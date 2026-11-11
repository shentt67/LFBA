"""Checkpoint-audited completion receipts for reports after model cleanup."""
import hashlib
import json
from pathlib import Path

from lfba_comparison.common import digest, read_json, write_json
from utils.artifacts import cleanup_runtime, runtime_checkpoint, runtime_cleanup_paths
from utils.checkpoint import load_checkpoint


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def local_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError('Artifact path must stay inside the experiment output')
    return path


def methods_for(protocol):
    if protocol['kind'] == 'core_ablation':
        from lfba_comparison.core_policy import METHODS
    elif protocol['kind'] == 'anchor_vote_ablation':
        from lfba_comparison.anchor_vote_policy import METHODS
    else:
        raise ValueError('Unsupported ablation protocol')
    return METHODS


def durable_names(protocol):
    names = ['selection.json', 'clean_prefix/config.json', 'clean_prefix/experiment_data.json']
    for method in methods_for(protocol):
        names.extend(method+'/'+name for name in ('config.json', 'metrics.json', 'experiment_data.json'))
    if protocol['kind'] == 'core_ablation':
        names.append('selection.pth.tar')  # Numeric selection evidence, not a model checkpoint.
    elif protocol.get('source_manifest'):
        names.append('references.json')
    return names


def load_completion(folder, protocol, seed):
    """Return None only when no receipt exists; invalid receipts must never fall back."""
    folder = Path(folder).resolve()
    path = local_path(folder, 'completion_proof.json')
    if not path.exists():
        return None
    proof = read_json(path)
    if (not isinstance(proof, dict) or proof.get('version') != 1 or proof.get('seed') != seed
            or proof.get('protocol_digest') != digest(protocol)
            or set(proof.get('files', {})) != set(durable_names(protocol))
            or set(proof.get('records', {})) != set(methods_for(protocol))):
        raise ValueError('Invalid completion proof identity')
    for relative, expected in proof['files'].items():
        if file_hash(local_path(folder, relative)) != expected:
            raise ValueError('Completed artifact changed: '+relative)
    selection = read_json(folder/'selection.json')
    if not selection or selection.get('prefix_digest') != proof['prefix']['digest']:
        raise ValueError('Completed selection prefix differs')
    choices = selection['choices' if protocol['kind'] == 'core_ablation' else 'results']
    for method, record in proof['records'].items():
        if (record != read_json(folder/method/'metrics.json')
                or record['seed'] != seed or record['method'] != method
                or record['epoch'] != protocol['epochs'] or record['status'] != 'complete'
                or record['start_state'] != proof['prefix']['start_state']
                or record['data_digest'] != proof['prefix']['data_digest']
                or record['choice_digest'] != digest(choices[method])
                or record['test'] != record['independent_evaluation']['test']
                or record['final_model_digest'] != record['independent_evaluation']['final_model_digest']):
            raise ValueError('Completed result differs from audited identity')
    return proof


def create_completion(folder, protocol, seed):
    """Mint a receipt only after the original full-state audits pass for every arm."""
    from lfba_comparison.core_runner import saved_result, verify_prefix
    folder = Path(folder).resolve()
    existing = load_completion(folder, protocol, seed)
    if existing is not None:
        return existing
    # Resolve the fixed allowlist before reading or writing within this seed.
    for name in durable_names(protocol):
        local_path(folder, name)
    prefix = load_checkpoint(runtime_checkpoint(folder/'clean_prefix'))
    verify_prefix(prefix)
    state = prefix['comparison']
    prefix_records = {
        'config.json': dict(prefix['config'], data_state=prefix['data_state'],
                            trigger_dimensions=prefix['trigger_dimensions']),
        'experiment_data.json': dict(version=1, history=prefix['history'], choice=state['choice'],
                                     plans=state['plans'], audit=state['audit'])}
    for name, value in prefix_records.items():
        path = local_path(folder, 'clean_prefix/'+name)
        value = json.loads(json.dumps(value, allow_nan=False))
        # Formal anchor runs copy only the prefix model. Materialize its durable
        # records from that verified state, including when finalizing an old run.
        if not path.exists() and protocol.get('source_manifest'):
            write_json(path, value)
        if read_json(path) != value:
            raise ValueError('Durable clean prefix records differ from checkpoint')
    selection = read_json(folder/'selection.json')
    if not selection or selection['prefix_digest'] != digest(prefix):
        raise ValueError('Selection prefix differs')
    if protocol['kind'] == 'core_ablation':
        from lfba_comparison.core_policy import diagnostics
        evidence = load_checkpoint(folder/'selection.pth.tar')
        if (evidence['prefix_digest'] != digest(prefix) or selection['choices'] != evidence['results']
                or selection['diagnostics'] != diagnostics(evidence)):
            raise ValueError('Selection differs from saved numeric evidence')
        choices = selection['choices']
    else:
        from lfba_comparison.anchor_vote_policy import select_fixed
        expected = select_fixed(prefix['comparison']['observations'], int(len(prefix['data_state']['train_ids'])*.1))
        if selection['results'] != expected['results']:
            raise ValueError('Fixed selection differs from gradient evidence')
        choices = selection['results']
        manifest = protocol.get('source_manifest')
        if manifest:
            from lfba_comparison.anchor_vote_runner import verify_manifest
            verify_manifest(manifest['files'])
            references = []
            info = manifest['seeds'][str(seed)]
            if info['prefix_digest'] != digest(prefix):
                raise ValueError('Reference prefix differs')
            for method, reference in info['references'].items():
                directory = Path(reference['directory'])
                result = saved_result(directory.parent, method, seed, 40, prefix, reference['choice'])
                if result is None:
                    raise ValueError('Referenced result is incomplete')
                references.append(dict(result=result, data=read_json(directory/'experiment_data.json'),
                                       config=read_json(directory/'config.json'), directory=str(directory)))
            write_json(folder/'references.json', dict(version=1, source_manifest_digest=digest(manifest),
                                                     records=references))
            verify_manifest(manifest['files'])
    records = {}
    for method in methods_for(protocol):
        result = saved_result(folder, method, seed, protocol['epochs'], prefix, choices[method])
        if result is None:
            raise ValueError('Cannot certify an incomplete branch: '+method)
        records[method] = result
    proof = dict(version=1, seed=seed, protocol_digest=digest(protocol), records=records,
                 prefix=dict(digest=digest(prefix), epoch=3, data_digest=digest(prefix['data_state']),
                             start_state=digest(dict(model=prefix['state_dict'], optimizer=prefix['optimizer'],
                                                     rng=prefix['rng_state']))),
                 files={name:file_hash(folder/name) for name in durable_names(protocol)})
    # JSON normalization makes the read-back comparison independent of tuple/list representation.
    proof = json.loads(json.dumps(proof, allow_nan=False))
    write_json(folder/'completion_proof.json', proof)
    if load_completion(folder, protocol, seed) != proof:
        raise ValueError('Completion proof read-back differs')
    return proof


def all_completed(root, protocol):
    return all(load_completion(local_path(root, f'seed_{seed}'), protocol, seed) is not None
               for seed in protocol['seeds'])


def finish_ablation(root, report, keep_checkpoints=False):
    """Validate every result and report before deleting any registered model file."""
    root = Path(root).resolve()
    protocol = read_json(root/'protocol.json')
    if not report.get('all_complete') or report.get('missing') or read_json(root/'failures.json'):
        return
    if read_json(root/'summary.json') != report:
        raise ValueError('Summary read-back differs')
    for name in ('results.md', 'all_epoch_metrics.csv', 'validation_curves.png'):
        path = local_path(root, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing or empty report artifact: '+name)
    folders = []
    for seed in protocol['seeds']:
        folder = local_path(root, f'seed_{seed}')
        create_completion(folder, protocol, seed)
        folders.extend(local_path(folder, name) for name in ('clean_prefix', *methods_for(protocol)))
    for seed in protocol['seeds']:
        load_completion(local_path(root, f'seed_{seed}'), protocol, seed)
    # Prevalidate every target before deleting the first checkpoint.
    for folder in folders:
        runtime_cleanup_paths(folder)
    if keep_checkpoints or protocol.get('keep_checkpoints', False):
        return
    for folder in folders:
        cleanup_runtime(folder)
