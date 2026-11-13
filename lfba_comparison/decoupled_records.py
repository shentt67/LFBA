"""Read-only provenance checks and parameter-aware audits for the new experiment."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from lfba_comparison.common import digest
from lfba_comparison.decoupled_policy import METHODS


def required_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot(paths):
    return {str(Path(p).resolve()): file_hash(p) for p in sorted(set(paths))}


def verify_manifest(manifest):
    changed = []
    for name, expected in manifest.items():
        try:
            if file_hash(name) != expected:
                changed.append(name)
        except OSError:
            changed.append(name)
    return dict(unchanged=not changed, files_checked=len(manifest), changed=changed)


def verify_bound_manifest(manifest, protocol):
    """An unchanged subset is insufficient: preserve the original frozen inventory."""
    if (not isinstance(manifest, dict) or not manifest
            or set(manifest) != set(protocol.get('input_paths', []))
            or digest(manifest) != protocol.get('input_manifest_digest')):
        return dict(unchanged=False, files_checked=0, changed=[],
                    reason='Missing, truncated or altered protocol-bound input manifest')
    return verify_manifest(manifest)


def validate_metrics(metrics, formal=False):
    counts = metrics.get('counts', {})
    for key in ('acc', 'asr', 'poison_acc', 'target_acc'):
        value, denominator = metrics.get(key), counts.get(key)
        if not isinstance(denominator, int) or denominator <= 0:
            raise ValueError('Invalid evaluation denominator: ' + key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Missing or nonfinite evaluation metric: ' + key)
        if abs(value * denominator - round(value * denominator)) > 1e-7:
            raise ValueError('Metric cannot be recomputed from integer numerator: ' + key)
    if counts['asr'] + counts['target_acc'] != counts['acc'] or counts['poison_acc'] != counts['acc']:
        raise ValueError('Inconsistent target-specific evaluation denominators')
    if formal and counts != dict(acc=10000, poison_acc=10000, asr=9000, target_acc=1000):
        raise ValueError('Unexpected CIFAR-10 formal denominators')


def load_references(reference_root, seed):
    from lfba_comparison.decoupled_protocol import RATES, case_name
    from lfba_comparison.records import verified_pair
    refs = []
    for rate in RATES:
        case = Path(reference_root) / case_name(rate)
        saved = required_json(case / 'protocol.json')
        expected = dict(dataset='CIFAR10', poison_rate=rate, select_rate=.5, epochs=40,
                        smoke=False, anchor=23470, anchor_source='fixed', final_model='last',
                        optimizer='Adam', lr=.001, batch_size=128, lr_decay=False, early_stop=False,
                        baseline_similarity='legacy_l1', vote_similarity='cosine')
        if any(saved.get(k) != v for k, v in expected.items()) or seed not in saved.get('seeds', []):
            raise ValueError('Historical protocol differs: ' + str(case))
        folder = case / f'seed_{seed}'
        records = verified_pair(folder, 40, seed, rate, .5, 'fixed', 23470)
        for method, record in records.items():
            output = folder / method
            config = required_json(output / 'config.json')
            data = required_json(output / 'experiment_data.json')
            checks = dict(dataset='CIFAR10', epoch=40, batch_size=128, lr=.001,
                          early_stop=0, train_limit=None, eval_limit=None, attack='LFBA',
                          client_num=2, attack_client_num=1, validation_rate=.1,
                          poison_all=False, random_select=False, retain_anchor=True,
                          similarity='cosine' if method == 'auto_vote' else 'legacy_l1')
            if any(config.get(k) != v for k, v in checks.items()):
                raise ValueError('Historical training configuration differs: ' + str(output))
            state = config['data_state']
            if record['data_digest'] != digest(state) or [len(state[k]) for k in ('train_ids', 'val_ids', 'test_ids')] != [45000, 5000, 10000]:
                raise ValueError('Historical data identity differs')
            if len(data['choice']['candidates']) != int(45000 * rate):
                raise ValueError('Historical candidate budget differs')
            validate_metrics(record['test'], formal=True)
            refs.append(dict(seed=seed, p=rate, method=method, source=str(output.resolve()),
                             record=record, data_state=state, trigger_dimensions=config['trigger_dimensions'],
                             clean_history=data['history'][:3 if method == 'auto_vote' else 1]))
    return refs


def verify_reference_prefix(prefix, refs):
    for ref in refs:
        record = ref['record']
        state = prefix['comparison']
        if (prefix['data_state'] != ref['data_state'] or prefix['trigger_dimensions'] != ref['trigger_dimensions']
                or state['initial_digest'] != record['initial_digest']
                or state['prefix'][0] != record['first_prefix']):
            raise ValueError('Historical first-round/data/initialization identity differs: ' + ref['source'])
        clean = 3 if ref['method'] == 'auto_vote' else 1
        if prefix['history'][:clean] != ref['clean_history']:
            raise ValueError('Historical clean metric trajectory differs: ' + ref['source'])
    return dict(verified=True, reference_count=len(refs),
                initial_digest=prefix['comparison']['initial_digest'],
                data_digest=digest(prefix['data_state']), first_prefix=prefix['comparison']['prefix'][0],
                clean_history_digest=digest(prefix['history'][:3]),
                rounds_2_3_full_state_available=False,
                interpretation='Exact first-round state digests and clean metric history; no claim of historical round-2/3 full-state equality')


def verify_prefix(prefix):
    state = prefix['comparison']
    if (prefix['epoch'] != 3 or state['method'] != 'capture' or state['choice'] is not None
            or state['plans'] or state['cumulative'] or prefix['attack_state']['current_plan']
            or prefix['attack_state']['label_overrides']
            or [o['epoch'] for o in state['observations']] != [1, 2, 3]
            or len(state['prefix']) != 3 or len(state['audit']) != 3 or len(prefix['history']) != 3):
        raise ValueError('Invalid three-round clean prefix')


def verify_final(final, prefix, choice, epochs, rate):
    verify_prefix(prefix)
    state = final['comparison']
    count = int(len(prefix['data_state']['train_ids']) * rate)
    q = count // 2
    expected = [0] * 3 + [q] * (epochs - 3)
    start = digest(dict(model=prefix['state_dict'], optimizer=prefix['optimizer'], rng=prefix['rng_state']))
    if (choice['method'] not in METHODS or final['epoch'] != epochs or state['method'] != choice['method']
            or state['choice'] != choice or state['start_state'] != start
            or state['initial_digest'] != prefix['comparison']['initial_digest']
            or final['data_state'] != prefix['data_state'] or state['prefix'] != prefix['comparison']['prefix']
            or digest(state['observations']) != digest(prefix['comparison']['observations'])
            or final['history'][:3] != prefix['history'] or final['trigger_dimensions'] != prefix['trigger_dimensions']
            or final['config']['poison_rate'] != rate or final['config']['select_rate'] != .5):
        raise ValueError('Final identity, prefix, selection or protocol differs')
    pool, train = set(choice['candidates']), set(prefix['data_state']['train_ids'])
    if q < 1 or len(pool) != count or len(choice['candidates']) != count or not pool <= train or choice['anchor_id'] in pool:
        raise ValueError('Invalid frozen candidates')
    if (state['cumulative'] != sum(expected) or len(state['audit']) != epochs or len(final['history']) != epochs
            or final['attack_state']['poison_indexes'] != choice['candidates'] or final['attack_state']['label_overrides']):
        raise ValueError('Final budget or attack state differs')
    for i, row in enumerate(state['audit']):
        if (row['epoch'] != i + 1 or row['recipient_count'] != expected[i]
                or row['cumulative_replacements'] != sum(expected[:i + 1])
                or row['candidate_count'] != (0 if i < 3 else count)
                or (i >= 3 and row['candidate_digest'] != digest(choice['candidates']))):
            raise ValueError('Per-round audit differs')
    if [p['epoch'] for p in state['plans']] != list(range(4, epochs + 1)):
        raise ValueError('Attack epochs differ')
    for plan in state['plans']:
        recipients, donors = plan['recipients'], plan['donors']
        if (len(recipients) != q or len(set(recipients)) != q or not set(recipients) <= pool
                or len(donors) != q or len(set(donors)) != q or not set(donors) <= train - pool):
            raise ValueError('Invalid replacement plan')
    if final['attack_state']['current_plan'] != dict(zip(state['plans'][-1]['recipients'], state['plans'][-1]['donors'])):
        raise ValueError('Final applied plan differs')


def exposure_diagnostics(labels, choice, plans):
    candidates = choice['candidates']
    target = int(labels[choice['anchor_id']])
    recipients = [i for p in plans for i in p['recipients']]
    donors = [i for p in plans for i in p['donors']]
    uses = Counter(recipients)
    wrong = sum(int(labels[i]) != target for i in recipients)
    histogram = lambda ids: dict(sorted(Counter(str(int(labels[i])) for i in ids).items()))
    return dict(candidate_class_counts=histogram(candidates), recipient_class_counts=histogram(recipients),
                donor_class_counts=histogram(donors), wrong_exposures=wrong,
                total_exposures=len(recipients), unique_recipients=len(uses), unique_donors=len(set(donors)),
                recipient_coverage=len(uses) / len(candidates), max_reuse=max(uses.values(), default=0),
                recipient_use_histogram=dict(sorted(Counter(str(uses[i]) for i in candidates).items())),
                candidate_purity=sum(int(labels[i]) == target for i in candidates) / len(candidates),
                recipient_purity=1 - wrong / len(recipients) if recipients else None)


def completed_record(root, output, protocol):
    """Validate durable receipts by hashing files, without loading models/inference."""
    root, output = Path(root).resolve(), Path(output).resolve()
    receipt_path = output / 'completion.json'
    if not receipt_path.exists():
        return None
    receipt = required_json(receipt_path)
    if receipt.get('protocol_digest') != digest(protocol):
        raise ValueError('Completion protocol differs')
    files = receipt.get('files', {})
    required = [output / name for name in ('metrics.json', 'config.json', 'experiment_data.json',
                                          '.runtime/last_checkpoint.pth.tar')]
    seed_folder = output.parent.parent
    required += [seed_folder / name for name in ('selection.json', 'selection.pth.tar',
                 'reference_check.json', 'clean_prefix/.runtime/last_checkpoint.pth.tar')]
    if set(files) != {str(p.relative_to(root)) for p in required}:
        raise ValueError('Incomplete completion file manifest')
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if root not in path.parents or file_hash(path) != expected:
            raise ValueError('Completion file changed: ' + str(path))
    record = required_json(output / 'metrics.json')
    config = required_json(output / 'config.json')
    data = required_json(output / 'experiment_data.json')
    selection = required_json(seed_folder / 'selection.json')
    reference = required_json(seed_folder / 'reference_check.json')
    from lfba_comparison.decoupled_protocol import case_name
    key = case_name(record['p'])
    choice = selection['choices'][key][record['method']]
    if (output != root / f"seed_{record['seed']}" / key / record['method']
            or record['method'] not in protocol['methods'] or record['seed'] not in protocol['seeds']
            or record['epoch'] != protocol['epochs'] or record['status'] != 'complete'
            or record['protocol_digest'] != digest(protocol) or data['choice'] != choice
            or record['choice_digest'] != digest(choice) or record['candidate_digest'] != digest(choice['candidates'])
            or record['anchor'] != choice['anchor_id'] or config['anchor_idx'] != record['anchor']
            or config['target_label'] != record['target_class'] or config['poison_rate'] != record['p']
            or record['data_digest'] != digest(config['data_state'])
            or record['prefix_digest'] != selection['prefix_digest']
            or reference['prefix_digest'] != selection['prefix_digest']
            or (not protocol['smoke'] and reference.get('verified') is not True)):
        raise ValueError('Saved branch identity differs')
    independent = record['independent_evaluation']
    if (independent['epoch'] != record['epoch'] or independent['test'] != record['test']
            or independent['final_model_digest'] != record['final_model_digest']):
        raise ValueError('Independent evaluation differs')
    validate_metrics(record['test'], formal=not protocol['smoke'])
    if len(data['history']) != protocol['epochs'] or record['cumulative_replacements'] != (protocol['epochs'] - 3) * (len(choice['candidates']) // 2):
        raise ValueError('Saved budget or history differs')
    return record
