"""Read-only reuse of old automatic arms; only fixed controls can train."""
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
import traceback

from dataset.experiment import build_datasets
from lfba_comparison.common import read_json, write_json, digest, batch_lock
from lfba_comparison.core_runner import (assemble, close, verify_prefix, run_branch,
                                         saved_result, source_hashes)
from lfba_comparison.anchor_vote_policy import METHODS, AUTO_METHODS, ANCHOR, select_fixed
from utils.artifacts import runtime_checkpoint
from utils.checkpoint import load_checkpoint
from lfba_comparison.completion import load_completion, all_completed, finish_ablation

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_METHODS = AUTO_METHODS + ('auto_mean',)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify_manifest(manifest):
    for name, expected in manifest.items():
        if not Path(name).is_file() or file_hash(name) != expected:
            raise ValueError('Frozen source file changed or missing: '+name)


def copy_prefix(source, target, expected_hash):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve(): raise ValueError('Cannot overwrite source prefix')
    if target.exists():
        if file_hash(target) != expected_hash: raise ValueError('Copied prefix differs')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix+'.copying')
        shutil.copyfile(source, temporary)
        if file_hash(temporary) != expected_hash: raise ValueError('Source changed during copy')
        temporary.replace(target)
    if file_hash(source) != expected_hash: raise ValueError('Source prefix changed')


def current_sources():
    result = source_hashes()
    path = ROOT/'scripts/run_anchor_vote_ablation.py'
    result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def check_compatibility(protocol):
    required = dict(kind='core_ablation', dataset='CIFAR10', smoke=False, epochs=40,
                    batch_size=128, optimizer='Adam', lr=.001, lr_decay=False,
                    early_stop=False, poison_rate=.1, select_rate=.5, clean_rounds=3,
                    attack_start=4, retained_split_anchor=ANCHOR, final_model='last', similarity='cosine')
    for key, value in required.items():
        if protocol.get(key) != value: raise ValueError('Incompatible source protocol: '+key)
    if not protocol.get('source_hashes'): raise ValueError('Missing source code provenance')
    # Newly added modules are permitted; every recorded source must remain unchanged.
    for relative, expected in protocol['source_hashes'].items():
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != expected:
            raise ValueError('Original source code differs: '+relative)


def inspect_sources(options):
    """CPU-only validation, no model construction, CUDA initialization or writes."""
    source = options.source_results
    protocol_path = source/'protocol.json'
    protocol = read_json(protocol_path)
    if protocol is None: raise ValueError('Missing source protocol: '+str(protocol_path))
    check_compatibility(protocol)
    files = {str(protocol_path): file_hash(protocol_path)}
    seeds = {}
    for seed in options.seeds:
        if seed not in protocol['seeds']: raise ValueError('Seed absent from source protocol')
        folder = source/f'seed_{seed}'
        prefix_path = runtime_checkpoint(folder/'clean_prefix')
        if not prefix_path.is_file():
            raise ValueError('Source model checkpoints were cleaned; run the source core ablation with --keep-checkpoints before using it for dependent training')
        prefix_hash = file_hash(prefix_path)
        prefix = load_checkpoint(prefix_path)
        verify_prefix(prefix)
        cfg = prefix['config']
        if cfg['seed'] != seed or cfg['dataset'] != 'CIFAR10': raise ValueError('Source seed/dataset differs')
        if [len(prefix['data_state'][k]) for k in ('train_ids','val_ids','test_ids')] != [45000,5000,10000]:
            raise ValueError('Unexpected source data split')
        if ANCHOR not in prefix['data_state']['train_ids']: raise ValueError('Fixed anchor not in training set')
        args = SimpleNamespace(**cfg); args.data_dir = str(options.data_dir)
        train, _, _, state = build_datasets(args, prefix['data_state'])
        if state != prefix['data_state']: raise ValueError('Source data fingerprints or split differ')
        selection = read_json(folder/'selection.json')
        if not selection or selection['prefix_digest'] != digest(prefix): raise ValueError('Source selection prefix differs')
        reference = {}
        for method in REFERENCE_METHODS:
            choice = selection['choices'][method]
            result = saved_result(folder, method, seed, 40, prefix, choice)
            if result is None: raise ValueError(f'Source result incomplete: {seed}/{method}')
            if result['target_class'] != int(train.targets[result['anchor']]):
                raise ValueError('Source target label differs from dataset')
            reference[method] = dict(directory=str(folder/method), choice=choice)
            for name in ('metrics.json','config.json','experiment_data.json','.runtime/last_checkpoint.pth.tar'):
                path = folder/method/name; files[str(path)] = file_hash(path)
        anchors = [reference[m]['choice']['anchor_id'] for m in REFERENCE_METHODS]
        if len(set(anchors)) != 1: raise ValueError('Automatic arms do not share an anchor')
        for path in (prefix_path, folder/'selection.json', folder/'selection.pth.tar'):
            files[str(path)] = file_hash(path)
        if files[str(prefix_path)] != prefix_hash: raise ValueError('Source prefix changed during inspection')
        seeds[str(seed)] = dict(prefix_path=str(prefix_path), prefix_digest=digest(prefix),
                                prefix_file_hash=prefix_hash, references=reference,
                                data_digest=digest(state))
        del train, prefix
    return dict(protocol_path=str(protocol_path), files=files, seeds=seeds)


def preflight(options):
    count = 45 if options.smoke else 4500
    result = dict(training_started=False, ready=False, smoke=options.smoke, seeds=options.seeds,
                  methods=list(METHODS), tasks=2*len(options.seeds), epochs=5 if options.smoke else 40,
                  candidates=count, replacements_per_epoch=count//2,
                  cumulative_replacements=(2 if options.smoke else 37)*(count//2),
                  source_results=None if options.smoke else str(options.source_results),
                  output=str(options.output), output_occupied=options.output.exists() and any(options.output.iterdir()),
                  errors=[])
    try:
        for name in [f'data_batch_{i}' for i in range(1,6)]+['test_batch','batches.meta']:
            if not (options.data_dir/'cifar-10-batches-py'/name).is_file():
                raise FileNotFoundError('Missing CIFAR file: '+name)
        if not options.smoke:
            # Do not expose large manifests in terminal output.
            result['_manifest'] = inspect_sources(options)
        result['ready'] = True
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def protocol(options, manifest):
    return dict(version=1, kind='anchor_vote_ablation', dataset='CIFAR10',
                seeds=options.seeds, methods=list(METHODS), smoke=options.smoke,
                epochs=5 if options.smoke else 40, device=options.device, block_size=128,
                data_dir=str(options.data_dir), source_results=None if options.smoke else str(options.source_results),
                source_manifest=manifest, source_hashes=current_sources(), anchor=ANCHOR,
                poison_rate=.1, select_rate=.5, clean_rounds=3, attack_start=4,
                batch_size=128, lr=.001, optimizer='Adam', lr_decay=False, early_stop=False,
                final_model='last', similarity='cosine', keep_checkpoints=getattr(options, 'keep_checkpoints', False))


def require_identical_protocol(saved, expected):
    if saved != expected: raise ValueError('Resume requires identical protocol, code and source manifest')


def run(options, manifest=None):
    from lfba_comparison.anchor_vote_report import write_report
    root = options.output
    expected = protocol(options, manifest)
    failures = []
    with batch_lock(root):
        occupied = any(p.name != 'protocol.json.tmp' for p in root.iterdir())
        if occupied:
            if not options.resume: raise FileExistsError('Output occupied; explicitly use --resume')
            require_identical_protocol(read_json(root/'protocol.json'), expected)
        else:
            write_json(root/'protocol.json', expected)
        completed = all_completed(root, expected)
        if manifest and not completed: verify_manifest(manifest['files'])
        for seed in options.seeds:
            folder = root/f'seed_{seed}'
            if load_completion(folder, expected, seed) is not None:
                print(f'SKIP seed={seed} completed and audited', flush=True)
                continue
            prefix_path = runtime_checkpoint(folder/'clean_prefix')
            try:
                if options.smoke:
                    prefix = load_checkpoint(prefix_path) if prefix_path.exists() else None
                    if prefix is None or prefix['epoch'] < 3:
                        trainer, handler = assemble(options, seed, 'capture', folder/'clean_prefix', prefix)
                        try: trainer.train()
                        finally: close(trainer, handler)
                        del trainer
                else:
                    info = manifest['seeds'][str(seed)]
                    copy_prefix(info['prefix_path'], prefix_path, info['prefix_file_hash'])
                prefix = load_checkpoint(prefix_path); verify_prefix(prefix)
                selection_path = folder/'selection.json'
                if selection_path.exists():
                    selection = read_json(selection_path)
                    if not selection or selection['prefix_digest'] != digest(prefix):
                        raise ValueError('Frozen selection prefix differs')
                    fresh = select_fixed(prefix['comparison']['observations'], int(len(prefix['data_state']['train_ids'])*.1))
                    if selection['results'] != fresh['results']: raise ValueError('Frozen selection differs')
                else:
                    selection = select_fixed(prefix['comparison']['observations'], int(len(prefix['data_state']['train_ids'])*.1))
                    selection['prefix_digest'] = digest(prefix)
                    write_json(selection_path, selection)
            except Exception as error:
                failures.append(dict(seed=seed, method='prefix', error=f'{type(error).__name__}: {error}'))
                traceback.print_exc(); continue
            for method in METHODS:
                try:
                    choice = selection['results'][method]
                    if saved_result(folder, method, seed, expected['epochs'], prefix, choice):
                        print(f'SKIP seed={seed} method={method}', flush=True)
                    else:
                        print(f'RUN seed={seed} method={method}', flush=True)
                        run_branch(options, seed, method, folder, prefix, choice)
                        if not saved_result(folder, method, seed, expected['epochs'], prefix, choice):
                            raise ValueError('Final durable audit missing')
                        print(f'COMPLETE seed={seed} method={method}', flush=True)
                except Exception as error:
                    failures.append(dict(seed=seed, method=method, error=f'{type(error).__name__}: {error}'))
                    traceback.print_exc()
                write_json(root/'failures.json', failures)
                write_report(root)
        if manifest and not completed: verify_manifest(manifest['files'])
        write_json(root/'failures.json', failures)
        result = write_report(root)
        if not failures:
            finish_ablation(root, result, getattr(options, 'keep_checkpoints', False))
    success = result['all_complete'] and not failures
    print('ALL_TASKS_COMPLETE' if success else 'BATCH_INCOMPLETE', root, flush=True)
    return 0 if success else 1
