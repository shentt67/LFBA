"""Frozen development matrix and explicit manual execution configuration."""
from pathlib import Path

from lfba_comparison.decoupled_policy import METHODS
from lfba_comparison.decoupled_report import THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2004, 592, 3383, 100]
RATES = (.01, .05)


def case_name(rate):
    if rate not in RATES:
        raise ValueError('Only the preregistered p=.01/.05 cases are supported')
    return 'p_' + str(rate).replace('.', 'p') + '_s_0p5'


def source_files():
    paths = [ROOT / 'main.py', ROOT / 'scripts/run_decoupled_selection.py']
    for directory in ('attack', 'dataset', 'models', 'utils', 'lfba_comparison'):
        paths.extend(sorted((ROOT / directory).glob('*.py')))
    return paths


def protocol(options):
    from lfba_comparison.decoupled_records import file_hash
    return dict(kind='cifar10_decoupled_selection', version=1, dataset='CIFAR10',
                seeds=[2004] if options.smoke else SEEDS, methods=list(METHODS),
                rates=list(RATES), select_rate=.5, reference_rate=.10,
                epochs=5 if options.smoke else 40, clean_rounds=3, attack_start=4,
                smoke=options.smoke, batch_size=128, optimizer='Adam', lr=.001,
                lr_decay=False, early_stop=False, final_model='last',
                retained_split_anchor=23470, thresholds=dict(THRESHOLDS),
                ranking_rules=dict(decoupled_mean=['mean_cosine_desc', 'id_asc'],
                    decoupled_vote=['top_K_votes_desc', 'mean_cosine_desc', 'id_asc']),
                train_source_limit=500 if options.smoke else None,
                eval_limit=100 if options.smoke else None, device=options.device,
                block_size=options.block_size, data_dir=str(options.data_dir),
                reference_root=str(options.reference_root), keep_checkpoints=True,
                historical_identity='first-round full-state digests + three-round metric history; rounds 2/3 state digests unavailable',
                source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in source_files()})


def input_files(options):
    files = list(source_files())
    files.extend(options.data_dir / 'cifar-10-batches-py' / name
                 for name in [*(f'data_batch_{i}' for i in range(1, 6)), 'test_batch', 'batches.meta'])
    if not options.smoke:
        for rate in RATES:
            folder = options.reference_root / case_name(rate)
            # All durable inputs in the two referenced batches, not model checkpoints.
            files.extend(p for p in folder.rglob('*')
                         if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.log', '.png'))
    return sorted(set(files))
