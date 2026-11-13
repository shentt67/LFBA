"""Manual CIFAR-10 selection experiment. No training without explicit --run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ('run', 'dry-run', 'report-only'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--reference-root', type=Path, default=ROOT / 'results/lfba_stability_full')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.block_size < 1 or args.device < -1:
        parser.error('Positive block size and device >= -1 required')
    args.data_dir, args.reference_root = args.data_dir.resolve(), args.reference_root.resolve()
    args.output = (args.output or ROOT / 'results' / ('cifar10_decoupled_selection_v1_smoke'
                   if args.smoke else 'cifar10_decoupled_selection_v1')).resolve()
    for protected in (args.data_dir, args.reference_root):
        if args.output == protected or protected in args.output.parents or args.output in protected.parents:
            parser.error('Output must not overlap data or historical references')
    return args


def preflight(args):
    from lfba_comparison.decoupled_protocol import RATES, SEEDS
    from lfba_comparison.decoupled_records import load_references
    n, epochs = (450, 5) if args.smoke else (45000, 40)
    seeds = [2004] if args.smoke else SEEDS
    errors, refs = [], 0
    required = [*(f'data_batch_{i}' for i in range(1, 6)), 'test_batch', 'batches.meta']
    for name in required:
        if not (args.data_dir / 'cifar-10-batches-py' / name).is_file():
            errors.append('Missing CIFAR input: ' + name)
    if not args.smoke:
        for seed in seeds:
            try:
                refs += len(load_references(args.reference_root, seed))
            except Exception as error:
                errors.append(f'Historical reference seed={seed}: {type(error).__name__}: {error}')
    return dict(training_started=False, inference_started=False, seeds=seeds,
                tasks=len(seeds) * 4, reference_count=int(n * .1), epochs=epochs,
                cases=[dict(p=p, candidates=int(n * p), replacements_per_epoch=int(n * p) // 2,
                            cumulative_replacements=(int(n * p) // 2) * (epochs - 3)) for p in RATES],
                historical_reference_count=refs, errors=errors,
                output=str(args.output), output_occupied=args.output.exists() and any(args.output.iterdir()),
                pending_checks=['New first-round state and three-round metric history after clean prefix',
                                'Current raw data fingerprints against historical splits at dataset construction'],
                historical_rounds_2_3_full_state_available=False)


def main(argv=None):
    args = parse_options(argv)
    if args.report_only:
        from lfba_comparison.common import batch_lock
        from lfba_comparison.decoupled_report import write_report
        if not (args.output / 'protocol.json').is_file():
            raise FileNotFoundError('No saved experiment protocol')
        with batch_lock(args.output):
            report = write_report(args.output)
        return 0 if report['all_complete'] else 1
    check = preflight(args)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if check['errors']:
        return 2
    if not args.run:
        return 0
    from lfba_comparison.decoupled_runner import run
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
