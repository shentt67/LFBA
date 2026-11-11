"""Explicit, isolated four-arm core ablation. No training without --run."""
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
        modes.add_argument('--'+name, action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Keep models after success for dependent experiments')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.resume and not args.run: parser.error('--resume requires --run')
    if args.seeds is None: args.seeds = [2004] if args.smoke else [2004, 592, 3383, 100]
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error('Seeds must be unique integers within [0,2**32)')
    if args.block_size < 1: parser.error('Positive block size required')
    args.output = (args.output or ROOT/'results'/('core_ablation_v1_smoke' if args.smoke else 'core_ablation_v1')).resolve()
    args.data_dir = args.data_dir.resolve()
    if args.output == args.data_dir or args.data_dir in args.output.parents:
        parser.error('Output cannot be inside input data')
    return args


def preflight(args):
    from lfba_comparison.core_policy import METHODS, schedule
    required = [f'data_batch_{i}' for i in range(1, 6)]+['test_batch', 'batches.meta']
    missing = [str(args.data_dir/'cifar-10-batches-py'/name) for name in required
               if not (args.data_dir/'cifar-10-batches-py'/name).is_file()]
    epochs = 5 if args.smoke else 40
    count = 45 if args.smoke else 4500
    return dict(training_started=False, seeds=args.seeds, methods=list(METHODS),
                tasks=4*len(args.seeds), epochs=epochs, clean_rounds=3, attack_start=4,
                candidates=count, replacements_per_epoch=count//2,
                cumulative_replacements=sum(schedule(epochs, count)),
                missing_data_files=missing, output=str(args.output),
                output_occupied=args.output.exists() and any(args.output.iterdir()))


def main(argv=None):
    args = parse_options(argv)
    if args.report_only:
        from lfba_comparison.common import batch_lock
        from lfba_comparison.core_report import write_report
        if not (args.output/'protocol.json').exists(): raise FileNotFoundError('No saved protocol')
        from lfba_comparison.completion import finish_ablation
        with batch_lock(args.output):
            result = write_report(args.output)
            finish_ablation(args.output, result, args.keep_checkpoints)
        return 0 if result['all_complete'] else 1
    if args.resume:
        from lfba_comparison.common import read_json
        from lfba_comparison.completion import all_completed
        saved = read_json(args.output/'protocol.json')
        if saved and all_completed(args.output, saved):
            from lfba_comparison.core_runner import run
            return run(args)
    check = preflight(args)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if check['missing_data_files']: return 2
    if not args.run: return 0
    from lfba_comparison.core_runner import run
    return run(args)


if __name__ == '__main__': raise SystemExit(main())
