"""Manual LFBA comparison: training requires explicit --run."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def default_data_directory(code_root):
    repository=code_root.parent.parent if code_root.parent.name in ('.worktrees','worktrees') else code_root
    return repository/'data'


def parse_options(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--report-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,default=[106,107,108])
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--data-dir',type=Path,default=default_data_directory(ROOT))
    parser.add_argument('--output',type=Path)
    options=parser.parse_args(argv)
    if options.resume and not options.run:parser.error('--resume requires --run')
    if len(set(options.seeds))!=len(options.seeds) or any(s<0 for s in options.seeds) or options.block_size<1:
        parser.error('Unique nonnegative seeds and positive block size required')
    if options.output is None:
        options.output=ROOT/'results'/('lfba_auto_vote_smoke' if options.smoke else 'lfba_auto_vote_formal')
    options.output=options.output.resolve();options.data_dir=options.data_dir.resolve();options.code_root=ROOT
    return options


def dry_run(options):
    required=[f'data_batch_{i}' for i in range(1,6)]+['test_batch','batches.meta']
    missing=[str(options.data_dir/'cifar-10-batches-py'/name) for name in required
             if not (options.data_dir/'cifar-10-batches-py'/name).is_file()]
    if options.output==options.data_dir or options.data_dir in options.output.parents:
        raise ValueError('Output must not be inside shared input data')
    print(json.dumps(dict(mode='DRY_RUN_ONLY',training_started=False,seeds=options.seeds,
        methods=['lfba_original','lfba_matched','auto_vote'],tasks=3*len(options.seeds),
        data_dir=str(options.data_dir),missing_data_files=missing,output=str(options.output),
        output_occupied=options.output.exists() and any(options.output.iterdir()),fixed_lfba_anchor=23470,
        epochs=5 if options.smoke else 20,
        formal_budgets=None if options.smoke else dict(lfba_original=42750,lfba_matched=38250,auto_vote=38250),
        note='No model creation, training, inference or CUDA initialization.'),ensure_ascii=False,indent=2))
    return 2 if missing else 0


def main(argv=None):
    options=parse_options(argv)
    if options.report_only:
        from lfba_comparison.report import write_report
        write_report(options.output);return 0
    if not options.run:return dry_run(options)
    if dry_run(options):raise FileNotFoundError('CIFAR-10 files missing; experiment not started')
    from lfba_comparison.runner import run_experiment
    run_experiment(options);return 0


if __name__=='__main__':raise SystemExit(main())
