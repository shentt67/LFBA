"""Anchor x voting controls; training is only entered with explicit --run."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parse_options(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for name in ('run','dry-run','report-only'): modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--keep-checkpoints',action='store_true',help='Keep models after success for dependent experiments')
    parser.add_argument('--source-results',type=Path,default=ROOT/'results/core_ablation')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--seeds',nargs='+',type=int,choices=(2004,592,3383))
    args=parser.parse_args(argv)
    if args.resume and not args.run: parser.error('--resume requires --run')
    args.seeds=args.seeds or ([2004] if args.smoke else [2004,592,3383])
    if len(set(args.seeds))!=len(args.seeds): parser.error('Duplicate seeds')
    args.block_size=128
    args.output=(args.output or ROOT/'results'/('anchor_vote_ablation_v1_smoke' if args.smoke else 'anchor_vote_ablation_v1')).resolve()
    args.source_results=args.source_results.resolve();args.data_dir=args.data_dir.resolve()
    if (args.output==args.data_dir or args.data_dir in args.output.parents
            or args.output==args.source_results or args.source_results in args.output.parents
            or args.output in args.source_results.parents):
        parser.error('Output must be separate from input data and source results')
    return args


def main(argv=None):
    args=parse_options(argv)
    if args.report_only:
        from lfba_comparison.common import batch_lock
        from lfba_comparison.anchor_vote_report import write_report
        if not (args.output/'protocol.json').exists(): raise FileNotFoundError('No saved protocol')
        from lfba_comparison.completion import finish_ablation
        with batch_lock(args.output):
            result=write_report(args.output)
            finish_ablation(args.output,result,args.keep_checkpoints)
        return 0 if result['all_complete'] else 1
    if args.resume:
        from lfba_comparison.common import read_json
        from lfba_comparison.completion import all_completed
        saved=read_json(args.output/'protocol.json')
        if saved and all_completed(args.output,saved):
            from lfba_comparison.anchor_vote_runner import run
            return run(args,saved.get('source_manifest'))
    from lfba_comparison.anchor_vote_runner import preflight,run
    result=preflight(args)
    manifest=result.pop('_manifest',None)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if not result['ready']: return 2
    if not args.run: return 0
    return run(args,manifest)


if __name__=='__main__': raise SystemExit(main())
