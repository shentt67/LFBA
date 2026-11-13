"""Manual LFBA comparison: training requires explicit --run."""
import argparse
import json
import math
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
    parser.add_argument('--stability',action='store_true')
    parser.add_argument('--poison-rates',nargs='+',type=float)
    parser.add_argument('--select-rates',nargs='+',type=float)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reduce-seeds',action='store_true',help='Explicitly shrink a saved stability seed list during resume; preserve all existing data')
    parser.add_argument('--seeds',nargs='+',type=int,default=None)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--poison-rate',type=float,default=.1)
    parser.add_argument('--select-rate',type=float,default=.5)
    parser.add_argument('--poison-dimensions',type=int,default=None)
    parser.add_argument('--trigger-feature-clip',type=float,default=None)
    parser.add_argument('--dataset',choices=('CIFAR10','UCIHAR','NUSWIDE','NUSWID'),default=('CIFAR10'))
    parser.add_argument('--anchor-source',choices=('fixed','auto_vote'),default=None)
    parser.add_argument('--data-dir',type=Path,default=default_data_directory(ROOT))
    parser.add_argument('--output',type=Path)
    options=parser.parse_args(argv)
    if options.dataset=='NUSWID':options.dataset='NUSWIDE'
    from lfba_comparison.stability_config import (DEFAULT_POISON_RATES,DEFAULT_SELECT_RATES,
        DEFAULT_SEEDS,build_cases,train_size,validate_budget)
    if options.reduce_seeds and not (options.stability and options.run and options.resume and options.seeds is not None):
        parser.error('--reduce-seeds requires --stability --run --resume and explicit --seeds')
    if options.resume and not options.run:parser.error('--resume requires --run')
    if options.seeds is None:
        options.seeds=list(DEFAULT_SEEDS if options.stability else
                          ([2004,592,3383,100] if options.dataset=='NUSWIDE' else [2004,592,3383]))
    if len(set(options.seeds))!=len(options.seeds) or any(not 0<=s<=2**32-1 for s in options.seeds) or options.block_size<1:
        parser.error('Unique seeds within [0,2**32-1] and positive block size required')
    if options.stability:
        if options.dataset!='CIFAR10':parser.error('Stability mode currently requires CIFAR10')
        if options.anchor_source=='auto_vote':parser.error('New stability compares full strategies with a fixed LFBA anchor; auto_vote is the legacy same-anchor protocol')
        options.anchor_source='fixed'
        if options.poison_rate!=.1 or options.select_rate!=.5:
            parser.error('Use --poison-rates/--select-rates for stability scans; baseline is p=.1,s=.5')
        options.poison_rates=list(DEFAULT_POISON_RATES if options.poison_rates is None else options.poison_rates)
        options.select_rates=list(DEFAULT_SELECT_RATES if options.select_rates is None else options.select_rates)
        for values in (options.poison_rates,options.select_rates):
            if len(set(values))!=len(values):parser.error('Duplicate parameter values are not allowed')
        cases=build_cases(options)
    else:
        if options.poison_rates is not None or options.select_rates is not None:
            parser.error('Parameter lists require --stability')
        options.anchor_source=options.anchor_source or 'fixed'
        cases=[dict(poison_rate=options.poison_rate,select_rate=options.select_rate)]
    try:
        for case in cases:
            if options.dataset=='NUSWIDE':
                # The filtered population is data-dependent; check its budget at preflight.
                # Parsing/report reconstruction must work without the original data files.
                if any(not math.isfinite(v) or not 0<v<=1 for v in (case['poison_rate'],case['select_rate'])):
                    raise ValueError('Poison and select rates must be finite and within (0,1]')
            else:validate_budget(train_size(options),case['poison_rate'],case['select_rate'],5 if options.smoke else 40)
    except ValueError as error:parser.error(str(error))
    if options.output is None:
        options.output=ROOT/'results'/('lfba_stability_full_smoke' if options.smoke else 'lfba_stability_full') if options.stability else ROOT/'results'/('lfba_auto_vote_smoke' if options.smoke else 'lfba_auto_vote_formal')
        if options.dataset=='NUSWIDE':
            options.output=ROOT/'results'/('nuswide_lfba_auto_vote_smoke' if options.smoke else 'nuswide_lfba_auto_vote_formal')
        elif options.dataset=='UCIHAR':
            options.output=ROOT/'results'/('ucihar_lfba_auto_vote_smoke' if options.smoke else 'ucihar_lfba_auto_vote_formal')
    options.output=options.output.resolve();options.data_dir=options.data_dir.resolve();options.code_root=ROOT
    return options


def dry_run(options):
    from lfba_comparison.policy import ANCHORS, METHODS
    if options.output==options.data_dir or options.data_dir in options.output.parents:
        raise ValueError('Output must not be inside shared input data')
    if options.dataset=='NUSWIDE':
        from lfba_comparison.nuswide import dataset_profile
        from lfba_comparison.stability_config import validate_budget
        try:
            profile=dataset_profile(options.data_dir,options.smoke)
        except FileNotFoundError as error:
            print(json.dumps(dict(mode='DRY_RUN_ONLY',training_started=False,dataset=options.dataset,
                data_dir=str(options.data_dir),missing_data_files=[str(error)]),ensure_ascii=False,indent=2))
            return 2
        if not 0<=ANCHORS[options.dataset]<profile['source_train_size']:
            raise ValueError('Fixed LFBA anchor is outside the filtered NUS-WIDE training source')
        epochs=5 if options.smoke else 40
        budget=validate_budget(profile['train_size'],options.poison_rate,options.select_rate,epochs)
        print(json.dumps(dict(mode='DRY_RUN_ONLY',training_started=False,dataset=options.dataset,
            seeds=options.seeds,methods=list(METHODS),tasks=len(METHODS)*len(options.seeds),
            epochs=epochs,anchor_source=options.anchor_source,fixed_lfba_anchor=ANCHORS[options.dataset],
            poison_rate=options.poison_rate,select_rate=options.select_rate,
            data_dir=str(options.data_dir),missing_data_files=[],output=str(options.output),
            output_occupied=options.output.exists() and any(options.output.iterdir()),
            formal_budgets=None if options.smoke else budget['budgets'],**budget,**profile,
            note='Label counts and file manifest only; feature shapes are checked on load. No training or CUDA initialization.'),
            ensure_ascii=False,indent=2))
        return 0
    if options.dataset=='CIFAR10':
        required=[Path('cifar-10-batches-py')/name for name in
                  [f'data_batch_{i}' for i in range(1,6)]+['test_batch','batches.meta']]
    else:
        required=[Path('UCI-HAR')/'UCI HAR Dataset'/split/f'{prefix}_{split}.txt'
                  for split in ('train','test') for prefix in ('X','y')]
    missing=[str(options.data_dir/name) for name in required if not (options.data_dir/name).is_file()]
    if options.output==options.data_dir or options.data_dir in options.output.parents:
        raise ValueError('Output must not be inside shared input data')
    epochs=5 if options.smoke else 40
    train_size=(450 if options.smoke else (45000 if options.dataset=='CIFAR10' else 6616))
    per_epoch=int(int(train_size*options.poison_rate)*options.select_rate)
    if options.stability:
        from lfba_comparison.stability_config import build_cases,validate_budget
        cases=[dict(case,**validate_budget(train_size,case['poison_rate'],case['select_rate'],epochs))
               for case in build_cases(options)]
        pairs=len(cases)*len(options.seeds)
        print(json.dumps(dict(mode='DRY_RUN_ONLY',training_started=False,stability=True,
            seeds=options.seeds,dataset=options.dataset,anchor_source='fixed',fixed_lfba_anchor=ANCHORS[options.dataset],execution_order=['auto_vote','lfba_original'],
            methods=['auto_vote','lfba_original'],pairs=pairs,tasks=2*pairs,epochs=epochs,
            train_size=train_size,train_source_limit=500 if options.smoke else None,
            cases=cases,data_dir=str(options.data_dir),missing_data_files=missing,
            output=str(options.output),output_occupied=options.output.exists() and any(options.output.iterdir())),
            ensure_ascii=False,indent=2))
        return 2 if missing else 0
    print(json.dumps(dict(mode='DRY_RUN_ONLY',training_started=False,seeds=options.seeds,
        methods=list(METHODS),tasks=len(METHODS)*len(options.seeds),dataset=options.dataset,
        data_dir=str(options.data_dir),missing_data_files=missing,output=str(options.output),
        output_occupied=options.output.exists() and any(options.output.iterdir()),fixed_lfba_anchor=ANCHORS[options.dataset],
        epochs=epochs,poison_rate=options.poison_rate,select_rate=options.select_rate,
        candidates=int(train_size*options.poison_rate),replacements_per_epoch=per_epoch,
        formal_budgets=None if options.smoke else dict(
            lfba_original=per_epoch*(epochs-1),
            auto_vote=per_epoch*(epochs-3)),
        note='No model creation, training, inference or CUDA initialization.'),ensure_ascii=False,indent=2))
    return 2 if missing else 0


def main(argv=None):
    options=parse_options(argv)
    if options.stability:
        if options.report_only:
            from lfba_comparison.stability import report_stability
            return report_stability(options.output)
        if not options.run:return dry_run(options)
        if dry_run(options):raise FileNotFoundError('CIFAR10 files missing; experiment not started')
        from lfba_comparison.stability import run_stability
        return run_stability(options)
    if options.report_only:
        from lfba_comparison.report import write_report
        from lfba_comparison.runner import cleanup_completed_batch
        from lfba_comparison.common import batch_lock
        with batch_lock(options.output):
            write_report(options.output)
            cleanup_completed_batch(options.output)
        return 0
    if not options.run:return dry_run(options)
    if dry_run(options):raise FileNotFoundError(f'{options.dataset} files missing; experiment not started')
    from lfba_comparison.runner import run_experiment
    run_experiment(options);return 0


if __name__=='__main__':raise SystemExit(main())
