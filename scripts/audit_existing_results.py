"""Read-only saved-result analysis; writes exclusively to a fresh output directory."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from lfba_comparison.results_audit import (scan,snapshot,verify_snapshot,deduplicate,
    aggregate,paired,write_json,write_csv,update_batch_counts)
from lfba_comparison.audit_diagnostics import diagnose
from lfba_comparison.audit_report import render,plot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--refresh-audit',action='store_true',help='Refresh only a directory created by this audit tool')
    opts=parser.parse_args(argv)
    root=opts.root.resolve();data=(opts.data_dir or root/'data').resolve()
    output=(opts.output or root/'results/existing_results_audit_20261112').resolve()
    if output==root or output==data or data in output.parents:
        raise ValueError('Output must be separate from code root and raw inputs')
    if output.exists():
        from lfba_comparison.results_audit import read
        marker=read(output/'audit_config.json',{})
        if (not opts.refresh_audit or marker.get('root')!=str(root)
                or marker.get('training') is not False or marker.get('inference') is not False):
            raise FileExistsError('Use a fresh directory, or --refresh-audit for an existing audit output: '+str(output))
    # Snapshot all lightweight historical artifacts, not just files eventually cited.
    inputs=[]
    for base in (root/'results',root/'scripts/results'):
        if base.exists():
            inputs.extend(p for p in base.rglob('*') if p.is_file() and not p.is_relative_to(output) and
                          p.suffix.lower() in ('.json','.csv','.md','.log','.png'))
    inputs.extend((data/'cifar-10-batches-py').glob('data_batch_*'))
    inputs.extend((data/'UCI-HAR/UCI HAR Dataset/train').glob('*.txt'))
    inputs.extend((data/'NUS_WIDE/Groundtruth/TrainTestLabels').glob('Labels_*_Train.txt'))
    for name in ('dataset/experiment.py','utils/trainer.py','attack/neighbor_vote.py'):
        if (root/name).exists():inputs.append(root/name)
    print('Hashing historical artifacts and raw inputs...',flush=True)
    before=snapshot(inputs)
    rows,batches,issues=scan(root,output)
    deduplicate(rows);update_batch_counts(rows,batches)
    groups=aggregate(rows);pairs=paired(rows)
    for row in rows:
        if any('Conflicting metrics' in p for p in row.get('problems',[])):
            issues.append(dict(source=row['source'],error='; '.join(row['problems'])))
    print(f'Inventory: {len(rows)} records; diagnosing actual saved plans...',flush=True)
    diagnostics,curves,triggers,used_inputs=diagnose(rows,data,issues)
    if any(str(p.resolve()) not in before for p in used_inputs):
        raise ValueError('A raw input was not snapshotted before analysis')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'verification.json',dict(status='in_progress',original_inputs_unchanged=None,
        training_started=False,inference_started=False))
    write_json(output/'audit_config.json',dict(root=str(root),data_dir=str(data),
        training=False,inference=False,original_reports_modified=False,
        duplicate_policy='data + seed + target + evaluation epoch + candidate + final model digest',
        raw_label_assumption='original source ordering; UCIHAR full training source fingerprint checked'))
    for name,records in [('master',rows),('batch_status',batches),('group_summary',groups),
                         ('paired',pairs),('diagnostics',diagnostics),('validation_history',curves),
                         ('ucihar_triggers',triggers)]:
        write_json(output/(name+'.json'),records);write_csv(output/(name+'.csv'),records)
    write_json(output/'duplicates.json',[{k:r.get(k) for k in ('source','experiment_id','duplicate_of','reference_to','origin')}
                                       for r in rows if r.get('duplicate_of') or r.get('origin')=='reference'])
    write_json(output/'issues.json',issues)
    write_json(output/'input_manifest.json',before)
    plot(output,diagnostics,curves)
    render(output,rows,batches,groups,pairs,diagnostics,triggers,issues,len(before))
    verify_snapshot(before)
    write_json(output/'verification.json',dict(status='complete',original_inputs_unchanged=True,files_checked=len(before),
        complete_records=sum(r['status']=='complete' for r in rows),diagnosed_records=len(diagnostics),
        issues=len(issues),training_started=False,inference_started=False))
    print('AUDIT_COMPLETE',output,flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
