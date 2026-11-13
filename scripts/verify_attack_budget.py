"""Independent checkpoint/CSV/manifest checks and fresh final-checkpoint evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.attack_budget import GROUPS, tree_digest
from experiments.json_io import atomic_write_json
from main import main
from utils.checkpoint import load_checkpoint
from utils.config import parse_args


def verify(root):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifests=json.loads((root/'source_hashes.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(root/'source_snapshot.zip') as archive:
        assert all(hashlib.sha256(archive.read(p.replace('\\','/'))).hexdigest()==h for p,h in manifests.items())
    changed=[p for p,h in manifests.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert all(p.replace('\\','/')=='scripts/run_attack_budget.py' for p in changed),changed
    checks={}
    for group in GROUPS:
        output=root/group
        checkpoint=load_checkpoint(output/'last_checkpoint.pth.tar')
        state=checkpoint['budget_experiment']
        result=summary[group]
        assert checkpoint['epoch']==result['epoch']==20
        assert len(checkpoint['data_state']['train_ids'])==45000
        assert len(checkpoint['data_state']['val_ids'])==5000
        assert len(checkpoint['data_state']['test_ids'])==10000
        assert checkpoint['config']['seed']==100 and checkpoint['config']['anchor_idx']==23470
        assert len(checkpoint['attack_state']['poison_indexes'])==4500
        assert tree_digest(checkpoint['state_dict'])==result['final_model_digest']
        rows=list(csv.DictReader((output/'epoch_metrics.csv').open(encoding='utf-8')))
        assert len(rows)==len(state['audit_rows'])==20
        expected=[0]*20
        if group=='early_single': expected[1:]=[2250]*19
        elif group=='early_matched': expected[1:4]=[2014]*3;expected[4:]=[2013]*16
        else: expected[3:]=[2250]*17
        counts=[int(row['recipient_count']) for row in rows]
        assert counts==expected==state['schedule']
        assert sum(counts)==result['cumulative_replacements']==state['cumulative_replacements']
        assert sum(counts)==(42750 if group=='early_single' else 38250)
        assert len({row['candidate_digest'] for row in rows if int(row['candidate_count'])})==1
        eval_dir=root/'independent_eval'/group
        if (eval_dir/'metrics.json').exists():
            reevaluated=json.loads((eval_dir/'metrics.json').read_text(encoding='utf-8'))
        else:
            reevaluated=main(parse_args(['--test_checkpoint',str(output/'last_checkpoint.pth.tar'),
                                        '--results_dir',str(eval_dir)]))
        assert reevaluated['checkpoint_epoch']==20
        assert reevaluated['test']==result['test'],group
        assert result['test']['counts']['acc']==10000 and result['test']['counts']['asr']==9000
        last5=[float(row['validation_asr']) for row in rows[-5:]]
        checks[group]=dict(epoch=20,budget=sum(counts),checkpoint_and_csv_match=True,
            independent_evaluation_identical=True,final_test=result['test'],
            last5_validation_asr_mean=statistics.mean(last5),
            last5_validation_asr_population_std=statistics.pstdev(last5))
    assert len({r['initial_digest'] for r in summary.values()})==1
    assert len({r['data_digest'] for r in summary.values()})==1
    assert summary['delayed_single']['prefix_digests']==summary['multi_mean']['prefix_digests']
    result=dict(groups=checks,source_archive_verified=True,
        post_training_changed_source=changed,training_policy_source_unchanged=True,
        initialization_identical=True,data_splits_identical=True,delayed_clean_prefix_identical=True)
    atomic_write_json(root/'final_verification.json',result)
    print('FINAL_VERIFICATION_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify(Path(sys.argv[1]))
