"""Run actual CIFAR/GPU paths on explicitly bounded data, preserving all artifacts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from main import main
from utils.config import parse_args


def run(root):
    results={}
    variants={'baseline':[], 'rsa':['--attack','rsa'], 'lra':['--attack','lra'],
              'lfba':['--attack','LFBA'], 'poison_all':['--attack','LFBA','--poison_all'],
              'random_select':['--attack','LFBA','--random_select']}
    common=['--dataset','CIFAR10','--data_dir',str(ROOT/'data'),'--device','0',
            '--epoch','2','--batch_size','32','--seed','100','--early_stop','0',
            '--anchor_idx','23470','--target_label','6','--poison_rate','.1','--select_rate','.5',
            '--train_limit','200','--eval_limit','100','--print_steps','100']
    for name,flags in variants.items():
        results[name]=main(parse_args([*common,*flags,'--results_dir',str(root/name)]))
    assert results['lfba']['test']==results['lfba']['independent_evaluation']['test']
    results['independent_eval']=results['lfba']['independent_evaluation']
    (root/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('SMOKE_ACCEPTANCE_PASSED',root,flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]))
