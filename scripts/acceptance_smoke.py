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
    original=results['lfba']['test']
    evaluated=main(parse_args(['--test_checkpoint',str(root/'lfba'/'best_checkpoint.pth.tar'),
                              '--results_dir',str(root/'independent_eval')]))
    assert original==evaluated['test'], 'Standalone GPU evaluation differs'
    results['independent_eval']=evaluated
    results['resume']=main(parse_args(['--pretrained_checkpoint',str(root/'lfba'/'last_checkpoint.pth.tar'),
                                       '--epoch','3','--results_dir',str(root/'resume')]))
    (root/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('SMOKE_ACCEPTANCE_PASSED',root,flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]))
