"""Compare real CIFAR/GPU continuous training with the saved smoke resume."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from main import main
from utils.config import parse_args
from utils.checkpoint import load_checkpoint


def run(smoke_root,output):
    flags=['--dataset','CIFAR10','--data_dir',str(ROOT/'data'),'--device','0',
           '--epoch','3','--batch_size','32','--seed','100','--early_stop','0',
           '--anchor_idx','23470','--target_label','6','--poison_rate','.1','--select_rate','.5',
           '--train_limit','200','--eval_limit','100','--print_steps','100','--attack','LFBA',
           '--results_dir',str(output/'continuous')]
    main(parse_args(flags))
    resumed=load_checkpoint(smoke_root/'resume'/'last_checkpoint.pth.tar')
    full=load_checkpoint(output/'continuous'/'last_checkpoint.pth.tar')
    for key in ['poison_indexes','current_plan','label_overrides']:
        assert resumed['attack_state'][key]==full['attack_state'][key],key
    assert resumed['best_epoch']==full['best_epoch']
    assert resumed['data_state']==full['data_state']
    maximum=0.
    for a,b in zip(resumed['state_dict'],full['state_dict']):
        for key,value in a.items():
            torch.testing.assert_close(value,b[key],rtol=1e-5,atol=1e-6)
            maximum=max(maximum,float((value.double()-b[key].double()).abs().max()))
    for a,b in zip(resumed['optimizer'],full['optimizer']):
        assert a['param_groups']==b['param_groups']
        for index,state in a['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,b['state'][index][key],rtol=1e-5,atol=1e-6)
    report=dict(passed=True,epochs=3,train_candidates=200,seed=100,rtol=1e-5,atol=1e-6,
                maximum_model_absolute_difference=maximum,
                resumed_checkpoint=str(smoke_root/'resume'/'last_checkpoint.pth.tar'))
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GPU_RESUME_COMPARISON_PASSED',json.dumps(report),flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
