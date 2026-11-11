"""Compare real CIFAR/GPU continuous training with the saved smoke resume."""
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from main import main
from utils.config import parse_args
from utils.checkpoint import load_checkpoint
from utils.artifacts import cleanup_runtime, runtime_checkpoint


def run(smoke_root,output):
    flags=['--dataset','CIFAR10','--data_dir',str(ROOT/'data'),'--device','0',
           '--epoch','3','--batch_size','32','--seed','100','--early_stop','0',
           '--anchor_idx','23470','--target_label','6','--poison_rate','.1','--select_rate','.5',
           '--train_limit','200','--eval_limit','100','--print_steps','100','--attack','LFBA',
           '--results_dir',str(output/'continuous')]
    snapshots=[]
    def capture_then_cleanup(directory):
        snapshots.append(load_checkpoint(runtime_checkpoint(directory)))
        cleanup_runtime(directory)
    # Capture verification state in memory; successful outputs keep no models.
    with patch('main.cleanup_runtime',side_effect=capture_then_cleanup):
        main(parse_args(flags))
        full=snapshots[-1]
        partial=list(flags)
        partial[partial.index('--epoch')+1]='2'
        partial[partial.index('--results_dir')+1]=str(output/'part')
        main(parse_args(partial))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'resume.pth.tar'
            torch.save(snapshots[-1],checkpoint)
            main(parse_args(['--pretrained_checkpoint',str(checkpoint),'--epoch','3',
                             '--results_dir',str(output/'resumed')]))
        resumed=snapshots[-1]
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
                resumed_checkpoint='temporary verification state, removed after completion')
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GPU_RESUME_COMPARISON_PASSED',json.dumps(report),flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
