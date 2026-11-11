"""Reproducible two-party LFBA experiment entry point."""
import copy
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.experiment import build_datasets
from models.CIFAR10_models import GlobalModelForCifar10, LocalModelForCifar10
from models.UCIHAR_models import GlobalModelForUCIHAR, LocalModelForUCIHAR
from models.PHISHING_models import GlobalModelForPHISHING, LocalModelForPHISHING
from models.NUSWIDE_models import GlobalModelForNUSWIDE, LocalModelForNUSWIDE
from utils.checkpoint import VERSION, load_checkpoint, cpu_tree
from utils.artifacts import save_configuration, save_experiment_data, verify_saved_result, cleanup_runtime
from lfba_comparison.common import batch_lock, write_json, digest
from utils.config import parse_args, validate_args
from utils.trainer import Trainer
from utils.utils import set_seed


def resolve_checkpoint(args):
    path=args.pretrained_checkpoint or args.test_checkpoint
    if not path:
        return validate_args(args),None
    checkpoint=load_checkpoint(path)
    if checkpoint.get('format_version')!=VERSION:
        if args.pretrained_checkpoint:
            raise ValueError('Legacy checkpoint cannot be strictly resumed')
        supplied=set(args._provided)
        if not {'dataset','target_label'}.issubset(supplied):
            raise ValueError('Legacy evaluation requires explicit --dataset and --target_label')
        if args.dataset not in ('CIFAR10',) and args.trigger_dimensions is None:
            raise ValueError('Legacy table-data evaluation requires --trigger_dimensions')
        return validate_args(args),checkpoint
    if args.pretrained_checkpoint and checkpoint.get('evaluation_only'):
        raise ValueError('This reconstructed best checkpoint is evaluation-only; resume from last_checkpoint')
    supplied=set(args._provided)
    allowed={'device','results_dir','epoch','pretrained_checkpoint','test_checkpoint'}
    saved=checkpoint['config']
    for key in supplied-allowed:
        value=getattr(args,key)
        if key=='dataset' and value=='NUSWIDET':
            value='NUSWIDE'
        if key=='data_dir':
            value=str(Path(value).resolve())
        if value!=saved.get(key):
            raise ValueError('Checkpoint configuration conflicts with --'+key)
    merged=dict(saved)
    for key in allowed:
        if key in supplied or key in ('results_dir','pretrained_checkpoint','test_checkpoint'):
            merged[key]=getattr(args,key)
    merged['_provided']=args._provided
    merged['start_epoch']=checkpoint['epoch'] if args.pretrained_checkpoint else 0
    args=validate_args(SimpleNamespace(**merged))
    if args.pretrained_checkpoint and args.epoch<=checkpoint['epoch']:
        raise ValueError('Resume epoch must exceed the number of completed epochs')
    return args,checkpoint


def build_models(args):
    pairs={'CIFAR10':(GlobalModelForCifar10,LocalModelForCifar10),
           'UCIHAR':(GlobalModelForUCIHAR,LocalModelForUCIHAR),
           'PHISHING':(GlobalModelForPHISHING,LocalModelForPHISHING),
           'NUSWIDE':(GlobalModelForNUSWIDE,LocalModelForNUSWIDE)}
    global_class,local_class=pairs[args.dataset]
    local=[local_class(args) if args.dataset=='CIFAR10' else local_class(args,i) for i in range(2)]
    return [global_class(args),*local]


def triggered_view(view,args,dimensions):
    triggered=copy.copy(view)
    triggered.base=copy.copy(view.base)
    triggered.set_attack({i:None for i in view.ids},args,dimensions)
    return triggered


def trigger_dimensions(args,checkpoint):
    if args.dataset=='CIFAR10':
        return []
    start,end={'PHISHING':(15,30),'UCIHAR':(281,561),'NUSWIDE':(634,1634)}[args.dataset]
    if checkpoint and checkpoint.get('format_version')==VERSION:
        dims=list(checkpoint['trigger_dimensions'])
    elif args.trigger_dimensions is not None:
        dims=list(args.trigger_dimensions)
    else:
        if args.poison_dimensions>end-start:
            raise ValueError('poison_dimensions exceeds attacker feature count')
        dims=np.random.choice(np.arange(start,end),args.poison_dimensions,replace=False).tolist()
    if len(dims)==0 or len(set(dims))!=len(dims) or any(not start<=i<end for i in dims):
        raise ValueError('trigger_dimensions must be unique attacker-side feature indices')
    return dims


def validate_data(args,train,val,test):
    classes={'CIFAR10':10,'UCIHAR':6,'PHISHING':2,'NUSWIDE':5}[args.dataset]
    if not 0<=args.target_label<classes:
        raise ValueError('target_label is outside the dataset class range')
    for name,view in (('train',train),('val',val),('test',test)):
        if not len(view):
            raise ValueError('Empty '+name+' split')
        labels=np.asarray(view.targets)[np.asarray(view.ids)-view.offset]
        if not np.isfinite(labels).all() or not np.equal(labels,labels.astype(np.int64)).all() or np.any((labels<0)|(labels>=classes)):
            raise ValueError('Invalid labels in '+name+' split')
    if args.attack=='LFBA' and not args.test_checkpoint:
        count=int(len(train)*args.poison_rate)
        if count<1:
            raise ValueError('poison_rate selects no LFBA samples')
        selected=int(count*args.select_rate)
        if not args.poison_all and (selected<1 or selected>len(train)-count):
            raise ValueError('LFBA replacement needs selected samples and enough clean donors')
        if args.anchor_idx not in train.ids:
            raise ValueError('anchor_idx must belong to training split')


def main(args):
    args,checkpoint=resolve_checkpoint(args)
    with batch_lock(Path(args.results_dir)):
        return _run_experiment(args,checkpoint)


def _run_experiment(args,checkpoint):
    args.data_dir=str(Path(args.data_dir).resolve())
    output=Path(args.results_dir)
    # Never overwrite historical or completed experiment outputs accidentally.
    if output.exists() and any((output/name).exists() for name in ('experiment.log','config.json','metrics.json')):
        raise FileExistsError('Experiment output is already occupied: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('lfba')
    logger.setLevel(logging.INFO)
    logger.propagate=False
    handler=logging.FileHandler(output/'experiment.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    stream=logging.StreamHandler()
    stream.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.addHandler(stream)
    try:
        started=time.perf_counter()
        set_seed(args.seed)
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        device=torch.device('cuda:'+str(args.device) if args.device>=0 and torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        data_state=checkpoint.get('data_state') if checkpoint and checkpoint.get('format_version')==VERSION else None
        train,val,test,data_state=build_datasets(args,data_state)
        if args.attack=='LFBA' and not args.test_checkpoint:
            if args.anchor_idx not in train.ids:
                raise ValueError('anchor_idx is not in training split')
            args.target_label=int(train.targets[args.anchor_idx])
        validate_data(args,train,val,test)
        dims=trigger_dimensions(args,checkpoint)
        args.trigger_dimensions=dims
        val_asr=triggered_view(val,args,dims)
        test_asr=triggered_view(test,args,dims)
        datasets=dict(train=train,val=val,val_asr=val_asr,test=test,test_asr=test_asr)
        loaders={name:DataLoader(data,batch_size=args.batch_size,shuffle=name=='train',num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed+index+1))
                 for index,(name,data) in enumerate(datasets.items())}
        models=[model.to(device) for model in build_models(args)]
        opts=[torch.optim.Adam(model.parameters(),lr=args.lr) for model in models]
        trainer=Trainer(device,models,[],None,opts,nn.CrossEntropyLoss(),loaders['train'],
                        loaders['test'],loaders['test_asr'],dims,logger,args,
                        checkpoint if args.pretrained_checkpoint else None,
                        loaders['val'],loaders['val_asr'],data_state)
        save_configuration(output,trainer)
        logger.info('Configuration: %s',json.dumps(vars(args)))
        logger.info('Split sizes train=%d validation=%d test=%d',len(train),len(val),len(test))
        if args.test_checkpoint:
            if len(checkpoint['state_dict'])!=len(models):
                raise ValueError('Checkpoint model count differs from this experiment')
            for model,state in zip(models,checkpoint['state_dict']):
                model.load_state_dict(state)
            evaluated_epoch=checkpoint['epoch']
            best_epoch=checkpoint.get('best_epoch',evaluated_epoch)
            validation=next((row['validation'] for row in checkpoint.get('history',[])
                             if row['epoch']==evaluated_epoch),None)
        else:
            trainer.train()
            if trainer.best_state_dict is None:
                raise ValueError('No defined validation score; cannot select a best model')
            for model,state in zip(models,trainer.best_state_dict):
                model.load_state_dict(state)
            best_epoch=trainer.best_epoch
            evaluated_epoch=best_epoch
            validation=trainer.best_metrics

        metrics=trainer.test(evaluated_epoch-1)
        evaluation_state=cpu_tree([m.state_dict() for m in models])
        independent_models=[model.to(device) for model in build_models(args)]
        for model,state in zip(independent_models,evaluation_state):
            model.load_state_dict(state)
        evaluator=copy.copy(trainer)
        evaluator.model_list=independent_models
        independent=evaluator.test(evaluated_epoch-1)
        if independent!=metrics:
            raise ValueError('Independent evaluation differs')
        save_experiment_data(output,trainer)
        result=dict(best_epoch=best_epoch,checkpoint_epoch=evaluated_epoch,validation=validation,test=metrics,
                    elapsed_seconds=time.perf_counter()-started,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                    status='complete',independent_evaluation=dict(epoch=evaluated_epoch,test=independent,
                    final_model_digest=digest(evaluation_state)))
        write_json(output/'metrics.json',result)
        verify_saved_result(output,result,trainer)
        cleanup_runtime(output)
        logger.info('Final result: %s',json.dumps(result))
        return result
    finally:
        for active in (handler,stream):
            logger.removeHandler(active)
            active.close()


if __name__=='__main__':
    main(parse_args())
