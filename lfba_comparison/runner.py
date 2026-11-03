"""Explicit orchestration. Importing this module performs no experiment."""
import csv
import hashlib
import logging
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.experiment import build_datasets
from main import build_models, triggered_view, validate_data
from utils.checkpoint import load_checkpoint
from utils.config import parse_args
from utils.utils import set_seed
from lfba_comparison.common import batch_lock, digest, peak_host_memory_bytes, read_json, write_json
from lfba_comparison.policy import ANCHOR, METHODS, purity, schedule
from lfba_comparison.trainer import ComparisonTrainer


def frozen_protocol(options):
    return dict(version=1,seeds=options.seeds,smoke=options.smoke,device=options.device,
        block_size=options.block_size,data_dir=str(options.data_dir),anchor=ANCHOR,
        methods=list(METHODS),epochs=5 if options.smoke else 20,batch_size=128,lr=.001,
        optimizer='Adam',lr_decay=False,early_stop=False,candidate_rate=.1,
        split='stratified 90/10 with anchor 23470 retained for all methods',
        final_model='last',baseline_similarity='legacy_l1',vote_similarity='cosine',
        primary='auto_vote - lfba_original',supplement='auto_vote - lfba_matched',
        thresholds=dict(mean_asr_gain=.02,mean_acc_loss_limit=.01,positive_fraction=2/3),
        interpretation='full strategy; fixed LFBA target versus free automatic target')


def runtime_sources(root):
    files=[root/'main.py']
    for name in ('attack','dataset','models','utils','scripts','lfba_comparison'):
        files.extend((root/name).rglob('*.py'))
    return {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def configuration(options,seed,method,output):
    flags=['--dataset','CIFAR10','--data_dir',str(options.data_dir),'--results_dir',str(output),
        '--device',str(options.device),'--seed',str(seed),'--epoch',str(1 if method=='capture' else (5 if options.smoke else 20)),
        '--batch_size','128','--lr','.001','--early_stop','0','--anchor_idx',str(ANCHOR),
        '--attack','LFBA','--poison_rate','.1','--select_rate','.5','--pretrain_stage',
        '3' if method=='auto_vote' else '1','--print_steps','200','--target_label','0']
    if options.smoke: flags+=['--train_limit','200','--eval_limit','100']
    args=parse_args(flags)
    args.comparison_method=method
    args.block_size=options.block_size
    args.similarity='cosine' if method=='auto_vote' else 'legacy_l1'
    args.trigger_dimensions=[]
    return args


def assemble(options,seed,method,output,checkpoint=None):
    args=configuration(options,seed,method,output)
    set_seed(seed);torch.set_num_threads(4)
    device=torch.device('cpu' if options.device<0 else f'cuda:{options.device}')
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; explicitly choose --device -1 for CPU')
        torch.cuda.set_device(device)
    train,val,test,state=build_datasets(args,checkpoint['data_state'] if checkpoint else None)
    validate_data(args,train,val,test)
    if not options.smoke and [len(train),len(val),len(test)]!=[45000,5000,10000]:
        raise ValueError('Formal split must be 45000/5000/10000')
    views=dict(train=train,val=val,val_asr=triggered_view(val,args,[]),test=test,test_asr=triggered_view(test,args,[]))
    loaders={name:DataLoader(view,batch_size=128,shuffle=name=='train',num_workers=0,
        generator=torch.Generator().manual_seed(seed+i+1)) for i,(name,view) in enumerate(views.items())}
    models=[m.to(device) for m in build_models(args)]
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
    output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('lfba_comparison.'+str(output));logger.setLevel(logging.INFO);logger.propagate=False
    handler=logging.FileHandler(output/'experiment.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));logger.addHandler(handler)
    trainer=ComparisonTrainer(device,models,[],None,optimizers,nn.CrossEntropyLoss(),
        loaders['train'],loaders['test'],loaders['test_asr'],[],logger,args,checkpoint,
        loaders['val'],loaders['val_asr'],state)
    return trainer,handler


def close(trainer,handler):
    trainer.logger.removeHandler(handler);handler.close()


def export(output,trainer):
    write_json(output/'config.json',vars(trainer.args))
    write_json(output/'history.json',trainer.history)
    write_json(output/'replacement_ids.json',trainer.plans)
    if trainer.choice:write_json(output/'choice.json',trainer.choice)
    rows=[]
    for audit in trainer.audit:
        row={k:v for k,v in audit.items() if k!='validation'}
        row.update({'val_'+k:audit['validation'].get(k) for k in ('acc','asr','poison_acc')})
        rows.append(row)
    with (output/'epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_checkpoint(checkpoint,method,epochs,prefix):
    state=checkpoint['comparison'];choice=state['choice']
    if checkpoint['epoch']!=epochs or state['method']!=method or choice is None:
        raise ValueError('Final checkpoint method or epoch differs')
    if checkpoint['data_state']!=prefix['data_state']:
        raise ValueError('Data split differs from shared prefix')
    count=int(len(prefix['data_state']['train_ids'])*.1)
    candidates=choice['candidates'];pool=set(candidates)
    if len(pool)!=count or len(candidates)!=count or not pool<=set(prefix['data_state']['train_ids']):
        raise ValueError('Candidate count or membership differs')
    clean=3 if method=='auto_vote' else 1
    expected=schedule(method,epochs,count)
    if [o['epoch'] for o in state['observations']]!=list(range(1,clean+1)):
        raise ValueError('Observation history differs')
    expected_start=digest(dict(model=prefix['state_dict'],optimizer=prefix['optimizer'],rng=prefix['rng_state']))
    if (state['initial_digest']!=prefix['comparison']['initial_digest'] or state['start_state']!=expected_start
            or state['prefix'][0]!=prefix['comparison']['prefix'][0]):
        raise ValueError('Initialization or actual first-round restore differs')
    if state['cumulative']!=sum(expected) or len(state['audit'])!=epochs:
        raise ValueError('Total budget or epoch history differs')
    for i,row in enumerate(state['audit']):
        if (row['epoch']!=i+1 or row['recipient_count']!=expected[i]
                or row['cumulative_replacements']!=sum(expected[:i+1])):
            raise ValueError('Per-epoch budget differs')
        if i<clean:
            if row['candidate_count']!=0:raise ValueError('Premature candidate construction')
        elif row['candidate_count']!=count or row['candidate_digest']!=digest(candidates):
            raise ValueError('Frozen candidate digest differs')
    plans=state['plans']
    if [p['epoch'] for p in plans]!=list(range(clean+1,epochs+1)):
        raise ValueError('Poisoning start or plan history differs')
    train=set(prefix['data_state']['train_ids'])
    for p in plans:
        n=expected[p['epoch']-1]
        if (len(p['recipients'])!=n or len(set(p['recipients']))!=n or len(p['donors'])!=n
                or len(set(p['donors']))!=n or not set(p['recipients'])<=pool
                or not set(p['donors'])<=train-pool):raise ValueError('Invalid replacement plan')
    if checkpoint['attack_state']['poison_indexes']!=candidates or checkpoint['attack_state']['label_overrides']:
        raise ValueError('Candidates or label overrides differ')


def run_task(options,seed,method,directory,prefix):
    output=directory/method;last=output/'last_checkpoint.pth.tar'
    checkpoint=load_checkpoint(last) if last.exists() else prefix
    trainer,handler=assemble(options,seed,method,output,checkpoint)
    try:
        trainer.train() # only reachable through explicit CLI --run
        final=load_checkpoint(last);verify_checkpoint(final,method,trainer.args.epoch,prefix)
        started=time.perf_counter();metrics=trainer.test(trainer.args.epoch-1,'test')
        evaluation_seconds=time.perf_counter()-started
        export(output,trainer)
        labels=np.asarray(trainer.train_loader.dataset.targets)
        quality=purity(labels,trainer.choice['candidates'],trainer.choice['anchor_id'])
        replaced=[i for plan in trainer.plans for i in plan['recipients']]
        rows=trainer.audit;clean=trainer.clean_window
        result=dict(seed=seed,method=method,epoch=trainer.args.epoch,test=metrics,
            anchor=trainer.choice['anchor_id'],target_class=trainer.args.target_label,
            candidate_purity=quality,recipient_purity=float(np.mean(labels[replaced]==trainer.args.target_label)),
            cumulative_replacements=trainer.cumulative,schedule=schedule(method,trainer.args.epoch,len(trainer.choice['candidates'])),
            candidate_digest=digest(trainer.choice['candidates']),initial_digest=trainer.initial_digest,
            data_digest=digest(trainer.data_state),first_prefix=trainer.prefix[0],start_state=trainer.start_state,
            final_model_digest=digest(final['state_dict']),
            timing=dict(shared_first_round=trainer.shared_clean_seconds,
                additional_clean_epochs=sum(r['epoch_seconds'] for r in rows[1:clean]),
                selection=trainer.selection_seconds,
                poison_epochs=sum(r['epoch_seconds']-r['selection_seconds'] for r in rows[clean:]),
                elapsed_branch=final['comparison']['elapsed_seconds'],final_evaluation=evaluation_seconds),
            peak_host_memory_bytes=peak_host_memory_bytes(),
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type=='cuda' else 0)
    finally:
        close(trainer,handler);del trainer
        if options.device>=0:torch.cuda.empty_cache()
    evaluator,handler=assemble(options,seed,method,output,final)
    try:
        started=time.perf_counter();evaluator._restore()
        if digest([m.state_dict() for m in evaluator.model_list])!=result['final_model_digest']:
            raise ValueError('Independent evaluator loaded a different model')
        independent=evaluator.test(evaluator.args.epoch-1,'test')
        if independent!=metrics:raise ValueError('Independent evaluation differs')
        result['timing']['independent_evaluation']=time.perf_counter()-started
        write_json(output/'independent_evaluation.json',dict(epoch=result['epoch'],test=independent,
            final_model_digest=result['final_model_digest']))
    finally:close(evaluator,handler)
    write_json(output/'metrics.json',result)
    return result


def is_complete(output,method,epochs,prefix,seed):
    record=read_json(output/'metrics.json');proof=read_json(output/'independent_evaluation.json')
    if not record or not proof:return False
    required=('candidate_purity','recipient_purity','candidate_digest','initial_digest','data_digest',
              'first_prefix','start_state','timing','final_model_digest','test')
    if not all(k in record for k in required):return False
    if (record.get('seed')!=seed or record.get('method')!=method or record.get('epoch')!=epochs
            or proof.get('epoch')!=epochs or record['test']!=proof.get('test')):return False
    checkpoint=load_checkpoint(output/'last_checkpoint.pth.tar')
    verify_checkpoint(checkpoint,method,epochs,prefix)
    return (record['final_model_digest']==proof.get('final_model_digest')==digest(checkpoint['state_dict'])
            and record.get('cumulative_replacements')==checkpoint['comparison']['cumulative']
            and record['candidate_digest']==digest(checkpoint['comparison']['choice']['candidates']))


def run_experiment(options):
    from lfba_comparison.report import write_report
    root=options.output;expected=frozen_protocol(options);manifest=runtime_sources(options.code_root)
    occupied=root.exists() and any(root.iterdir())
    if occupied and not options.resume:raise FileExistsError('Output occupied; explicitly use --resume')
    with batch_lock(root):
        if occupied:
            if read_json(root/'protocol.json')!=expected or read_json(root/'source_hashes.json')!=manifest:
                raise ValueError('Resume requires the identical frozen protocol and runtime source')
        else:
            write_json(root/'protocol.json',expected);write_json(root/'source_hashes.json',manifest)
        if not (root/'source_snapshot.zip').exists():
            with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in manifest:archive.write(options.code_root/name,name)
        write_json(root/'environment.json',dict(python=sys.version,executable=sys.executable,
            torch=str(torch.__version__),platform=platform.platform(),command=sys.argv,
            cuda=torch.version.cuda))
        for seed in options.seeds:
            directory=root/f'seed_{seed}';prefix_path=directory/'clean_first'/'last_checkpoint.pth.tar'
            if not prefix_path.exists():
                trainer,handler=assemble(options,seed,'capture',directory/'clean_first')
                try:trainer.train();export(directory/'clean_first',trainer)
                finally:close(trainer,handler)
                del trainer
                if options.device>=0:torch.cuda.empty_cache()
            prefix=load_checkpoint(prefix_path)
            if (prefix['epoch']!=1 or prefix['comparison']['method']!='capture'
                    or prefix['comparison']['choice'] is not None or prefix['comparison']['plans']
                    or prefix['attack_state']['current_plan']):raise ValueError('Invalid shared clean first round')
            for method in METHODS:
                if is_complete(directory/method,method,expected['epochs'],prefix,seed):continue
                print(f'RUN seed={seed} method={method}',flush=True)
                run_task(options,seed,method,directory,prefix)
        write_report(root)
        print('ALL_TASKS_COMPLETE',root,flush=True)
