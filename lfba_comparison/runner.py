"""Explicit orchestration. Importing this module performs no experiment."""
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.experiment import build_datasets
from main import build_models, trigger_dimensions, triggered_view, validate_data
from utils.checkpoint import load_checkpoint
from utils.config import parse_args
from utils.utils import set_seed
from utils.artifacts import runtime_checkpoint, save_experiment_data, cleanup_runtime, load_experiment_data
from lfba_comparison.records import verified_group, verified_pair
from lfba_comparison.common import batch_lock, digest, peak_host_memory_bytes, read_json, write_json
from lfba_comparison.policy import ANCHORS, METHODS, purity, schedule
from lfba_comparison.trainer import ComparisonTrainer


def frozen_protocol(options):
    protocol=dict(version=4,seeds=options.seeds,smoke=options.smoke,device=options.device,
        block_size=options.block_size,data_dir=str(options.data_dir),dataset=options.dataset,
        anchor=ANCHORS[options.dataset],anchor_source=options.anchor_source,
        methods=list(METHODS),epochs=5 if options.smoke else 40,batch_size=128,lr=.001,
        train_source_limit=500 if options.smoke else None,eval_limit=100 if options.smoke else None,
        optimizer='Adam',lr_decay=False,early_stop=False,poison_rate=options.poison_rate,
        select_rate=options.select_rate,
        split=f'stratified 90/10 on {options.dataset}; LFBA arms use '
              f'{"the anchor selected by auto_vote" if options.anchor_source=="auto_vote" else f"anchor {ANCHORS[options.dataset]}"}',
        final_model='last',baseline_similarity='legacy_l1',vote_similarity='cosine',
        primary='auto_vote - lfba_original',
        thresholds=dict(mean_asr_gain=.02,mean_acc_loss_limit=.01,positive_fraction=2/3),
        interpretation=('full strategy; shared automatically selected anchor'
                        if options.anchor_source=='auto_vote' else
                        'full strategy; fixed LFBA target versus free automatic target'))
    if options.dataset=='NUSWIDE':
        from lfba_comparison.nuswide import dataset_profile
        protocol.update(dataset_profile(options.data_dir,options.smoke))
    if getattr(options,'stability',False):
        protocol.update(stability=True,methods=['auto_vote','lfba_original'])
        if options.anchor_source=='auto_vote':
            protocol.update(anchor=None,split='stratified 90/10 without a retained preset anchor')
        else:
            protocol.update(version=5,split=f'stratified 90/10 retaining fixed LFBA anchor {ANCHORS[options.dataset]} for both methods')
    return protocol


def configuration(options,seed,method,output,anchor=None):
    anchor=ANCHORS[options.dataset] if anchor is None else anchor
    flags=['--dataset',options.dataset,'--data_dir',str(options.data_dir),'--results_dir',str(output),
        '--device',str(options.device),'--seed',str(seed),'--epoch',str(1 if method=='capture' else (5 if options.smoke else 40)),
        '--batch_size','128','--lr','.001','--early_stop','0','--anchor_idx',str(anchor),
        '--attack','LFBA','--poison_rate',str(options.poison_rate),'--select_rate',str(options.select_rate),
        '--pretrain_stage',
        '3' if method=='auto_vote' else '1','--print_steps','200','--target_label','0']
    if options.smoke: flags+=['--train_limit','500','--eval_limit','100']
    if options.dataset=='NUSWIDE':
        from lfba_comparison.nuswide import POISON_DIMENSIONS
        flags+=['--poison_dimensions',str(POISON_DIMENSIONS)]
    args=parse_args(flags)
    args.retain_anchor=not getattr(options,'stability',False) or options.anchor_source=='fixed'
    args.comparison_method=method
    args.block_size=options.block_size
    args.similarity='cosine' if method=='auto_vote' else 'legacy_l1'
    args.trigger_dimensions=[] if options.dataset=='CIFAR10' else None
    return args


def assemble(options,seed,method,output,checkpoint=None,anchor=None):
    args=configuration(options,seed,method,output,anchor)
    set_seed(seed);torch.set_num_threads(4)
    device=torch.device('cpu' if options.device<0 else f'cuda:{options.device}')
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; explicitly choose --device -1 for CPU')
        torch.cuda.set_device(device)
    train,val,test,state=build_datasets(args,checkpoint['data_state'] if checkpoint else None)
    if not args.retain_anchor and anchor is None:
        choice=checkpoint.get('comparison',{}).get('choice') if checkpoint else None
        args.anchor_idx=int(choice['anchor_id']) if choice else int(train.ids[0])
    validate_data(args,train,val,test)
    expected={'CIFAR10':[45000,5000,10000],'UCIHAR':[6616,736,2947]}.get(args.dataset)
    if args.dataset=='NUSWIDE':
        from lfba_comparison.nuswide import dataset_profile
        profile=dataset_profile(options.data_dir,options.smoke)
        expected=[profile[k] for k in ('train_size','validation_size','test_size')]
    if not options.smoke and expected and [len(train),len(val),len(test)]!=expected:
        raise ValueError(f'Formal split for {args.dataset} must be {expected}')
    dimensions=trigger_dimensions(args,checkpoint)
    views=dict(train=train,val=val,val_asr=triggered_view(val,args,dimensions),
        test=test,test_asr=triggered_view(test,args,dimensions))
    loaders={name:DataLoader(view,batch_size=128,shuffle=name=='train',num_workers=0,
        generator=torch.Generator().manual_seed(seed+i+1)) for i,(name,view) in enumerate(views.items())}
    models=[m.to(device) for m in build_models(args)]
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
    output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('lfba_comparison.'+str(output));logger.setLevel(logging.INFO);logger.propagate=False
    handler=logging.FileHandler(output/'experiment.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));logger.addHandler(handler)
    trainer=ComparisonTrainer(device,models,[],None,optimizers,nn.CrossEntropyLoss(),
        loaders['train'],loaders['test'],loaders['test_asr'],dimensions,logger,args,checkpoint,
        loaders['val'],loaders['val_asr'],state)
    if not (output/'config.json').exists():
        write_json(output/'config.json',dict(vars(args),data_state=state,trigger_dimensions=dimensions))
    return trainer,handler


def close(trainer,handler):
    trainer.logger.removeHandler(handler);handler.close()


def export(output,trainer):
    config=dict(vars(trainer.args),data_state=trainer.data_state,
                trigger_dimensions=trainer.trigger_dimensions)
    write_json(output/'config.json',config)
    save_experiment_data(output,trainer)


def verify_checkpoint(checkpoint,method,epochs,prefix,poison_rate=.1,select_rate=.5):
    state=checkpoint['comparison'];choice=state['choice']
    if checkpoint['epoch']!=epochs or state['method']!=method or choice is None:
        raise ValueError('Final checkpoint method or epoch differs')
    if checkpoint['data_state']!=prefix['data_state']:
        raise ValueError('Data split differs from shared prefix')
    if checkpoint['trigger_dimensions']!=prefix['trigger_dimensions']:
        raise ValueError('Trigger dimensions differ from shared prefix')
    if choice['anchor_id']!=checkpoint['config']['anchor_idx']:
        raise ValueError('Choice anchor differs from configured anchor')
    count=int(len(prefix['data_state']['train_ids'])*poison_rate)
    candidates=choice['candidates'];pool=set(candidates)
    if len(pool)!=count or len(candidates)!=count or not pool<=set(prefix['data_state']['train_ids']):
        raise ValueError('Candidate count or membership differs')
    clean=3 if method=='auto_vote' else 1
    expected=schedule(method,epochs,count,select_rate)
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


def run_task(options,seed,method,directory,prefix,anchor=None):
    output=directory/method;last=runtime_checkpoint(output)
    checkpoint=load_checkpoint(last) if last.exists() else prefix
    trainer,handler=assemble(options,seed,method,output,checkpoint,anchor)
    try:
        trainer.train() # only reachable through explicit CLI --run
        final=load_checkpoint(last);verify_checkpoint(final,method,trainer.args.epoch,prefix,
            trainer.args.poison_rate,trainer.args.select_rate)
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
            cumulative_replacements=trainer.cumulative,
            schedule=schedule(method,trainer.args.epoch,len(trainer.choice['candidates']),trainer.args.select_rate),
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
    evaluator,handler=assemble(options,seed,method,output,final,anchor)
    try:
        started=time.perf_counter();evaluator._restore()
        if digest([m.state_dict() for m in evaluator.model_list])!=result['final_model_digest']:
            raise ValueError('Independent evaluator loaded a different model')
        independent=evaluator.test(evaluator.args.epoch-1,'test')
        if independent!=metrics:raise ValueError('Independent evaluation differs')
        result['timing']['independent_evaluation']=time.perf_counter()-started
        result['independent_evaluation']=dict(epoch=result['epoch'],test=independent,
            final_model_digest=result['final_model_digest'])
    finally:close(evaluator,handler)
    trainer_epoch=5 if options.smoke else 40
    result['status']='complete'
    write_json(output/'metrics.json',result)
    saved,_,_=verified_group(output,method,trainer_epoch,seed,options.select_rate,strict=True)
    if saved!=result:raise ValueError('Final result read-back differs')
    return result


def is_complete(output,method,epochs,prefix,seed,poison_rate=.1,select_rate=.5):
    try:
        record,_,_=verified_group(output,method,epochs,seed,select_rate,strict=True)
        if prefix is not None:
            expected=digest(dict(model=prefix['state_dict'],optimizer=prefix['optimizer'],rng=prefix['rng_state']))
            return record['start_state']==expected
        return True
    except (ValueError,KeyError,TypeError,OSError):
        return False


def cleanup_completed_batch(root):
    root=Path(root)
    protocol=read_json(root/'protocol.json') or {}
    if protocol.get('version',0)<3:
        return
    if not (root/'results.md').is_file() or not (root/'validation_curves_lfba_vs_vote.png').is_file():
        return
    for seed in protocol['seeds']:
        for method in METHODS:
            verified_group(root/f'seed_{seed}'/method,method,protocol['epochs'],seed,
                           protocol.get('select_rate',.5),strict=True)
    if protocol.get('stability'):
        for seed in protocol['seeds']:
            verified_pair(root/f'seed_{seed}',protocol['epochs'],seed,protocol['poison_rate'],protocol['select_rate'],
                          anchor_source=protocol['anchor_source'],fixed_anchor=protocol.get('anchor'))
    for seed in protocol['seeds']:
        for method in (*METHODS,'clean_first'):
            cleanup_runtime(root/f'seed_{seed}'/method)


def _run_seed(options,root,seed,expected):
    directory=root/f'seed_{seed}'
    if all(is_complete(directory/m,m,expected['epochs'],None,seed,
                       options.poison_rate,options.select_rate) for m in METHODS):
        if getattr(options,'stability',False):
            verified_pair(directory,expected['epochs'],seed,options.poison_rate,options.select_rate,
                          anchor_source=options.anchor_source,fixed_anchor=expected.get('anchor'))
        return
    prefix_path=runtime_checkpoint(directory/'clean_first')
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
    methods=(('auto_vote','lfba_original') if options.anchor_source=='auto_vote' or getattr(options,'stability',False) else METHODS)
    for method in methods:
        anchor=None
        if method!='auto_vote' and options.anchor_source=='auto_vote':
            if not is_complete(directory/'auto_vote','auto_vote',expected['epochs'],prefix,seed,
                               options.poison_rate,options.select_rate):
                raise ValueError('Automatic arm must complete before the original LFBA arm')
            choice=load_experiment_data(directory/'auto_vote').get('choice')
            if not choice:raise ValueError('auto_vote anchor is unavailable for original LFBA')
            anchor=int(choice['anchor_id'])
        if is_complete(directory/method,method,expected['epochs'],prefix,seed,
                       options.poison_rate,options.select_rate):continue
        displayed_anchor=anchor if anchor is not None else ('automatic' if method=='auto_vote' else ANCHORS[options.dataset])
        print(f'RUN seed={seed} method={method} anchor={displayed_anchor}',flush=True)
        run_task(options,seed,method,directory,prefix,anchor)
    if getattr(options,'stability',False):
        verified_pair(directory,expected['epochs'],seed,options.poison_rate,options.select_rate,
                      anchor_source=options.anchor_source,fixed_anchor=expected.get('anchor'))


def run_experiment(options):
    from lfba_comparison.report import write_report
    root=options.output;expected=frozen_protocol(options)
    occupied=root.exists() and any(root.iterdir())
    if occupied and not options.resume:raise FileExistsError('Output occupied; explicitly use --resume')
    failures=[]
    with batch_lock(root):
        if occupied:
            if read_json(root/'protocol.json')!=expected:
                raise ValueError('Resume requires the identical frozen protocol')
        else:
            write_json(root/'protocol.json',expected)
        for seed in options.seeds:
            try:
                _run_seed(options,root,seed,expected)
            except Exception as error:
                if not getattr(options,'continue_on_error',False):raise
                failure=dict(seed=seed,error=f'{type(error).__name__}: {error}',stage='trial')
                failures.append(failure)
                directory=root/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
                with (directory/'experiment.log').open('a',encoding='utf-8') as stream:
                    stream.write(failure['error']+'\n')
                write_json(root/'summary.json',dict(all_complete=False,failures=failures))
                if options.device>=0:torch.cuda.empty_cache()
        if not failures:
            try:
                write_report(root)
            except Exception as error:
                if not getattr(options,'continue_on_error',False):raise
                failures.append(dict(seed=None,error=f'{type(error).__name__}: {error}',stage='report'))
        if failures:
            write_json(root/'summary.json',dict(all_complete=False,failures=failures))
        elif not getattr(options,'defer_cleanup',False):
            cleanup_completed_batch(root)
        print('ALL_TASKS_COMPLETE' if not failures else 'BATCH_INCOMPLETE',root,flush=True)
    return dict(all_complete=not failures,failures=failures)
