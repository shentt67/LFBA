"""Run four fixed-horizon attacks; evaluate final models, never select by ASR."""
import argparse
import csv
import hashlib
import json
import logging
import platform
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from main import build_models, triggered_view, validate_data
from dataset.experiment import build_datasets
from experiments.attack_budget import GROUPS, BudgetTrainer, tree_digest
from experiments.json_io import atomic_write_json, read_json_if_valid
from utils.checkpoint import load_checkpoint
from utils.config import parse_args
from utils.utils import set_seed


def write_json(path,value):
    atomic_write_json(path,value)


def saved_run_config(output):
    saved=read_json_if_valid(output/'config.json')
    if not isinstance(saved,dict):
        saved=load_checkpoint(output/'last_checkpoint.pth.tar').get('config')
    if not isinstance(saved,dict):
        raise ValueError('No valid saved configuration or checkpoint configuration')
    return saved


def completed_run_result(output,group,epoch):
    result=read_json_if_valid(output/'metrics.json')
    if not isinstance(result,dict) or result.get('group')!=group or result.get('epoch')!=epoch:
        return None
    required=('initial_digest','data_digest','prefix_digests','schedule','cumulative_replacements',
              'candidate_purity','weighted_recipient_purity','elapsed_seconds')
    metrics=result.get('test')
    if not all(k in result for k in required) or not isinstance(metrics,dict):
        return None
    if not all(k in metrics for k in ('acc','asr','poison_acc','counts')):
        return None
    return result


def make_args(root,group,smoke,seed=100,anchor_idx=23470):
    total=5 if smoke else 20
    clean=1 if group.startswith('early') else 3
    flags=['--dataset','CIFAR10','--data_dir',str(ROOT/'data'),'--device','0','--attack','LFBA',
           '--epoch',str(total),'--pretrain_stage',str(clean),'--batch_size','128','--seed',str(seed),
           '--early_stop','0','--lr','.001','--anchor_idx',str(anchor_idx),'--poison_rate','.1',
           '--select_rate','.5','--similarity','legacy_l1','--print_steps','200',
           '--results_dir',str(root/group)]
    if smoke: flags+=['--train_limit','200','--eval_limit','100']
    args=parse_args(flags)
    args.experiment_group=group
    args.experiment_total_epochs=total
    args.observation_window=3
    return args


def run_group(root,group,smoke,resume,seed=100,anchor_idx=23470,quiet=False):
    output=root/group
    args=make_args(root,group,smoke,seed=seed,anchor_idx=anchor_idx)
    checkpoint=None
    if (output/'config.json').exists() or (output/'last_checkpoint.pth.tar').exists():
        if not resume: raise FileExistsError('Use --resume for existing experiment outputs')
        saved=saved_run_config(output)
        current=vars(args)
        ignore={'target_label','trigger_dimensions'}
        if any(saved.get(k)!=v for k,v in current.items() if k not in ignore):
            raise ValueError('Saved experiment configuration differs')
        completed=completed_run_result(output,group,args.epoch)
        if completed is not None:
            print('COMPLETED_GROUP_VERIFIED',group,flush=True)
            return completed
        if (output/'last_checkpoint.pth.tar').exists():
            checkpoint=load_checkpoint(output/'last_checkpoint.pth.tar')
    output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('budget.'+group)
    logger.setLevel(logging.INFO);logger.propagate=False
    handlers=[logging.FileHandler(output/'experiment.log',encoding='utf-8')]
    if not quiet: handlers.append(logging.StreamHandler())
    for h in handlers: logger.addHandler(h)
    started=time.perf_counter()
    try:
        torch.set_num_threads(4)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            torch.cuda.set_device(device);torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        set_seed(args.seed)
        train,val,test,data_state=build_datasets(args,checkpoint['data_state'] if checkpoint else None)
        # This label is evaluation metadata only. Candidate selection never receives it.
        args.target_label=int(train.targets[args.anchor_idx])
        args.trigger_dimensions=[]
        validate_data(args,train,val,test)
        datasets=dict(train=train,val=val,val_asr=triggered_view(val,args,[]),
                      test=test,test_asr=triggered_view(test,args,[]))
        loaders={name:DataLoader(data,batch_size=128,shuffle=name=='train',num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed+i+1))
                 for i,(name,data) in enumerate(datasets.items())}
        models=[model.to(device) for model in build_models(args)]
        optimizers=[torch.optim.Adam(model.parameters(),lr=args.lr) for model in models]
        trainer=BudgetTrainer(device,models,[],None,optimizers,nn.CrossEntropyLoss(),loaders['train'],
            loaders['test'],loaders['test_asr'],[],logger,args,checkpoint,
            loaders['val'],loaders['val_asr'],data_state)
        write_json(output/'config.json',vars(args))
        write_json(output/'data_state.json',data_state)
        write_json(output/'environment.json',dict(python=sys.version,executable=sys.executable,
            torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform(),
            device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None))
        print('GROUP_START',group,'seed='+str(seed),'anchor='+str(anchor_idx),flush=True)
        trainer.train()
        final=load_checkpoint(output/'last_checkpoint.pth.tar')
        assert trainer.completed_epoch==args.epoch==final['epoch']
        assert tree_digest([m.state_dict() for m in models])==tree_digest(final['state_dict'])
        assert trainer.cumulative_replacements==sum(trainer.schedule)
        assert [r['recipient_count'] for r in trainer.audit_rows]==trainer.schedule
        metrics=trainer.test(args.epoch-1,split='test')
        write_json(output/'history.json',trainer.history)
        write_json(output/'budget_audit.json',trainer.audit_rows)
        write_json(output/'prefix_digests.json',trainer.prefix_digests)
        combined=[dict(row,**{'validation_'+k:v for k,v in history['validation'].items()
                              if k in ('acc','asr','poison_acc','target_acc','loss')})
                  for row,history in zip(trainer.audit_rows,trainer.history)]
        with (output/'epoch_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
        total=trainer.cumulative_replacements
        weighted=sum((row['recipient_purity'] or 0)*row['recipient_count'] for row in trainer.audit_rows)/total
        previous_elapsed=checkpoint.get('budget_elapsed_seconds',0.) if checkpoint else 0.
        result=dict(group=group,epoch=args.epoch,test=metrics,
            candidate_purity=trainer.audit_rows[-1]['candidate_purity'],
            weighted_recipient_purity=weighted,cumulative_replacements=total,
            initial_digest=trainer.initial_digest,data_digest=tree_digest(data_state),
            final_model_digest=tree_digest(final['state_dict']),
            elapsed_seconds=previous_elapsed+time.perf_counter()-started,
            elapsed_note='wall time of this invocation; partial prior invocation time excluded on resume',
            prefix_digests=trainer.prefix_digests,schedule=trainer.schedule,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
        write_json(output/'metrics.json',result)
        print('GROUP_DONE',group if quiet else json.dumps(result),flush=True)
        return result
    finally:
        for h in handlers: logger.removeHandler(h);h.close()


def report(root,results,smoke,seed=100,anchor_idx=23470):
    assert list(results)==list(GROUPS)
    assert len({r['initial_digest'] for r in results.values()})==1,'Initial models differ'
    assert len({r['data_digest'] for r in results.values()})==1,'Data splits differ'
    assert results['delayed_single']['prefix_digests']==results['multi_mean']['prefix_digests'],'Clean prefixes differ'
    write_json(root/'summary.json',results)
    delta_asr=results['multi_mean']['test']['asr']-results['delayed_single']['test']['asr']
    delta_acc=results['multi_mean']['test']['acc']-results['delayed_single']['test']['acc']
    if delta_asr>=.02 and delta_acc>=-.01:
        conclusion='满足预设初步正向标准：ASR提高至少2个百分点且ACC下降不超过1个百分点。'
    elif delta_asr>0 and delta_acc<-.01:
        conclusion='ASR改善伴随超过1个百分点的ACC下降，应解释为攻击效果与正常任务性能的权衡。'
    elif delta_asr<=-.02:
        conclusion='多轮相对延迟单轮的ASR下降至少2个百分点，本配置不支持实际攻击收益。'
    else:
        conclusion='未观察到达到预设幅度的ASR改善；纯度改善不能等同于实际攻击更强。'
    checks=dict(initialization_identical=True,data_splits_identical=True,delayed_clean_prefix_identical=True,
                all_final_epochs=all(r['epoch']==(5 if smoke else 20) for r in results.values()),
                delta_asr=delta_asr,delta_acc=delta_acc,conclusion=conclusion,smoke=smoke)
    write_json(root/'comparison_checks.json',checks)
    lines=['# 单种子实际投毒与预算对照','',
        '**短训练执行路径检查，不解释攻击性能。**' if smoke else '**固定第20轮完整测试集结果；单种子初步证据。**','',
        f'seed={seed}，anchor={anchor_idx}，legacy_l1，Adam lr=.001，batch=128，不早停。',
        '正式训练/验证/测试=45,000/5,000/10,000；短训练检查使用200条候选和100条测试数据。',
        '每组从相同初始化开始，独立在线采集所需干净反馈；不读取已有五轮轨迹。候选冻结、标签仅用于评估。',
        '沿用父训练器保存的best checkpoint不参与最终评测；最终始终使用last checkpoint。','',
        '| 组别 | ACC | ASR | 触发样本准确率 | 候选纯度 | 实际替换样本加权纯度 | 累计替换次数 | 耗时秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        m=r['test']
        lines.append(f"| {name} | {m['acc']:.2%} | {m['asr']:.2%} | {m['poison_acc']:.2%} | {r['candidate_purity']:.2%} | {r['weighted_recipient_purity']:.2%} | {r['cumulative_replacements']} | {r['elapsed_seconds']:.1f} |")
    lines+=['','## 主要配对：multi_mean - delayed_single','',
        f'ASR差异：{delta_asr*100:+.3f}个百分点；ACC差异：{delta_acc*100:+.3f}个百分点。',
        '短训练结果不用于性能判断。' if smoke else conclusion,'',
        '初始化、数据划分以及延迟两组前三轮模型/优化器/随机状态摘要均一致。',
        '早期原始组投毒19轮，延迟及多轮组投毒17轮；补充早期组累计次数匹配，但改变每轮投毒强度，不能视为完全相同的时序条件。','',
        '## 逐轮预算','',
        '| 轮次 | early_single | delayed_single | multi_mean | early_matched |','|---|---:|---:|---:|---:|']
    for ep in range(len(results['early_single']['schedule'])):
        lines.append('| '+str(ep+1)+' | '+' | '.join(str(results[g]['schedule'][ep]) for g in GROUPS)+' |')
    lines+=['','## 复现与边界','',
        '- 入口：scripts/run_attack_budget.py；短训练加--smoke；已存在目录需显式--resume。',
        '- 各组epoch_metrics.csv包含逐轮纯度、验证指标和累计预算；last checkpoint包含干净观察历史、候选集、供体RNG及审计状态。',
        '- ASR只统计真实标签不为目标类的样本；正式分母为9,000；ACC分母10,000。',
        '- 验证ASR只记录，不参与启动、选样、训练停止或最终模型选择。',
        '- 单锚点、单种子、单数据集，不进行统计显著性或普适性推断；不更换锚点追求正结果。','']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for name in GROUPS:
            history=json.loads((root/name/'history.json').read_text(encoding='utf-8'))
            for ax,key in zip(axes,('acc','asr')):
                ax.plot([x['epoch'] for x in history],[100*x['validation'][key] for x in history],label=name)
                ax.set(xlabel='Epoch',ylabel=key.upper()+' (%)',title='Validation '+key.upper(),ylim=(0,100))
                ax.grid(alpha=.25)
        axes[1].legend(fontsize=8)
        fig.savefig(root/'validation_curves.png',dpi=160);plt.close(fig)
        lines+=['![Validation curves](validation_curves.png)','']
    except ImportError:
        lines+=['绘图库不可用，完整曲线数值保存在各组epoch_metrics.csv。','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('EXPERIMENT_COMPLETE',root,conclusion,flush=True)


def run(root,smoke,resume,seed=100,anchor_idx=23470):
    protocol=dict(seed=seed,anchor=anchor_idx,groups=list(GROUPS),smoke=smoke,epochs=5 if smoke else 20,
                  selection='online only',final_selection='last',asr_gain_threshold=.02,acc_loss_limit=.01)
    files=[ROOT/'main.py',ROOT/'scripts'/'run_attack_budget.py']
    files += [p for folder in ('attack','dataset','models','utils','experiments') for p in (ROOT/folder).glob('*.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if (root/'protocol.json').exists():
        if not resume: raise FileExistsError('Output exists; choose a new directory or --resume')
        if json.loads((root/'protocol.json').read_text())!=protocol: raise ValueError('Protocol differs')
        if json.loads((root/'source_hashes.json').read_text())!=hashes: raise ValueError('Source differs from experiment snapshot')
    else:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'protocol.json',protocol);write_json(root/'source_hashes.json',hashes)
    results={}
    for group in GROUPS:
        results[group]=run_group(root,group,smoke,resume,seed=seed,anchor_idx=anchor_idx)
        write_json(root/'progress.json',results)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    report(root,results,smoke,seed=seed,anchor_idx=anchor_idx)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--anchor-idx',type=int,default=23470)
    flags=parser.parse_args()
    output=flags.output or ROOT/'results'/('attack_budget_smoke' if flags.smoke else 'attack_budget_seed100')
    run(output,flags.smoke,flags.resume,seed=flags.seed,anchor_idx=flags.anchor_idx)
