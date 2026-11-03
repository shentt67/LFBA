"""Saved-results-only comparison reports. This module never trains or evaluates."""
import csv
import math
from pathlib import Path
import statistics

from lfba_comparison.common import digest, read_json, write_json
from lfba_comparison.policy import METHODS, schedule


def summarize(records):
    comparisons={name:[] for name in ('lfba_original','lfba_matched')}
    for seed in sorted({r['seed'] for r in records}):
        rows={r['method']:r for r in records if r['seed']==seed}
        if len(rows)!=3 or len([r for r in records if r['seed']==seed])!=3:raise ValueError('Incomplete or duplicate methods')
        for key in ('initial_digest','data_digest','first_prefix','start_state'):
            if len({digest(r[key]) for r in rows.values()})!=1:raise ValueError(f'Paired {key} differs')
        if not rows['auto_vote']['start_state']:raise ValueError('Actual shared-prefix restore not recorded')
        if rows['lfba_original']['candidate_digest']!=rows['lfba_matched']['candidate_digest']:
            raise ValueError('Original and matched LFBA candidate sets differ')
        for baseline in comparisons:
            base,new=rows[baseline],rows['auto_vote']
            comparisons[baseline].append(dict(seed=seed,baseline=baseline,
                baseline_anchor=base['anchor'],new_anchor=new['anchor'],
                baseline_target=base['target_class'],new_target=new['target_class'],
                same_target=base['target_class']==new['target_class'],
                delta_asr=new['test']['asr']-base['test']['asr'],delta_acc=new['test']['acc']-base['test']['acc'],
                baseline_asr_saturated=base['test']['asr']>=.98))
    result={}
    for baseline,pairs in comparisons.items():
        count=len(pairs)
        mean_asr=statistics.mean(p['delta_asr'] for p in pairs)
        mean_acc=statistics.mean(p['delta_acc'] for p in pairs)
        positives=sum(p['delta_asr']>0 and p['delta_acc']>=-.01 for p in pairs)
        result[baseline]=dict(pairs=pairs,mean_delta_asr=mean_asr,mean_delta_acc=mean_acc,
            positive_count=positives,total=count,
            meets_threshold=mean_asr>=.02 and mean_acc>=-.01 and positives>=math.ceil(2*count/3))
    return result


def write_report(root):
    root=Path(root);protocol=read_json(root/'protocol.json')
    if not protocol:raise ValueError('No saved protocol; report-only never runs missing experiments')
    records=[];all_epochs=[];budget=[]
    for seed in protocol['seeds']:
        for method in METHODS:
            folder=root/f'seed_{seed}'/method
            r=read_json(folder/'metrics.json');proof=read_json(folder/'independent_evaluation.json')
            choice=read_json(folder/'choice.json')
            if (not r or not proof or not choice or r.get('method')!=method or r.get('seed')!=seed
                    or r.get('epoch')!=protocol['epochs'] or proof.get('epoch')!=protocol['epochs']
                    or proof.get('test')!=r.get('test') or proof.get('final_model_digest')!=r.get('final_model_digest')):
                raise ValueError(f'Incomplete or unverified saved result: {seed}/{method}')
            if r['candidate_digest']!=digest(choice['candidates']):raise ValueError('Saved candidate digest differs')
            with (folder/'epoch_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            expected=schedule(method,protocol['epochs'],len(choice['candidates']))
            if (len(rows)!=len(expected) or [int(row['epoch']) for row in rows]!=list(range(1,len(expected)+1))
                    or [int(row['recipient_count']) for row in rows]!=expected
                    or [int(row['cumulative_replacements']) for row in rows]!=[sum(expected[:i+1]) for i in range(len(expected))]
                    or r['cumulative_replacements']!=sum(expected)):raise ValueError('Budget audit failed')
            clean=3 if method=='auto_vote' else 1
            if any(int(row['candidate_count'])!=0 for row in rows[:clean]) or any(
                row['candidate_digest']!=r['candidate_digest'] or int(row['candidate_count'])!=len(choice['candidates'])
                for row in rows[clean:]):raise ValueError('Frozen selection audit failed')
            all_epochs.extend(dict(seed=seed,method=method,**row) for row in rows)
            budget.append(dict(seed=seed,method=method,expected=sum(expected),actual=r['cumulative_replacements'],passed=True))
            records.append(r)
    summary=summarize(records)
    applicable=not protocol['smoke'] and protocol['seeds']==[106,107,108]
    summary.update(smoke=protocol['smoke'],formal_verdict_applicable=applicable)
    write_json(root/'summary.json',summary)
    for filename,rows in (('all_epoch_metrics.csv',all_epochs),('budget_check.csv',budget)):
        with (root/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/'timing_resources.json',dict(
        shared_first_round_seconds=sum(next(r['timing']['shared_first_round'] for r in records if r['seed']==seed) for seed in protocol['seeds']),
        groups=[dict(seed=r['seed'],method=r['method'],timing=r['timing'],
            peak_host_memory_bytes=r['peak_host_memory_bytes'],peak_cuda_memory_bytes=r['peak_cuda_memory_bytes']) for r in records]))
    lines=['# LFBA原始策略与自动锚点＋多轮投票对照','',
        '**短训练执行路径检查，不能判断攻击性能。**' if protocol['smoke'] else '**固定第20轮完整测试集；三种子、单个LFBA预设锚点。**','',
        '| seed | 方法 | 锚点 | 目标类 | ACC | ASR | 触发准确率 | 候选纯度 | 去锚点纯度（分母） | 替换纯度 | 累计替换 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in records:
        m=r['test'];p=r['candidate_purity'];without='—' if p['without_anchor'] is None else f"{p['without_anchor']:.4%} ({p['without_anchor_denominator']})"
        lines.append(f"|{r['seed']}|{r['method']}|{r['anchor']}|{r['target_class']}|{m['acc']:.2%}|{m['asr']:.2%}|{m['poison_acc']:.2%}|{p['full']:.4%}|{without}|{r['recipient_purity']:.4%}|{r['cumulative_replacements']}|")
    for method,label in (('lfba_original','主要比较：投票减原始策略'),('lfba_matched','补充比较：投票减预算匹配')):
        part=summary[method]
        lines+=['',f'## {label}','',
            f"平均ASR差 {part['mean_delta_asr']*100:+.3f}个百分点，平均ACC差 {part['mean_delta_acc']*100:+.3f}个百分点；满足逐种子正向条件 {part['positive_count']}/{part['total']}。",
            ('达到预设初步收益门槛。' if part['meets_threshold'] else '未达到预设初步收益门槛。') if applicable else '本次不适用正式性能判据。','',
            '| seed | ASR差(pp) | ACC差(pp) | 目标类别相同 |','|---|---|---|---|']
        for p in part['pairs']:lines.append(f"|{p['seed']}|{p['delta_asr']*100:+.3f}|{p['delta_acc']*100:+.3f}|{p['same_target']}|")
    lines+=['','## 解释边界','',
        'LFBA原始策略在修复后的统一训练框架中运行，不恢复旧实现错误。自动投票自由选择目标，LFBA固定目标来自锚点23470，因此是完整策略比较，不是固定语义目标比较，也不能单独归因于投票。',
        '预算匹配组改变每轮强度；与投票虽累计次数相同，启动时间仍不同，不能声称消除了时序差异。',
        '候选纯度列分母为完整候选数量；去锚点列明确分母。高纯度与高ASR上限保留在汇总中，不事后换种子、目标或终点。',
        '共享首轮只计一次；阶段epoch时间不含全部检查点I/O，elapsed_branch含该分支记录到最后快照的耗时。主机内存为进程历史峰值，GPU峰值也可能跨阶段累计。',
        '三种子、一个LFBA锚点仅提供初步证据，不作统计显著性或普适性结论。','']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_methods=('lfba_original','auto_vote')
    plot_labels={'lfba_original':'Original LFBA','auto_vote':'Improved: Auto-anchor + Multi-round Vote'}
    plot_colors={'lfba_original':'#3568a8','auto_vote':'#d65f36'}
    fig,axes=plt.subplots(len(protocol['seeds']),2,figsize=(12,3.5*len(protocol['seeds'])),squeeze=False)
    for i,seed in enumerate(protocol['seeds']):
        for method in plot_methods:
            history=read_json(root/f'seed_{seed}'/method/'history.json')
            for j,key in enumerate(('acc','asr')):
                axis=axes[i,j]
                axis.plot([r['epoch'] for r in history],
                    [float('nan') if r['validation'].get(key) is None else r['validation'][key] for r in history],
                    label=plot_labels[method],color=plot_colors[method],linewidth=2)
                axis.set_title(f'Seed {seed}: Validation {key.upper()}')
                axis.set_xlabel('Epoch');axis.set_ylabel('Score');axis.set_ylim(0,1)
                axis.grid(True,alpha=.25);axis.legend()
    fig.tight_layout();fig.savefig(root/'validation_curves_lfba_vs_vote.png',dpi=200,bbox_inches='tight');plt.close(fig)
    lines+=['![Validation curves](validation_curves_lfba_vs_vote.png)','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')
