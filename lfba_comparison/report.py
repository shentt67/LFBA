"""Saved-results-only comparison reports. This module never trains or evaluates."""
import csv
import math
from pathlib import Path
import statistics

from lfba_comparison.common import digest, read_json, write_json
from lfba_comparison.policy import METHODS
from lfba_comparison.records import verified_group


def summarize(records, methods=None):
    methods=tuple(methods or METHODS)
    comparisons={name:[] for name in methods if name!='auto_vote'}
    for seed in sorted({r['seed'] for r in records}):
        rows={r['method']:r for r in records if r['seed']==seed}
        if set(rows)!=set(methods) or len([r for r in records if r['seed']==seed])!=len(methods):raise ValueError('Incomplete or duplicate methods')
        for key in ('initial_digest','data_digest','first_prefix','start_state'):
            if len({digest(r[key]) for r in rows.values()})!=1:raise ValueError(f'Paired {key} differs')
        if not rows['auto_vote']['start_state']:raise ValueError('Actual shared-prefix restore not recorded')
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
    methods=tuple(protocol.get('methods',METHODS))
    records=[];all_epochs=[];budget=[];candidate_total=None
    for seed in protocol['seeds']:
        for method in methods:
            folder=root/f'seed_{seed}'/method
            r,data,rows=verified_group(folder,method,protocol['epochs'],seed,protocol.get('select_rate',.5),strict=protocol.get('version',0)>=3)
            choice=data['choice']
            expected_total=r['cumulative_replacements']
            all_epochs.extend(dict(seed=seed,method=method,**row) for row in rows)
            candidate_total=len(choice['candidates'])
            budget.append(dict(seed=seed,method=method,expected=expected_total,actual=r['cumulative_replacements'],passed=True))
            records.append(r)
    summary=summarize(records,methods)
    applicable=(not protocol['smoke'] and len(protocol['seeds'])>=3
                and not protocol.get('seeds_are_descriptive_only',False))
    summary.update(smoke=protocol['smoke'],formal_verdict_applicable=applicable)
    summary['budget_checks']=budget
    summary['timing_resources']=dict(
        shared_first_round_seconds=sum(next(r['timing']['shared_first_round'] for r in records if r['seed']==seed) for seed in protocol['seeds']),
        groups=[dict(seed=r['seed'],method=r['method'],timing=r['timing'],
            peak_host_memory_bytes=r['peak_host_memory_bytes'],peak_cuda_memory_bytes=r['peak_cuda_memory_bytes']) for r in records])
    previous=read_json(root/'summary.json') or {}
    if 'artifact_metadata' in previous:summary['artifact_metadata']=previous['artifact_metadata']
    write_json(root/'summary.json',summary)
    with (root/'all_epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(key for row in all_epochs for key in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_epochs)
    lines=[('# LFBA原始策略与自动锚点＋多轮投票对照' if 'lfba_original' in methods else
            '# 自动锚点与多轮近邻策略对照'),'',
        '**短训练执行路径检查，不能判断攻击性能。**' if protocol['smoke'] else f'**固定第{protocol["epochs"]}轮完整测试集；{len(protocol["seeds"])}种子，LFBA锚点来源：{protocol.get("anchor_source", "fixed")}。**','',
        f'毒化比例 p={protocol.get("poison_rate",.1)}，切换比例 s={protocol.get("select_rate",.5)}，候选 {candidate_total} 个。','',
        '| seed | 方法 | 锚点 | 目标类 | ACC | ASR | 触发准确率 | 候选纯度 | 去锚点纯度（分母） | 替换纯度 | 累计替换 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in records:
        m=r['test'];p=r['candidate_purity'];without='—' if p.get('without_anchor') is None else f"{p['without_anchor']:.4%} ({p['without_anchor_denominator']})"
        lines.append(f"|{r['seed']}|{r['method']}|{r['anchor']}|{r['target_class']}|{m['acc']:.2%}|{m['asr']:.2%}|{m['poison_acc']:.2%}|{p['full']:.4%}|{without}|{r['recipient_purity']:.4%}|{r['cumulative_replacements']}|")
    for method in methods:
        if method=='auto_vote':continue
        label=f'投票减 {method}'
        part=summary[method]
        lines+=['',f'## {label}','',
            f"平均ASR差 {part['mean_delta_asr']*100:+.3f}个百分点，平均ACC差 {part['mean_delta_acc']*100:+.3f}个百分点；满足逐种子正向条件 {part['positive_count']}/{part['total']}。",
            ('达到预设初步收益门槛。' if part['meets_threshold'] else '未达到预设初步收益门槛。') if applicable else '本次不适用正式性能判据。','',
            '| seed | ASR差(pp) | ACC差(pp) | 目标类别相同 |','|---|---|---|---|']
        for p in part['pairs']:lines.append(f"|{p['seed']}|{p['delta_asr']*100:+.3f}|{p['delta_acc']*100:+.3f}|{p['same_target']}|")
    lines+=['','## 解释边界','',
        ('两种方法共用自动投票选择的锚点和目标；原始LFBA仍使用第一轮梯度构建候选集。'
         if protocol.get('anchor_source')=='auto_vote' else
         (f'原始LFBA使用预设锚点{protocol.get("anchor", 23470)}，自动投票自由选择锚点和目标。'
          if 'lfba_original' in methods else '各自动方法使用各自选出的锚点与目标。'))
        + '本实验比较完整策略，不能单独归因于投票。',
        ('历史批次包含预算匹配组；累计预算匹配不消除启动时序差异。' if 'lfba_matched' in methods else
         '各方法的启动轮次和预算见保存的实验配置及逐轮数据。'),
        '候选纯度列分母为完整候选数量；去锚点列明确分母。高纯度与高ASR上限保留在汇总中，不事后换种子、目标或终点。',
        '共享首轮只计一次；阶段epoch时间不含全部检查点I/O，elapsed_branch含该分支记录到最后快照的耗时。主机内存为进程历史峰值，GPU峰值也可能跨阶段累计。',
        f'本批次{len(protocol["seeds"])}个种子与有限锚点仅提供初步证据，不作统计显著性或普适性结论。','']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_methods=methods
    plot_labels={'lfba_original':'Original LFBA','auto_vote':'Improved: Auto-anchor + Multi-round Vote'}
    plot_colors={'lfba_original':'#3568a8','auto_vote':'#d65f36'}
    fig,axes=plt.subplots(len(protocol['seeds']),2,figsize=(12,3.5*len(protocol['seeds'])),squeeze=False)
    for i,seed in enumerate(protocol['seeds']):
        for method in plot_methods:
            _,saved,_=verified_group(root/f'seed_{seed}'/method,method,protocol['epochs'],seed,protocol.get('select_rate',.5),strict=protocol.get('version',0)>=3)
            history=saved['history']
            for j,key in enumerate(('acc','asr')):
                axis=axes[i,j]
                axis.plot([r['epoch'] for r in history],
                    [float('nan') if r['validation'].get(key) is None else r['validation'][key] for r in history],
                    label=plot_labels.get(method,method),color=plot_colors.get(method),linewidth=2)
                axis.set_title(f'Seed {seed}: Validation {key.upper()}')
                axis.set_xlabel('Epoch');axis.set_ylabel('Score');axis.set_ylim(0,1)
                axis.grid(True,alpha=.25);axis.legend()
    fig.tight_layout();fig.savefig(root/'validation_curves_lfba_vs_vote.png',dpi=200,bbox_inches='tight');plt.close(fig)
    lines+=['![Validation curves](validation_curves_lfba_vs_vote.png)','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')
