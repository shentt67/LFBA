"""Model-free stability reports, including incomplete and failed experiments."""
import csv
import math
from pathlib import Path
import statistics

from lfba_comparison.common import read_json,write_json
from lfba_comparison.records import verified_group

METHODS=('auto_vote','lfba_original')


def statistics_record(values):
    values=[float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite stability metric')
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values)>1 else None,
                min=min(values) if values else None,max=max(values) if values else None)


LABELS={'auto_vote':'Auto-anchor + multi-round vote','lfba_original':'Original LFBA'}
COLORS={'auto_vote':'#d65f36','lfba_original':'#3568a8'}
STYLES={'auto_vote':'-','lfba_original':'--'}


def _protocol_label(protocol):
    return ('完整方法稳定性对照：自动锚点＋多轮投票 vs 固定锚点原始LFBA'
            if protocol.get('anchor_source')=='fixed' else '同锚点稳定性对照（历史协议）')


def _interpretation(protocol):
    anchor=(f"原始LFBA固定锚点{protocol['fixed_lfba_anchor']}，自动方法自由选择锚点。"
            if protocol.get('anchor_source')=='fixed' else '双方共用自动方法选出的锚点。')
    return anchor+'ASR分别针对各自目标类别计算，分母为排除各自目标类后的测试样本数。保留各自投毒时序与不同累计预算，结果比较完整策略，不能单独归因于投票。'


def _budget(case,protocol):
    n=450 if protocol.get('smoke') else 45000
    p=math.floor(n*case['poison_rate']);q=math.floor(p*case['select_rate'])
    return p,q


def _budget_text(case,protocol):
    p,q=_budget(case,protocol);epochs=protocol['epochs']
    return f"候选P={p}，每轮替换Q={q}；原始LFBA累计{(epochs-1)*q}（{epochs-1}Q），自动方法累计{(epochs-3)*q}（{epochs-3}Q）。"


def _plot_context(case,protocol):
    p,q=_budget(case,protocol)
    return f"{protocol['dataset']}: p={case['poison_rate']}, s={case['select_rate']}, P={p}, Q={q}; {protocol['epochs']} epochs; validation curves"


def _plot_note(protocol):
    mode=(f"LFBA fixed anchor {protocol['fixed_lfba_anchor']}; auto selects freely" if protocol.get('anchor_source')=='fixed' else 'Legacy same automatic anchor')
    return f"{mode}. ASR uses each method's own target. Budgets: LFBA {protocol['epochs']-1}Q, auto {protocol['epochs']-3}Q."


def _describe_case(case,protocol):
    lines=[f"完成配对{case['completed_pairs']}/{case['expected_pairs']}；失败{case['failed_pairs']}，缺失{case['missing_pairs']}。"+
           ('阶段性结果，尚不能视为完整稳定性结论。' if case['completed_pairs']<case['expected_pairs'] else '')]
    if protocol.get('smoke'):lines.append('短训练仅验证执行流程，不用于判断正式攻击性能。')
    for method in METHODS:
        stats=case['methods'][method];a,b=stats['acc'],stats['asr']
        if not a['n']:
            lines.append(f"{method}：暂无有效结果。")
            continue
        lines.append(f"{method}（n={a['n']}）：ACC {_display(a['mean'])}% ± {_display(a['std'])}，最差{_display(a['min'])}%；ASR {_display(b['mean'])}% ± {_display(b['std'])}，最差{_display(b['min'])}%。")
    delta=case['paired']['delta_asr'];acc=case['paired']['delta_acc']
    if delta['n']:
        direction='更高' if delta['mean']>0 else ('更低' if delta['mean']<0 else '相同')
        lines.append(f"配对差值（自动−原始）：ASR均值{_display(delta['mean'])}个百分点，标准差{_display(delta['std'])}，范围[{_display(delta['min'])}, {_display(delta['max'])}]；ACC均值{_display(acc['mean'])}个百分点，标准差{_display(acc['std'])}，范围[{_display(acc['min'])}, {_display(acc['max'])}]。自动方法平均ASR{direction}，ASR提升种子比例{_display(case['paired']['positive_fraction'])}%。")
    auto,base=case['methods']['auto_vote']['asr'],case['methods']['lfba_original']['asr']
    if auto['std'] is not None and base['std'] is not None:
        comparison='较小' if auto['std']<base['std'] else ('较大' if auto['std']>base['std'] else '相同')
        lines.append(f"自动方法ASR的种子间波动{comparison}；稳定性应结合平均性能及最差表现判断。双方统计样本不齐时，波动比较仅作描述。")
    return lines


def _describe_trends(cases,protocol):
    lines=[];baseline=protocol['baseline']
    reference=next(c for c in cases if all(c[k]==v for k,v in baseline.items()))
    for parameter,fixed,label in (('poison_rate','select_rate','p'),('select_rate','poison_rate','s')):
        selected=sorted((c for c in cases if c[fixed]==baseline[fixed]),key=lambda c:c[parameter])
        lines.append(f"固定{'s' if label=='p' else 'p'}={baseline[fixed]}，扫描{label}：")
        for method in METHODS:
            for metric in ('acc','asr'):
                ref=reference['methods'][method][metric]['mean']
                parts=[]
                for c in selected:
                    value=c['methods'][method][metric]['mean']
                    change='' if value is None or ref is None else f"（较基准{100*(value-ref):+.3f}个百分点）"
                    parts.append(f"{label}={c[parameter]}: {_display(value)}%{change}")
                lines.append(f"{method} {metric.upper()}："+'；'.join(parts)+'。')
        deltas=[c['paired']['delta_asr']['mean'] for c in selected if c['paired']['delta_asr']['mean'] is not None]
        if deltas:
            lines.append('自动方法的平均ASR优势随配置发生反转。' if min(deltas)<0<max(deltas) else
                         ('已有配对配置中自动方法平均ASR均更高。' if min(deltas)>0 else '已有配对配置未显示自动方法始终具有平均ASR优势。'))
        if any(c['completed_pairs']<c['expected_pairs'] for c in selected):lines.append('本扫描尚未全部完成，以上趋势为阶段性描述。')
    return lines


def _pair_records(directory,epochs,seed,p,s,anchor_source='auto_vote',fixed_anchor=None):
    from lfba_comparison.records import verified_pair
    return verified_pair(directory/f'seed_{seed}',epochs,seed,p,s,anchor_source=anchor_source,fixed_anchor=fixed_anchor)


def _failures(root):
    saved=read_json(root/'failures.json') or (read_json(root/'summary.json') or {}).get('failures',[])
    if isinstance(saved,dict):saved=saved.get('failures',[])
    return {item['seed']:item for item in saved if isinstance(item,dict) and 'seed' in item}


def _display(value):
    return '—' if value is None else f'{value*100:.3f}'


def _case_protocol_error(folder,case,parent):
    saved=read_json(folder/'protocol.json')
    if saved is None:
        return 'Missing case protocol' if any(folder.glob('seed_*/*/metrics.json')) else None
    from types import SimpleNamespace
    from lfba_comparison.runner import frozen_protocol
    options=SimpleNamespace(stability=True,dataset=parent['dataset'],seeds=parent['seeds'],
        smoke=parent['smoke'],device=parent['device'],block_size=parent['block_size'],
        data_dir=parent['data_dir'],anchor_source=parent.get('anchor_source','auto_vote'),
        poison_rate=case['poison_rate'],select_rate=case['select_rate'])
    expected=frozen_protocol(options)
    return 'Case protocol differs from the frozen matrix' if saved!=expected else None


def _case_report(folder,case,seeds,groups,pairs,statuses,protocol,entry):
    folder.mkdir(parents=True,exist_ok=True)
    lines=['# '+_protocol_label(protocol),'',f"p={case['poison_rate']}, s={case['select_rate']}；固定第{protocol['epochs']}轮测试集结果。",'',
           '| seed | 状态 | 方法 | 锚点 | 目标类 | ACC(%) | ASR(%) | 累计替换 |',
           '|---|---|---|---|---|---|---|---|']
    for seed in seeds:
        for method in METHODS:
            item=groups.get((seed,method))
            if item:
                r=item[0]
                lines.append(f"|{seed}|{statuses[seed]['status']}|{method}|{r['anchor']}|{r['target_class']}|"
                             f"{_display(r['test']['acc'])}|{_display(r['test']['asr'])}|{r.get('cumulative_replacements','—')}|")
            else:lines.append(f"|{seed}|{statuses[seed]['status']}|{method}|—|—|—|—|—|")
    lines+=['','## 逐种子配对差值（自动−原始）','',
            '| seed | ACC差(pp) | ASR差(pp) |','|---|---|---|']
    for pair in pairs:
        lines.append(f"|{pair['seed']}|{_display(pair['delta_acc'])}|{_display(pair['delta_asr'])}|")
    lines+=['',*_describe_case(entry,protocol),'',_budget_text(case,protocol),'',_interpretation(protocol),'', '原始LFBA第2轮开始投毒，自动方法第4轮开始；累计预算不同。',
            '不完整配对不进入差值统计，已完成方法仍保留单项统计。','',
            '![Validation curves](validation_curves_lfba_vs_vote.png)','']
    (folder/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(max(1,len(seeds)),2,figsize=(11,3*max(1,len(seeds))),squeeze=False)
    for row,seed in enumerate(seeds):
        for col,key in enumerate(('acc','asr')):
            axis=axes[row,col]
            for method in METHODS:
                item=groups.get((seed,method))
                if item:
                    history=item[1]['history']
                    axis.plot([v['epoch'] for v in history],
                              [v['validation'].get(key) if v['validation'].get(key) is not None else float('nan') for v in history],label=LABELS[method],color=COLORS[method],linestyle=STYLES[method])
            axis.axvline(2,color=COLORS['lfba_original'],linestyle=':',alpha=.6)
            axis.axvline(4,color=COLORS['auto_vote'],linestyle=':',alpha=.6)
            axis.text(2,.02,'LFBA start: 2',rotation=90,fontsize=7,va='bottom')
            axis.text(4,.02,'Auto start: 4',rotation=90,fontsize=7,va='bottom')
            axis.set(title=f'Seed {seed}: validation {key.upper()}',xlabel='Epoch',ylabel='Score',ylim=(0,1))
            axis.grid(alpha=.25)
            if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=8)
    fig.suptitle(_plot_context(case,protocol),fontsize=11)
    fig.text(.5,.005,_plot_note(protocol),ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.95));fig.savefig(folder/'validation_curves_lfba_vs_vote.png',dpi=160);plt.close(fig)


def write_stability_report(root):
    root=Path(root);protocol=read_json(root/'protocol.json')
    if not protocol or protocol.get('kind')!='stability':raise ValueError('Missing stability protocol')
    for case in protocol['cases']:
        identifier=case['id']
        if (not isinstance(identifier,str) or Path(identifier).name!=identifier or
                (root/identifier).resolve().parent!=root.resolve()):
            raise ValueError('Stability case directory must stay directly inside the batch')
    seeds=protocol['seeds'];epochs=protocol['epochs'];all_rows=[];cases=[]
    for case in protocol['cases']:
        folder=root/case['id'];groups={};pairs=[];statuses={};failures=_failures(folder)
        protocol_error=_case_protocol_error(folder,case,protocol)
        for seed in seeds:
            errors={}
            for method in METHODS:
                try:
                    if protocol_error:raise ValueError(protocol_error)
                    item=verified_group(folder/f'seed_{seed}'/method,method,epochs,seed,case['select_rate'],strict=True)
                    config=read_json(folder/f'seed_{seed}'/method/'config.json')
                    if config is not None and (config.get('poison_rate')!=case['poison_rate'] or
                                               config.get('select_rate')!=case['select_rate']):
                        raise ValueError('Saved method rates differ from scan configuration')
                    groups[(seed,method)]=item
                    for row in item[2]:all_rows.append(dict(case_id=case['id'],poison_rate=case['poison_rate'],
                        select_rate=case['select_rate'],seed=seed,method=method,**row))
                except (ValueError,KeyError,TypeError,OSError) as error:errors[method]=str(error)
            if not errors:
                try:
                    records=(_pair_records(folder,epochs,seed,case['poison_rate'],case['select_rate'],
                        anchor_source='fixed',fixed_anchor=protocol['fixed_lfba_anchor']) if protocol.get('anchor_source')=='fixed' else
                        _pair_records(folder,epochs,seed,case['poison_rate'],case['select_rate']))
                    auto,original=records['auto_vote'],records['lfba_original']
                    if protocol.get('anchor_source','auto_vote')=='auto_vote' and (auto['anchor']!=original['anchor'] or auto['target_class']!=original['target_class']):
                        raise ValueError('Paired anchor or target differs')
                    pairs.append(dict(seed=seed,auto_anchor=auto['anchor'],original_anchor=original['anchor'],
                        auto_target=auto['target_class'],original_target=original['target_class'],
                        auto_counts=auto['test'].get('counts'),original_counts=original['test'].get('counts'),
                        delta_acc=auto['test']['acc']-original['test']['acc'],
                        delta_asr=auto['test']['asr']-original['test']['asr']))
                except (ValueError,KeyError,TypeError,OSError) as error:errors['pair']=str(error)
            if not errors:statuses[seed]=dict(status='complete')
            else:
                invalid_present=any((folder/f'seed_{seed}'/m/'metrics.json').exists()
                                    for m in METHODS if m in errors)
                status='failed' if protocol_error or seed in failures or 'pair' in errors or invalid_present else 'missing'
                statuses[seed]=dict(status=status,errors=errors,failure=failures.get(seed))
        paired={key:statistics_record([p[key] for p in pairs]) for key in ('delta_acc','delta_asr')}
        paired.update(positive_count=sum(p['delta_asr']>0 for p in pairs),
                      positive_fraction=(sum(p['delta_asr']>0 for p in pairs)/len(pairs)) if pairs else None,
                      seeds=[p['seed'] for p in pairs],records=pairs)
        methods={method:{key:statistics_record([item[0]['test'][key] for (seed,m),item in groups.items() if m==method])
                         for key in ('acc','asr')} for method in METHODS}
        entry=dict(case,expected_pairs=len(seeds),completed_pairs=len(pairs),
                   failed_pairs=sum(v['status']=='failed' for v in statuses.values()),
                   missing_pairs=sum(v['status']=='missing' for v in statuses.values()),
                   completed_methods=len(groups),methods=methods,paired=paired,
                   statuses={str(k):v for k,v in statuses.items()})
        candidate_count,q=_budget(case,protocol)
        entry.update(candidate_count=candidate_count,replacements_per_epoch=q,
                     budgets=dict(lfba_original=(epochs-1)*q,auto_vote=(epochs-3)*q))
        cases.append(entry)
        # Report-only must not occupy future trial directories before initialization.
        if (folder/'protocol.json').is_file() and not protocol_error:
            _case_report(folder,case,seeds,groups,pairs,statuses,protocol,entry)
            existing=read_json(folder/'summary.json') or {}
            existing['stability']=entry
            write_json(folder/'summary.json',existing)
    summary=dict(kind='stability',anchor_source=protocol.get('anchor_source','auto_vote'),
        fixed_lfba_anchor=protocol.get('fixed_lfba_anchor'),expected_pairs=len(cases)*len(seeds),expected_methods=2*len(cases)*len(seeds),
        completed_pairs=sum(c['completed_pairs'] for c in cases),completed_methods=sum(c['completed_methods'] for c in cases),
        failed_pairs=sum(c['failed_pairs'] for c in cases),missing_pairs=sum(c['missing_pairs'] for c in cases),
        seeds=seeds,cases=cases)
    summary['all_complete']=summary['completed_pairs']==summary['expected_pairs']
    # Training completion alone must not mark a batch successful before plotting.
    write_json(root/'summary.json',dict(summary,all_complete=False))
    fields=list(dict.fromkeys(k for row in all_rows for k in row)) or ['case_id','poison_rate','select_rate','seed','method','epoch']
    with (root/'all_epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    lines=['# '+_protocol_label(protocol),'',
           f"完整配对 {summary['completed_pairs']}/{summary['expected_pairs']}；失败 {summary['failed_pairs']}；缺失 {summary['missing_pairs']}。",'',
           '| p | s | 完成/预期 | ACC差均值±标准差(pp) | ASR差均值±标准差(pp) | ASR提升种子比例 |',
           '|---|---|---|---|---|---|']
    for c in cases:
        a,b=c['paired']['delta_acc'],c['paired']['delta_asr']
        lines.append(f"|{c['poison_rate']}|{c['select_rate']}|{c['completed_pairs']}/{c['expected_pairs']}|"
                     f"{_display(a['mean'])} ± {_display(a['std'])}|{_display(b['mean'])} ± {_display(b['std'])}|"
                     f"{_display(c['paired']['positive_fraction'])}%|")
    lines+=['','## 各方法的种子间波动','',
            '| p | s | 方法 | 完成种子数 | ACC均值±标准差(pp) | ASR均值±标准差(pp) |',
            '|---|---|---|---|---|---|']
    for c in cases:
        for method in METHODS:
            a,b=c['methods'][method]['acc'],c['methods'][method]['asr']
            lines.append(f"|{c['poison_rate']}|{c['select_rate']}|{method}|{a['n']}|"
                         f"{_display(a['mean'])} ± {_display(a['std'])}|{_display(b['mean'])} ± {_display(b['std'])}|")
    lines+=['','标准差为种子间样本标准差；不足两个观测时不定义，显示为 —。',
            f"每轮替换 Q=floor(floor(Np)s)，原始LFBA累计 {epochs-1}Q，自动方法累计 {epochs-3}Q。",
            _interpretation(protocol),
            '失败与缺失不按零计入，不完整批次不作完整稳定性结论。','',
            '![Stability](stability_parameters.png)','']
    lines+=['## 分配置结果说明','']
    for case in cases:
        lines += [f"### p={case['poison_rate']}, s={case['select_rate']}",'',_budget_text(case,protocol),*_describe_case(case,protocol),'']
    lines+=['## 参数变化趋势','',*_describe_trends(cases,protocol),'']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    _parameter_plot(root,protocol,cases)
    write_json(root/'summary.json',summary)
    return summary


def _parameter_plot(root,protocol,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    baseline=protocol['baseline'];fig,axes=plt.subplots(2,2,figsize=(15,10),squeeze=False)
    for col,(parameter,fixed) in enumerate((('poison_rate','select_rate'),('select_rate','poison_rate'))):
        selected=sorted([c for c in cases if c[fixed]==baseline[fixed]],key=lambda c:c[parameter])
        for row,metric in enumerate(('acc','asr')):
            axis=axes[row,col]
            for index,method in enumerate(METHODS):
                valid=[c for c in selected if c['methods'][method][metric]['mean'] is not None]
                if valid:
                    x=[c[parameter] for c in valid];y=[c['methods'][method][metric]['mean'] for c in valid]
                    axis.plot(x,y,marker='o' if index==0 else 's',color=COLORS[method],linestyle=STYLES[method],label=LABELS[method])
                    for c,xv,yv in zip(valid,x,y):
                        stat=c['methods'][method][metric]
                        if stat['std'] is not None:
                            axis.errorbar([xv],[yv],yerr=[stat['std']],capsize=4,color=COLORS[method])
                        label=f"{yv:.2%} (n={stat['n']})"+('*' if stat['std'] is None else '')
                        axis.annotate(label,(xv,yv),xytext=(-5 if c is valid[-1] else 5,12 if index==0 else -21),
                                      ha='right' if c is valid[-1] else 'left',textcoords='offset points',fontsize=8,color=COLORS[method])
            axis.axvline(baseline[parameter],color='gray',linestyle=':',label='Baseline parameter')
            axis.set(xlabel='p' if parameter=='poison_rate' else 's',ylabel=f'Test {metric.upper()}',ylim=(-.08,1.15),
                     title=f"Test {metric.upper()} vs {'p' if parameter=='poison_rate' else 's'}; fixed {'s' if fixed=='select_rate' else 'p'}={baseline[fixed]}")
            axis.set_xticks([c[parameter] for c in selected]);axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.margins(x=.18);axis.grid(alpha=.25)
            if axis.lines:axis.legend(fontsize=8,loc='lower left')
    fig.suptitle(f"{protocol['dataset']} stability: epoch {protocol['epochs']} test metrics; mean +/- sample SD",fontsize=13)
    fig.text(.5,.02,_plot_note(protocol)+'\nMissing/failed seeds excluded; n shown per point; * n=1: SD unavailable.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(root/'stability_parameters.png',dpi=180);plt.close(fig)
