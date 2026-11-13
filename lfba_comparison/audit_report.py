"""Standalone Chinese audit reports and descriptive research plots."""
from collections import defaultdict
import statistics


def pct(value):
    return '—' if value is None else f'{100*value:.3f}'


def markdown_table(headers, rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+[
        '|'+'|'.join(str(v).replace('|','/') for v in row)+'|' for row in rows]


def render(output, rows, batches, groups, pairs, diagnostics, triggers, issues, manifest_count):
    complete=[r for r in rows if r.get('status')=='complete' and r.get('origin')!='reference']
    ids={r['experiment_id'] for r in complete if r.get('experiment_id')}
    references=[r for r in rows if r.get('origin')=='reference']
    lines=['# 现有实验统一主表与失败机制诊断','',
        '本报告只读取保存结果与原始数据，没有训练、推理、重建旧报告或修改旧协议。',
        '历史实验按保存配置解释；当前默认锚点不用于改写历史身份。所有百分比均乘100；差值单位为百分点。','',
        '## 1. 清点与统计口径','',
        f'扫描得到 {len(complete)} 条已完成的实体结果记录，另有 {len(references)} 条旧结果引用；'
        f'其中具有足够摘要可确认身份的独立训练/评测结果有 {len(ids)} 个。'
        '这个数字覆盖正式、短训及历史记录，不能直接当作正式独立实验样本量。',
        '完全相同的数据、种子、目标、评测轮次、候选与末轮模型摘要才合并为同一实验身份。'
        '单轮与三轮共享前缀摘要语义不同，不以这个差异否定相同末轮模型的复用事实。',
        '同一实验可以出现在不同协议的展示表中；跨协议展示的 n 不得相加。'
        '同一统计组内每个种子只计一次；无法确认是否重复的同种子记录排除出统计并保留清单。',
        '主表空值表示未保存或不适用，不替换为0；源数据损坏、指标缺失与低ASR分开处理。'
        '分母保持各自目标定义，跨目标比较属于完整策略比较。','']
    important=[b for b in batches if b['category']=='formal']
    lines+=markdown_table(['批次','完成/预期','缺失','异常','引用完成/预期'],[
        [b['batch'],f"{b['complete']}/{b['expected']}",b['missing'],b['corrupt'],
         f"{b['complete_references']}/{b['expected_references']}"] for b in important])
    lines+=['','父级稳定性矩阵是子批次的合计，不新增训练；锚点×投票批次训练6组，另引用9组旧结果，'
            '其中6组进入四方法主矩阵，3组auto_mean作为补充参照。','',
        '## 2. 当前正式批次的分组结果','',
        '下表按当前主要批次展示，不将不同协议、参数和评测终点混合。标准差为种子间样本标准差；n=1时不定义。','']
    main_names=('lfba_stability_full/','ucihar_lfba_vs_vote_','nuswide_lfba_vs_vote_','core_ablation','anchor_vote_ablation_v1')
    selected=[g for g in groups if any(any(n in b.replace('\\','/') for n in main_names) for b in g['batches'])]
    lines+=markdown_table(['批次','方法','n','ACC均值±标准差(%)','ASR均值±标准差(%)'],[
        [','.join(g['batches']),g['method'],g['n'],
         pct(g['acc_mean'])+' ± '+pct(g['acc_std']),pct(g['asr_mean'])+' ± '+pct(g['asr_std'])] for g in selected])
    lines+=['','所有历史组与逐种子值见 `master.csv`、`group_summary.csv`；配对明细见 `paired.csv`。','',
        '## 3. CIFAR-10：候选比例与失败模式','',
        '以下覆盖六配置的全部四个种子。候选纯度衡量构建质量，错误事件计数衡量非目标标签接收到触发特征的次数；'
        '同一原始样本可以贡献多次事件。验证轨迹仅用于诊断，不替换第40轮测试终点。','']
    cifar=[d for d in diagnostics if d['dataset']=='CIFAR10']
    lines+=diagnostic_table(cifar)
    small=[d for d in cifar if d['poison_rate']==.01 and d['method']=='auto_vote']
    for d in small:
        baseline=next((b for b in cifar if b['poison_rate']==.01 and b['method']=='lfba_original' and b['seed']==d['seed']),None)
        if baseline:
            lines+=['',f"- seed={d['seed']}：自动ASR {pct(d['asr'])}%、原始 {pct(baseline['asr'])}%，差值 "
                f"{100*(d['asr']-baseline['asr']):+.3f}pp；自动候选纯度 {pct(d['candidate_purity_recomputed'])}%、"
                f"错误替换 {d['wrong_recipient_events']} 次；原始对应 {pct(baseline['candidate_purity_recomputed'])}%、"
                f"{baseline['wrong_recipient_events']} 次；目标类别分别为 {d['target_class']} / {baseline['target_class']}。"]
    if small and all(d['wrong_candidates']==0 and d['wrong_recipient_events']==0 for d in small):
        lines+=['','**直接结论：p=.01的四个自动组均没有错误候选，也没有错误替换事件。** '
            '因此，这批小候选池的负向收益不能用候选污染或错误暴露放大来解释。'
            '与默认p=.1相比，候选数从4500降至450，每轮替换也从2250降至225，自动目标类别由1/8变为2/4；'
            '候选规模、预算和目标同时变化，现有结果不能单独锁定哪一项导致退化。']
    large=[d for d in cifar if d['poison_rate']==.2 and d['method']=='auto_vote']
    if large:
        lines+=['',f"**p=.2存在候选容量限制：** 自动组每个候选池9000条，目标类训练样本只有 "
            f"{sorted({d['target_class_train_count'] for d in large})} 条，且锚点被排除；"
            f"纯度理论上限为 {pct(large[0]['candidate_purity_upper_bound'])}%。"
            f"实际候选纯度范围 {pct(min(d['candidate_purity_recomputed'] for d in large))}—"
            f"{pct(max(d['candidate_purity_recomputed'] for d in large))}%，错误替换事件 "
            f"{min(d['wrong_recipient_events'] for d in large)}—{max(d['wrong_recipient_events'] for d in large)} 次，"
            f"占全部事件 {pct(min(d['wrong_event_fraction'] for d in large))}—"
            f"{pct(max(d['wrong_event_fraction'] for d in large))}%。"
            '这确认了多类候选及错误暴露，尚不能量化其对末轮ASR的独立因果贡献。']
    lines+=['','**解释边界：** 高纯度却低ASR的种子不能归因为候选污染；目标变化、较少替换预算和后门学习轨迹'
        '都可能参与差异。若存在错误候选且被反复替换，只能确认错误暴露，不能据此证明它独立导致全部ASR下降。',
        'p=.2 的候选规模达到训练集20%，应结合目标类别本身的样本数、候选组成和替换暴露判断；'
        '增加候选规模并不保证仍然属于同一个类别。','',
        '## 4. NUS-WIDE：低比例目标与候选错位','']
    nus=[d for d in diagnostics if d['dataset']=='NUSWIDE']
    lines+=diagnostic_table(nus)
    for p,seed in [(.005,2004),(.001,3383)]:
        d=next((d for d in nus if d['poison_rate']==p and d['seed']==seed and d['method']=='auto_vote'),None)
        if d:
            lines+=['',f"**重点案例 p={p}、s={d['select_rate']}、seed={seed}：** 锚点 {d['anchor']}，"
                f"评测目标 {d['target_class']}，候选主类 {d['candidate_majority_class']}；"
                f"候选类别计数 {d['candidate_class_counts']}。候选纯度 {pct(d['candidate_purity_recomputed'])}%，"
                f"实际替换纯度 {pct(d['recipient_purity_recomputed'])}%，错误替换 {d['wrong_recipient_events']}/"
                f"{d['replacement_events']} 次，ASR {pct(d['asr'])}%。"
                f"末五轮验证ASR均值 {pct(d['validation_asr_last5_mean'])}%。"]
            if p==.005:
                lines+=['',f"候选中的非目标比例为 {pct(1-d['candidate_purity_recomputed'])}%，"
                    f"实际替换事件中的非目标比例为 {pct(d['wrong_event_fraction'])}%。"
                    '重复选择放大了非目标暴露；该种子候选主类仍为目标类，应描述为混入非目标候选，'
                    '不能与极低比例种子的完全目标错位混为一谈。']
    lines+=['','**直接事实与解释：** 候选主类与锚点目标错位时，选择出的梯度邻域未保持锚点的真实语义；'
        '如果替换计划继续从这些候选中取样，就会给触发器提供非目标类监督。该事实能定位选择阶段失败，'
        '但不能从现有记录还原局部间隔为何给该锚点高分，因为这些批次未保留逐轮梯度及全部候选分数。',
        '同配置其它种子是对照；p=.005/s=.3 与 p=.001/s=.1 同时改变两个参数，不能把两者差异归于单一预算因素。'
        'p=.05/s=.5只登记已完成的1个分支，不生成四种子性能结论。','',
        '## 5. UCIHAR：高纯度未转化为攻击效果','']
    har=[d for d in diagnostics if d['dataset']=='UCIHAR']
    lines+=diagnostic_table(har)
    train_triggers=[t for t in triggers if t['population']=='train']
    if train_triggers:
        lines+=['',f"已检查 {len(train_triggers)} 条方法×维度训练集统计；触发维度均位于攻击方281—560索引，"
            f"原始训练源指纹与保存配置匹配：{all(t['source_fingerprint_matches'] is True for t in train_triggers)}。",
            f"原始维度最小值范围 [{min(t['min'] for t in train_triggers):.5f}, {max(t['min'] for t in train_triggers):.5f}]，"
            f"最大值范围 [{min(t['max'] for t in train_triggers):.5f}, {max(t['max'] for t in train_triggers):.5f}]；"
            f"设为1.0后的平均绝对改变量范围 [{min(t['mean_absolute_change'] for t in train_triggers):.5f}, "
            f"{max(t['mean_absolute_change'] for t in train_triggers):.5f}]。完整分位数、目标/非目标分层见 `ucihar_triggers.csv`。"]
        lines+=['','这些维度的注入值处于原始特征范围边界，且平均绝对改变量明显非零，'
            '不支持“触发器根本没有改变输入”的解释。每次只设置5个攻击方维度；'
            '这是否足以改变模型决策，仍需模型证据，不能由输入统计单独判定。']
    if har:
        lines+=['',f"自动组接收者覆盖率为 {pct(min(d['recipient_coverage'] for d in har if d['method']=='auto_vote'))}—"
            f"{pct(max(d['recipient_coverage'] for d in har if d['method']=='auto_vote'))}%；"
            '原始组覆盖率均为100%。自动组虽有样本使用集中现象，但只有四个种子，'
            '且目标类别不同，不能据此断言覆盖率是ASR差异的根因。']
    lines+=['','**可确认：** 逐样本重算的候选纯度、实际替换纯度和数量可与保存指标核对；'
        '候选没有污染并不意味着模型必须利用触发器。现有输入应用代码只替换281维之后的攻击方特征，'
        '维持接收者标签与诚实方特征。此项属于当前源码静态核对，不能代替历史逐批次张量级重放。',
        '**待验证解释：** 特征触发强度、目标类别难度、诚实方特征提供的分类信号、接收者反复暴露及供体组成'
        '可能影响触发器学习。原始取值与改变量只能排查维度越界、触发退化等输入问题，不能确定模型内部使用了哪些特征。',
        '**证据不足：** 已清理模型，且没有保存逐样本预测与完整梯度，不能恢复混淆矩阵、触发后错误流向、'
        '梯度竞争或逐轮锚点分数。不能断言唯一根因，也不能以候选100%纯度宣称攻击应该成功。','',
        '## 6. 证据、验收与后续判断','',
        f'输入清单覆盖 {manifest_count} 个结果、原始数据及所引用源码文件；完成后逐一重新计算SHA-256。'
        '具体一致性结果见 `verification.json`。原始大模型文件从未加载或写入。',
        'CIFAR-10、NUS-WIDE标签按原训练源顺序读取；NUS-WIDE按五类恰好一个正标签过滤。'
        '标签文件的本次摘要保存在输入清单中，历史完整特征源未重载，不能把本次文件摘要当成历史源码/数据版本证明。',
        f'发现 {len(issues)} 条异常或证据缺口，逐项列在 `issues.json`；异常分支不按零指标汇总。',
        '优先依据本报告区分“选样错误”“目标错位”“高纯度但学习不足”，再设计固定目标、预算和终点的反事实实验。'
        '现阶段不从描述性相关推导因果关系，不按观察结果更换种子或评测终点。','',
        '![候选纯度与攻击效果](candidate_quality.png)','',
        '![错误事件与攻击效果](wrong_exposures.png)','',
        '![CIFAR-10验证轨迹](validation_cifar10.png)','',
        '![NUS-WIDE验证轨迹](validation_nuswide.png)','',
        '![UCIHAR验证轨迹](validation_ucihar.png)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def diagnostic_table(rows):
    return markdown_table(['p/s','seed','方法','目标','ASR(%)','候选纯度(%)','错误事件/总事件',
                           '接收者覆盖(%)','最大重复次数','末5轮验证ASR(%)'],[
        [f"{d['poison_rate']}/{d['select_rate']}",d['seed'],d['method'],d['target_class'],pct(d['asr']),
         pct(d['candidate_purity_recomputed']),f"{d['wrong_recipient_events']}/{d['replacement_events']}",
         pct(d['recipient_coverage']),d['max_recipient_uses'],pct(d['validation_asr_last5_mean'])]
        for d in sorted(rows,key=lambda d:(d['poison_rate'],d['select_rate'],d['seed'],d['method']))])


def plot(output,diagnostics,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    colors={'auto_vote':'#d65137','lfba_original':'#2878a0'}
    for name,field,xlabel in [('candidate_quality','candidate_purity_recomputed','Candidate purity (%)'),
                              ('wrong_exposures','wrong_recipient_events','Wrong recipient events')]:
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        for ax,dataset in zip(axes,['CIFAR10','NUSWIDE','UCIHAR']):
            for method,color in colors.items():
                ds=[d for d in diagnostics if d['dataset']==dataset and d['method']==method]
                scale=100 if field=='candidate_purity_recomputed' else 1
                ax.scatter([d[field]*scale for d in ds],[100*d['asr'] for d in ds],
                           color=color,label=method,alpha=.75,edgecolors='white',linewidth=.5)
            ax.set(title=dataset,xlabel=xlabel,ylabel='Final test ASR (%)',ylim=(-3,103))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(output/(name+'.png'),dpi=160);plt.close(fig)
    for dataset in ('CIFAR10','NUSWIDE','UCIHAR'):
        ds=[d for d in diagnostics if d['dataset']==dataset]
        batches=sorted({d['batch'] for d in ds});seeds=sorted({d['seed'] for d in ds})
        if not batches or not seeds:continue
        fig,axes=plt.subplots(len(batches),len(seeds),figsize=(13,2.3*len(batches)),
                              squeeze=False,layout='constrained')
        for i,batch in enumerate(batches):
            for j,seed in enumerate(seeds):
                ax=axes[i,j]
                for method,color in colors.items():
                    values=[r for r in curves if r['batch']==batch and r['seed']==seed
                            and r['method']==method and r['asr'] is not None]
                    ax.plot([r['epoch'] for r in values],[100*r['asr'] for r in values],color=color,label=method)
                d=next(d for d in ds if d['batch']==batch)
                ax.set(title=f"p={d['poison_rate']}, s={d['select_rate']}, seed={seed}",
                       ylim=(-3,103),xlabel='Epoch',ylabel='Validation ASR (%)')
                ax.grid(alpha=.2)
                if i==0 and j==0:ax.legend(fontsize=7)
        fig.savefig(output/('validation_'+dataset.lower()+'.png'),dpi=150);plt.close(fig)
