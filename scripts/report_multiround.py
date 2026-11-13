"""Summarize paired seed-level selection diagnostics, without significance claims."""
import csv
import json
from pathlib import Path
import statistics
import sys


def report(root):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    raw=list(csv.DictReader((root/'selection_metrics.csv').open(encoding='utf-8')))
    seeds=sorted({x['seed'] for x in summary})
    lines=['# 多轮构建毒化候选集合：可行性验证','',
           '日期：2026-10-30。分支：codex/multiround-feasibility。',
           '本实验直接评估候选集合的同类纯度，未进行投毒训练，不报告攻击有效性或 ASR 改善。','',
           '## 实验口径','',
           '- CIFAR-10：每个种子完整训练 45,000 条，验证 5,000 条，各正常训练 5 轮；种子 100、101、102。',
           '- 每个种子按类随机选择 50 个评估锚点；历史锚点 23470 单列，不用其结果替代随机锚点总体结果。',
           '- 三种方法：首轮单轮、窗口末轮单轮、窗口内逐轮相似度的算术平均。固定相同候选数量。',
           '- 窗口 W=2/3/4/5；候选比例 .01/.05/.1；legacy_l1 和标准 cosine 分别比较。',
           '- 预指定主要配置：W=3，候选率=.1，legacy_l1。cosine 用于核对原论文的标准余弦定义。',
           '- 噪声比为噪声标准差/该轮梯度全局 RMS，取 0、1、3。额外噪声只用于离线选样诊断，不进入正常训练的反向传播。',
           '- 选样接口不接收标签；标签仅用于选择评估锚点和事后计算纯度。纯度排除锚点自身；不访问测试集进行选样。',
           '- 验证指标只用于既有训练入口保存模型，梯度分析始终使用预先规定的全部 5 轮，不按验证表现选轮。','',
           '## W=3、候选率 10%：跨种子均值','',
           '| 相似度 | 离线噪声比 | 首轮纯度 | 第3轮纯度 | 3轮平均纯度 | 平均−第3轮（百分点） | 各种子的配对差异（百分点） |',
           '|---|---:|---:|---:|---:|---:|---|']
    for mode in ('legacy_l1','cosine'):
        for noise in (0.,1.,3.):
            part=[x for x in summary if x['window']==3 and x['rate']==.1 and x['similarity']==mode and x['noise_rms_ratio']==noise]
            means={m:statistics.mean(x[m] for x in part) for m in ('single_first','single_last','multi_mean','delta_last')}
            lines.append(f"| {mode} | {noise:g} | {means['single_first']:.2%} | {means['single_last']:.2%} | {means['multi_mean']:.2%} | {100*means['delta_last']:+.3f} | "+', '.join(f"{x['seed']}: {100*x['delta_last']:+.3f}" for x in part)+' |')
    lines+=['','## 无额外噪声、候选率 10%：窗口敏感性','','| 相似度 | W | 首轮 | 末轮 | 多轮平均 | 平均−末轮（百分点） |','|---|---:|---:|---:|---:|---:|']
    for mode in ('legacy_l1','cosine'):
        for window in (2,3,4,5):
            part=[x for x in summary if x['window']==window and x['rate']==.1 and x['similarity']==mode and x['noise_rms_ratio']==0]
            vals=[statistics.mean(x[m] for x in part) for m in ('single_first','single_last','multi_mean','delta_last')]
            lines.append(f'| {mode} | {window} | {vals[0]:.2%} | {vals[1]:.2%} | {vals[2]:.2%} | {100*vals[3]:+.3f} |')
    lines+=['','## 无额外噪声、W=3：候选规模敏感性','','| 相似度 | 候选比例 | 首轮 | 末轮 | 多轮平均 | 平均-末轮（百分点） |','|---|---:|---:|---:|---:|---:|']
    for mode in ('legacy_l1','cosine'):
        for rate in (.01,.05,.1):
            part=[x for x in summary if x['window']==3 and x['rate']==rate and x['similarity']==mode and x['noise_rms_ratio']==0]
            vals=[statistics.mean(x[m] for x in part) for m in ('single_first','single_last','multi_mean','delta_last')]
            lines.append(f'| {mode} | {rate:.0%} | {vals[0]:.2%} | {vals[1]:.2%} | {vals[2]:.2%} | {100*vals[3]:+.3f} |')
    lines+=['','## 主配置逐种子明细','','| seed | 首轮 | 末轮 | 多轮平均 | 差异（百分点） |','|---|---:|---:|---:|---:|']
    for x in summary:
        if x['window']==3 and x['rate']==.1 and x['similarity']=='legacy_l1' and x['noise_rms_ratio']==0:
            lines.append(f"| {x['seed']} | {x['single_first']:.2%} | {x['single_last']:.2%} | {x['multi_mean']:.2%} | {100*x['delta_last']:+.3f} |")
    lines+=['','## 历史锚点：W=3、候选率10%、无额外噪声','','| seed | 相似度 | 首轮 | 末轮 | 多轮平均 |','|---|---|---:|---:|---:|']
    for seed in seeds:
        for mode in ('legacy_l1','cosine'):
            part={x['method']:float(x['purity']) for x in raw if int(x['seed'])==seed and int(x['window'])==3
                  and float(x['rate'])==.1 and x['similarity']==mode and float(x['noise_rms_ratio'])==0 and int(x['historical_anchor'])}
            lines.append(f"| {seed} | {mode} | {part['single_first']:.2%} | {part['single_last']:.2%} | {part['multi_mean']:.2%} |")
    lines+=['','## 当前结论','',
            '- 候选率10%、W=3时，多轮平均在三个种子及两种相似度下均改善纯度；相对同等待时间的末轮单轮，legacy_l1提高5.781个百分点，cosine提高5.649个百分点。',
            '- 标准cosine只选前1%时，末轮单轮已达99.82%，多轮平均为99.48%，下降0.342个百分点；不能声称所有设置均更优。',
            '- 离线噪声比1时，多轮相对末轮提高约17个百分点，但尚未验证噪声进入反向传播的真实反馈场景。',
            '- 结果支持继续研究构建质量，尚不能证明ASR提高。下一步需固定总训练预算比较早期单轮、延迟单轮及多轮攻击。','',
            '## 解释边界','',
            '- 首轮与多轮的比较同时包含训练阶段变化；末轮与多轮的比较才控制了启动等待时间。',
            '- 这是沿共同干净轨迹比较构建质量，不能替代单轮方法从第2轮实际启动投毒后的端到端对照。',
            '- 额外离线噪声的改善只能说明选样诊断结果，不能声称抵抗真实梯度防御。',
            '- 三个种子及单一数据集只提供初步可行性证据，不作统计显著性、跨任务普适性或论文录用预测。',
            '- 可复现入口：scripts/multiround_feasibility.py；报告生成：scripts/report_multiround.py；逐锚点数据：selection_metrics.csv；种子汇总：summary.json。','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:35]))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    report(Path(sys.argv[1]))
