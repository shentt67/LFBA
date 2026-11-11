"""Audit durable results and regenerate reports without training or inference."""
import csv
from pathlib import Path
import statistics

from lfba_comparison.common import read_json, write_json
from lfba_comparison.core_policy import METHODS
from lfba_comparison.completion import load_completion


def stats(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def cell(stat, scale=100):
    if stat['mean'] is None: return '—'
    return f"{stat['mean']*scale:.3f} ± {stat['std']*scale:.3f}" if stat['std'] is not None else f"{stat['mean']*scale:.3f} (n=1)"


def write_report(root):
    from lfba_comparison.core_runner import saved_result, verify_prefix
    from utils.checkpoint import load_checkpoint
    from utils.artifacts import runtime_checkpoint
    root = Path(root)
    protocol = read_json(root/'protocol.json')
    if not protocol: raise ValueError('Missing saved protocol')
    records = []; missing = []; epochs = []; selections = {}
    for seed in protocol['seeds']:
        folder = root/f'seed_{seed}'
        try:
            proof = load_completion(folder, protocol, seed)
            prefix = None
            if proof is None:
                prefix = load_checkpoint(runtime_checkpoint(folder/'clean_prefix'))
                verify_prefix(prefix)
            selection = read_json(folder/'selection.json')
            if selection is None: raise ValueError('Missing selection')
            from lfba_comparison.common import digest
            expected_prefix = proof['prefix']['digest'] if proof else digest(prefix)
            if selection['prefix_digest'] != expected_prefix: raise ValueError('Selection prefix differs')
            anchors = [selection['choices'][m]['anchor_id'] for m in METHODS[1:]]
            if len(set(anchors)) != 1: raise ValueError('B/C/D anchors differ')
            selections[str(seed)] = selection['diagnostics']
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing.extend(dict(seed=seed, method=m, error=str(error)) for m in METHODS)
            continue
        for method in METHODS:
            try:
                result = (proof['records'][method] if proof else
                          saved_result(folder, method, seed, protocol['epochs'], prefix, selection['choices'][method]))
                if result is None: raise ValueError('Not completed')
                records.append(result)
                data = read_json(folder/method/'experiment_data.json')
                epochs.extend(dict(seed=seed, method=method, epoch=row['epoch'],
                                   acc=row['validation']['acc'], asr=row['validation']['asr'],
                                   recipient_count=row['recipient_count'],
                                   cumulative_replacements=row['cumulative_replacements']) for row in data['audit'])
            except (OSError, ValueError, KeyError, TypeError) as error:
                missing.append(dict(seed=seed, method=method, error=str(error)))
    methods = {}
    for method in METHODS:
        rows = [r for r in records if r['method'] == method]
        methods[method] = dict(n=len(rows), acc=stats([r['test']['acc'] for r in rows]),
                               asr=stats([r['test']['asr'] for r in rows]),
                               candidate_purity=stats([r['candidate_purity']['without_anchor'] for r in rows]),
                               recipient_purity=stats([r['recipient_purity'] for r in rows]))
    pairs = {}
    for base, new in zip(METHODS[:-1], METHODS[1:]):
        rows = []
        for seed in protocol['seeds']:
            a = next((r for r in records if r['seed'] == seed and r['method'] == base), None)
            b = next((r for r in records if r['seed'] == seed and r['method'] == new), None)
            if a is None or b is None: continue
            if a['start_state'] != b['start_state'] or a['data_digest'] != b['data_digest']:
                raise ValueError('Paired starting state differs')
            if base != 'auto_single' and (a['anchor'] != b['anchor'] or a['target_class'] != b['target_class']):
                raise ValueError('Same-anchor comparison differs')
            rows.append(dict(seed=seed, delta_acc=b['test']['acc']-a['test']['acc'],
                             delta_asr=b['test']['asr']-a['test']['asr'], same_target=a['target_class']==b['target_class']))
        pairs[new+' - '+base] = dict(rows=rows, delta_acc=stats([r['delta_acc'] for r in rows]),
                                     delta_asr=stats([r['delta_asr'] for r in rows]))
    result = dict(all_complete=len(records)==4*len(protocol['seeds']), complete=len(records),
                  expected=4*len(protocol['seeds']), methods=methods, pairs=pairs,
                  missing=missing, records=records, diagnostics=selections)
    write_json(root/'summary.json', result)
    lines = ['# 四组核心消融实验', '', f"完成 {len(records)}/{result['expected']}；固定第 {protocol['epochs']} 轮评测。", '',
             '| 方法 | n | ACC (%) | ASR (%) | 去锚点候选纯度 (%) | 累计替换纯度 (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for method, summary in methods.items():
        lines.append('|'+method+'|'+str(summary['n'])+'|'+'|'.join(cell(summary[k]) for k in ('acc','asr','candidate_purity','recipient_purity'))+'|')
    lines += ['', '标准差为种子间样本标准差；缺失不补零。共同干净状态训练三轮，所有组第4轮开始投毒。', '',
              '| seed | 方法 | 锚点 | 目标类 | ACC (%) | ASR (%) | 错误候选数 | 累计替换数 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['seed']}|{r['method']}|{r['anchor']}|{r['target_class']}|{r['test']['acc']*100:.3f}|{r['test']['asr']*100:.3f}|{r['wrong_candidates']}|{r['cumulative_replacements']}|")
    lines += ['', '## 配对比较', '', '| 比较 | n | ACC差 (pp) | ASR差 (pp) |', '|---|---:|---:|---:|']
    for name, pair in pairs.items():
        lines.append(f"|{name}|{pair['delta_asr']['n']}|{cell(pair['delta_acc'])}|{cell(pair['delta_asr'])}|")
    lines += ['', '| 比较 | seed | ACC差 (pp) | ASR差 (pp) | 同目标类别 |', '|---|---:|---:|---:|---|']
    for name, pair in pairs.items():
        for row in pair['rows']:
            lines.append(f"|{name}|{row['seed']}|{row['delta_acc']*100:+.3f}|{row['delta_asr']*100:+.3f}|{row['same_target']}|")
    lines += ['', '## 候选构建诊断', '', '| seed | B∩C / B∪C | B∩D / B∪D | C∩D / C∪D | 截止票数 | 边界同票数 | 需相似度决胜数 | 共享选样耗时(s) |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed, d in selections.items():
        j = d['jaccard']; v = d['vote']
        lines.append(f"|{seed}|{j['auto_rank_last__auto_mean']:.4f}|{j['auto_rank_last__auto_vote']:.4f}|{j['auto_mean__auto_vote']:.4f}|{v['cutoff_votes']}|{v['boundary_tied_count']}|{v['selected_by_tiebreak']}|{d['shared_selection_seconds']:.3f}|")
    lines += ['', '选样由同一次共享计算产生，耗时按种子计一次，不将共享计算成本重复归到各方法。',
              'B−A比较锚点选择策略，目标类别可能变化；C−B及D−C固定锚点与目标。',
              '固定终点，保留持平、负向和饱和结果，不据结果更换种子或终点。', '',
              '![验证集曲线](validation_curves.png)', '']
    if missing: lines += ['未完成或审计失败的分支见 summary.json 的 missing 字段。', '']
    (root/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    with (root/'all_epoch_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed','method','epoch','acc','asr','recipient_count','cumulative_replacements'])
        writer.writeheader(); writer.writerows(epochs)
    if epochs: plot(root, epochs, protocol)
    return result


def plot(root, rows, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(protocol['seeds']), 2, figsize=(12, 3.3*len(protocol['seeds'])), squeeze=False)
    for i, seed in enumerate(protocol['seeds']):
        for j, metric in enumerate(('acc', 'asr')):
            ax = axes[i, j]
            for method in METHODS:
                points = [r for r in rows if r['seed']==seed and r['method']==method and r[metric] is not None]
                if points: ax.plot([r['epoch'] for r in points], [100*r[metric] for r in points], label=method)
            ax.axvline(4, color='gray', linestyle='--', label='Poison starts')
            ax.set(xlabel='Epoch', ylabel='Validation '+metric.upper()+' (%)', title=f'Seed {seed}; p=0.1, s=0.5; P={45 if protocol["smoke"] else 4500}, Q={22 if protocol["smoke"] else 2250}')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/'validation_curves.png', dpi=150); plt.close(fig)
