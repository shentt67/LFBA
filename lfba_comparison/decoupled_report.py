"""Fixed-threshold development evaluation; missing runs never become zero ASR."""
import math
from statistics import mean, stdev
import csv
import json
from pathlib import Path

THRESHOLDS = dict(mean_asr_delta_pp=-5., min_asr_delta_pp=-10.,
                  min_asr_pp=80., mean_acc_delta_pp=-1., max_std_ratio=.75)


def acceptance(new, baseline, seeds):
    def index(rows):
        result = {}
        for row in rows:
            if row['seed'] in result:
                raise ValueError('Duplicate seed in acceptance group')
            if row['seed'] not in seeds:
                raise ValueError('Unexpected seed in acceptance group')
            result[row['seed']] = row
        return result

    left, right = index(new), index(baseline)
    def valid(row):
        return row and all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k])
                           and 0 <= row[k] <= 1 for k in ('asr', 'acc'))
    paired = [s for s in seeds if valid(left.get(s)) and valid(right.get(s))]
    delta_asr = [100 * (left[s]['asr'] - right[s]['asr']) for s in paired]
    delta_acc = [100 * (left[s]['acc'] - right[s]['acc']) for s in paired]
    new_std = stdev([100 * left[s]['asr'] for s in paired]) if len(paired) > 1 else None
    old_std = stdev([100 * right[s]['asr'] for s in paired]) if len(paired) > 1 else None
    ratio = new_std / old_std if old_std is not None and old_std > 0 else None
    complete = len(paired) == len(seeds) and bool(seeds)
    checks = dict.fromkeys(('mean_asr', 'min_asr_delta', 'min_asr', 'mean_acc', 'asr_std'))
    if complete:
        # Tiny rounding tolerance only; no rounding before making the decision.
        checks.update(mean_asr=mean(delta_asr) >= THRESHOLDS['mean_asr_delta_pp'] - 1e-10,
                      min_asr_delta=min(delta_asr) >= THRESHOLDS['min_asr_delta_pp'] - 1e-10,
                      min_asr=min(100 * left[s]['asr'] for s in paired) >= THRESHOLDS['min_asr_pp'] - 1e-10,
                      mean_acc=mean(delta_acc) >= THRESHOLDS['mean_acc_delta_pp'] - 1e-10,
                      asr_std=None if ratio is None else ratio <= THRESHOLDS['max_std_ratio'] + 1e-12)
    return dict(complete=complete, paired_seeds=paired,
                mean_asr_delta_pp=mean(delta_asr) if paired else None,
                min_asr_delta_pp=min(delta_asr) if paired else None,
                mean_acc_delta_pp=mean(delta_acc) if paired else None,
                new_asr_std_pp=new_std, baseline_asr_std_pp=old_std, std_ratio=ratio,
                checks=checks, passed=complete and all(value is True for value in checks.values()),
                undefined_std_reason='Fewer than two observations or zero baseline variance' if ratio is None else None)


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def write_report(root):
    """Only read records/hash evidence; never import runner, models, or a trainer."""
    from lfba_comparison.common import read_json, write_json
    root = Path(root).resolve()
    write_json(root / 'verification.json', dict(status='in_progress'))
    try:
        return _write_report(root)
    except Exception as error:
        previous = read_json(root / 'summary.json')
        if isinstance(previous, dict):
            previous['all_complete'] = False
            previous['method_passed'] = {m: False for m in previous.get('method_passed', {})}
            for item in previous.get('acceptance', []):
                item['eligible'] = item['passed'] = False
            previous.setdefault('issues', []).append(dict(stage='report', error=f'{type(error).__name__}: {error}'))
            write_json(root / 'summary.json', previous)
        revoked = previous.get('acceptance', []) if isinstance(previous, dict) else []
        write_json(root / 'acceptance.json', revoked)
        _csv(root / 'acceptance.csv', revoked)
        (root / 'report.md').write_text(
            '# 报告重新核验失败\n\n先前通过结论已撤销。请检查协议和输入文件后重新生成报告。\n\n'
            + f'错误：{type(error).__name__}: {error}\n\n最终状态见 `verification.json`。\n', encoding='utf-8')
        write_json(root / 'verification.json', dict(status='failed', original_inputs_unchanged=False,
                   report_started_training=False, report_started_inference=False,
                   error=f'{type(error).__name__}: {error}'))
        raise


def _write_report(root):
    from lfba_comparison.common import digest, read_json, write_json
    from lfba_comparison.decoupled_policy import METHODS
    from lfba_comparison.decoupled_protocol import RATES, SEEDS, case_name
    from lfba_comparison.decoupled_records import (
        completed_record, load_references, required_json, verify_bound_manifest,
    )
    root = Path(root).resolve()
    protocol = required_json(root / 'protocol.json')
    if (protocol.get('kind') != 'cifar10_decoupled_selection' or protocol.get('version') != 1
            or protocol.get('thresholds') != THRESHOLDS or protocol.get('rates') != list(RATES)
            or protocol.get('methods') != list(METHODS)
            or protocol.get('seeds') != ([2004] if protocol.get('smoke') else SEEDS)):
        raise ValueError('Unrecognized or altered preregistered protocol')
    manifest = read_json(root / 'input_manifest.json')
    input_check = verify_bound_manifest(manifest, protocol)
    issues, rows, references, curves, details = [], [], [], [], []
    if not input_check['unchanged']:
        issues.append(dict(stage='inputs', error='Input verification unavailable or changed', detail=input_check))
    for seed in protocol['seeds']:
        if not protocol['smoke']:
            try:
                refs = load_references(protocol['reference_root'], seed)
            except Exception as error:
                issues.append(dict(seed=seed, stage='references', error=f'{type(error).__name__}: {error}'))
                refs = []
            for ref in refs:
                record = ref['record']
                references.append(dict(seed=seed, p=ref['p'], method=ref['method'], source=ref['source'],
                    data_digest=record['data_digest'], initial_digest=record['initial_digest'],
                    candidate_digest=record['candidate_digest'], final_model_digest=record['final_model_digest'],
                    new_training=False))
                rows.append(dict(seed=seed, p=ref['p'], method=ref['method'], source=ref['source'],
                    status='reference', acc=record['test']['acc'], asr=record['test']['asr'],
                    acc_denominator=record['test']['counts']['acc'], asr_denominator=record['test']['counts']['asr'],
                    anchor=record['anchor'], target_class=record['target_class'],
                    candidate_purity=record['candidate_purity']['full'], recipient_purity=record['recipient_purity'],
                    cumulative_replacements=record['cumulative_replacements']))
                data = required_json(Path(ref['source']) / 'experiment_data.json')
                curves.extend(dict(seed=seed, p=ref['p'], method=ref['method'], epoch=h['epoch'],
                                   acc=h['validation']['acc'], asr=h['validation']['asr']) for h in data['history'])
        for rate in RATES:
            for method in METHODS:
                output = root / f'seed_{seed}' / case_name(rate) / method
                row = dict(seed=seed, p=rate, method=method, source=str(output), status='missing', acc=None, asr=None)
                try:
                    record = completed_record(root, output, protocol)
                    if record is not None:
                        row.update(status='complete', acc=record['test']['acc'], asr=record['test']['asr'],
                            acc_denominator=record['test']['counts']['acc'], asr_denominator=record['test']['counts']['asr'],
                            anchor=record['anchor'], target_class=record['target_class'],
                            candidate_purity=record['candidate_purity'], recipient_purity=record['recipient_purity'],
                            candidate_count=record['candidate_count'], reference_count=record['reference_count'],
                            cumulative_replacements=record['cumulative_replacements'])
                        data = required_json(output / 'experiment_data.json')
                        choice = data['choice']
                        details.append(dict(seed=seed, p=rate, method=method, target_class=record['target_class'],
                                            **record['diagnostics'], **choice['diagnostics']))
                        curves.extend(dict(seed=seed, p=rate, method=method, epoch=h['epoch'],
                                           acc=h['validation']['acc'], asr=h['validation']['asr']) for h in data['history'])
                    elif (output / 'metrics.json').exists():
                        row['status'] = 'unverified'
                except Exception as error:
                    row['status'] = 'invalid'
                    issues.append(dict(seed=seed, p=rate, method=method, stage='branch', error=f'{type(error).__name__}: {error}'))
                rows.append(row)
    accepted, groups, pairs, overlaps = [], [], [], []
    for rate in RATES:
        for method in (*METHODS, 'lfba_original', 'auto_vote'):
            values = [r for r in rows if r['p'] == rate and r['method'] == method and r['status'] in ('complete', 'reference')]
            groups.append(dict(p=rate, method=method, n=len(values),
                               acc_mean=mean(r['acc'] for r in values) if values else None,
                               asr_mean=mean(r['asr'] for r in values) if values else None,
                               acc_std=stdev(r['acc'] for r in values) if len(values) > 1 else None,
                               asr_std=stdev(r['asr'] for r in values) if len(values) > 1 else None))
        base = [r for r in rows if r['p'] == rate and r['method'] == 'lfba_original' and r['status'] == 'reference']
        for method in METHODS:
            current = [r for r in rows if r['p'] == rate and r['method'] == method and r['status'] == 'complete']
            assessment = acceptance(current, base, protocol['seeds'])
            eligible = not protocol['smoke'] and input_check['unchanged'] and not issues
            if not eligible:
                assessment['passed'] = False
            accepted.append(dict(p=rate, method=method, eligible=eligible, **assessment))
            for new in current:
                for old in [r for r in rows if r['p'] == rate and r['seed'] == new['seed'] and r['status'] == 'reference']:
                    pairs.append(dict(seed=new['seed'], p=rate, method=method, reference=old['method'],
                                      asr_delta_pp=100 * (new['asr'] - old['asr']),
                                      acc_delta_pp=100 * (new['acc'] - old['acc']),
                                      new_target=new['target_class'], old_target=old['target_class'],
                                      new_asr_denominator=new['asr_denominator'], old_asr_denominator=old['asr_denominator']))
        for seed in protocol['seeds']:
            ready = [r for r in rows if r['seed'] == seed and r['p'] == rate and r['status'] == 'complete']
            if len(ready) == 2:
                mean_row = next(r for r in ready if r['method'] == 'decoupled_mean')
                vote_row = next(r for r in ready if r['method'] == 'decoupled_vote')
                mean_data = required_json(Path(mean_row['source']) / 'experiment_data.json')
                vote_data = required_json(Path(vote_row['source']) / 'experiment_data.json')
                a, b = (set(d['choice']['candidates']) for d in (mean_data, vote_data))
                overlaps.append(dict(seed=seed, p=rate, mean_vote_jaccard=len(a & b) / len(a | b),
                                     identical_candidates=a == b))
                pairs.append(dict(seed=seed, p=rate, method='decoupled_vote', reference='decoupled_mean',
                    asr_delta_pp=100 * (vote_row['asr'] - mean_row['asr']),
                    acc_delta_pp=100 * (vote_row['acc'] - mean_row['acc']),
                    new_target=vote_row['target_class'], old_target=mean_row['target_class'],
                    new_asr_denominator=vote_row['asr_denominator'], old_asr_denominator=mean_row['asr_denominator']))
    failures = read_json(root / 'failures.json') or []
    expected = len(protocol['seeds']) * 4
    complete = sum(r['status'] == 'complete' for r in rows)
    all_complete = complete == expected and not issues and not failures and input_check['unchanged']
    # Infrastructure failures must not be hidden by previously completed numerical results.
    if failures:
        for item in accepted:
            item['eligible'] = item['passed'] = False
    summary = dict(expected=expected, complete=complete, historical_references=len(references),
                   all_complete=all_complete, smoke=protocol['smoke'], groups=groups, acceptance=accepted,
                   method_passed={m: all_complete and all(a['passed'] for a in accepted if a['method'] == m) for m in METHODS},
                   issues=issues, failures=failures, input_verification=input_check)
    for name, content in [('experiments', rows), ('references', references), ('group_summary', groups),
                          ('paired', pairs), ('diagnostics', details), ('candidate_overlap', overlaps),
                          ('validation_history', curves), ('acceptance', accepted)]:
        write_json(root / (name + '.json'), content)
        _csv(root / (name + '.csv'), content)
    write_json(root / 'summary.json', summary)
    _markdown(root, summary)
    if curves:
        _plots(root, rows, curves)
    # Recheck inputs after report generation; the completion receipt is written last.
    final_check = verify_bound_manifest(manifest, protocol)
    if not final_check['unchanged']:
        summary['all_complete'] = False
        summary['method_passed'] = dict.fromkeys(METHODS, False)
        for item in summary['acceptance']:
            item['eligible'] = item['passed'] = False
        summary['input_verification'] = final_check
        write_json(root / 'summary.json', summary)
        write_json(root / 'acceptance.json', summary['acceptance'])
        _csv(root / 'acceptance.csv', summary['acceptance'])
        _markdown(root, summary)
    write_json(root / 'verification.json', dict(status='complete' if summary['all_complete'] else 'incomplete',
               original_inputs_unchanged=final_check['unchanged'], files_checked=final_check['files_checked'],
               completed_branches=complete, protocol_digest=digest(protocol),
               report_started_training=False, report_started_inference=False))
    return summary


def _markdown(root, summary):
    def number(value, factor=1):
        return '—' if value is None else f'{value * factor:.3f}'
    text = ['# CIFAR-10 固定参考邻域稳定性实验', '',
            f"新方法完成 {summary['complete']}/{summary['expected']}，历史引用 {summary['historical_references']} 条。",
            '短训只验证流程，不作性能验收。' if summary['smoke'] else '四个已有种子属于开发验收，不构成统计显著性或泛化保证。', '',
            'K固定为训练集10%；p只截取最终候选。平均与投票共享锚点、干净前缀、启动轮次和替换预算。',
            '原始LFBA从第2轮攻击，新方法从第4轮攻击；跨目标ASR保留各自分母，属于完整策略比较。',
            '历史可核验首轮完整状态摘要及前三轮指标轨迹；第二、三轮完整状态摘要未保存，不能声称这些历史状态逐位一致。', '',
            '|p|方法|n|ACC均值±样本标准差(%)|ASR均值±样本标准差(%)|',
            '|---|---|---|---|---|']
    for g in summary['groups']:
        text.append(f"|{g['p']}|{g['method']}|{g['n']}|{number(g['acc_mean'], 100)} ± {number(g['acc_std'], 100)}|{number(g['asr_mean'], 100)} ± {number(g['asr_std'], 100)}|")
    text += ['', '## 固定开发验收', '',
             '要求：平均ASR差≥−5pp；每种子差≥−10pp且ASR≥80%；平均ACC差≥−1pp；ASR样本标准差≤原始的75%。',
             '基线方差为零时稳定性比例不定义，不能据此宣称降低25%。缺失、异常及训练失败均阻止完整验收；不补零。', '',
             '|p|方法|平均ASR差(pp)|最差ASR差(pp)|标准差比|五项检查|通过|',
             '|---|---|---|---|---|---|---|']
    for a in summary['acceptance']:
        checks = ' / '.join('待定' if v is None else '通过' if v else '未通过' for v in a['checks'].values())
        text.append(f"|{a['p']}|{a['method']}|{number(a['mean_asr_delta_pp'])}|{number(a['min_asr_delta_pp'])}|{number(a['std_ratio'])}|{checks}|{'是' if a['passed'] else '否'}|")
    text += ['', '检查顺序为平均ASR差、最差ASR差、绝对ASR、平均ACC差、标准差比例。', '',
             '## 完成与证据', '',
             f"批次完整：{summary['all_complete']}；输入摘要保持一致：{summary['input_verification']['unchanged']}。",
             f"方法跨两个p通过情况：{summary['method_passed']}。", '',
             '候选纯度、覆盖率、重复次数、供体组成与验证轨迹是诊断证据，不参与在线选择。',
             '结果同时展示两方法，不按种子选择优胜者；相近表现不被解释为等效。', '',
             '逐种子、来源、配对、候选重叠、诊断及验证轨迹见同目录CSV/JSON。检查点与梯度证据保留，不自动清理。', '',
             '## 异常与未完成', '',
             '```json', json.dumps(dict(issues=summary['issues'], failures=summary['failures']), ensure_ascii=False, indent=2), '```', '']
    if summary['complete'] or summary['historical_references']:
        text += ['![ASR分布](asr_distribution.png)', '', '![验证轨迹](validation_curves.png)', '']
    (root / 'report.md').write_text('\n'.join(text), encoding='utf-8')


def _plots(root, rows, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lfba_comparison.decoupled_policy import METHODS
    from lfba_comparison.decoupled_protocol import RATES
    names = ('lfba_original', 'auto_vote', *METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, p in zip(axes, RATES):
        for x, method in enumerate(names):
            values = [100 * r['asr'] for r in rows if r['p'] == p and r['method'] == method and r['asr'] is not None]
            if values:
                ax.scatter([x] * len(values), values, s=30)
                ax.plot([x - .18, x + .18], [mean(values)] * 2, color='black')
        ax.axhline(80, linestyle='--', color='grey', label='ASR floor 80%')
        ax.set(xticks=range(4), xticklabels=names, ylim=(0, 103), title=f'p={p}', ylabel='Final test ASR (%)')
        ax.tick_params(axis='x', labelrotation=25)
    fig.savefig(root / 'asr_distribution.png', dpi=160)
    plt.close(fig)
    seeds = sorted(set(c['seed'] for c in curves))
    fig, axes = plt.subplots(2, len(seeds), figsize=(4 * len(seeds), 7), squeeze=False, constrained_layout=True)
    for row, p in enumerate(RATES):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            for method in names:
                data = [c for c in curves if c['p'] == p and c['seed'] == seed and c['method'] == method and c['asr'] is not None]
                if data:
                    ax.plot([c['epoch'] for c in data], [100 * c['asr'] for c in data], label=method, linewidth=1)
            ax.set(title=f'p={p}, seed={seed}', ylim=(0, 103), xlabel='Epoch', ylabel='Validation ASR (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=7)
    fig.savefig(root / 'validation_curves.png', dpi=160)
    plt.close(fig)
