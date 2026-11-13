"""Saved-only factorial report: four explicit contrasts and their interaction."""
import csv
from pathlib import Path

from lfba_comparison.common import read_json, write_json, digest
from lfba_comparison.core_report import stats, cell
from lfba_comparison.core_runner import saved_result, verify_prefix
from lfba_comparison.anchor_vote_policy import METHODS, AUTO_METHODS, ANCHOR
from utils.artifacts import runtime_checkpoint
from utils.checkpoint import load_checkpoint
from lfba_comparison.completion import load_completion

PRIMARY = METHODS + AUTO_METHODS
CONTRASTS = (('fixed_single','auto_rank_last'), ('fixed_vote','auto_vote'),
             ('fixed_single','fixed_vote'), ('auto_rank_last','auto_vote'))


def summarize(records, seeds, smoke=False):
    methods = METHODS if smoke else PRIMARY
    required = {(s,m) for s in seeds for m in methods}
    indexed = {}
    for row in records:
        key = (row['seed'],row['method'])
        if key in indexed: raise ValueError('Duplicate result')
        indexed[key] = row
    for seed in seeds:
        rows = [indexed[(seed,m)] for m in methods if (seed,m) in indexed]
        if len({r['start_state'] for r in rows}) > 1 or len({r['data_digest'] for r in rows}) > 1:
            raise ValueError('Paired starting state or data differs')
        for group in (METHODS, AUTO_METHODS):
            pair = [indexed[(seed,m)] for m in group if (seed,m) in indexed]
            if len({(r['anchor'],r['target_class']) for r in pair}) > 1:
                raise ValueError('Within-anchor comparison changes anchor or target')
    comparisons = {}
    for base, new in (() if smoke else CONTRASTS):
        paired = []
        for seed in seeds:
            a,b = indexed.get((seed,base)), indexed.get((seed,new))
            if a is None or b is None: continue
            paired.append(dict(seed=seed, acc=b['test']['acc']-a['test']['acc'],
                               asr=b['test']['asr']-a['test']['asr'],
                               same_target=a['target_class']==b['target_class']))
        comparisons[new+' - '+base] = dict(rows=paired,
                                           **{k:stats([r[k] for r in paired]) for k in ('acc','asr')})
    interaction = []
    if not smoke:
        for seed in seeds:
            if not all((seed,m) in indexed for m in PRIMARY): continue
            fs,fv,as_,av = [indexed[(seed,m)] for m in PRIMARY]
            interaction.append(dict(seed=seed, **{k:av['test'][k]-as_['test'][k]-fv['test'][k]+fs['test'][k]
                                                   for k in ('acc','asr')}))
    complete = len(required & indexed.keys())
    return dict(all_complete=required <= indexed.keys(), complete=complete, expected=len(required),
                comparisons=comparisons,
                interaction=dict(rows=interaction, **{k:stats([r[k] for r in interaction]) for k in ('acc','asr')}),
                methods={m:dict(n=sum((s,m) in indexed for s in seeds),
                                **{k:stats([indexed[(s,m)]['test'][k] for s in seeds if (s,m) in indexed])
                                   for k in ('acc','asr')}) for m in methods})


def load_records(root, protocol):
    from lfba_comparison.anchor_vote_runner import verify_manifest, file_hash
    manifest = protocol['source_manifest']
    records=[]; missing=[]; histories=[]; diagnostics={}
    archived_data = {}
    for seed in protocol['seeds']:
        folder = root/f'seed_{seed}'
        try:
            proof = load_completion(folder, protocol, seed)
        except (OSError, ValueError, KeyError, TypeError) as error:
            required = METHODS if protocol['smoke'] else PRIMARY
            missing.extend(dict(seed=seed, method=m, error=str(error)) for m in required)
            continue
        if proof is not None:
            selection = read_json(folder/'selection.json')
            diagnostics[str(seed)] = {m:selection['results'][m]['diagnostics'] for m in METHODS}
            if manifest:
                archive = read_json(folder/'references.json')
                if archive['source_manifest_digest'] != digest(manifest):
                    raise ValueError('Archived reference provenance differs')
                for reference in archive['records']:
                    result = reference['result']
                    records.append(dict(result, origin='reused', result_directory=reference['directory']))
                    archived_data[(seed, result['method'])] = reference['data']
            records.extend(dict(proof['records'][m], origin='new', result_directory=str(folder/m)) for m in METHODS)
            continue
        prefix = None
        if manifest:
            verify_manifest(manifest['files'])
            info = manifest['seeds'][str(seed)]
            prefix = load_checkpoint(info['prefix_path']); verify_prefix(prefix)
            if digest(prefix) != info['prefix_digest']: raise ValueError('Source prefix digest differs')
            for method, ref in info['references'].items():
                directory = Path(ref['directory'])
                result = saved_result(directory.parent, method, seed, 40, prefix, ref['choice'])
                if result is None: raise ValueError('Referenced completed result disappeared')
                row = dict(result, origin='reused', result_directory=str(directory))
                records.append(row)
        copied = runtime_checkpoint(folder/'clean_prefix')
        try:
            if prefix is None:
                prefix = load_checkpoint(copied); verify_prefix(prefix)
            elif copied.exists() and file_hash(copied) != manifest['seeds'][str(seed)]['prefix_file_hash']:
                raise ValueError('Local clean prefix differs from frozen source')
            selection = read_json(folder/'selection.json')
            if not selection or selection['prefix_digest'] != digest(prefix):
                raise ValueError('Missing or inconsistent selection')
            from lfba_comparison.anchor_vote_policy import select_fixed
            expected = select_fixed(prefix['comparison']['observations'], int(len(prefix['data_state']['train_ids'])*.1))
            if selection['results'] != expected['results']:
                raise ValueError('Saved fixed selection differs from gradient evidence')
            diagnostics[str(seed)] = {m:selection['results'][m]['diagnostics'] for m in METHODS}
        except Exception as error:
            missing.extend(dict(seed=seed,method=m,error=f'{type(error).__name__}: {error}') for m in METHODS)
            continue
        for method in METHODS:
            try:
                choice = selection['results'][method]
                if choice['anchor_id'] != ANCHOR: raise ValueError('Unexpected fixed anchor')
                result = saved_result(folder, method, seed, protocol['epochs'], prefix, choice)
                if result is None: raise ValueError('Not completed')
                records.append(dict(result,origin='new',result_directory=str(folder/method)))
            except Exception as error:
                missing.append(dict(seed=seed,method=method,error=f'{type(error).__name__}: {error}'))
    for record in records:
        directory = Path(record['result_directory'])
        data = archived_data.get((record['seed'], record['method']))
        if data is None:
            data = read_json(directory/'experiment_data.json')
        for row in data['audit']:
            histories.append(dict(seed=record['seed'],method=record['method'],epoch=row['epoch'],
                                  acc=row['validation']['acc'],asr=row['validation']['asr'],
                                  recipient_count=row['recipient_count'],
                                  cumulative_replacements=row['cumulative_replacements']))
        record['wrong_recipient_exposures'] = round(sum(
            row['recipient_count']*(1-row['recipient_purity']) for row in data['audit'] if row['recipient_count']))
    return records,missing,histories,diagnostics


def write_report(root):
    root = Path(root)
    protocol = read_json(root/'protocol.json')
    if not protocol or protocol.get('kind') != 'anchor_vote_ablation': raise ValueError('Missing anchor/vote protocol')
    records,missing,histories,diagnostics = load_records(root,protocol)
    summary = summarize(records,protocol['seeds'],protocol['smoke'])
    summary.update(new_complete=sum(r['method'] in METHODS for r in records),new_expected=2*len(protocol['seeds']),
                   records=records,missing=missing,diagnostics=diagnostics)
    write_json(root/'summary.json',summary)
    lines=['# 自动锚点 × 多轮投票消融','',
           f"新增训练完成 {summary['new_complete']}/{summary['new_expected']}；主矩阵完成 {summary['complete']}/{summary['expected']}。",'',
           '| 方法 | 来源 | n | ACC (%) | ASR (%) | 候选纯度 (%) | 错误候选累计攻击次数 |',
           '|---|---|---:|---:|---:|---:|---:|']
    display = list(METHODS) if protocol['smoke'] else list(PRIMARY)+['auto_mean']
    for method in display:
        rows=[r for r in records if r['method']==method]
        lines.append(f"|{method}|{'新训练' if method in METHODS else '旧结果只读引用'}|{len(rows)}|"+
                     '|'.join(cell(stats([r['test'][k] for r in rows])) for k in ('acc','asr'))+'|'+
                     cell(stats([r['candidate_purity']['without_anchor'] for r in rows]))+'|'+
                     cell(stats([r['wrong_recipient_exposures'] for r in rows]),1)+'|')
    lines += ['', '| seed | 方法 | 锚点 | 目标类别 | ACC (%) | ASR (%) | 候选纯度 (%) | 错误候选攻击次数 | 累计替换次数 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['seed']}|{r['method']}|{r['anchor']}|{r['target_class']}|{r['test']['acc']*100:.3f}|{r['test']['asr']*100:.3f}|{r['candidate_purity']['without_anchor']*100:.4f}|{r['wrong_recipient_exposures']}|{r['cumulative_replacements']}|")
    lines += ['', '## 配对差值与交互作用','', '| 对比 | n | ACC差 (pp) | ASR差 (pp) |', '|---|---:|---:|---:|']
    for name,pair in list(summary['comparisons'].items())+[('交互作用 (A-V−A-S)−(F-V−F-S)',summary['interaction'])]:
        lines.append(f"|{name}|{pair['asr']['n']}|{cell(pair['acc'])}|{cell(pair['asr'])}|")
    lines += ['', '| 对比 | seed | ACC差 (pp) | ASR差 (pp) | 同目标类别 |','|---|---:|---:|---:|---|']
    for name,pair in list(summary['comparisons'].items())+[('交互作用',summary['interaction'])]:
        for row in pair['rows']:
            lines.append(f"|{name}|{row['seed']}|{row['acc']*100:+.3f}|{row['asr']*100:+.3f}|{row.get('same_target','—')}|")
    lines += ['', '标准差为种子间样本标准差；缺失不补零，配对比较只包含对应组均有效的种子。',
              '四组均从同一种子的三轮干净状态分支，第4轮开始投毒，固定第40轮测试；短训练只作流程验收。',
              '固定锚点为23470；自动锚点沿用三轮邻域间隔百分位排名平均。投票按票数优先、平均相似度处理同票。',
              '自动锚点与固定锚点允许目标类别变化，比较的是自动选择策略；同锚点内的投票比较保持目标类别相同。',
              'auto_mean仅作补充参照，以区分多轮信息与离散投票的作用。未达到收益门槛不等于证明无差异。',
              '旧结果按文件摘要只读核验并归档必要记录；新分支独立复评后保留完成证明，成功时默认清理临时模型。','']
    if missing: lines += ['尚未完成或审计失败的新增分支见summary.json的missing字段。','']
    if histories: lines += ['![验证集曲线](validation_curves.png)','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    with (root/'all_epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','method','epoch','acc','asr','recipient_count','cumulative_replacements'])
        writer.writeheader();writer.writerows(histories)
    if histories: plot(root,histories,protocol)
    return summary


def plot(root,histories,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(protocol['seeds']),2,figsize=(12,3.4*len(protocol['seeds'])),squeeze=False)
    methods=list(METHODS) if protocol['smoke'] else list(PRIMARY)+['auto_mean']
    for i,seed in enumerate(protocol['seeds']):
        for j,metric in enumerate(('acc','asr')):
            ax=axes[i,j]
            for method in methods:
                rows=[r for r in histories if r['seed']==seed and r['method']==method and r[metric] is not None]
                if rows: ax.plot([r['epoch'] for r in rows],[r[metric]*100 for r in rows],label=method)
            ax.axvline(4,color='gray',linestyle='--',label='Poison starts')
            ax.set(xlabel='Epoch',ylabel='Validation '+metric.upper()+' (%)',title=f'Seed {seed}; p=.1, s=.5; final epoch={protocol["epochs"]}')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'validation_curves.png',dpi=150);plt.close(fig)
