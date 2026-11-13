"""Redraw saved validation curves in Chinese without altering experiment sources."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    'fixed_single': dict(label='固定锚点＋末轮近邻', color='#4477AA', linestyle='-',
                         marker='s', markevery=(0, 6), linewidth=1.65, zorder=2),
    'fixed_vote': dict(label='固定锚点＋三轮投票', color='#EE7733', linestyle=(0, (5, 2)),
                       marker='^', markevery=(2, 6), linewidth=1.8, zorder=3),
    'auto_rank_last': dict(label='自动锚点＋末轮近邻', color='#228833', linestyle='-',
                           marker='v', markevery=(4, 6), linewidth=1.7, zorder=4),
    'auto_vote': dict(label='自动锚点＋三轮投票', color='#CC3311', linestyle=(0, (4, 3)),
                      marker='o', markevery=(0, 6), linewidth=1.8, zorder=6),
    'auto_mean': dict(label='自动锚点＋三轮相似度均值', color='#AA4499', linestyle='-',
                      marker='D', markevery=(3, 6), linewidth=2.7, zorder=5),
}


def configure_font():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        family = next((f for f in ('Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei') if f in available), None)
        if family is None:
            raise RuntimeError('A Chinese font is required for this figure')
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10})


def render(root, output):
    configure_font()
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    with (root/'all_epoch_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    methods = list(STYLES)
    seeds = protocol['seeds']
    series = {}
    for seed in seeds:
        for metric in ('acc', 'asr'):
            for method in methods:
                points = sorted((int(r['epoch']), 100*float(r[metric])) for r in rows
                                if int(r['seed']) == seed and r['method'] == method and r[metric] != '')
                if not points:
                    raise ValueError(f'Missing curve: {seed}/{method}/{metric}')
                series[seed, metric, method] = points
    overlaps = [seed for seed in seeds
                if series[seed, 'asr', 'auto_mean'] == series[seed, 'asr', 'auto_vote']]
    fig, axes = plt.subplots(len(seeds), 2, figsize=(16, 12.4), squeeze=False, sharex=True)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.075, right=.985, bottom=.205, top=.885, wspace=.15, hspace=.30)
    fig.suptitle('自动锚点与多轮投票：验证集训练曲线', y=.975, fontsize=21, fontweight='bold')
    fig.text(.5, .938, '比较锚点选择与候选构建方式；所有方法均从第 4 轮开始投毒',
             ha='center', fontsize=12, color='#526171')
    handles = [Line2D([], [], **{k:v for k,v in style.items() if k not in ('markevery', 'zorder')},
                      markersize=6, markerfacecolor='white', markeredgewidth=1.2)
               for style in STYLES.values()]
    handles.append(Line2D([], [], color='#7E8793', linestyle=':', linewidth=1.5, label='第4轮开始投毒'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008), ncol=3,
               frameon=False, fontsize=12, handlelength=3.5, columnspacing=2.5, labelspacing=.9)
    for i, seed in enumerate(seeds):
        for j, metric in enumerate(('acc', 'asr')):
            ax = axes[i, j]
            ax.set_facecolor('#FCFDFE')
            ax.axvspan(.5, 3.5, color='#E9EDF2', alpha=.65, zorder=0)
            for method, style in STYLES.items():
                points = series[seed, metric, method]
                trace_style = dict(style)
                marker_size = 4.8
                if metric == 'asr' and seed in overlaps:
                    if method == 'auto_mean':
                        trace_style.update(linewidth=4.3, zorder=5)
                        marker_size = 6
                    elif method == 'auto_vote':
                        trace_style.update(linewidth=2.1, linestyle=(0, (5, 4)), zorder=6)
                        marker_size = 7
                ax.plot([p[0] for p in points], [p[1] for p in points], **trace_style,
                        markersize=marker_size, markerfacecolor='white', markeredgewidth=1.15,
                        solid_capstyle='round', dash_capstyle='butt')
            ax.axvline(4, color='#7E8793', linestyle=':', linewidth=1.4, zorder=1)
            title = f'随机种子：{seed}'
            if metric == 'asr' and seed in overlaps:
                title += '（均值与投票曲线完全重合）'
            ax.set_title(title, loc='left', pad=9, fontweight='bold')
            ax.set_ylabel('干净准确率 ACC（%）' if metric == 'acc' else '攻击成功率 ASR（%）')
            ax.set_xlim(.5, protocol['epochs']+.8)
            ax.set_ylim((40, 85) if metric == 'acc' else (40, 102.5))
            ax.set_xticks([1]+list(range(5, protocol['epochs']+1, 5)))
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.grid(axis='y', color='#D8DFE7', linewidth=.7, alpha=.8)
            ax.grid(axis='x', color='#E7ECF1', linewidth=.6, alpha=.6)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            for name in ('bottom', 'left'):
                ax.spines[name].set_color('#AAB4C1')
            if i == len(seeds)-1: ax.set_xlabel('训练轮次')
            # Five data traces must exist in every panel, including exact overlaps.
            assert len([line for line in ax.lines if line.get_label() in
                        [style['label'] for style in STYLES.values()]]) == 5
    fig.text(.075, .125,
             f"参数：候选比例 p＝0.1，替换比例 s＝0.5；总训练 {protocol['epochs']} 轮。灰色区域为前三轮干净训练。",
             fontsize=11, color='#526171')
    fig.text(.075, .099,
             '显示说明：相似度均值用紫色实线与菱形标记，邻居投票用红色虚线与圆点标记；曲线重合时仍保留各自原始数值。',
             fontsize=10.5, color='#526171')
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.with_name(output.stem+'_original'+output.suffix)
    if output.exists() and not backup.exists(): shutil.copy2(output, backup)
    temporary = output.with_name(output.stem+'.rendering.png')
    fig.savefig(temporary, dpi=220, facecolor='white')
    plt.close(fig)
    temporary.replace(output)
    return dict(output=str(output), exact_asr_overlaps=overlaps,
                panels=len(seeds)*2, data_traces=len(seeds)*2*len(methods))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/anchor_vote_ablation_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--style', choices=('original', 'enhanced', 'overlay'), default='original',
                        help='Keep original chart styling by default; enhanced is the previous redesign')
    args = parser.parse_args()
    name = 'validation_curves_v2.png' if args.style == 'overlay' else 'validation_curves.png'
    output = args.output or args.results/name
    result = (render(args.results, output) if args.style == 'enhanced'
              else render_original(args.results, output, vote_dashed=args.style == 'overlay'))
    print(json.dumps(result, ensure_ascii=False))


def render_original(root, output, vote_dashed=False):
    """Preserve the source plot style; only translate and relocate the legend."""
    plt.rcdefaults()
    configure_font()
    chinese_font = font_manager.FontProperties(family=plt.rcParams['font.family'])
    plt.rcdefaults()  # Keep English axes/titles and the original Matplotlib font.
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    with (root/'all_epoch_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        histories = list(csv.DictReader(stream))
    fig, axes = plt.subplots(len(protocol['seeds']), 2,
                             figsize=(12, 3.4*len(protocol['seeds'])), squeeze=False)
    handles = []
    for i, seed in enumerate(protocol['seeds']):
        for j, metric in enumerate(('acc', 'asr')):
            ax = axes[i, j]
            for method, style in STYLES.items():
                rows = [r for r in histories if int(r['seed']) == seed
                        and r['method'] == method and r[metric] != '']
                if not rows: raise ValueError(f'Missing data for {seed}/{method}/{metric}')
                line_style = {}
                if vote_dashed and method == 'auto_vote':
                    line_style = dict(linestyle='--', zorder=4)
                elif vote_dashed and method == 'auto_mean':
                    line_style = dict(linestyle='-', zorder=3)
                line, = ax.plot([int(r['epoch']) for r in rows],
                                [float(r[metric])*100 for r in rows], label=style['label'], **line_style)
                if i == 0 and j == 0: handles.append(line)
            start_line = ax.axvline(4, color='gray', linestyle='--', label='第4轮开始投毒')
            if i == 0 and j == 0: handles.append(start_line)
            ax.set(xlabel='Epoch', ylabel='Validation '+metric.upper()+' (%)',
                   title=f'Seed {seed}; p=.1, s=.5; final epoch={protocol["epochs"]}')
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.yaxis.set_major_locator(MultipleLocator(5 if metric == 'acc' or span < 45 else 10))
            ax.grid(alpha=.2)
            assert len(ax.lines) == 6
    fig.tight_layout(rect=(0, .085, 1, 1))
    chinese_font.set_size(10)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .009), ncol=3,
               prop=chinese_font, frameon=True, handlelength=2.8, columnspacing=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.with_name(output.stem+'_original'+output.suffix)
    if output.exists() and not backup.exists(): shutil.copy2(output, backup)
    temporary = output.with_name(output.stem+'.rendering.png')
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    temporary.replace(output)
    return dict(output=str(output), style='overlay' if vote_dashed else 'original', panels=len(protocol['seeds'])*2,
                data_traces=len(protocol['seeds'])*2*len(STYLES), legend='Chinese, bottom')


if __name__ == '__main__': main()
