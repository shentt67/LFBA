# UCIHAR：自动锚点投票与原始 LFBA 对照

准备日期：2026-11-06。工作分支：`codex/ucihar-lfba-p01-s05`。

本批次仅比较 `lfba_original` 与 `auto_vote`，固定 p=0.1、s=0.5。原始 LFBA 固定锚点 0，自动方法自由选择锚点。准备工作不启动真实数据训练；短训练和正式训练均由用户手动执行。

## 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | 本地 UCIHAR，保留官方测试集 |
| 训练／验证／测试 | 6616／736／2947；官方训练数据按种子分层 90/10 划分，并保留锚点 0 |
| 特征切分 | 第一方前 281 维，攻击方后 280 维 |
| 种子 | 2004、592、3383、100，共 8 个方法训练分支 |
| 训练 | 40 轮、batch 128、Adam、lr=0.001，无衰减、无早停 |
| 触发器 | 攻击方原始维度索引 [281,561) 内固定 5 维，赋值 1.0 |
| 候选与替换 | P=floor(6616×0.1)=661；Q=floor(661×0.5)=330 |
| 最终评估 | 第 40 轮模型；重新加载后独立复评 |

| 方法 | 候选构建 | 首次投毒 | 累计替换 |
|---|---|---:|---:|
| lfba_original | 首轮梯度、锚点 0、legacy_l1 | 第 2 轮 | 39×330=12870 |
| auto_vote | 前三轮梯度、自动锚点、cosine 邻居投票 | 第 4 轮 | 37×330=12210 |

同一种子的两组共享首轮模型、优化器、随机状态、划分与触发维度。触发维度随种子确定，写入各组配置；候选集构建后冻结。选样不接收标签，目标类别在选样后绑定。

两组可能采用不同目标类别，投毒起点和预算也不同，因此这是完整策略比较，不把差异单独归因于投票。不启用稳定性扫描，不增加消融组、额外参数或其它数据集。

## PowerShell 初始化

在新开的 PowerShell 窗口中先运行以下设置。已有环境无需安装或升级依赖。

```powershell
$projectRoot = 'D:\MyHomework\2026_2\LFBA\second\LFBA'
Set-Location -LiteralPath $projectRoot
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
$comparisonScript = Join-Path $projectRoot 'scripts\compare_lfba_auto_vote.py'
$formalOutput = Join-Path $projectRoot 'results\ucihar_lfba_vs_vote_p01_s05_v1'
$smokeOutput = Join-Path $projectRoot 'results\ucihar_lfba_vs_vote_p01_s05_v1_smoke'
$comparisonArgs = @(
    '--dataset', 'UCIHAR', '--anchor-source', 'fixed',
    '--poison-rate', '0.1', '--select-rate', '0.5', '--device', '0',
    '--data-dir', (Join-Path $projectRoot 'data')
)
$formalArgs = $comparisonArgs + @('--seeds', '2004', '592', '3383', '100', '--output', $formalOutput)
$smokeArgs = $comparisonArgs + @('--smoke', '--seeds', '2004', '--output', $smokeOutput)
git branch --show-current
```

分支应为 `codex/ucihar-lfba-p01-s05`。若不符，先保存其它任务的工作并核对分支，不要强制切换或回滚。当前分支继承了原工作区未提交内容，本次没有自动提交、暂存或清理。

## 手动运行顺序

先预检，不创建模型、不进行推理或训练、不初始化 CUDA：

```powershell
& $experimentPython -B $comparisonScript @formalArgs --dry-run
& $experimentPython -B $comparisonScript @smokeArgs --dry-run
```

正式预检应显示 `tasks=8`、`candidates=661`、`replacements_per_epoch=330`、预算 12870 和 12210，且 `missing_data_files=[]`。开始新批次时输出目录应未占用；已有批次只能以相同协议续训，不能覆盖。

短训练由用户启动：

```powershell
& $experimentPython -B $comparisonScript @smokeArgs --run
```

短训练仅种子 2004：500 条源样本、划分 450/50、测试 100、5 轮，P=45、Q=22，原始 LFBA 累计替换 88 条，auto_vote 累计 44 条。短训练完成应显示 `ALL_TASKS_COMPLETE`，并生成两组结果和验证曲线；它只用于流程验收，不作为性能证据。

短训练成功后，启动正式实验：

```powershell
& $experimentPython -B $comparisonScript @formalArgs --run
```

正式实验使用 GPU 0 串行执行四种子、两方法，复用当前入口的执行顺序。不要同时向同一输出目录启动第二个进程。

## 续训和报告重建

在保留上述变量的窗口执行；新窗口需先重新执行初始化块。续训必须沿用相同种子、设备、数据路径和其余协议参数。

```powershell
# 正式实验续训
& $experimentPython -B $comparisonScript @formalArgs --run --resume

# 短训练续训
& $experimentPython -B $comparisonScript @smokeArgs --run --resume

# 正式报告重建：只读取已保存结果，不训练或推理
& $experimentPython -B $comparisonScript @formalArgs --report-only
```

完成组可以跳过；中断或失败时保留临时恢复状态。成功生成报告和图片后，入口按现有规则统一清理临时检查点。正式结果长期保留日志、配置、候选、逐轮攻击计划、指标和独立复评证明。

## 结果与统计

正式输出目录为 `results/ucihar_lfba_vs_vote_p01_s05_v1`，短训练目录在末尾增加 `_smoke`。根目录包含 `protocol.json`、`results.md`、`summary.json`、`all_epoch_metrics.csv`、`validation_curves_lfba_vs_vote.png`。每方法目录包含 `experiment.log`、`config.json`、`experiment_data.json`、`metrics.json`。

先查看逐种子 ACC、ASR、候选纯度、去锚点纯度、替换纯度、目标类别与累计替换。ASR 分别排除各组目标类，应同时核对 `counts.asr`。保留负向、持平与饱和结果，不更换种子、目标或终点。

现有双方法报告提供逐种子结果和配对均值。以下补充统计在全部正式结果完成后手动运行，使用现有校验器读取记录，不加载模型，也不修改原报告。它将双方均值、样本标准差和配对差值写入 `ucihar_statistics.json`；缺失或未完成时明确报错，不补零。

```powershell
@'
import json, statistics, sys
from pathlib import Path
root = Path(r'D:\MyHomework\2026_2\LFBA\second\LFBA')
sys.path.insert(0, str(root))
from lfba_comparison.common import read_json, write_json
from lfba_comparison.records import verified_group
output = root / 'results/ucihar_lfba_vs_vote_p01_s05_v1'
protocol = read_json(output / 'protocol.json')
assert protocol and protocol['dataset'] == 'UCIHAR' and not protocol['smoke']
assert protocol['seeds'] == [2004, 592, 3383, 100]
assert protocol['anchor_source'] == 'fixed' and protocol['epochs'] == 40
assert protocol['poison_rate'] == .1 and protocol['select_rate'] == .5
methods = ('lfba_original', 'auto_vote')
records = {m: [verified_group(output/f'seed_{s}'/m, m, 40, s, .5, strict=True)[0]
               for s in protocol['seeds']] for m in methods}
def stat(values):
    values = [100 * float(v) for v in values]
    return dict(n=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values))
result = dict(units='metrics: percent; paired differences: percentage points', methods={}, paired={})
for m, rows in records.items():
    result['methods'][m] = {
        'acc': stat([r['test']['acc'] for r in rows]),
        'asr': stat([r['test']['asr'] for r in rows]),
        'candidate_purity': stat([r['candidate_purity']['full'] for r in rows]),
        'candidate_purity_without_anchor': stat([r['candidate_purity']['without_anchor'] for r in rows]),
        'recipient_purity': stat([r['recipient_purity'] for r in rows])}
for metric in ('acc', 'asr'):
    differences = [b['test'][metric] - a['test'][metric]
                   for a, b in zip(records['lfba_original'], records['auto_vote'])]
    result['paired']['delta_'+metric] = stat(differences)
write_json(output / 'ucihar_statistics.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
'@ | & $experimentPython -B -
```

## 准备阶段验证记录

工作区基线：`docs/ucihar-comparison-baseline-2026-11-06.json`，记录原分支、提交、状态与已有相关文件摘要。

真实数据预检：`note/ucihar-comparison-preflight-2026-11-06.json`，记录正式／短训练协议、四种子划分、数据指纹、锚点与触发维度。验证直接检查替换前后特征和标签，不创建模型，不启动真实数据训练。

相关回归：`note/ucihar-comparison-regression-2026-11-06.log`，42 项测试通过，涵盖配置、选样、划分、攻击、检查点、输出、微型 CPU 续训、独立复评和恢复状态清理。
