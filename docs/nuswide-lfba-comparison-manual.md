# NUS-WIDE：auto_vote 与原始 LFBA 对比

准备日期：2026-11-06。复用 `scripts/compare_lfba_auto_vote.py`，使用项目已有 Python 环境。本次准备不启动真实数据训练，以下短训练和正式训练由用户手动运行。

## 实验协议

| 项目 | 设置 |
|---|---|
| 数据集参数 | `NUSWIDE`（兼容 `NUSWID` 拼写，保存时统一为 `NUSWIDE`） |
| 类别顺序 | 0 buildings、1 grass、2 animal、3 water、4 person |
| 筛选 | 五个所选标签之和恰好为 1，沿用现有加载器 |
| 数据规模 | 筛选后的官方训练集 69966、官方测试集 46693 |
| 训练／验证／测试 | 62969／6997／46693；官方训练集按种子分层 90/10，保留固定锚点 |
| 两方特征 | 第一方 634 维图像特征；攻击方 1000 维文本特征 |
| 种子 | 2004、592、3383、100；每种子先 lfba_original，再 auto_vote，共 8 个分支 |
| 训练 | 40 轮、batch 128、Adam、lr=0.001，无学习率衰减、无早停 |
| 比例与预算 | p=0.1、s=0.5；P=floor(62969×0.1)=6296，Q=floor(6296×0.5)=3148 |
| 触发器 | 按种子选定攻击方索引 [634,1634) 内的 10 维，设为 1.0，两方法共享 |
| 最终评估 | 固定第 40 轮，重新加载最终检查点独立复评 |

| 方法 | 锚点／选样 | 投毒轮次 | 累计替换 |
|---|---|---|---:|
| lfba_original | 固定锚点 33930，首轮梯度、legacy_l1 | 第 2—40 轮 | 122772 |
| auto_vote | 自由选锚点，前三轮 cosine 邻居投票 | 第 4—40 轮 | 116476 |

固定锚点 33930 和 10 维触发器沿用项目 README 的 NUS-WIDE 示例。锚点 ID 指筛选后官方训练集中的样本 ID，当前文件对应 water（类别 3）。训练时长、batch 和 s 沿用此前对比协议，因此不能将本批次描述为论文全部超参数的原样复现。

同种子两方法共享首轮模型、优化器、随机状态、划分和触发维度；候选构建后冻结。候选选择不接收标签，选样后绑定目标类别。保留现有困难样本选择和候选集外供体采样规则。这是完整策略比较：目标类别、投毒起点、累计预算可能不同，不把性能差异单独归因于投票。

## PowerShell 初始化

在 PowerShell 中先执行下面整段。`--data-dir` 指向 `NUS_WIDE` 的父目录。

```powershell
Set-Location -LiteralPath 'D:\MyHomework\2026_2\LFBA\second\LFBA'
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
$comparisonScript = '.\scripts\compare_lfba_auto_vote.py'
$comparisonArgs = @(
    '--dataset', 'NUSWIDE', '--anchor-source', 'fixed',
    '--poison-rate', '0.1', '--select-rate', '0.5', '--device', '0',
    '--data-dir', '.\data'
)
$formalArgs = $comparisonArgs + @(
    '--seeds', '2004', '592', '3383', '100',
    '--output', '.\results\nuswide_lfba_vs_vote_p01_s05_v1'
)
$smokeArgs = $comparisonArgs + @(
    '--smoke', '--seeds', '2004',
    '--output', '.\results\nuswide_lfba_vs_vote_p01_s05_v1_smoke'
)
```

系统默认 `python` 和项目环境可能不同。上述完整路径使用已有 PyTorch 环境，无需安装或升级依赖。

## 运行顺序

预检只读取标签计数和核对文件清单，不创建模型、不启动训练：

```powershell
& $experimentPython -B $comparisonScript @formalArgs --dry-run
& $experimentPython -B $comparisonScript @smokeArgs --dry-run
```

正式预检应显示 `tasks=8`、`epochs=40`、`candidates=6296`、`replacements_per_epoch=3148`，双方预算分别为 122772、116476，`missing_data_files=[]`。新批次输出目录应未占用。特征表的完整行列数在数据加载时检查。

先运行种子 2004 的短训练：

```powershell
& $experimentPython -B $comparisonScript @smokeArgs --run
```

短训练为 500 条源训练样本（450 训练／50 验证）、100 测试、5 轮、P=45、Q=22；原始 LFBA 累计替换 88 次，auto_vote 为 44 次。保留固定锚点。短训练仍需读取原始特征文件后再取子集，启动阶段不会像训练样本数一样缩小。短训练仅用于流程验收，其 ASR 不作为正式性能证据。

看到 `ALL_TASKS_COMPLETE` 且生成两方法报告后，运行正式实验：

```powershell
& $experimentPython -B $comparisonScript @formalArgs --run
```

正式实验在 GPU 0 串行运行。NUS-WIDE 的训练集和测试集均大于 UCIHAR，自动方法还要进行分块近邻计算；不要用 UCIHAR 的耗时估算本实验。选样阶段仍使用原有精确选择器和默认 `--block-size 128`。

## 中断续训与报告重建

中断后，在相同目录、环境和参数下运行：

```powershell
& $experimentPython -B $comparisonScript @formalArgs --run --resume
```

短训练中断则使用：

```powershell
& $experimentPython -B $comparisonScript @smokeArgs --run --resume
```

续训要求冻结协议一致，并在加载数据时核对数据指纹。不要在同一批次中更换种子、设备、比例或代码。两方法入口当前不强制检查源码摘要，因此应由运行者保留同一份代码。

已完成结果的报告重建不训练，也不需要原始数据文件：

```powershell
& $experimentPython -B $comparisonScript @formalArgs --report-only
```

## 输出及解释

正式结果位于 `results/nuswide_lfba_vs_vote_p01_s05_v1`，短训练使用独立的 `_smoke` 目录。主要文件：

- `protocol.json`：冻结协议，包含类别顺序、数据规模、特征切分和触发维数。
- `results.md`、`summary.json`：逐种子两方法结果及配对差值。
- `validation_curves_lfba_vs_vote.png`：验证集 ACC／ASR 曲线。
- `seed_*/{lfba_original,auto_vote}/`：配置、日志、训练审计、候选／供体记录、测试指标与独立复评记录。

ASR 排除各自目标类，分母由最终指标记录；因此不同目标的 ASR 分母可能不同。候选纯度、去锚点纯度、替换纯度与 ACC／ASR 一起解释。保留负向、持平及饱和结果，不据此更换种子、锚点或终点。

沿用现有输出策略：全部方法验证通过并成功生成报告后清理临时检查点；中断、复评失败或报告失败时保留恢复状态。长期结果记录足以重建报告，清理后不能仅靠它们再次加载模型。

## 本次验证记录

- `note/nuswide-comparison-preflight-2026-11-06.json`：真实数据正式预检。
- `note/nuswide-comparison-readiness-2026-11-06.json`：真实特征加载、数值有限性、四种子正式／短训练划分及触发维度核验。
- `note/nuswide-comparison-regression-2026-11-06.log`：新增测试与既有对比、检查点、续训和稳定性配置回归。
- `note/nuswide-core-entry-regression-2026-11-06.log`：四组核心消融与原入口回归。

两批测试共 65 项通过（54+11）；说明中的 7 段 PowerShell 命令均通过语法检查。已用真实数据运行正式和短训练预检，并完整加载特征验证形状及数值有限性；模型接口和替换检查使用合成数据，未启动真实 NUS-WIDE 训练。

真实 NUS-WIDE 短训练与正式训练仍由用户启动，以上检查不代替两者。
