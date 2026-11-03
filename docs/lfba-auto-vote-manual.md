# LFBA原始策略与自动锚点＋多轮投票：手动运行说明

## 分支与当前状态

独立分支 `codex/lfba-vs-auto-vote`，worktree为 `.worktrees/lfba-vs-auto-vote`，起点是已提交的修复版 `971f9f5`。旧实验文件和结果不变，新批次不复用旧梯度或训练结果。数据目录只读共用。

自动投票模块直接复用上一批正式实验归档中的源码，已核对SHA256：

```text
attack/neighbor_vote.py
ee720d602bcedebca7090acbe05f964079f180bb0c090573db4545a9e2f93ad8
```

交付阶段只进行纯算法、模拟状态、配置和入口检查，没有训练或模型推理。GPU完整执行路径由用户手动验收。

## 固定实验协议

种子106、107、108；LFBA预先固定锚点23470。各组统一45000/5000/10000数据划分，数据准备阶段均保留23470在训练集中。自动选择器只收到ID与梯度，不收到预设锚点或真实标签。

| 方法名 | 构建依据 | 首个投毒轮次 | 每轮替换 | 累计预算 |
|---|---|---|---|---|
|lfba_original|第1轮、固定锚点、legacy_l1|第2轮|2250|42750|
|lfba_matched|与原始策略同一候选集合|第2轮|第2—4轮2014，其余2013|38250|
|auto_vote|第1—3轮、自动锚点、余弦近邻投票|第4轮|2250|38250|

每组候选4500条冻结；batch128、Adam、lr=.001，20轮内不衰减，不早停。共享第1轮干净检查点；自动投票独立再进行两轮干净训练，LFBA不能使用这两轮未来信息。LFBA允许锚点参与原始排序，不强制其一定入选；投票仍排除锚点。

每轮从候选中按上一轮嵌入梯度L2范数选困难样本；供体从候选以外抽样，使用seed+17000独立Python随机状态，标签和诚实方特征沿用修复基线的保护方式。候选改变会改变供体池，不能宣称供体ID相同。触发器不变，不加入噪声或新评分。

所有最终结果来自最后第20轮，best检查点不参与最终评测。验证ACC/ASR仅记录；父训练器可能保存best文件，程序不会将其加载为最终模型。每组结束后重新组装评估器，加载last检查点独立复评，一致才写完成指标。

## 运行命令

进入新worktree并设置Python：

```powershell
Set-Location 'D:\MyHomework\2026_2\LFBA\second\LFBA\.worktrees\lfba-vs-auto-vote'
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
```

检查配置、输入数据文件及任务，不创建模型、不训练、不推理：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --dry-run
```

手动执行短训练路径检查：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --run --smoke
```

短训练使用200条源样本后划分为180/20，测试100条，共5轮；候选18条。原始组总替换36次，匹配组第2—5轮依次5、5、4、4次共18次，投票组第4—5轮各9次共18次。短训练不能用于性能判断。

短训练通过后，正式九组：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --run
```

中断后继续正式实验，或继续短训练：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --run --resume
& $experimentPython scripts\compare_lfba_auto_vote.py --run --smoke --resume
```

全部结束后从保存指标重建报告，不进行模型推理：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --report-only
```

正式目录为 `results/lfba_auto_vote_formal`，短训练目录为 `results/lfba_auto_vote_smoke`。可用 `--output PATH` 指定新目录；恢复和重建报告时必须使用原目录。`--data-dir PATH`指定原始CIFAR-10数据目录，默认指向原工作区的data。`--device -1`显式使用CPU，否则默认GPU0，无CUDA时明确报错。

默认及 `--dry-run` 均不启动训练；只有显式 `--run` 才能执行。实验顺序运行，日志写各组文件，所有任务结束后统一生成分析，不需要助手监听。输出已占用时拒绝覆盖，续训严格核对协议和运行源码。实验开始后不要修改运行源码；若需修改block-size等配置，请使用新输出目录开始新批次。

## 输出及结果判断

根目录保存协议、运行源码SHA256和ZIP、环境信息，以及：

- `results.md`：完整对照报告。
- `summary.json`：主要和预算补充比较。
- `all_epoch_metrics.csv`：所有逐轮指标。
- `budget_check.csv`：各组预算核对。
- `timing_resources.json`：共享首轮、额外观察、选样、投毒训练及复评耗时。
- `validation_curves_lfba_vs_vote.png`：原始LFBA与自动投票的验证曲线；预算匹配组的逐轮数据仍在CSV和history.json中。

每个种子保存 `clean_first/last_checkpoint.pth.tar`。每个方法目录保存最后检查点、choice.json、config.json、experiment.log、history.json、epoch_metrics.csv、replacement_ids.json、metrics.json及independent_evaluation.json。投票另保存selection.pt，其中有局部间隔、排名、各轮近邻、票数与同票诊断。

每个检查点均包含完整观察历史、冻结候选、累计预算、实际替换计划及随机状态。计时字段区分共享首轮、额外干净轮次、选样、投毒轮次和最终两次评测；epoch时间不包括全部检查点I/O，elapsed_branch记录该分支到最后快照的耗时，不应将这些存在重叠的字段再次全部相加。内存记录为进程/设备分配峰值，不能直接解释为选样独占内存。

主要比较为投票减原始LFBA，补充比较为投票减预算匹配LFBA。分别要求：平均ASR提高至少2个百分点、平均ACC下降不超过1个百分点，且至少2/3种子ASR提高并且ACC下降不超过1个百分点。保留负向与饱和结果，不替换种子、锚点或终点。改变默认种子列表后的结果只作探索性汇总，不适用正式门槛。

候选纯度报告完整集合及去锚点两种口径，并列出分母。目标类别可能不同：原版固定锚点对应类别，自动方法自由选目标。即便通过门槛，也只能描述本协议下完整策略收益，不能单独归因于投票或声称固定目标攻击提升。预算匹配仍有启动时间和每轮强度差异。三种子、一个LFBA锚点不支持统计显著性和普适性结论。

## 手动验收清单

- 先完成短训练，检查九组完成、预算核对通过、最后检查点独立复评一致。
- 短训练另开一个输出目录连续运行，再开一个输出目录中断续训；逐种子比较观察历史、候选、实际替换计划、最终模型摘要和随机状态。纯状态测试不能替代实际GPU恢复验证。
- 同种子三组初始化、数据划分、首轮摘要及实际加载的模型/优化器/RNG摘要一致。
- 两种LFBA候选摘要相同，构建只用第一轮；投票前三轮零投毒，候选随后冻结。
- 正式原始预算42750，另外两组38250；核对各轮而非只核对总数。
- 验证接收者标签和诚实方特征不变；最终评测仅使用last检查点。

## 纯代码检查命令

```powershell
& $experimentPython -m unittest tests.test_lfba_comparison tests.test_config tests.test_checkpoint_io -v
```

此命令只使用合成梯度、模拟状态和合成报告数据，不进行模型前向、反向或训练循环。不要运行 `unittest discover`：基线其他测试包含微型训练。

2026-10-31交付检查：上述23项检查通过；dry-run确认输入文件齐全、九组任务及42750/38250/38250预算，正式输出目录未创建。自动投票模块SHA256与上一批归档一致。此次没有启动任何训练或模型评测，代码尚未提交Git。

## 首轮恢复摘要误报修复

2026-10-31手动短训练发现：`Initialization or actual first-round restore differs`。根因是Adam加载检查点时可能保留CPU状态张量的引用（尤其step计数），分支训练因此会改写内存中的共享首轮参考。磁盘上的首轮文件未改变；重新读取磁盘首轮后，失败组保存的第5轮检查点可通过全部审计。

修复在加载前复制优化器状态，隔离共享参考；未移除或放宽状态一致性检查。新增真实Adam状态别名回归检查，只直接修改状态张量，不调用模型或优化器step。修复后24项不训练检查通过，本次没有重新启动实验。

由于运行源码已改变，原失败目录的源码摘要与修复版不同，不要直接对原目录使用 `--resume`，也不要改写其摘要。保留旧目录，用新目录手动重跑短训练：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --run --smoke --output results\lfba_auto_vote_smoke_fixed
```

该修复批次若中断，使用相同输出目录续训：

```powershell
& $experimentPython scripts\compare_lfba_auto_vote.py --run --smoke --output results\lfba_auto_vote_smoke_fixed --resume
```

短训练通过后，正式目录尚未启动，可以使用前述 `--run` 命令正常启动正式九组。
