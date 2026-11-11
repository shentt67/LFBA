# LFBA 与自动锚点多轮投票运行说明

当前 main 分支只运行 `lfba_original` 和 `auto_vote`。普通单配置模式默认三个种子；稳定性模式默认五个种子、六个单因素配置，每对先自动方法、再使用固定锚点23470的原始LFBA。

## 运行命令

使用既有 Python 环境，在项目根目录运行：

```powershell
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
& $experimentPython scripts/compare_lfba_auto_vote.py --dry-run
& $experimentPython scripts/compare_lfba_auto_vote.py --run --smoke --output results/two_arm_smoke
& $experimentPython scripts/compare_lfba_auto_vote.py --run --output results/two_arm_formal
```

续训使用相同输出目录和协议：

```powershell
& $experimentPython scripts/compare_lfba_auto_vote.py --run --output results/two_arm_formal --resume
& $experimentPython scripts/compare_lfba_auto_vote.py --report-only --output results/two_arm_formal
```

默认不训练，必须显式传入 `--run`。`--report-only` 仅读取保存的指标、历史和审计记录。

## 实验协议

正式实验 40 轮，短训练 5 轮；batch 128，Adam，学习率 .001，无学习率衰减、无早停。默认 CIFAR10，训练/验证/测试划分为 45000/5000/10000。

| 方法 | 候选构建 | 首个投毒轮次 | 默认每轮替换 | 40 轮累计替换 |
|---|---|---|---|---|
| lfba_original | 第一轮梯度、固定锚点 23470、legacy_l1 | 第 2 轮 | 2250 | 87750 |
| auto_vote | 前三轮梯度、自动锚点、余弦近邻投票 | 第 4 轮 | 2250 | 83250 |

用 `--poison-rate` 和 `--select-rate` 调整比例。候选构建后冻结，选样接口不接收标签。最终结果取最后一轮，并重新加载 last 检查点独立复评。

可用 `--dataset UCIHAR` 运行 UCIHAR，默认固定锚点为 0；`--anchor-source auto_vote` 会先执行自动方法，再将其锚点用于原始 LFBA。这属于另一种比较协议，需使用独立输出目录。

## 输出文件

批次根目录保留 `protocol.json`、`results.md`、`summary.json`、`all_epoch_metrics.csv` 和实验图片。预算、耗时及资源统计合并在汇总中；并发锁放在系统临时目录。

各方法目录长期只保留 `experiment.log`、`config.json`、`experiment_data.json`、`metrics.json`。配置包含数据划分和触发器参数；实验数据包含逐轮指标、候选、接收者及供体；结果包含独立复评证明。临时检查点位于各组 `.runtime/last_checkpoint.pth.tar`，失败或中断时保留，报告和图片成功生成后统一删除。共享首轮的恢复状态同样临时保存。

不再计算运行源码哈希，不生成 `source_hashes.json`、`source_snapshot.zip`、`environment.json` 、`selection.pt` 或 `best_checkpoint.pth.tar`。

续训只核对实验协议和检查点状态，不检查源码是否改变。数据指纹、模型状态、候选集和预算的一致性检查仍保留。

新单配置协议版本为4，完整方法稳定性子配置为5、批次协议为2（旧同锚点子配置为4、批次为1），短训练源样本数固定500并写入协议。历史版本仍可重建报告，但不直接续训为新协议；新实验请使用新目录。已完成组无需检查点即可跳过；报告重建不加载模型。旧文件不会因运行而自动迁移。报告入口按历史协议中的方法列表读取，可重建旧三组及早期投票实验；新实验仍只运行两组。重建会更新所选目录的报告和曲线，已有历史报告本次清理未覆盖。

原始 LFBA 与自动方法可能选择不同目标，且启动轮次和累计预算不同；结果应解释为完整策略比较，不能单独归因于投票。

## 检查

```powershell
& $experimentPython -B -m unittest tests.test_comparison_outputs tests.test_lfba_comparison tests.test_config tests.test_checkpoint_io -v
```

以上检查使用合成数据和模拟调度，不启动正式实验。`tests.test_compact_comparison` 额外验证微型 CPU 训练、中断恢复、复评/写入/绘图失败及成功清理。

## 历史清理

`python -B scripts/compact_experiment_outputs.py` 只检查；添加 `--apply` 才迁移清理主目录及两个工作树的实验输出。保留历史方法、报告、所有图片、日志和额外分析文件；成功实验合并数据后删除模型与冗余产物，未完成实验保留恢复状态。清理日志写入 `note/artifact-cleanup-*.json`。


## 完整方法稳定性测试

`--stability` 启用 CIFAR10 单因素扫描：基准 p=.1、s=.5；p 列表为 .01/.05/.1/.2，s 列表为 .3/.5/.7，基准去重后六个配置。默认种子为 2004、592、3383、100、101，共三十对、六十次方法训练。每个配置独立存放在 p_..._s_... 子目录。

```powershell
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --dry-run
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --run --smoke --output results/lfba_stability_full_smoke
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --run --output results/lfba_stability_full
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --run --resume --output results/lfba_stability_full
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --report-only --output results/lfba_stability_full
```

只有显式 `--run` 才训练。上述命令是运行说明，不代表已执行正式实验。

可用 `--poison-rates`、`--select-rates` 和 `--seeds` 自定义列表，例如 `--poison-rates .01 .05 .1 .2 --select-rates .3 .5 .7 --seeds 2004 592 3383 100 101`。这仍是单因素扫描，固定另一参数为基准值，不作笛卡尔积。续训必须重复相同自定义列表、种子及其余参数；报告重建直接读取保存的协议。

每对先完整运行自动锚点＋前三轮近邻投票，再运行固定锚点23470的原始LFBA；双方分别记录锚点ID、目标类别，不再传递锚点。共同划分保证23470在训练集中，短训练也将其纳入500条源样本。两组使用相同首轮模型、优化器、随机状态和划分；自动方法仍自由选择锚点。稳定性模式默认 `--anchor-source fixed`，显式指定 `auto_vote` 会被拒绝。

原始LFBA使用第一轮梯度和legacy_l1，第二轮开始投毒；自动方法使用前三轮梯度和cosine投票，第四轮开始。正式终点均为40轮；Q=floor(floor(45000p)s)，预算为39Q和37Q。本实验是完整策略比较，预算与时序不同，不将收益单独归因于投票。ASR分别针对各自锚点确定的目标类别计算，因此两组目标类别和ASR分母可以不同。

短训练使用500条源样本，划分450/50，测试100条、5轮。最低默认p=.01仍有4个候选且替换至少1条。开始前检查候选数、供体、替换预算及2P+1约束；运行中有效梯度不足会明确失败，不缩小集合。

根目录和各配置保留 results.md、summary.json、all_epoch_metrics.csv 及图片。总报告包含各方法ACC/ASR均值与样本标准差，配对差值的均值、标准差、范围与正向比例；参数图误差棒是样本标准差，不是置信区间。缺失与失败不当零分，不完整配对不进入差值统计。

单个种子失败后保留恢复状态并继续其它种子和配置。模型临时状态在所有配置及总报告、图片均成功后统一清理。稳定性入口失败或不完整返回非零；report-only能够记录缺失而不占用尚未初始化的配置目录，也不会启动模型。

### 稳定性报告与图片标注

每个配置的 `results.md` 包含p、s、P、Q、双方预算、完成状态、逐种子指标与配对差值，并描述ACC和ASR的均值、样本标准差、最差表现、ASR提升种子比例。批次报告分别分析固定s扫描p、固定p扫描s的变化，并标明优势反转与阶段性结果；缺失和失败不补零。

`stability_parameters.png` 的四个面板标注固定参数、基准配置、均值、样本标准差误差棒和各点有效种子数n。单种子不画标准差误差棒。各配置训练曲线注明p、s、P、Q、总轮次、验证集指标和双方投毒起始轮次。图注注明预算差异与各自目标类别。最终表格为第40轮测试结果，短训练仅检查流程。

新实验默认写入 `results/lfba_stability_full`，短训练为 `results/lfba_stability_full_smoke`。不要向正在运行的旧目录启动新协议。旧同锚点批次保持原样，不能用新协议续训或复用训练结果；可显式指定旧目录以旧协议重建报告：

```powershell
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --report-only --output results/stability_formal_v1
```

### 已有批次缩减为四个种子

先在原训练窗口按一次Ctrl+C，等待进程退出。运行中的进程不会热加载修改后的代码。第一组的前四对结果已经完成时，以下命令会跳过它们，后续配置统一使用2004、592、3383、100：

```powershell
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --run --resume --reduce-seeds --seeds 2004 592 3383 100 --output results/lfba_stability_full
```

此操作仅允许缩减原种子列表且保持原顺序；其余协议参数必须完全一致。批次锁或配置锁仍被占用时会拒绝操作。迁移前先检查全部已有配置，并在 `seed_reduction.json` 保存原协议、子配置备份和迁移状态。迁移中断后重复同一命令可恢复。原始日志、数据、结果和被排除种子的检查点全部保留，排除种子不参与新的汇总。六配置、四种子共24对、48次方法训练。

后续恢复不再需要迁移选项，但仍须明确指定四个种子；报告重建自动读取保存的种子列表：

```powershell
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --run --resume --seeds 2004 592 3383 100 --output results/lfba_stability_full
& $experimentPython scripts/compare_lfba_auto_vote.py --stability --report-only --output results/lfba_stability_full
```
