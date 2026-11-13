# 四组核心消融运行说明

本入口独立于原LFBA双方法及稳定性实验。仅运行CIFAR-10，四方法依次为A `auto_single`、B `auto_rank_last`、C `auto_mean`、D `auto_vote`。

每种子先干净训练三轮，保存模型、Adam、全部随机数和加载器状态、三轮梯度。四分支独立恢复这个完整状态，第4轮开始投毒，固定第40轮测试。B/C/D共享百分位排名平均所选锚点；B用末轮邻居，C用三轮相似度均值，D用邻居票数及平均相似度决胜。候选集排除锚点、构建后冻结；标签只用于选择完成后的审计和评估。

正式数据45000/5000/10000；p=.1，s=.5，P=4500，Q=2250，37轮累计83250条替换。batch128、Adam、lr=.001，无衰减、无早停。四种子2004、592、3383、100，共16分支，GPU0串行执行。保留原锚点23470参与划分，不限制自动方法选择锚点。

在项目根目录用既有环境执行：

```powershell
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
& $experimentPython scripts/run_core_ablation.py --dry-run
& $experimentPython scripts/run_core_ablation.py --run --smoke
& $experimentPython scripts/run_core_ablation.py --run
```

默认无`--run`只预检，不创建模型或启动CUDA。短训练默认仅种子2004，源样本500、划分450/50、测试100、5轮，P45、Q22、累计44。短训练结果不得作为性能证据。

正式目录`results/core_ablation_v1`，短训练目录`results/core_ablation_v1_smoke`；可显式`--output`指定新目录。输入目录使用`--data-dir`，设备使用`--device`（-1为CPU），种子使用`--seeds`；同一批次恢复必须重复相同协议参数。

```powershell
& $experimentPython scripts/run_core_ablation.py --run --resume
& $experimentPython scripts/run_core_ablation.py --report-only --output results/core_ablation_v1
```

恢复校验完整协议及相关源码摘要，不支持变更种子或迁移旧稳定性结果。已有有效完成分支跳过；失败保留状态、继续其它分支。批次未完成返回非零。报告重建读取完成证明、候选和指标；尚未生成完成证明的旧结果仍使用检查点校验，不进行推理或训练，不完整结果不补零。

训练期间每种子保存`clean_prefix/.runtime/last_checkpoint.pth.tar`和各分支的临时检查点，支持中断恢复。最终模型重新加载独立测试，指标必须完全相同。整个批次的分支审计、结果写入和报告绘图均成功后，生成并回读验证每种子的`completion_proof.json`，然后默认删除临时模型及空的`.runtime`/`runtime`目录；任一步失败均保留恢复文件。

长期保留`selection.pth.tar`（选样数值证据，不是模型）、`selection.json`、完成证明及各分支的`config.json`、`experiment.log`、`experiment_data.json`和`metrics.json`。模型删除后仍可执行`--report-only`重建报告、`--run --resume`跳过完整批次，无需原始数据。完成证明缺失或持久记录校验失败时，不把缺失检查点的结果视为完成。`--report-only`成功补全报告后也会执行上述清理；未知文件不会被删除。

如需将核心消融作为后续“自动锚点 × 多轮投票消融”的训练来源，在首次运行时显式保留模型：

```powershell
& $experimentPython scripts/run_core_ablation.py --run --keep-checkpoints --output results/core_ablation_for_reuse
& $experimentPython scripts/run_anchor_vote_ablation.py --run --source-results results/core_ablation_for_reuse
```

`--keep-checkpoints`写入冻结协议；续训时须重复传入。该策略下`--report-only`也保留模型。已经清理的来源不能仅凭完成证明提供新的训练起点，应在新目录以保留选项重新运行；已有结果目录不会被自动批量清理。

根目录产出`protocol.json`、`summary.json`、`results.md`、`all_epoch_metrics.csv`、`validation_curves.png`及`failures.json`。主表为均值±样本标准差，配对表列B−A、C−B、D−C；另有逐种子锚点/目标、错误候选、候选重合度及投票边界诊断。共享选样耗时每种子计一次，不作为独立算法耗时比较。最终评测固定终点，保留持平、负向和饱和结果。
