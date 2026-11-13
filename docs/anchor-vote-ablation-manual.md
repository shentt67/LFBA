# 自动锚点 × 多轮投票消融运行说明

只新增 `fixed_single` 和 `fixed_vote` 两组，固定锚点23470。正式种子为2004、592、3383，共六个新增训练分支，不补种子100。

`fixed_single` 使用第三轮cosine Top-P；`fixed_vote` 使用前三轮Top-P邻居票数，平均相似度处理同票。共同有效梯度、归一化和同分ID规则沿用旧选择器；候选排除锚点，选定后冻结。固定锚点失效时明确失败，不选择替代锚点。所有方法第4轮开始攻击，第40轮测试；P4500、Q2250、累计83250次替换，batch128、Adam、lr=.001，无衰减和早停。

正式实验只读复用 `results/core_ablation` 中三个种子的第三轮干净检查点，以及 `auto_rank_last`、`auto_vote`、`auto_mean` 的已完成结果。原数据、代码摘要、分支检查点、独立复评与审计记录均在预检时校验。旧自动组不重新训练，也不会复制为伪造的新训练分支。

新生成的核心消融默认在成功后清理模型；如准备继续运行本实验，须在运行来源实验时使用`--keep-checkpoints`，并用`--source-results`指定对应目录。来源模型已清理时预检会明确报错，完成证明不能代替训练起点。已有来源仍须通过原有协议和源码摘要校验。

## 手动执行

在PowerShell执行以下初始化，后续命令在同一个窗口运行：

```powershell
Set-Location 'D:\MyHomework\2026_2\LFBA\second\LFBA'
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
$ablationScript = 'D:\MyHomework\2026_2\LFBA\second\LFBA\scripts\run_anchor_vote_ablation.py'
```

1. 正式预检（读取旧结果和当前CIFAR数据，不训练、推理或初始化CUDA）：

```powershell
& $experimentPython $ablationScript --dry-run
```

应显示 `ready: true`、`tasks: 6`、三个种子、每轮2250、累计83250和空的 `errors`。预检会读取检查点及计算文件摘要，可能需要数十秒。它不创建输出目录。

2. 运行两组短训练验收：

```powershell
& $experimentPython $ablationScript --run --smoke
```

短训练独立生成前三轮干净状态，仅种子2004，源样本500、训练/验证450/50、测试100、共5轮。P45、Q22、累计44。完成时显示 `ALL_TASKS_COMPLETE`，短训练报告新增分支2/2，不包含旧正式结果。

3. 短训练验收通过后运行正式实验：

```powershell
& $experimentPython $ablationScript --run
```

GPU0按2004、592、3383顺序运行，每个种子先fixed_single再fixed_vote。干净状态按摘要复制到新目录，原目录保持只读。运行完毕应显示 `ALL_TASKS_COMPLETE`，新训练6/6，主矩阵12/12。

4. 中断后续训；已完成且校验通过的分支会跳过：

```powershell
& $experimentPython $ablationScript --run --resume
```

短训练续训须同时传入 `--smoke --run --resume`。不要在同一输出目录启动多个进程。旧结果、源码或协议变化后会拒绝续训，不修改已有协议来绕过检查。

5. 只重建正式报告，不训练或推理：

```powershell
& $experimentPython $ablationScript --report-only
```

短训练报告重建使用 `--report-only --smoke`。也可用 `--output` 明确指定已有批次，报告以保存的协议为准。

输入和输出可用 `--source-results`、`--data-dir`、`--output` 指定；GPU用 `--device` 指定，-1为CPU。自定义参数必须在续训时重复。默认目录为 `results/anchor_vote_ablation_v1`，短训练目录为 `results/anchor_vote_ablation_v1_smoke`，均与旧结果分离。

## 输出与解释

根目录保存冻结协议、来源路径与文件摘要、`summary.json`、`results.md`、`all_epoch_metrics.csv`、`validation_curves.png` 和 `failures.json`。每个新增分支长期保留配置、日志、完整攻击计划、审计历史和独立复评指标。

训练期间保存临时检查点用于恢复。整个批次的独立复评、结果审计、报告及曲线全部成功后，每种子生成并回读验证`completion_proof.json`，默认删除本批次模型及空的`.runtime`/`runtime`目录。中断、评测失败、记录写入或绘图失败时保留恢复文件；未知文件和来源实验不受清理影响。

正式实验同时在每种子目录的`references.json`归档已核验的引用指标、配置、历史与来源证明。成功后的`--report-only`与已完成批次的`--run --resume`只依赖本地持久记录，即使来源目录或原始数据不可用也能重建报告、跳过训练。缺失证明或持久记录损坏时不会误判完成；`--report-only`补全报告成功后同样执行清理。

需要保留本批次模型用于后续用途时，首次运行显式传入`--keep-checkpoints`；续训须重复该选项。保存的保留策略同样适用于报告重建。默认清理后的模型无法用于新的独立模型评测或依赖训练。本次代码修改不会批量清理已有目录。

主矩阵对应F-S=fixed_single、F-V=fixed_vote、A-S=auto_rank_last、A-V=auto_vote；auto_mean只作补充参照。报告四项差值A-S−F-S、A-V−F-V、F-V−F-S、A-V−A-S，以及交互作用 `(A-V−A-S)−(F-V−F-S)`，展示逐种子值、均值及样本标准差。

锚点策略比较允许目标类别变化，不能解释为固定同类内部的锚点提升。同锚点的投票比较保持相同目标。固定第40轮终点，保留负向、持平和饱和结果；缺失不补零，未达到收益门槛不等于证明等效。失败分支保留状态并继续其它分支，批次不完整时返回非零。
