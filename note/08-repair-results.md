# LFBA 修复与验收记录

本轮按照用户批准的计划逐项修复，使用既有 `D:\MyHomework\AIAIAIAI\environment`，未升级依赖。分支：`codex/lfba-repair-validation`。原有改动补丁及源码备份保存在 `note/repair/preexisting.patch`、`note/repair/source-before.zip`，历史 results 未修改。

## 修复及证据

| 问题 | 修复 | 回归证据 |
|---|---|---|
| 训练模式 | 每轮 train；评测 eval + no_grad | 两轮 BatchNorm 计数持续增加，评测无梯度 |
| RSA 副作用 | 对拷贝注入 | NumPy 与 Tensor 输入不变 |
| CIFAR 边界 | 延迟攻击计划：共享增强后，只替换右半图，再注入右下角 patch | 强制翻转、边界裁剪、仅触发器与标签替换检查 |
| PHISHING 泄漏 | 固定且共享 train/val/test，训练专属 scaler | 集合互斥；改变测试特征不改变 scaler；锚点预留在拟合前完成 |
| NUS-WIDE 名称 | 统一规范名与 T 别名，拒绝 I 变体 | 真实合成文件读取、634/1000 切分与模型反向 |
| 相似度 | 默认 legacy_l1，可选 cosine，过滤无效值、按原 ID 稳定同分 | 两种排序、零向量、NaN 和有效候选不足检查 |
| 参数 | 生效的预训练轮数、自定义目录、冲突与能力检查 | 参数、延迟攻击和无效配置专项 |
| 验证集 | 独立分层验证，原 ID 不变，诚实测试集不参与选模 | 禁止访问测试加载器时训练仍通过 |
| checkpoint | v2 完整 last/best 与随机状态、梯度、选样、donor、划分、scaler、配置 | CPU 连续 / 中断续训参数逐位一致；最佳记录不重置 |
| 独立评测 | 删除未实现 t-SNE 路径，配置自动恢复，旧格式要求显式元数据 | 新格式 train/eval/resume；缺文件、配置漂移与旧格式续训拒绝 |
| GPU/Windows | 先初始化 CUDA；checkpoint 原子替换遇临时文件锁时限次重试 | 真实 GPU 运行及文件锁专项 |

所有指标带样本数；空分母返回 null 和原因；不参与攻击选模。新增原子 last checkpoint，训练中断可从最近完成的轮次恢复。若续训新目录没有更优模型，保留先前最佳权重为 evaluation-only，严格续训使用 last。

## 已完成验证

- **24 项单元与集成测试全部通过**，日志：`note/repair/unittest.log`。另通过 34 个 Python 文件语法检查和 `git diff --check`。
- 只读复核补充的三项边界问题也已覆盖：独立评测末轮模型时使用同轮验证指标，并区分 checkpoint_epoch/best_epoch；PHISHING 明确拒绝 eval_limit；NUS-WIDE 按特征文件名排序，避免目录枚举改变列顺序。
- 六条真实 CIFAR GPU 短训练、独立评测与 GPU 续训：`results/repair_validation/smoke_gpu_retry/summary.json`。
- GPU 连续训练 / 中断续训模型及 Adam 状态容差验收通过（rtol=1e-5，atol=1e-6），本次模型最大绝对差为 **0**，毒化集合和 donor 映射相同。结果：`results/repair_validation/gpu_resume_comparison/comparison.json`。这是一组本机短训练证据，不承诺跨硬件逐位一致。
- 短训练只使用 200 条训练候选和 100 条测试样本，用于执行路径检查，不能解释为攻击性能结果。
- 旧 LFBA 模型完整评测：ACC **72.19%**，ASR **96.1778%**，触发样本准确率 **13.07%**，与原保存指标一致。结果：`results/repair_validation/legacy_eval/metrics.json`。
- PHISHING / UCIHAR / NUS-WIDE 尚无本地真实数据，本次通过与实际读取函数兼容的合成文件验证，未声称完成其真实数据实验。

## 完整实验

两组 seed=100、各 20 轮实验由 `scripts/acceptance_full.py` 执行；结果写入 `results/repair_validation/full_seed100`，控制台日志写入 `note/repair/full-console.log`。

两组均已完成 **20 轮**，训练划分一致，均使用 seed=100、batch=128、Adam、lr=.001；LFBA 为 legacy_l1、poison_rate=.1、select_rate=.5、anchor=23470、目标类 6，每轮实际替换 2,250 个样本。两组关闭早停。原始测试集保持 10,000 条；ASR 分母为 9,000 个非目标类样本。

| 方法 | 验证集选中的轮次 | 独立测试 ACC | 独立测试 ASR | 触发样本准确率 | 总耗时 |
|---|---:|---:|---:|---:|---:|
| 干净基线 | 19 | 77.31% | 2.88% | 76.67% | 928.7 秒 |
| LFBA | 16 | 74.83% | 96.38% | 11.61% | 984.5 秒 |

结果表：`results/repair_validation/full_seed100/results.md`；完整参数和指标：同目录 `summary.json`。各方法目录保存 config.json、environment.json、history.json、metrics.json、experiment.log 以及 last/best_checkpoint.pth.tar。模型由验证集选择，测试结果没有用于调参或选轮。

运行启动时的源码快照为 `source_snapshot.zip`，全部 18 个文件已逐一匹配 `source_hashes.json`。启动后的三项边界修复不改变本次 CIFAR 训练路径和选模结果。旧模型结果只作为兼容性证据，不并入新对照。本轮仅一个 seed，不声称统计稳定性，也没有新增防御实验。

## 运行与接口说明

当前只支持两客户端、攻击方 1。默认就是选择性替换，`--select_replace` 为弃用兼容别名；只加触发器使用 `--poison_all`。`pretrain_stage=0` 或 1 均先完成一轮干净训练；更大值表示干净轮数。`--early_stop 0` 关闭早停。

`--train_limit` 和 `--eval_limit` 仅供显式有界验收，正式对照不传；PHISHING 不支持 train_limit。标准余弦用 `--similarity cosine`，默认仍为 legacy_l1。

评测返回统一字典：acc、asr、poison_acc、target_acc、loss、counts、undefined。数据读取保持四元组，最后一项始终是原始 ID；官方测试源的 ID 使用训练源长度作偏移，避免与训练源混淆。CIFAR 的 transform 只负责确定性的 ToTensor/Normalize，随机几何增强在内部共享且只执行一次。

旧 checkpoint 只可评测，需明确 dataset 和 target_label；表格数据还需 trigger_dimensions。新 checkpoint 自动恢复这些参数。严格续训只允许改变 device、results_dir、epoch，冲突参数报错。结果目录已有实验文件时拒绝覆盖。

```powershell
# 回归测试
& 'D:\MyHomework\AIAIAIAI\environment\python.exe' -m unittest discover -v

# 独立评测新模型：指定一个新的输出目录
& 'D:\MyHomework\AIAIAIAI\environment\python.exe' main.py --test_checkpoint results/repair_validation/full_seed100/lfba/best_checkpoint.pth.tar --results_dir results/repair_validation/new_eval

# 精确续训新模型
& 'D:\MyHomework\AIAIAIAI\environment\python.exe' main.py --pretrained_checkpoint results/repair_validation/full_seed100/lfba/last_checkpoint.pth.tar --epoch 25 --results_dir results/repair_validation/resume_to25
```
