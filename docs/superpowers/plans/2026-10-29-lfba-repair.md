# LFBA 逐项修复与实验验收计划（已批准）

日期：2026-10-29。依据：用户批准的逐项修复计划与 `note/07-code-review.md`。

执行约束：使用 `D:\MyHomework\AIAIAIAI\environment\python.exe`；不升级依赖；保留已有改动、数据和历史结果；按顺序进行失败复现、根因修复、专项检查与累计回归；使用 unittest。连续三次失败需回到根因和设计检查。默认使用 legacy_l1，cosine 只作可选对照。

## 顺序任务

- [x] 每轮恢复 train 模式，评测使用 eval 与 no_grad。
- [x] RSA 操作深拷贝，输入保持不变。
- [x] CIFAR 先共享增强、再在攻击方右半图替换和注入，最后转 Tensor 并归一化。
- [x] PHISHING 共享固定分层划分，只在最终训练子集拟合 scaler。
- [x] 统一 NUSWIDE / NUSWIDET，明确拒绝 NUSWIDEI。
- [x] 保留 L1 默认，新增 cosine，排除无效梯度并稳定处理同分。
- [x] 统一 select_replace、poison_all、pretrain_stage、results_dir 和两方能力边界。
- [x] 独立分层验证集、保留原始样本 ID、锚点留在训练集；测试集不参与选模。
- [x] 完整且版本化的 last / best checkpoint，恢复模型、优化器、最佳记录、攻击、划分和随机状态。
- [x] 普通独立评测入口与旧 checkpoint 显式元数据兼容；拒绝旧格式严格续训。

## 验收与交付

- [x] CPU 专项与累计回归，包括连续训练 / 中断续训严格等价。
- [x] 四数据集接口检查：CIFAR 本地真实数据；表格数据集通过真实文件接口的合成文件。
- [x] CIFAR GPU 短训练覆盖 baseline、RSA、LRA、LFBA、poison_all、random_select。
- [x] 新格式 GPU 独立评测和续训。
- [x] 历史真实 CIFAR checkpoint 完整评测兼容性。
- [x] seed=100，两组各 20 轮的 CIFAR 干净基线与 LFBA 完整实验。
- [x] 汇总独立测试集结果、命令、环境、耗时与模型，更新修复说明。

最终验收：24 项 unittest 通过；真实 GPU 连续/中断训练容差比较通过，本次模型最大绝对差为 0；两组完整对照通过。结果详见 `note/08-repair-results.md`。

完整实验固定：两方、攻击方 1、batch=128、Adam、lr=.001、poison_rate=.1、select_rate=.5、anchor=23470、legacy_l1；不早停；两组共享划分。干净基线按验证 ACC 选模，LFBA 按验证 (ACC+ASR)/2 选模，最终只评测选中的模型。

排除范围：多种子完整实验、cosine 完整对照、多客户端扩展及 CV-Guard。不以恢复旧 ASR 数值作为修复成功条件，不合并新旧结果。
