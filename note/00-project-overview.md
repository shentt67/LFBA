# LFBA 项目基本情况

> 本文记录修复前的项目情况与行号。修复后的行为、参数和验收结果以 [08-repair-results.md](./08-repair-results.md) 为准。

整理日期：2026-10-29。范围：本地源码、笔记、数据、日志与模型文件；本次只整理文档并做轻量检查，没有重新训练，也没有核验外部论文或防御方法的新颖性。

## 1. 项目定位

README 将本项目标为 AAAI-25 论文《Label-Free Backdoor Attacks in Vertical Federated Learning》的 PyTorch 实现。研究场景是纵向联邦学习：各参与方持有相同样本的不同特征，本地模型输出嵌入，再交给全局模型分类。[VERIFY: README.md:1] [VERIFY: dataset/utils.py:48] [VERIFY: models/CIFAR10_models.py:14]

当前实现是单进程中模拟多方计算，没有在训练路径中实现真实网络通信。入口在 `main.py`，统一创建各方模型和优化器，`Trainer` 直接调用它们并反向传播。[VERIFY: main.py:87] [VERIFY: utils/trainer.py:150] [VERIFY: utils/trainer.py:171]

代码包含干净训练、RSA、LRA 和 LFBA 四种路径。RSA 随机选样加触发器；LRA 还会在训练时替换标签；LFBA 通过梯度选样，默认执行选择性特征替换。[VERIFY: main.py:183] [VERIFY: attack/attack.py:13] [VERIFY: attack/attack.py:30] [VERIFY: attack/attack.py:39]

## 2. 目录与职责

| 位置 | 职责 | 依据 |
|---|---|---|
| `main.py` | 参数、数据增强、数据与模型装配、攻击分流、启动训练 | [VERIFY: main.py:20] |
| `dataset/dataset.py` | 四类数据集，返回当前样本、干净快照、标签、样本下标 | [VERIFY: dataset/dataset.py:18] [VERIFY: dataset/dataset.py:64] |
| `dataset/utils.py` | NUS-WIDE 数据读取和纵向特征切分 | [VERIFY: dataset/utils.py:9] [VERIFY: dataset/utils.py:48] |
| `models/` | 各数据集对应的本地模型与全局分类器 | [VERIFY: models/CIFAR10_models.py:6] [VERIFY: models/UCIHAR_models.py:6] [VERIFY: models/PHISHING_models.py:6] [VERIFY: models/NUSWIDE_models.py:5] |
| `attack/attack.py` | 攻击入口、锚点候选辅助函数、梯度相似度选样 | [VERIFY: attack/attack.py:13] [VERIFY: attack/attack.py:80] |
| `attack/add_trigger.py` | 图像 patch 或表格特征触发器 | [VERIFY: attack/add_trigger.py:8] |
| `attack/add_trigger_replace.py` | 跨样本替换攻击方特征并注入触发器 | [VERIFY: attack/add_trigger_replace.py:6] |
| `utils/trainer.py` | 梯度收集、毒化样本选择、训练、评测和保存模型 | [VERIFY: utils/trainer.py:35] [VERIFY: utils/trainer.py:257] |
| `utils/utils.py` | 随机种子与异常处理 | [VERIFY: utils/utils.py:20] |
| `note/` | 代码解读、实验规划、防御研究设想 | [VERIFY: note/README.md:5] [VERIFY: note/06-defense-research-proposal.md:1] |
| `results/` | 按数据集与时间保存实验日志及最佳模型 | [VERIFY: main.py:259] [VERIFY: utils/trainer.py:236] |

## 3. 数据和模型

| 数据集 | 两方输入划分 | 本地模型 / 全局输出 | 依据 |
|---|---|---|---|
| CIFAR-10 | 图像左、右各 16 列 | 两个 ResNet18，各输出 128 维；拼接后输出 10 类 | [VERIFY: dataset/utils.py:49] [VERIFY: models/CIFAR10_models.py:6] [VERIFY: models/CIFAR10_models.py:25] |
| UCIHAR | 561 维分为 281 / 280 | MLP，各输出 16 维；输出 6 类 | [VERIFY: dataset/utils.py:54] [VERIFY: models/UCIHAR_models.py:6] [VERIFY: models/UCIHAR_models.py:25] |
| PHISHING | 30 维分为 15 / 15 | MLP，各输出 4 维；输出 2 类 | [VERIFY: dataset/utils.py:59] [VERIFY: models/PHISHING_models.py:6] [VERIFY: models/PHISHING_models.py:23] |
| NUS-WIDE | 634 维图像特征 / 1000 维文本标签特征 | MLP，输出 40 / 60 维；输出 5 类；入口与切分命名有冲突，见后文 | [VERIFY: dataset/utils.py:64] [VERIFY: models/NUSWIDE_models.py:5] [VERIFY: models/NUSWIDE_models.py:24] |

本次实际检查：`data/` 下已有 CIFAR-10，加载训练集 50,000 条、测试集 10,000 条；原始数组形状为 `(50000, 32, 32, 3)`。没有发现其余三类数据集文件。

## 4. LFBA 核心流程

1. 第一个训练轮次（`ep=0`）先训练并收集攻击方嵌入对损失的梯度、样本下标和标签。[VERIFY: utils/trainer.py:180]
2. 第二个训练轮次开始（`ep=1`），用 `anchor_idx` 找到锚点，按梯度相似度选出 `poison_rate × N` 个样本，毒化候选集合之后保持固定。相似度实现为点积除以双方 **L1 范数乘积**，不能直接称为标准余弦相似度。[VERIFY: utils/trainer.py:62] [VERIFY: utils/trainer.py:69] [VERIFY: attack/attack.py:80]
3. 每轮在候选集合内按上一轮梯度 L2 范数选取困难样本，数量为 `int(num_poisons × select_rate)`；从候选集合之外抽取 donor 样本。[VERIFY: utils/trainer.py:65] [VERIFY: utils/trainer.py:86] [VERIFY: utils/trainer.py:94]
4. 默认把 donor 的攻击方特征加上触发器，再复制给选中的目标样本，保留接收样本的标签；每轮从 `data_p` 干净快照重建数据。`--poison_all` 改为只加触发器；`--random_select` 可改为随机选择子集。[VERIFY: utils/trainer.py:100] [VERIFY: utils/trainer.py:124] [VERIFY: attack/add_trigger_replace.py:39]
5. 每轮评估干净准确率 ACC 和攻击成功率 ASR，按 `(ACC + ASR) / 2` 保存最佳模型。ASR 只统计真实标签不是目标类的触发样本。[VERIFY: utils/trainer.py:213] [VERIFY: utils/trainer.py:224] [VERIFY: utils/trainer.py:326]

默认 `client_num=2`，攻击方下标为 1。虽然有客户端数量参数，但切分函数只返回两方数据、全局模型只拼接两个输入，目前应按两方实验使用。[VERIFY: main.py:233] [VERIFY: main.py:247] [VERIFY: dataset/utils.py:48] [VERIFY: models/CIFAR10_models.py:14]

## 5. 已有实验与进度

下面是历史运行结果，本次同时读取了两个 checkpoint 的标量元数据，与日志一致；没有重新计算全测试集指标。

| 运行目录（`results/CIFAR10/`） | 方法 | 总训练轮数 | 保存轮次 | ACC | ASR | 依据 |
|---|---|---:|---:|---:|---:|---|
| `20261023_174949` | 无攻击训练 | 5 | 5 | 63.77% | 6.37% | [VERIFY: results/CIFAR10/20261023_174949/experiment.log:1] [VERIFY: results/CIFAR10/20261023_174949/experiment.log:422] |
| `20261023_181047` | LFBA | 20 | 16 | 72.19% | 96.18% | [VERIFY: results/CIFAR10/20261023_181047/experiment.log:1] [VERIFY: results/CIFAR10/20261023_181047/experiment.log:763] |

LFBA 的历史配置：seed=100、batch_size=128、lr=0.001、anchor_idx=23470、poison_rate=0.1、select_rate=0.5，训练中目标标签为 6。实际替换规模约为全训练集的 5%，即 2,500 个样本。[VERIFY: results/CIFAR10/20261023_181047/experiment.log:1] [VERIFY: results/CIFAR10/20261023_181047/experiment.log:53] [VERIFY: utils/trainer.py:65]

第 20 轮本身的 ACC=74.49%、ASR=77.74%，与第 16 轮保存的最佳结果不同，引用结果时应注明轮次。两组历史实验轮数和 batch size 不同，不能据此判断 LFBA 提升了干净准确率。[VERIFY: results/CIFAR10/20261023_181047/experiment.log:952] [VERIFY: results/CIFAR10/20261023_174949/experiment.log:1]

`note/06-defense-research-proposal.md` 提出了 CV-Guard 检测、阻断与净化方案。本次目录检查没有发现对应防御实现或结果，应将其视为待验证研究方案。[VERIFY: note/06-defense-research-proposal.md:1] [VERIFY: note/06-defense-research-proposal.md:10]

## 6. 指定运行环境与本次检查

后续运行固定使用：`D:\MyHomework\AIAIAIAI\environment\python.exe`。

| 项目 | 本次实测值 |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| CUDA（PyTorch 构建版本） | 12.6 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU；CUDA 可用 |
| NumPy / pandas / scikit-learn | 1.26.4 / 2.3.1 / 1.7.1 |

已通过的轻量检查：

- 15 个 Python 源文件语法解析。
- `main.py --help`，主要模块可导入。
- CIFAR-10 训练集和测试集完整性检查与加载，未下载数据。
- 两个本地模型和全局模型的 GPU 前向：输入拆为两个 `(1, 3, 32, 16)`，输出为 `(1, 10)`。这次使用随机初始化权重，只验证接口和执行路径。
- 两个历史 checkpoint 用 `weights_only=True` 读取，核对保存轮次、ACC、ASR。

本次没有启动训练、执行反向传播或完整 checkpoint 评测；以上检查不代表整个项目没有缺陷。模型构造出现 torchvision `pretrained` 参数弃用提示，前向执行成功。

## 7. 当前需注意的问题

以下代码问题本次仅记录，未修改。

| 问题 | 影响 / 说明 | 依据 |
|---|---|---|
| 评测后没有恢复训练模式 | `train()` 仅在 epoch 循环前调用一次，而每次 `test()` 都调用 `eval()`；下一轮仍会更新参数，但 ResNet 的 BatchNorm 等保持评测行为，需优先核验和修正 | [VERIFY: utils/trainer.py:45] [VERIFY: utils/trainer.py:58] [VERIFY: utils/trainer.py:213] [VERIFY: utils/trainer.py:260] |
| 单独 checkpoint 评测入口不完整 | `args.do_tsne` 没有参数定义，`Trainer` 也没有 `perform_tsne` 方法 | [VERIFY: main.py:207] [VERIFY: main.py:213] [VERIFY: main.py:225] |
| NUS-WIDE 名称不一致 | 入口识别 `NUSWIDE`，切分函数仅识别 `NUSWIDET/NUSWIDEI`，会导致当前入口无法直接完成该数据集训练 | [VERIFY: main.py:73] [VERIFY: dataset/utils.py:64] |
| PHISHING 数据拆分与标准化 | 每次构造训练/测试对象都会重新抽样，存在训练测试重叠风险；标准化也在拆分前对全数据拟合 | [VERIFY: dataset/dataset.py:127] |
| LFBA 使用真实标签 | 代码用标签计算候选集一致率，并从锚点真实标签设置评测目标；样本相似度排序本身不接收标签，研究时要明确这两种用途 | [VERIFY: utils/trainer.py:78] [VERIFY: utils/trainer.py:129] |
| `--select_replace` 没有行为作用 | 仅声明参数，实际是否切换由 `--poison_all` 决定 | [VERIFY: main.py:252] [VERIFY: attack/attack.py:43] |
| 命令行默认值不适合直接本地启动 | 默认数据集是 NUSWIDE，默认数据目录是 Linux 风格路径；传入的 `results_dir` 又会被入口覆盖 | [VERIFY: main.py:225] [VERIFY: main.py:259] |

历史笔记校正：`note/README.md`、`note/05-paper-map-and-next-steps.md` 中“LFBA 尚未执行”已被上面的 20 轮日志推翻。旧笔记中的 `--attack None` 会传入字符串而非 Python `None`，干净训练应完全省略 `--attack`。[VERIFY: note/README.md:32] [VERIFY: note/05-paper-map-and-next-steps.md:69] [VERIFY: main.py:183] [VERIFY: main.py:240]

旧笔记所列 CIFAR 切片轴疑点不能成立：替换时原始数组为 NHWC，`new_data[target, :, 16:, :]` 改的是右半宽度；经过 `ToTensor` 后为 NCHW，`data[:, :, :, 16:]` 同样取右半宽度。两者的轴语义一致。随机裁剪和翻转对触发器位置的影响是另一个需要独立实验的问题。[VERIFY: dataset/dataset.py:60] [VERIFY: attack/add_trigger_replace.py:51] [VERIFY: dataset/utils.py:52] [VERIFY: main.py:46]

## 8. 运行命令备忘

在项目根目录的 PowerShell 中运行，显式指定解释器即可，无需修改系统默认 Python。下列训练命令仅作为备忘，本次未执行。

```powershell
# 查看参数（本次已执行）
& 'D:\MyHomework\AIAIAIAI\environment\python.exe' main.py --help

# 同配置干净基线：省略 --attack
& 'D:\MyHomework\AIAIAIAI\environment\python.exe' main.py --data_dir ./data/ --dataset CIFAR10 --device 0 --epoch 20 --batch_size 128 --lr 0.001 --seed 100

# 对齐历史 LFBA 配置
& 'D:\MyHomework\AIAIAIAI\environment\python.exe' main.py --data_dir ./data/ --dataset CIFAR10 --device 0 --epoch 20 --batch_size 128 --lr 0.001 --seed 100 --attack LFBA --anchor_idx 23470 --poison_rate 0.1 --select_rate 0.5
```

后续工作顺序建议：先解决训练模式及评测入口问题，再补相同配置的干净基线与多随机种子实验，最后验证防御方案。现有结果可用于理解流程，尚不足以证明防御有效或获得稳定复现结论。
