# 01 · main.py 逐行注释

> 文件：`main.py`（261 行）
> 一句话：把「数据 → 模型 → 触发器 → Trainer」装配起来，本身**不包含**任何模型训练逻辑。

## 函数地图

| 位置 | 作用 |
|---|---|
| `main(args)` [L20-L215](../main.py#L20-L215) | 全流程装配 |
| `if __name__ == '__main__'` [L218-L261](../main.py#L218-L261) | 参数定义与启动 |

---

## 1. 导入与日志（L1-L36）

```python
import argparse, copy, logging, math
from datetime import datetime

import torch.nn as nn
import torchvision.transforms as transforms
from dataset.dataset import CIFAR10_VFL, UCIHAR_VFL, PHISHING_VFL, NUSWIDE_VFL   # L9
from models.CIFAR10_models import GlobalModelForCifar10, LocalModelForCifar10      # L10
...
from utils.trainer import Trainer                                                  # L14
from attack.attack import attack_lra, attack_rsa                                  # L15
# from utils.trigger_visualization import trigger_visualization                    # L16 你已注释
from utils.utils import *                                                          # L17 星号导入，带来 torch/np/os
```

- [L16](../main.py#L16)：原仓库引用了不存在的 `utils.trigger_visualization`，注释掉后启动才不报 ImportError。
- [L17](../main.py#L17)：`from utils.utils import *` 是隐式依赖来源——`os`、`torch`、`np` 都靠它带进来，所以别在 `utils/utils.py` 里随便删导入。

```python
def main(args):
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")  # L21 有卡用卡，无卡回退 CPU
    logger = logging.getLogger(__name__)                                                       # L24
    os.makedirs(args.results_dir, exist_ok=True)                                               # L29 你改成了递归创建
    fh = logging.FileHandler(os.path.join(args.results_dir, 'experiment.log'))                 # L30
    logger.addHandler(fh)                                                                      # L33
    logger.info(args)                                                                          # L36 日志第一行是全部超参，复现时先看它
```

---

## 2. 数据集构造（L38-L78）

```python
if args.dataset == 'CIFAR10':
    transform_train = transforms.Compose([...])        # L42-L47 随机裁剪+翻转+归一化
    transform_test  = transforms.Compose([...])        # L49-L52 只做 ToTensor+归一化
    train_data = CIFAR10_VFL(root=args.data_dir, train=True,  download=True, transform=transform_train)  # L54
    test_data  = CIFAR10_VFL(root=args.data_dir, train=False, download=True, transform=transform_test)   # L55
    test_data_asr = copy.deepcopy(test_data)           # L56 ASR 专用副本，后续只改它的 .data
```

逐行要点：

- [L54-L55](../main.py#L54-L55)：CIFAR10 是唯一会**自动下载**的数据集，所以最适合先跑通。
- [L56](../main.py#L56)：`deepcopy` 的时机很关键——`test_data_asr` 在**任何毒化之前**复制，因此它保留干净标签，只把特征换成带触发器的版本。
- [L69-L72](../main.py#L69-L72)：NUS-WIDE 只选 5 个类 `['buildings','grass','animal','water','person']`，拼接图像特征（634 维）与文本标签特征（1000 维）。
- [L76-L78](../main.py#L76-L78)：`trigger_visualization` 分支已注释，相应参数 `--trigger_visualization` 仍存在但无效。

**坑 1**：`--data_dir` 默认 `/data/data_raw/`，而 NUSWIDE/PHISHING 用 `root + 'NUS_WIDE'`、`root + 'Phishing/CM1.csv'` 直拼字符串，路径**必须以 `/` 结尾**。

---

## 3. 模型装配（L80-L124）

```python
model_list = []            # L82 下标 0 = 全局模型，1..client_num = 各客户端
extractor_list = []        # L83 同一结构的第二份副本，用于加载"特征提取器"权重
model_list.append(GlobalModelForCifar10(args))          # L84
extractor_list.append(GlobalModelForCifar10(args))      # L85
for i in range(args.client_num):                        # L86
    model_list.append(LocalModelForCifar10(args))       # L87
    ...
optimizer_list = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in model_list]  # L89 每个子模型独立优化器
criterion = nn.CrossEntropyLoss().to(device)            # L90
```

要点：

- `model_list[0]` 是 Top 模型，`model_list[i+1]` 是第 i 个客户端。[L123](../main.py#L123) 统一 `.to(device)`。
- 每个子模型**独立优化器**，但 [L175-L178](../utils/trainer.py#L175-L178) 是对整条计算图做一次 `loss.backward()`，所以梯度天然对齐。
- `extractor_list` 只是 checkpoint 容器，训练时不参与计算（见下方「坑 3」）。

---

## 4. 断点续训（L125-L162）

```python
if args.test_checkpoint:                      # L126 纯评测：只加载，不训练
    model_list[i].load_state_dict(...)        # L131
    optimizer_list[i].load_state_dict(...)    # L132

checkpoint = None                             # L138 正常训练路径的变量
if args.pretrained_checkpoint:                # L139 续训：加载后覆盖 start_epoch
    checkpoint = torch.load(...)              # L142
    args.start_epoch = checkpoint['epoch']    # L143
```

- [L131](../main.py#L131)：`state_dict[i]` 依赖**保存时的模型顺序**，改动 `client_num` 后旧 checkpoint 不能复用。
- [L143](../main.py#L143)：续训时 epoch 数会覆盖命令行 `--start_epoch`，注意别和 `--epoch` 冲突。

---

## 5. 触发器维度（L164-L178）

```python
if args.dataset == 'CIFAR10':
    trigger_dimensions = []                      # L165 图像用空间 patch，不需要维度列表
elif args.dataset == 'UCIHAR':
    ranges = range(math.ceil(train_data.data.shape[1] / args.client_num), train_data.data.shape[1])  # L168-L169 → 281..560
    trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)             # L170
elif args.dataset == 'PHISHING':
    ranges = range(15, 30)                       # L172 第二个客户端的 15 维
elif args.dataset == 'NUSWIDE':
    ranges = range(634, 1634)                    # L175 文本特征 1000 维
```

**核心规则：触发器维度一定落在攻击者客户端所持有的特征段内。** 这也是为什么 `--attack_client_num` 默认是 1（第二个客户端）。

- `np.random.choice(..., replace=False)` 保证维度不重复。
- 维度在**一次运行内固定**（同一个 `trigger_dimensions` 用于训练和测试），换个 seed 就换一组维度。

---

## 6. 训练前注入（L180-L194）

```python
if args.attack is None:                        # L180
    test_data_asr.data = attack_rsa(..., rate=1, 'test')        # L181 测试集全量加触发器
elif args.attack == 'rsa':                     # L182
    train_data.data = attack_rsa(..., args.poison_rate, 'train')# L183 随机毒化训练集
    test_data_asr.data = attack_rsa(..., 1, 'test')             # L184
elif args.attack == 'lra':                     # L185
    train_data.data, train_data.targets = attack_lra(...)       # L186-L188 随机毒化 + 改标签
elif args.attack == 'LFBA':                    # L191
    test_data_asr.data = attack_rsa(..., 1, 'test')             # L192 训练集不在这里毒化！
```

**这是全项目最重要的分支。** `LFBA` 分支只处理测试集，因为 LFBA 的训练集毒化必须**每轮动态做**，发生在 `Trainer.train()` 内部（见 04 篇）。

- `rate=1` 表示测试集 100% 加触发器，用于算 ASR。
- `x_p`（干净副本）在整个训练里没被使用，训练只用 `x_n`。

---

## 7. 组装 Trainer（L196-L215）

```python
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)   # L196
test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=True)   # L197
test_asr_loader = DataLoader(test_data_asr, batch_size=args.batch_size, shuffle=True)  # L198

trainer = Trainer(device, model_list, extractor_list, extractor_list[args.attack_client_num + 1],
                  optimizer_list, criterion, train_loader, test_loader, test_asr_loader,
                  trigger_dimensions, logger, args, checkpoint)                    # L199-L202
trainer.train()                                                                    # L215
```

- [L199](../main.py#L199)：第 3、4 个实参是 `extractor_list` 和 `extractor`。`extractor` 在 `Trainer` 里被保存为 `self.extractor`，**从未使用**。
- DataLoader 的顺序：`shuffle=True` 意味着每轮 `index` 顺序不同，所以 `grad_vec_epoch` 必须靠 `indexes_epoch` 才能映射回原始样本下标（见 04 篇）。

**坑 2**：`--test_checkpoint` 分支引用 `args.do_tsne` 与 `trainer.perform_tsne()`（[L203-L211](../main.py#L203-L211)），二者都不存在。只用正常训练/评测时不会触发；一旦传 `--test_checkpoint` 且 `--do_tsne`（参数本身也未定义）就会崩。

---

## 8. argparse（L218-L261）

按作用分组记忆：

| 组 | 参数 |
|---|---|
| 设备与数据 | `--device --data_dir --dataset --seed` |
| 训练 | `--epoch --batch_size --lr --start_epoch --early_stop --pretrain_stage` |
| 攻击 | `--attack --target_label --poison_rate --poison_dimensions --trigger_feature_clip --attack_client_num --anchor_idx` |
| 切换策略 | `--select_rate --select_replace --random_select --poison_all` |
| 权重 | `--pretrained_checkpoint --test_checkpoint --feature_extractor` |

你新增的 [L236](../main.py#L236) `--pretrain_stage` 只为修复 `Trainer` 中未定义属性，默认 0 表示第一轮就开始正常训练（没有单独的预训练阶段）。

[L256](../main.py#L256) 把结果目录改到 `results/<dataset>/<时间戳>`，与你在日志中看到的结构一致。

---

## 自测题

1. 为什么 `attack=None` 时测试集也会被加触发器？这会不会影响 ACC？
2. `extractor_list` 和 `model_list` 结构相同，为什么需要两份？
3. LFBA 分支为什么不在 `main.py` 里毒化训练集？
4. 若要攻击**第一个**客户端，需要同时改哪两个参数/常量？

<details>
<summary>参考答案</summary>

1. 计算 ASR 必须让测试样本带触发器，所以即使不攻击也统一加；ACC 在 `test_loader` 上算，不受影响。
2. 一份用于训练，一份用于加载外部特征提取器权重做对比实验，当前训练流程未使用后者。
3. 因为 LFBA 的毒化集与切换比例依赖每一轮的嵌入梯度，只能放在训练循环里动态生成。
4. `--attack_client_num 0`；同时确认 `trigger_dimensions` 落在客户端 0 的特征段内（当前代码按攻击者=第二个客户端硬编码了 ranges）。
</details>
