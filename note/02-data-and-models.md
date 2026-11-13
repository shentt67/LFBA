# 02 · 数据与模型（dataset/ + models/）

> 一句话：数据集负责「同一批样本、按特征列切给两个客户端」，模型负责「本地编码器 → 拼接 → 全局分类器」。

## 一、数据集契约

所有 `Dataset.__getitem__` 都返回四元组：

```python
return x, x_p, y, index
```

| 变量 | 含义 | 训练里是否使用 |
|---|---|---|
| `x` | 当前特征（可能已被毒化） | 使用 |
| `x_p` | 初始化时的干净特征快照 | **不使用**（`trainer.py` L151 只取 `x_n`） |
| `y` | 标签 | 使用（LFBA 里还用于选 target label） |
| `index` | 原始样本下标 | 使用（把乱序 batch 映射回全局下标） |

`index` 是整套代码的“锚”：DataLoader 每轮乱序，只有靠它才能把梯度、毒化集、数据集行号对上。

---

## 二、`dataset/dataset.py` 逐行

### 1. `CIFAR10_VFL`（L18-L80）

```python
self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
self.data = self.data.transpose((0, 2, 3, 1))     # L59-L60 → (N,32,32,3) 即 HWC
self.data_p = copy.deepcopy(self.data)            # L64 干净快照
```

```python
def __getitem__(self, index):
    img, img_p, target = self.data[index], self.data_p[index], self.targets[index]  # L67
    img = Image.fromarray(img)                                                       # L69
    img_poisoned = Image.fromarray(img_p)                                            # L71
    if self.transform is not None:
        img = self.transform(img)                                                      # L74
        img_poisoned = self.transform(img_poisoned)                                    # L75
    return img, img_poisoned, target, index                                            # L80
```

要点：

- **触发器写在 uint8 原始数组上**（0/255），不是归一化后的张量；`ToTensor()+Normalize` 在 `__getitem__` 里才发生。
- `self.data` 是 HWC，但 `split_vfl` 的切分维是 `dim=2`（宽），不是高。这一点后面会引发一个 bug（见 03 篇）。

### 2. `UCIHAR_VFL`（L83-L101）

```python
self.data   = np.loadtxt(root + '/UCI-HAR/.../X_train.txt')      # L86 561 维
self.data_p = np.loadtxt(root + '/UCI-HAR/.../X_train.txt')      # L87 独立再读一遍
self.targets = np.loadtxt(.../y_train.txt') - 1                  # L88 标签 1..6 → 0..5
```

- 数据是 numpy `float64`，训练时 `x.to(device).float()` 才转 float32。
- `data_p` 与 `data` 内容相同但**不是同一对象**，后续只改 `data`。

### 3. `NUSWIDE_VFL`（L104-L124）

```python
X_image, X_text, Y = get_labeled_data(root + 'NUS_WIDE', selected_labels, None, 'Train')  # L107
self.data   = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)            # L108 → 1634 维
self.data_p = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)            # L109
self.targets = torch.tensor(Y)                                                            # L110
```

- 拼接顺序固定：**前 634 维图像 → 客户端 0，后 1000 维文本 → 客户端 1**。
- `root + 'NUS_WIDE'` 是字符串直拼，`--data_dir` 必须以 `/` 结尾。

### 4. `PHISHING_VFL`（L127-L159）—— 有一个真实缺陷

```python
train_indexes = sample(list(indexes_list), 8844)                    # L139
test_indexes  = list(set(list(indexes_list)).difference(set(train_indexes)))  # L140
...
if train: self.data = train_data                                     # L143
else:     self.data = test_data                                      # L148
```

**坑**：`train_indexes` 在 `__init__` 里**每次实例化都重新采样**。`main.py` 先构造 `train=True` 再构造 `train=False`，第二次采样得到的是另一组 8844 个下标，于是：

- 训练集和测试集**不是互补关系，会重叠**；
- `StandardScaler` 也在全量数据上各自重新 `fit`，存在轻微信息泄漏。

复现论文数值时建议固定切分（例如按索引做一次性划分），否则指标不可比。

---

## 三、`dataset/utils.py` 逐行

### 1. `get_labeled_data`（L9-L45）

```python
for label in selected_label:                        # L13 逐类读 Groundtruth/TrainTestLabels/Labels_<label>_<dtype>.txt
    ...
selected = data_labels[data_labels.sum(axis=1) == 1]  # L21 只保留"仅属于一个选中类"的样本
data_X_image_selected = data_XA.loc[selected.index]    # L34
data_X_text_selected  = tagsdf.loc[selected.index]     # L41
return ..., np.argmax(selected.values[:], 1)           # L43 标签 = 5 类中的 one-hot 位置
```

- 这是标准 NUS-WIDE 预处理：图像低级特征 + 1k 标签向量。
- `n_samples=None` 时返回全量；本仓库调用都传 `None`。

### 2. `split_vfl`（L48-L75）

```python
if args.dataset == 'CIFAR10':
    x_a = data[:, :, :, :16]; x_b = data[:, :, :, 16:]     # L51-L52 按宽切一半
elif args.dataset == 'UCIHAR':
    x_a = data[:, :281]; x_b = data[:, 281:]               # L56-L57
elif args.dataset == 'PHISHING':
    x_a = data[:, :15]; x_b = data[:, 15:]                 # L61-L62
elif args.dataset == 'NUSWIDET':
    x_a = data[:, :634]; x_b = data[:, 634:]               # L65-L66
elif args.dataset == 'NUSWIDEI':
    x_a = data[:, 634:]; x_b = data[:, :634]               # L69-L70
else:
    raise_dataset_exception()                               # L75
```

**坑**：分支名是 `NUSWIDET` / `NUSWIDEI`，而 `main.py --dataset` 传的是 `NUSWIDE`。若你直接跑 NUSWIDE，会在第一个 batch 抛 “Unknown dataset”。两条修复路径：把 `main.py` 的 dataset 改成 `NUSWIDET`（并同步所有 `args.dataset == 'NUSWIDE'` 判断），或在 `split_vfl` 里增加 `NUSWIDE` 分支。

---

## 四、`models/` 逐行

### 结构总表

| 数据集 | 客户端 0 | 客户端 1 | 全局模型 | 类别数 |
|---|---|---|---|---|
| CIFAR10 | ResNet18 → 128 | ResNet18 → 128 | 256→256→128→10 | 10 |
| UCIHAR | 281→140→70→35→16 | 280→140→70→35→16 | 32→32→16→6 | 6 |
| PHISHING | 15→8→4 | 15→8→4 | 8→4→2 | 2 |
| NUSWIDE | 634→320→160→80→40 | 1000→500→250→125→60 | 100→100→50→5 | 5 |

### 1. `CIFAR10_models.py`

```python
class GlobalModelForCifar10(nn.Module):
    def __init__(self, args):
        self.linear1 = nn.Linear(256, 256)     # L9 两个客户端各 128 维，拼成 256
        self.linear2 = nn.Linear(256, 128)     # L10
        self.classifier = nn.Linear(128, 10)   # L11

    def forward(self, input_list):
        tensor_t = torch.cat((input_list[0], input_list[1]), dim=1)  # L15 按特征维拼接
        x = self.linear1(tensor_t); x = self.linear2(x); x = self.classifier(x)  # L17-L20
        return x
```

```python
class LocalModelForCifar10(nn.Module):
    def __init__(self, args):
        self.backbone = models.resnet18(pretrained=False)     # L29
        num_ftrs = self.backbone.fc.in_features
        if self.args.client_num == 2:
            self.backbone.fc = nn.Linear(num_ftrs, 128)       # L32 只改最后一层
```

- 两个客户端用的是**同一个类、同样结构**，没有客户端 id 参数，因此两个本地模型初始化不同但结构对称。
- 全局模型**没有激活函数**，纯线性堆叠；这也意味着特征空间是线性的，理解梯度语义时更干净。

### 2. `UCIHAR_models.py`

```python
if client_number == 0:
    nn.Linear(math.ceil(561 / self.args.client_num), 140)   # L31 → 281
else:
    nn.Linear(round(561 / self.args.client_num), 140)       # L42 → 280
```

- `math.ceil(280.5)=281`，`round(280.5)=280`（Python 银行家舍入）。两个客户端输入维分别是 281 / 280，正好对上 `split_vfl`。
- 输出各 16 维 → 全局输入 32。

### 3. `PHISHING_models.py`

```python
nn.Linear(15, 8), nn.ReLU(), nn.Linear(8, 4)   # L29-L31 / L35-L37
```

- 两个客户端结构完全相同（都写了两遍），各输出 4 维 → 全局 8 维。
- 全局 `8→4→2`，二分类。

### 4. `NUSWIDE_models.py`

```python
backbone_I = nn.Sequential(nn.Linear(634,320), ReLU, ..., nn.Linear(80,40), ReLU)    # L28-L37
backbone_T = nn.Sequential(nn.Linear(1000,500), ReLU, ..., nn.Linear(125,60), ReLU)  # L38-L47
if client_number == 0: self.backbone = backbone_I
else:                  self.backbone = backbone_T                                   # L48-L51
```

- 两个客户端**结构不同**（输入维和隐藏维都不同），这是唯一非对称的数据集。
- 输出 40 + 60 = 100，正好是全局模型第一层输入。

---

## 自测题

1. 为什么 `x_p` 在各数据集里都被完整保存，却在训练中从未使用？
2. CIFAR10 的 `split_vfl` 按哪个维度切？把 `x_a` 打印出来是什么 shape？
3. UCIHAR 两个客户端的输入维为什么是 281 和 280，而不是都 280？
4. PHISHING 的训练集/测试集为什么可能重叠？

<details>
<summary>参考答案</summary>

1. 它是“干净参照”，本可用于可视化或对比实验；当前训练路径只读 `x_n`，属于预留字段。
2. 按 `dim=2`（宽）切：`x_a` 为 `(B,32,16,3)`，`x_b` 为 `(B,32,16,3)`。
3. `ceil(561/2)=281`、`round(561/2)=280`，源码分别写了 ceil 和 round。
4. 每个 `PHISHING_VFL` 实例都会重新 `sample` 一次 8844 个训练下标，两次实例化结果不同，导致两集合不是互补划分。
</details>
