# 03 · attack 模块逐行注释

> 一句话：`attack.py` 决定「攻击谁」，`add_trigger*.py` 决定「怎么改特征」。

## 一、`attack/attack.py`

### 1. 三种攻击入口

```python
def attack_lra(args, logger, data, trigger_dimensions, targets, rate, mode):     # L9
    new_data = copy.deepcopy(data)                                               # L10
    new_targets = copy.deepcopy(targets)                                         # L11
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]  # L12 随机选毒化样本
    new_data, new_targets = add_trigger_to_data(..., replace_label=True)         # L13-L15 加触发器 + 改标签
    return new_data, new_targets
```

```python
def attack_rsa(args, logger, data, trigger_dimensions, rate, mode):              # L19
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]  # L21
    new_data, _ = add_trigger_to_data(..., replace_label=False)                  # L22-L23 只加触发器，不改标签
    return new_data
```

| 函数 | 毒化比例 | 改标签 | 用途 |
|---|---|---|---|
| `attack_lra` | `rate` | 是 → `target_label` | 有标签攻击基线 |
| `attack_rsa` | `rate` | 否 | 随机选样攻击 / 给测试集加触发器 |
| `attack_LFBA` | 指定下标集合 | 否 | 训练时动态毒化 |

> `lra` = label-replace attack，`rsa` = random-sample attack。二者都只用于**对照实验**。

### 2. `attack_LFBA`（L27-L38）

```python
if args.poison_all:                                                              # L30
    new_data, _ = add_trigger_to_data(..., poison_indexes, ..., replace_label=False)     # L31-L33 全量加触发器
else:
    new_data, _ = add_trigger_to_data_replace(args, logger, replace_indexes_others,
                                              replace_indexes_target, train_indexes,
                                              poison_indexes, ..., replace_label=False)  # L34-L37 切换版
```

- `--poison_all`：只加触发器，不替换特征（消融用）。
- 默认路径：**只替换 `replace_indexes_target` 这批样本的攻击端特征**，其余毒化样本保持原样。
- 两条路径都**不改标签**，这是 LFBA 与 lra 的本质区别。

### 3. 梯度选样：`get_near_index`（L63-L67）

```python
anchor_feature_l1 = torch.norm(anchor_feature, p=1)                              # L64
train_features_l1 = torch.norm(train_features, p=1, dim=1)                       # L65
vals, indices = torch.topk(
    torch.div((train_features @ anchor_feature), (train_features_l1 * anchor_feature_l1)),
    k=num_poisons, dim=0)                                                        # L66
return indices
```

- 输入：`train_features` 形状 `(N, d)`，是攻击端嵌入对损失的梯度；`anchor_feature` 是其中一个样本的梯度。
- 输出：与 anchor 最相似的 `num_poisons` 个样本在**当前 epoch 顺序**中的位置下标。

**坑 1（数学）**：分子是内积，分母却是 **L1 范数**乘积，所以它不是余弦相似度。若论文写的是 cosine，这里存在实现偏差。做敏感性分析时值得单独验证：改成 L2 范数后毒化集是否变化、ASR 是否变化。

**坑 2（口径）**：返回值是 `train_features` 中的行号，不是数据集原始下标。调用方必须再经 `train_indexes[...]` 映射（见 04 篇 L77）。

### 4. 未被调用的两个函数

```python
def select_LFBA(train_features, num_poisons):        # L41-L50
    anchor_idx = get_anchor_LFBA(train_features, num_poisons)
    ...
def get_anchor_LFBA(train_features, num_poisons):    # L53-L60
    consistency = train_features @ train_features.T                    # L54 相似度矩阵
    w = torch.cat((torch.ones(num_poisons), -torch.ones(num_poisons))) # L55 前 k 个 +1，后 k 个 -1
    top_con = torch.topk(consistency, 2 * num_poisons, dim=1)[0]       # L57 每行取 top-2k
    mean_top_con = torch.matmul(top_con, w)                            # L58 前 k 大之和 − 次 k 大之和
    idx = torch.argmax(mean_top_con)                                   # L59 选"最合群"的样本
    return idx
```

- 论文的“自动选 anchor”在这里实现了：找与自己最相似的一批样本、且与下一批差异最大的样本。
- **训练流程并没有调用它们**，实际 anchor 来自命令行 `--anchor_idx`。这是论文与开源实现的一处差异，写报告时要注明。

---

## 二、`attack/add_trigger.py`（只加触发器，不替换）

### 分发器（L5-L28）

按数据集把请求转到三种写触发器方式：CIFAR10 用空间 patch，其余用“指定维度赋值”。

### 1. CIFAR10：右下角三角 patch（L31-L45）

```python
height, width, channels = new_data.shape[1:]          # L32 HWC
for idx in poison_indexes:
    if replace_label and mode == 'train':
        new_targets[idx] = args.target_label           # L34-L35 只有 lra 才改标签
    for c in range(channels):
        new_data[idx, height-3:, width-3:, c] = 0      # L37 3x3 区域清零
        new_data[idx, height-3, width-1, c] = 255      # L38-L41 四个角点置白 → 三角形
```

- 触发器位于**右下角**，即 width ≥ 16 的一半，属于客户端 1（攻击者）的特征域。
- 直接写 uint8 数组，`__getitem__` 里才做 ToTensor/Normalize。

### 2. UCIHAR：维度置 `trigger_feature_clip`（L48-L57）

```python
new_data[idx][trigger_dimensions] = args.trigger_feature_clip   # L53 默认 1.0
```

- UCIHAR 特征已标准化，1.0 是一个“偏离均值”的固定值。

### 3. PHISHING / NUSWIDE：维度置 1（L60-L69）

```python
new_data[idx][trigger_dimensions] = 1    # L65
```

三者形式相同，只是触发维度范围不同（见 01 篇 L164-L176）。

---

## 三、`attack/add_trigger_replace.py`（触发器 + 跨样本替换）

### 核心思想

把样本 A（干净样本）的攻击端特征整段搬到样本 B（毒化样本）上，B 的标签保持不变。这样 B 的本地视图变成“别人的特征 + 触发器”，全局模型无法再用原始特征解释标签，只能依赖触发器。

### 1. 向量型数据集（L73-L88）

```python
temp = copy.deepcopy(new_data)                       # L76 整份数据副本
for i, idx in enumerate(replace_indexes_others):     # L78 遍历"提供特征的干净样本"
    temp[idx][trigger_dimensions] = 1                # L79 先给干净样本打触发器
    new_data[replace_indexes_target[i]][634:] = temp[idx][634:]   # L84 整段搬到毒化样本（NUSWIDE）
```

- PHISHING 对应 `[15:]`（L80），UCIHAR 对应 `[281:]`（L66）。
- 复制范围 = **攻击客户端持有的整个特征段**，所以触发器维度自然被一起搬过去。
- 成对关系由 04 篇 L94-L98 决定：`replace_indexes_others`（随机干净样本） ↔ `replace_indexes_target`（毒化集里的硬样本）。

**坑 3（性能）**：每轮都 `deepcopy` 整个数据集。CIFAR10 训练集约 50000×3072 字节 ≈ 150 MB，每轮一次完整拷贝，是主要的内存/时间开销之一。

### 2. CIFAR10（L39-L55）—— 维度对不上

```python
temp[idx, height-3:, width-3:, c] = 0                 # L46-L50 在干净样本右下角打三角
new_data[replace_indexes_target[i], :, 16:, :] = temp[idx, :, 16:, :]   # L51
```

`new_data` 是 `(N, H, W, C)`，因此 `[:, 16:, :]` 切的是**高度**（dim=1）的后 16 行；但 `split_vfl` 用的是 `data[:, :, :, 16:]`，切的是**宽度**（dim=2）的后 16 列。

**坑 4**：替换区域与客户端实际持有的特征区域不一致。理论上应写成 `new_data[target, :, :, 16:] = temp[idx, :, :, 16:]`。这很可能影响 CIFAR10 上的 ASR，建议作为你下一步的验证点之一。

### 3. 分发器（L6-L36）

与 `add_trigger.py` 同名同构，只是把 `poison_indexes` 换成了三元组
`(replace_indexes_others, replace_indexes_target, train_indexes)`。`replace_label` 一路传 `False`，所以**标签始终不变**。

---

## 自测题

1. `attack_rsa` 与 `attack_lra` 的唯一区别是什么？
2. `get_near_index` 返回的下标能直接拿去索引 `dataset.data` 吗？为什么？
3. 为什么替换时要复制“整个攻击端特征段”，而不是只复制触发器维度？
4. CIFAR10 的替换区域写错在哪个维度？修正后预期 ASR 会怎么变？

<details>
<summary>参考答案</summary>

1. 是否把毒化样本标签改成 `target_label`（`replace_label` 开关）。
2. 不能。它是 `train_features`/`train_indexes` 中的位置，需再经 `train_indexes[...]` 映射成原始下标。
3. 只复制触发器会让“原特征 + 触发器”仍然可学；整段替换才真正切断原始特征与标签的关联，迫使模型依赖触发器。
4. 错在把宽（dim=2）写成了高（dim=1）；修正后攻击端特征被真正替换，ASR 预期显著上升。
</details>
