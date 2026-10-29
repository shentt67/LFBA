# 04 · utils/trainer.py 逐行注释（核心）

> 一句话：LFBA 的全部攻击逻辑都在这 346 行里。读懂 `train()` 的 L62-L140，就读懂了论文的方法部分。

## 类地图

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | L11-L27 | 接住 `main.py` 装配好的对象 |
| `adjust_learning_rate` | L29-L33 | 每 20 轮学习率 ×0.1 |
| `train` | L35-L253 | 主循环：采集梯度 → 构建毒化集 → 切换样本 → 训练 → 评测 |
| `test` | L257-L346 | 算 ACC / target acc / ASR |

---

## 一、`__init__`（L11-L27）

```python
self.model_list = model_list          # L16 [全局, 客户端1, 客户端2, ...]
self.extractor_list = extractor_list  # L17 全程未使用
self.extractor = extractor            # L18 全程未使用
self.optimizer_list = optimizer_list  # L19 与 model_list 一一对应
self.train_loader / test_loader / test_asr_loader   # L21-L23
self.args = args                      # L25 超参全部从这里取
self.checkpoint = checkpoint          # L26 续训时恢复 best_acc
```

`self.extractor`、`self.select_his`（L54）都是死代码，读的时候可以跳过。

---

## 二、`train()` 的骨架

```python
for ep in range(self.args.start_epoch, self.args.epoch):   # L58
    [A] 若 attack == 'LFBA' 且 ep >= 1：构建/更新毒化数据        # L62-L140
    [B] 遍历 train_loader：前向、反向、记录梯度、更新参数        # L145-L204
    [C] 轮末：拼接梯度、调学习率、评测、保存最优、早停           # L205-L253
```

**时间线（务必记住）**

```
ep=0 : 干净数据训练一轮，把攻击端梯度存进 grad_vec_epoch
ep=1 : 用 ep=0 的梯度构建毒化集（只此一次）；本轮数据据此改写过
ep>=2: 毒化集不变；每轮用"上一轮（已中毒数据）"的梯度重选硬样本并再次改写数据
```

---

## 三、[A] LFBA 分支（L62-L140）

### 1. 取出上一轮缓存的梯度（L62-L66）

```python
if ep >= 1 and self.args.attack == 'LFBA':
    self.train_features = self.grad_vec_epoch   # L63 (N, d) 攻击端嵌入对 loss 的梯度
    self.train_labels   = self.target_epoch     # L63 真实标签（评估/取 target 用）
    self.train_indexes  = self.indexes_epoch    # L63 对应的原始样本下标
    ... .cpu()                                  # L64 梯度搬到 CPU，省显存
    self.num_poisons = int(self.args.poison_rate * len(dataset.data))   # L65 例如 5000
    self.num_select  = int(self.num_poisons * self.args.select_rate)    # L66 例如 2500
```

- 梯度 shape 对 CIFAR10 是 `(50000, 128)`，对 UCIHAR 是 `(7352, 16)`。
- `grad_vec_epoch` 在 [L180-L183](#4-记录攻击端梯度l180-l183) 每轮重新收集，所以这里拿到的永远是**上一轮**的梯度。

### 2. 只构建一次毒化集（L69-L80）

```python
if ep == 1:
    self.anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()  # L71
    self.indexes = get_near_index(self.train_features[self.anchor_idx_t],
                                  self.train_features, self.num_poisons)                     # L72-L73
    self.poison_indexes = self.train_indexes[self.indexes]                                  # L77
    self.consistent_rate = float(
        (self.train_labels[self.indexes] == int(self.train_labels[self.anchor_idx_t])).sum()
        / len(self.indexes))                                                                 # L78-L80
```

逐行解释：

- [L71](../utils/trainer.py#L71)：`--anchor_idx` 是**数据集原始下标**，在乱序的 `train_indexes` 里找到它的位置。
- [L72-L73](../utils/trainer.py#L72-L73)：在梯度空间找与 anchor 最相似的 `num_poisons` 个样本，得到的是**位置** `self.indexes`。
- [L77](../utils/trainer.py#L77)：位置 → 原始下标，得到真正的毒化集 `poison_indexes`。
- [L78-L80](../utils/trainer.py#L78-L80)：`consistent_rate` 是“毒化集里与 anchor 同标签的比例”，**只用于打印/分析**，不参与任何计算。

> 这就是论文里的 GPC（gradient-based poison construction）。因为只在 `ep == 1` 执行，后面所有轮都复用同一批 `poison_indexes`。

### 3. 选硬样本并配对（L83-L98）

```python
self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy())  # L83
temp = np.array(range(len(self.train_indexes)))                                                # L84
self.indexes = temp[self.indexes]        # L85 毒化样本在 train_indexes 中的位置
self.l2_norm_features = torch.norm(self.train_features[self.indexes], p=2, dim=1)              # L86
self.poison_features, self.select_indexes = self.l2_norm_features.topk(
    self.num_select, dim=0, largest=True, sorted=True)                                          # L88-L90
```

- [L83-L85](../utils/trainer.py#L83-L85)：布尔掩码 → 位置数组。**`self.indexes` 在这里被复用了**：L72 时它表示“毒化样本位置”，L85 后表示“毒化样本在 `train_indexes` 里的位置”，含义相同但长度不同，读代码时不要混淆。
- [L86](../utils/trainer.py#L86)：只对毒化集算梯度的 L2 范数。
- [L88-L90](../utils/trainer.py#L88-L90)：取范数最大的 `num_select` 个——即“最难学/最被欠拟合”的样本。这正是论文的 Hard-Sample Selection（日志里的 "hard-sample selection time"）。

```python
num_of_replace = int(len(self.poison_indexes) * self.args.select_rate)                          # L94
replace_all_list = list(set(self.train_indexes.numpy()).difference(set(self.poison_indexes.numpy())))  # L95
replace_indexes_others = sample(replace_all_list, num_of_replace)                               # L96
random_indexes_target = sample(list(self.poison_indexes), num_of_replace)                       # L97
selected_indexes_target = self.train_indexes[self.indexes[self.select_indexes]]                 # L98
```

- [L95-L96](../utils/trainer.py#L95-L96)：从**非毒化样本**里随机抽同样数量，作为“特征提供者”。
- [L98](../utils/trainer.py#L98)：两次映射 `select_indexes → indexes → train_indexes`，把 top-k 位置还原成**原始下标**，作为“被替换者”。
- 若 `--random_select`，则改为随机抽毒化样本（L97 + L125-L126），这是消融开关。

### 4. 两条毒化路径

**（a）`--poison_all`（L100-L123）**

```python
self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)          # L108
self.train_loader.dataset.data = attack_LFBA(..., self.poison_indexes, ...) # L110-L123
```

只加触发器、不替换特征。`clean_data_p` 是每轮的**干净起点快照**，保证毒化不跨轮累积。

**（b）默认：选择性切换（L124-L139）**

```python
self.poisoning_labels = np.array(self.train_labels)[self.indexes]            # L129 仅记录
self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])  # L130
self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)          # L131
self.train_loader.dataset.data = attack_LFBA(self.args, self.logger,
                                             replace_indexes_others,
                                             replace_indexes_target,
                                             self.train_indexes, self.poison_indexes,
                                             self.clean_data_p, ...)         # L132-L138
self.args.target_label = self.anchor_label                                   # L139
```

- [L130](../utils/trainer.py#L130)：**target label 取自 anchor 的真实标签**。论文声称不用标签；这里是实现层面的“上帝视角”捷径，严格说应改成由梯度语义推断。写报告/质疑时这是重要观察点。
- [L131](../utils/trainer.py#L131)：每次从 `data_p`（初始化时的干净副本）深拷贝，再写回 `data`。所以第 5 轮的数据 = 干净数据 + 本轮的触发器/替换，不叠加历史。
- [L139](../utils/trainer.py#L139)：`target_label` 每轮被重设成同一个值（anchor 不变），供 `test()` 计算 ASR 使用。

---

## 四、[B] 一个 batch 的前向/反向（L145-L204）

### 1. 取数与纵向切分（L150-L157）

```python
for step, (x_n, x_p, y, index) in enumerate(self.train_loader):   # L150
    x = x_n.to(self.device).float()                               # L152 只用 x_n，x_p 丢弃
    y = y.to(self.device).long()                                  # L153
    x_split_list = split_vfl(x, self.args)                        # L155 按数据集切给各客户端
    local_output_list, global_input_list = [], []                 # L156-L157
```

### 2. 本地模型前向（L159-L160）

```python
for i in range(self.args.client_num):
    local_output_list.append(model_list[i + 1](x_split_list[i]))  # L160 客户端嵌入 z_i
```

### 3. 为取梯度做准备（L162-L169）

```python
global_input_t = local_output_list[i].detach().clone()            # L163 全局模型输入用"断图"副本
global_input_t.requires_grad_(True)                               # L164
global_input_list.append(global_input_t)
local_output_list[i].requires_grad_(True)                         # L166 让 z_i 成为可求导叶子
local_output_list[i].retain_grad()                                # L167 保留 z_i 的梯度
x_split_list[i].requires_grad_(True)                              # L168
x_split_list[i].retain_grad()                                     # L169
```

- 注意：全局模型实际用的是 `global_input_list`（L171），而本地模型输出 `local_output_list` 仍挂在计算图上，是为了拿到 `z_i.grad`。
- `retain_grad()` 是整段攻击的“眼睛”：没有它，非叶子张量的梯度 backward 后即被释放。

### 4. 全局前向 + 反向（L171-L178）

```python
global_output = model_list[0](local_output_list)                   # L171 注意传的是未 detach 的 z_i
loss = self.criterion(global_output, y)                            # L174
for opt in self.optimizer_list: opt.zero_grad()                    # L175-L176
loss.backward()                                                    # L178 一次 backward 更新所有子模型
```

### 5. 记录攻击端梯度（L180-L183）

```python
if self.args.attack == 'LFBA':
    self.grad_vec_epoch.append(local_output_list[self.args.attack_client_num].grad.to(self.device))  # L181
    self.indexes_epoch.append(index)                                                                 # L182
    self.target_epoch.append(y)                                                                      # L183
```

- 只保存**攻击客户端**的嵌入梯度，其它客户端的梯度不落地。
- `index` 是原始样本下标，用于下一轮把梯度与样本对应起来。
- 每轮结束在 [L205-L208](../utils/trainer.py#L205-L208) 用 `torch.cat` 拼成整轮张量。

### 6. 参数更新与训练指标（L185-L204）

```python
for opt in self.optimizer_list: opt.step()                        # L185-L186
_, predicted = global_output.max(1)                               # L190
total += y.size(0); correct += predicted.eq(y).sum().item()       # L191-L192
```

注意 `predicted` 是在 `opt.step()` **之后**用旧 logits 算的，所以 train acc 略滞后于参数状态——不影响结论。

---

## 五、[C] 轮末处理（L205-L253）

```python
self.grad_vec_epoch = torch.cat(self.grad_vec_epoch)      # L206 下一轮的 train_features
...
self.adjust_learning_rate(ep + 1)                         # L212 每 20 轮 ×0.1
test_acc, test_poison_accuracy, test_target, test_asr = self.test(ep)  # L213
test_trade_off = (test_acc + test_asr) / 2                # L214
if test_trade_off > best_trade_off:                       # L215
    ... 保存 best_checkpoint.pth.tar                     # L225-L237
else:
    if ep > self.args.pretrain_stage: no_change += 1      # L239-L240
if no_change == self.args.early_stop: ... return          # L246-L253
```

- **模型选择标准是 `(ACC + ASR)/2`**，不是单纯 ACC。所以最终 checkpoint 往往不是 ACC 最高的那一轮。
- `pretrain_stage` 默认 0，因此从第 1 轮（ep=2）起就开始累计 `no_change`。
- 早停计数只统计“未刷新 trade-off”的连续轮数。

---

## 六、`test()`（L257-L346）

### 1. 干净测试集：ACC 与 target acc（L267-L294）

```python
for step, (x, x_p, y, index) in enumerate(self.test_loader):      # L267
    ...
    total += y.size(0); correct += predicted.eq(y).sum().item()                    # L291-L292
    total_target += (y == self.args.target_label).float().sum()                    # L293
    correct_target += predicted.eq(y)[y == self.args.target_label].float().sum()    # L294
```

- `test_acc` = 总体准确率。
- `test_target` = **目标类被正确识别的比例**（不是攻击指标，不要和 ASR 混）。

### 2. 带触发器测试集：ASR（L301-L327）

```python
y_attack_target = torch.ones(size=y.shape).long() * self.args.target_label   # L304-L305 定义后未使用
...
total_asr += (y != self.args.target_label).float().sum()                    # L326 分母排除目标类
correct_asr += (predicted[y != self.args.target_label] == self.args.target_label).float().sum()  # L327
```

- ASR 定义：**真实标签不是 target 的样本中，被预测成 target 的比例**。
- 分母排除目标类样本，是为了避免“本来就该预测成 target”的样本虚高 ASR。
- [L304-L305](../utils/trainer.py#L304-L305) 的 `y_attack_target` 是死代码。

### 3. 汇总（L330-L335）

```python
test_acc = correct / total
test_poison_accuracy = correct_poison / total_poison   # 带触发器样本的整体 ACC
test_asr = correct_asr / total_asr
test_target = correct_target / total_target
```

`test_poison_accuracy` 会被返回但 `train()` 只用它记录 best epoch，不参与选择。

---

## 七、读这段代码最容易踩的 5 个点

1. **`ep` 从 0 开始**：`ep == 1` 是第二轮，不是第一轮。
2. **`self.indexes` 一处三义**：位置 / 毒化样本位置 / 毒化样本在 `train_indexes` 中的位置。
3. **梯度是上一轮的**：`train_features` 永远滞后一轮；从 ep=2 起，它来自已被毒化的数据。
4. **毒化集固定、硬样本每轮变**：`poison_indexes` 只在 ep=1 生成；`selected_indexes_target` 每轮重算。
5. **数据每轮重置**：`clean_data_p` 从 `data_p` 重新拷贝，毒化效果不会累积。

## 自测题

1. 如果 `select_rate=0.3`、`poison_rate=0.1`、N=50000，那么 `num_poisons`、`num_select`、`num_of_replace` 各是多少？
2. 为什么 `local_output_list[i]` 既要 `requires_grad_` 又要 `retain_grad()`？
3. 从 ep=2 开始，硬样本选择用的梯度来自干净数据还是毒化数据？这会带来什么风险？
4. 最终保存的 checkpoint 一定是 ACC 最高的吗？

<details>
<summary>参考答案</summary>

1. 5000 / 1500 / 1500。
2. `requires_grad_` 让它成为可求导张量，`retain_grad()` 保证 backward 后 `.grad` 不被释放。
3. 毒化数据（上一轮已经改写过 `dataset.data`）；风险是梯度被攻击自身扭曲，硬样本选择可能自激震荡。
4. 不一定，按 `(ACC+ASR)/2` 选择。
</details>
