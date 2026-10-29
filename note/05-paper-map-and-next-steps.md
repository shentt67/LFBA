# 05 · 论文↔代码映射与下一步实验计划

## 一、论文概念 ↔ 代码符号

| 论文 | 代码 | 位置 |
|---|---|---|
| 纵向联邦、共享样本 | `split_vfl` 按特征列切分 | `dataset/utils.py` L48-L75 |
| 客户端本地嵌入 $z_i$ | `local_output_list[i]` | `trainer.py` L159-L160 |
| 嵌入梯度（标签语义载体） | `grad_vec_epoch` | `trainer.py` L180-L183 |
| 后门目标 / anchor | `--anchor_idx` | `main.py` L251 |
| 毒化集构建（GPC） | `get_near_index` + L69-L80 | `attack.py` L63-L67 |
| 硬样本选择（HS） | 梯度 L2 范数 top-k | `trainer.py` L86-L90 |
| 选择性样本切换 | `attack_LFBA` → `add_trigger_to_data_replace` | `trainer.py` L132-L138 |
| 触发器 | 指定维度置 1 / 右下角 patch | `attack/add_trigger*.py` |
| 主任务精度 | `test_acc` | `trainer.py` L330 |
| 攻击成功率 | `test_asr` | `trainer.py` L332 |
| 权衡指标 | `(ACC + ASR) / 2` | `trainer.py` L214 |

## 二、变量速查

| 变量 | 形状/类型 | 含义 |
|---|---|---|
| `x_split_list` | list，元素 `(B, d_i)` | 各客户端输入 |
| `local_output_list` | list，元素 `(B, k_i)` | 各客户端嵌入 |
| `grad_vec_epoch` | `(N, k_att)` | 攻击端嵌入梯度，**上一轮** |
| `train_indexes` | `(N,)` | 每轮的原始下标（乱序） |
| `poison_indexes` | `(num_poisons,)` | 毒化集原始下标，**固定** |
| `indexes` | 变长 | 位置数组，三种含义（见 04 篇） |
| `select_indexes` | `(num_select,)` | 硬样本在 `indexes` 内的位置 |
| `replace_indexes_others` | list | 提供特征的干净样本 |
| `replace_indexes_target` | list | 被替换特征的毒化样本 |

## 三、当前进度与缺口

| 项目 | 状态 |
|---|---|
| 环境/数据/CIFAR10 下载 | 已完成 |
| baseline（`attack=None`）5 轮 | 已跑通，ACC 0.638 |
| **LFBA 分支** | **尚未执行** |
| rsa / lra 对照 | 未执行 |
| 超参敏感性 | 未执行 |
| 多 seed 稳定性 | 未执行 |

所以“跑通”目前只覆盖了 `main.py` 的公共路径和 `Trainer` 的非攻击路径。LFBA 的梯度采集、毒化集构建、样本切换这三段代码**一次都没被执行过**，这是下一步的重点。

## 四、下一步：按优先级执行

### 第 1 步：先把 LFBA 跑起来（验证代码路径，不看指标）

```powershell
python main.py --data_dir ./data/ --dataset CIFAR10 --device 0 ^
  --epoch 20 --batch_size 128 --lr 0.001 ^
  --attack LFBA --anchor_idx 23470 --poison_rate 0.1 --select_rate 0.5
```

验收（日志里必须出现）：

1. `Target label: <x>` —— 说明 anchor 被正确定位、target 被设定；
2. `The poison set construction time: ...` —— 说明 `ep == 1` 的 GPC 分支执行了；
3. `The hard-sample selection time: ...` —— 说明硬样本选择执行了；
4. 每轮 ASR 有变化，且 trade-off 被记录。

若第 2、3 条不出现，说明 `--attack` 值没传对（大小写敏感，必须是 `LFBA`）。

### 第 2 步：同配置补 baseline

```powershell
python main.py --data_dir ./data/ --dataset CIFAR10 --device 0 ^
  --epoch 20 --batch_size 128 --lr 0.001 --attack None
```

只有同 epoch、同 batch、同 seed 的 baseline 才能和 LFBA 比较。baseline 的 ASR 应接近随机（约 0.1 附近）。

### 第 3 步：跑对照实验（每次只改一个变量）

| 编号 | 命令关键参数 | 想验证什么 |
|---|---|---|
| R1 | `--attack None` | 干净上界 |
| R2 | `--attack rsa --poison_rate 0.1` | 随机选样是否也能成功 |
| R3 | `--attack lra --poison_rate 0.1` | 有标签攻击的 ACC 代价 |
| R4 | `--attack LFBA --poison_rate 0.1` | 完整方法（默认即替换） |
| R5 | `--attack LFBA --poison_rate 0.1 --poison_all` | 只加触发器、不切换 |
| R6 | `--attack LFBA --poison_rate 0.1 --random_select` | 随机选硬样本 vs 按范数选 |
| R7 | `--attack LFBA --select_rate 1.0` | 全量切换 |
| R8 | `--attack LFBA --poison_rate 0.2` | 毒化比例敏感性 |

**重要发现**：`--select_replace` 在代码里**从未被读取**（全局搜索只有 `main.py` L249 的参数定义）。也就是说，只要不是 `--poison_all`，默认就走替换分支；README 里"加该参数才启用切换"的说法与实际代码不符。做消融时以 `--poison_all` 作为“不切换”的对照组。

### 第 4 步：验证 4 个可疑实现点（这是最能出成果的部分）

| 编号 | 疑点 | 验证方式 | 预期 |
|---|---|---|---|
| V1 | CIFAR10 替换区域按高切（`[:, 16:, :]`），而客户端按宽切（`[:, :, 16:]`） | 在 `add_trigger_replace.py` L51 打断点，打印被改写的像素坐标范围 | 修正后 ASR 应上升 |
| V2 | `get_near_index` 分母用 L1 范数，不是余弦 | 断点处用 L2 重算一次相关系数，比较毒化集重叠率 | 若重叠率低，说明实现与论文口径不同 |
| V3 | target label 取自 anchor 真实标签 | 对比“显式给 target label”与“用 anchor 标签”两种设置 | 指标应几乎一致，说明攻击不依赖标签选择 |
| V4 | PHISHING 训练/测试集重复采样导致重叠 | 统计两集合交集大小 | 交集非空即确认 |

V1/V2 若要改代码验证，需要你同意后再动源码；也可以复制一份脚本做对照，不动仓库。

### 第 5 步：稳定性与规模

1. 用 `--seed 100 / 101 / 102` 各跑一次，报告均值±标准差；
2. `--poison_dimensions` 取 5 / 10 / 20，看触发器维度对 ASR 的影响；
3. `--client_num 2` 之外的结构暂不支持（模型和切分都按 2 客户端写死），不要贸然改。

### 第 6 步：记录与产出

建议每跑完一组，把 `experiment.log` 的关键行抽成一张表（可以手写 Markdown，也可以写个解析脚本）：

| run | attack | poison_rate | select_rate | dims | best epoch | ACC | ASR | trade-off |
|---|---|---|---|---|---|---|---|---|

最终学习成果建议是三样东西：

1. 本笔记（论文-代码对照）；
2. 一张完整的实验结果表 + 与论文趋势的差异说明；
3. 一段 300 字结论：LFBA 为什么在无标签条件下可行、代码里哪些地方偏离论文、这些偏离对指标的影响。

## 五、“学懂了”的验收标准

能不看代码回答下面 6 个问题，就算过关：

1. 攻击者到底能看到哪些张量？哪些变量是“上帝视角”？
2. 毒化集为什么用梯度相似度而不是特征相似度？
3. 为什么切换样本能提升 ASR 而不明显掉 ACC？
4. 毒化集为什么只建一次，而硬样本每轮重选？
5. ASR 的分母为什么排除目标类？
6. 若把 `select_rate` 从 0.5 调到 1.0，ACC 和 ASR 分别会怎么动，为什么？

## 六、命令备忘（PowerShell）

```powershell
# 看全部参数
python main.py --help

# 只评测已有 checkpoint（注意：该分支引用了未定义的 args.do_tsne，当前会崩）
python main.py --dataset CIFAR10 --test_checkpoint results/CIFAR10/<时间戳>/best_checkpoint.pth.tar

# 续训
python main.py --dataset CIFAR10 --pretrained_checkpoint results/CIFAR10/<时间戳>/best_checkpoint.pth.tar --epoch 100
```

> `--test_checkpoint` 分支目前不可用，这是代码缺陷，不是你操作问题。
