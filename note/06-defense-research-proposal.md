# 06 · 抗 LFBA 的检测与保护机制：研究方案（投稿版）

> 目标：面向论文投稿，提出并验证一套针对 **Label-Free Backdoor Attacks in VFL (LFBA)** 的检测、阻断与净化机制。
> 约束：本阶段不修改仓库源码；所有分析基于现有实现与已有实验日志。

---

## 0. 一句话概括

LFBA 的“无标签”能力依赖两个可被服务器镜像的结构性痕迹——**梯度引导的毒化集选择（GPC）**与**选择性样本切换（SAW）造成的嵌入复制**。本文提出 **CV-Guard**：在训练期用“镜像锚点检测 + Δ-复制簇检测”定位毒化/切换样本，在外科式屏蔽阻止触发器学习，并在推理期估计触发器方向完成检测与净化，从而同时降低 ASR、保持 MTA、恢复被攻击样本的正确预测（UR）。

---

## 1. 研究问题与文献空白

### 1.1 LFBA 的攻击面（基于本仓库实现）

| 步骤 | 机制 | 代码位置 |
|---|---|---|
| 毒化集构建（GPC） | 用 anchor 与全体样本的**嵌入梯度余弦相似度**取 top-p | `attack/attack.py` L56-L86；`utils/trainer.py` L69-L77 |
| 硬样本选择（HS） | 在毒化集内按**梯度 L2 范数**取 top-Ns | `utils/trainer.py` L86-L90 |
| 选择性样本切换（SAW） | 用干净样本 `x_j + δ` 替换毒化样本的攻击端特征，受害端特征保留 | `attack/add_trigger_replace.py` L39-L55；`utils/trainer.py` L94-L138 |
| 最终训练集 | `D_f = D_c + (D_p − D_s) + D_sf`，**只有 D_s 带触发器** | 论文 Eq. (9)；代码同上 |
| 目标标签 | 代码直接取 anchor 的真实标签 | `utils/trainer.py` L105-L107 |

两个关键事实：

1. 毒化集 `D_p` 在**第 1 个 epoch 构建一次**，之后固定；切换子集 `D_s` 每轮重选。
2. 触发器只存在于 `D_s`，规模约为 `p·s·N`（默认 0.1×0.5×50000 = 2500，占 5%）。

### 1.2 现有防御为什么不够

| 工作 | 核心假设/机制 | 对 LFBA 的结构性盲点 |
|---|---|---|
| VFLIP (ESORICS'24) | 推理期 MAE 重建 + 参与者级异常检测 | 需要干净嵌入分布作参照；BVBench 实测 MTA 严重下降（Satellite 63.73→18.73），UR 远低于 MTA |
| VFLMonitor (TIFS'25) | 单方劫持攻击的检测与恢复 | 面向嵌入操纵型攻击；TPR/FPR 难以兼顾 |
| GBD (TDSC) | 通用测试期后门检测 | 非 VFL 专用；未针对标签一致毒化 |
| UBD (IJCAI'25) | 标签一致性聚类 + 潜在触发器合成 + 线性探测 | 依赖“预测 vs 真实标签不一致”；LFBA 毒化样本**真实标签就是目标类**，该信号天然失效；且只支持单目标 |
| FILTER (AAAI'26) | 嵌入过滤器 + 损失过滤器 + 遗忘 | 过滤器只在**同标签**嵌入高度相似时生效；LFBA 的捐赠样本是**跨类随机样本**，且 FILTER 假设攻击者已知标签 |
| ConFirm (TIFS'26) | 扰动 top model，观察预测置信度稳定性 | 毒化样本本身是目标类干净样本，置信度不一定异常；未见 LFBA 评测 |
| BVBench (arXiv 2608.12962, 2026-08) | 9 攻击 × 8 防御的系统评测 | 结论：**没有防御能同时保住 MTA、压低 ASR、恢复正确预测**；LFBA 是其 benchmark 中效果最好的攻击 |

LFBA 原文只测了**梯度压缩**与**高斯噪声**，两者都以主任务精度为代价，不能算有效防御。

### 1.3 本文要回答的三个问题

- **RQ1（可检测性）**：在没有原始数据、没有真实标签的条件下，服务器能否从梯度与嵌入中稳定识别 LFBA 的 `D_p`、`D_s` 与触发器方向？
- **RQ2（检测性能）**：该检测在跨数据集、跨毒化率/切换率、跨触发器维度时，能否在训练早期取得高 TPR、低 FPR？
- **RQ3（保护与恢复）**：检测后采用“屏蔽/替换可疑攻击端嵌入”能否阻断后门学习并保持 MTA？推理期净化能否把 UR 提升到接近 MTA？

> 这三个问题正好对应 BVBench 提出的三个评价维度（efficacy / utility / recovery），也是现有工作没有同时解决的空白。

---

## 2. 威胁模型与防御者能力

### 2.1 攻击者（与 LFBA 论文一致）

- 控制一个被动参与方（默认 client 1），可修改自己的**原始特征**与本地 bottom model。
- 不能访问其他参与方数据、top model、真实标签。
- 触发器为固定模式：CIFAR-10 为右下角 patch；表格数据为固定维度置常值。
- 攻击目标是让含触发器的样本被分为 anchor 所属类，同时保持干净样本精度。

### 2.2 防御者（active party / server）

- 可见：每个参与方的**逐样本嵌入** `z_i^k`、**逐样本嵌入梯度** `g_i^k`、真实标签 `y_i`、全局预测与损失。
- 不可见：任何参与方的原始特征与本地模型参数。
- 可做：记录/统计嵌入与梯度；决定 top model 如何融合各参与方嵌入；对可疑样本屏蔽、替换或降权；在推理期对嵌入做检测与净化。
- 协议兼容性：不新增通信轮次，不要求参与方上传额外信息；所有操作都在 active party 已有的计算与模型控制范围内。

### 2.3 不在本文范围

- 攻击者直接篡改发送的嵌入（embedding manipulation，如 VILLAIN / HijackVFL）。
- 多攻击者串谋、多目标后门（后续工作）。
- 参与方数量 > 2 的扩展（当前仓库写死 2 客户端）。

---

## 3. 关键观察：LFBA 的三个可检测痕迹

### O1. GPC 在梯度空间留下“锥形”结构

攻击者选的是与 anchor 梯度最相似的 p 个样本。对服务器可见的梯度矩阵 `G ∈ R^{N×d}`：
- `D_p` 内样本的梯度彼此近共线，形成一个窄锥；
- anchor 的 `top-p` 平均相似度与 `top-(p+1)` 平均相似度之间存在人为断层；
- 只要按类做 SVD，目标类的第一奇异值占比会异常高。

**这是攻击者自己用来实现“无标签”的信号，服务器本来就能算。**

### O2. SAW 在嵌入空间留下“复制-平移”结构

对 `i ∈ D_s`，攻击端输入是 `x_j + δ`，其中 `j` 是随机干净样本（可能跨类）。因此：

```
z_i^m = F_m(x_j + δ) ≈ F_m(x_j) + Δ = z_j^m + Δ
```

而受害端仍是 `z_i^v = F_v(x_i)`，与 `z_j^v` 语义无关。于是存在大量配对 `(i, j)`：
- 攻击端相似度高（复制 + 固定平移）；
- 受害端相似度低（不同样本、甚至不同类）；
- 差值 `Δ_ij` 的方向与模长高度一致（共享触发器偏移）。

**这是 LFBA 最独特的指纹，且与标签无关。**

### O3. 共享触发偏移可用于推理期检测与净化

所有触发样本共享同一个 `Δ`。一旦估出 `Δ̂`：
- 推理期可用投影/复制检验判断输入是否被触发；
- 净化可用 `z − Δ̂` 或“跨视图检索代理”恢复正确预测。

### 3.1 代码映射与可观测性

| 痕迹 | 服务器可见的量 | 代码依据 | 备注 |
|---|---|---|---|
| 梯度锥 | `g_i = ∂L/∂z_i^m` | `trainer.py` L180-L183 | 服务器本来就计算并回传该梯度 |
| 复制-平移 | `z_i^m`、`z_i^v` | `trainer.py` L159-L169 | 两方嵌入在 top model 前拼接 |
| 共享偏移 | 同一 `Δ` 的多对差值 | `add_trigger_replace.py` L44-L51 | 触发 patch/维度固定 |
| 时间结构 | 每轮 `D_s` 重选、`D_p` 固定 | `trainer.py` L62-L98 | 跨 epoch 聚合可提升稳定性 |

### 3.2 一个重要推论：可以在投毒生效前检测

LFBA 在第 0 轮用**干净梯度**选 `D_p`，第 1 轮才开始投毒。服务器在第 0 轮结束时就持有这些梯度。因此：
- **A2（镜像锚点）有机会在第 1 轮投毒开始前就标出 `D_p`**；
- A1（Δ-复制簇）在第 1 轮结束后即可确认 `D_s`。

这为“先发制人”的防御提供了时间窗口，是现有推理期防御不具备的。

---

## 4. 方法：CV-Guard

### 4.1 总览

```
训练期（每轮结束，服务器侧）
  ┌─ A2 镜像锚点检测 ──→ D̂_p, anchor, 目标类
  ├─ A1 Δ-复制簇检测 ──→ D̂_s, Δ̂
  └─ A3 跨视图确认 ─────→ 去除自然近邻误报
        │
        ├─ 训练期保护：对 D̂_s 屏蔽/替换攻击端嵌入
        └─ 记录 Δ̂ 供推理期使用

推理期
  检测：投影 + 复制检验 → 是否触发
  净化：z − Δ̂ 或跨视图检索代理 → 恢复正确预测
```

### 4.2 A1：Δ-复制簇检测（针对 SAW）

**输入**：攻击端嵌入 `Z_m ∈ R^{N×d}`、受害端嵌入 `Z_v ∈ R^{N×d}`。

**步骤**：

1. 分块计算攻击端余弦相似度，得到候选配对集合 `P`（每个样本保留 top-K 近邻，或全局 top-M 配对）。
2. 对每个配对 `(i,j)` 计算跨视图不一致分数：

   ```
   s_ij = cos(z_i^m, z_j^m) − λ · cos(z_i^v, z_j^v)
   ```

   取 `s_ij` 最高的若干配对作为候选复制对。`λ` 在干净轮（第 0 轮）上校准，使干净数据的假阳率低于目标水平。
3. 计算差值 `Δ_ij = z_i^m − z_j^m`，对 `{Δ_ij}` 做**方向 + 模长联合聚类**（球面聚类 / mean-shift / DBSCAN）。
4. 若存在一个支配簇 `C*`，其支持度显著高于干净轮零分布，且方向一致性 `cos(Δ_ij, Δ_i'j') > τ_dir`，则判定存在触发器，`Δ̂ = median(C*)`。
5. `D̂_s` = 参与支配簇的样本集合；再用 `Δ̂` 做一次全量精化：

   ```
   i ∈ D̂_s  ⇔  ∃j: ||(z_i^m − Δ̂) − z_j^m|| < ε 且 s_ij > 0
   ```

**零分布校准**：第 0 轮没有投毒，可直接用其 Δ 簇支持度、方向一致性作为 null；无需人工设定绝对阈值。

### 4.3 A2：镜像锚点检测（针对 GPC / 不切换模式）

服务器重跑攻击者的选择统计量。对候选 anchor `r` 与毒化率 `p`：

```
T(r,p) = mean_{i∈Top_p sim(g_r,g_i)} sim(g_r,g_i)
       − mean_{i∈Next_p sim(g_r,g_i)} sim(g_r,g_i)
```

- 在 `p` 的网格上扫描（如 0.01–0.30，步长 0.01），取 `max T`；
- 与干净轮零分布比较，超过阈值则判定存在 GPC 结构；
- `D̂_p = Top_p(sim(g_anchor, ·))`；
- 辅助确认：对 `D̂_p` 的梯度矩阵做 SVD，第一奇异值占比应显著高于同类随机子集。

**作用**：
- 覆盖 `--poison_all`（只加触发器、不切换）——此时 A1 失效，但 A2 仍有效；
- 可在第 0 轮结束、投毒尚未生效时给出预警。

### 4.4 A3：跨视图确认

A1 的候选对可能包含自然跨类近邻。用以下证据过滤：
- 复制对应在**多个 epoch** 重复出现（`D_p` 固定，`D_s` 重选，配对关系会变化但触发器方向不变）；
- 复制对的受害端语义不相邻，但攻击端差值方向与 `Δ̂` 一致；
- 若服务器有少量干净验证集，可用 `Δ̂` 合成触发样本，检验 top model 是否被诱导到同一目标类（模型级确认）。

### 4.5 训练期保护

对 `i ∈ D̂_s`，在 top model 融合时二选一（做消融）：

- **M1 屏蔽**：将 `z_i^m` 置零（或置为同类干净嵌入均值），仅用受害端嵌入参与该样本的损失与反传；对攻击端回传零梯度。
- **M2 跨视图检索替换**：用受害端 `z_i^v` 在干净样本中检索 k 近邻，取其攻击端嵌入均值作为 `z_i^m` 的代理。

**为什么代价小**：`D_s` 只占 5%，且其真实标签属于目标类（`D_p` 本身来自目标类）。屏蔽/替换这些样本对主任务精度的影响有限；而 LFBA 的触发器关联**只由 `D_s` 建立**（Eq. 9），阻断 `D_s` 即可从源头阻止后门学习。

### 4.6 推理期检测与净化

**检测**（两种可组合）：
- **D-proj**：`t(z) = <z − μ_m, Δ̂> / ||Δ̂||`，以干净嵌入的高分位数为阈值；
- **D-copy**：检验 `z − Δ̂` 是否在训练嵌入集合中存在近邻（复制检验）。

**净化**（两种，做消融）：
- **P1 减法**：`z' = z − Δ̂`；
- **P2 跨视图检索代理**：用受害端 `z^v` 检索 k 个干净训练样本，以它们的攻击端嵌入均值替换 `z^m`。P2 不依赖线性可加假设，通常更利于 UR。

### 4.7 复杂度与工程可行性

- 逐样本嵌入/梯度存储：`N×d×4B`，CIFAR-10 约 25.6 MB/方/轮，可忽略。
- 两两相似度：`O(N²d)`；N=50k、d=128 时用 GPU 分块矩阵乘可在数秒内完成，或用 FAISS/随机投影近似。
- 检测可按轮执行，也可每 k 轮执行；Δ 聚类只对 top-M 配对做，成本远低于全量。
- 所有计算都在 active party 本地，符合 VFL 协议。

### 4.8 伪代码

```python
# 每个 epoch 结束，服务器侧执行
def cv_guard_epoch(Zm, Zv, G, Y, null_stats, lam, grid_p):
    # A2: mirror anchor
    Sg = cosine_matrix(G)                     # chunked
    best = None
    for p in grid_p:
        T = mirror_statistic(Sg, p)           # per-anchor top-p gap
        if max(T) > null_stats.T_p99:
            r = argmax(T); Dp = top_p_indices(Sg[r], p)
            best = (r, p, Dp, max(T))
    # A1: delta-cluster
    pairs = top_cross_view_pairs(Zm, Zv, lam) # s_ij = cos_m - lam*cos_v
    deltas = [Zm[i] - Zm[j] for (i, j) in pairs]
    clusters = cluster_by_direction_and_norm(deltas)
    C = dominant_cluster(clusters)
    if support(C) > null_stats.delta_p99 and direction_consistency(C) > tau_dir:
        delta_hat = median(C)
        Ds = refine_by_shift(Zm, delta_hat, pairs)
    # target class (server-side, for reporting/verification only)
    target = mode(Y[Dp]) if best else None
    return Dp, Ds, delta_hat, target
```

---

## 5. 可检验命题（Claims）

- **C1（检测一致性）**：在 LFBA 的 GPC 下，`D_p` 的梯度矩阵第一奇异值占比随 `p` 与 ASR 增大而显著高于同类随机子集；镜像锚点统计量在攻击轮是零分布的离群点。
- **C2（复制可辨识）**：在 SAW 下，`D_s` 的配对差值 `Δ_ij` 在方向与模长上形成支配簇，支持度约等于 `N_s`，显著高于干净轮的 Δ 簇支持度。
- **C3（阻断充分性）**：由于 `D_f = D_c + (D_p − D_s) + D_sf`，把 `D_s` 从 top model 的损失中屏蔽即可消除触发器与目标类的关联；剩余样本不含触发器。
- **C4（恢复可行性）**：`z − Δ̂` 或跨视图检索代理能把被触发样本的预测恢复到与受害端语义一致，从而把 UR 提升到接近 MTA。

这些命题先做**分析与经验验证**，不强行包装成定理；论文中以“观察 + 实验 + 消融”呈现。

---

## 6. 实验设计

### 6.1 攻击与防御基线

**攻击**：
- LFBA（主攻击，含 `--select_replace`、`--poison_all`、`--random_select` 三种变体）
- RSA / LRA（攻击内部对照）
- 可选：BadVFL、VILLAIN（验证防御的通用性，非本文重点）

**防御基线**：
1. 无防御
2. 梯度压缩 / 高斯噪声（LFBA 原文的两种）
3. 嵌入范数过滤 / 嵌入随机丢弃（通用基线）
4. FILTER 风格：同标签嵌入去重 + 损失过滤
5. UBD 风格：标签一致性筛选 + 线性探测
6. VFLIP 风格：MAE 异常检测 + 重建（工程量大，可放补充实验）
7. **Oracle**：直接使用真实 `D_s` 屏蔽（上界）

> Oracle 很关键：它能证明“检测误差”与“阻断方法本身”各自的贡献，避免把检测失败误判为防御失败。

### 6.2 数据、模型与配置

- 数据集：CIFAR-10（主）、NUSWIDE、UCIHAR、PHISHING；后续可加 BVBench 的现实数据集（Satellite/KUHAR/PTB-XL/Vehicle）。
- 模型：沿用仓库（ResNet18 bottom + MLP top）；补一组 MLP bottom 验证架构无关性。
- 关键超参：`poison_rate ∈ {0.05, 0.1, 0.2}`、`select_rate ∈ {0.3, 0.5, 1.0}`、`poison_dimensions ∈ {5, 10, 20}`、`client_num=2`、`attack_client_num=1`。
- 已有基线：CIFAR-10、20 epoch 下 ACC 72.19%、ASR 96.18%（`results/CIFAR10/20261023_181047`）。
- 实现前提校验：先确认 CIFAR 张量布局与切片轴一致（note/05 的 V1 疑点），避免“替换区域跨到受害端”污染跨视图统计。

### 6.3 指标

**检测**：
- 样本级 ROC-AUC、TPR@FPR=1%、Precision/Recall；
- 恶意客户端识别准确率；
- 检测时延（第几个 epoch 首次报警）；
- 目标类识别准确率。

**防御**（BVBench 三件套）：
- MTA（干净主任务精度）；
- defended ASR；
- UR（被攻击样本在防御后预测正确的比例）；
- trade-off `(MTA + (1−ASR))/2`。

**开销**：每轮检测耗时、显存峰值、额外存储。

### 6.4 消融

1. 检测器组件：仅 A1 / 仅 A2 / A1+A2 / A1+A2+A3；
2. 阈值来源：干净轮零分布 vs 固定阈值 vs 分位数；
3. 时间聚合：单轮 vs 多轮中位数；
4. 训练期保护：M1 屏蔽 vs M2 检索替换；
5. 推理期净化：P1 减法 vs P2 检索代理；
6. 触发器强度：`poison_dimensions` / patch 大小；
7. 复制检测的配对策略：全局 top-M vs 互近邻 vs 跨视图分数 top-M。

### 6.5 自适应攻击（审稿人必问）

| 自适应策略 | 预期效果 | 防御应对 |
|---|---|---|
| 对 Δ 加逐样本噪声 | Δ 簇变松，A1 退化 | A2 仍可检测；且噪声会降低 ASR，形成 trade-off |
| 随机化 donor 选择 | 复制结构仍在 | A1 不受影响 |
| 不做切换（`poison_all`） | A1 失效 | A2 覆盖；但该变体 ASR 本身较低 |
| 延迟到第 5 轮才投毒 | 检测时延变长 | 每轮检测可持续发现；报告检测时延 |
| 用标签推断替代真实标签 | 不影响服务器观测 | 防御不依赖标签，仍有效 |
| 逐样本随机触发器 | Δ 方向不一致 | 需 A2 + 梯度锥；同时 ASR 可能下降 |
| 直接篡改嵌入 | 超出 LFBA 威胁模型 | 明确声明范围，不承诺防御 |

### 6.6 统计与复现

- 主表：5 seeds，报告 mean±std；
- 其余数据集与消融：3 seeds；
- 固定 seed、记录环境与命令；检测阈值只在干净轮校准，避免测试集泄漏；
- 所有检测统计量先在小规模（N=5k）上调通，再上全量。

---

## 7. 预期贡献与新颖性边界

### 7.1 预期贡献

1. **首个针对 LFBA 的可检测性分析**：给出 GPC 梯度锥与 SAW 嵌入复制两类服务器可观测痕迹，并证明现有防御的假设与 LFBA 不匹配。
2. **CV-Guard 机制**：训练期“镜像锚点 + Δ-复制簇”双检测器，标签无关、协议兼容，可在投毒生效前预警。
3. **外科式阻断 + 推理期净化**：只处理 5% 的切换样本，既阻止触发器学习，又通过跨视图检索净化恢复正确预测，直接补上 BVBench 指出的 UR 缺口。
4. **系统评测**：跨数据集、跨毒化率/切换率、含自适应攻击与 Oracle 上界。

### 7.2 与最近工作的差异（必须写清楚）

| 对比对象 | 差异 |
|---|---|
| FILTER | FILTER 只查“同标签嵌入重复”，且假设攻击者已知标签；本文查**跨类复制 + 跨视图不一致**，并估计共享触发器方向；本文还做推理期净化与 UR |
| UBD | UBD 依赖“预测与标签不一致”；LFBA 标签一致，UBD 结构性失效；本文不依赖标签一致性 |
| VFLIP | VFLIP 只在推理期做 MAE 异常检测；本文在训练期检测并阻断，且提供可解释的触发器方向估计 |
| ConFirm | 置信度稳定性检测不定位触发器方向，也不做净化；本文给出方向估计与恢复方案 |
| BVBench | BVBench 是 benchmark，不提出防御；本文借用其指标与结论，填补其“无有效防御”的空白 |

### 7.3 标题与投稿方向

- 英文候选：*Mirroring the Label-Free Backdoor: Detecting and Mitigating Gradient-Guided Poisoning in Vertical Federated Learning*
- 中文候选：《以彼之矛：面向纵向联邦无标签后门攻击的镜像检测与净化》
- 适配会议/期刊：IEEE S&P / CCS / USENIX Security / NDSS（安全主线）；或 IEEE TIFS / TDSC（期刊主线）；AAAI 可作为 ML 安全方向备选。

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 干净数据的自然跨类近邻造成 A1 假阳 | 误屏蔽干净样本 | 用第 0 轮零分布校准；A3 多轮一致性过滤；Oracle 对照 |
| 表格数据上 Δ 不明显（本地模型非线性/维度低） | A1 检测性能下降 | 以 A2 为主；A1 仅作确认；分数据集报告 |
| 检测阈值对数据分布敏感 | 跨数据集不可迁移 | 统一用干净轮零分布 + 分位数，不设绝对阈值 |
| 攻击者自适应加噪 | Δ 簇退化 | 报告 ASR-检测率 trade-off；A2 兜底 |
| BVBench 现实设置下 LFBA ASR 常 <30% | “防住弱攻击”的说服力不足 | 同时在原始设置（ASR 96%）与 BVBench 设置下评测，明确讨论 |
| 仓库实现与论文口径有偏差（如 target label 取真实标签、切片轴不一致） | 结果可复现性受质疑 | 在论文中显式说明实现差异；主实验用修正后的干净实现或做双版本对照 |
| 2 客户端写死 | 扩展性受限 | 本文限定 2 方；多方作为 future work |

---

## 9. 论文结构建议

1. Introduction：LFBA 的无标签威胁 → 现有防御为何失效 → 两个可观测痕迹 → 贡献。
2. Background & Threat Model：VFL 协议、LFBA 的 GPC/SAW、防御者能力。
3. Motivation / Pilot Study：在真实 LFBA 轨迹上展示梯度锥与 Δ 簇（一张图：干净轮 vs 攻击轮）。
4. Method：CV-Guard 的三个检测器、阻断与净化。
5. Evaluation：主表（MTA/ASR/UR）、检测 ROC、跨数据集、消融。
6. Adaptive Attacks：加噪、延迟、不切换、随机触发器。
7. Discussion / Limitations：2 客户端、固定触发器、嵌入篡改不在范围。
8. Related Work：VFL 后门攻击与防御两条线。
9. Conclusion。

---

## 10. 不改源码的第一步（离线预研）

在获得改码许可前，可以只新增 `defense/` 目录完成可行性验证：

1. `defense/replay.py`：加载 `results/CIFAR10/20261023_181047/best_checkpoint.pth.tar`，在训练集上重算两方嵌入与嵌入梯度。
2. `defense/simulate_lfba.py`：用第 0 轮/最终模型的梯度重放 GPC，按 SAW 规则离线构造 `D_s` 与触发嵌入（不改 `trainer.py`）。
3. `defense/detectors.py`：实现 A1/A2/A3 的统计量与 ROC 评估。
4. `defense/README.md`：记录命令、阈值、结果。

> 说明：离线重放能验证“信号是否存在”，但不能替代真实训练轨迹；真实轨迹验证需要后续给 `trainer.py` 加最小记录 hook（需另行确认）。

---

## 11. 参考文献（已核验）

- Shen et al., *Label-Free Backdoor Attacks in Vertical Federated Learning*, AAAI-25. https://ojs.aaai.org/index.php/AAAI/article/view/34246
- Cho et al., *VFLIP: A Backdoor Defense for Vertical Federated Learning via Identification and Purification*, ESORICS 2024. https://arxiv.org/pdf/2408.15591
- Chen et al., *Universal Backdoor Defense via Label Consistency in Vertical Federated Learning*, IJCAI 2025. https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4900.pdf
- Hu et al., *FILTER: A Framework for Defending against Backdoor Attacks in Vertical Federated Learning*, AAAI-26. https://ojs.aaai.org/index.php/AAAI/article/download/40859/44820
- Zhao et al., *Understanding Backdoor Vulnerabilities in Vertical Federated Learning: The Gap Between Research and Practice* (BVBench), arXiv:2608.12962, 2026. https://arxiv.org/html/2608.12962
- *ConFirm: Leveraging Confidence Consistency for Poisoned Sample Detection in Vertical Federated Learning*, IEEE TIFS, vol. 21, 2026. https://xplorestaging.ieee.org/document/11480883/references
- Xu et al., *VFLMonitor: Defending One-Party Hijacking Attacks in Vertical Federated Learning*, IEEE TIFS, vol. 20, pp. 4828–4843, 2025.
- Yuan et al., *General Test-Time Backdoor Detection in Split Neural Network-Based Vertical Federated Learning*, IEEE TDSC.
- Naseri et al., *BadVFL: Backdoor Attacks in Vertical Federated Learning*, 2024. https://arxiv.org/html/2304.08847v2
- Bai et al., *VILLAIN: Backdoor Attacks Against Vertical Split Learning*, USENIX Security 2023.
- *Hijack Vertical Federated Learning Models as One Party*, IEEE TDSC, 2024.
- *ProtoGuard-SL: Prototype Consistency Based Backdoor Defense for Vertical Split Learning*, arXiv:2604.03595, 2026. https://arxiv.org/html/2604.03595v1

> 投稿前请再用 DBLP / Google Scholar 复核一次“LFBA defense”相关新作，避免与最新预印本撞车。
