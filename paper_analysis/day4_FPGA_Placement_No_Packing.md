# Day 4: FPGA 布局新范式 —— 无需显式打包的 Place-Legalize 流程

> **论文标题**: A New Paradigm for FPGA Placement without Explicit Packing
>
> **作者**: Wuxi Li (Student Member, IEEE), David Z. Pan (Fellow, IEEE)
>
> **机构**: Department of Electrical and Computer Engineering, The University of Texas at Austin
>
> **期刊**: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)
>
> **卷/期/页码**: Vol. 38, No. 11, pp. 2113–2126
>
> **DOI**: [10.1109/TCAD.2018.2877017](https://doi.org/10.1109/TCAD.2018.2877017)
>
> **关键词**: Placement, Legalization, Packing, FPGA, Parallel Algorithm
>
> **资助**: Xilinx Inc.
>
> **分析日期**: 2026-06-08
>
> **与前三天论文的关系**: 第一作者 Wuxi Li 是 DREAMPlace（Day 1）的第三作者，也是 UTPlaceF（ISPD 2016/2017 冠军）的一作。本文基于 UTPlaceF 框架，但采用了**二次布局**（而非 ePlace 静电模型），并引入了完全创新的直接合法化算法。

---

## 目录

1. [FPGA 布局与打包的根本矛盾](#1-fpga-布局与打包的根本矛盾)
2. [核心贡献：Place-Legalize 新范式](#2-核心贡献place-legalize-新范式)
3. [FPGA 架构与问题定义](#3-fpga-架构与问题定义)
4. [创新点一：动态 LUT/FF 面积调整（DAA）](#4-创新点一动态-lutff-面积调整daa)
5. [创新点二：完全可并行的直接合法化（DL）](#5-创新点二完全可并行的直接合法化dl)
6. [算法流程](#6-算法流程)
7. [实验结果与分析](#7-实验结果与分析)
8. [创新点深度分析](#8-创新点深度分析)
9. [参考文献](#9-参考文献)

---

## 1. FPGA 布局与打包的根本矛盾

### 1.1 传统 FPGA 设计流程的四种范式

本文将已有工作归纳为四种流程范式：

```mermaid
graph TD
    A["<b>Pack-Place-Legalize</b><br/>传统流程<br/>T-VPack → VPR"] --> B["<b>Place-Pack-Place-Legalize</b><br/>物理感知流程<br/>FIP → Packing → 放置"]
    B --> C["<b>Place-SemiPack-Legalize</b><br/>半融合流程<br/>FIP → BLE打包 → 合法化"]
    C --> D["<b>Place-Legalize (本文)</b><br/>无显式打包<br/>FIP → 直接合法化"]
```

| 流程 | 代表 | Packing 阶段 | 核心问题 |
|------|------|-------------|---------|
| Pack-Place-Legalize | T-VPack, VPR | 完全独立，不考虑物理位置 | Packing 决策盲目 |
| Place-Pack-Place-Legalize | UTPlaceF, LSC | 先做 FIP，再基于物理信息 Packing | FIP 与最终解偏差大 |
| Place-SemiPack-Legalize | RippleFPGA | 仅做 BLE 级打包，CLB 打包留给合法化 | 仍存在信息断层 |
| **Place-Legalize (本文)** | — | **完全消除** | 需要解决新的挑战 |

### 1.2 顺序化流程的核心问题

传统 Place-Pack-Place-Legalize 流程存在两个关键问题：

**问题 1：FIP 与最终合法解的巨大偏差**

Packing 阶段将 BLE 打包进 CLB 时，CLB 的位置是通过对内部 BLE 位置取平均估计的——这个估计可能远离 CLB 的最终合法位置。特别是对于**控制集（control set）密集的设计**，FIP 中精心优化的线长、时序和可布线性指标在合法化后可能被严重破坏。

**问题 2：面积溢出不等于资源溢出**

FIP 只检查 LUT+FF 的总面积是否溢出，但不区分 LUT 和 FF 的资源类型。可能出现总面积不溢出、但 FF 密度溢出的情况（如下图）。传统布局器无法捕捉这种**资源类型不平衡**。

```
资源需求                                    资源需求
LUT  FF  LUT+FF  面积容量  CLB容量         LUT  FF  LUT+FF  面积容量  CLB容量
 │   │    │       ─ ─ ─     ─ ─            │   │    │       ─ ─ ─     ─ ─
 │   █    │       ─ ─ ─     ─ ─            │   │    │       ─ ─ ─     ─ ─
 │   █    │       ─ ─ ─     ─ ─            │   │    │       ─ ─ ─     ─ ─
 │   ██   │       ─ ─ ─     ─ ─            │   │    │       ─ ─ ─     ─ ─

 (a) 面积不溢出，但FF溢出！              (b) 合法：LUT和FF均不溢出
```

---

## 2. 核心贡献：Place-Legalize 新范式

### 2.1 核心主张

> **消除显式 Packing 阶段，直接从扁平初始布局（FIP）通过合法化获得最终合法解，在合法化过程中同时探索 Placement 和 Packing 的解空间。**

### 2.2 三大贡献

1. **动态 LUT/FF 面积调整（DAA）**：在 FIP 迭代中动态调整每个 LUT/FF 的面积，以考虑 Packing 效应和资源类型利用率，使 FIP 尽量接近真正合法的布局
2. **完全可并行的直接合法化（DL）**：受 Gale-Shapley 大学录取问题启发，每个 CLB slice 独立并行地寻找最优 BLE 聚类，同时满足 Placement 和 Packing 合法性
3. **实验验证**：ISPD 2016 基准上 routed wirelength 比 UTPlaceF 改善 **4.4%**，在难打包设计 FPGA-10 上改善 **29.5%**

---

## 3. FPGA 架构与问题定义

### 3.1 目标 FPGA 架构：Xilinx UltraScale VU095

本文采用 ISPD 2016 竞赛使用的 Xilinx UltraScale VU095 架构：

```
CLB (Configurable Logic Block)
├── Half CLB 0
│   ├── BLE 0: LUT_A + LUT_B + FF_A + FF_B  ← 共享 CK₀, SR₀, CEA₀, CEB₀
│   ├── BLE 1: LUT_A + LUT_B + FF_A + FF_B
│   ├── BLE 2: LUT_A + LUT_B + FF_A + FF_B
│   └── BLE 3: LUT_A + LUT_B + FF_A + FF_B
└── Half CLB 1
    ├── BLE 4: LUT_A + LUT_B + FF_A + FF_B  ← 共享 CK₁, SR₁, CEA₁, CEB₁
    ├── BLE 5: LUT_A + LUT_B + FF_A + FF_B
    ├── BLE 6: LUT_A + LUT_B + FF_A + FF_B
    └── BLE 7: LUT_A + LUT_B + FF_A + FF_B
```

**架构规则**：
- 每个 BLE 包含 **2 个 LUT**（可实现 1 个 6-input LUT 或 2 个总输入 ≤ 5 的小 LUT）和 **2 个 FF**
- 同一 BLE 中的 2 个 FF 必须共享 **CK（时钟）和 SR（置位/复位）**信号，但 **CE（时钟使能）**可以不同
- 同一 Half CLB 中的 4 个 BLE 共享相同的 CK, SR, CEA, CEB

> **控制集（Control Set）**：FF 的控制集定义为 (CK, SR, CE)；Half CLB 的控制集定义为 (CK, SR, CEA, CEB)。控制集的多样性直接决定了 Packing 的难度——不同控制集的 FF 无法放入同一个 Half CLB。

### 3.2 直接合法化（DL）问题的数学定义

给定 FIP 中所有单元的坐标 \( (\bar{x}, \bar{y}) \)，直接合法化问题定义为：

\[
\max_{\mathbf{x}, \mathbf{y}} \sum_{s \in S} \Psi(\{v \in V \mid z_{v,s} = 1\}) - \frac{1}{\eta} \cdot \text{HPWL}(\mathbf{x}, \mathbf{y})
\]

\[
\text{s.t.} \quad \sum_{s \in S} z_{v,s} = 1, \quad \forall v \in V \quad \text{(每个单元恰好分配到一个 slice)}
\]

\[
\{v \mid v \in V, z_{v,s} = 1\} \text{ 是架构合法的}, \quad \forall s \in S \quad \text{(满足所有控制集/容量规则)}
\]

\[
|x_v - \bar{x}_v| + |y_v - \bar{y}_v| \leq D, \quad \forall v \in V \quad \text{(最大位移约束)}
\]

其中：
- \( V \) 是所有 LUT 和 FF 的集合，\( S \) 是所有 CLB slice 的集合
- \( z_{v,s} \) 是二元变量：\( z_{v,s} = 1 \) 当且仅当单元 \( v \) 分配到 slice \( s \)
- \( \Psi(\cdot) \) 是聚类评分函数（捕捉引脚/网络共享、时序影响等）
- \( \eta > 0 \) 是线长归一化参数
- \( D \) 是最大位移约束（软约束）

> **公式解读**：目标函数同时最大化聚类质量（Packing 目标）和最小化线长（Placement 目标）。约束 (3b) 保证每个单元恰好属于一个 slice，约束 (3c) 保证架构合法性（控制集兼容、容量不超），约束 (3d) 限制位移以保持 FIP 质量。**这是第一个同时保证 Placement 和 Packing 合法性的公式化表述。**

---

## 4. 创新点一：动态 LUT/FF 面积调整（DAA）

### 4.1 核心思想

在 FIP 的每次迭代后，动态调整每个 LUT 和 FF 的面积，使得：
1. **难打包的单元获得更大面积**（反映其实际资源需求）
2. **不同资源类型的利用率分别控制**（解决 LUT/FF 不平衡问题）
3. **布线拥塞区域避免过度压缩**

### 4.2 局部资源利用率

对于单元 \( v \)，定义其局部利用率为：

\[
U_v = \frac{\sum_{i \in N_v^+} A_i}{C_v}
\]

其中：
- \( N_v^+ = \{v\} \cup N_v \)，\( N_v \) 是与 \( v \) 同类型且在 \( (x_v - L, y_v - L, x_v + L, y_v + L) \) 范围内的邻居
- \( A_i \) 是单元 \( i \) 的资源需求（反映打包难度）
- \( C_v \) 是该范围内的 CLB slice 数量
- \( L = 5 \)（经验值）

> **公式解读**：\( U_v \) 衡量了单元 \( v \) 附近区域的"拥挤程度"。\( U_v > 1 \) 意味着该区域的资源需求超过供给，需要膨胀单元面积来推开它们。关键创新是**对 LUT 和 FF 分别计算利用率**——这解决了"总面积不溢出但 FF 溢出"的问题。

### 4.3 面积更新规则

\[
a_v = \begin{cases}
\min(a_v^+, A_v \cdot U_v \cdot \nu_v), & \text{if } A_v \cdot U_v \cdot \nu_v > a_v \\[4pt]
\max(a_v^-, A_v \cdot U_v \cdot \nu_v), & \text{if } A_v \cdot U_v \cdot \nu_v < a_v \text{ and } R_v < R_{\max} \\[4pt]
a_v, & \text{otherwise}
\end{cases}
\]

其中：
- \( a_v \) 是当前面积，\( a_v^+ = a_v \cdot 1.1 \)（膨胀率），\( a_v^- = a_v \cdot 0.95 \)（收缩率）
- \( \nu_v \geq 1 \) 是累积布线膨胀比
- \( R_v \) 是局部布线利用率，\( R_{\max} = 0.65 \)

> **三种情况的含义**：
> - **情况 1（膨胀）**：当估计的资源需求超过当前面积时，增大面积。上限为 \( a_v^+ \) 确保平滑增长
> - **情况 2（收缩）**：当区域资源充足且布线不拥塞时，缩小面积让其他单元进入
> - **情况 3（保持）**：当布线拥塞（\( R_v \geq R_{\max} \)）时，即使资源充足也不收缩——避免进一步加重布线压力

### 4.4 LUT 资源需求估计

每个 LUT 的资源需求基于其与邻居的架构兼容性：

\[
A_v^{(\text{LUT})} = \frac{|N_v'|}{|N_v^+|} \cdot \frac{1}{16} + \frac{|N_v^+| - |N_v'|}{|N_v^+|} \cdot \frac{1}{8}
\]

其中 \( N_v' \) 是 \( N_v^+ \) 中与 \( v \) 架构兼容（可放入同一 BLE）的 LUT 集合。

> **公式解读**：在 UltraScale 中，每个 BLE 可容纳 2 个兼容 LUT。如果 \( v \) 的大部分邻居都与之兼容，\( |N_v'|/|N_v^+| \) 接近 1，需求为 1/16 CLB（紧凑打包，两个 LUT 共享一个 BLE）；如果很少有兼容邻居，需求为 1/8 CLB（独占一个 BLE）。这是一个**概率估计**，不需要实际的 Packing 解。

### 4.5 FF 资源需求估计

FF 的需求估计更复杂，因为控制集规则限制了哪些 FF 可以共处同一个 Half CLB。本文采用**最紧打包下界估计**：

\[
A_v^{(\text{FF})} = \frac{1}{2} \cdot \left\lceil \frac{\sum_{i=0}^{m} \lceil n_i / 4 \rceil}{2} \right\rceil \cdot \frac{1}{n_0} \cdot \beta^{(\text{FF})}
\]

其中：
- FF \( v \) 的控制集为 \( (\text{CK}_0, \text{SR}_0, \text{CE}_0) \)
- \( \{\text{CE}_0, \text{CE}_1, \ldots, \text{CE}_m\} \) 是邻居 \( N_v^+ \) 中与 \( v \) 共享 \( (\text{CK}_0, \text{SR}_0) \) 的 FF 的不同 CE 集合
- \( n_i \) 是控制集为 \( (\text{CK}_0, \text{SR}_0, \text{CE}_i) \) 的 FF 数量
- \( \beta^{(\text{FF})} = 1.1 \)，补偿最紧打包与实际打包之间的差距

> **推导思路**：4 个共享同一 (CK, SR, CE) 的 FF 组成一个"quarter CLB"，2 个共享同一 (CK, SR) 的 quarter CLB 组成一个"half CLB"。先计算每种 CE 变体需要多少 quarter CLB（\( \lceil n_i/4 \rceil \)），再计算总共需要多少 half CLB（\( \lceil \text{sum}/2 \rceil \)），最后将需求均摊到每个 FF。FF 需求范围 [1/16, 1/2]，差异可达 **8 倍**——这正是传统方法忽略的关键方差来源。

---

## 5. 创新点二：完全可并行的直接合法化（DL）

### 5.1 大学录取问题类比

DL 算法受 **Gale-Shapley 大学录取问题** 启发：

```
大学录取问题                    直接合法化
──────────────────────────────────────────
大学 (College)                 CLB Slice
学生 (Student)                 LUT/FF 单元
朋友 (Friends)                 共享网络的单元
大学录取学生                   Slice 接受 BLE 聚类
学生选择大学                   单元选择 Slice
```

关键差异：
1. 单元对 Slice 的偏好取决于其**连接单元的选择**（朋友也去了哪所"大学"）
2. 某些单元**不能同处一个 Slice**（架构合法性约束）

### 5.2 节点中心算法

每个 Slice 作为一个**计算节点**独立运行，维护以下数据结构：

| 数据结构 | 含义 |
|---------|------|
| `det` | 已确定分配到该 Slice 的单元集合 |
| `pq` | 优先队列，存储当前 Top-K 候选聚类 |
| `nbr` | 邻居单元集合（距离 ≤ d 的单元） |
| `scl` | 种子聚类集合（用于生成新候选） |
| `i` | 自上次最优候选变化以来的迭代数 |

```mermaid
graph TD
    A["检查 pq.top()<br/>是否稳定且被接受?"] -->|Yes| B["提交到 det<br/>重置 pq"]
    A -->|No| C["移除无效候选/单元"]
    C --> D{"邻居太少<br/>且 d < D?"}
    D -->|Yes| E["扩展搜索范围<br/>d += Δd<br/>添加更多邻居"]
    D -->|No| F["生成新候选<br/>nbr × scl → pq"]
    E --> F
    F --> G["广播 pq.top()<br/>给涉及单元"]
    G --> H["单元选择<br/>最高 SCORE 的 Slice"]
```

### 5.3 评分函数

给定 Slice \( s \) 和聚类 \( c \)：

\[
\text{SCORE}(c, s) = \sum_{e \in \text{Net}(c)} \frac{\text{InternalPins}(e, c)}{\text{TotalPins}(e)} - \frac{1}{\eta} \cdot \text{HPWL}(c, s)
\]

其中：
- \( \text{Net}(c) \)：至少包含聚类 \( c \) 中一个单元的网络集合
- \( \text{InternalPins}(e, c) \)：网络 \( e \) 在聚类 \( c \) 内的引脚数
- \( \text{TotalPins}(e) \)：网络 \( e \) 的总引脚数
- \( \text{HPWL}(c, s) \)：将聚类 \( c \) 中的单元从 FIP 位置移到 Slice \( s \) 位置的线长增量
- \( \eta = 0.02 \)

> **评分函数的双重含义**：
>
> **第一项（Packing 质量）**：\( \text{InternalPins}/\text{TotalPins} \) 衡量了聚类将外部网络"吸收"为内部网络的比例。高比例 = 更多网络被封装在 CLB 内部 = 更少的 CLB 间布线需求 = 更好的可布线性。这是 Packing 阶段优化目标的直接体现。
>
> **第二项（Placement 质量）**：线长增量越小 = 从 FIP 移动的代价越小 = 更好地保持 FIP 质量。

### 5.4 并行化与串行等价性

**并行化**：所有 Slice 计算节点可以**完全并行**执行（Algorithm 1 line 9-11），所有单元也可以并行处理 offer 并返回决策（line 12-16）。

**串行等价性（Serial Equivalency）**：这是本文的一个重要理论贡献——保证无论使用多少线程，算法产生**完全相同**的解。关键机制是 Lemma 1：

> **Lemma 1**：如果在同一 DL 迭代中，任意两个计算节点提供相同的 SCORE 改进，则通过基于 Slice 唯一标识符的确定性 tie-breaking，可以保证算法收敛。

> **为什么串行等价性重要？** 它意味着可以放心使用尽可能多的并行资源（甚至 GPU/FPGA 加速），而不会牺牲结果质量。这与模拟退火等随机算法形成鲜明对比——后者通常在并行化时结果不一致。

### 5.5 后处理：异常处理

DL 完成后，通常有 < 1% 的单元无法在位移约束 \( D \) 内找到合法位置。处理策略是**拆解重分配（Rip-up and Reallocate）**：

\[
\text{SCORE}_{\text{ripup}}(v, s, c) = -\frac{1}{\eta_1} \cdot \text{HPWL}(v, s) - \eta_2 \cdot \text{SCORE}(c, s) - \eta_3 \cdot \text{Area}(c)
\]

> 选择拆解哪个 Slice 的原则：线长代价小 + 原聚类评分低 + 原聚类面积小（面积大意味着单元多或难打包，拆解后难以重新合法化）。参数 \( \eta_1 = 0.02, \eta_2 = 1.0, \eta_3 = 4.0 \)。

---

## 6. 算法流程

```mermaid
graph TD
    A["<b>输入</b><br/>LUT/FF 网表<br/>FPGA 架构"] --> B["<b>扁平初始布局 FIP</b><br/>二次布局 + 粗合法化<br/>+ 动态面积调整 DAA"]
    B --> C{"<b>收敛?</b>"}
    C -->|No| B
    C -->|Yes| D["<b>并行直接合法化 DL</b><br/>Gale-Shapley 式<br/>Slice↔BLE 双向选择<br/>同时满足 Packing+Placement 合法性"]
    D --> E["<b>后处理</b><br/>拆解重分配<br/>(< 1% 单元)"]
    E --> F["<b>详细布局</b><br/>局部优化"]
    F --> G["<b>输出</b><br/>合法布局"]
```

### 运行时间分布

| 阶段 | 占比 | 说明 |
|------|------|------|
| 二次布局 (QP + 粗合法化) | 64.5% | 主要时间消耗 |
| 详细布局 | 18.5% | — |
| 直接合法化 (DL) | 12.5% | 16 线程并行 |
| 动态面积调整 (DAA) | 2.2% | 开销极小 |
| 后处理异常处理 | 0.7% | — |

---

## 7. 实验结果与分析

### 7.1 实验设置

| 项目 | 配置 |
|------|------|
| **CPU** | Intel Core i9-7900X (3.30 GHz, 10 核, 13.75 MB L3) |
| **内存** | 128 GB RAM |
| **基准** | ISPD 2016 (12 个设计) + ISPD 2017 (13 个设计) |
| **FPGA** | Xilinx UltraScale VU095 (67K CLB slices) |
| **布线器** | Xilinx Vivado v2015.4 (2016) / v2016.4 (2017) |
| **并行** | DL 使用 16 线程 (OpenMP)，其余单线程 |

### 7.2 消融实验（ISPD 2016）

| 方法 | Routed WL (归一化) | 运行时间 (归一化) |
|------|-------------------|-----------------|
| UTPlaceF（原始） | **1.044** | 1.24 |
| UTPlaceF + DAA | 1.030 | 1.29 |
| DAA + 贪心 DL | 1.060 | 0.93 |
| **Proposed (DAA + DL)** | **1.000** | **1.00** |

> **关键发现**：
> - DAA 单独带来 1.4% 的 WL 改善（FIP 更接近合法解）
> - DL 比 UTPlaceF 的 Packing+合法化流程好 3.0%
> - DL 比贪心 DL 好 6.0%（并行探索更大解空间的价值）
> - 整体改善 4.4%，**同时运行更快**（1.24× 加速）

### 7.3 FIP 与合法解的位移对比

| 方法 | 平均位移 | 最大位移 |
|------|---------|---------|
| UTPlaceF | 21.4 | 162.5 |
| UTPlaceF + DAA | 11.7 | 135.0 |
| DAA + 贪心 DL | 1.5 | 57.4 |
| **Proposed (DAA + DL)** | **1.2** | **11.6** |

> **Proposed 方法将最大位移从 162.5 降到 11.6**（恰好低于预设的位移约束 D = 12）。这意味着 FIP 中优化的指标（线长、时序、可布线性）在合法化后几乎完全保留。

### 7.4 难打包设计的突出表现

| 设计 | 控制集数 | vs UTPlaceF WL 改善 | 说明 |
|------|---------|-------------------|------|
| FPGA-10 | 2541 | **-29.5%** | 控制集最多，传统 Packing 最难 |
| FPGA-06 | 2541 | -12.1% | |
| FPGA-07 | 2541 | -5.6% | |
| FPGA-09 | 1281 | -3.7% | |

> **核心洞察**：传统流程在控制集密集的设计上表现差，因为 Packing 阶段无法预见 CLB 的物理位置，导致大量位移。本文方法通过在合法化过程中同时考虑 Packing 和 Placement，特别适合这类**难打包设计**。

### 7.5 与其他学术布局器的对比（ISPD 2016）

| 对比方法 | Routed WL (归一化) | 运行时间 (归一化) |
|---------|-------------------|-----------------|
| ISPD 2016 第 1 名 | 1.080 | 3.96 |
| ISPD 2016 第 2 名 | 1.140 | 4.90 |
| ISPD 2016 第 3 名 | 1.444 | 5.84 |
| GPlace | 1.254 | 0.88 |
| RippleFPGA | 1.041 | 0.64 |
| **Proposed** | **1.000** | **1.00** |

> 比 ISPD 2016 冠军好 8.0%，比 RippleFPGA 好 4.1%。RippleFPGA 是最接近的竞争者——它采用了类似的方法论（Place-SemiPack-Legalize），但使用静态面积和贪心合法化。

### 7.6 DL 并行化扩展性

| 线程数 | 加速比 |
|--------|-------|
| 1 | 1.00× |
| 2 | 1.65× |
| 4 | 3.15× |
| 8 | 6.19× |
| 16 (超线程) | 8.68× |

> 近线性扩展到 8 线程，16 线程时因超线程资源竞争开始饱和。

---

## 8. 创新点深度分析

### 8.1 创新点一：消除显式 Packing —— 从"先打包再放置"到"放置中自然打包"

**本质**：将 FPGA CAD 中分离了二十多年的两个阶段合并为一个统一过程。

**为什么此前没人做到？**

1. **思维惯性**：经典 VPR 框架（Betz, Rose, 1997）将 Packing 和 Placement 定义为两个独立阶段，后续所有工作都在这个框架内改进
2. **技术障碍**：没有 DAA，FIP 解与合法解之间偏差太大，直接合法化会产生极差的结果。DAA 是使 Place-Legalize 流程可行的**必要前提**
3. **合法化算法瓶颈**：传统 Tetris 式贪心合法化一次只考虑一个单元，解空间极窄。本文的 Gale-Shapley 式并行合法化使解空间探索成为可能

**本文的关键洞察**：不是"如何做更好的 Packing"，而是"Packing 是否真的需要作为独立阶段存在？"。答案是否定的——如果 FIP 已经考虑了 Packing 效应（通过 DAA），且合法化算法足够强（通过 DL），Packing 就会自然涌现。

### 8.2 创新点二：DAA —— 将 Packing 信息隐式注入 FIP

**巧妙之处**：DAA 不直接求解 Packing 问题，而是通过**调整单元面积**间接影响 FIP 的密度分布。

- 面积大的单元 → 在粗合法化中被推开更远 → 自然占据更多空间 → 反映其"难打包"的特性
- 不同类型（LUT/FF）分别计算利用率 → 解决资源类型不平衡
- 布线拥塞区域禁止面积收缩 → 维护可布线性

> **反直觉的发现**：DAA 使 FIP 的 HPWL **增大了 2.5%**（Table VI），但最终 routed wirelength **改善了 1.4%**。这说明更大的 FIP HPWL 不是质量退化——而是 FIP 更接近合法解的信号。传统流程中 FIP HPWL 偏小是因为它忽略了 Packing 效应，看似好实则不可实现。

### 8.3 创新点三：Gale-Shapley 式并行合法化

**与经典 Gale-Shapley 的差异**：

| 维度 | 经典 Gale-Shapley | 本文 DL |
|------|-------------------|---------|
| 对象偏好 | 固定且独立 | 取决于"朋友"的选择 |
| 约束 | 仅容量 | 容量 + 架构合法性（控制集兼容） |
| 解空间 | 每个"学生"去一个"大学" | 聚类级别的绑定 |
| 收敛保证 | 经典理论 | Lemma 1（确定性 tie-breaking） |

**串行等价性**的工程价值极高——它允许用户在任何并行度下运行，总是得到相同结果。这是工业工具的基本要求（可复现性），也是后续 GPU/FPGA 加速实现的基础。

### 8.4 与前三天论文的方法论对比

| 维度 | ePlace/RePlAce/DREAMPlace | 本文 |
|------|--------------------------|------|
| **基础框架** | 静电模型 + Nesterov 优化 | 二次布局 + 粗合法化 |
| **密度处理** | Poisson 方程 + FFT | 动态面积调整 + 粗合法化 |
| **合法化** | 不涉及（ASIC 标准单元简单对齐） | **核心创新**（FPGA 架构约束极复杂） |
| **优化目标** | 线长 + 密度 | 线长 + Packing 质量 + 位移约束 |
| **并行化** | GPU 张量运算 | CPU 多线程 + 串行等价性 |

> ASIC 布局与 FPGA 布局的根本差异在于**合法化的复杂度**：ASIC 中标准单元只需要对齐到 site row，而 FPGA 中还需要满足 BLE/CLB 的架构约束（控制集兼容、容量限制）。这导致 FPGA 的合法化本质上是**装箱+分配的组合问题**，需要全新的算法思路。

---

## 9. 参考文献

1. W. Li and D. Z. Pan, "A New Paradigm for FPGA Placement without Explicit Packing," *IEEE Trans. Computer-Aided Design (TCAD)*, vol. 38, no. 11, pp. 2113–2126, 2019. DOI: [10.1109/TCAD.2018.2877017](https://doi.org/10.1109/TCAD.2018.2877017)

2. W. Li, S. Dhar, and D. Z. Pan, "UTPlaceF: A Routability-Driven FPGA Placer with Physical and Congestion Aware Packing," *IEEE TCAD*, 2017.

3. G. Chen et al., "RippleFPGA: Routability-Driven Simultaneous Packing and Placement for Modern FPGAs," *IEEE TCAD*, 2017.

4. D. Gale and L. S. Shapley, "College Admissions and the Stability of Marriage," *The American Mathematical Monthly*, vol. 69, no. 1, pp. 9–15, 1962.

5. V. Betz and J. Rose, "VPR: A New Packing, Placement and Routing Tool for FPGA Research," in *FPL*, 1997.

6. J. Lu et al., "ePlace: Electrostatics-Based Placement Using Fast Fourier Transform and Nesterov's Method," *ACM TODAES*, vol. 20, no. 2, 2015.

7. Y. Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement," in *Proc. DAC*, 2019.

---

*本文档由 Claude Code 于 2026-06-08 基于论文全文重写，作为 EDA 论文每日分析系列的第 4 天内容。与前三天 ASIC 布局论文不同，本文展示了 FPGA 布局的独特挑战——架构约束使得合法化成为核心难题，而 Gale-Shapley 式并行算法提供了一种优雅的解决方案。*
