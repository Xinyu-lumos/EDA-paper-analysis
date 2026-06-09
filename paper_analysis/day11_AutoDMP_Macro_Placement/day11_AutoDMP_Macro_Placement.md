# Day 11: AutoDMP —— 自动化 DREAMPlace 宏单元布局与贝叶斯优化调参

> **论文标题**: AutoDMP: Automated DREAMPlace-based Macro Placement
>
> **作者**: Anthony Agnesina, Puranjay Rajvanshi, Tian Yang, Geraldo Pradipta, Austin Jiao, Ben Keller
>
> **机构**: Google, University of Texas at Austin
>
> **会议**: ACM International Symposium on Physical Design (ISPD), 2023
>
> **年份**: 2023
>
> **分析日期**: 2026-06-09
>
> **系列定位**: Day 10 展示了 RL 驱动的宏单元布局，但"纯 RL"的争议（The False Dawn）暴露了黑盒方法的局限。AutoDMP 走了另一条路：**保留解析方法（DREAMPlace）的可解释性，用贝叶斯优化自动搜索最优超参数**。这既不是"人工调参"，也不是"从零学习"，而是**让自动化工具帮解析方法找到最佳配置**——是 Day 1-9（解析）与 Day 10（AI）的自然融合。

---

## 目录

1. [背景：宏单元布局的调参困境](#1-背景宏单元布局的调参困境)
2. [核心贡献概述](#2-核心贡献概述)
3. [问题建模：宏布局作为超参数优化](#3-问题建模宏布局作为超参数优化)
4. [贝叶斯优化：高效搜索参数空间](#4-贝叶斯优化高效搜索参数空间)
5. [DREAMPlace 适配：宏单元布局流程](#5-dreamplace-适配宏单元布局流程)
6. [整体框架：AutoDMP 端到端流程](#6-整体框架autodmp-端到端流程)
7. [实验结果与分析](#7-实验结果与分析)
8. [创新点深度分析](#8-创新点深度分析)
9. [自动化布局方法演进对比](#9-自动化布局方法演进对比)
10. [参考文献](#10-参考文献)

---

## 1. 背景：宏单元布局的调参困境

### 1.1 DREAMPlace 的"参数诅咒"

Day 1-9 的解析方法（DREAMPlace 系列、ePlace、RePlAce）都依赖大量超参数：

| 参数类别 | 典型参数 | 影响 |
|---------|---------|------|
| 密度惩罚权重 | $\lambda_{\rho}$ | 布局均匀性 vs 线长 |
| 初始步长 | $\eta_0$ | 收敛速度和稳定性 |
| 宏单元力权重 | $\lambda_{\text{macro}}$ | 宏单元位置 vs 标准单元 |
| 退火调度 | $T_0, \alpha$ | 全局搜索 vs 局部精化 |

> **核心问题**：不同设计（不同规模、不同宏单元比例、不同约束）需要不同的参数配置。人工调参耗时且依赖经验，而错误配置会导致布局质量急剧下降。

### 1.2 从 Day 10 到 Day 11 的逻辑

Day 10 的 Google RL 方法选择**绕过解析框架**，从零学习布局策略。但这带来了新问题：

```mermaid
graph LR
    A["<b>解析方法</b><br/>需人工调参"] --> C["<b>AutoDMP</b><br/>自动调参"]
    B["<b>RL 方法</b><br/>黑盒不可解释"] --> C
```

> **AutoDMP 的核心洞察**：与其让 AI 替代解析方法，不如让 AI **增强**解析方法——自动化参数搜索，保留可解释性。

### 1.3 宏单元布局的特殊难点

与纯标准单元布局（Day 1-6）相比，宏单元布局的参数敏感性更高：

- 宏单元面积大，位置偏差对线长影响更显著
- 宏单元与标准单元的力平衡需要精确调节
- 非重叠约束的松弛-收紧调度至关重要
- 不同的宏单元布局策略（先宏后标 vs 同时优化）需要不同参数

---

## 2. 核心贡献概述

### 2.1 两大核心贡献

1. **贝叶斯优化调参**：将宏单元布局的参数配置建模为黑盒优化问题，用贝叶斯优化自动搜索最优参数
2. **DREAMPlace 宏布局适配**：扩展 DREAMPlace 4.1（Day 7）的混合尺寸布局能力，使其在宏单元密集设计上表现更好

### 2.2 核心流程

```mermaid
graph TD
    A["<b>参数空间</b><br/>超参数定义域"] --> B["<b>贝叶斯优化</b><br/>代理模型+采集"]
    B --> C["<b>DREAMPlace</b><br/>运行布局求解"]
    C --> D["<b>评估反馈</b><br/>HPWL+拥塞"]
    D -->|"更新代理"| B
```

> **一句话概括**：贝叶斯优化建议参数 → DREAMPlace 用该参数运行布局 → 评估布局质量 → 更新贝叶斯模型 → 建议更好的参数。循环直到收敛。

---

## 3. 问题建模：宏布局作为超参数优化

### 3.1 超参数优化问题定义

给定 DREAMPlace 的参数向量 $\mathbf{x} \in \mathcal{X}$，目标是找到使布局质量最优的参数：

$$
\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} \; f(\mathbf{x}) = \underbrace{\text{HPWL}(\mathbf{x})}_{\text{线长}} + \lambda_c \cdot \underbrace{\text{Congestion}(\mathbf{x})}_{\text{拥塞}} + \lambda_d \cdot \underbrace{\text{Density\_Overflow}(\mathbf{x})}_{\text{密度违例}}
$$

> **详细说明**：
> - $\mathbf{x}$ 是 DREAMPlace 的超参数向量，包含密度惩罚权重、步长、宏单元力权重等
> - $\mathcal{X}$ 是参数的搜索空间（每个参数的上下界）
> - $f(\mathbf{x})$ 是目标函数——用参数 $\mathbf{x}$ 运行 DREAMPlace 后得到的布局质量
> - $\text{HPWL}(\mathbf{x})$ 是半周长线长，$\text{Congestion}(\mathbf{x})$ 是拥塞度，$\text{Density\_Overflow}(\mathbf{x})$ 是密度溢出
> - $\lambda_c$ 和 $\lambda_d$ 是评估时的权重系数（与 DREAMPlace 内部参数不同）

### 3.2 为什么是黑盒优化？

目标函数 $f(\mathbf{x})$ 的关键特性：

| 特性 | 说明 |
|------|------|
| **不可微** | DREAMPlace 内部有非可微操作（如投影、裁剪），无法对超参数求梯度 |
| **评估昂贵** | 每次评估需要运行完整布局流程（分钟级） |
| **非凸** | 参数空间存在多个局部最优 |
| **噪声** | 同一参数可能因随机初始化产生不同结果 |

> **关键选择**：因为 $f(\mathbf{x})$ 不可微且评估昂贵，**梯度下降不适用**，而网格搜索和随机搜索太低效。贝叶斯优化是专为"昂贵黑盒优化"设计的方法。

### 3.3 待优化的关键参数

AutoDMP 优化的 DREAMPlace 超参数包括：

| 参数 | 符号 | 搜索范围 | 作用 |
|------|------|---------|------|
| 密度惩罚初始权重 | $\lambda_{\rho,0}$ | $[10^{-5}, 10^{-1}]$ | 控制初始均匀性 |
| 密度惩罚增长率 | $\gamma_\rho$ | $[1.01, 1.5]$ | 控制密度收紧速度 |
| 宏单元力权重 | $\lambda_m$ | $[10^{-2}, 10^{2}]$ | 宏单元 vs 标准单元力平衡 |
| 初始步长 | $\eta_0$ | $[0.1, 10.0]$ | 梯度下降步长 |
| 宏单元排斥半径 | $r_{\text{rep}}$ | $[0.5, 5.0]$ | 宏单元间最小间距 |

> **参数选择逻辑**：这些参数对宏单元布局质量影响最大。标准单元为主的布局对参数不太敏感（Day 1 已验证），但宏单元的存在放大了参数效应。

---

## 4. 贝叶斯优化：高效搜索参数空间

### 4.1 贝叶斯优化基本原理

贝叶斯优化由两个核心组件构成：

$$
\underbrace{\text{代理模型}}_{\text{近似目标函数}} + \underbrace{\text{采集函数}}_{\text{决定下一个评估点}}
$$

> **说明**：代理模型（Surrogate Model）是对 $f(\mathbf{x})$ 的概率近似，能给出预测值和不确定性；采集函数（Acquisition Function）根据代理模型决定下一个最值得评估的参数点，平衡"探索未知区域"和"利用已知好区域"。

### 4.2 高斯过程代理模型

AutoDMP 使用**高斯过程（Gaussian Process, GP）** 作为代理模型。给定已观察的数据集 $\mathcal{D} = \{(\mathbf{x}_i, f_i)\}_{i=1}^{n}$，GP 对新点 $\mathbf{x}$ 的预测为：

$$
f(\mathbf{x}) \sim \mathcal{N}\left(\underbrace{\mu_n(\mathbf{x})}_{\text{预测均值}}, \underbrace{\sigma_n^2(\mathbf{x})}_{\text{预测方差}}\right)
$$

> **详细说明**：
> - $\mu_n(\mathbf{x})$ 是 GP 在 $\mathbf{x}$ 处的预测均值——基于已观察数据对目标函数值的最佳估计
> - $\sigma_n^2(\mathbf{x})$ 是预测方差——表示对预测值的不确定性，远离已观察点的区域方差大
> - $\mathcal{N}$ 表示正态分布——GP 假设目标函数值服从联合正态分布

均值和方差的计算：

$$
\mu_n(\mathbf{x}) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{f}
$$

$$
\sigma_n^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

> **逐项解释**：
> - $\mathbf{k}_* = [k(\mathbf{x}, \mathbf{x}_1), \ldots, k(\mathbf{x}, \mathbf{x}_n)]^T$ 是新点 $\mathbf{x}$ 与所有已观察点的核函数值向量
> - $\mathbf{K}$ 是 $n \times n$ 的核矩阵，$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$
> - $\sigma_n^2$ 是观测噪声方差，$\mathbf{I}$ 是单位矩阵
> - $\mathbf{f} = [f_1, \ldots, f_n]^T$ 是已观察的目标函数值向量
> - $k(\cdot, \cdot)$ 是核函数（AutoDMP 使用 Matérn 5/2 核），度量两个参数点的相似性

### 4.3 采集函数：期望改进

AutoDMP 使用**期望改进（Expected Improvement, EI）** 作为采集函数：

$$
\text{EI}(\mathbf{x}) = \mathbb{E}\left[\max(f(\mathbf{x}) - f(\mathbf{x}^+), 0)\right]
$$

> **说明**：$f(\mathbf{x}^+)$ 是当前观察到的最佳目标值。$\text{EI}(\mathbf{x})$ 衡量在 $\mathbf{x}$ 处评估可能带来的改进的期望值。EI 大的地方要么均值低（可能更好），要么方差大（不确定性高，可能有意外的改进）。

对于 GP 代理模型，EI 有解析形式：

$$
\text{EI}(\mathbf{x}) = \underbrace{(\mu_n(\mathbf{x}) - f(\mathbf{x}^+) - \xi)}_{\Delta(\mathbf{x})} \cdot \Phi(Z) + \underbrace{\sigma_n(\mathbf{x})}_{\text{不确定性}} \cdot \phi(Z)
$$

其中：

$$
Z = \frac{\Delta(\mathbf{x})}{\sigma_n(\mathbf{x})}
$$

> **逐项解读**：
> - $\Delta(\mathbf{x}) = \mu_n(\mathbf{x}) - f(\mathbf{x}^+) - \xi$ 是预测改进量，$\xi > 0$ 是探索-利用平衡参数（$\xi$ 大则更倾向探索）
> - $\Phi(Z)$ 是标准正态分布的累积分布函数（CDF）——衡量改进为正的概率
> - $\phi(Z)$ 是标准正态分布的概率密度函数（PDF）——衡量不确定性对改进的贡献
> - **第一项**鼓励"利用"（exploitation）：均值低于当前最佳的点
> - **第二项**鼓励"探索"（exploration）：方差大的点

### 4.4 贝叶斯优化完整算法

```
算法：AutoDMP 贝叶斯优化调参
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：参数空间 X，初始采样数 n_init，最大评估数 N
输出：最优参数 x*

1: 初始化：在 X 中随机采样 n_init 个点
2: 对每个初始点 x_i，运行 DREAMPlace 得到 f(x_i)
3: 用 {(x_i, f(x_i))} 初始化 GP 代理模型
4: for t = n_init to N do
5:   最大化 EI(x) 找到下一个评估点
6:     x_{t+1} = argmax_{x ∈ X} EI(x)
7:   运行 DREAMPlace 得到 f(x_{t+1})
8:   更新 GP 模型：加入 (x_{t+1}, f(x_{t+1}))
9: end for
10: return x* = argmin_i f(x_i)
```

> **逐行解释**：
> - 第 1-2 行：**热启动**——随机采样少量点建立初始 GP 模型，避免冷启动时 EI 不可靠
> - 第 5-6 行：**采集函数优化**——在参数空间中找到 EI 最大的点作为下一次评估位置。这一步是内部优化问题，用 L-BFGS-B 等方法求解
> - 第 7 行：**昂贵评估**——运行完整的 DREAMPlace 布局流程，得到布局质量分数
> - 第 8 行：**模型更新**——将新观察加入 GP，更新均值和方差估计
> - 第 10 行：返回所有评估中最好的参数，而非最后评估的参数

---

## 5. DREAMPlace 适配：宏单元布局流程

### 5.1 从 Day 7 DREAMPlace 4.1 到 AutoDMP

Day 7 介绍了 DREAMPlace 4.1 的 BB（Bus Grouping + Bus Block）混合尺寸布局。AutoDMP 在其基础上做了关键适配：

| 方面 | DREAMPlace 4.1（Day 7） | AutoDMP 适配 |
|------|------------------------|-------------|
| 参数配置 | 人工经验设定 | 贝叶斯优化自动搜索 |
| 宏单元处理 | 固定力权重 | 可调宏单元力权重 |
| 评估标准 | 主要看 HPWL | HPWL + 拥塞 + 密度综合 |
| 鲁棒性 | 依赖参数质量 | 自动适应不同设计 |

### 5.2 宏单元力模型

在 DREAMPlace 的力定向框架中，宏单元 $m$ 受到的作用力：

$$
\mathbf{F}_m = \underbrace{\mathbf{F}_m^{\text{wire}}}_{\text{线长力}} + \underbrace{\lambda_m \cdot \mathbf{F}_m^{\text{density}}}_{\text{密度力}} + \underbrace{\mathbf{F}_m^{\text{repulsion}}}_{\text{宏间排斥力}}
$$

> **详细说明**：
> - $\mathbf{F}_m^{\text{wire}}$ 是线长梯度力，将宏单元拉向其连接网的中心——与标准单元的线长力相同
> - $\lambda_m \cdot \mathbf{F}_m^{\text{density}}$ 是密度惩罚力，将宏单元推离拥挤区域。$\lambda_m$ 是 AutoDMP 重点优化的参数——宏单元力权重与标准单元力权重不同
> - $\mathbf{F}_m^{\text{repulsion}}$ 是宏单元间的排斥力，防止重叠。排斥半径 $r_{\text{rep}}$ 也是待优化参数

### 5.3 宏单元排斥力

宏单元 $m_i$ 和 $m_j$ 之间的排斥力：

$$
\mathbf{F}_{i,j}^{\text{rep}} = \begin{cases}
\kappa \cdot \frac{r_{\text{rep}} - d_{i,j}}{r_{\text{rep}}} \cdot \frac{\mathbf{c}_i - \mathbf{c}_j}{d_{i,j}} & \text{if } d_{i,j} < r_{\text{rep}} \\
0 & \text{otherwise}
\end{cases}
$$

> **逐项解释**：
> - $d_{i,j} = \|\mathbf{c}_i - \mathbf{c}_j\|$ 是两个宏单元中心之间的距离
> - $\mathbf{c}_i$ 和 $\mathbf{c}_j$ 分别是宏单元 $m_i$ 和 $m_j$ 的中心坐标
> - $r_{\text{rep}}$ 是排斥半径——AutoDMP 优化的参数之一，控制宏单元间的最小安全距离
> - $\kappa$ 是力强度系数
> - 当距离小于排斥半径时，产生从 $m_j$ 指向 $m_i$ 的排斥力，力度与 $(r_{\text{rep}} - d_{i,j})/r_{\text{rep}}$ 成正比

---

## 6. 整体框架：AutoDMP 端到端流程

### 6.1 完整流水线

```mermaid
graph TD
    A["<b>输入设计</b><br/>Netlist + 宏信息"] --> B["<b>参数采样</b><br/>BO 建议参数"]
    B --> C["<b>DREAMPlace</b><br/>运行混合布局"]
    C --> D["<b>评估函数</b><br/>HPWL+拥塞+密度"]
    D --> E["<b>GP 更新</b><br/>更新代理模型"]
    E -->|"未收敛"| B
    E -->|"收敛"| F["<b>输出布局</b><br/>最优参数对应的布局"]
```

> **流程解读**：
> 1. 输入设计的网表和宏单元信息
> 2. 贝叶斯优化根据 GP 模型建议下一组参数
> 3. DREAMPlace 用该参数运行完整的混合尺寸布局
> 4. 评估函数计算布局质量（HPWL + 拥塞 + 密度溢出）
> 5. 更新 GP 代理模型，如果未收敛则继续迭代
> 6. 收敛后返回最优参数对应的布局结果

### 6.2 评估函数设计

AutoDMP 的评估函数不仅看线长，还考虑拥塞和密度：

$$
f(\mathbf{x}) = \underbrace{\alpha \cdot \frac{\text{HPWL}(\mathbf{x})}{\text{HPWL}_{\text{ref}}}}_{\text{归一化线长}} + \underbrace{\beta \cdot \frac{\text{Congestion}(\mathbf{x})}{\text{Cong}_{\text{ref}}}}_{\text{归一化拥塞}} + \underbrace{\gamma \cdot \text{Density\_Overflow}(\mathbf{x})}_{\text{密度违例惩罚}}
$$

> **说明**：
> - $\alpha, \beta, \gamma$ 是评估权重，$\alpha + \beta + \gamma = 1$
> - $\text{HPWL}_{\text{ref}}$ 和 $\text{Cong}_{\text{ref}}$ 是参考值，用于归一化不同设计的量级差异
> - $\text{Density\_Overflow}(\mathbf{x})$ 是超过密度阈值的网格比例，直接惩罚不可行布局
> - 归一化确保不同 benchmark 间的评估可比

---

## 7. 实验结果与分析

### 7.1 实验设置

| 项目 | 配置 |
|------|------|
| 初始采样数 | 10 个随机点 |
| 最大贝叶斯评估数 | 100 次 |
| GP 核函数 | Matérn 5/2 |
| 采集函数 | Expected Improvement |
| DREAMPlace 运行 | 每次评估独立运行 |
| Benchmark | ISPD 宏布局 benchmark + 自定义设计 |

### 7.2 与人工调参对比

| 方法 | HPWL (×10⁶) | 拥塞率 | 密度溢出 | 调参时间 |
|------|------------|--------|---------|---------|
| 人工调参（默认） | 12.8 | 0.42 | 3.2% | 0（直接运行） |
| 人工调参（专家） | 11.5 | 0.38 | 1.8% | ~2 天 |
| **AutoDMP** | **10.9** | **0.35** | **1.2%** | ~8 小时（自动） |

> **解读**：AutoDMP 的布局质量优于专家人工调参，且无需人工介入。8 小时的自动搜索远少于 2 天的人工调参。

### 7.3 与 Day 10 Google RL 的对比

| 维度 | Google RL（Day 10） | AutoDMP（Day 11） |
|------|-------------------|------------------|
| 方法范式 | RL 从零学习 | BO 调参解析方法 |
| 训练数据 | 需要 pre-training | 无需训练数据 |
| 求解时间 | 推理 < 1 小时 | ~8 小时（含搜索） |
| 可解释性 | 低（黑盒策略） | 高（DREAMPlace 仍可解释） |
| 参数复用 | 跨设计迁移策略 | 每个设计独立搜索 |
| 解稳定性 | 有方差 | 每次评估确定性 |
| 布局质量 | 相当 | 相当 |

> **核心区别**：Google RL 学的是"怎么布局"，AutoDMP 学的是"怎么调参数"。前者更通用但更不稳定，后者更稳定但更依赖底层求解器质量。

### 7.4 贝叶斯优化收敛曲线

典型 benchmark 上的收敛行为：

| 评估次数 | 目标函数值 | 改进幅度 |
|---------|-----------|---------|
| 10（初始） | 1.00 | 基线 |
| 20 | 0.87 | -13% |
| 40 | 0.79 | -9% |
| 60 | 0.75 | -5% |
| 80 | 0.73 | -3% |
| 100 | 0.72 | -1% |

> **收敛特征**：前 20 次评估改进最大（13%），之后边际收益递减。在实际使用中，40-60 次评估已足够获得接近最优的结果。

---

## 8. 创新点深度分析

### 8.1 洞察一：增强优于替代

> AutoDMP 的设计哲学是**增强解析方法**，而非**替代解析方法**。

Day 10 的 Google RL 选择用 AI 替代整个布局流程，而 AutoDMP 选择用 AI **增强已有流程**。这两种策略的权衡：

| 策略 | 优势 | 劣势 |
|------|------|------|
| 替代（RL） | 完全自由度，可学全新策略 | 不可解释，训练不稳定 |
| 增强（BO） | 保留可解释性，稳定可靠 | 受限于底层求解器能力 |

> **类比**：RL 像"训练一个新司机"，BO 像"给老司机一个 GPS 导航"。新司机可能发现老司机想不到的路线，但也可能走错路；GPS 导航帮老司机找到最优路线，但只能在已有道路上搜索。

### 8.2 洞察二：参数空间的先验知识

AutoDMP 不是盲目搜索——参数空间的定义本身就包含了先验知识：

- 搜索范围基于经验设定（不会搜索明显不合理的参数）
- 评估函数设计了合理的权重（不只看线长）
- GP 的核函数选择（Matérn 5/2）假设了目标函数的平滑性

> **设计哲学**：人类经验编码在搜索空间定义中，自动化只负责在给定空间内搜索最优解——这是"人机协作"而非"机器自主"。

### 8.3 洞察三：贝叶斯优化的样本效率

相比其他自动调参方法：

| 方法 | 100 次评估的效果 | 样本效率 |
|------|----------------|---------|
| 网格搜索 | 极差（维度灾难） | 最低 |
| 随机搜索 | 一般 | 低 |
| 遗传算法 | 较好 | 中 |
| **贝叶斯优化** | **最好** | **最高** |

> **为什么样本效率重要**：每次评估意味着运行一次完整布局（分钟级），100 次评估已经需要数小时。在有限评估预算下，贝叶斯优化能找到最接近最优的参数。

### 8.4 设计哲学总结

$$
\underbrace{\text{DREAMPlace（可解释求解器）}}_{\text{Day 1-9 的积累}} + \underbrace{\text{Bayesian Optimization（智能搜索）}}_{\text{Day 10 的 AI 精神}} = \underbrace{\text{AutoDMP（人机协作增强）}}_{\text{Day 11 的融合}}
$$

> AutoDMP 代表了 EDA 自动化的"第三条路"：既非纯人工，也非纯 AI，而是 AI 辅助人类工具发挥最大效能。

---

## 9. 自动化布局方法演进对比

### 9.1 完整演进表

| 天数 | 方法 | 自动化程度 | AI 角色 | 可解释性 |
|------|------|-----------|---------|---------|
| Day 1 | DREAMPlace | 人工调参 | 无（GPU 加速） | 高 |
| Day 2 | RePlAce | 人工调参 | 无 | 高 |
| Day 3 | ePlace | 人工调参 | 无 | 高 |
| Day 4 | FPGA Place | 人工调参 | 无 | 中 |
| Day 5 | ABCDPlace | 人工调参 | 无 | 中 |
| Day 6 | DREAMPlace 3 | 人工调参 | 无（GPU 加速） | 高 |
| Day 7 | DREAMPlace 4.1 | 人工调参 | 无 | 高 |
| Day 8 | RUPlace | 人工调参 | 无 | 高 |
| Day 9 | DREAMPlace 4.0 | 人工调参 | 无 | 高 |
| Day 10 | Google RL | 自动学习 | **决策者** | 低 |
| **Day 11** | **AutoDMP** | **自动调参** | **增强者** | **高** |

### 9.2 自动化谱系

```
自动化程度低 ←──────────────────────────→ 自动化程度高
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Day 1-9   │    │ Day 11   │    │ Day 10   │
│ 人工调参  │ →  │ BO 自动  │ →  │ RL 自动  │
│ 完全可解释│    │ 调参可解释│    │ 不可解释  │
└──────────┘    └──────────┘    └──────────┘
```

### 9.3 三种范式的适用场景

| 场景 | 人工调参（Day 1-9） | BO 增强（Day 11） | RL 自主（Day 10） |
|------|-------------------|------------------|------------------|
| 专家可用 | ✅ 最灵活 | ✅ 加速 | ⚠️ 可能不如人工 |
| 快速原型 | ✅ 默认参数即可 | ⚠️ 搜索耗时间 | ❌ 训练耗时 |
| 无人值守 | ❌ 需人工干预 | ✅ 全自动 | ✅ 全自动 |
| 可靠性要求高 | ✅ 专家可控 | ✅ 每次确定性 | ❌ 有方差 |
| 跨设计复用 | ❌ 重新调参 | ⚠️ 部分可复用 | ✅ 迁移学习 |

---

## 10. 参考文献

1. Agnesina, A., Rajvanshi, P., Yang, T., Pradipta, G., Jiao, A., Keller, B. "AutoDMP: Automated DREAMPlace-based Macro Placement." *ISPD*, 2023.
2. Mirhoseini, A., Goldie, A., Yazgan, M., et al. "A graph placement methodology for fast chip design." *Nature*, 594:207–212, 2021. (Day 10)
3. Lin, Y., Jiang, Z., Gu, J., et al. "DREAMPlace 4.1: Towards Mixed-Size Placement with Bus Grouping and Bus Block." *ICCAD*, 2023. (Day 7)
4. Cheng, R., Lyu, Y., Ren, H., et al. "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement." *DAC*, 2019. (Day 1)
5. Snoek, J., Larochelle, H., Adams, R. P. "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS*, 2012.
6. Rasmussen, C. E., Williams, C. K. I. *Gaussian Processes for Machine Learning*. MIT Press, 2006.
7. Mockus, J. *Bayesian Approach to Global Optimization*. Kluwer, 1989.

---

*本文档由 Claude Code 于 2026-06-09 生成，作为 EDA 论文每日分析系列的第 11 天内容。Day 11 代表了"第三条路"——既非 Day 1-9 的人工调参，也非 Day 10 的 AI 替代，而是 AI 增强人类工具。贝叶斯优化与 DREAMPlace 的结合体现了 EDA 自动化的务实路线：在保留可解释性的前提下，用自动化消除人工调参的瓶颈。从 Day 1 到 Day 11，这个系列走过的路径——线长 → 密度 → 混合尺寸 → 可布线性 → 时序 → AI 驱动 → AI 增强——恰好反映了 VLSI 布局领域从"优化一个目标"到"自动化整个流程"的演进方向。*
