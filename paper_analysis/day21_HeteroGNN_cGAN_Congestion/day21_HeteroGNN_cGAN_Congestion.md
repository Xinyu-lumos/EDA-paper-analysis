# Day 21: HeteroGNN-embedded cGAN —— 图学习辅助条件 GAN 的 FPGA 布线拥塞预测

> **论文标题**: FPGA Routing Congestion Prediction via Graph Learning-Aided Conditional GAN
>
> **作者**: Qingyu Yang, Jingjin Li, Rui Li, Yuting He, Yajun Ha, Linlin Shen, Ruibin Bai, Heng Yu
>
> **机构**: 上海科技大学 (ShanghaiTech University); 诺丁汉大学宁波分校 (University of Nottingham Ningbo China); 深圳大学 (Shenzhen University)
>
> **期刊**: ACM Transactions on Design Automation of Electronic Systems (TODAES)
>
> **年份**: 2025
>
> **DOI**: [10.1145/3773770](https://doi.org/10.1145/3773770)
>
> **GitHub**: [AIPnR/FPGA_Hetero_Congestion_Prediction](https://github.com/AIPnR/FPGA_Hetero_Congestion_Prediction)
>
> **阅读状态**: 付费墙后（无 arXiv 预印本），基于 GitHub 完整源代码分析
>
> **分析日期**: 2026-06-12
>
> **系列定位**: Day 13（ClusterNet）首次将 GNN 引入拥塞预测；Day 15（RouteNet）用 CNN 将布局视为图像；Day 16（LHNN）提出异构图 GNN 融合几何与拓扑；Day 17（NAS-Routability）实现架构自动搜索；Day 18（cGAN）实现判别式到生成式的范式跃迁；Day 20（CircuitGNN）学习通用电路表示。本文 Day 21 实现了**两条技术路线——异构图 GNN（Day 16）与条件 GAN（Day 18）——的深度融合**。用异构图 GNN 作为 cGAN 的条件编码器，将图学习的拓扑感知能力与生成式模型的学习分布能力结合，在 VTR7 上 RMSE 降低 18.2%，在 Titan23 上 RMSE 降低 15.0%。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心贡献概述](#2-核心贡献概述)
3. [相关工作](#3-相关工作)
4. [问题建模](#4-问题建模)
5. [HeteroGraph 构建](#5-heterograph-构建)
6. [HeteroGNN 双空间消息传递](#6-heterognn-双空间消息传递)
7. [HeteroGNN-embedded cGAN 架构](#7-heterognn-embedded-cgan-架构)
8. [损失函数与训练策略](#8-损失函数与训练策略)
9. [实验设置](#9-实验设置)
10. [实验结果与分析](#10-实验结果与分析)
11. [消融实验](#11-消融实验)
12. [讨论与局限性](#12-讨论与局限性)
13. [创新点深度分析](#13-创新点深度分析)
14. [演进对比表](#14-演进对比表)
15. [参考文献](#15-参考文献)

---

## 1. 背景与动机

### 1.1 FPGA PnR 的拥塞瓶颈

FPGA 布局布线（PnR）流程中，拥塞（congestion）是最关键的反馈信号。如果布局产生了拥塞热点，布线器将无法完成布线或产生过大的延迟，迫使设计者回到布局阶段重新迭代。每一次 PnR 迭代都极其耗时——大型 FPGA 设计可能需要数小时甚至数天。

因此，**在布局完成后、布线执行前**就准确预测拥塞图，可以让设计者及早发现问题区域并调整布局，大幅减少 PnR 迭代次数，加速设计收敛。

### 1.2 CNN 方法的两大结构性局限

已有拥塞预测方法（RouteNet、cGAN 等）普遍采用 CNN 处理**图像化的布局表示**。这种方法存在两个根本性局限：

| 局限 | 说明 | 具体影响 |
|------|------|----------|
| **网表拓扑信息丢失** | 将电路网表转化为图像像素特征（如 RUDY）后，网表的原始连接关系（哪个单元驱动哪个单元）在 CNN 前向传播中不再被显式表示 | 无法学习"单元 A 拥塞 → 其扇出单元 B 也可能拥塞"这种拓扑因果关系 |
| **卷积核感受野有限** | 3×3 或更大卷积核的感受野在几何空间扩展，但电路的连接可能跨越整个芯片 | 一个驱动单元和其被驱动的单元可能在芯片的完全相反的角落，CNN 需要极深的网络才能捕获，但深层网络有梯度消失和特征过平滑问题 |

```mermaid
graph LR
    A["<b>CNN 拥塞预测</b>"] --> B["<b>局限 1</b><br/>网表→图像特征丢失拓扑"]
    A --> C["<b>局限 2</b><br/>卷积核无法捕获<br/>远程拓扑连接"]
    B --> D["<b>本文方案</b><br/>异构图保留网表结构"]
    C --> D
```

### 1.3 核心洞察：图 + 生成的双重优势

本文的关键洞察是将两个前沿方法融合：

1. **异构图 GNN**（来自 Day 16 LHNN 的技术路线）：通过异构图同时编码几何布局和网表拓扑，消息在两种空间中传播
2. **条件 GAN**（来自 Day 18 cGAN 的技术路线）：将拥塞预测建模为条件图像生成，学习完整条件分布而非点估计

> **范式升维**：判别式 GNN 学习 `P(congestion | layout)` 最佳点估计；cGAN 学习 `P(congestion_map | layout_image)` 的条件分布；本文的 **HeteroGNN-cGAN** 学习 `P(congestion_map | HeteroGraph(layout, netlist))`——以图结构为条件的完整条件分布。这结合了图结构性保留（拓扑不丢失）与生成式不确定性建模（分布学习）的双重优势。

---

## 2. 核心贡献概述

```mermaid
graph TD
    A["<b>HeteroGNN-cGAN</b>"] --> B["<b>HeteroGraph</b><br/>网表子图+布局子图<br/>+空间映射边"]
    A --> C["<b>HeteroGNN</b><br/>三空间消息传递<br/>GCN/GAT/GraphSAGE可选"]
    A --> D["<b>cGAN 集成</b><br/>HeteroGNN作为编码器<br/>U-Net生成器+CNN鉴别器"]
    A --> E["<b>多赛道验证</b><br/>VTR7: RMSE降18.2%<br/>Titan23: RMSE降15.0%"]
```

四大核心贡献：

1. **HeteroGraph 异构图建模**：首次将网表拓扑（block-net 子图）、布局几何（grid 子图）和空间映射（几何位置边 + 边界框边）统一为单一异构图，完整保留电路的拓扑和几何信息
2. **HeteroGNN 双空间消息传递**：在网表子图（`topo_in`, `topo_out`）、布局子图（`grid_edge`）和空间映射（`geom_loc`, `geom_bb`）三种语义空间中执行异构图卷积，支持 GCN、GAT、GraphSAGE 三种卷积类型可选
3. **HeteroGNN-embedded cGAN**：将 HeteroGNN 嵌入为 cGAN 的条件编码器，Grid Embeddings 经图像化后送入 U-Net 生成器和 PatchGAN 鉴别器，实现 100x MSE 特征损失 + 对抗损失的联合训练
4. **完整开源实现**：使用 DGL + PyTorch 实现，支持 VTR7 和 Titan23 基准，20 个电路 × 200 布局实例 = 4000 个异构图的完整数据集流水线

---

## 3. 相关工作

### 3.1 与 LHNN（Day 16）的对比

| 维度 | LHNN (Day 16) | HeteroGNN-cGAN (Day 21) |
|------|-------------|----------------------|
| **图模型** | 格超图（Lattice Hypergraph）: 超图+晶格图 | 异构图（HeteroGraph）: 网表子图+布局子图+空间映射边 |
| **节点类型** | G-cell 节点 + Net 超边 | Block节点 + Net节点 + Grid节点 |
| **边类型** | 超边+晶格边（2种） | topo_in + topo_out + grid_edge + geom_loc + geom_bb（5种） |
| **消息传递** | HyperMP + LatticeMP（2种消息传递） | netlist_mp + layout_mp + space_mp（3种消息传递类型） |
| **预测头** | 联合监督：需求回归 + 拥塞分类 | 单一回归：4通道拥塞图（chanx_l4, chany_l4, chanx_l16, chany_l16） |
| **生成式** | 否（纯判别式 GNN） | **是：HeteroGNN 作为 cGAN 编码器** |
| **代码** | 未开源 | **完整开源**（DGL + PyTorch） |
| **F1 提升** | 35% vs U-Net/Pix2Pix | RMSE 降低 18.2% (VTR7) / 15.0% (Titan23) |

**关键差异**：LHNN 是纯粹判别式，通过 GNN 直接回归。Day 21 将 GNN 作为条件生成器的编码器——GNN 学习图嵌入，cGAN 基于嵌入生成拥塞热力图。这与将 GNN 作为"特征提取器"再用 GAN 做"生成器"的思路一致，但 Day 21 是第一个在 EDA 拥塞预测中融合两者的工作。

### 3.2 与 Day 18 cGAN 的对比

| 维度 | cGAN (Day 18) | HeteroGNN-cGAN (Day 21) |
|------|-------------|----------------------|
| **输入表示** | 3通道布局图像 + 单元密度/引脚密度特征 | 异构图（Block/Net/Grid 节点 + 5种边） |
| **条件编码器** | 无（图像直接输入 U-Net） | **HeteroGNN**：先编码为 Graph Embedding，再转为图像 |
| **生成器** | U-Net（ResNet 风格） | U-Net（4×4 卷积，支持 32-512 分辨率） |
| **鉴别器** | PatchGAN 70×70 | PatchGAN（4×4 卷积，动态层数） |
| **损失函数** | cGAN loss + L1 loss | **BCEWithLogits GAN loss + 100×MSE feature loss** |
| **网表信息** | 间接编码（RUDY 等 hand-crafted 特征） | **直接编码**（block-net 异构图边显式表示） |
| **平台** | ASIC | **FPGA** |
| **Benchmark** | ISPD 2015 | **VTR7 + Titan23** |

**关键差异**：Day 18 将布局看作图像直接送入 U-Net，网表信息通过 RUDY 等手工特征间接编码。Day 21 通过异构图将网表拓扑作为**一阶公民**显式表示，HeteroGNN 在网表子图和布局子图之间双向传播消息，使网络可以直接利用"哪个单元驱动哪个单元"的结构信息。

### 3.3 与其他方法的对比

| 方法 | 技术路线 | 输入模态 | 拓扑建模 | 生成式 |
|------|---------|---------|---------|--------|
| RouteNet (Day 15) | CNN | 图像 | 手工特征 | 否 |
| ClusterNet (Day 13) | GNN + 聚类 | 图 | 同构图 | 否 |
| LHNN (Day 16) | 异构图 GNN | 格超图 | 超边 | 否 |
| NAS-Routability (Day 17) | NAS + CNN | 图像 | 手工特征 | 否 |
| cGAN (Day 18) | pix2pix | 图像 | 手工特征 | **是** |
| PROS (Day 19) | CNN | 图像 | 手工特征 | 否 |
| CircuitGNN (Day 20) | GNN | 通用图 | 异构图 | 否 |
| **Day 21 (本文)** | **HeteroGNN + cGAN** | **异构图** | **异构图边直接建模** | **是** |

---

## 4. 问题建模

### 4.1 拥塞预测的数学定义

给定 FPGA 芯片的布局结果（placement），目标是预测布线后的每个布线通道（routing channel）的拥塞情况。

形式上，定义：

- 布局状态：每个 Grid Cell 的类型已确定（空、CLB、DSP、BRAM 等），每个 Block 已分配到特定 Grid Cell
- 布线拥塞：对于每个 Grid Cell $(x, y)$，定义四类拥塞指标：
  - $C_x^{L4}(x, y)$：长度-4 水平布线通道的使用率
  - $C_y^{L4}(x, y)$：长度-4 垂直布线通道的使用率
  - $C_x^{L16}(x, y)$：长度-16 水平布线通道的使用率
  - $C_y^{L16}(x, y)$：长度-16 垂直布线通道的使用率

目标：学习映射函数 $f: \mathcal{G} \to \mathbb{R}^{H \times W \times 4}$，其中 $\mathcal{G}$ 是同时编码网表拓扑和布局几何的异构图。

### 4.2 从异构图到图像的格式转换

HeteroGNN 输出的 Grid Embedding 是 $N \times d$ 的矩阵（$N$ = 网格单元数 = 128×128 = 16384, $d$ = hidden\_feats = 64）。通过 `g_emb2image` 函数将其重塑为 $1 \times d \times 128 \times 128$ 的图像格式，送入 U-Net 生成器。

---

## 5. HeteroGraph 构建

### 5.1 异构图整体结构

HeteroGraph 是一个 DGL 异构（heterogeneous）图，包含三种节点类型和五种边类型：

```mermaid
graph TD
    subgraph Netlist["网表子图 Topology Space"]
        BLOCK["Block Nodes<br/>特征: type, height, pin数"]
        NET["Net Nodes<br/>特征: bb高, bb宽, bb面积, 密度"]
    end

    subgraph Layout["布局子图 Geometry Space"]
        GRID["Grid Nodes<br/>特征: type, RUDY, pin密度"]
    end

    BLOCK -->|"topo_out<br/>驱动关系"| NET
    NET -->|"topo_in<br/>扇入关系"| BLOCK
    GRID -->|"grid_edge<br/>几何邻接"| GRID
    BLOCK -->|"geom_loc<br/>位置映射"| GRID
    NET -->|"geom_bb<br/>边界框映射"| GRID
```

### 5.2 三种节点及其特征

#### Block Node（块节点）
每个物理块（CLB、DSP、BRAM 等）是一个节点，维度为 4：
| 特征 | 维度 | 含义 |
|------|------|------|
| `type_embed` | 1 | 块类型的可学习嵌入 |
| `block_height` | 1 | 块占用的格点数（高度） |
| `num_in_pin` | 1 | 输入引脚数 |
| `num_out_pin` | 1 | 输出引脚数 |

#### Net Node（网节点）
每条网线是一个节点，维度为 4：
| 特征 | 维度 | 含义 |
|------|------|------|
| `bb_high` | 1 | 边界框高度（行数） |
| `bb_length` | 1 | 边界框宽度（列数） |
| `bb_area` | 1 | 边界框面积（high × length） |
| `bb_block_density` | 1 | 边界框内块密度 |

#### Grid Node（格点节点）
FPGA 芯片被离散化为 $W \times H$ 的格点，每个格点是一个节点，维度为 5：
| 特征 | 维度 | 含义 |
|------|------|------|
| `grid_type_embed` | 1 | 格点类型嵌入（空/CLB/DSP/BRAM 等） |
| `grid_h_net_density` | 1 | 通过该格点的线网水平密度 |
| `grid_v_net_density` | 1 | 通过该格点的线网垂直密度 |
| `grid_rudy` | 1 | RUDY 拥塞估计（经 q 因子校正） |
| `pin_density` | 1 | 该格点上的引脚密度 |

### 5.3 五种边类型

| 边类型 | 源节点 | 目标节点 | 语义 | 数量级 |
|--------|--------|---------|------|--------|
| `topo_out` | Block | Net | 块驱动网线（输出） | $O(B)$ |
| `topo_in` | Net | Block | 网线驱动块（输入） | $O(B)$ |
| `grid_edge` | Grid | Grid | 相邻格点的几何连接 | $O(W \times H)$ |
| `geom_loc` | Block | Grid | 块位置到所在格点 | $O(B)$ |
| `geom_bb` | Net | Grid | 网线边界框覆盖的格点 | $O(B \times bb\_area)$ |

### 5.4 数据集规模

根据代码分析：

- **电路数量**：20 个 VTR7 基准电路（`BENCHMARK_LIST` 从 `arch_blif_source/vtr7/` 读取所有 `.blif` 文件）
- **每电路布局实例**：200 个（`seed_range=200`）
- **总图数量**：20 × 200 = 4000 个异构图
- **训练/测试划分**：90%/10%（`test_ratio=0.1`），按电路随机划分种子
- **数据存储**：DGL 二进制格式（`dgl.save_graphs`），每图一个 `.bin` 文件

### 5.5 关键特征计算：RUDY 与 q 因子

RUDY（Rectangular Uniform wire DensitY）是拥塞预测中的经典手工特征。本文对标准 RUDY 进行了改进：

$$RUDY(net) = q(|terminals|) \cdot \frac{bb\_high + bb\_length}{bb\_high \cdot bb\_length}$$

其中：
- $q(|t|)$ 是 Steiner 树布线因子的校正系数，查表获得（1-50 端点预计算，>50 线性外推 $q = 2.7933 + 0.02616 \times (|t| - 50)$）
- $bb\_high$ 和 $bb\_length$ 分别是网线边界框的高和宽

对每个 Net，其边界框覆盖的每个 Grid 累积 RUDY 值。边界框计算考虑了 macro 块的高度——macro 块在其占据的所有列上扩展边界框。

---

## 6. HeteroGNN 双空间消息传递

### 6.1 架构总览

根据 `source/model/FHGNN.py` 的完整代码：

```mermaid
graph TD
    subgraph Input["输入投影"]
        B1["Block Feats → Linear→LeakyReLU"]
        N1["Net Feats → Linear→LeakyReLU"]
        G1["Grid Feats → Linear→LeakyReLU"]
    end

    subgraph Layer["GNN Layer × layer_num"]
        MP["HeteroGraphConv<br/>5种边类型并行MP"]
        ACT["ReLU激活"]
    end

    subgraph Output["输出"]
        OUT["Linear → pred_out<br/>Grid Embedding"]
    end

    B1 --> MP
    N1 --> MP
    G1 --> MP
    MP --> ACT
    ACT --> MP
    ACT --> OUT
```

### 6.2 特征投影（两阶段）

输入特征先经过两层投影：

**第一阶段**（投影到 `hidden_feats // 2`）：
$$h_{block}^{(0)} = \text{LeakyReLU}(W_{block1} \cdot x_{block})$$
$$h_{net}^{(0)} = \text{LeakyReLU}(W_{net1} \cdot x_{net})$$
$$h_{grid}^{(0)} = \text{LeakyReLU}(W_{grid1} \cdot x_{grid})$$

**第二阶段**（投影到 `hidden_feats`，默认 64）：
$$h_{block}^{(1)} = \text{LeakyReLU}(W_{block2} \cdot h_{block}^{(0)})$$
$$h_{net}^{(1)} = \text{LeakyReLU}(W_{net2} \cdot h_{net}^{(0)})$$
$$h_{grid}^{(1)} = \text{LeakyReLU}(W_{grid2} \cdot h_{grid}^{(0)})$$

### 6.3 三空间异构消息传递

每一层 GNN 对 5 种边类型执行异构图卷积。三种消息传递类型对应三个语义空间：

#### 网表空间（`netlist_mp_type`，作用于 `topo_in` 和 `topo_out` 边）

消息在 Block 和 Net 双向流动：
- `topo_out`：Block → Net（驱动关系消息："这个 Block 的特征会影响它驱动的 Net"）
- `topo_in`：Net → Block（扇入关系消息："这些 Net 的特征会影响被它们驱动的 Block"）

三种可选卷积：

| 卷积类型 | 操作 | 特点 |
|---------|------|------|
| **GCN** (`gcn`) | $\sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}HW)$ | 归一化邻域平均 |
| **GAT** (`gat`) | $\sigma(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j)$ | 4头注意力，每头维度 hidden_feats/4 |
| **GraphSAGE** (`sage`) | $\sigma(W \cdot [h_i \| \max_{j \in \mathcal{N}(i)} \sigma(W_{pool}h_j + b)])$ | Pool 聚合器 |

#### 布局空间（`layout_mp_type`，作用于 `grid_edge` 边）

消息在几何相邻的 Grid 之间传播。Grid 形成一个 2D 网格图（128×128），边连接上下左右邻居。对于 macro 块覆盖的 Grid，额外增加了同 macro 内 Grid 之间的边。

同样支持 GCN、GAT、GraphSAGE 三种卷积。

#### 空间映射（`space_mp_type`，作用于 `geom_loc` 和 `geom_bb` 边）

消息在拓扑空间和几何空间之间跨空间传播：
- `geom_loc`：Block → Grid（"块的位置信息影响所在格点"）
- `geom_bb`：Net → Grid（"网的边界框范围影响覆盖格点"）

### 6.4 异构聚合（`hagg_type`）

所有边类型的消息在各自的目标节点聚合。两种聚合策略：

- **`sum`**（默认）：对所有边类型的结果求和
  $$h_v^{(l+1)} = \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \text{MP}_r(h_u^{(l)}, h_v^{(l)})$$

- **`stack`**：对所有边类型的结果拼接后线性投影
  $$h_v^{(l+1)} = W_{agg} \cdot [h_{v,r_1} \| h_{v,r_2} \| \dots \| h_{v,r_k}]$$

对于 GAT 卷积，多头注意力输出需要额外 reshape（`view` 操作展开多头维度）。对于 Grid 节点，如果使用 `stack` 聚合，需要经过两个额外的 Linear 层（`lin_grid_agg` 和 `lin_grid_agg2`）。

### 6.5 默认配置

```python
netlist_mp_type = 'sage'   # GraphSAGE 用于拓扑空间
layout_mp_type = 'sage'    # GraphSAGE 用于布局空间
space_mp_type = 'gcn'      # GCN 用于空间映射
hagg_type = 'mean'         # 平均聚合
layer_num = 2              # 2层GNN
hidden_num = 64            # 64维隐藏层
nn_out_dim = 4             # 4通道输出（对应4种拥塞）
cnn_image_size = 128       # 128×128图像
```

---

## 7. HeteroGNN-embedded cGAN 架构

### 7.1 整体流水线

```mermaid
graph TD
    subgraph Encoder["条件编码器: HeteroGNN"]
        HG["HeteroGraph<br/>3节点类型, 5边类型"]
        GNN["FHGNN<br/>2层异构GNN"]
        EMB["Grid Embeddings<br/>16384×64"]
    end

    subgraph Generator["生成器: U-Net"]
        IMG["g_emb2image<br/>→1×64×128×128"]
        UNET["U-Net Encoder-Decoder<br/>Skip Connections"]
        PRED["Predicted Congestion<br/>1×4×128×128"]
    end

    subgraph Discriminator["鉴别器: PatchGAN"]
        COND["Grid Feature Image<br/>1×4×128×128"]
        CONCAT["Concat[Cond, Real/Fake]"]
        DISC["CNN Discriminator<br/>Patch-level Real/Fake"]
    end

    HG --> GNN
    GNN --> EMB
    EMB --> IMG
    IMG --> UNET
    UNET --> PRED
    COND --> CONCAT
    PRED --> CONCAT
    CONCAT --> DISC
```

### 7.2 PREDICTOR_GEN：生成器封装

```python
class PREDICTOR_GEN(nn.Module):
    """
    FHGNN 生成 Grid Embeddings → 转为图像 → U-Net 生成拥塞图
    """
    def forward(self, graph):
        _, grid_emb = self.fhgnn(graph)            # HeteroGNN 前向
        pred_image = self.gen(g_emb2image(grid_emb)) # U-Net 生成
        real_image = g_emb2image(graph.nodes['grid'].data['label'])
        return pred_image, real_image
```

关键流程：
1. FHGNN 对异构图进行 2 层消息传递，输出 Grid Embeddings（16384 × 64）
2. `g_emb2image` 将 Embedding 重塑为 `[1, 64, 128, 128]` 图像
3. U-Net 生成器对 Embedding 图像进行编码-解码，输出 `[1, 4, 128, 128]` 拥塞预测
4. `g_emb2image` 同时将真实标签（4维 per-grid）转为图像格式用于损失计算

### 7.3 U-Net 生成器（`UNET_adaptive`）

根据 `source/model/UNET_adaptive.py`：

**核心参数**：
- `in_channels=64`（HeteroGNN 的隐藏维度）
- `out_channels=4`（4 通道拥塞：chanx_l4, chany_l4, chanx_l16, chany_l16）
- `features=64`（基础特征数，每层翻倍）
- `image_size=128`（自适应计算层数 `layer_num = ceil(log2(128)) = 7`）

**架构结构**（`decoder=True` 时的完整 U-Net）：

| 阶段 | 层 | 输入通道 | 输出通道 | 操作 |
|------|-----|---------|---------|------|
| 初始下采样 | `initial_down` | 64 | 64 | Conv2d(k=4,s=2)+LeakyReLU |
| 下采样块 | `down_blocks[1]` | 64 | 128 | Conv(k=4,s=2)+BN+LeakyReLU |
| 下采样块 | `down_blocks[2]` | 128 | 256 | 同上 |
| 下采样块 | `down_blocks[3]` | 256 | 512 | 同上 + Dropout(0.5) |
| 下采样块 | `down_blocks[4-5]` | 512 | 512 | 同上 |
| 瓶颈 | `bottleneck` | 512 | 512 | Conv+ReLU |
| 初始上采样 | `initial_up` | 512 | 512 | ConvT+BN+ReLU+Dropout |
| 上采样块 | `up_blocks[3]` | 1024 | 256 | ConvT(k=4,s=2)+Skip Cat |
| 上采样块 | `up_blocks[2]` | 512 | 128 | 同上 |
| 上采样块 | `up_blocks[1]` | 256 | 64 | 同上 |
| 最终上采样 | `final_up` | 128 | 4 | ConvT(k=4,s=2) |

**设计特点**：
- 自适应层数：`layer_num = ceil(log2(image_size))`，支持 32/64/128/256/512 分辨率
- 也支持 `decoder=False` 模式：瓶颈输出经过 Linear(512, 1) 输出标量预测（用于线长预测等辅助任务）
- Skip connections：上采样块与对应分辨率的 Skip 连接

### 7.4 鉴别器（Discriminator）

根据 `source/model/Discriminator.py`：

**核心参数**：
- `channels_xny = in_grid_feats + nn_out_dim`（条件特征 + 预测通道）
- `features = 64`
- `image_size = 128`（`layer_num = ceil(log2(128)) - 4 = 3`）

**架构**（PatchGAN 风格的鉴别器）：

| 层 | 输入通道 | 输出通道 | 步长 | 操作 |
|-----|---------|---------|------|------|
| `initial` | channels_xny | 64 | 2 | Conv(k=4)+LeakyReLU(0.2) |
| `blocks[1]` | 64 | 128 | 2 | Conv(k=4)+BN+LeakyReLU(0.2) |
| `blocks[2]` | 128 | 1 | 1 | Conv(k=4)+BN+LeakyReLU(0.2) |

鉴别器输出一个 **Patch 级别的真假判断**：输入的 real/fake 拥塞图和条件 Grid 特征图拼接后，经过卷积下采样，输出一个 $N \times N$ 的矩阵，每个元素对应该 patch 区域的真实性判断。

### 7.5 PREDICTOR_DISC：鉴别器封装

```python
class PREDICTOR_DISC(nn.Module):
    """
    与生成器共享 FHGNN 编码器（但冻结/独立复制）
    将 Grid 特征嵌入转为图像，与拥塞图拼接送入 CNN 鉴别器
    """
    def forward(self, graph, chan_usage_image):
        _, grid_emb = self.fhgnn(graph)
        d_out = self.disc(g_emb2image(grid_emb), chan_usage_image)
        return d_out
```

**注意**：代码中生成器和鉴别器使用**独立的 FHGNN 实例**（`fhgnn_gen` 和 `fhgnn_disc`），不共享参数。鉴别器的 FHGNN 使用 `hidden_feats=nn_out_dim`（而非 64），输出维度更小。

---

## 8. 损失函数与训练策略

### 8.1 生成器损失

生成器对抗两项损失的加权和：

$$\mathcal{L}_G = \mathcal{L}_{GAN}(G) + \lambda \cdot \mathcal{L}_{feature}$$

其中：

**对抗损失**（BCEWithLogitsLoss）：
$$\mathcal{L}_{GAN}(G) = -\log D(G(z|c), c)$$

即：生成器希望鉴别器将假拥塞图误判为真实。

**特征损失**（MSE，权重 $\lambda = 100$）：
$$\mathcal{L}_{feature} = 100 \cdot \| \hat{y} - y \|_2^2$$

即：预测拥塞图与真实拥塞图之间的逐像素均方误差。

**总生成器损失**：
$$\mathcal{L}_G = \text{BCE}(D(G(z|c), c), \mathbf{1}) + 100 \times \text{MSE}(\hat{y}, y)$$

### 8.2 鉴别器损失

标准的 cGAN 鉴别器损失：

$$\mathcal{L}_D = \frac{1}{2}[\mathcal{L}_{D,real} + \mathcal{L}_{D,fake}]$$

- 真实样本损失：$\mathcal{L}_{D,real} = \text{BCE}(D(x, c), \mathbf{1})$
- 虚假样本损失：$\mathcal{L}_{D,fake} = \text{BCE}(D(G(z|c), c), \mathbf{0})$

其中 `.detach()` 确保生成器梯度不会通过鉴别器反向传播。

### 8.3 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 优化器 | Adam | 生成器和鉴别器独立优化 |
| 学习率 | $2 \times 10^{-4}$ | 初始学习率 |
| Adam $\beta$ | (0.5, 0.999) | 标准 GAN 训练配置 |
| 学习率衰减 | StepLR($\gamma = 0.98$) | 每 epoch 衰减 2% |
| 训练轮数 | 200 | 完整训练 |
| 梯度裁剪 | `max_norm=1.0` | 稳定 GAN 训练 |
| 混合精度 | AMP (GradScaler) | 加速训练 |
| 批量大小 | 1 | 每个异构图单独训练 |

### 8.4 替代训练配置（`train_1hg2cgan.py`）

在 `train_1hg2cgan.py` 中，鉴别器**不使用**独立的 FHGNN：
- 生成器：FHGNN + U-Net（正常结构）
- 鉴别器：仅 CNN，输入为 Grid **原始特征**图像（非 FHGNN 嵌入），与拥塞图拼接

这种变体让鉴别器看到原始特征而非编码后的嵌入，可用于消融对比（评估 FHGNN 编码对鉴别器的影响）。

对比 `train_2hg2cgan.py`（默认配置）：
- 生成器：FHGNN_gen + U-Net
- 鉴别器：FHGNN_disc + CNN（**双 FHGNN 架构**）

---

## 9. 实验设置

### 9.1 基准数据集

| 数据集 | 电路数 | 每电路布局数 | FPGA 架构 | 规模 |
|--------|--------|------------|-----------|------|
| **VTR7** | 20 | 200 | k6_frac_N10 | 小型-中型 |
| **Titan23** | 若干 | 200 | Stratix IV | 大型 |

### 9.2 评估指标

| 指标 | 全称 | 公式 | 含义 |
|------|------|------|------|
| **NRMSE** | Normalized Root Mean Square Error | $\sqrt{\text{MSE}}$ （代码中未归一化，直接是 RMSE） | 预测与真实的偏差 |
| **SSIM** | Structural Similarity Index | $\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$ | 结构相似性 |
| **Pearson** | Pearson Correlation | $\frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ | 线性相关性 |
| **Spearman** | Spearman Rank Correlation | 排序相关性 | 单调关系 |
| **Kendall** | Kendall's $\tau$ | $\frac{\text{concordant} - \text{discordant}}{\binom{n}{2}}$ | 排序一致性 |

评估时还单独计算了 4 个 channel 各自的分段指标（`l4_x`, `l4_y`, `l16_x`, `l16_y`）。

### 9.3 对比方法

根据代码分析，仓库中包含以下对比模型的训练脚本：

| 模型 | 训练脚本 | 技术路线 |
|------|---------|---------|
| **1hg2cgan** | `train_1hg2cgan.py` | 单 FHGNN cGAN（鉴别器无 GNN） |
| **2hg2cgan** | `train_2hg2cgan.py` | **双 FHGNN cGAN**（默认/最优配置） |
| **fhgnn_homo** | `train_fhgnnhomo.py` | 纯 FHGNN（无 cGAN，直接将 GNN 输出作为预测） |
| **pix2pix** | `train_pix2pix.py` | 标准 pix2pix（图像到图像，无 GNN） |
| **unet** | `train_unet.py` | 纯 U-Net（无 GAN loss，仅 MSE） |
| **wcpnet** | `train_wcpnet.py` | WCPNet（Wirelength-Congestion Predictor Network） |

---

## 10. 实验结果与分析

### 10.1 VTR7 基准结果

基于论文摘要和代码分析，主要结果总结如下：

| 方法 | RMSE (NRMS) | SSIM | Pearson | Spearman | Kendall |
|------|-------------|------|---------|----------|---------|
| U-Net | 基准 | 基准 | 基准 | 基准 | 基准 |
| pix2pix (cGAN) | 基准+ | 基准+ | 基准+ | 基准+ | 基准+ |
| FHGNN (纯 GNN) | 基准+ | 基准+ | 基准+ | 基准+ | 基准+ |
| **2hg2cgan (本文)** | **-18.2% vs SOTA** | 最优 | 最优 | 最优 | 最优 |

> **关键数据**：在 VTR7 基准上，RMSE（均方根误差）降低 18.2%。这是对最优对比方法（可能为 pix2pix 或 FHGNN）的提升幅度。

### 10.2 Titan23 基准结果

| 方法 | RMSE 降低幅度 |
|------|-------------|
| SOTA 方法 | 基准 |
| **本文 (2hg2cgan)** | **-15.0%** |

Titan23 是更大规模的基准，包含了 Stratix IV 架构的大型设计。15.0% 的 RMSE 降低表明方法在大规模设计上仍然有效。

### 10.3 多指标综合优势

HeteroGNN-cGAN 在所有五个指标上均优于对比方法：
- **NRMSE**：更低（预测偏差更小）
- **SSIM**：更高（结构相似性更好）
- **Pearson**：更高（线性相关性更强）
- **Spearman**：更高（排序更准确）
- **Kendall**：更高（排序一致性更好）

四项相关性指标的提升对各种下游任务（如跳过布线、优化引导）特别重要：即使绝对拥塞值预测有误差，只要峰值排序准确，就能有效指导优化。

### 10.4 推理时间

根据代码中的计时日志，模型推理时间被记录在 TensorBoard 中。由于模型在推理时只需前向一次（无对抗训练），且 GNN 推理较为高效，推理时间应当与 pix2pix 或 U-Net 相当或略慢。

---

## 11. 消融实验

### 11.1 代码中的消融控制参数

训练脚本中预设了以下消融控制参数，支持系统的消融研究：

| 参数 | 默认值 | 消融操作 | 目的 |
|------|--------|---------|------|
| `use_grid_feature` | 1 | 设为 0 时清零 Grid 节点特征 | 评估 Grid 特征的重要性 |
| `use_geom_bb_edge` | 1 | 设为 0 时删除 geom_bb 边 | 评估边界框边的重要性 |
| `use_geom_loc_edge` | 1 | 设为 0 时删除 geom_loc 边 | 评估位置映射边的重要性 |
| `retain_net_ratio` | 1.0 | 设为 <1.0 时随机删除 Net 节点 | 评估网表采样的鲁棒性 |
| `retain_block_ratio` | 1.0 | 设为 <1.0 时随机删除 Block 节点 | 评估块采样的鲁棒性 |

### 11.2 组件消融推断

基于代码设计的消融维度和已知对比方法，推断消融结果：

#### 图组件消融

| 消融配置 | 预期影响 | 原因 |
|---------|---------|------|
| 完整 HeteroGraph（5 边类型） | 最优 | 同时利用拓扑和几何信息 |
| 去除 `geom_bb` 边 | RMSE 上升明显 | Net-Grid 映射是连接拓扑和几何的关键桥梁 |
| 去除 `geom_loc` 边 | RMSE 上升 | Block-Grid 映射连接块位置和网格特征 |
| 去除 Grid 特征 | RMSE 上升 | RUDY/pin_density 等特征提供重要的先验信息 |
| 仅 `grid_edge`（退化为纯布局图） | RMSE 显著上升 | 完全丢失网表拓扑信息 |

#### 模型架构消融

| 配置 | 说明 | 预期效果 |
|------|------|---------|
| 2hg2cgan（双 FHGNN cGAN） | 默认最优配置 | 最优 |
| 1hg2cgan（单 FHGNN cGAN） | 鉴别器无 GNN 编码 | 略差，鉴别器缺少图感知 |
| FHGNN only（纯 GNN 回归） | 无 cGAN | 较差，丢失生成式优势 |
| pix2pix（纯图像 cGAN） | 无 GNN | 较差，丢失拓扑信息 |
| U-Net（纯判别式） | 无 GAN 无 GNN | 最差 |

### 11.3 消息传递类型对比

代码支持三种卷积类型（GCN、GAT、GraphSAGE）在不同空间的自由组合。根据默认配置（netlist/layout=SAGE, space=GCN）推断：

| 空间 | 最优选择 | 推理 |
|------|---------|------|
| 网表空间 | GraphSAGE | Pool 聚合器适合处理变长邻居（不同块的引脚数差异大） |
| 布局空间 | GraphSAGE | 同样适合处理变长邻居（边界格点的度不同） |
| 空间映射 | GCN | 简单的加权平均可能更适合固定拓扑的几何关系 |

---

## 12. 讨论与局限性

### 12.1 贡献总结

1. **首次将异构图 GNN 与 cGAN 深度融合**：不同于 LHNN 的纯判别式 GNN 或 cGAN 的纯图像生成，本文的 HeteroGNN-cGAN 融合了两种范式的优势
2. **完整开源实现**：包括数据生成（VPR 自动化）、图构建、模型训练和评估的完整流水线
3. **双基准验证**：在 VTR7（小型）和 Titan23（大型）两个基准上验证了方法有效性
4. **灵活的消融框架**：代码中预设了系统化的消融参数，方便研究各部分贡献

### 12.2 局限性

1. **ACM 付费墙限制**：论文 PDF 无法公开获取，限制了独立验证和复现
2. **GitHub Stars 较少**：2 stars / 1 fork，社区关注度有限
3. **仅支持 FPGA**：方法为 FPGA 架构定制（利用 VTR 工具链），难以直接迁移到 ASIC
4. **训练成本**：GAN 训练需要同时优化生成器和鉴别器，且每步需要 GNN 前向，训练时间较长
5. **GNN 推理开销**：推理时需要执行 GNN 消息传递，相比纯 CNN 方法推理速度更慢
6. **每电路 200 布局实例**：数据量相对有限，模型可能过拟合到特定布局分布
7. **未与 LHNN 直接对比**：Day 16 的 LHNN 是最相关的前序工作，但基准不同（VTR7 vs ISPD），难以直接比较

### 12.3 未来方向

- 扩展到 ASIC 设计（需要适配不同的工具链和图结构）
- 将拥塞预测集成到强化学习布局优化器中
- 探索更高效的 GNN 架构以减少推理开销
- 利用预训练 + 微调范式的跨电路迁移学习

---

## 13. 创新点深度分析

### 13.1 创新点 1：HeteroGraph 统一表示

**创新程度**: ★★★★☆（高）

HeteroGraph 是本文最核心的创新。它将三种异构实体（Block、Net、Grid）和五种语义关系（驱动、扇入、几何相邻、位置映射、边界框覆盖）统一为单一异构图。

**与 LHNN 的区别**：
- LHNN 使用超图（hypergraph）建模网表：一条超边连接多个 G-cell
- Day 21 使用二部图建模网表：Net 节点单独存在，通过 `topo_in` 和 `topo_out` 边连接 Block
- 二部图建模更自然地支持异构图 GNN 的消息传递框架

**与 CircuitGNN（Day 20）的区别**：
- CircuitGNN 追求通用电路表示（覆盖多种 EDA 任务）
- Day 21 专门针对拥塞预测，特征设计高度任务特定

### 13.2 创新点 2：双空间消息传递

**创新程度**: ★★★★☆（高）

通过 5 种边类型的异构图卷积，实现消息在拓扑空间、几何空间和交叉空间的三向传播。

**数学本质**：k 层消息传递后，每个 Grid 节点的感受野同时覆盖：
- 几何邻域（通过 `grid_edge`）：$k$ 步内的所有相邻格点
- 拓扑邻域（通过 `topo_in/out` 传递到 `geom_loc`）：与同网线相连的远距离 Block 的几何信息
- 结构邻域（通过 `geom_bb`）：所有覆盖该格点的网线的特征

这实现了 CNN 卷积核无法做到的**拓扑距离感知**：两个几何上相距甚远但被同一网线连接的单元，其信息可以在 GNN 中通过 2 跳传播聚合。

### 13.3 创新点 3：HeteroGNN-embedded cGAN

**创新程度**: ★★★★☆（高）

将异构图 GNN 作为条件 GAN 的编码器是一个有意义的架构创新：

1. **编码器部分**（HeteroGNN）：将异构图结构编码为 Grid Embeddings。这是一个"图到特征"的过程。
2. **生成器部分**（U-Net）：将 Embedding 图像解码为拥塞热力图。这是一个"特征到图像"的过程。
3. **鉴别器部分**（CNN）：在 patch 级别判断生成的拥塞图是否与 Grid 特征图一致。

这种"先编码后生成"的架构比纯 pix2pix（直接从布局图像生成）或纯 GNN（直接回归）具有更好的信息保留能力。

**与 Day 18 cGAN 的关键不同**：
- Day 18：`layout_image → U-Net → congestion_image`（编码器是隐式的，信息在 U-Net 压缩路径中丢失）
- Day 21：`HeteroGraph → FHGNN → grid_emb → U-Net → congestion_image`（编码器是显式的，拓扑信息在 GNN 中保留）

### 13.4 创新点 4：自适应 U-Net 生成器

**创新程度**: ★★★☆☆（中等）

UNET_adaptive 的 `layer_num = ceil(log2(image_size))` 设计使其能无缝适配不同分辨率的 FPGA（32-512）。这使得同一代码库可以处理小到 VTR7 到大到 Titan23 的不同规模设计。

---

## 14. 演进对比表

| 维度 | RouteNet (D15) | ClusterNet (D13) | LHNN (D16) | NAS-Routability (D17) | cGAN (D18) | PROS (D19) | CircuitGNN (D20) | **Day 21 (本文)** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **年份** | 2018 | 2023 | 2022 | 2021 | 2019 | 2020 | 2022 | **2025** |
| **技术路线** | CNN | GNN+聚类 | 异构图GNN | NAS+CNN | cGAN | CNN | 通用GNN | **HeteroGNN+cGAN** |
| **输入模态** | 图像 | 图 | 格超图 | 图像 | 图像 | 图像 | 通用图 | **异构图** |
| **网表拓扑** | 手工特征(间接) | GNN(直接) | 超边(直接) | 手工特征(间接) | 手工特征(间接) | 手工特征(间接) | GNN(直接) | **异构图边(直接)** |
| **几何建模** | CNN感受野 | - | 晶格图 | CNN感受野 | CNN感受野 | CNN感受野 | - | **Grid边+GNN** |
| **生成式** | 否 | 否 | 否 | 否 | **是** | 否 | 否 | **是** |
| **GAN** | - | - | - | - | pix2pix | - | - | **HeteroGNN-cGAN** |
| **图类型** | - | 同构图 | 异构图(超图) | - | - | - | 异构图 | **异构图(二部图)** |
| **消息传递** | - | 单空间 | 双空间 | - | - | - | 多任务 | **三空间** |
| **预测头** | 单任务 | 单任务 | 联合监督 | 单任务 | 单任务 | 单任务 | 多任务 | **单回归(4通道)** |
| **平台** | ASIC | FPGA | ASIC | ASIC | ASIC | ASIC | 通用 | **FPGA** |
| **开源** | 否 | 否 | 否 | 否 | 否 | 否 | 部分 | **是** |
| **NAS** | - | - | - | 是 | - | - | - | - |
| **主要提升** | 首次CNN | 首次GNN+聚类 | F1+35% vs CNN | Kendall+5.85% | 首次生成式 | 优化引导 | 通用表示 | **RMSE -18.2%/-15.0%** |
| **框架** | TensorFlow | PyTorch | PyTorch | PyTorch | - | - | PyTorch | **DGL+PyTorch** |

---

## 15. 参考文献

1. Yang, Q., Li, J., Li, R., He, Y., Ha, Y., Shen, L., Bai, R., & Yu, H. (2025). FPGA Routing Congestion Prediction via Graph Learning-Aided Conditional GAN. *ACM Transactions on Design Automation of Electronic Systems (TODAES)*. DOI: [10.1145/3773770](https://doi.org/10.1145/3773770). GitHub: [AIPnR/FPGA_Hetero_Congestion_Prediction](https://github.com/AIPnR/FPGA_Hetero_Congestion_Prediction).

2. Xie, Z., Huang, Y., Fang, G., Ren, H., Fang, S., Chen, Y., & Hu, J. (2018). RouteNet: Routability Prediction for Mixed-Size Designs Using Convolutional Neural Network. *ICCAD 2018*. (Day 15)

3. Wang, B., Shen, G., Li, D., Hao, J., Liu, W., Huang, Y., Wu, H., Lin, Y., Chen, G., & Heng, P. A. (2022). LHNN: Lattice Hypergraph Neural Network for VLSI Congestion Prediction. *arXiv:2203.12831*. (Day 16)

4. Chang, C., Pan, J., Zhang, T., Xie, Z., Hu, J., Qi, W., Lin, C., Liang, R., Mitra, J., Fallon, E., & Chen, Y. (2021). Automatic Routability Predictor Development Using Neural Architecture Search. *ICCAD 2021*. (Day 17)

5. Yu, C., & Zhang, Z. (2019). Painting on Placement: Forecasting Routing Congestion using Conditional Generative Adversarial Nets. *DAC 2019*. (Day 18)

6. Chen, J., Kuang, J., Zhao, G., Huang, D. J.-H., & Young, E. F. Y. (2020). PROS: A Plug-in for Routability Optimization Applied in the State-of-the-art Commercial EDA Tool Using Deep Learning. *ICCAD 2020*. (Day 19)

7. Wang, H., Wang, K., Yang, J., Shen, L., Sun, N., Lee, H., & Han, S. (2022). GNN-based End-to-End Circuit Representation Learning for Electronic Design Automation. *NeurIPS 2022*. (Day 20)

8. Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017*.

9. Wang, M., Zheng, D., Ye, Z., Gan, Q., Li, M., Song, X., Zhou, J., Ma, C., Yu, L., Gai, Y., Xiao, T., He, T., Karypis, G., Li, J., & Zhang, Z. (2019). Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks. *arXiv:1909.01315*.
