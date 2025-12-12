# GNN三维地质建模系统 - 项目架构说明

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [核心模块详解](#核心模块详解)
- [数据流程](#数据流程)
- [模型架构](#模型架构)
- [扩展指南](#扩展指南)

---

## 项目概述

本项目是一个基于图神经网络(GNN)的三维地质建模系统，用于根据稀疏钻孔数据预测地下岩性分布。

### 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch | 模型构建与训练 |
| 图神经网络 | PyTorch Geometric | GNN层实现 |
| 数据处理 | NumPy, Pandas, SciPy | 数据加载与预处理 |
| 机器学习 | Scikit-learn | 数据划分、评估指标 |
| 可视化 | Plotly, Streamlit | 3D可视化与Web界面 |

### 核心思想

```
钻孔数据 → 图结构 → GNN模型 → 岩性预测 → 3D地质模型
```

1. **图构建**: 将钻孔采样点视为图节点，基于空间距离建立边连接
2. **特征编码**: 节点特征包括空间坐标(x,y,z)和地质属性(孔隙度等)
3. **消息传递**: GNN通过邻居信息聚合学习空间模式
4. **分类预测**: 输出每个点的岩性类别概率

---

## 目录结构

```
dizhijianmo/
│
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── models.py                 # GNN模型定义
│   ├── data_loader.py            # 数据加载与图构建
│   └── trainer.py                # 训练与评估
│
├── configs/                      # 配置文件目录
│   ├── __init__.py
│   └── config.py                 # 参数配置
│
├── data/                         # 数据目录
│   ├── (your_data.csv)           # 用户数据
│   └── predictions.csv           # 预测结果输出
│
├── models/                       # 模型保存目录
│   ├── best_model.pt             # 训练好的模型
│   └── preprocessor.json         # 预处理器状态
│
├── docs/                         # 文档目录
│   ├── STARTUP_GUIDE.md          # 启动指南
│   └── ARCHITECTURE.md           # 架构说明 (本文件)
│
├── app.py                        # Streamlit可视化前端
├── main.py                       # 主入口脚本
├── requirements.txt              # 依赖列表
└── README.md                     # 项目简介
```

---

## 核心模块详解

### 1. 模型模块 (`src/models.py`)

定义了四种GNN模型架构:

#### 1.1 GeoGCN - 图卷积网络

```python
class GeoGCN(nn.Module):
    """
    基础GCN模型
    - 使用 GCNConv 层进行图卷积
    - 支持边权重 (基于距离)
    - 适合中小规模数据
    """
```

**结构图:**
```
输入特征 [N, F_in]
    ↓
GCNConv + BatchNorm + ReLU + Dropout
    ↓
GCNConv + BatchNorm + ReLU + Dropout
    ↓
GCNConv (输出层)
    ↓
输出 [N, num_classes]
```

#### 1.2 GeoGraphSAGE - 采样聚合网络

```python
class GeoGraphSAGE(nn.Module):
    """
    GraphSAGE模型
    - 采样邻居并聚合信息
    - 支持 mean/max/lstm 聚合
    - 适合大规模数据，可扩展性好
    """
```

**优势:**
- 可处理大规模图 (百万节点)
- 支持归纳学习 (推理时可处理新节点)
- 计算效率高

#### 1.3 GeoGAT - 图注意力网络

```python
class GeoGAT(nn.Module):
    """
    GAT模型
    - 使用注意力机制学习邻居重要性
    - 多头注意力增强表达能力
    - 适合邻居重要性不均匀的场景
    """
```

**结构图:**
```
输入 [N, F]
    ↓
GATConv (heads=4) → [N, F*4]
    ↓
GATConv (heads=4) → [N, F*4]
    ↓
GATConv (heads=1, concat=False) → [N, C]
```

#### 1.4 Geo3DGNN - 混合地质模型

```python
class Geo3DGNN(nn.Module):
    """
    专为三维地质数据设计的混合模型

    特点:
    - 空间位置编码器: 学习坐标的隐式表示
    - 特征编码器: 处理其他地质属性
    - 多尺度卷积: 结合SAGE和GCN
    - 跳跃连接: 融合多层特征
    """
```

**架构图:**
```
输入 [N, F]
    ↓
┌─────────────────┬──────────────────┐
│ 坐标 (x,y,z)    │ 其他特征          │
│      ↓          │       ↓          │
│ 空间编码器       │  特征编码器       │
│  MLP [3→32]     │  MLP [F-3→32]    │
└────────┬────────┴────────┬─────────┘
         │      拼接        │
         └────────┬─────────┘
                  ↓
         SAGEConv + BatchNorm
                  ↓
          GCNConv + BatchNorm ──────┐
                  ↓                 │
         SAGEConv + BatchNorm       │ 跳跃连接
                  ↓                 │
          GCNConv + BatchNorm ──────┤
                  ↓                 │
              Fusion ←──────────────┘
                  ↓
           分类器 MLP
                  ↓
            输出 [N, C]
```

#### 模型工厂函数

```python
def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    统一的模型创建接口

    Args:
        model_name: 'gcn', 'graphsage', 'gat', 'geo3d'
        **kwargs: 模型参数

    Returns:
        初始化好的模型实例
    """
```

---

### 2. 数据加载模块 (`src/data_loader.py`)

#### 2.1 BoreholeDataProcessor

核心数据处理类，负责:

```python
class BoreholeDataProcessor:
    """
    钻孔数据处理器

    功能:
    1. 数据加载 (CSV/Excel)
    2. 图结构构建 (KNN/Radius/Delaunay)
    3. 特征标准化
    4. 数据集划分
    """
```

**图构建方法:**

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| `knn` | K近邻图，每个点连接最近的K个邻居 | 通用场景，推荐默认使用 |
| `radius` | 半径图，连接指定距离内的所有点 | 点分布密度均匀时 |
| `delaunay` | Delaunay三角剖分 | 需要保持几何拓扑时 |

**处理流程:**

```
原始CSV数据
    ↓
提取坐标 (x, y, z)
    ↓
提取特征 (porosity, permeability, ...)
    ↓
编码标签 (lithology → 0, 1, 2, ...)
    ↓
标准化 (StandardScaler)
    ↓
构建图 (KNN/Radius/Delaunay)
    ↓
创建掩码 (train/val/test)
    ↓
PyG Data 对象
```

#### 2.2 GridInterpolator

用于将稀疏预测结果插值到规则三维网格:

```python
class GridInterpolator:
    """
    网格插值器

    用途:
    - 生成规则网格用于可视化
    - 支持体渲染和切片显示
    """
```

---

### 3. 训练模块 (`src/trainer.py`)

#### 3.1 GeoModelTrainer

完整的训练管理类:

```python
class GeoModelTrainer:
    """
    训练器

    功能:
    - 训练循环 (前向传播、反向传播、参数更新)
    - 验证评估 (损失、准确率、F1)
    - 早停机制 (防止过拟合)
    - 学习率调度 (Plateau/Cosine)
    - 模型保存/加载
    - 类别权重 (处理不平衡数据)
    """
```

**训练流程:**

```
初始化
    ↓
┌──────────────────────────────────────┐
│  for epoch in range(epochs):        │
│      ↓                               │
│  train_epoch()                       │
│      - 前向传播                       │
│      - 计算损失 (CrossEntropy)        │
│      - 反向传播                       │
│      - 更新参数                       │
│      ↓                               │
│  validate()                          │
│      - 计算验证损失和准确率            │
│      - 计算F1分数                     │
│      ↓                               │
│  更新学习率 (scheduler.step)          │
│      ↓                               │
│  早停检查                            │
│      - 如果验证损失不再下降，停止训练   │
│      ↓                               │
│  保存最佳模型                         │
└──────────────────────────────────────┘
    ↓
返回训练历史
```

**关键方法:**

| 方法 | 功能 |
|------|------|
| `train_epoch()` | 单个epoch的训练 |
| `validate()` | 验证集评估 |
| `train()` | 完整训练流程 |
| `evaluate()` | 测试集评估，输出详细报告 |
| `predict()` | 对所有节点进行预测 |
| `save_model()` / `load_model()` | 模型持久化 |

#### 3.2 类别权重计算

```python
def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    计算类别权重，处理样本不平衡问题

    公式: weight[c] = total_samples / (num_classes * count[c])
    """
```

---

### 4. 可视化模块 (`app.py`)

基于 Streamlit 的交互式Web界面:

```
┌─────────────────────────────────────────────────────────┐
│                    侧边栏 (参数设置)                      │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 数据配置                                            ││
│  │  - 数据来源 (模拟/上传)                              ││
│  │  - 钻孔数量、采样点数                                ││
│  ├─────────────────────────────────────────────────────┤│
│  │ 图构建                                              ││
│  │  - 图类型 (KNN/Radius/Delaunay)                     ││
│  │  - K邻居数                                          ││
│  ├─────────────────────────────────────────────────────┤│
│  │ 模型配置                                            ││
│  │  - 模型类型                                         ││
│  │  - 隐藏维度、层数、Dropout                           ││
│  ├─────────────────────────────────────────────────────┤│
│  │ 训练配置                                            ││
│  │  - 学习率、轮数、早停                                ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    主区域 (标签页)                        │
│                                                         │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ 数据探索  │ 模型训练  │ 结果分析  │ 三维可视化│         │
│  └──────────┴──────────┴──────────┴──────────┘         │
│                                                         │
│  数据探索:                                               │
│    - 数据预览表格                                        │
│    - 3D钻孔散点图                                        │
│    - 岩性分布柱状图                                      │
│    - 深度分布直方图                                      │
│                                                         │
│  模型训练:                                               │
│    - 实时训练进度条                                      │
│    - 损失曲线图                                         │
│    - 准确率曲线图                                        │
│                                                         │
│  结果分析:                                               │
│    - 关键指标卡片 (准确率、F1)                           │
│    - 混淆矩阵热力图                                      │
│    - 详细分类报告表格                                    │
│                                                         │
│  三维可视化:                                             │
│    - 预测结果3D散点图                                    │
│    - 剖面视图 (X/Y方向)                                  │
│    - 错误预测高亮                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 数据流程

### 完整数据流

```
                    ┌─────────────────┐
                    │   原始钻孔数据    │
                    │   (CSV/Excel)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ BoreholeData    │
                    │ Processor       │
                    │                 │
                    │ - 加载数据       │
                    │ - 提取特征       │
                    │ - 编码标签       │
                    │ - 标准化         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   构建图结构     │
                    │                 │
                    │ - KNN/Radius    │
                    │ - 计算边权重     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PyG Data对象   │
                    │                 │
                    │ - x: 节点特征    │
                    │ - edge_index    │
                    │ - edge_weight   │
                    │ - y: 标签       │
                    │ - masks         │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
     ┌──────────┐     ┌──────────┐     ┌──────────┐
     │ 训练集    │     │ 验证集    │     │ 测试集    │
     │ (70%)    │     │ (10%)    │     │ (20%)    │
     └────┬─────┘     └────┬─────┘     └────┬─────┘
          │                │                │
          ▼                ▼                │
     ┌─────────────────────────┐            │
     │      GNN 模型训练        │            │
     │                         │            │
     │  for epoch:             │            │
     │    train on 训练集       │            │
     │    validate on 验证集    │            │
     │    early stopping       │            │
     └───────────┬─────────────┘            │
                 │                          │
                 ▼                          ▼
          ┌─────────────┐           ┌─────────────┐
          │  最佳模型    │           │  测试评估    │
          │  保存       │──────────▶│             │
          └─────────────┘           │ - 准确率     │
                                    │ - F1分数     │
                                    │ - 混淆矩阵   │
                                    └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   预测输出   │
                                    │             │
                                    │ - 岩性类别   │
                                    │ - 置信度     │
                                    │ - 可视化     │
                                    └─────────────┘
```

### PyG Data 对象结构

```python
Data(
    x=[N, F],           # 节点特征矩阵
                        #   N = 节点数 (数据点数)
                        #   F = 特征维度 (3坐标 + K个属性)

    edge_index=[2, E],  # 边索引
                        #   E = 边数
                        #   [0, :] = 源节点索引
                        #   [1, :] = 目标节点索引

    edge_weight=[E],    # 边权重 (基于距离的相似度)

    y=[N],              # 节点标签 (岩性类别索引)

    train_mask=[N],     # 训练集掩码 (bool)
    val_mask=[N],       # 验证集掩码 (bool)
    test_mask=[N],      # 测试集掩码 (bool)

    coords=[N, 3]       # 原始坐标 (用于可视化)
)
```

---

## 模型架构

### GNN消息传递机制

GNN的核心是消息传递(Message Passing):

```
                    邻居节点 j
                   ┌─────────┐
                   │  h_j    │
                   └────┬────┘
                        │ 消息
                        ▼
┌─────────┐       ┌─────────┐       ┌─────────┐
│ 邻居 i  │──────▶│ 聚合    │◀──────│ 邻居 k  │
│  h_i    │       │ AGG     │       │  h_k    │
└─────────┘       └────┬────┘       └─────────┘
                       │
                       ▼
                 ┌───────────┐
                 │  更新      │
                 │  h_v^new   │
                 │           │
                 │ = UPDATE( │
                 │   h_v,    │
                 │   AGG({   │
                 │    h_u    │
                 │   })      │
                 │ )         │
                 └───────────┘
```

**各模型的聚合方式:**

| 模型 | 聚合函数 | 更新函数 |
|------|----------|----------|
| GCN | 归一化求和 | 线性变换 |
| GraphSAGE | Mean/Max/LSTM | Concat + 线性 |
| GAT | 注意力加权求和 | 线性变换 |

### 损失函数

使用带权重的交叉熵损失:

```
L = -Σ w_c * y_c * log(p_c)

其中:
- w_c: 类别c的权重 (处理不平衡)
- y_c: 真实标签 (one-hot)
- p_c: 预测概率
```

---

## 扩展指南

### 添加新模型

1. 在 `src/models.py` 中定义新模型类:

```python
class MyNewGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        # 定义层

    def forward(self, x, edge_index, edge_weight=None):
        # 前向传播
        return out

    def predict(self, x, edge_index, edge_weight=None):
        return torch.argmax(self.forward(x, edge_index, edge_weight), dim=1)
```

2. 在 `get_model()` 函数中注册:

```python
models = {
    'gcn': GeoGCN,
    'graphsage': GeoGraphSAGE,
    'gat': GeoGAT,
    'geo3d': Geo3DGNN,
    'mynew': MyNewGNN  # 添加这行
}
```

### 添加新特征

1. 在数据文件中添加新列
2. 确保列名不在排除列表中:

```python
# data_loader.py
exclude_cols = {'x', 'y', 'z', 'lithology', 'label', 'borehole_id', 'id'}
```

3. 特征将自动被包含在处理中

### 自定义图构建

在 `BoreholeDataProcessor.build_graph()` 中添加新方法:

```python
elif self.graph_type == 'custom':
    # 你的图构建逻辑
    edge_index = ...
    edge_weight = ...
```

### 添加新的可视化

在 `app.py` 中添加新的绘图函数:

```python
def plot_my_visualization(data, predictions):
    fig = go.Figure()
    # Plotly绑图逻辑
    return fig
```

然后在对应的标签页中调用即可。

---

## 性能优化建议

1. **大规模数据**: 使用 GraphSAGE + 小批量采样
2. **GPU加速**: 确保数据和模型都在GPU上
3. **内存优化**: 减少 k_neighbors 或使用稀疏矩阵
4. **训练加速**: 使用混合精度训练 (torch.cuda.amp)

---

## 参考资料

- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
- [Graph Neural Networks 综述](https://arxiv.org/abs/1901.00596)
- [GraphSAGE 论文](https://arxiv.org/abs/1706.02216)
- [GAT 论文](https://arxiv.org/abs/1710.10903)
