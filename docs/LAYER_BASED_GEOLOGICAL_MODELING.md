# 基于层序累加的GNN地质建模方案

## 目录
1. [问题分析](#1-问题分析)
2. [层序累加建模原理](#2-层序累加建模原理)
3. [GNN厚度预测模型设计](#3-gnn厚度预测模型设计)
4. [实现方案详解](#4-实现方案详解)
5. [与当前方法对比](#5-与当前方法对比)
6. [技术实现细节](#6-技术实现细节)

---

## 1. 问题分析

### 1.1 当前方法的问题

当前的"直接预测法"采用的是**逐点分类**策略：

```
对于每个网格点 (x, y, z):
    预测该点的岩性类别 = argmax(GNN(features))
```

这种方法存在以下根本性问题：

| 问题 | 原因 | 后果 |
|------|------|------|
| **岩体冲突** | 相邻点独立预测，无约束 | 同一位置出现多种岩性交错 |
| **空缺区域** | 预测边界不连续 | 某些区域没有被正确填充 |
| **违背地质规律** | 忽略了地层的层序关系 | 模型不符合实际沉积规律 |
| **边界锯齿** | 分类决策的离散性 | 岩层边界不平滑 |

### 1.2 地质建模的物理本质

真实的地质建模应该遵循**沉积地层学**原理：

```
┌─────────────────────────────────────────────────────────────┐
│                    地层沉积规律示意                           │
└─────────────────────────────────────────────────────────────┘

        地表 ═══════════════════════════════════════════
              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← 第N层 (最新)
        ─────────────────────────────────────────────────
              ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ← 第N-1层
        ─────────────────────────────────────────────────
              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 第N-2层
        ─────────────────────────────────────────────────
              ████████████████████████████████████████  ← 第1层 (最老)
        ═════════════════════════════════════════════════
        底面 (基准面)

关键特征:
1. 地层是按时间顺序沉积的，从下到上依次叠加
2. 每一层在空间上是连续的曲面
3. 相邻层之间不会相交（除非有断层等构造）
4. 每个位置的岩性由其所在的层决定
```

### 1.3 核心洞察

**问题本质的转换**：

| 原方法 | 新方法 |
|--------|--------|
| 预测每个点的**岩性类别**（分类） | 预测每层的**厚度分布**（回归） |
| 输出：离散类别标签 | 输出：连续厚度值 |
| 可能产生冲突 | 数学上保证无冲突 |
| 边界不连续 | 曲面天然连续 |

---

## 2. 层序累加建模原理

### 2.1 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│                   层序累加建模流程                            │
└─────────────────────────────────────────────────────────────┘

步骤1: 确定底面
────────────────────────────────────────────────────────────
    Z_bottom(x, y) = 插值(所有钻孔的最大深度)

    这是整个模型的基准面

步骤2: 第1层（最深层）建模
────────────────────────────────────────────────────────────
    thickness_1(x, y) = GNN预测 或 插值(钻孔第1层厚度)

    Z_top_1(x, y) = Z_bottom(x, y) + thickness_1(x, y)

    第1层岩体 = Z_bottom 和 Z_top_1 之间的空间

步骤3: 第2层建模
────────────────────────────────────────────────────────────
    thickness_2(x, y) = GNN预测 或 插值(钻孔第2层厚度)

    Z_top_2(x, y) = Z_top_1(x, y) + thickness_2(x, y)

    第2层岩体 = Z_top_1 和 Z_top_2 之间的空间

...以此类推...

步骤N: 第N层（最浅层）建模
────────────────────────────────────────────────────────────
    thickness_N(x, y) = GNN预测 或 插值(钻孔第N层厚度)

    Z_top_N(x, y) = Z_top_{N-1}(x, y) + thickness_N(x, y)

    第N层岩体 = Z_top_{N-1} 和 Z_top_N 之间的空间

    注: Z_top_N 即为地表面
```

### 2.2 数学表达

设有 $N$ 个地层，从深到浅编号为 $1, 2, ..., N$。

**底面定义**：
$$Z_0(x, y) = Z_{bottom}(x, y)$$

**第 $i$ 层顶面**：
$$Z_i(x, y) = Z_{i-1}(x, y) + T_i(x, y)$$

其中 $T_i(x, y)$ 是第 $i$ 层在位置 $(x, y)$ 的厚度。

**第 $i$ 层岩体区域**：
$$Layer_i = \{(x, y, z) \mid Z_{i-1}(x, y) \leq z < Z_i(x, y)\}$$

**关键性质**：
- **无冲突**：$Layer_i \cap Layer_j = \emptyset$ 当 $i \neq j$（各层不相交）
- **无空缺**：$\bigcup_{i=1}^{N} Layer_i = [Z_0, Z_N]$（完全覆盖）
- **连续性**：$Z_i(x, y)$ 是连续曲面（由插值保证）

### 2.3 图示说明

```
                        钻孔A        钻孔B        钻孔C
                          │            │            │
    地表 ═════════════════╪════════════╪════════════╪═════════
                          │            │            │
         ░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░│░░░░░░░░░░░░│░░░░░░░░  层3: 表土
         ─────────────────┼────────────┼────────────┼─────────  Z_3(x,y)
                          │ T3_A       │ T3_B       │ T3_C
         ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│▒▒▒▒▒▒▒▒▒▒▒▒│▒▒▒▒▒▒▒▒▒▒▒▒│▒▒▒▒▒▒▒▒  层2: 砂岩
         ─────────────────┼────────────┼────────────┼─────────  Z_2(x,y)
                          │ T2_A       │ T2_B       │ T2_C
         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓  层1: 泥岩
         ═════════════════╧════════════╧════════════╧═════════  Z_0(x,y) 底面
                          │ T1_A       │ T1_B       │ T1_C

    任意位置 (x, y) 的建模:
    ┌────────────────────────────────────────────────────────┐
    │  Z_0(x,y) = 插值底面深度                                │
    │  Z_1(x,y) = Z_0(x,y) + T1(x,y)  ← GNN预测T1            │
    │  Z_2(x,y) = Z_1(x,y) + T2(x,y)  ← GNN预测T2            │
    │  Z_3(x,y) = Z_2(x,y) + T3(x,y)  ← GNN预测T3            │
    └────────────────────────────────────────────────────────┘
```

### 2.4 处理岩层缺失

在实际钻孔数据中，某些钻孔可能缺少某些岩层：

```
钻孔A: 表土(5m) → 砂岩(10m) → 泥岩(8m) → 煤层(3m)
钻孔B: 表土(4m) → 砂岩(12m) → [缺失泥岩] → 煤层(4m)
钻孔C: 表土(6m) → [缺失砂岩] → 泥岩(15m) → 煤层(2m)
```

**处理策略**：
1. **厚度为0**：缺失的层在该钻孔处厚度记为0
2. **插值处理**：GNN/插值会根据周围钻孔预测合理的厚度
3. **尖灭现象**：某层在某区域厚度趋近于0是正常的地质现象

```python
# 处理示例
layer_thickness = {
    '钻孔A': {'表土': 5, '砂岩': 10, '泥岩': 8, '煤层': 3},
    '钻孔B': {'表土': 4, '砂岩': 12, '泥岩': 0, '煤层': 4},  # 泥岩厚度为0
    '钻孔C': {'表土': 6, '砂岩': 0, '泥岩': 15, '煤层': 2},  # 砂岩厚度为0
}
```

---

## 3. GNN厚度预测模型设计

### 3.1 问题重新定义

**原问题（分类）**：
```
输入: 点坐标 (x, y, z) + 特征
输出: 岩性类别 (0, 1, 2, ..., N-1)
损失函数: CrossEntropyLoss
```

**新问题（回归）**：
```
输入: 平面坐标 (x, y) + 地质特征
输出: 各层厚度 [T1, T2, ..., TN]
损失函数: MSELoss 或 SmoothL1Loss
```

### 3.2 GNN模型架构调整

```
┌─────────────────────────────────────────────────────────────┐
│               GNN厚度预测模型架构                             │
└─────────────────────────────────────────────────────────────┘

                    输入特征 (每个钻孔节点)
                    ┌─────────────────────┐
                    │ x, y 坐标           │
                    │ 各层厚度 [T1...TN]  │
                    │ 总深度              │
                    │ 地质特征            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   GNN编码器         │
                    │   (GraphSAGE/GAT)   │
                    │   学习空间相关性     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   回归头            │
                    │   Linear → ReLU    │
                    │   → Linear(N)      │
                    │   输出N个厚度值     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 输出: [T1, T2...TN] │
                    │ 每层的预测厚度       │
                    └─────────────────────┘
```

### 3.3 训练数据构建

```python
# 从钻孔数据构建训练样本
def build_thickness_dataset(df, layer_order):
    """
    构建厚度预测数据集

    Args:
        df: 钻孔数据DataFrame
        layer_order: 岩层顺序列表，从深到浅

    Returns:
        节点特征, 标签(各层厚度)
    """
    samples = []

    for bh_id in df['borehole_id'].unique():
        bh_data = df[df['borehole_id'] == bh_id]

        # 提取坐标
        x, y = bh_data['x'].iloc[0], bh_data['y'].iloc[0]

        # 提取各层厚度
        thicknesses = []
        for layer_name in layer_order:
            layer_data = bh_data[bh_data['lithology'] == layer_name]
            if len(layer_data) > 0:
                thickness = layer_data['layer_thickness'].sum()
            else:
                thickness = 0.0  # 该层缺失
            thicknesses.append(thickness)

        samples.append({
            'x': x,
            'y': y,
            'features': [...],  # 其他地质特征
            'thicknesses': thicknesses  # 标签
        })

    return samples
```

### 3.4 损失函数设计

```python
class ThicknessLoss(nn.Module):
    """厚度预测损失函数"""

    def __init__(self, layer_weights=None):
        super().__init__()
        self.layer_weights = layer_weights  # 可以给不同层不同权重

    def forward(self, pred_thickness, true_thickness):
        """
        pred_thickness: [batch, num_layers] 预测厚度
        true_thickness: [batch, num_layers] 真实厚度
        """
        # 基础MSE损失
        mse_loss = F.mse_loss(pred_thickness, true_thickness, reduction='none')

        # 按层加权
        if self.layer_weights is not None:
            mse_loss = mse_loss * self.layer_weights

        # 非负约束：厚度不能为负
        negative_penalty = F.relu(-pred_thickness).sum()

        # 总厚度一致性约束（可选）
        total_pred = pred_thickness.sum(dim=1)
        total_true = true_thickness.sum(dim=1)
        total_loss = F.mse_loss(total_pred, total_true)

        return mse_loss.mean() + 0.1 * negative_penalty + 0.1 * total_loss
```

---

## 4. 实现方案详解

### 4.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                   层序累加建模完整流程                        │
└─────────────────────────────────────────────────────────────┘

Phase 1: 数据准备
────────────────────────────────────────────────────────────
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  加载钻孔    │ ──→ │  确定层序    │ ──→ │  提取厚度    │
    │  数据        │     │  (从深到浅)  │     │  数据        │
    └─────────────┘     └─────────────┘     └─────────────┘

Phase 2: 模型训练
────────────────────────────────────────────────────────────
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  构建图结构  │ ──→ │  训练GNN    │ ──→ │  验证模型    │
    │  (钻孔邻接)  │     │  厚度预测    │     │  精度        │
    └─────────────┘     └─────────────┘     └─────────────┘

Phase 3: 三维建模
────────────────────────────────────────────────────────────
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  创建网格    │ ──→ │  预测厚度    │ ──→ │  累加构建    │
    │  (x, y)     │     │  T(x,y)     │     │  曲面        │
    └─────────────┘     └─────────────┘     └─────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────┐
                                          │  体素化填充  │
                                          │  生成模型    │
                                          └─────────────┘
```

### 4.2 核心类设计

```python
class LayerBasedGeologicalModeling:
    """基于层序累加的地质建模"""

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        layer_order: List[str] = None,  # 岩层顺序，从深到浅
        use_gnn: bool = True,  # 是否使用GNN预测厚度
        interpolation_method: str = 'rbf'  # 不用GNN时的插值方法
    ):
        self.resolution = resolution
        self.layer_order = layer_order
        self.use_gnn = use_gnn
        self.interpolation_method = interpolation_method

        # 结果存储
        self.surfaces = {}  # 各层曲面 {layer_name: Z(x,y)}
        self.lithology_3d = None  # 最终岩性模型

    def build_model(self, df, trainer=None):
        """构建三维地质模型"""

        # 1. 确定岩层顺序
        if self.layer_order is None:
            self.layer_order = self._infer_layer_order(df)

        # 2. 构建底面
        bottom_surface = self._build_bottom_surface(df)

        # 3. 逐层构建
        current_surface = bottom_surface
        for i, layer_name in enumerate(self.layer_order):

            # 预测该层厚度
            if self.use_gnn and trainer is not None:
                thickness = self._predict_thickness_gnn(df, layer_name, trainer)
            else:
                thickness = self._interpolate_thickness(df, layer_name)

            # 累加得到顶面
            top_surface = current_surface + thickness

            # 存储曲面
            self.surfaces[layer_name] = {
                'bottom': current_surface,
                'top': top_surface,
                'thickness': thickness
            }

            # 更新当前面
            current_surface = top_surface

        # 4. 体素化
        self._voxelize()

        return self.lithology_3d
```

### 4.3 厚度预测实现

```python
def _predict_thickness_gnn(self, df, layer_name, trainer):
    """使用GNN预测某层的厚度分布"""

    # 获取网格坐标
    nx, ny, nz = self.resolution
    x_grid = np.linspace(self.bounds['x'][0], self.bounds['x'][1], nx)
    y_grid = np.linspace(self.bounds['y'][0], self.bounds['y'][1], ny)

    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # 构建预测图（将网格点连接到钻孔节点）
    # ... 与之前类似的图构建逻辑 ...

    # GNN预测
    with torch.no_grad():
        # 获取该层的厚度预测
        layer_idx = self.layer_order.index(layer_name)
        all_thickness = trainer.model(extended_data)  # [N, num_layers]
        predicted_thickness = all_thickness[:, layer_idx]

    # 重塑为网格
    thickness_grid = predicted_thickness.reshape(nx, ny)

    # 后处理：确保非负
    thickness_grid = np.maximum(thickness_grid, 0)

    return thickness_grid


def _interpolate_thickness(self, df, layer_name):
    """使用传统插值预测厚度"""

    # 提取该层的厚度数据
    layer_data = df[df['lithology'] == layer_name].groupby('borehole_id').agg({
        'x': 'first',
        'y': 'first',
        'layer_thickness': 'sum'
    }).reset_index()

    # 对于缺失该层的钻孔，厚度为0
    all_boreholes = df.groupby('borehole_id')[['x', 'y']].first().reset_index()
    layer_data = all_boreholes.merge(
        layer_data[['borehole_id', 'layer_thickness']],
        on='borehole_id',
        how='left'
    ).fillna(0)

    # 插值
    from scipy.interpolate import RBFInterpolator

    points = layer_data[['x', 'y']].values
    values = layer_data['layer_thickness'].values

    interpolator = RBFInterpolator(points, values, kernel='thin_plate_spline')

    # 在网格上插值
    thickness_grid = interpolator(self.grid_xy).reshape(self.nx, self.ny)

    # 确保非负
    thickness_grid = np.maximum(thickness_grid, 0)

    return thickness_grid
```

### 4.4 体素化实现

```python
def _voxelize(self):
    """将层曲面转换为体素模型"""

    nx, ny, nz = self.resolution
    z_grid = np.linspace(self.bounds['z'][0], self.bounds['z'][1], nz)

    # 初始化岩性数组
    self.lithology_3d = np.zeros((nx, ny, nz), dtype=np.int32)

    # 从下到上填充
    for layer_idx, layer_name in enumerate(self.layer_order):
        surface_info = self.surfaces[layer_name]
        bottom = surface_info['bottom']  # [nx, ny]
        top = surface_info['top']        # [nx, ny]

        for i in range(nx):
            for j in range(ny):
                z_bottom = bottom[i, j]
                z_top = top[i, j]

                # 找到该层对应的z索引范围
                k_start = np.searchsorted(z_grid, z_bottom)
                k_end = np.searchsorted(z_grid, z_top)

                # 填充该层岩性
                self.lithology_3d[i, j, k_start:k_end] = layer_idx
```

---

## 5. 与当前方法对比

### 5.1 方法对比表

| 特性 | 逐点分类法（当前） | 层序累加法（新） |
|------|-------------------|-----------------|
| **核心任务** | 分类（预测类别） | 回归（预测厚度） |
| **输出空间** | 离散（N个类别） | 连续（厚度值） |
| **物理约束** | 无内置约束 | 层序关系内置 |
| **冲突问题** | 可能产生 | 数学上不可能 |
| **空缺问题** | 可能产生 | 数学上不可能 |
| **边界连续性** | 不保证 | 由曲面插值保证 |
| **可解释性** | 低 | 高（符合地质逻辑） |
| **计算复杂度** | O(nx*ny*nz) | O(nx*ny*N_layers) |

### 5.2 优势详解

**1. 无冲突保证**
```
逐点法：
    点(100, 200, -50): 预测=砂岩
    点(100, 200, -51): 预测=泥岩
    点(100, 200, -52): 预测=砂岩  ← 不合理的交错！

层序法：
    Z_1(100, 200) = -80  (泥岩底面)
    Z_2(100, 200) = -60  (泥岩顶面=砂岩底面)
    Z_3(100, 200) = -40  (砂岩顶面)

    → -80到-60: 泥岩
    → -60到-40: 砂岩
    → 严格分层，无交错
```

**2. 连续曲面**
```
层序法使用曲面插值，天然保证：
- 同一层的顶/底面是光滑曲面
- 不会出现孤立的"岛屿"
- 边界过渡自然
```

**3. 计算效率**
```
逐点法：需要预测 nx * ny * nz 个点
层序法：只需预测 nx * ny 个点的 N_layers 个厚度

例如 50x50x50 网格，5个岩层：
逐点法：125,000 次预测
层序法：2,500 次预测（每次输出5个值）
效率提升约 50 倍！
```

### 5.3 潜在挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| **岩层顺序不一致** | 基于统计确定主要层序，异常作为局部变化处理 |
| **岩层尖灭** | 允许厚度为0，插值会自然处理过渡 |
| **断层构造** | 可以分区建模，或在曲面上引入不连续 |
| **透镜体** | 作为独立层处理，或使用后处理添加 |
| **岩层重复** | 同名岩层的不同层位分别建模 |

---

## 6. 技术实现细节

### 6.1 岩层顺序确定

```python
def infer_layer_order(df):
    """从钻孔数据推断岩层顺序"""

    # 统计每种岩性的平均深度
    layer_depths = df.groupby('lithology').agg({
        'z': 'mean',  # 平均高程（负值表示深度）
        'borehole_id': 'count'  # 出现次数
    }).reset_index()

    # 按平均高程排序（从小到大 = 从深到浅）
    layer_depths = layer_depths.sort_values('z')

    # 返回层序
    layer_order = layer_depths['lithology'].tolist()

    return layer_order
```

### 6.2 底面确定

```python
def build_bottom_surface(df, method='max_depth'):
    """构建模型底面"""

    if method == 'max_depth':
        # 使用各钻孔最大深度插值
        bottom_data = df.groupby('borehole_id').agg({
            'x': 'first',
            'y': 'first',
            'z': 'min'  # 最深点
        }).reset_index()

    elif method == 'fixed':
        # 使用固定深度
        z_min = df['z'].min()
        return np.full((nx, ny), z_min)

    # 插值
    interpolator = RBFInterpolator(
        bottom_data[['x', 'y']].values,
        bottom_data['z'].values
    )

    bottom_surface = interpolator(grid_xy).reshape(nx, ny)

    return bottom_surface
```

### 6.3 GNN回归模型修改

```python
class GNNThicknessPredictor(nn.Module):
    """GNN厚度预测模型"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_output_layers: int,  # 输出的岩层数量
        dropout: float = 0.3
    ):
        super().__init__()

        # GNN编码器（与原模型类似）
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # 回归头：输出每层厚度
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_output_layers)
        )

        # 最后一层用Softplus确保输出非负
        self.output_activation = nn.Softplus()

    def forward(self, x, edge_index, edge_weight=None):
        # 编码
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # 回归
        thickness = self.regressor(x)

        # 确保非负
        thickness = self.output_activation(thickness)

        return thickness
```

### 6.4 文件结构规划

```
src/
├── layer_modeling.py           # 新增：层序累加建模
│   ├── LayerBasedGeologicalModeling
│   ├── ThicknessPredictor
│   └── LayerDataProcessor
├── models.py                   # 修改：添加回归模型
│   └── GNNThicknessPredictor
├── trainer.py                  # 修改：支持回归训练
│   └── ThicknessTrainer
├── gnn_modeling.py             # 保留：直接预测法对比
└── modeling.py                 # 保留：传统插值法对比
```

---

## 附录

### A. 数据格式要求

```python
# 钻孔数据需要包含以下字段
required_columns = [
    'borehole_id',      # 钻孔编号
    'x', 'y',           # 平面坐标
    'lithology',        # 岩性名称
    'top_depth',        # 层顶深度
    'bottom_depth',     # 层底深度
    'layer_thickness',  # 层厚度
    'layer_order'       # 层序号（从上到下）
]
```

### B. 参考文献

1. 地层学原理与三维地质建模
2. Graph Neural Networks for Spatial Prediction
3. Geological Surface Interpolation Methods

---

*文档版本: 1.0*
*创建日期: 2024*
*方案: 层序累加建模*
