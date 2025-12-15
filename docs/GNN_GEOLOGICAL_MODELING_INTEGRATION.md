# GNN模型与三维地质建模整合方案

## 目录
1. [问题背景](#1-问题背景)
2. [当前系统架构分析](#2-当前系统架构分析)
3. [三种整合方案详解](#3-三种整合方案详解)
4. [对照实验设计](#4-对照实验设计)
5. [实现路线图](#5-实现路线图)
6. [技术细节与代码结构](#6-技术细节与代码结构)

---

## 1. 问题背景

### 1.1 核心问题

当前系统存在一个关键的**断层**：

- **GNN模型**：训练完成后可以预测任意位置的岩性类别（当前验证准确率约77%）
- **地质建模**：仍然使用传统的RBF/IDW插值方法，完全基于原始钻孔数据
- **结果**：模型的预测能力没有被利用到最终的三维地质模型中

### 1.2 为什么需要整合？

| 方法 | 优势 | 局限 |
|------|------|------|
| **传统插值** | 物理意义明确，保证地层连续性 | 无法学习复杂的非线性关系 |
| **GNN预测** | 可学习空间相关性和特征模式 | 可能产生地质上不合理的预测 |
| **整合方案** | 结合两者优势 | 需要设计合理的融合机制 |

### 1.3 目标

将训练好的GNN模型的预测能力与传统地质建模的物理约束相结合，生成更准确、更符合地质规律的三维模型。

---

## 2. 当前系统架构分析

### 2.1 数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        当前系统数据流                                  │
└──────────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │  钻孔数据    │
                         │  (CSV/Excel) │
                         └──────┬──────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
         ┌──────────────────┐    ┌──────────────────┐
         │  data_loader.py  │    │   modeling.py    │
         │  图数据构建       │    │   传统地质建模   │
         └────────┬─────────┘    └────────┬─────────┘
                  │                       │
                  ▼                       ▼
         ┌──────────────────┐    ┌──────────────────┐
         │  models.py       │    │  RBF/IDW插值     │
         │  GNN模型定义      │    │  地层界面        │
         └────────┬─────────┘    └────────┬─────────┘
                  │                       │
                  ▼                       ▼
         ┌──────────────────┐    ┌──────────────────┐
         │  trainer.py      │    │  体素化模型      │
         │  模型训练/预测    │    │  (独立输出)      │
         └────────┬─────────┘    └────────┬─────────┘
                  │                       │
                  ▼                       ▼
         ┌──────────────────┐    ┌──────────────────┐
         │  预测结果        │    │  3D地质模型      │
         │  (未使用!)       │    │  (最终输出)      │
         └──────────────────┘    └──────────────────┘
                  │                       │
                  └───────── ✗ ───────────┘
                        (未连接)
```

### 2.2 关键代码位置

#### 模型预测 (`trainer.py` 第580-619行)
```python
def predict(self, data: Data, return_probs: bool = False):
    """
    预测所有节点的岩性类别
    返回: predictions (类别索引), probabilities (各类别概率)
    """
    out = self.model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)
    pred = out.argmax(dim=1)
    return predictions, probabilities
```

#### 传统建模 (`modeling.py` 第246-350行)
```python
def build_stratigraphic_model(self, df, lithology_classes):
    """
    传统层状建模流程:
    1. 构建三维网格
    2. 提取地层界面 (从钻孔数据)
    3. RBF插值每个界面
    4. 根据界面位置分配岩性
    """
```

### 2.3 问题诊断

| 组件 | 状态 | 问题 |
|------|------|------|
| `trainer.predict()` | ✅ 正常 | 可输出预测，但结果未被使用 |
| `modeling.build_stratigraphic_model()` | ✅ 正常 | 只使用原始钻孔数据 |
| 两者连接 | ❌ 缺失 | 没有将预测结果传递给建模模块 |

---

## 3. 三种整合方案详解

### 3.1 方案一：直接预测法 (Direct Prediction)

#### 核心思想
完全使用GNN模型直接预测三维网格中每个点的岩性类别，绕过传统插值。

#### 工作流程
```
┌─────────────────────────────────────────────────────────────┐
│                    方案一：直接预测法                         │
└─────────────────────────────────────────────────────────────┘

  钻孔数据 ──→ 构建训练图 ──→ 训练GNN ──→ 模型收敛
                                              │
                                              ▼
  三维网格点 ──→ 构建预测图 ──→ GNN推理 ──→ 每个体素的岩性
      │              │
      │              └── 将网格点作为新节点加入图
      │                  连接到最近的钻孔节点
      │
      └── 为每个网格点生成特征:
          - 空间坐标 (x, y, z)
          - 深度特征
          - 与已知钻孔的距离特征
```

#### 实现要点

```python
class DirectPredictionModeling:
    """直接预测法建模"""

    def __init__(self, trainer, data_processor):
        self.trainer = trainer  # 训练好的GNN模型
        self.processor = data_processor

    def build_model(self, resolution=(50, 50, 50)):
        # 1. 创建三维网格
        grid_points = self._create_grid(resolution)

        # 2. 为网格点生成特征
        grid_features = self._generate_features(grid_points)

        # 3. 构建包含网格点的扩展图
        extended_data = self._build_extended_graph(grid_features)

        # 4. 使用GNN预测所有网格点
        predictions, probabilities = self.trainer.predict(
            extended_data,
            return_probs=True
        )

        # 5. 提取网格点的预测结果
        grid_predictions = predictions[self.grid_mask]

        return grid_predictions.reshape(resolution)
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 充分利用GNN学习到的模式 | 可能产生地质上不连续的结果 |
| 实现相对简单 | 远离钻孔的区域预测可能不准 |
| 可以给出预测置信度 | 不保证地层的层序关系 |
| 计算效率高 | 依赖模型泛化能力 |

#### 适用场景
- 钻孔数据密集
- 地质条件相对简单
- 需要快速建模

---

### 3.2 方案二：混合融合法 (Hybrid Fusion)

#### 核心思想
将GNN预测结果作为传统插值的"软约束"或辅助信息，两种方法的结果按权重融合。

#### 工作流程
```
┌─────────────────────────────────────────────────────────────┐
│                    方案二：混合融合法                         │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │    钻孔数据      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │   传统插值建模    │          │    GNN预测       │
    │   (RBF/IDW)      │          │   (岩性+概率)    │
    └────────┬─────────┘          └────────┬─────────┘
             │                              │
             ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  插值岩性结果     │          │  预测岩性结果     │
    │  P_interp(x,y,z) │          │  P_gnn(x,y,z)    │
    └────────┬─────────┘          └────────┬─────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
                  ┌──────────────────┐
                  │   加权融合       │
                  │ P = α·P_interp   │
                  │   + (1-α)·P_gnn  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  最终3D模型      │
                  └──────────────────┘
```

#### 融合策略

**策略A：全局固定权重**
```python
alpha = 0.5  # 可调参数
final_result = alpha * interp_result + (1 - alpha) * gnn_result
```

**策略B：基于距离的动态权重**
```python
def compute_weight(point, boreholes):
    """距离钻孔越近，传统插值权重越高"""
    min_dist = min_distance_to_boreholes(point, boreholes)

    # 距离越远，GNN权重越高
    alpha = np.exp(-min_dist / scale)  # scale为超参数

    return alpha  # 传统插值权重
```

**策略C：基于置信度的权重**
```python
def compute_weight(interp_confidence, gnn_confidence):
    """根据两种方法的置信度分配权重"""
    total = interp_confidence + gnn_confidence
    alpha = interp_confidence / total
    return alpha
```

#### 实现要点

```python
class HybridFusionModeling:
    """混合融合法建模"""

    def __init__(self, trainer, resolution=(50, 50, 50)):
        self.trainer = trainer
        self.resolution = resolution
        self.fusion_strategy = 'distance'  # 'fixed', 'distance', 'confidence'

    def build_model(self, df, lithology_classes):
        # 1. 传统插值建模
        trad_model = StratigraphicModel3D(resolution=self.resolution)
        trad_model.build_stratigraphic_model(df, lithology_classes)
        trad_lithology, trad_confidence = trad_model.get_voxel_model()

        # 2. GNN预测建模
        gnn_lithology, gnn_probs = self._gnn_predict_grid()

        # 3. 计算融合权重
        weights = self._compute_fusion_weights(df)

        # 4. 融合结果
        if self.fusion_strategy == 'probability':
            # 概率级融合 (软融合)
            fused_probs = weights * trad_probs + (1 - weights) * gnn_probs
            final_lithology = np.argmax(fused_probs, axis=-1)
        else:
            # 决策级融合 (硬融合)
            final_lithology = self._decision_fusion(
                trad_lithology, gnn_lithology, weights
            )

        return final_lithology
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 结合两种方法的优势 | 实现复杂度较高 |
| 可调节融合权重 | 需要调优融合参数 |
| 保留一定的地质约束 | 可能引入两种方法的误差 |
| 灵活性高 | 计算量增加 |

#### 适用场景
- 需要平衡精度和地质合理性
- 对结果可解释性有要求
- 有足够的验证数据调优权重

---

### 3.3 方案三：两阶段优化法 (Two-Stage Optimization)

#### 核心思想
先用传统方法建立符合地质规律的初始模型，再用GNN预测结果对边界和不确定区域进行优化。

#### 工作流程
```
┌─────────────────────────────────────────────────────────────┐
│                  方案三：两阶段优化法                         │
└─────────────────────────────────────────────────────────────┘

阶段一：初始模型构建
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  钻孔数据 ──→ 传统插值 ──→ 初始3D模型 ──→ 识别不确定区域    │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                     │
│                    │ 高置信度区域(保留)│                     │
│                    │ 低置信度区域(标记)│                     │
│                    │ 边界区域(标记)    │                     │
│                    └──────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

阶段二：GNN优化
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  标记区域 ──→ GNN预测 ──→ 地质约束检验 ──→ 选择性更新       │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                     │
│                    │ 通过约束 → 更新   │                     │
│                    │ 未通过 → 保留原值 │                     │
│                    └──────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 不确定区域识别

```python
def identify_uncertain_regions(model):
    """识别需要优化的区域"""
    lithology_3d, confidence_3d = model.get_voxel_model()

    uncertain_mask = np.zeros_like(lithology_3d, dtype=bool)

    # 1. 低置信度区域
    low_confidence = confidence_3d < 0.7
    uncertain_mask |= low_confidence

    # 2. 岩性边界区域 (岩性变化的邻域)
    boundary_mask = detect_lithology_boundaries(lithology_3d)
    uncertain_mask |= boundary_mask

    # 3. 远离钻孔的区域
    far_from_borehole = compute_distance_field(boreholes) > threshold
    uncertain_mask |= far_from_borehole

    return uncertain_mask
```

#### 地质约束检验

```python
def geological_constraints_check(original, predicted, position):
    """
    检验GNN预测是否符合地质约束
    """
    constraints_satisfied = True

    # 约束1: 层序关系 (上覆地层不能比下伏地层更老)
    if not check_stratigraphic_order(predicted, position):
        constraints_satisfied = False

    # 约束2: 空间连续性 (同一层在空间上应该连续)
    if not check_spatial_continuity(predicted, position, neighbors):
        constraints_satisfied = False

    # 约束3: 厚度合理性 (地层厚度应在合理范围内)
    if not check_thickness_validity(predicted, position):
        constraints_satisfied = False

    return constraints_satisfied
```

#### 实现要点

```python
class TwoStageOptimizationModeling:
    """两阶段优化法建模"""

    def __init__(self, trainer, resolution=(50, 50, 50)):
        self.trainer = trainer
        self.resolution = resolution

    def build_model(self, df, lithology_classes):
        # ========== 阶段一：初始模型 ==========
        print("阶段一：构建初始模型...")

        # 传统插值建模
        initial_model = StratigraphicModel3D(resolution=self.resolution)
        initial_model.build_stratigraphic_model(df, lithology_classes)
        lithology_3d, confidence_3d = initial_model.get_voxel_model()

        # 识别不确定区域
        uncertain_mask = self._identify_uncertain_regions(
            lithology_3d, confidence_3d, df
        )

        print(f"  不确定区域: {uncertain_mask.sum()} 个体素 "
              f"({100*uncertain_mask.mean():.1f}%)")

        # ========== 阶段二：GNN优化 ==========
        print("阶段二：GNN优化不确定区域...")

        # 获取不确定区域的GNN预测
        uncertain_indices = np.where(uncertain_mask)
        gnn_predictions = self._predict_uncertain_regions(uncertain_indices)

        # 地质约束检验与选择性更新
        updated_count = 0
        for idx, pred in zip(zip(*uncertain_indices), gnn_predictions):
            if self._check_geological_constraints(lithology_3d, pred, idx):
                lithology_3d[idx] = pred
                updated_count += 1

        print(f"  更新体素: {updated_count} 个 "
              f"({100*updated_count/uncertain_mask.sum():.1f}%)")

        return lithology_3d
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 保证基本的地质合理性 | 实现最为复杂 |
| 只在必要时使用GNN | 需要定义地质约束规则 |
| 可解释性最强 | 可能过于保守 |
| 风险最小 | 优化效果可能有限 |

#### 适用场景
- 对地质合理性要求最高
- 需要保守稳健的结果
- 有明确的地质约束知识

---

## 4. 对照实验设计

### 4.1 实验框架

```
┌─────────────────────────────────────────────────────────────┐
│                     对照实验设计                             │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │  测试数据集  │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  留出钻孔    │   │  交叉验证    │   │  合成数据    │
│  (真实验证)  │   │  (稳定性)    │   │  (可控实验)  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
              ┌──────────────────┐
              │  四种方法对比     │
              │  1. 传统插值      │
              │  2. 直接预测      │
              │  3. 混合融合      │
              │  4. 两阶段优化    │
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  定量指标   │ │  地质合理性  │ │  可视化对比  │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 4.2 评估指标

#### 定量指标

| 指标 | 计算方法 | 说明 |
|------|----------|------|
| **准确率** | `(预测正确体素数) / (总体素数)` | 整体预测正确率 |
| **Kappa系数** | Cohen's Kappa | 考虑随机一致性的准确率 |
| **各类F1** | 每个岩性类别的F1 | 类别级别性能 |
| **边界准确率** | 边界区域的准确率 | 关键区域性能 |
| **置信度校准** | 预测概率与实际准确率的一致性 | 不确定性估计质量 |

#### 地质合理性指标

| 指标 | 计算方法 | 说明 |
|------|----------|------|
| **层序违规率** | 违反层序关系的体素比例 | 地层顺序合理性 |
| **空间连续性** | 同一岩性的连通性分析 | 地层连续性 |
| **厚度方差** | 预测地层厚度的变异系数 | 厚度稳定性 |
| **边界平滑度** | 岩性边界的梯度分析 | 边界自然程度 |

### 4.3 实验代码框架

```python
class ComparativeExperiment:
    """对照实验框架"""

    def __init__(self, df, lithology_classes, test_boreholes=None):
        self.df = df
        self.lithology_classes = lithology_classes
        self.test_boreholes = test_boreholes or self._select_test_boreholes()

        # 分割训练/测试数据
        self.train_df = df[~df['borehole_id'].isin(self.test_boreholes)]
        self.test_df = df[df['borehole_id'].isin(self.test_boreholes)]

        self.methods = {}
        self.results = {}

    def register_method(self, name, method_class, **kwargs):
        """注册一个建模方法"""
        self.methods[name] = (method_class, kwargs)

    def run_all(self):
        """运行所有方法"""
        for name, (method_class, kwargs) in self.methods.items():
            print(f"\n{'='*50}")
            print(f"运行方法: {name}")
            print(f"{'='*50}")

            method = method_class(**kwargs)
            model = method.build_model(self.train_df, self.lithology_classes)

            # 评估
            metrics = self._evaluate(model, name)
            self.results[name] = {
                'model': model,
                'metrics': metrics
            }

        return self.results

    def _evaluate(self, model, method_name):
        """评估模型"""
        metrics = {}

        # 1. 定量指标：在留出钻孔位置验证
        metrics['accuracy'] = self._compute_accuracy_at_boreholes(model)
        metrics['f1_per_class'] = self._compute_f1_per_class(model)
        metrics['kappa'] = self._compute_kappa(model)

        # 2. 地质合理性指标
        metrics['stratigraphic_violations'] = self._check_stratigraphic_order(model)
        metrics['spatial_continuity'] = self._check_spatial_continuity(model)
        metrics['thickness_variance'] = self._compute_thickness_variance(model)

        return metrics

    def generate_report(self):
        """生成对比报告"""
        report = []
        report.append("# 对照实验结果报告\n")

        # 汇总表格
        report.append("## 定量指标对比\n")
        report.append("| 方法 | 准确率 | Kappa | F1-Macro | 层序违规率 |\n")
        report.append("|------|--------|-------|----------|------------|\n")

        for name, result in self.results.items():
            m = result['metrics']
            report.append(f"| {name} | {m['accuracy']:.4f} | "
                         f"{m['kappa']:.4f} | {m['f1_macro']:.4f} | "
                         f"{m['stratigraphic_violations']:.4f} |\n")

        return "".join(report)
```

### 4.4 实验配置示例

```python
# 实验配置
experiment_config = {
    'methods': {
        'baseline_rbf': {
            'class': 'StratigraphicModel3D',
            'params': {'interpolation_method': 'rbf'}
        },
        'baseline_idw': {
            'class': 'StratigraphicModel3D',
            'params': {'interpolation_method': 'idw'}
        },
        'direct_gnn': {
            'class': 'DirectPredictionModeling',
            'params': {'model_path': 'models/best_gnn.pt'}
        },
        'hybrid_fixed': {
            'class': 'HybridFusionModeling',
            'params': {'fusion_strategy': 'fixed', 'alpha': 0.5}
        },
        'hybrid_distance': {
            'class': 'HybridFusionModeling',
            'params': {'fusion_strategy': 'distance', 'scale': 100}
        },
        'two_stage': {
            'class': 'TwoStageOptimizationModeling',
            'params': {'confidence_threshold': 0.7}
        }
    },
    'evaluation': {
        'test_ratio': 0.2,  # 留出20%钻孔用于验证
        'cross_validation': 5,  # 5折交叉验证
        'metrics': ['accuracy', 'kappa', 'f1', 'geological_validity']
    }
}
```

---

## 5. 实现路线图

### 5.1 开发阶段

```
┌─────────────────────────────────────────────────────────────┐
│                     实现路线图                               │
└─────────────────────────────────────────────────────────────┘

Phase 1: 基础设施 (1-2天)
├── 重构modeling.py，添加模型预测接口
├── 创建统一的建模基类
└── 实现评估指标计算模块

Phase 2: 方案实现 (3-5天)
├── 实现方案一：DirectPredictionModeling
├── 实现方案二：HybridFusionModeling
├── 实现方案三：TwoStageOptimizationModeling
└── 单元测试

Phase 3: 实验系统 (2-3天)
├── 实现ComparativeExperiment类
├── 实现可视化对比功能
└── 自动报告生成

Phase 4: 调优与文档 (2-3天)
├── 超参数调优
├── 结果分析
└── 文档完善
```

### 5.2 文件结构规划

```
src/
├── modeling.py              # 现有：传统建模 (重构)
├── gnn_modeling.py          # 新增：GNN相关建模方法
│   ├── DirectPredictionModeling
│   ├── HybridFusionModeling
│   └── TwoStageOptimizationModeling
├── evaluation.py            # 新增：评估指标模块
│   ├── QuantitativeMetrics
│   └── GeologicalValidityMetrics
├── experiment.py            # 新增：对照实验框架
│   ├── ComparativeExperiment
│   └── ReportGenerator
└── utils/
    ├── grid_utils.py        # 网格操作工具
    └── geological_constraints.py  # 地质约束检验
```

---

## 6. 技术细节与代码结构

### 6.1 核心接口设计

```python
from abc import ABC, abstractmethod

class BaseGeologicalModeling(ABC):
    """地质建模基类"""

    def __init__(self, resolution=(50, 50, 50)):
        self.resolution = resolution
        self.lithology_3d = None
        self.confidence_3d = None
        self.grid_info = None

    @abstractmethod
    def build_model(self, df, lithology_classes, **kwargs):
        """构建三维模型 (子类实现)"""
        pass

    def get_voxel_model(self):
        """获取体素模型"""
        return self.lithology_3d, self.confidence_3d

    def export_vtk(self, filepath):
        """导出VTK格式"""
        # 通用实现
        pass

    def get_statistics(self):
        """获取统计信息"""
        # 通用实现
        pass


class TraditionalModeling(BaseGeologicalModeling):
    """传统插值建模 (当前StratigraphicModel3D的重构)"""
    pass


class GNNBasedModeling(BaseGeologicalModeling):
    """GNN建模基类"""

    def __init__(self, trainer, resolution=(50, 50, 50)):
        super().__init__(resolution)
        self.trainer = trainer

    def _predict_grid(self, grid_features, grid_edges):
        """使用GNN预测网格点"""
        return self.trainer.predict(grid_data, return_probs=True)
```

### 6.2 与现有代码的集成点

#### 修改 `main.py`

```python
# 在 main.py 中添加建模方法选择

def run_modeling(args):
    # ... 现有代码 ...

    # 新增：选择建模方法
    if args.modeling_method == 'traditional':
        model = StratigraphicModel3D(resolution=args.resolution)
        model.build_stratigraphic_model(df, lithology_classes)

    elif args.modeling_method == 'direct_gnn':
        model = DirectPredictionModeling(trainer)
        model.build_model(df, lithology_classes)

    elif args.modeling_method == 'hybrid':
        model = HybridFusionModeling(trainer, fusion_strategy=args.fusion)
        model.build_model(df, lithology_classes)

    elif args.modeling_method == 'two_stage':
        model = TwoStageOptimizationModeling(trainer)
        model.build_model(df, lithology_classes)
```

#### 修改 `configs/config.py`

```python
# 新增建模配置

MODELING_CONFIG = {
    'method': 'hybrid',  # 'traditional', 'direct_gnn', 'hybrid', 'two_stage'
    'resolution': (50, 50, 50),

    # 混合融合参数
    'fusion': {
        'strategy': 'distance',  # 'fixed', 'distance', 'confidence'
        'alpha': 0.5,  # 固定权重时使用
        'distance_scale': 100,  # 距离权重时使用
    },

    # 两阶段优化参数
    'two_stage': {
        'confidence_threshold': 0.7,
        'boundary_width': 2,
        'constraints': ['stratigraphic_order', 'spatial_continuity']
    }
}
```

### 6.3 关键算法伪代码

#### 网格点特征生成

```python
def generate_grid_features(grid_points, borehole_data, feature_config):
    """
    为三维网格点生成特征向量

    特征包括:
    1. 空间坐标 (归一化)
    2. 深度特征
    3. 到最近钻孔的距离
    4. 最近钻孔的岩性分布统计
    """
    features = []

    for point in grid_points:
        x, y, z = point

        # 基础空间特征
        spatial = normalize([x, y, z])

        # 深度特征
        depth_features = compute_depth_features(z, borehole_data)

        # 距离特征
        distances, nearest_boreholes = find_k_nearest_boreholes(point, k=5)
        distance_features = compute_distance_features(distances)

        # 邻域岩性统计
        neighbor_stats = compute_neighbor_lithology_stats(
            point, nearest_boreholes, borehole_data
        )

        # 合并特征
        feature_vector = concatenate([
            spatial,
            depth_features,
            distance_features,
            neighbor_stats
        ])

        features.append(feature_vector)

    return np.array(features)
```

#### 地质约束检验

```python
def check_all_geological_constraints(lithology_3d, position, predicted_class):
    """
    综合检验所有地质约束

    返回: (是否通过, 违反的约束列表)
    """
    violations = []

    # 约束1: 层序关系
    if not check_stratigraphic_order(lithology_3d, position, predicted_class):
        violations.append('stratigraphic_order')

    # 约束2: 空间连续性
    if not check_spatial_continuity(lithology_3d, position, predicted_class):
        violations.append('spatial_continuity')

    # 约束3: 局部一致性
    if not check_local_consistency(lithology_3d, position, predicted_class):
        violations.append('local_consistency')

    passed = len(violations) == 0
    return passed, violations


def check_stratigraphic_order(lithology_3d, position, predicted_class):
    """
    检验层序关系

    原理: 在垂直方向上，地层应该保持合理的层序
    """
    i, j, k = position

    # 获取上下相邻体素的岩性
    if k > 0:
        below = lithology_3d[i, j, k-1]
    if k < lithology_3d.shape[2] - 1:
        above = lithology_3d[i, j, k+1]

    # 检验层序 (需要根据具体地质情况定义)
    # 例如: 表土应该在最上层，煤层不应该在表土之上
    return is_valid_stratigraphic_sequence(above, predicted_class, below)
```

---

## 附录

### A. 参考文献

1. Graph Neural Networks for Geological Modeling
2. 3D Geological Modeling Methods Comparison
3. Machine Learning in Subsurface Prediction

### B. 相关代码文件

- `src/trainer.py`: GNN训练器，包含predict方法
- `src/modeling.py`: 当前传统建模实现
- `src/models.py`: GNN模型定义
- `src/data_loader.py`: 数据处理和图构建

### C. 术语表

| 术语 | 解释 |
|------|------|
| **体素 (Voxel)** | 三维空间中的最小单元，类似于二维像素 |
| **RBF** | 径向基函数插值 |
| **IDW** | 反距离加权插值 |
| **Kriging** | 克里金插值，一种地统计学方法 |
| **GNN** | 图神经网络 |
| **层序** | 地层在垂直方向上的排列顺序 |

---

*文档版本: 1.0*
*创建日期: 2024*
*作者: Claude Code Assistant*
