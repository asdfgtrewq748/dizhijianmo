# GNN三维地质建模系统

基于图神经网络(GNN)的三维地质建模项目，使用PyTorch Geometric实现。提供高性能PyQt6桌面应用和Web可视化界面。

## 核心理念

本项目采用**正确的地质建模逻辑**：

```
钻孔数据 → GNN预测厚度(回归) → 层序累加 → 三维地质模型
              ↑                    ↑
       替代传统插值        保证层间无冲突
```

**关键创新**：用GNN预测各层**厚度**（回归问题），而不是预测体素**岩性**（分类问题），然后使用层序累加算法构建三维模型，数学上保证层间无重叠无空缺。

## 项目结构

```
dizhijianmo/
├── src/
│   ├── gnn_thickness_modeling.py  # [核心] GNN厚度预测建模
│   ├── thickness_data_loader.py   # [核心] 厚度预测数据加载
│   ├── thickness_trainer.py       # [核心] 厚度预测训练器
│   ├── pyvista_renderer.py        # [核心] PyVista GPU渲染
│   ├── models.py                  # GNN模型定义
│   ├── data_loader.py             # 原数据加载（兼容）
│   ├── trainer.py                 # 原训练模块（兼容）
│   ├── visualization.py           # SCI论文配图模块
│   └── modeling.py                # 原建模模块（兼容）
├── geological_modeling_algorithms/ # 成熟的传统建模算法库
│   ├── interpolation.py           # 插值算法（13种方法）
│   ├── coal_seam_blocks/          # 层序累加建模
│   └── exporters/                 # 多格式导出（STL/DXF/FLAC3D）
├── configs/
│   └── config.py                  # 配置文件
├── data/                          # 钻孔数据目录
├── output/                        # 输出目录
├── app_qt.py                      # [推荐] PyQt6高性能桌面应用 ⭐
├── run_qt.bat                     # PyQt6快速启动脚本
├── install_qt.bat                 # PyQt6依赖安装脚本
├── app.py                         # Streamlit Web可视化（备选）
├── main_thickness.py              # 命令行厚度预测入口
├── main.py                        # 原命令行入口（兼容）
└── requirements.txt               # 依赖列表
```

## 快速开始

### 方式一：PyQt6桌面应用（推荐）⭐

**高性能原生桌面应用，GPU加速渲染，60+ FPS流畅交互**

#### 1. 安装PyQt6环境

```bash
# 创建虚拟环境
conda create -n geomodel python=3.10
conda activate geomodel

# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装PyTorch Geometric
pip install torch-geometric

# 安装PyQt6依赖（一键脚本）
install_qt.bat

# 或手动安装
pip install PyQt6==6.6.1
pip install pyvistaqt==0.11.0
pip install pyvista>=0.43.0
```

#### 2. 启动PyQt6应用

```bash
# 方式1：直接运行（推荐）
python app_qt.py

# 方式2：使用启动脚本
run_qt.bat

# 方式3：双击 run_qt.bat
```

#### 3. 使用PyQt6界面

界面布局：
```
┌──────────────┬──────────────┬─────────────────────┐
│  控制面板    │  信息面板    │   3D渲染区域         │
│  (左侧)      │  (中间)      │   (右侧)            │
├──────────────┼──────────────┼─────────────────────┤
│ 数据配置     │ 统计信息      │                     │
│ 预测方法     │ 进度条       │    PyVista          │
│ 建模配置     │ 控制台日志    │    GPU渲染窗口       │
│ 渲染控制     │              │                     │
│ 导出按钮     │              │                     │
└──────────────┴──────────────┴─────────────────────┘
```

**工作流程**：
1. 加载数据 → 选择数据目录，配置参数
2. 训练模型 → 选择传统方法或GNN深度学习
3. 构建模型 → 设置分辨率，构建三维模型
4. 渲染控制 → 选择地层、调节透明度、切换渲染模式
5. 导出结果 → PNG/HTML/OBJ/STL/VTK/FLAC3D多格式导出

**性能优势**：
| 指标 | Streamlit | PyQt6 | 提升 |
|------|----------|-------|------|
| 建模速度 | 8-15秒 | 1-2秒 | **5-10倍** |
| 渲染帧率 | 5-15 FPS | 60+ FPS | **4-12倍** |
| GPU利用率 | 10-20% | 70-90% | **4-8倍** |

**详细文档**：[PYQT_FEATURES.md](PYQT_FEATURES.md)

---

### 方式二：命令行工具

**适合批量处理和脚本自动化**

#### 1. 安装基础环境

```bash
# 创建虚拟环境
conda create -n geomodel python=3.10
conda activate geomodel

# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch Geometric
pip install torch-geometric

# 安装其他依赖
pip install -r requirements.txt
```

#### 2. 准备数据

将钻孔数据放入 `data/` 目录：
- 每个钻孔一个CSV文件（如 `ZK001.csv`）
- 一个坐标文件（文件名包含"坐标"）

#### 3. 运行训练和建模

```bash
# 完整流程：训练GNN + 构建三维模型
python main_thickness.py all --epochs 300 --resolution 50

# 仅训练
python main_thickness.py train --epochs 300

# 仅构建模型
python main_thickness.py build --resolution 50

# 传统插值方法（对比基准）
python main_thickness.py baseline --method linear
```

---

### 方式三：Streamlit Web界面（备选）

**适合远程访问和演示展示**

```bash
# 激活环境
conda activate geomodel

# 启动Web界面
streamlit run app.py
```

浏览器访问：http://localhost:8501

**注意**：Streamlit版本存在性能限制，GPU利用率较低，建议使用PyQt6版本获得最佳性能。

## 数据格式

### 钻孔数据文件 (如 ZK001.csv)

| 序号 | 名称 | 厚度/m | 弹性模量/GPa | 容重/kN·m-3 | 抗拉强度/MPa |
|------|------|--------|--------------|-------------|--------------|
| 1 | 腐殖土 | 2.5 | - | - | - |
| 2 | 砂岩 | 5.2 | 15.3 | 25.6 | 3.8 |
| 3 | 煤 | 1.8 | 3.5 | 14.0 | 0.5 |
| ... | ... | ... | ... | ... | ... |

### 坐标文件 (如 钻孔坐标.csv)

| 钻孔名 | 坐标x | 坐标y |
|--------|-------|-------|
| ZK001 | 39645876.2 | 4438976.5 |
| ZK002 | 39646120.8 | 4439102.3 |
| ... | ... | ... |

## 命令行参数

```bash
python main_thickness.py <command> [options]

命令:
  train      训练GNN厚度预测模型
  build      构建三维地质模型
  all        完整流程（训练+构建）
  baseline   传统插值方法（对比）

通用参数:
  --data-dir      数据目录 (默认: data)
  --output        输出目录 (默认: output)

训练参数:
  --epochs        训练轮数 (默认: 200)
  --lr            学习率 (默认: 0.001)
  --hidden-dim    隐藏层维度 (默认: 128)
  --gnn-layers    GNN层数 (默认: 3)
  --conv-type     卷积类型: gatv2/transformer/sage (默认: gatv2)
  --patience      早停耐心值 (默认: 30)

建模参数:
  --resolution    网格分辨率 (默认: 50)
  --base-level    基准面高程 (默认: 0.0)
  --gap           层间间隙 (默认: 0.0)
```

## 代码示例

### 使用新的厚度预测API

```python
from src.thickness_data_loader import ThicknessDataProcessor
from src.gnn_thickness_modeling import GNNGeologicalModeling

# 1. 加载数据
processor = ThicknessDataProcessor(merge_coal=False)
result = processor.process_directory('data/')

# 2. 创建建模器
modeling = GNNGeologicalModeling(
    layer_order=result['layer_order'],
    resolution=50
)

# 3. 训练GNN厚度预测模型
history = modeling.fit(result['data'], epochs=200)

# 4. 构建三维模型
block_models = modeling.build(
    borehole_coords=result['borehole_coords'],
    borehole_features=result['data'].x.numpy(),
    edge_index=result['data'].edge_index
)

# 5. 获取体素模型
voxel_grid, grid_info = modeling.get_voxel_model(nz=50)
```

### 使用传统插值（对比基准）

```python
from src.gnn_thickness_modeling import (
    TraditionalThicknessInterpolator,
    GeologicalModelBuilder
)

# 使用传统线性插值
interpolator = TraditionalThicknessInterpolator(
    method='linear',
    layer_order=layer_order
)

thickness_grids = interpolator.interpolate_thickness(
    borehole_data=df_layers,
    grid_x=grid_x,
    grid_y=grid_y
)

# 层序累加构建模型
builder = GeologicalModelBuilder(layer_order=layer_order)
block_models, XI, YI = builder.build_model(thickness_grids, x_range, y_range)
```

## 输出文件

训练和建模完成后，`output/` 目录包含：

```
output/
├── best_model.pt              # 最佳模型权重
├── training_metadata.json     # 训练元数据
├── model_info.json            # 三维模型信息
├── evaluation_report.txt      # 评估报告
└── layer_*.npz                # 各层网格数据
```

## 技术特点

### PyQt6版本特性 ⭐
1. **GPU加速渲染**：OpenGL硬件加速，60+ FPS流畅交互
2. **多线程架构**：数据加载、训练、建模全异步，UI永不卡顿
3. **完整渲染控制**：
   - 层选择显示（多选）
   - 渲染模式切换（增强材质/基础/线框）
   - 透明度实时调节
   - 侧面显示开关（提升性能）
   - 钻孔标记显示
4. **多格式导出**：PNG/HTML/OBJ/STL/VTK/FLAC3D
5. **FLAC3D网格导出**：层间节点共享，确保应力/位移正确传导
6. **性能优越**：建模速度5-10倍提升，GPU利用率70-90%

### 核心算法特性
1. **GNN厚度回归**：用图神经网络预测各层厚度，替代传统插值
2. **层序累加算法**：自下而上逐层累加，数学保证层间关系正确
3. **垂向顺序修正**：`enforce_columnwise_order` 强制修正异常层序
4. **多种GNN架构**：支持GATv2、Transformer、GraphSAGE
5. **SCI论文配图**：内置专业可视化模块

## 许可证

MIT License
