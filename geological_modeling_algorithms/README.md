# 地质建模与导出算法模块

本文件夹包含地质建模系统的核心算法实现，仅包含后端算法部分，不包含前端界面代码。

## 📁 文件结构

### 核心算法文件

- **`interpolation.py`** - 插值算法模块
  - 支持多种插值方法（克里金、RBF、IDW等）
  - 包含交叉验证和智能插值选择
  - 地质建模的核心计算引擎

- **`cad_section_importer.py`** - CAD剖面图导入模块
  - 解析DXF格式剖面图
  - 与地质模型生成的剖面进行对比验证
  - 偏差分析和误差计算

- **`z_section_slicer.py`** - Z截面切片模块
  - 生成指定Z高度的地质剖面
  - 支持多层次模型切片
  - 用于剖面图导出和可视化

- **`key_strata_calculator.py`** - 关键地层计算模块
  - 关键层识别和分析
  - 地层稳定性评估
  - 支撑设计参考计算

- **`visualization_enhanced.py`** - 可视化增强模块
  - 3D地质模型可视化
  - 多种渲染和展示功能
  - 支持模型预览和导出

- **`batch_operations.py`** - 批量操作模块
  - 批量建模处理
  - 批量导出功能
  - 大规模数据处理优化

- **`data_validation.py`** - 数据验证模块
  - 输入数据验证和清洗
  - 数据质量检查
  - 异常值检测和处理

- **`statistical_analysis.py`** - 统计分析模块
  - 地质数据统计分析
  - 趋势分析和相关性计算
  - 数据质量评估

### 子模块

#### `coal_seam_blocks/` - 煤层块建模模块
- **`modeling.py`** - 核心建模算法
  - BlockModel数据结构
  - 分层建模主流程
  - 层间间隙修复
  - 顶板生成算法

- **`aggregator.py`** - 数据聚合模块
  - 钻孔数据聚合
  - 多源数据融合
  - 数据预处理

- **`__init__.py`** - 模块初始化文件

#### `exporters/` - 导出器模块
所有支持的地质模型导出格式：

- **`base_exporter.py`** - 导出器基类
  - 统一的导出接口
  - 通用导出逻辑

- **`stl_exporter.py`** - STL格式导出
  - 三角网格导出
  - 支持3D打印格式

- **`layered_stl_exporter.py`** - 分层STL导出
  - 按地层分别导出STL
  - 多文件管理

- **`dxf_exporter.py`** - DXF格式导出
  - CAD格式导出
  - 剖面图和平面图导出

- **`flac3d_exporter.py`** - FLAC3D格式导出
  - FLAC3D网格导出
  - 支持数值模拟

- **`f3grid_exporter.py`** - F3Grid格式导出
  - FLAC3D标准网格格式
  - 包含区域和分组信息

- **`tetra_f3grid_exporter.py`** - 四面体F3Grid导出
  - 四面体网格生成
  - 高级网格导出

- **`obj_exporter.py`** - OBJ格式导出
  - Wavefront OBJ格式
  - 通用3D模型格式

- **`__init__.py`** - 导出器模块初始化

### 配置文件

- **`requirements.txt`** - Python依赖包列表
  - 所需的核心算法库
  - 科学计算和数据处理库

## 🔧 主要功能

### 1. 地质建模
- 多点钻孔数据插值
- 自动分层建模
- 层间关系处理
- 顶板自动生成

### 2. 数据导出
- 支持多种3D格式（STL、OBJ）
- 支持CAD格式（DXF）
- 支持数值模拟格式（FLAC3D、F3Grid）
- 分层导出和批量导出

### 3. 数据处理
- 数据验证和清洗
- 统计分析
- 批量操作
- 质量控制

### 4. 可视化
- 3D模型渲染
- 剖面图生成
- 交互式预览

## 📦 依赖库

主要依赖的Python库：
- `numpy` - 数值计算
- `scipy` - 科学计算和插值
- `scikit-learn` - 机器学习和交叉验证
- `pandas` - 数据处理
- `ezdxf` - DXF文件处理
- `pyvista` - 3D可视化
- `trimesh` - 网格处理
- 其他详见 `requirements.txt`

## 🚀 使用方式

### 独立使用
这些算法模块可以独立导入和使用，无需前端界面：

```python
# 示例：使用插值模块
from interpolation import interpolate_smart, get_interpolator

# 示例：使用建模模块
from coal_seam_blocks.modeling import build_block_models

# 示例：使用导出器
from exporters.stl_exporter import STLExporter
```

### 集成使用
可以集成到其他项目中作为地质建模算法库使用。

## 📝 注意事项

1. **纯算法代码**：本文件夹只包含后端算法，不包含任何前端UI代码
2. **独立运行**：可以独立于原项目使用，只需安装依赖
3. **依赖关系**：某些模块之间有相互依赖，建议保持文件夹结构
4. **缓存文件**：包含了`__pycache__`文件夹，可以删除

## 🔄 从原项目复制

本文件夹的内容从以下位置复制：
- 源路径：`backend/`
- 复制内容：
  - 核心算法文件
  - `coal_seam_blocks/` 模块
  - `exporters/` 模块
  - 算法依赖配置

## 📅 创建日期
2025年12月16日
