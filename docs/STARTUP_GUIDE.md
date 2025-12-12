# GNN三维地质建模系统 - 启动指南

## 目录

- [环境准备](#环境准备)
- [安装步骤](#安装步骤)
- [快速启动](#快速启动)
- [数据准备](#数据准备)
- [常见问题](#常见问题)

---

## 环境准备

### 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11, Linux, macOS |
| Python | 3.9 - 3.11 (推荐 3.10) |
| 内存 | 最低 8GB，推荐 16GB+ |
| GPU | 可选，支持 CUDA 11.8+ 的 NVIDIA 显卡 |

### 前置软件

1. **Python**: 推荐使用 [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Git**: 用于版本控制 (可选)
3. **CUDA**: 如需GPU加速，安装对应版本的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

---

## 安装步骤

### 步骤 1: 创建虚拟环境

```bash
# 使用 conda 创建环境
conda create -n geomodel python=3.10
conda activate geomodel

# 或使用 venv
python -m venv geomodel_env
# Windows:
geomodel_env\Scripts\activate
# Linux/macOS:
source geomodel_env/bin/activate
```

### 步骤 2: 安装 PyTorch

根据你的硬件选择安装命令:

**有 NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**有 NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**仅 CPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 步骤 3: 安装 PyTorch Geometric

```bash
pip install torch-geometric
```

如果安装遇到问题，可以尝试:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

### 步骤 4: 安装其他依赖

```bash
cd E:/xiangmu/dizhijianmo
pip install -r requirements.txt
```

### 步骤 5: 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

预期输出:
```
PyTorch: 2.x.x
PyG: 2.x.x
CUDA可用: True (或 False，取决于你的硬件)
```

---

## 快速启动

### 方式 1: 运行演示 (推荐新手)

使用模拟数据快速体验完整流程:

```bash
cd D:/xiangmu/dizhijianmo
python main.py demo
```

这将:
1. 生成 50 个模拟钻孔数据
2. 构建 KNN 图结构
3. 训练 GraphSAGE 模型
4. 评估并输出预测结果

### 方式 2: 启动可视化界面

```bash
python main.py webapp
```

或直接运行:
```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`

**界面功能:**
- 📊 数据探索: 加载数据、3D可视化、统计分析
- 🚀 模型训练: 参数配置、实时训练监控
- 📈 结果分析: 混淆矩阵、分类报告
- 🗺️ 三维可视化: 预测结果、剖面图

### 方式 3: 使用自定义数据训练

```bash
python main.py train --data path/to/your_data.csv --model graphsage --epochs 300
```

完整参数:
```bash
python main.py train \
    --data your_data.csv \     # 数据文件路径 (必需)
    --model graphsage \        # 模型类型: gcn, graphsage, gat, geo3d
    --hidden 64 \              # 隐藏层维度
    --layers 3 \               # GNN层数
    --epochs 200 \             # 训练轮数
    --lr 0.01 \                # 学习率
    --output output/           # 输出目录
```

---

## 数据准备

### 数据格式要求

钻孔数据需要保存为 CSV 或 Excel 文件，包含以下列:

| 列名 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `x` | float | ✅ | X坐标 (东方向, 米) |
| `y` | float | ✅ | Y坐标 (北方向, 米) |
| `z` | float | ✅ | Z坐标 (高程/深度, 米, 向下为负) |
| `lithology` | string | ✅ | 岩性标签 (如 "砂岩", "泥岩") |
| `borehole_id` | string | ❌ | 钻孔编号 (可选) |
| `porosity` | float | ❌ | 孔隙度 (可选特征) |
| `permeability` | float | ❌ | 渗透率 (可选特征) |
| `density` | float | ❌ | 密度 (可选特征) |
| ... | ... | ❌ | 其他地质特征 |

### 示例数据

```csv
borehole_id,x,y,z,lithology,porosity,permeability,density
BH_001,100.5,200.3,-10.0,砂岩,0.25,150.5,2.35
BH_001,100.5,200.3,-20.0,砂岩,0.22,120.3,2.40
BH_001,100.5,200.3,-30.0,泥岩,0.08,0.5,2.55
BH_001,100.5,200.3,-40.0,灰岩,0.05,0.1,2.70
BH_002,350.2,180.7,-10.0,砂岩,0.28,180.2,2.30
...
```

### 数据放置

将你的数据文件放入 `data/` 目录:
```
dizhijianmo/
└── data/
    └── your_borehole_data.csv
```

---

## 常见问题

### Q1: 安装 torch-geometric 失败

**解决方案:**
```bash
# 先安装依赖包
pip install wheel
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

### Q2: CUDA out of memory

**解决方案:**
1. 减少 `hidden_channels` (如 64 → 32)
2. 减少 `num_layers` (如 3 → 2)
3. 减少 `k_neighbors` (如 8 → 5)
4. 使用 CPU 训练: 在代码中设置 `device='cpu'`

### Q3: Streamlit 无法启动

**解决方案:**
```bash
# 检查端口占用
netstat -ano | findstr :8501

# 指定其他端口
streamlit run app.py --server.port 8502
```

### Q4: 数据加载报错

**检查项:**
1. 确保 CSV 文件编码为 UTF-8
2. 确保列名正确: `x`, `y`, `z`, `lithology`
3. 确保没有空值或异常值

### Q5: 训练损失不下降

**解决方案:**
1. 降低学习率: `--lr 0.001`
2. 增加隐藏层维度: `--hidden 128`
3. 检查数据是否有问题 (标签是否正确)
4. 尝试不同的模型: `--model gat`

---

## 下一步

1. 阅读 [项目架构说明](./ARCHITECTURE.md) 了解代码结构
2. 准备你的钻孔数据
3. 在可视化界面中探索和训练模型
4. 根据需要修改配置文件 `configs/config.py`

如有问题，请检查日志输出或在代码中添加调试信息。
