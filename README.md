# GNN三维地质建模系统

基于图神经网络(GNN)的三维地质建模项目，使用PyTorch Geometric实现。

## 项目结构

```
dizhijianmo/
├── src/
│   ├── models.py        # GNN模型定义 (GCN, GraphSAGE, GAT, Geo3D)
│   ├── data_loader.py   # 数据加载与预处理
│   └── trainer.py       # 训练、验证与评估
├── configs/
│   └── config.py        # 配置文件
├── data/                # 数据目录
├── models/              # 模型保存目录
├── app.py               # Streamlit可视化前端
├── main.py              # 主入口脚本
└── requirements.txt     # 依赖列表
```

## 安装

```bash
# 创建虚拟环境 (推荐)
conda create -n geomodel python=3.10
conda activate geomodel

# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch Geometric
pip install torch-geometric

# 安装其他依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 运行演示

```bash
python main.py demo
```

### 2. 启动可视化界面

```bash
python main.py webapp
# 或直接运行
streamlit run app.py
```

### 3. 使用自定义数据训练

```bash
python main.py train --data your_data.csv --model graphsage --epochs 300
```

## 数据格式

钻孔数据CSV文件需包含以下列:

| 列名 | 必需 | 说明 |
|------|------|------|
| x | 是 | X坐标 |
| y | 是 | Y坐标 |
| z | 是 | Z坐标(深度) |
| lithology | 是 | 岩性标签 |
| porosity | 否 | 孔隙度 |
| permeability | 否 | 渗透率 |
| density | 否 | 密度 |
| ... | 否 | 其他地质特征 |

## 可用模型

- **GCN**: 图卷积网络
- **GraphSAGE**: 采样聚合网络
- **GAT**: 图注意力网络
- **Geo3D**: 专为三维地质建模设计的混合模型

## 代码示例

```python
from src.models import get_model
from src.data_loader import BoreholeDataProcessor
from src.trainer import GeoModelTrainer

# 1. 加载数据
processor = BoreholeDataProcessor(k_neighbors=8)
df = processor.load_borehole_data("your_data.csv")
result = processor.process(df)

# 2. 创建模型
model = get_model(
    'graphsage',
    in_channels=result['num_features'],
    hidden_channels=64,
    out_channels=result['num_classes']
)

# 3. 训练
trainer = GeoModelTrainer(model)
trainer.train(result['data'], epochs=200)

# 4. 预测
predictions, probs = trainer.predict(result['data'])
```
