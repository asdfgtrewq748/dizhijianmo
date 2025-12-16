# Legacy Code - 旧版代码（已弃用）

⚠️ **警告：此目录包含旧版代码，逻辑有误，请勿使用！**

## 为什么这些代码被弃用？

旧版代码的核心逻辑是**错误的**：

| 旧版（错误） | 新版（正确） |
|-------------|-------------|
| GNN预测体素**岩性**（分类） | GNN预测各层**厚度**（回归） |
| 直接生成体素网格 | 层序累加构建层面 |
| 地质上不连续 | 数学保证层间无冲突 |

## 旧版文件说明

```
legacy/
├── src/
│   ├── modeling.py        # 旧建模模块 - 层面插值方式有问题
│   ├── gnn_modeling.py    # 旧GNN建模 - 直接预测岩性（错误！）
│   ├── layer_modeling.py  # 旧层序建模 - 逻辑混乱
│   ├── data_loader.py     # 旧数据加载 - 层内密集采样（分类用）
│   ├── trainer.py         # 旧训练器 - 分类任务
│   └── augmentation.py    # 数据增强
├── main.py                # 旧主入口
└── app.py                 # 旧Streamlit前端
```

## 正确的代码在哪里？

请使用项目根目录下的新版代码：

```
dizhijianmo/
├── src/
│   ├── gnn_thickness_modeling.py  # ✅ 新版GNN厚度预测
│   ├── thickness_data_loader.py   # ✅ 新版数据加载
│   └── thickness_trainer.py       # ✅ 新版训练器
├── main_thickness.py              # ✅ 新版主入口
└── app_new.py                     # ✅ 新版前端（待创建）
```

## 启动命令

```bash
# ✅ 正确方式
python main_thickness.py all --epochs 300

# ❌ 错误方式（勿用）
python legacy/main.py
```
