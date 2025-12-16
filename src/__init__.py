# 初始化src包
"""
GNN地质建模系统 - 源代码包

重构后的模块结构：
├── gnn_thickness_modeling.py   - GNN厚度预测建模核心模块（新）
├── thickness_data_loader.py    - 厚度预测数据加载（新）
├── thickness_trainer.py        - 厚度预测训练器（新）
├── models.py                   - GNN模型定义
├── data_loader.py              - 原数据加载模块（保留兼容）
├── trainer.py                  - 原训练模块（保留兼容）
├── visualization.py            - 可视化模块
└── modeling.py                 - 原建模模块（保留兼容）

新的正确工作流：
1. 使用 thickness_data_loader.ThicknessDataProcessor 加载层表数据
2. 使用 gnn_thickness_modeling.GNNThicknessPredictor 训练厚度回归模型
3. 使用 gnn_thickness_modeling.GeologicalModelBuilder 层序累加构建三维模型
"""
