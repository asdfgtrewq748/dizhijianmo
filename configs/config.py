"""
配置文件
包含模型和训练的默认参数
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    # 图构建参数
    "k_neighbors": 30,          # KNN邻居数 (优化：调整为30)
    "max_distance": None,       # 最大连接距离 (None表示不限制)
    "graph_type": "knn",        # 图类型: 'knn', 'radius', 'delaunay'

    # 数据预处理
    "normalize_coords": True,   # 是否标准化坐标
    "normalize_features": True, # 是否标准化特征

    # 数据划分
    "test_size": 0.2,           # 测试集比例
    "val_size": 0.1,            # 验证集比例
    "random_seed": 42           # 随机种子
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # 模型选择
    "model_type": "transformer", # 模型类型: 'gcn', 'graphsage', 'gat', 'geo3d', 'transformer'

    # 网络结构
    "hidden_channels": 384,     # 隐藏层维度 (优化：调整为384，增加容量)
    "num_layers": 6,            # GNN层数 (优化：保持深度)
    "dropout": 0.4,             # Dropout比率 (优化：略微增加以防过拟合)

    # GAT专用参数
    "gat_heads": 8,             # 注意力头数 (优化：增加头数)

    # GraphSAGE专用参数
    "sage_aggr": "mean"         # 聚合方式: 'mean', 'max', 'lstm'
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    # 优化器
    "optimizer": "adamw",       # 优化器: 'adam', 'adamw'
    "learning_rate": 0.002,     # 学习率 (优化：降低初始学习率以稳定训练)
    "weight_decay": 1e-4,       # L2正则化

    # 学习率调度
    "scheduler": "cosine_restart", # 调度器: 'plateau', 'cosine', 'cosine_restart', 'none' (优化：使用cosine_restart)
    "scheduler_patience": 15,   # plateau调度器耐心值
    "scheduler_factor": 0.5,    # 学习率衰减因子

    # 训练控制
    "epochs": 300,              # 最大训练轮数
    "early_stopping_patience": 50,  # 早停耐心值
    "min_delta": 1e-4,          # 最小改进阈值

    # 类别平衡
    "use_class_weights": True,  # 是否使用类别权重
    
    # 损失函数优化
    "loss_type": "focal",       # 'ce', 'focal', 'label_smoothing'
    "focal_gamma": 2.0,         # Focal Loss gamma (优化：回调至2.0，平衡难易样本)
    "label_smoothing": 0.1      # 标签平滑系数
}

# ==================== 可视化配置 ====================
VIS_CONFIG = {
    # 网格插值
    "grid_resolution": (50, 50, 50),  # 三维网格分辨率

    # 颜色方案
    "colormap": "Set3",         # 岩性颜色方案

    # 默认显示
    "default_marker_size": 4,   # 散点大小
    "default_opacity": 0.8      # 透明度
}

# ==================== 岩性定义 (示例) ====================
# 在此定义你的岩性类别及其属性
LITHOLOGY_CONFIG = {
    "classes": [
        # ============== 占位符: 替换为你的实际岩性 ==============
        {"name": "砂岩", "code": "SS", "color": "#F5DEB3"},
        {"name": "泥岩", "code": "MS", "color": "#8B4513"},
        {"name": "灰岩", "code": "LS", "color": "#808080"},
        {"name": "页岩", "code": "SH", "color": "#2F4F4F"},
        {"name": "砾岩", "code": "CG", "color": "#CD853F"}
    ],

    # 特征列定义
    "feature_columns": [
        # ============== 占位符: 替换为你的实际特征 ==============
        {"name": "porosity", "description": "孔隙度", "unit": "%"},
        {"name": "permeability", "description": "渗透率", "unit": "mD"},
        {"name": "density", "description": "密度", "unit": "g/cm³"}
    ]
}


def get_config(config_type: str = "all") -> dict:
    """
    获取配置

    Args:
        config_type: 配置类型 ('data', 'model', 'train', 'vis', 'lithology', 'all')

    Returns:
        配置字典
    """
    configs = {
        "data": DATA_CONFIG,
        "model": MODEL_CONFIG,
        "train": TRAIN_CONFIG,
        "vis": VIS_CONFIG,
        "lithology": LITHOLOGY_CONFIG
    }

    if config_type == "all":
        return configs
    elif config_type in configs:
        return configs[config_type]
    else:
        raise ValueError(f"未知配置类型: {config_type}")
