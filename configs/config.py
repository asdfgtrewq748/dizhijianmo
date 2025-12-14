"""
配置文件
包含模型和训练的默认参数
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    # 图构建参数
    "k_neighbors": 16,          # KNN邻居数 (16是最佳值)
    "max_distance": None,       # 最大连接距离 (None表示不限制)
    "graph_type": "knn",        # 图类型: 'knn', 'radius', 'delaunay'

    # 数据预处理
    "normalize_coords": True,   # 是否标准化坐标
    "normalize_features": True, # 是否标准化特征

    # 数据划分
    "test_size": 0.15,          # 测试集比例
    "val_size": 0.15,           # 验证集比例
    "random_seed": 42           # 随机种子
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # 模型选择 - enhanced效果最好
    "model_type": "enhanced",   # 恢复使用 enhanced 模型

    # 网络结构
    "hidden_channels": 384,     # 隐藏层维度
    "num_layers": 2,            # GNN层数 (2层防止过平滑)
    "dropout": 0.15,            # Dropout比率

    # GAT专用参数
    "gat_heads": 12,            # 注意力头数

    # GraphSAGE专用参数
    "sage_aggr": "mean"         # 聚合方式
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    # 优化器
    "optimizer": "adamw",       # 优化器: 'adam', 'adamw'
    "learning_rate": 0.001,     # 学习率
    "weight_decay": 1e-4,       # L2正则化

    # 学习率调度 - 使用plateau调度器，根据性能自动调整
    "scheduler": "plateau",     # 调度器: 'plateau' (自适应降低学习率)
    "scheduler_patience": 50,   # plateau调度器耐心值 (增加到50，更稳定)
    "scheduler_factor": 0.5,    # 学习率衰减因子

    # 训练控制
    "epochs": 600,              # 最大训练轮数
    "early_stopping_patience": 120,  # 早停耐心值 (增加到120)
    "min_delta": 1e-5,          # 最小改进阈值

    # 类别平衡
    "use_class_weights": True,  # 是否使用类别权重

    # 损失函数优化 - 使用focal loss处理类别不平衡
    "loss_type": "focal",       # 'ce', 'focal', 'label_smoothing'
    "focal_gamma": 2.0,         # Focal Loss gamma
    "label_smoothing": 0.1,     # 标签平滑系数

    # 数据增强配置 - 降低增强强度以减少波动
    "use_augmentation": True,   # 启用数据增强
    "augment_noise_std": 0.02,  # 节点特征噪声标准差 (从0.03降低到0.02)
    "augment_edge_drop": 0.03,  # 边丢弃概率 (从0.05降低到0.03)
    "use_mixup": False,         # 是否使用Mixup (对图数据效果有限)
    "mixup_alpha": 0.2,         # Mixup alpha参数

    # EMA配置 - 新增，用于平滑训练
    "use_ema": True,            # 启用EMA平滑
    "ema_decay": 0.995          # EMA衰减率 (越接近1越平滑)
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
