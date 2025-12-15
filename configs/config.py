"""
配置文件
包含模型和训练的默认参数
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    # 图构建参数
    "k_neighbors": 24,          # KNN邻居数 (增加到24获取更多空间上下文)
    "max_distance": None,       # 最大连接距离 (None表示不限制)
    "graph_type": "knn",        # 图类型: 'knn', 'radius', 'delaunay'

    # 数据预处理
    "normalize_coords": True,   # 是否标准化坐标
    "normalize_features": True, # 是否标准化特征

    # 数据划分
    "test_size": 0.10,          # 测试集比例 (减少以增加训练数据)
    "val_size": 0.10,           # 验证集比例
    "random_seed": 42           # 随机种子
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # 模型选择 - enhanced效果最好
    "model_type": "enhanced",   # 恢复使用 enhanced 模型

    # 网络结构
    "hidden_channels": 512,     # 隐藏层维度 (增加模型容量)
    "num_layers": 3,            # GNN层数 (3层防止过平滑)
    "dropout": 0.3,             # Dropout比率 (增加对抗过拟合)

    # GAT专用参数
    "gat_heads": 8,             # 注意力头数 (增加)

    # GraphSAGE专用参数
    "sage_aggr": "mean"         # 聚合方式
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    # 优化器
    "optimizer": "adamw",       # 优化器: 'adam', 'adamw'
    "learning_rate": 0.0005,    # 学习率 (降低以提高稳定性)
    "weight_decay": 5e-4,       # L2正则化 (增加)

    # 学习率调度 - 使用cosine调度器
    "scheduler": "cosine",      # 调度器: 'cosine' (更平滑的衰减)
    "scheduler_patience": 50,   # plateau调度器耐心值
    "scheduler_factor": 0.5,    # 学习率衰减因子

    # 训练控制
    "epochs": 1500,             # 最大训练轮数 (延长cosine周期，保持高学习率更久)
    "early_stopping_patience": 80,  # 早停耐心值
    "min_delta": 1e-5,          # 最小改进阈值

    # 类别平衡
    "use_class_weights": True,  # 是否使用类别权重

    # 损失函数优化 - 使用focal loss处理类别不平衡
    "loss_type": "focal",       # 'ce', 'focal', 'label_smoothing'
    "focal_gamma": 3.0,         # Focal Loss gamma (增加以关注难分类样本)
    "label_smoothing": 0.1,     # 标签平滑系数

    # 数据增强配置 - 降低强度减少波动
    "use_augmentation": True,   # 启用数据增强
    "augment_noise_std": 0.005, # 节点特征噪声标准差 (进一步降低)
    "augment_edge_drop": 0.01,  # 边丢弃概率 (进一步降低)
    "use_mixup": False,         # 是否使用Mixup (对图数据效果有限)
    "mixup_alpha": 0.2,         # Mixup alpha参数

    # EMA配置 - 用于平滑训练
    "use_ema": True,            # 启用EMA平滑
    "ema_decay": 0.9995         # EMA衰减率 (提高以更平滑)
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


def validate_config(config: dict = None) -> bool:
    """
    验证配置参数的有效性

    Args:
        config: 要验证的配置字典，如果为None则验证所有默认配置

    Returns:
        是否验证通过

    Raises:
        ValueError: 当参数不合法时
    """
    if config is None:
        config = get_config("all")

    errors = []

    # 验证数据配置
    data_cfg = config.get("data", DATA_CONFIG)
    if data_cfg.get("k_neighbors", 0) < 1:
        errors.append("k_neighbors 必须大于等于1")
    if data_cfg.get("test_size", 0) < 0 or data_cfg.get("test_size", 0) > 1:
        errors.append("test_size 必须在 0-1 之间")
    if data_cfg.get("val_size", 0) < 0 or data_cfg.get("val_size", 0) > 1:
        errors.append("val_size 必须在 0-1 之间")
    if data_cfg.get("test_size", 0) + data_cfg.get("val_size", 0) >= 1:
        errors.append("test_size + val_size 必须小于 1")
    if data_cfg.get("graph_type") not in ["knn", "radius", "delaunay"]:
        errors.append("graph_type 必须是 'knn', 'radius' 或 'delaunay'")

    # 验证模型配置
    model_cfg = config.get("model", MODEL_CONFIG)
    if model_cfg.get("hidden_channels", 0) < 1:
        errors.append("hidden_channels 必须大于等于1")
    if model_cfg.get("num_layers", 0) < 1:
        errors.append("num_layers 必须大于等于1")
    if not (0 <= model_cfg.get("dropout", 0) <= 1):
        errors.append("dropout 必须在 0-1 之间")
    if model_cfg.get("gat_heads", 0) < 1:
        errors.append("gat_heads 必须大于等于1")
    if model_cfg.get("model_type") not in ["gcn", "gat", "graphsage", "sage", "enhanced"]:
        errors.append("model_type 必须是 'gcn', 'gat', 'graphsage', 'sage' 或 'enhanced'")

    # 验证训练配置
    train_cfg = config.get("train", TRAIN_CONFIG)
    if train_cfg.get("learning_rate", 0) <= 0:
        errors.append("learning_rate 必须大于0")
    if train_cfg.get("weight_decay", -1) < 0:
        errors.append("weight_decay 必须大于等于0")
    if train_cfg.get("epochs", 0) < 1:
        errors.append("epochs 必须大于等于1")
    if train_cfg.get("early_stopping_patience", 0) < 1:
        errors.append("early_stopping_patience 必须大于等于1")
    if train_cfg.get("focal_gamma", -1) < 0:
        errors.append("focal_gamma 必须大于等于0")
    if not (0 <= train_cfg.get("label_smoothing", 0) < 1):
        errors.append("label_smoothing 必须在 0-1 之间")
    if not (0 <= train_cfg.get("augment_noise_std", 0) <= 1):
        errors.append("augment_noise_std 必须在 0-1 之间")
    if not (0 <= train_cfg.get("augment_edge_drop", 0) <= 1):
        errors.append("augment_edge_drop 必须在 0-1 之间")
    if not (0 < train_cfg.get("ema_decay", 0) < 1):
        errors.append("ema_decay 必须在 0-1 之间 (不含边界)")
    if train_cfg.get("loss_type") not in ["ce", "focal", "label_smoothing"]:
        errors.append("loss_type 必须是 'ce', 'focal' 或 'label_smoothing'")
    if train_cfg.get("optimizer") not in ["adam", "adamw"]:
        errors.append("optimizer 必须是 'adam' 或 'adamw'")
    if train_cfg.get("scheduler") not in ["plateau", "cosine", "step", "onecycle", "none", None]:
        errors.append("scheduler 必须是 'plateau', 'cosine', 'step', 'onecycle' 或 'none'")

    # 验证可视化配置
    vis_cfg = config.get("vis", VIS_CONFIG)
    resolution = vis_cfg.get("grid_resolution", (50, 50, 50))
    if not (isinstance(resolution, tuple) and len(resolution) == 3):
        errors.append("grid_resolution 必须是长度为3的元组")
    elif any(r < 1 for r in resolution):
        errors.append("grid_resolution 中的每个值必须大于等于1")
    if not (0 < vis_cfg.get("default_opacity", 0.5) <= 1):
        errors.append("default_opacity 必须在 0-1 之间")

    if errors:
        error_msg = "配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return True


def print_config(config_type: str = "all"):
    """打印配置内容"""
    config = get_config(config_type)

    if config_type == "all":
        for name, cfg in config.items():
            print(f"\n{'='*20} {name.upper()} {'='*20}")
            for k, v in cfg.items():
                print(f"  {k}: {v}")
    else:
        print(f"\n{'='*20} {config_type.upper()} {'='*20}")
        for k, v in config.items():
            print(f"  {k}: {v}")
