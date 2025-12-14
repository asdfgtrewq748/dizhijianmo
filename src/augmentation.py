"""
数据增强模块
包含图数据增强和特征增强方法
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Tuple
import copy


class NodeFeatureNoise(nn.Module):
    """节点特征噪声增强"""

    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class DropEdge(nn.Module):
    """随机丢弃边 - 防止过拟合"""

    def __init__(self, drop_rate: float = 0.1):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if training and self.drop_rate > 0:
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges, device=edge_index.device) > self.drop_rate
            edge_index = edge_index[:, mask]
            if edge_weight is not None:
                edge_weight = edge_weight[mask]
        return edge_index, edge_weight


class FeatureMixup:
    """特征Mixup增强 - 在训练时混合样本"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        对mask内的样本进行mixup

        Returns:
            mixed_x: 混合后的特征
            y_a: 原始标签
            y_b: 混合标签
            lam: 混合系数
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 只对训练样本进行mixup
        train_indices = mask.nonzero(as_tuple=True)[0]
        batch_size = train_indices.size(0)

        # 随机打乱索引
        perm = torch.randperm(batch_size, device=x.device)
        shuffled_indices = train_indices[perm]

        # 混合特征
        mixed_x = x.clone()
        mixed_x[train_indices] = lam * x[train_indices] + (1 - lam) * x[shuffled_indices]

        y_a = y[train_indices]
        y_b = y[shuffled_indices]

        return mixed_x, y_a, y_b, lam


class GraphAugmentation:
    """
    综合图数据增强
    """

    def __init__(
        self,
        node_noise_std: float = 0.05,
        edge_drop_rate: float = 0.1,
        feature_mask_rate: float = 0.1
    ):
        self.node_noise = NodeFeatureNoise(node_noise_std)
        self.drop_edge = DropEdge(edge_drop_rate)
        self.feature_mask_rate = feature_mask_rate

    def augment(self, data: Data, training: bool = True) -> Data:
        """
        应用数据增强

        Args:
            data: PyG Data对象
            training: 是否为训练模式

        Returns:
            augmented_data: 增强后的数据
        """
        if not training:
            return data

        # 创建数据副本
        aug_data = copy.copy(data)

        # 节点特征噪声
        aug_data.x = self.node_noise(data.x, training)

        # 特征随机遮蔽
        if self.feature_mask_rate > 0:
            mask = torch.rand(data.x.size(1), device=data.x.device) > self.feature_mask_rate
            aug_data.x = aug_data.x * mask.float()

        # 边丢弃
        aug_data.edge_index, aug_data.edge_weight = self.drop_edge(
            data.edge_index,
            getattr(data, 'edge_weight', None),
            training
        )

        return aug_data


def compute_neighborhood_features(data: Data, num_classes: int) -> torch.Tensor:
    """
    计算邻域统计特征

    Args:
        data: PyG Data对象
        num_classes: 类别数量

    Returns:
        neighborhood_features: 邻域特征 [num_nodes, num_features]
    """
    num_nodes = data.x.size(0)
    device = data.x.device

    # 初始化邻域特征
    # 1. 邻居数量
    # 2. 邻居类别分布 (如果有标签)
    # 3. 邻居特征统计

    edge_index = data.edge_index

    # 计算每个节点的邻居数量
    neighbor_counts = torch.zeros(num_nodes, device=device)
    for i in range(edge_index.size(1)):
        neighbor_counts[edge_index[0, i]] += 1

    # 归一化邻居数量
    max_neighbors = neighbor_counts.max() + 1e-6
    normalized_neighbor_counts = neighbor_counts / max_neighbors

    # 计算邻居特征的均值和标准差
    neighbor_mean = torch.zeros(num_nodes, data.x.size(1), device=device)
    neighbor_std = torch.zeros(num_nodes, data.x.size(1), device=device)

    # 使用scatter操作高效计算
    src_features = data.x[edge_index[1]]  # 源节点特征

    # 累加邻居特征
    neighbor_sum = torch.zeros(num_nodes, data.x.size(1), device=device)
    neighbor_sum.scatter_add_(0, edge_index[0].unsqueeze(1).expand(-1, data.x.size(1)), src_features)

    # 计算均值
    counts = neighbor_counts.unsqueeze(1).clamp(min=1)
    neighbor_mean = neighbor_sum / counts

    # 计算标准差
    diff_sq = (src_features - neighbor_mean[edge_index[0]]) ** 2
    neighbor_var = torch.zeros(num_nodes, data.x.size(1), device=device)
    neighbor_var.scatter_add_(0, edge_index[0].unsqueeze(1).expand(-1, data.x.size(1)), diff_sq)
    neighbor_std = torch.sqrt(neighbor_var / counts + 1e-6)

    # 聚合特征
    # 使用均值的范数和标准差的范数作为紧凑表示
    mean_norm = torch.norm(neighbor_mean, dim=1, keepdim=True)
    std_norm = torch.norm(neighbor_std, dim=1, keepdim=True)

    neighborhood_features = torch.cat([
        normalized_neighbor_counts.unsqueeze(1),
        mean_norm,
        std_norm
    ], dim=1)

    return neighborhood_features


def add_positional_encoding(coords: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """
    添加位置编码 (类似Transformer)

    Args:
        coords: 坐标张量 [num_nodes, 3]
        dim: 编码维度

    Returns:
        pos_encoding: 位置编码 [num_nodes, dim * 3]
    """
    device = coords.device
    num_nodes = coords.size(0)

    # 归一化坐标到[0, 1]
    coords_min = coords.min(dim=0, keepdim=True)[0]
    coords_max = coords.max(dim=0, keepdim=True)[0]
    coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-6)

    # 生成频率
    frequencies = torch.pow(10000, -torch.arange(0, dim, 2, device=device).float() / dim)

    encodings = []
    for i in range(3):  # x, y, z
        coord = coords_norm[:, i:i+1]  # [num_nodes, 1]

        # 计算sin和cos编码
        angles = coord * frequencies  # [num_nodes, dim//2]
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        enc = torch.cat([sin_enc, cos_enc], dim=1)  # [num_nodes, dim]
        encodings.append(enc)

    pos_encoding = torch.cat(encodings, dim=1)  # [num_nodes, dim * 3]

    return pos_encoding


class BalancedBatchSampler:
    """
    平衡类别采样器
    确保每个批次中各类别样本数量相近
    """

    def __init__(self, labels: torch.Tensor, batch_size: int):
        self.labels = labels
        self.batch_size = batch_size

        # 获取每个类别的索引
        self.class_indices = {}
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            self.class_indices[label.item()] = (labels == label).nonzero(as_tuple=True)[0].tolist()

        self.num_classes = len(self.class_indices)
        self.samples_per_class = max(1, batch_size // self.num_classes)

    def __iter__(self):
        # 每个类别随机采样
        batch = []
        for class_idx, indices in self.class_indices.items():
            if len(indices) >= self.samples_per_class:
                selected = np.random.choice(indices, self.samples_per_class, replace=False)
            else:
                selected = np.random.choice(indices, self.samples_per_class, replace=True)
            batch.extend(selected)

        np.random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return self.samples_per_class * self.num_classes
