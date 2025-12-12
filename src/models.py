"""
GNN模型定义模块
包含GCN、GraphSAGE等图神经网络模型，用于三维地质建模中的岩性分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional, List, Tuple


class GeoGCN(nn.Module):
    """
    基于GCN的地质建模网络
    用于根据钻孔数据预测三维空间中的岩性类别
    """

    def __init__(
        self,
        in_channels: int,           # 输入特征维度 (x, y, z坐标 + 其他地质特征)
        hidden_channels: int,        # 隐藏层维度
        out_channels: int,           # 输出类别数 (岩性种类数)
        num_layers: int = 3,         # GCN层数
        dropout: float = 0.5,        # Dropout比率
        use_batch_norm: bool = True  # 是否使用批归一化
    ):
        super(GeoGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # 构建GCN层列表
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels))

        # 中间层: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))

        # 最后一层: hidden_channels -> out_channels
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 (可选) [num_edges]

        Returns:
            out: 节点分类logits [num_nodes, out_channels]
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层不加激活函数
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index, edge_weight)
        return torch.argmax(logits, dim=1)


class GeoGraphSAGE(nn.Module):
    """
    基于GraphSAGE的地质建模网络
    GraphSAGE通过采样和聚合邻居信息，适合处理大规模地质数据
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = 'mean',          # 聚合方式: 'mean', 'max', 'lstm'
        use_batch_norm: bool = True
    ):
        super(GeoGraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))

        # 输出层
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]

        Returns:
            out: 分类logits [num_nodes, out_channels]
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index)
        return torch.argmax(logits, dim=1)


class GeoGAT(nn.Module):
    """
    基于图注意力网络(GAT)的地质建模网络
    使用注意力机制自适应地学习邻居节点的重要性
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,              # 注意力头数
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super(GeoGAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels * heads))

        # 中间层 (输入维度需要乘以heads)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels * heads))

        # 输出层 (concat=False, 对多头结果取平均)
        self.convs.append(GATConv(hidden_channels * heads, out_channels,
                                  heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index)
        return torch.argmax(logits, dim=1)


class Geo3DGNN(nn.Module):
    """
    专为三维地质建模设计的混合GNN模型
    结合空间特征编码和多尺度图卷积
    """

    def __init__(
        self,
        in_channels: int,            # 原始输入特征维度
        hidden_channels: int = 64,
        out_channels: int = 10,      # 岩性类别数
        num_layers: int = 4,
        dropout: float = 0.5,
        spatial_encoding_dim: int = 32  # 空间位置编码维度
    ):
        super(Geo3DGNN, self).__init__()

        self.dropout = dropout

        # 空间位置编码器 (处理x, y, z坐标)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, spatial_encoding_dim),
            nn.ReLU(),
            nn.Linear(spatial_encoding_dim, spatial_encoding_dim),
            nn.ReLU()
        )

        # 特征编码器 (处理其他地质特征)
        feature_dim = in_channels - 3  # 除去坐标的其他特征
        self.feature_encoder = nn.Sequential(
            nn.Linear(max(feature_dim, 1), hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) if feature_dim > 0 else None

        # 计算融合后的特征维度
        fused_dim = spatial_encoding_dim + (hidden_channels // 2 if feature_dim > 0 else 0)

        # 多尺度图卷积层
        self.conv1 = SAGEConv(fused_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)

        self.batch_norms = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(4)
        ])

        # 跳跃连接融合层
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)

        # 输出分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_channels], 前3列为(x,y,z)坐标
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 (可选)
        """
        # 分离坐标和其他特征
        coords = x[:, :3]  # (x, y, z)

        # 空间位置编码
        spatial_feat = self.spatial_encoder(coords)

        # 其他特征编码
        if self.in_channels > 3 and self.feature_encoder is not None:
            other_feat = x[:, 3:]
            other_feat = self.feature_encoder(other_feat)
            # 融合特征
            h = torch.cat([spatial_feat, other_feat], dim=1)
        else:
            h = spatial_feat

        # 多尺度图卷积
        h1 = F.relu(self.batch_norms[0](self.conv1(h, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.relu(self.batch_norms[1](self.conv2(h1, edge_index, edge_weight)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = F.relu(self.batch_norms[2](self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        h4 = F.relu(self.batch_norms[3](self.conv4(h3, edge_index, edge_weight)))

        # 跳跃连接
        h_skip = torch.cat([h2, h4], dim=1)
        h_fused = self.fusion(h_skip)

        # 分类输出
        out = self.classifier(h_fused)
        return out

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index, edge_weight)
        return torch.argmax(logits, dim=1)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    模型工厂函数

    Args:
        model_name: 模型名称 ('gcn', 'graphsage', 'gat', 'geo3d')
        **kwargs: 模型参数

    Returns:
        model: 对应的GNN模型实例
    """
    models = {
        'gcn': GeoGCN,
        'graphsage': GeoGraphSAGE,
        'gat': GeoGAT,
        'geo3d': Geo3DGNN
    }

    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")

    return models[model_name.lower()](**kwargs)


# ============== 测试代码 ==============
if __name__ == "__main__":
    # 测试模型
    print("测试GNN模型...")

    # 模拟数据
    num_nodes = 100
    in_channels = 6   # x, y, z + 3个其他特征
    out_channels = 5  # 5种岩性

    # 随机生成节点特征和边
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 300))

    # 测试各个模型
    for model_name in ['gcn', 'graphsage', 'gat', 'geo3d']:
        print(f"\n测试 {model_name.upper()} 模型:")
        model = get_model(
            model_name,
            in_channels=in_channels,
            hidden_channels=32,
            out_channels=out_channels
        )
        out = model(x, edge_index)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
