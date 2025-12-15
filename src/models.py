"""
GNN模型定义模块
包含GCN、GraphSAGE等图神经网络模型，用于三维地质建模中的岩性分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, TransformerConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional, List, Tuple


class GeoGCN(nn.Module):
    """
    基于GCN的地质建模网络 (增强版)
    用于根据钻孔数据预测三维空间中的岩性类别
    
    优化改进:
    - 添加残差连接
    - 添加输入投影层
    - 支持更深的网络结构
    """

    def __init__(
        self,
        in_channels: int,           # 输入特征维度 (x, y, z坐标 + 其他地质特征)
        hidden_channels: int,        # 隐藏层维度
        out_channels: int,           # 输出类别数 (岩性种类数)
        num_layers: int = 3,         # GCN层数
        dropout: float = 0.5,        # Dropout比率
        use_batch_norm: bool = True, # 是否使用批归一化
        use_residual: bool = True    # 是否使用残差连接
    ):
        super(GeoGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # 构建GCN层列表
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 输入投影层
        self.input_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

        # 所有层: hidden_channels -> hidden_channels (除了最后一层)
        for _ in range(num_layers - 1):
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
        # 输入投影
        if self.input_proj is not None:
            x = self.input_proj(x)
            x = F.relu(x)
        
        for i in range(self.num_layers - 1):
            x_input = x  # 保存输入用于残差连接
            
            x = self.convs[i](x, edge_index, edge_weight)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # 残差连接
            if self.use_residual and x.shape == x_input.shape:
                x = x + x_input
            
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层不加激活函数和残差连接
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index, edge_weight)
        return torch.argmax(logits, dim=1)


class GeoGraphSAGE(nn.Module):
    """
    基于GraphSAGE的地质建模网络 (增强版)
    GraphSAGE通过采样和聚合邻居信息，适合处理大规模地质数据
    
    优化改进:
    - 添加残差连接，缓解深层网络训练困难
    - 使用LayerNorm替代BatchNorm，提升训练稳定性
    - 添加输入投影层，统一特征维度
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = 'mean',          # 聚合方式: 'mean', 'max', 'lstm'
        use_batch_norm: bool = True,
        use_residual: bool = True    # 是否使用残差连接
    ):
        super(GeoGraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 输入投影层 - 将输入维度转换为hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

        # 第一层
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        if use_batch_norm:
            self.batch_norms.append(nn.LayerNorm(hidden_channels)) # 优化：使用LayerNorm

        # 中间层 (带残差连接)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            if use_batch_norm:
                self.batch_norms.append(nn.LayerNorm(hidden_channels)) # 优化：使用LayerNorm

        # 输出层
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 (可选, SAGEConv不使用但保留以统一接口)

        Returns:
            out: 分类logits [num_nodes, out_channels]
        """
        # 输入投影
        if self.input_proj is not None:
            x = self.input_proj(x)
            x = F.relu(x)
        
        # 前面的层使用残差连接
        for i in range(self.num_layers - 1):
            x_input = x  # 保存输入用于残差连接
            
            x = self.convs[i](x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # 残差连接
            if self.use_residual and x.shape == x_input.shape:
                x = x + x_input
            
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层不使用残差连接
        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index, edge_weight)
        return torch.argmax(logits, dim=1)


class GeoGAT(nn.Module):
    """
    基于图注意力网络(GAT)的地质建模网络 (增强版)
    使用注意力机制自适应地学习邻居节点的重要性
    
    优化改进:
    - 添加残差连接
    - 使用LayerNorm替代BatchNorm
    - 添加输入投影层
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,              # 注意力头数
        dropout: float = 0.5,
        use_batch_norm: bool = True, # 保持参数名兼容，实际可能使用LayerNorm
        use_residual: bool = True
    ):
        super(GeoGAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        
        # 输入投影层
        self.input_proj = nn.Linear(in_channels, hidden_channels * heads)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() # 使用LayerNorm或BatchNorm
        self.skips = nn.ModuleList() # 跳跃连接的线性变换

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.skips.append(nn.Linear(in_channels, hidden_channels * heads))

        # 中间层 (输入维度需要乘以heads)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.skips.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        # 输出层 (concat=False, 对多头结果取平均)
        self.convs.append(GATConv(hidden_channels * heads, out_channels,
                                  heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        
        for i in range(self.num_layers - 1):
            x_in = x
            
            # GAT卷积
            x = self.convs[i](x, edge_index)
            
            # 残差连接 (需要维度匹配)
            if self.use_residual:
                if x_in.shape[-1] != x.shape[-1]:
                    x_in = self.skips[i](x_in)
                x = x + x_in
            
            # 归一化
            x = self.norms[i](x)
            
            # 激活和Dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层
        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """预测岩性类别"""
        logits = self.forward(x, edge_index, edge_weight)
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


class EnhancedGeoGNN(nn.Module):
    """
    增强版地质GNN模型 (v2)
    特点：
    1. 使用GATv2Conv替代GATConv，支持动态注意力
    2. 利用边权重(距离)作为边特征
    3. 深度残差连接和LayerNorm
    4. 层次化特征聚合
    5. 更宽的网络结构以增加表达能力
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 10,
        num_layers: int = 4,
        dropout: float = 0.3,
        heads: int = 4
    ):
        super(EnhancedGeoGNN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # 确保 hidden_channels 能被 heads 整除
        if hidden_channels % heads != 0:
            hidden_channels = (hidden_channels // heads) * heads

        self.hidden_channels = hidden_channels

        # 深度空间编码器 - 增强版
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # 特征编码器 - 增强版
        feature_dim = max(in_channels - 3, 1)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # 特征融合 - 增强版
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # 图注意力层 (多层) - 使用GATv2
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()  # 添加FFN层增强表达能力

        for i in range(num_layers):
            # GATv2Conv, edge_dim=1 (距离权重)
            self.gat_layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads,
                          dropout=dropout, concat=True, edge_dim=1)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
            # 添加FFN (Feed-Forward Network) 增强表达能力
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
            ))

        # 层次化聚合权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # 分类头 - 增强版
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, out_channels)
        )

        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 分离坐标和特征
        coords = x[:, :3]
        features = x[:, 3:] if x.shape[1] > 3 else torch.zeros(x.shape[0], 1, device=x.device)

        # 编码
        spatial_feat = self.spatial_encoder(coords)
        other_feat = self.feature_encoder(features)

        # 融合
        h = self.fusion_layer(torch.cat([spatial_feat, other_feat], dim=1))

        # 准备边特征 (距离权重)
        edge_attr = None
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)

        # 多层GAT with 残差连接、FFN和层次聚合
        layer_outputs = []
        for i in range(self.num_layers):
            h_residual = h

            # GAT卷积
            h = self.gat_layers[i](h, edge_index, edge_attr=edge_attr)
            h = self.layer_norms[i](h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # 残差连接
            h = h + h_residual

            # FFN with residual
            h_ffn = self.ffn_layers[i](h)
            h = h + h_ffn

            layer_outputs.append(h)

        # 层次化聚合
        weights = F.softmax(self.layer_weights, dim=0)
        h_final = sum(w * out for w, out in zip(weights, layer_outputs))

        # 分类
        out = self.classifier(h_final)
        return out

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(x, edge_index, edge_weight)
        return torch.argmax(logits, dim=1)


class GeoTransformer(nn.Module):
    """
    基于Transformer的地质建模网络 (增强版)
    结合了深度空间编码、TransformerConv和Jumping Knowledge
    用于捕捉长距离地质依赖并防止过平滑

    优化改进:
    1. 更深的编码器
    2. 添加FFN层
    3. 更好的残差连接
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 10,
        num_layers: int = 4,
        dropout: float = 0.3,
        heads: int = 4
    ):
        super(GeoTransformer, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # 确保 hidden_channels 能被 heads 整除
        if hidden_channels % heads != 0:
            hidden_channels = (hidden_channels // heads) * heads
        self.hidden_channels = hidden_channels

        # 1. 深度空间编码器 (处理XYZ坐标) - 增强版
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # 2. 特征编码器 (处理其他属性) - 增强版
        feature_dim = max(in_channels - 3, 1)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # 3. 特征融合层 - 增强版
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # 4. Transformer层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()  # 添加FFN

        for _ in range(num_layers):
            # beta=True 允许模型学习边权重的缩放
            self.convs.append(
                TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                                dropout=dropout, edge_dim=1, beta=True)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
            # FFN层
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 4, hidden_channels),
            ))

        # 5. 层次聚合权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # 6. 分类头 - 使用加权聚合而非JK拼接
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        # 分离坐标和特征
        coords = x[:, :3]
        features = x[:, 3:] if x.shape[1] > 3 else torch.zeros(x.shape[0], 1, device=x.device)

        # 编码
        spatial_feat = self.spatial_encoder(coords)
        other_feat = self.feature_encoder(features)

        # 融合
        h = self.fusion_layer(torch.cat([spatial_feat, other_feat], dim=1))

        # 准备边特征
        edge_attr = None
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)

        # 保存每一层的输出用于层次聚合
        layer_outputs = []

        # Transformer 卷积循环
        for i in range(self.num_layers):
            h_in = h

            # TransformerConv
            h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h = self.norms[i](h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # 残差连接
            h = h + h_in

            # FFN with residual
            h_ffn = self.ffn_layers[i](h)
            h = h + h_ffn

            layer_outputs.append(h)

        # 层次化聚合 (加权平均)
        weights = F.softmax(self.layer_weights, dim=0)
        h_final = sum(w * out for w, out in zip(weights, layer_outputs))

        return self.classifier(h_final)

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(x, edge_index, edge_weight)
        return torch.argmax(logits, dim=1)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    模型工厂函数

    Args:
        model_name: 模型名称 ('gcn', 'graphsage', 'gat', 'geo3d', 'enhanced', 'transformer')
        **kwargs: 模型参数

    Returns:
        model: 对应的GNN模型实例
    """
    models = {
        'gcn': GeoGCN,
        'graphsage': GeoGraphSAGE,
        'gat': GeoGAT,
        'geo3d': Geo3DGNN,
        'enhanced': EnhancedGeoGNN,
        'transformer': GeoTransformer
    }

    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")

    return models[model_name.lower()](**kwargs)


def get_thickness_model(model_type: str = 'sage', **kwargs) -> nn.Module:
    """
    获取厚度预测回归模型

    Args:
        model_type: 模型类型 ('sage', 'gat')
        **kwargs: 模型参数
            - in_channels: 输入特征维度
            - hidden_channels: 隐藏层维度
            - num_layers: GNN层数
            - num_output_layers: 输出的岩层数量
            - dropout: dropout比率

    Returns:
        model: 厚度预测模型
    """
    from src.layer_modeling import GNNThicknessPredictor

    return GNNThicknessPredictor(
        in_channels=kwargs.get('in_channels', 4),
        hidden_channels=kwargs.get('hidden_channels', 128),
        num_layers=kwargs.get('num_layers', 4),
        num_output_layers=kwargs.get('num_output_layers', 5),
        dropout=kwargs.get('dropout', 0.3),
        model_type=model_type
    )


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
