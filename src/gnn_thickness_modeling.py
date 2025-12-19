"""
GNN厚度预测建模模块 (重构版)

核心思想：
1. GNN预测的是每层的【厚度】（回归问题），而不是岩性（分类问题）
2. 用GNN预测的厚度网格取代传统插值方法
3. 将厚度预测结果代入成熟项目的【层序累加】流程，构建三维模型

这样做的好处：
- 符合地质沉积规律（自下而上逐层累加）
- 数学上保证层间无冲突无空缺
- GNN增强厚度预测精度，传统方法保证模型合理性
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, TransformerConv, SAGEConv
from scipy.spatial import KDTree
from scipy.interpolate import griddata, Rbf
from typing import Optional, Tuple, List, Dict, Union
import warnings


# =============================================================================
# 一、数据结构定义（来自成熟项目）
# =============================================================================

class BlockModel:
    """
    地层块体模型（来自成熟项目 geological_modeling_algorithms）

    每个BlockModel代表一个地层的三维几何形态：
    - top_surface: 顶面高程网格 (ny, nx)
    - bottom_surface: 底面高程网格 (ny, nx)
    - thickness_grid: 厚度网格 = top_surface - bottom_surface
    """

    def __init__(
        self,
        name: str,
        points: int,
        top_surface: np.ndarray,
        bottom_surface: np.ndarray
    ):
        self.name = name
        self.points = points  # 数据点数量
        self.top_surface = np.asarray(top_surface, dtype=float)
        self.bottom_surface = np.asarray(bottom_surface, dtype=float)

        # 计算厚度网格
        thickness = self.top_surface - self.bottom_surface
        thickness = np.clip(thickness, 0.0, None)
        self.thickness_grid = thickness

        # 统计信息
        def _safe_stat(func, array, default=0.0):
            try:
                value = func(array)
                if np.isfinite(value):
                    return float(value)
            except ValueError:
                pass
            return float(default)

        self.avg_thickness = _safe_stat(np.nanmean, thickness)
        self.max_thickness = _safe_stat(np.nanmax, thickness)
        self.avg_height = _safe_stat(np.nanmean, self.top_surface)
        self.max_height = _safe_stat(np.nanmax, self.top_surface)
        self.min_height = _safe_stat(np.nanmin, self.top_surface)
        self.avg_bottom = _safe_stat(np.nanmean, self.bottom_surface)
        self.min_bottom = _safe_stat(np.nanmin, self.bottom_surface)
        self.base = self.avg_bottom


# =============================================================================
# 二、GNN厚度预测模型
# =============================================================================

class GNNThicknessPredictor(nn.Module):
    """
    GNN厚度预测模型（回归任务）

    输入：钻孔节点特征（坐标、地质属性等）
    输出：每个钻孔位置各层的预测厚度

    与旧版分类模型的区别：
    - 输出是连续值（厚度），不是离散类别（岩性）
    - 使用MSE/MAE损失，不是交叉熵损失
    - Softplus确保输出非负
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_output_layers: int = 8,  # 预测的地层数量
        dropout: float = 0.2,
        heads: int = 4,
        conv_type: str = 'gatv2'  # 'gatv2' | 'transformer' | 'sage'
    ):
        super().__init__()

        # 确保 hidden_channels 能被 heads 整除
        if hidden_channels % heads != 0:
            old_hidden = hidden_channels
            hidden_channels = (hidden_channels // heads) * heads
            if hidden_channels < 64:
                hidden_channels = 64
            warnings.warn(f"hidden_channels {old_hidden} 不能被 heads {heads} 整除，"
                         f"已自动调整为 {hidden_channels}")

        self.num_layers = num_layers
        self.num_output_layers = num_output_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.hidden_channels = hidden_channels

        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # 图卷积层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == 'transformer':
                self.convs.append(
                    TransformerConv(hidden_channels, hidden_channels // heads,
                                   heads=heads, dropout=dropout, edge_dim=1, beta=True)
                )
            elif conv_type == 'gatv2':
                self.convs.append(
                    GATv2Conv(hidden_channels, hidden_channels // heads,
                             heads=heads, dropout=dropout, concat=True, edge_dim=1)
                )
            else:  # sage
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # 双头输出
        # 存在性头：预测该层是否存在（二分类）
        self.exist_head = nn.Linear(hidden_channels, num_output_layers)
        # 厚度头：预测厚度值（回归）
        self.thick_head = nn.Linear(hidden_channels, num_output_layers)
        # Softplus确保厚度非负
        self.softplus = nn.Softplus()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征（距离权重）[num_edges, 1]

        Returns:
            thickness: 预测厚度 [num_nodes, num_output_layers]
            exist_logit: 存在性logits [num_nodes, num_output_layers]
        """
        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            if isinstance(conv, (TransformerConv, GATv2Conv)):
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # 残差连接
            if h.shape == h_res.shape:
                h = h + h_res

        exist_logit = self.exist_head(h)
        thickness = self.softplus(self.thick_head(h))

        return thickness, exist_logit


# =============================================================================
# 三、厚度预测损失函数
# =============================================================================

class ThicknessLoss(nn.Module):
    """
    厚度预测损失函数 - 针对地质数据优化

    关键改进：
    1. 使用对数空间计算误差（处理0.5m~70m的大范围）
    2. 相对误差而非绝对误差
    3. 降低存在性损失权重
    """

    def __init__(
        self,
        thick_weight: float = 1.0,
        exist_weight: float = 0.1,  # 降低存在性损失权重
        smooth_weight: float = 0.0,  # 关闭平滑正则化
        huber_delta: float = 2.0
    ):
        super().__init__()
        self.thick_weight = thick_weight
        self.exist_weight = exist_weight
        self.smooth_weight = smooth_weight
        self.huber_delta = huber_delta

    def forward(
        self,
        pred_thick: torch.Tensor,
        pred_exist: torch.Tensor,
        target_thick: torch.Tensor,
        target_exist: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        if mask is None:
            mask = torch.ones_like(target_thick)

        # 仅在存在的层计算厚度损失
        thick_mask = mask * target_exist
        valid_count = thick_mask.sum() + 1e-8

        # 方法1：对数空间MSE（推荐）
        # 将厚度映射到对数空间，使小厚度和大厚度的误差权重相当
        eps = 0.1  # 防止log(0)
        pred_log = torch.log(pred_thick + eps)
        target_log = torch.log(target_thick + eps)
        log_mse = ((pred_log - target_log) ** 2 * thick_mask).sum() / valid_count

        # 方法2：相对误差（补充）
        relative_error = (pred_thick - target_thick) / (target_thick + 1.0)
        rel_mse = ((relative_error ** 2) * thick_mask).sum() / valid_count

        # 组合厚度损失
        thick_loss = 0.7 * log_mse + 0.3 * rel_mse

        # 简化的存在性损失
        exist_loss = F.binary_cross_entropy_with_logits(
            pred_exist, target_exist, reduction='none'
        )
        exist_loss = (exist_loss * mask).mean()

        # 总损失
        total_loss = self.thick_weight * thick_loss + self.exist_weight * exist_loss

        loss_dict = {
            'total': total_loss.item(),
            'thick': thick_loss.item(),
            'exist': exist_loss.item(),
            'smooth': 0.0
        }

        return total_loss, loss_dict


# =============================================================================
# 四、GNN厚度预测器（训练与预测接口）
# =============================================================================

class GNNThicknessEstimator:
    """
    GNN厚度预测器

    封装模型训练和预测的完整流程
    """

    def __init__(
        self,
        layer_order: List[str],
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = 'gatv2',
        device: str = 'auto'
    ):
        """
        初始化

        Args:
            layer_order: 地层顺序列表（从底到顶）
            hidden_channels: 隐藏层维度
            num_layers: GNN层数
            dropout: dropout比率
            conv_type: 卷积类型
            device: 计算设备
        """
        self.layer_order = layer_order
        self.layer_to_idx = {name: i for i, name in enumerate(layer_order)}
        self.num_output_layers = len(layer_order)

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

        self.model = None
        self.is_trained = False

    def _build_model(self, in_channels: int):
        """构建模型"""
        self.model = GNNThicknessPredictor(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            num_output_layers=self.num_output_layers,
            dropout=self.dropout,
            conv_type=self.conv_type
        ).to(self.device)

    def train(
        self,
        data: Data,
        epochs: int = 200,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 30,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            data: PyG Data对象，包含：
                - x: 节点特征
                - edge_index: 边索引
                - edge_attr: 边特征
                - y_thick: 目标厚度 [N, L]
                - y_exist: 目标存在性 [N, L]
                - y_mask: 有效掩码 [N, L]
                - train_mask, val_mask: 训练/验证掩码

        Returns:
            history: 训练历史
        """
        # 构建模型
        self._build_model(data.x.shape[1])

        # 移动数据到设备
        data = data.to(self.device)

        # 优化器和损失函数
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = ThicknessLoss()

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            optimizer.zero_grad()

            pred_thick, pred_exist = self.model(data.x, data.edge_index, data.edge_attr)

            train_loss, loss_dict = criterion(
                pred_thick[data.train_mask],
                pred_exist[data.train_mask],
                data.y_thick[data.train_mask],
                data.y_exist[data.train_mask],
                data.y_mask[data.train_mask]
            )

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # 计算训练MAE
            with torch.no_grad():
                train_mask_full = data.train_mask.unsqueeze(1) * data.y_exist.bool()
                train_mae = F.l1_loss(
                    pred_thick[train_mask_full],
                    data.y_thick[train_mask_full]
                ).item()

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                pred_thick, pred_exist = self.model(data.x, data.edge_index, data.edge_attr)

                val_loss, _ = criterion(
                    pred_thick[data.val_mask],
                    pred_exist[data.val_mask],
                    data.y_thick[data.val_mask],
                    data.y_exist[data.val_mask],
                    data.y_mask[data.val_mask]
                )

                val_mask_full = data.val_mask.unsqueeze(1) * data.y_exist.bool()
                val_mae = F.l1_loss(
                    pred_thick[val_mask_full],
                    data.y_thick[val_mask_full]
                ).item()

            # 学习率调度
            scheduler.step(val_loss)

            # 记录历史
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss.item():.4f}, "
                      f"val_loss={val_loss.item():.4f}, "
                      f"train_mae={train_mae:.4f}, "
                      f"val_mae={val_mae:.4f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_trained = True
        return history

    def predict_thickness_grid(
        self,
        borehole_coords: np.ndarray,
        borehole_features: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """
        预测厚度网格

        Args:
            borehole_coords: 钻孔坐标 [N, 2]
            borehole_features: 钻孔特征 [N, F]
            grid_x: X网格坐标 (nx,)
            grid_y: Y网格坐标 (ny,)
            edge_index: 边索引
            edge_attr: 边特征

        Returns:
            thickness_grids: 每层的厚度网格 {layer_name: (ny, nx)}
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        self.model.eval()

        # 预测钻孔位置的厚度
        x_tensor = torch.tensor(borehole_features, dtype=torch.float32).to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)

        with torch.no_grad():
            pred_thick, pred_exist = self.model(x_tensor, edge_index, edge_attr)
            pred_thick = pred_thick.cpu().numpy()
            pred_exist = torch.sigmoid(pred_exist).cpu().numpy()

        # 创建网格
        XI, YI = np.meshgrid(grid_x, grid_y)
        xi_flat = XI.flatten()
        yi_flat = YI.flatten()

        thickness_grids = {}

        for i, layer_name in enumerate(self.layer_order):
            # 获取该层的预测厚度
            layer_thick = pred_thick[:, i]
            layer_exist = pred_exist[:, i]

            # 仅使用存在该层的钻孔进行插值
            exist_mask = layer_exist > 0.5
            if exist_mask.sum() < 3:
                # 数据点太少，使用所有点
                exist_mask = np.ones(len(layer_thick), dtype=bool)

            x_valid = borehole_coords[exist_mask, 0]
            y_valid = borehole_coords[exist_mask, 1]
            z_valid = layer_thick[exist_mask]

            # 使用传统插值方法将点预测扩展到网格
            try:
                grid_thick = griddata(
                    (x_valid, y_valid), z_valid, (xi_flat, yi_flat),
                    method='linear'
                )
                # 填充NaN
                if np.any(np.isnan(grid_thick)):
                    nearest = griddata(
                        (x_valid, y_valid), z_valid, (xi_flat, yi_flat),
                        method='nearest'
                    )
                    grid_thick = np.where(np.isnan(grid_thick), nearest, grid_thick)
            except Exception as e:
                warnings.warn(f"插值失败: {e}，使用最近邻")
                grid_thick = griddata(
                    (x_valid, y_valid), z_valid, (xi_flat, yi_flat),
                    method='nearest'
                )

            # 确保非负
            grid_thick = np.clip(grid_thick, 0.5, None)
            grid_thick = grid_thick.reshape(XI.shape)

            thickness_grids[layer_name] = grid_thick

        return thickness_grids


# =============================================================================
# 五、三维地质模型构建器（核心逻辑来自成熟项目）
# =============================================================================

class GeologicalModelBuilder:
    """
    三维地质模型构建器

    核心算法来自成熟项目 geological_modeling_algorithms:
    1. 从底层开始，逐层累加厚度构建顶面
    2. 数学上保证层间无重叠无空缺
    3. enforce_columnwise_order 强制垂向顺序修正
    """

    def __init__(
        self,
        layer_order: List[str],
        resolution: int = 50,
        base_level: float = 0.0,
        gap_value: float = 0.0,
        min_thickness: float = 0.5
    ):
        """
        初始化

        Args:
            layer_order: 地层顺序（从底到顶）
            resolution: 网格分辨率
            base_level: 初始基准面高程
            gap_value: 层间间隙（米）
            min_thickness: 最小层厚（米）
        """
        self.layer_order = layer_order
        self.resolution = resolution
        self.base_level = base_level
        self.gap_value = gap_value
        self.min_thickness = min_thickness

    def build_model(
        self,
        thickness_grids: Dict[str, np.ndarray],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> Tuple[List[BlockModel], np.ndarray, np.ndarray]:
        """
        构建三维地质模型

        Args:
            thickness_grids: 每层的厚度网格 {layer_name: (ny, nx)}
            x_range: X坐标范围 (x_min, x_max)
            y_range: Y坐标范围 (y_min, y_max)

        Returns:
            block_models: BlockModel列表
            XI: X坐标网格
            YI: Y坐标网格
        """
        # 创建坐标网格
        x_grid = np.linspace(x_range[0], x_range[1], self.resolution)
        y_grid = np.linspace(y_range[0], y_range[1], self.resolution)
        XI, YI = np.meshgrid(x_grid, y_grid)

        block_models = []
        current_base_surface = np.full(XI.shape, self.base_level, dtype=float)

        print(f"\n[构建模型] 层序: {self.layer_order}")
        print(f"[构建模型] 网格分辨率: {self.resolution}x{self.resolution}")
        print(f"[构建模型] 基准面高程: {self.base_level}m")

        for layer_name in self.layer_order:
            if layer_name not in thickness_grids:
                print(f"  [警告] {layer_name} 无厚度数据，跳过")
                continue

            thickness_grid = thickness_grids[layer_name]

            # 确保尺寸匹配
            if thickness_grid.shape != XI.shape:
                # 需要重采样
                from scipy.ndimage import zoom
                zoom_factors = (XI.shape[0] / thickness_grid.shape[0],
                               XI.shape[1] / thickness_grid.shape[1])
                thickness_grid = zoom(thickness_grid, zoom_factors, order=1)

            # 处理无效值
            thickness_grid = np.nan_to_num(thickness_grid, nan=self.min_thickness)
            thickness_grid = np.clip(thickness_grid, self.min_thickness, None)

            # 计算顶底面
            bottom_surface = current_base_surface.copy()
            top_surface = bottom_surface + thickness_grid

            # 创建BlockModel
            block_model = BlockModel(
                name=layer_name,
                points=int(np.sum(thickness_grid > 0)),
                top_surface=top_surface,
                bottom_surface=bottom_surface
            )
            block_models.append(block_model)

            print(f"  [建模] {layer_name}: "
                  f"底面={bottom_surface.mean():.2f}m, "
                  f"厚度={thickness_grid.mean():.2f}m, "
                  f"顶面={top_surface.mean():.2f}m")

            # 更新下一层的基准面
            current_base_surface = top_surface + self.gap_value

        # 强制垂向顺序修正
        self.enforce_columnwise_order(block_models)

        return block_models, XI, YI

    def enforce_columnwise_order(
        self,
        block_models: List[BlockModel],
        min_gap: float = 0.0
    ):
        """
        逐列强制垂向顺序（来自成熟项目）

        对每个(y,x)垂直柱子强制重排层序，确保相邻层之间无重叠

        注意: min_gap 设为 0 以保证层间节点共享，这对 FLAC3D 导出很重要
        """
        if len(block_models) < 2:
            return

        nlay = len(block_models)
        bottoms = np.stack([bm.bottom_surface for bm in block_models])
        tops = np.stack([bm.top_surface for bm in block_models])

        ny, nx = bottoms.shape[1:]
        fixed_count = 0

        for j in range(ny):
            for i in range(nx):
                bcol = bottoms[:, j, i]
                tcol = tops[:, j, i]

                valid_idx = np.where(np.isfinite(bcol) & np.isfinite(tcol))[0]
                if valid_idx.size == 0:
                    continue

                order = valid_idx[np.argsort(bcol[valid_idx])]

                # 检查是否需要修复
                needs_fix = False
                for ii in range(len(order) - 1):
                    if tops[order[ii], j, i] + min_gap > bottoms[order[ii+1], j, i]:
                        needs_fix = True
                        break

                if not needs_fix:
                    continue

                fixed_count += 1
                z_cur = float(np.min(bcol[valid_idx]))

                for idx in order:
                    thick = float(tcol[idx] - bcol[idx])
                    if not np.isfinite(thick) or thick < self.min_thickness:
                        thick = self.min_thickness

                    bottoms[idx, j, i] = z_cur
                    tops[idx, j, i] = z_cur + thick
                    z_cur = tops[idx, j, i] + min_gap

        # 写回BlockModel
        for k, bm in enumerate(block_models):
            bm.bottom_surface = bottoms[k]
            bm.top_surface = tops[k]
            bm.thickness_grid = tops[k] - bottoms[k]

        total_cells = ny * nx
        print(f"[垂向修正] 修复了 {fixed_count}/{total_cells} 个柱子 ({100*fixed_count/total_cells:.1f}%)")

    def build_voxel_model(
        self,
        block_models: List[BlockModel],
        XI: np.ndarray,
        YI: np.ndarray,
        nz: int = 50
    ) -> Tuple[np.ndarray, Dict]:
        """
        从BlockModel构建体素模型

        Args:
            block_models: BlockModel列表
            XI, YI: 坐标网格
            nz: Z方向网格数

        Returns:
            voxel_grid: 体素网格 (nx, ny, nz)，值为岩性索引
            grid_info: 网格信息
        """
        # 确定Z范围
        z_min = min(bm.bottom_surface.min() for bm in block_models)
        z_max = max(bm.top_surface.max() for bm in block_models)

        # 稍微扩展范围
        z_range = z_max - z_min
        z_min -= z_range * 0.05
        z_max += z_range * 0.05

        ny, nx = XI.shape
        z_grid = np.linspace(z_min, z_max, nz)

        # 创建岩性到索引的映射
        layer_to_idx = {bm.name: i+1 for i, bm in enumerate(block_models)}

        # 初始化体素网格（0表示空）
        voxel_grid = np.zeros((nx, ny, nz), dtype=np.int32)

        # 填充体素
        for k, z in enumerate(z_grid):
            for bm in block_models:
                # 找出该层包含此z的位置
                mask = (z >= bm.bottom_surface) & (z < bm.top_surface)
                # 注意：mask的形状是(ny, nx)，需要转置
                for j in range(ny):
                    for i in range(nx):
                        if mask[j, i]:
                            voxel_grid[i, j, k] = layer_to_idx[bm.name]

        grid_info = {
            'x_grid': XI[0, :],
            'y_grid': YI[:, 0],
            'z_grid': z_grid,
            'layer_to_idx': layer_to_idx,
            'idx_to_layer': {v: k for k, v in layer_to_idx.items()}
        }

        return voxel_grid, grid_info


# =============================================================================
# 六、统一接口：GNN增强的地质建模
# =============================================================================

class GNNGeologicalModeling:
    """
    GNN增强的地质建模统一接口

    工作流程：
    1. 加载钻孔数据
    2. 训练GNN厚度预测模型
    3. 预测厚度网格
    4. 使用层序累加法构建三维模型
    """

    def __init__(
        self,
        layer_order: List[str],
        resolution: int = 50,
        base_level: float = 0.0,
        gap_value: float = 0.0
    ):
        self.layer_order = layer_order
        self.resolution = resolution
        self.base_level = base_level
        self.gap_value = gap_value

        self.thickness_estimator = None
        self.model_builder = None
        self.block_models = None
        self.grid_info = None

    def fit(
        self,
        data: Data,
        epochs: int = 200,
        lr: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        训练GNN厚度预测模型
        """
        self.thickness_estimator = GNNThicknessEstimator(
            layer_order=self.layer_order
        )
        history = self.thickness_estimator.train(
            data=data,
            epochs=epochs,
            lr=lr,
            verbose=verbose
        )
        return history

    def build(
        self,
        borehole_coords: np.ndarray,
        borehole_features: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None
    ) -> List[BlockModel]:
        """
        构建三维地质模型

        Args:
            borehole_coords: 钻孔坐标 [N, 2]
            borehole_features: 钻孔特征 [N, F]
            edge_index: 边索引
            edge_attr: 边特征
            x_range, y_range: 坐标范围（None则自动计算）

        Returns:
            block_models: BlockModel列表
        """
        if self.thickness_estimator is None or not self.thickness_estimator.is_trained:
            raise RuntimeError("请先调用fit()训练模型")

        # 确定坐标范围
        if x_range is None:
            x_range = (borehole_coords[:, 0].min(), borehole_coords[:, 0].max())
        if y_range is None:
            y_range = (borehole_coords[:, 1].min(), borehole_coords[:, 1].max())

        # 创建网格
        grid_x = np.linspace(x_range[0], x_range[1], self.resolution)
        grid_y = np.linspace(y_range[0], y_range[1], self.resolution)

        # 预测厚度网格
        thickness_grids = self.thickness_estimator.predict_thickness_grid(
            borehole_coords=borehole_coords,
            borehole_features=borehole_features,
            grid_x=grid_x,
            grid_y=grid_y,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # 构建模型
        self.model_builder = GeologicalModelBuilder(
            layer_order=self.layer_order,
            resolution=self.resolution,
            base_level=self.base_level,
            gap_value=self.gap_value
        )

        self.block_models, XI, YI = self.model_builder.build_model(
            thickness_grids=thickness_grids,
            x_range=x_range,
            y_range=y_range
        )

        self.grid_info = {
            'XI': XI,
            'YI': YI,
            'x_range': x_range,
            'y_range': y_range,
            'resolution': self.resolution
        }

        return self.block_models

    def get_voxel_model(self, nz: int = 50) -> Tuple[np.ndarray, Dict]:
        """
        获取体素模型
        """
        if self.block_models is None:
            raise RuntimeError("请先调用build()构建模型")

        return self.model_builder.build_voxel_model(
            block_models=self.block_models,
            XI=self.grid_info['XI'],
            YI=self.grid_info['YI'],
            nz=nz
        )


# =============================================================================
# 七、传统插值方法（作为对比基准）
# =============================================================================

class TraditionalThicknessInterpolator:
    """
    传统厚度插值方法

    用于：
    1. 作为GNN方法的对比基准
    2. 当GNN训练数据不足时的回退方案
    """

    def __init__(
        self,
        method: str = 'linear',
        layer_order: List[str] = None
    ):
        """
        初始化

        Args:
            method: 插值方法 ('linear', 'cubic', 'nearest', 'rbf', 'kriging')
            layer_order: 地层顺序
        """
        self.method = method
        self.layer_order = layer_order or []

    def interpolate_thickness(
        self,
        borehole_data: pd.DataFrame,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        layer_col: str = 'lithology',
        thickness_col: str = 'layer_thickness',
        x_col: str = 'x',
        y_col: str = 'y'
    ) -> Dict[str, np.ndarray]:
        """
        插值厚度网格

        Args:
            borehole_data: 钻孔层表数据
            grid_x, grid_y: 网格坐标
            layer_col: 地层列名
            thickness_col: 厚度列名
            x_col, y_col: 坐标列名

        Returns:
            thickness_grids: {layer_name: (ny, nx)}
        """
        XI, YI = np.meshgrid(grid_x, grid_y)
        xi_flat = XI.flatten()
        yi_flat = YI.flatten()

        thickness_grids = {}

        for layer_name in self.layer_order:
            layer_data = borehole_data[borehole_data[layer_col] == layer_name]

            if len(layer_data) < 3:
                # 数据点太少，使用默认厚度
                thickness_grids[layer_name] = np.full(XI.shape, 1.0)
                continue

            x = layer_data[x_col].values
            y = layer_data[y_col].values
            z = layer_data[thickness_col].values

            try:
                if self.method in ['linear', 'cubic', 'nearest']:
                    grid_thick = griddata((x, y), z, (xi_flat, yi_flat), method=self.method)
                elif self.method == 'rbf':
                    rbf = Rbf(x, y, z, function='thin_plate', smooth=0.5)
                    grid_thick = rbf(xi_flat, yi_flat)
                else:
                    grid_thick = griddata((x, y), z, (xi_flat, yi_flat), method='linear')

                # 处理NaN
                if np.any(np.isnan(grid_thick)):
                    fill_value = np.nanmedian(z)
                    grid_thick = np.nan_to_num(grid_thick, nan=fill_value)

                grid_thick = np.clip(grid_thick, 0.5, None)
                grid_thick = grid_thick.reshape(XI.shape)

            except Exception as e:
                warnings.warn(f"{layer_name} 插值失败: {e}")
                grid_thick = np.full(XI.shape, np.median(z))

            thickness_grids[layer_name] = grid_thick

        return thickness_grids


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("GNN厚度预测建模模块测试...")

    # 模拟数据
    np.random.seed(42)
    num_boreholes = 20
    num_layers = 5

    # 模拟钻孔坐标
    coords = np.random.rand(num_boreholes, 2) * 1000

    # 模拟厚度数据
    y_thick = np.random.rand(num_boreholes, num_layers) * 5 + 1
    y_exist = (np.random.rand(num_boreholes, num_layers) > 0.2).astype(float)
    y_thick = y_thick * y_exist

    # 模拟特征
    features = np.column_stack([
        coords,
        np.random.rand(num_boreholes, 2)
    ])

    # 构建图
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    edges = []
    for i in range(num_boreholes):
        dists, indices = tree.query(coords[i], k=5)
        for j in indices[1:]:
            edges.append([i, j])
            edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # 创建Data对象
    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.shape[1], 1),
        y_thick=torch.tensor(y_thick, dtype=torch.float32),
        y_exist=torch.tensor(y_exist, dtype=torch.float32),
        y_mask=torch.ones(num_boreholes, num_layers),
        train_mask=torch.zeros(num_boreholes, dtype=torch.bool),
        val_mask=torch.zeros(num_boreholes, dtype=torch.bool)
    )
    data.train_mask[:15] = True
    data.val_mask[15:] = True

    # 测试GNN厚度预测
    layer_order = [f'Layer_{i}' for i in range(num_layers)]
    modeling = GNNGeologicalModeling(
        layer_order=layer_order,
        resolution=30,
        base_level=0.0
    )

    print("\n训练GNN厚度预测模型...")
    history = modeling.fit(data, epochs=50, verbose=True)

    print("\n构建三维模型...")
    block_models = modeling.build(
        borehole_coords=coords,
        borehole_features=features,
        edge_index=edge_index
    )

    print(f"\n构建完成! 共 {len(block_models)} 层:")
    for bm in block_models:
        print(f"  {bm.name}: 平均厚度={bm.avg_thickness:.2f}m, "
              f"底面={bm.avg_bottom:.2f}m, 顶面={bm.avg_height:.2f}m")
