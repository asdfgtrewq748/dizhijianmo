"""
层序累加地质建模模块

核心思想:
1. 确定底面作为基准
2. 从最深层开始，逐层累加厚度构建曲面
3. 曲面之间的空间即为该层岩体
4. GNN用于预测每层厚度（回归问题）

优势:
- 无岩体冲突（数学上不可能）
- 无空缺区域（完全填充）
- 符合地质沉积规律
- 曲面连续光滑
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, List, Optional, Union
import os


class LayerDataProcessor:
    """层序数据处理器"""

    def __init__(
        self,
        k_neighbors: int = 10,
        normalize_coords: bool = True,
        normalize_thickness: bool = True
    ):
        self.k_neighbors = k_neighbors
        self.normalize_coords = normalize_coords
        self.normalize_thickness = normalize_thickness

        # 归一化参数
        self.coord_mean = None
        self.coord_std = None
        self.thickness_mean = None
        self.thickness_std = None

        # 层序信息
        self.layer_order = None
        self.num_layers = 0

    def infer_layer_order(self, df: pd.DataFrame) -> List[str]:
        """
        从钻孔数据推断岩层顺序（从深到浅）

        逻辑：按各岩性的平均深度排序
        """
        # 计算每种岩性的平均高程（z值，负值表示深度）
        layer_stats = df.groupby('lithology').agg({
            'z': 'mean',
            'borehole_id': 'nunique',  # 出现在多少钻孔中
            'layer_thickness': 'sum'   # 总厚度
        }).reset_index()

        # 按平均高程排序（从小到大 = 从深到浅）
        layer_stats = layer_stats.sort_values('z')

        self.layer_order = layer_stats['lithology'].tolist()
        self.num_layers = len(self.layer_order)

        print(f"推断的岩层顺序（从深到浅）:")
        for i, layer in enumerate(self.layer_order):
            stats = layer_stats[layer_stats['lithology'] == layer].iloc[0]
            print(f"  {i+1}. {layer}: 平均高程={stats['z']:.1f}m, "
                  f"出现在{int(stats['borehole_id'])}个钻孔")

        return self.layer_order

    def extract_thickness_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取每个钻孔各层的厚度数据

        返回: DataFrame，每行一个钻孔，列为各层厚度
        """
        if self.layer_order is None:
            self.infer_layer_order(df)

        thickness_data = []

        for bh_id in df['borehole_id'].unique():
            bh_data = df[df['borehole_id'] == bh_id]

            record = {
                'borehole_id': bh_id,
                'x': bh_data['x'].iloc[0],
                'y': bh_data['y'].iloc[0],
                'total_depth': bh_data['z'].min(),  # 最深点
                'surface_elevation': bh_data['z'].max()  # 地表高程
            }

            # 提取各层厚度
            for layer_name in self.layer_order:
                layer_data = bh_data[bh_data['lithology'] == layer_name]
                if len(layer_data) > 0:
                    # 同一岩性可能有多层，求总厚度
                    thickness = layer_data['layer_thickness'].sum()
                else:
                    thickness = 0.0  # 该层缺失
                record[f'thickness_{layer_name}'] = thickness

            thickness_data.append(record)

        return pd.DataFrame(thickness_data)

    def build_graph_data(
        self,
        thickness_df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[Data, Dict]:
        """
        构建用于GNN训练的图数据

        特征: 坐标 + 其他地质特征
        标签: 各层厚度
        """
        n_samples = len(thickness_df)

        # 1. 提取坐标
        coords = thickness_df[['x', 'y']].values.astype(np.float32)

        # 2. 提取厚度标签
        thickness_cols = [f'thickness_{layer}' for layer in self.layer_order]
        thicknesses = thickness_df[thickness_cols].values.astype(np.float32)

        # 3. 归一化
        if self.normalize_coords:
            self.coord_mean = coords.mean(axis=0)
            self.coord_std = coords.std(axis=0) + 1e-8
            coords_normalized = (coords - self.coord_mean) / self.coord_std
        else:
            coords_normalized = coords

        if self.normalize_thickness:
            self.thickness_mean = thicknesses.mean(axis=0)
            self.thickness_std = thicknesses.std(axis=0) + 1e-8
            thicknesses_normalized = (thicknesses - self.thickness_mean) / self.thickness_std
        else:
            thicknesses_normalized = thicknesses

        # 4. 构建特征（坐标 + 总深度 + 地表高程）
        total_depth = thickness_df['total_depth'].values.reshape(-1, 1)
        surface_elev = thickness_df['surface_elevation'].values.reshape(-1, 1)

        # 归一化深度特征
        depth_normalized = (total_depth - total_depth.mean()) / (total_depth.std() + 1e-8)
        elev_normalized = (surface_elev - surface_elev.mean()) / (surface_elev.std() + 1e-8)

        features = np.concatenate([
            coords_normalized,
            depth_normalized,
            elev_normalized
        ], axis=1).astype(np.float32)

        # 5. 构建图结构（KNN）
        tree = KDTree(coords)
        edge_list = []
        edge_weights = []

        for i in range(n_samples):
            distances, indices = tree.query(coords[i], k=min(self.k_neighbors + 1, n_samples))
            for j, (dist, idx) in enumerate(zip(distances[1:], indices[1:])):
                edge_list.append([i, idx])
                edge_list.append([idx, i])
                weight = np.exp(-dist / (distances[1:].mean() + 1e-8))
                edge_weights.extend([weight, weight])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

        # 去重边
        edge_index, unique_idx = torch.unique(edge_index, dim=1, return_inverse=True)
        # 对重复边的权重取平均
        edge_weight_new = torch.zeros(edge_index.shape[1])
        count = torch.zeros(edge_index.shape[1])
        for i, idx in enumerate(unique_idx):
            edge_weight_new[idx] += edge_weights[i]
            count[idx] += 1
        edge_weight = edge_weight_new / count

        # 6. 数据集划分
        indices = np.random.permutation(n_samples)
        test_end = int(n_samples * test_size)
        val_end = test_end + int(n_samples * val_size)

        test_mask = torch.zeros(n_samples, dtype=torch.bool)
        val_mask = torch.zeros(n_samples, dtype=torch.bool)
        train_mask = torch.zeros(n_samples, dtype=torch.bool)

        test_mask[indices[:test_end]] = True
        val_mask[indices[test_end:val_end]] = True
        train_mask[indices[val_end:]] = True

        # 7. 创建Data对象
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            y=torch.tensor(thicknesses_normalized, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            coords=torch.tensor(coords, dtype=torch.float32),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        # 保存原始厚度用于反归一化
        data.y_original = torch.tensor(thicknesses, dtype=torch.float32)

        result = {
            'data': data,
            'num_features': features.shape[1],
            'num_layers': self.num_layers,
            'layer_order': self.layer_order,
            'thickness_df': thickness_df
        }

        return data, result

    def denormalize_thickness(self, normalized_thickness: np.ndarray) -> np.ndarray:
        """反归一化厚度"""
        if self.normalize_thickness and self.thickness_mean is not None:
            return normalized_thickness * self.thickness_std + self.thickness_mean
        return normalized_thickness


class GNNThicknessPredictor(nn.Module):
    """
    GNN厚度预测模型

    输入: 节点特征 (坐标 + 地质特征)
    输出: 各层厚度预测值
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_output_layers: int,
        dropout: float = 0.3,
        model_type: str = 'sage'
    ):
        super().__init__()

        self.num_gnn_layers = num_layers
        self.dropout = dropout

        # GNN编码器
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if model_type == 'sage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif model_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels // 4, heads=4, concat=True))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True))

        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_channels))

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, num_output_layers)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # GNN编码
        for i, conv in enumerate(self.convs):
            if edge_weight is not None and hasattr(conv, 'edge_weight'):
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 回归预测
        thickness = self.regressor(x)

        return thickness


class ThicknessTrainer:
    """厚度预测模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )

        self.best_val_loss = float('inf')
        self.best_model_state = None

    def compute_loss(self, pred, target, mask=None):
        """计算损失"""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        # MSE损失
        mse_loss = F.mse_loss(pred, target)

        # 平滑L1损失（对异常值更鲁棒）
        smooth_l1 = F.smooth_l1_loss(pred, target)

        return 0.5 * mse_loss + 0.5 * smooth_l1

    def train(
        self,
        data: Data,
        epochs: int = 300,
        patience: int = 50,
        verbose: bool = True,
        save_path: str = None
    ) -> Dict:
        """训练模型"""
        data = data.to(self.device)

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': []
        }

        no_improve = 0

        for epoch in range(epochs):
            # 训练
            self.model.train()
            self.optimizer.zero_grad()

            out = self.model(data.x, data.edge_index, data.edge_weight)
            train_loss = self.compute_loss(out, data.y, data.train_mask)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x, data.edge_index, data.edge_weight)
                val_loss = self.compute_loss(out, data.y, data.val_mask)

                # MAE
                val_pred = out[data.val_mask]
                val_true = data.y[data.val_mask]
                val_mae = torch.abs(val_pred - val_true).mean()

            # 学习率调整
            self.scheduler.step(val_loss)

            # 记录
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['val_mae'].append(val_mae.item())

            # 早停
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0

                if save_path:
                    torch.save(self.best_model_state, save_path)
            else:
                no_improve += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss.item():.4f} | "
                      f"Val Loss: {val_loss.item():.4f} | "
                      f"Val MAE: {val_mae.item():.4f}")

            if no_improve >= patience:
                print(f"早停于 Epoch {epoch+1}")
                break

        # 恢复最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return history

    def predict(self, data: Data) -> np.ndarray:
        """预测厚度"""
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_weight)

        return out.cpu().numpy()

    def evaluate(self, data: Data, processor: LayerDataProcessor) -> Dict:
        """评估模型"""
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_weight)

            # 测试集
            test_pred = out[data.test_mask].cpu().numpy()
            test_true = data.y[data.test_mask].cpu().numpy()

            # 反归一化
            test_pred_real = processor.denormalize_thickness(test_pred)
            test_true_real = processor.denormalize_thickness(test_true)

            # 计算指标
            mae = np.abs(test_pred_real - test_true_real).mean()
            rmse = np.sqrt(((test_pred_real - test_true_real) ** 2).mean())

            # 每层的指标
            layer_metrics = {}
            for i, layer_name in enumerate(processor.layer_order):
                layer_mae = np.abs(test_pred_real[:, i] - test_true_real[:, i]).mean()
                layer_rmse = np.sqrt(((test_pred_real[:, i] - test_true_real[:, i]) ** 2).mean())
                layer_metrics[layer_name] = {'MAE': layer_mae, 'RMSE': layer_rmse}

        results = {
            'overall_mae': mae,
            'overall_rmse': rmse,
            'layer_metrics': layer_metrics
        }

        print(f"\n评估结果:")
        print(f"  总体 MAE: {mae:.2f} m")
        print(f"  总体 RMSE: {rmse:.2f} m")
        print(f"\n  各层指标:")
        for layer_name, metrics in layer_metrics.items():
            print(f"    {layer_name}: MAE={metrics['MAE']:.2f}m, RMSE={metrics['RMSE']:.2f}m")

        return results


class LayerBasedGeologicalModeling:
    """
    基于层序累加的三维地质建模

    核心流程:
    1. 确定底面
    2. 对每层预测厚度分布
    3. 逐层累加构建曲面
    4. 体素化填充
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        use_gnn: bool = True,
        interpolation_method: str = 'rbf',
        smooth_surfaces: bool = True,
        smooth_sigma: float = 1.0
    ):
        self.resolution = resolution
        self.use_gnn = use_gnn
        self.interpolation_method = interpolation_method
        self.smooth_surfaces = smooth_surfaces
        self.smooth_sigma = smooth_sigma

        # 结果
        self.bounds = None
        self.grid_info = None
        self.surfaces = {}
        self.lithology_3d = None
        self.layer_order = None

    def _build_grid(self, df: pd.DataFrame, padding: float = 0.05):
        """构建二维网格（用于厚度预测）和三维网格（用于体素化）"""
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        z_min, z_max = df['z'].min(), df['z'].max()

        # 添加边距
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= x_range * padding
        x_max += x_range * padding
        y_min -= y_range * padding
        y_max += y_range * padding

        self.bounds = {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max)
        }

        nx, ny, nz = self.resolution

        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)

        self.grid_info = {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'z_grid': z_grid,
            'resolution': self.resolution
        }

        # 2D网格坐标
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
        self.grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

        return x_grid, y_grid, z_grid

    def _build_bottom_surface(self, df: pd.DataFrame) -> np.ndarray:
        """构建模型底面"""
        # 获取各钻孔最深点
        bottom_data = df.groupby('borehole_id').agg({
            'x': 'first',
            'y': 'first',
            'z': 'min'
        }).reset_index()

        points = bottom_data[['x', 'y']].values
        values = bottom_data['z'].values

        # RBF插值
        interpolator = RBFInterpolator(
            points, values,
            kernel='thin_plate_spline',
            smoothing=0.1
        )

        nx, ny, _ = self.resolution
        bottom_surface = interpolator(self.grid_xy).reshape(nx, ny)

        if self.smooth_surfaces:
            bottom_surface = gaussian_filter(bottom_surface, sigma=self.smooth_sigma)

        return bottom_surface

    def _interpolate_thickness(
        self,
        thickness_df: pd.DataFrame,
        layer_name: str
    ) -> np.ndarray:
        """使用传统插值预测某层厚度"""
        col_name = f'thickness_{layer_name}'

        points = thickness_df[['x', 'y']].values
        values = thickness_df[col_name].values

        # RBF插值
        if self.interpolation_method == 'rbf':
            interpolator = RBFInterpolator(
                points, values,
                kernel='thin_plate_spline',
                smoothing=0.1
            )
        else:
            interpolator = LinearNDInterpolator(points, values, fill_value=0)

        nx, ny, _ = self.resolution
        thickness = interpolator(self.grid_xy).reshape(nx, ny)

        # 确保非负
        thickness = np.maximum(thickness, 0)

        if self.smooth_surfaces:
            thickness = gaussian_filter(thickness, sigma=self.smooth_sigma)

        return thickness

    def _predict_thickness_gnn(
        self,
        trainer: ThicknessTrainer,
        processor: LayerDataProcessor,
        thickness_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """使用GNN预测所有层的厚度"""
        nx, ny, _ = self.resolution

        # 构建预测数据
        # 需要为网格点创建特征并连接到训练图

        # 获取训练数据的坐标
        train_coords = thickness_df[['x', 'y']].values
        n_train = len(train_coords)

        # 归一化网格坐标
        if processor.normalize_coords:
            grid_coords_normalized = (self.grid_xy - processor.coord_mean) / processor.coord_std
        else:
            grid_coords_normalized = self.grid_xy

        # 为网格点插值其他特征（总深度、地表高程）
        tree = KDTree(train_coords)
        distances, indices = tree.query(self.grid_xy, k=min(5, n_train))
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # 插值深度和高程
        total_depth = thickness_df['total_depth'].values
        surface_elev = thickness_df['surface_elevation'].values

        grid_depth = np.sum(weights * total_depth[indices], axis=1)
        grid_elev = np.sum(weights * surface_elev[indices], axis=1)

        # 归一化
        depth_mean, depth_std = total_depth.mean(), total_depth.std() + 1e-8
        elev_mean, elev_std = surface_elev.mean(), surface_elev.std() + 1e-8

        grid_depth_norm = (grid_depth - depth_mean) / depth_std
        grid_elev_norm = (grid_elev - elev_mean) / elev_std

        # 构建网格点特征
        grid_features = np.column_stack([
            grid_coords_normalized,
            grid_depth_norm.reshape(-1, 1),
            grid_elev_norm.reshape(-1, 1)
        ]).astype(np.float32)

        n_grid = len(self.grid_xy)

        # 构建扩展图：将网格点连接到训练节点
        new_edges = []
        new_weights = []

        for i in range(n_grid):
            grid_idx = n_train + i
            for dist, train_idx in zip(distances[i], indices[i]):
                weight = np.exp(-dist / (distances[i].mean() + 1e-8))
                new_edges.append([grid_idx, train_idx])
                new_edges.append([train_idx, grid_idx])
                new_weights.extend([weight, weight])

        # 获取原始训练图
        # 重新构建训练数据
        train_data, _ = processor.build_graph_data(thickness_df, test_size=0.0, val_size=0.0)

        # 合并边
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        extended_edge_index = torch.cat([train_data.edge_index, new_edge_index], dim=1)

        new_edge_weight = torch.tensor(new_weights, dtype=torch.float32)
        extended_edge_weight = torch.cat([train_data.edge_weight, new_edge_weight])

        # 合并特征
        grid_x = torch.tensor(grid_features, dtype=torch.float32)
        extended_x = torch.cat([train_data.x, grid_x], dim=0)

        # 预测
        trainer.model.eval()
        device = trainer.device

        extended_x = extended_x.to(device)
        extended_edge_index = extended_edge_index.to(device)
        extended_edge_weight = extended_edge_weight.to(device)

        with torch.no_grad():
            out = trainer.model(extended_x, extended_edge_index, extended_edge_weight)
            grid_out = out[n_train:].cpu().numpy()

        # 反归一化
        grid_thickness = processor.denormalize_thickness(grid_out)

        # 确保非负
        grid_thickness = np.maximum(grid_thickness, 0)

        # 转换为各层字典
        thickness_dict = {}
        for i, layer_name in enumerate(processor.layer_order):
            layer_thickness = grid_thickness[:, i].reshape(nx, ny)
            if self.smooth_surfaces:
                layer_thickness = gaussian_filter(layer_thickness, sigma=self.smooth_sigma)
            thickness_dict[layer_name] = layer_thickness

        return thickness_dict

    def build_model(
        self,
        df: pd.DataFrame,
        processor: LayerDataProcessor,
        trainer: ThicknessTrainer = None,
        thickness_df: pd.DataFrame = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        构建三维地质模型

        Args:
            df: 原始钻孔数据
            processor: 数据处理器
            trainer: GNN训练器（如果use_gnn=True）
            thickness_df: 厚度数据（如果已预处理）
            verbose: 是否打印详细信息

        Returns:
            lithology_3d: 三维岩性数组
        """
        if verbose:
            print("\n" + "=" * 60)
            print("层序累加地质建模")
            print("=" * 60)

        self.layer_order = processor.layer_order

        # 1. 构建网格
        if verbose:
            print("\n[1/4] 构建网格...")
        x_grid, y_grid, z_grid = self._build_grid(df)
        nx, ny, nz = self.resolution

        if verbose:
            print(f"  分辨率: {nx} x {ny} x {nz}")

        # 2. 构建底面
        if verbose:
            print("\n[2/4] 构建底面...")
        bottom_surface = self._build_bottom_surface(df)
        self.surfaces['_bottom'] = bottom_surface

        if verbose:
            print(f"  底面深度范围: {bottom_surface.min():.1f} ~ {bottom_surface.max():.1f} m")

        # 3. 预测各层厚度
        if verbose:
            print("\n[3/4] 预测各层厚度...")

        if thickness_df is None:
            thickness_df = processor.extract_thickness_data(df)

        if self.use_gnn and trainer is not None:
            if verbose:
                print("  使用GNN预测...")
            thickness_dict = self._predict_thickness_gnn(trainer, processor, thickness_df)
        else:
            if verbose:
                print("  使用传统插值...")
            thickness_dict = {}
            for layer_name in processor.layer_order:
                thickness_dict[layer_name] = self._interpolate_thickness(thickness_df, layer_name)

        # 4. 逐层累加构建曲面
        if verbose:
            print("\n[4/4] 逐层累加构建曲面...")

        current_surface = bottom_surface.copy()

        for layer_name in processor.layer_order:
            thickness = thickness_dict[layer_name]
            top_surface = current_surface + thickness

            self.surfaces[layer_name] = {
                'bottom': current_surface.copy(),
                'top': top_surface.copy(),
                'thickness': thickness
            }

            if verbose:
                mean_thickness = thickness.mean()
                print(f"  {layer_name}: 平均厚度 {mean_thickness:.2f} m")

            current_surface = top_surface

        # 5. 体素化
        if verbose:
            print("\n体素化填充...")

        self.lithology_3d = np.zeros((nx, ny, nz), dtype=np.int32)

        for layer_idx, layer_name in enumerate(processor.layer_order):
            surface_info = self.surfaces[layer_name]
            bottom = surface_info['bottom']
            top = surface_info['top']

            for i in range(nx):
                for j in range(ny):
                    z_bottom = bottom[i, j]
                    z_top = top[i, j]

                    # 找到对应的z索引
                    k_start = np.searchsorted(z_grid, z_bottom)
                    k_end = np.searchsorted(z_grid, z_top)

                    k_start = max(0, min(k_start, nz))
                    k_end = max(0, min(k_end, nz))

                    if k_end > k_start:
                        self.lithology_3d[i, j, k_start:k_end] = layer_idx

        if verbose:
            print("\n建模完成!")
            unique, counts = np.unique(self.lithology_3d, return_counts=True)
            total = counts.sum()
            for u, c in zip(unique, counts):
                if u < len(processor.layer_order):
                    name = processor.layer_order[u]
                    print(f"  {name}: {c:,} 体素 ({100*c/total:.1f}%)")

        return self.lithology_3d

    def get_voxel_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取体素模型"""
        if self.lithology_3d is None:
            raise ValueError("请先调用build_model构建模型")

        # 置信度：这里简化为1（因为层序模型没有不确定性）
        confidence = np.ones_like(self.lithology_3d, dtype=np.float32)

        return self.lithology_3d, confidence

    def get_slice(
        self,
        axis: str = 'z',
        index: int = None,
        position: float = None
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """获取切片"""
        if self.lithology_3d is None:
            raise ValueError("请先构建模型")

        nx, ny, nz = self.resolution
        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']

        if axis == 'x':
            if index is None:
                if position is not None:
                    index = np.argmin(np.abs(x_grid - position))
                else:
                    index = nx // 2
            slice_data = self.lithology_3d[index, :, :]
            yy, zz = np.meshgrid(y_grid, z_grid, indexing='ij')
            slice_coords = {'y': yy, 'z': zz}
            slice_info = {'axis': 'x', 'index': index, 'position': x_grid[index]}

        elif axis == 'y':
            if index is None:
                if position is not None:
                    index = np.argmin(np.abs(y_grid - position))
                else:
                    index = ny // 2
            slice_data = self.lithology_3d[:, index, :]
            xx, zz = np.meshgrid(x_grid, z_grid, indexing='ij')
            slice_coords = {'x': xx, 'z': zz}
            slice_info = {'axis': 'y', 'index': index, 'position': y_grid[index]}

        else:  # z
            if index is None:
                if position is not None:
                    index = np.argmin(np.abs(z_grid - position))
                else:
                    index = nz // 2
            slice_data = self.lithology_3d[:, :, index]
            xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
            slice_coords = {'x': xx, 'y': yy}
            slice_info = {'axis': 'z', 'index': index, 'position': z_grid[index]}

        return slice_data, slice_coords, slice_info

    def get_statistics(self, lithology_names: List[str] = None) -> pd.DataFrame:
        """获取统计信息"""
        if self.lithology_3d is None:
            raise ValueError("请先构建模型")

        lithology_names = lithology_names or self.layer_order
        nx, ny, nz = self.resolution

        # 计算体素体积
        cell_volume = (
            (self.bounds['x'][1] - self.bounds['x'][0]) / (nx - 1) *
            (self.bounds['y'][1] - self.bounds['y'][0]) / (ny - 1) *
            (self.bounds['z'][1] - self.bounds['z'][0]) / (nz - 1)
        )

        stats = []
        total_voxels = self.lithology_3d.size

        for i, layer_name in enumerate(lithology_names):
            mask = self.lithology_3d == i
            count = mask.sum()
            volume = count * cell_volume

            # 从surfaces获取平均厚度
            if layer_name in self.surfaces:
                mean_thickness = self.surfaces[layer_name]['thickness'].mean()
            else:
                mean_thickness = 0

            stats.append({
                '岩性': layer_name,
                '体素数': count,
                '体积 (m³)': volume,
                '占比 (%)': 100.0 * count / total_voxels,
                '平均厚度 (m)': mean_thickness
            })

        return pd.DataFrame(stats)

    def export_vtk(self, filepath: str, lithology_names: List[str] = None):
        """导出VTK格式"""
        if self.lithology_3d is None:
            raise ValueError("请先构建模型")

        lithology_names = lithology_names or self.layer_order
        nx, ny, nz = self.resolution

        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']

        with open(filepath, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Layer-Based 3D Geological Model\n")
            f.write("ASCII\n")
            f.write("DATASET RECTILINEAR_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")

            f.write(f"X_COORDINATES {nx} float\n")
            f.write(" ".join(f"{x:.6f}" for x in x_grid) + "\n")

            f.write(f"Y_COORDINATES {ny} float\n")
            f.write(" ".join(f"{y:.6f}" for y in y_grid) + "\n")

            f.write(f"Z_COORDINATES {nz} float\n")
            f.write(" ".join(f"{z:.6f}" for z in z_grid) + "\n")

            f.write(f"POINT_DATA {nx * ny * nz}\n")

            f.write("SCALARS lithology int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{self.lithology_3d[i, j, k]}\n")

        print(f"VTK文件已导出: {filepath}")

    def export_numpy(self, filepath: str):
        """导出NumPy格式"""
        np.savez_compressed(
            filepath,
            lithology=self.lithology_3d,
            x_grid=self.grid_info['x_grid'],
            y_grid=self.grid_info['y_grid'],
            z_grid=self.grid_info['z_grid'],
            layer_order=self.layer_order,
            method='layer_based'
        )
        print(f"NumPy文件已导出: {filepath}")


def build_layer_based_model(
    df: pd.DataFrame,
    resolution: Tuple[int, int, int] = (50, 50, 50),
    use_gnn: bool = True,
    epochs: int = 300,
    hidden_dim: int = 128,
    num_gnn_layers: int = 4,
    output_dir: str = 'output',
    verbose: bool = True
) -> Tuple[LayerBasedGeologicalModeling, LayerDataProcessor, Optional[ThicknessTrainer]]:
    """
    便捷函数：构建层序累加地质模型

    Args:
        df: 钻孔数据DataFrame
        resolution: 网格分辨率
        use_gnn: 是否使用GNN预测厚度
        epochs: 训练轮数
        hidden_dim: GNN隐藏层维度
        num_gnn_layers: GNN层数
        output_dir: 输出目录
        verbose: 是否打印详细信息

    Returns:
        geo_model: 地质模型
        processor: 数据处理器
        trainer: GNN训练器（如果use_gnn=True）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 数据处理
    if verbose:
        print("\n" + "=" * 60)
        print("数据处理")
        print("=" * 60)

    processor = LayerDataProcessor(k_neighbors=10)
    processor.infer_layer_order(df)
    thickness_df = processor.extract_thickness_data(df)

    if verbose:
        print(f"\n提取了 {len(thickness_df)} 个钻孔的厚度数据")

    trainer = None

    if use_gnn:
        # 2. 构建训练数据
        if verbose:
            print("\n" + "=" * 60)
            print("GNN模型训练")
            print("=" * 60)

        data, result = processor.build_graph_data(thickness_df)

        if verbose:
            print(f"\n训练数据:")
            print(f"  节点数: {data.num_nodes}")
            print(f"  边数: {data.num_edges}")
            print(f"  特征维度: {result['num_features']}")
            print(f"  输出层数: {result['num_layers']}")

        # 3. 创建模型
        model = GNNThicknessPredictor(
            in_channels=result['num_features'],
            hidden_channels=hidden_dim,
            num_layers=num_gnn_layers,
            num_output_layers=result['num_layers'],
            dropout=0.3
        )

        # 4. 训练
        trainer = ThicknessTrainer(model, learning_rate=0.001)
        model_path = os.path.join(output_dir, 'thickness_model.pt')

        history = trainer.train(
            data,
            epochs=epochs,
            patience=50,
            verbose=verbose,
            save_path=model_path
        )

        # 5. 评估
        trainer.evaluate(data, processor)

    # 6. 构建地质模型
    geo_model = LayerBasedGeologicalModeling(
        resolution=resolution,
        use_gnn=use_gnn,
        smooth_surfaces=True,
        smooth_sigma=1.0
    )

    geo_model.build_model(
        df,
        processor,
        trainer=trainer,
        thickness_df=thickness_df,
        verbose=verbose
    )

    # 7. 导出
    geo_model.export_vtk(os.path.join(output_dir, 'layer_model.vtk'), processor.layer_order)
    geo_model.export_numpy(os.path.join(output_dir, 'layer_model.npz'))

    stats = geo_model.get_statistics(processor.layer_order)
    stats.to_csv(os.path.join(output_dir, 'layer_model_stats.csv'), index=False, encoding='utf-8-sig')

    if verbose:
        print("\n" + "=" * 60)
        print("输出文件")
        print("=" * 60)
        print(f"  VTK模型: {output_dir}/layer_model.vtk")
        print(f"  NumPy模型: {output_dir}/layer_model.npz")
        print(f"  统计信息: {output_dir}/layer_model_stats.csv")
        if use_gnn:
            print(f"  GNN模型: {output_dir}/thickness_model.pt")

    return geo_model, processor, trainer


if __name__ == "__main__":
    print("层序累加地质建模模块")
    print("请在main.py中运行完整测试")
