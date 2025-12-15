"""
GNN建模模块
将GNN预测结果与三维地质建模相结合

包含三种整合方案:
1. DirectPredictionModeling: 直接预测法
2. HybridFusionModeling: 混合融合法 (TODO)
3. TwoStageOptimizationModeling: 两阶段优化法 (TODO)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import os
import json


class BaseGNNModeling(ABC):
    """
    GNN地质建模基类
    定义统一的接口
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        bounds: Optional[Dict] = None
    ):
        self.resolution = resolution
        self.bounds = bounds

        # 模型结果
        self.grid_lithology = None
        self.grid_confidence = None
        self.grid_probabilities = None
        self.grid_info = None
        self.lithology_classes = []

    @abstractmethod
    def build_model(
        self,
        trainer,
        data: Data,
        result: Dict,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建三维地质模型

        Args:
            trainer: GeoModelTrainer实例 (已训练好)
            data: PyG Data对象 (训练数据)
            result: 数据处理结果字典 (包含raw_df, lithology_classes等)

        Returns:
            lithology_3d: 岩性三维数组 [nx, ny, nz]
            confidence_3d: 置信度三维数组 [nx, ny, nz]
        """
        pass

    def get_voxel_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取体素模型"""
        if self.grid_lithology is None:
            raise ValueError("请先调用build_model构建模型")
        nx, ny, nz = self.resolution
        lithology_3d = self.grid_lithology.reshape(nx, ny, nz)
        confidence_3d = self.grid_confidence.reshape(nx, ny, nz)
        return lithology_3d, confidence_3d

    def get_probabilities(self) -> Optional[np.ndarray]:
        """获取预测概率"""
        if self.grid_probabilities is None:
            return None
        nx, ny, nz = self.resolution
        num_classes = self.grid_probabilities.shape[-1]
        return self.grid_probabilities.reshape(nx, ny, nz, num_classes)

    def get_slice(
        self,
        axis: str = 'z',
        index: Optional[int] = None,
        position: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """获取切片"""
        lithology_3d, confidence_3d = self.get_voxel_model()
        nx, ny, nz = self.resolution

        if axis == 'x':
            if index is None:
                if position is not None:
                    x_grid = self.grid_info['x_grid']
                    index = np.argmin(np.abs(x_grid - position))
                else:
                    index = nx // 2
            slice_data = lithology_3d[index, :, :]
            slice_confidence = confidence_3d[index, :, :]
            y_coords, z_coords = np.meshgrid(
                self.grid_info['y_grid'],
                self.grid_info['z_grid'],
                indexing='ij'
            )
            slice_coords = {'y': y_coords, 'z': z_coords}
            slice_info = {'axis': 'x', 'index': index, 'position': self.grid_info['x_grid'][index]}

        elif axis == 'y':
            if index is None:
                if position is not None:
                    y_grid = self.grid_info['y_grid']
                    index = np.argmin(np.abs(y_grid - position))
                else:
                    index = ny // 2
            slice_data = lithology_3d[:, index, :]
            slice_confidence = confidence_3d[:, index, :]
            x_coords, z_coords = np.meshgrid(
                self.grid_info['x_grid'],
                self.grid_info['z_grid'],
                indexing='ij'
            )
            slice_coords = {'x': x_coords, 'z': z_coords}
            slice_info = {'axis': 'y', 'index': index, 'position': self.grid_info['y_grid'][index]}

        else:  # axis == 'z'
            if index is None:
                if position is not None:
                    z_grid = self.grid_info['z_grid']
                    index = np.argmin(np.abs(z_grid - position))
                else:
                    index = nz // 2
            slice_data = lithology_3d[:, :, index]
            slice_confidence = confidence_3d[:, :, index]
            x_coords, y_coords = np.meshgrid(
                self.grid_info['x_grid'],
                self.grid_info['y_grid'],
                indexing='ij'
            )
            slice_coords = {'x': x_coords, 'y': y_coords}
            slice_info = {'axis': 'z', 'index': index, 'position': self.grid_info['z_grid'][index]}

        return slice_data, slice_coords, slice_info

    def get_statistics(self, lithology_names: Optional[List[str]] = None) -> pd.DataFrame:
        """获取统计信息"""
        if self.grid_lithology is None:
            raise ValueError("请先构建模型")

        lithology_names = lithology_names or self.lithology_classes
        nx, ny, nz = self.resolution

        # 计算单元格体积 - 防止除零
        x_divisor = max(nx - 1, 1)
        y_divisor = max(ny - 1, 1)
        z_divisor = max(nz - 1, 1)

        cell_volume = (
            (self.bounds['x'][1] - self.bounds['x'][0]) / x_divisor *
            (self.bounds['y'][1] - self.bounds['y'][0]) / y_divisor *
            (self.bounds['z'][1] - self.bounds['z'][0]) / z_divisor
        )

        stats = []
        total_count = len(self.grid_lithology)
        if total_count == 0:
            return pd.DataFrame(stats)

        for i in range(int(self.grid_lithology.max()) + 1):
            mask = self.grid_lithology == i
            count = mask.sum()
            volume = count * cell_volume

            name = lithology_names[i] if lithology_names and i < len(lithology_names) else f"类别{i}"

            stats.append({
                '岩性': name,
                '体素数': count,
                '体积 (m³)': volume,
                '占比 (%)': 100.0 * count / total_count,
                '平均置信度': self.grid_confidence[mask].mean() if count > 0 else 0
            })

        return pd.DataFrame(stats)

    def export_vtk(self, filepath: str, lithology_names: Optional[List[str]] = None):
        """导出VTK格式"""
        if self.grid_lithology is None:
            raise ValueError("请先构建模型")

        nx, ny, nz = self.resolution
        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']

        lithology_3d, confidence_3d = self.get_voxel_model()

        with open(filepath, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("3D GNN Geological Model\n")
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

            # 岩性数据
            f.write("SCALARS lithology int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{lithology_3d[i, j, k]}\n")

            # 置信度数据
            f.write("SCALARS confidence float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{confidence_3d[i, j, k]:.4f}\n")

        print(f"VTK文件已导出: {filepath}")

    def export_numpy(self, filepath: str):
        """导出NumPy格式"""
        lithology_3d, confidence_3d = self.get_voxel_model()
        probs_3d = self.get_probabilities()

        save_dict = {
            'lithology': lithology_3d,
            'confidence': confidence_3d,
            'x_grid': self.grid_info['x_grid'],
            'y_grid': self.grid_info['y_grid'],
            'z_grid': self.grid_info['z_grid'],
            'lithology_classes': self.lithology_classes,
            'method': self.__class__.__name__
        }

        if probs_3d is not None:
            save_dict['probabilities'] = probs_3d

        np.savez_compressed(filepath, **save_dict)
        print(f"NumPy文件已导出: {filepath}")


class DirectPredictionModeling(BaseGNNModeling):
    """
    直接预测法 (Direct Prediction)

    核心思想:
    - 将三维网格中的每个点作为新节点
    - 使用训练好的GNN模型直接预测每个网格点的岩性
    - 网格点通过KNN连接到最近的钻孔节点

    优点:
    - 充分利用GNN学习到的空间模式
    - 实现相对简单
    - 可以给出预测置信度

    缺点:
    - 可能产生地质上不连续的结果
    - 远离钻孔的区域预测可能不准
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        k_neighbors: int = 8,  # 网格点连接到多少个最近钻孔节点
        bounds: Optional[Dict] = None,
        use_edge_weight: bool = True,  # 是否使用基于距离的边权重
        smooth_output: bool = True,  # 是否平滑输出
        smooth_sigma: float = 0.5  # 高斯平滑参数
    ):
        super().__init__(resolution, bounds)
        self.k_neighbors = k_neighbors
        self.use_edge_weight = use_edge_weight
        self.smooth_output = smooth_output
        self.smooth_sigma = smooth_sigma

    def _build_grid(self, df: pd.DataFrame, padding: float = 0.05):
        """根据钻孔数据范围构建三维网格"""
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()

        # 获取z范围 (注意z可能是负值表示深度)
        if 'z' in df.columns:
            z_min, z_max = df['z'].min(), df['z'].max()
        else:
            z_min = -df['bottom_depth'].max()
            z_max = -df['top_depth'].min()

        # 添加padding
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
            'resolution': self.resolution,
            'bounds': self.bounds,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'z_grid': z_grid,
            'cell_size': {
                'x': (x_max - x_min) / (nx - 1),
                'y': (y_max - y_min) / (ny - 1),
                'z': (z_max - z_min) / (nz - 1)
            },
            'total_cells': nx * ny * nz
        }

        # 生成所有网格点坐标
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        grid_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        print(f"网格构建完成:")
        print(f"  分辨率: {nx} x {ny} x {nz} = {nx*ny*nz} 个体素")
        print(f"  X范围: {x_min:.1f} ~ {x_max:.1f} m")
        print(f"  Y范围: {y_min:.1f} ~ {y_max:.1f} m")
        print(f"  Z范围: {z_min:.1f} ~ {z_max:.1f} m")

        return grid_coords

    def _generate_grid_features(
        self,
        grid_coords: np.ndarray,
        borehole_coords: np.ndarray,
        borehole_features: np.ndarray,
        coord_scaler,
        feature_scaler
    ) -> np.ndarray:
        """
        为网格点生成特征向量

        特征包括:
        1. 归一化的空间坐标 (x, y, z)
        2. 深度相关特征
        3. 基于最近钻孔的插值特征
        """
        n_grid = len(grid_coords)
        n_features = borehole_features.shape[1]

        # 输入校验
        if n_grid == 0:
            raise ValueError("网格坐标为空")
        if len(borehole_coords) == 0:
            raise ValueError("钻孔坐标为空")

        # 1. 归一化坐标
        if coord_scaler is not None and hasattr(coord_scaler, 'mean_'):
            coords_normalized = coord_scaler.transform(grid_coords)
        else:
            # 手动归一化 - 使用钻孔坐标的统计量以保持一致性
            coord_mean = borehole_coords.mean(axis=0)
            coord_std = borehole_coords.std(axis=0) + 1e-8
            coords_normalized = (grid_coords - coord_mean) / coord_std

        # 2. 使用KNN插值钻孔特征到网格点
        k = min(self.k_neighbors, len(borehole_coords))
        if k == 0:
            raise ValueError("没有足够的钻孔节点进行KNN插值")

        tree = KDTree(borehole_coords)
        distances, indices = tree.query(grid_coords, k=k)

        # 确保 distances 是2D的 (当k=1时可能是1D)
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # IDW插值权重
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # 插值特征 (除去坐标部分)
        if borehole_features.shape[1] > 3:
            other_features = borehole_features[:, 3:]  # 除去xyz坐标
            interpolated_features = np.zeros((n_grid, other_features.shape[1]))
            for i in range(n_grid):
                interpolated_features[i] = np.sum(
                    weights[i, :, np.newaxis] * other_features[indices[i]], axis=0
                )
            # 3. 合并特征
            grid_features = np.concatenate([coords_normalized, interpolated_features], axis=1)
        else:
            # 特征维度<=3时，只使用坐标作为特征
            grid_features = coords_normalized

        return grid_features.astype(np.float32)

    def _build_extended_graph(
        self,
        grid_coords: np.ndarray,
        grid_features: np.ndarray,
        borehole_coords: np.ndarray,
        original_edge_index: torch.Tensor,
        original_edge_weight: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        """
        构建扩展图，将网格点连接到钻孔节点

        返回:
            extended_x: 扩展后的节点特征 [n_borehole + n_grid, F]
            extended_edge_index: 扩展后的边索引
            extended_edge_weight: 扩展后的边权重
            n_borehole: 钻孔节点数量 (用于后续分离)
        """
        n_borehole = len(borehole_coords)
        n_grid = len(grid_coords)

        # 使用KNN找到每个网格点最近的k个钻孔节点
        tree = KDTree(borehole_coords)
        distances, indices = tree.query(grid_coords, k=min(self.k_neighbors, n_borehole))

        # 构建新边: 网格点 -> 钻孔节点
        new_edges = []
        new_weights = []

        for i in range(n_grid):
            grid_node_idx = n_borehole + i  # 网格点的索引从n_borehole开始

            for j, (dist, bh_idx) in enumerate(zip(distances[i], indices[i])):
                # 双向边
                new_edges.append([grid_node_idx, bh_idx])
                new_edges.append([bh_idx, grid_node_idx])

                # 基于距离的权重
                weight = np.exp(-dist / (distances[i].mean() + 1e-8))
                new_weights.extend([weight, weight])

        # 合并边
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        extended_edge_index = torch.cat([original_edge_index, new_edge_index], dim=1)

        # 合并边权重
        if self.use_edge_weight and original_edge_weight is not None:
            new_edge_weight = torch.tensor(new_weights, dtype=torch.float32)
            extended_edge_weight = torch.cat([original_edge_weight, new_edge_weight])
        else:
            extended_edge_weight = None

        # 合并特征
        extended_x = torch.tensor(grid_features, dtype=torch.float32)

        return extended_x, extended_edge_index, extended_edge_weight, n_borehole

    def build_model(
        self,
        trainer,
        data: Data,
        result: Dict,
        batch_size: int = 10000,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用GNN直接预测构建三维地质模型

        Args:
            trainer: 训练好的GeoModelTrainer
            data: PyG Data对象
            result: 数据处理结果字典
            batch_size: 批处理大小 (用于大规模网格)
            verbose: 是否打印详细信息

        Returns:
            lithology_3d: 岩性预测结果 [nx, ny, nz]
            confidence_3d: 置信度 [nx, ny, nz]
        """
        if verbose:
            print("\n" + "=" * 60)
            print("构建GNN直接预测模型")
            print("=" * 60)

        self.lithology_classes = result['lithology_classes']
        df = result['raw_df']

        # 1. 构建网格
        if verbose:
            print("\n[1/4] 构建三维网格...")
        grid_coords = self._build_grid(df)
        n_grid = len(grid_coords)

        # 2. 获取钻孔节点信息
        if verbose:
            print("\n[2/4] 准备节点特征...")
        borehole_coords = data.coords.cpu().numpy()
        borehole_features = data.x.cpu().numpy()

        # 从result获取scaler (如果有)
        coord_scaler = None
        feature_scaler = None
        # 这里暂时不使用scaler，因为特征已经在data中归一化了

        # 生成网格点特征
        grid_features = self._generate_grid_features(
            grid_coords,
            borehole_coords,
            borehole_features,
            coord_scaler,
            feature_scaler
        )

        if verbose:
            print(f"  钻孔节点数: {len(borehole_coords)}")
            print(f"  网格点数: {n_grid}")
            print(f"  特征维度: {grid_features.shape[1]}")

        # 3. 构建扩展图
        if verbose:
            print("\n[3/4] 构建扩展图...")

        # 将edge_index移到CPU进行图构建
        edge_index_cpu = data.edge_index.cpu()
        edge_weight_cpu = data.edge_weight.cpu() if hasattr(data, 'edge_weight') and data.edge_weight is not None else None

        extended_x, extended_edge_index, extended_edge_weight, n_borehole = self._build_extended_graph(
            grid_coords,
            grid_features,
            borehole_coords,
            edge_index_cpu,
            edge_weight_cpu
        )

        # 合并钻孔特征和网格特征
        full_x = torch.cat([data.x.cpu(), extended_x], dim=0)

        if verbose:
            print(f"  扩展后节点数: {full_x.shape[0]}")
            print(f"  扩展后边数: {extended_edge_index.shape[1]}")

        # 4. 使用GNN预测
        if verbose:
            print("\n[4/4] GNN预测...")

        device = trainer.device
        model = trainer.model
        model.eval()

        # 将数据移到设备上
        full_x = full_x.to(device)
        extended_edge_index = extended_edge_index.to(device)
        if extended_edge_weight is not None:
            extended_edge_weight = extended_edge_weight.to(device)

        # 使用EMA参数进行预测
        if trainer.ema is not None:
            trainer.ema.apply_shadow()

        try:
            with torch.no_grad():
                # 前向传播
                if extended_edge_weight is not None:
                    out = model(full_x, extended_edge_index, extended_edge_weight)
                else:
                    out = model(full_x, extended_edge_index)

                # 提取网格点的预测结果
                grid_out = out[n_borehole:]

                # 计算概率和预测类别
                probs = F.softmax(grid_out, dim=1)
                predictions = grid_out.argmax(dim=1)

                # 置信度 = 最高概率
                confidence = probs.max(dim=1).values

                # 转换为numpy
                predictions_np = predictions.cpu().numpy()
                confidence_np = confidence.cpu().numpy()
                probs_np = probs.cpu().numpy()

        finally:
            # 恢复原始参数
            if trainer.ema is not None:
                trainer.ema.restore()

        # 5. 后处理 (可选平滑)
        nx, ny, nz = self.resolution

        if self.smooth_output:
            if verbose:
                print("\n应用平滑处理...")

            # 对置信度进行高斯平滑
            confidence_3d = confidence_np.reshape(nx, ny, nz)
            confidence_3d = gaussian_filter(confidence_3d, sigma=self.smooth_sigma)
            confidence_np = confidence_3d.ravel()

            # 对概率进行平滑后重新决策
            probs_3d = probs_np.reshape(nx, ny, nz, -1)
            for c in range(probs_3d.shape[-1]):
                probs_3d[:, :, :, c] = gaussian_filter(probs_3d[:, :, :, c], sigma=self.smooth_sigma)

            # 重新归一化并决策
            probs_3d = probs_3d / (probs_3d.sum(axis=-1, keepdims=True) + 1e-8)
            predictions_np = probs_3d.argmax(axis=-1).ravel()
            probs_np = probs_3d.reshape(-1, probs_3d.shape[-1])

        # 保存结果
        self.grid_lithology = predictions_np
        self.grid_confidence = confidence_np
        self.grid_probabilities = probs_np

        # 统计信息
        if verbose:
            print(f"\n模型构建完成!")
            print(f"  总体素数: {n_grid}")
            unique, counts = np.unique(predictions_np, return_counts=True)
            for u, c in zip(unique, counts):
                name = self.lithology_classes[u] if u < len(self.lithology_classes) else f"类别{u}"
                print(f"  {name}: {c} ({100*c/n_grid:.1f}%)")
            print(f"  平均置信度: {confidence_np.mean():.4f}")

        lithology_3d = predictions_np.reshape(nx, ny, nz)
        confidence_3d = confidence_np.reshape(nx, ny, nz)

        return lithology_3d, confidence_3d


class HybridFusionModeling(BaseGNNModeling):
    """
    混合融合法 (Hybrid Fusion)

    将GNN预测结果与传统插值方法融合

    TODO: 待实现
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        fusion_strategy: str = 'distance',  # 'fixed', 'distance', 'confidence'
        alpha: float = 0.5,  # 固定权重时使用
        distance_scale: float = 100,  # 距离权重时使用
        bounds: Optional[Dict] = None
    ):
        super().__init__(resolution, bounds)
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.distance_scale = distance_scale

    def build_model(
        self,
        trainer,
        data: Data,
        result: Dict,
        traditional_model=None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """混合融合法建模 - 待实现"""
        raise NotImplementedError("HybridFusionModeling尚未实现，请使用DirectPredictionModeling")


class TwoStageOptimizationModeling(BaseGNNModeling):
    """
    两阶段优化法 (Two-Stage Optimization)

    先用传统方法建立初始模型，再用GNN优化边界和不确定区域

    TODO: 待实现
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        confidence_threshold: float = 0.7,
        boundary_width: int = 2,
        bounds: Optional[Dict] = None
    ):
        super().__init__(resolution, bounds)
        self.confidence_threshold = confidence_threshold
        self.boundary_width = boundary_width

    def build_model(
        self,
        trainer,
        data: Data,
        result: Dict,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """两阶段优化法建模 - 待实现"""
        raise NotImplementedError("TwoStageOptimizationModeling尚未实现，请使用DirectPredictionModeling")


# ==================== 便捷函数 ====================

def build_gnn_geological_model(
    trainer,
    data: Data,
    result: Dict,
    method: str = 'direct',
    resolution: Tuple[int, int, int] = (50, 50, 50),
    output_dir: str = 'output',
    **kwargs
) -> BaseGNNModeling:
    """
    构建GNN地质模型的便捷函数

    Args:
        trainer: 训练好的GeoModelTrainer
        data: PyG Data对象
        result: 数据处理结果字典
        method: 建模方法 ('direct', 'hybrid', 'two_stage')
        resolution: 网格分辨率
        output_dir: 输出目录
        **kwargs: 其他参数

    Returns:
        model: 构建好的地质模型
    """
    os.makedirs(output_dir, exist_ok=True)

    # 选择方法
    if method == 'direct':
        model = DirectPredictionModeling(
            resolution=resolution,
            k_neighbors=kwargs.get('k_neighbors', 8),
            use_edge_weight=kwargs.get('use_edge_weight', True),
            smooth_output=kwargs.get('smooth_output', True),
            smooth_sigma=kwargs.get('smooth_sigma', 0.5)
        )
    elif method == 'hybrid':
        model = HybridFusionModeling(
            resolution=resolution,
            fusion_strategy=kwargs.get('fusion_strategy', 'distance'),
            alpha=kwargs.get('alpha', 0.5)
        )
    elif method == 'two_stage':
        model = TwoStageOptimizationModeling(
            resolution=resolution,
            confidence_threshold=kwargs.get('confidence_threshold', 0.7)
        )
    else:
        raise ValueError(f"未知方法: {method}")

    # 构建模型
    model.build_model(trainer, data, result, **kwargs)

    # 导出结果
    model.export_vtk(os.path.join(output_dir, f'gnn_model_{method}.vtk'))
    model.export_numpy(os.path.join(output_dir, f'gnn_model_{method}.npz'))

    # 统计信息
    stats = model.get_statistics(result['lithology_classes'])
    print("\n岩性体积统计:")
    print(stats.to_string(index=False))
    stats.to_csv(os.path.join(output_dir, f'gnn_model_{method}_stats.csv'),
                 index=False, encoding='utf-8-sig')

    return model


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("GNN建模模块测试")
    print("请在main.py中运行完整测试")
