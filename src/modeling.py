"""
三维地质建模模块
将稀疏钻孔预测结果插值到规则三维网格，生成完整地质体模型
"""

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from typing import Tuple, Dict, List, Optional
import os
import json


class GeoModel3D:
    """
    三维地质模型
    基于GNN预测结果构建完整的三维地质体
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),  # 网格分辨率
        bounds: Optional[Dict] = None,  # 模型边界
        interpolation_method: str = 'knn',  # 插值方法: 'knn', 'nearest', 'idw'
        k_neighbors: int = 5  # KNN插值的邻居数
    ):
        self.resolution = resolution
        self.bounds = bounds
        self.interpolation_method = interpolation_method
        self.k_neighbors = k_neighbors

        # 模型数据
        self.grid_points = None      # 网格点坐标 [M, 3]
        self.grid_lithology = None   # 网格点岩性 [M]
        self.grid_confidence = None  # 网格点置信度 [M]
        self.grid_shape = None       # 网格形状 (nx, ny, nz)
        self.lithology_classes = []
        self.grid_info = {}

    def build_grid(
        self,
        borehole_coords: np.ndarray,
        padding: float = 0.1  # 边界扩展比例
    ) -> np.ndarray:
        """
        构建三维规则网格

        Args:
            borehole_coords: 钻孔点坐标 [N, 3]
            padding: 边界扩展比例

        Returns:
            grid_points: 网格点坐标 [M, 3]
        """
        # 确定边界
        if self.bounds is None:
            x_min, x_max = borehole_coords[:, 0].min(), borehole_coords[:, 0].max()
            y_min, y_max = borehole_coords[:, 1].min(), borehole_coords[:, 1].max()
            z_min, z_max = borehole_coords[:, 2].min(), borehole_coords[:, 2].max()

            # 添加padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min

            x_min -= x_range * padding
            x_max += x_range * padding
            y_min -= y_range * padding
            y_max += y_range * padding
            # z方向通常不扩展（地表和底部有物理意义）

            self.bounds = {
                'x': (x_min, x_max),
                'y': (y_min, y_max),
                'z': (z_min, z_max)
            }
        else:
            x_min, x_max = self.bounds['x']
            y_min, y_max = self.bounds['y']
            z_min, z_max = self.bounds['z']

        # 创建网格
        nx, ny, nz = self.resolution
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)

        # 生成所有网格点
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        self.grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        self.grid_shape = (nx, ny, nz)

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

        print(f"网格构建完成:")
        print(f"  分辨率: {nx} x {ny} x {nz} = {nx*ny*nz} 个体素")
        print(f"  X范围: {x_min:.1f} ~ {x_max:.1f} m")
        print(f"  Y范围: {y_min:.1f} ~ {y_max:.1f} m")
        print(f"  Z范围: {z_min:.1f} ~ {z_max:.1f} m")

        return self.grid_points

    def interpolate_lithology(
        self,
        borehole_coords: np.ndarray,
        borehole_lithology: np.ndarray,
        borehole_confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将钻孔岩性插值到网格点

        Args:
            borehole_coords: 钻孔点坐标 [N, 3]
            borehole_lithology: 钻孔点岩性标签 [N]
            borehole_confidence: 钻孔点置信度 [N] (可选)

        Returns:
            grid_lithology: 网格点岩性 [M]
            grid_confidence: 网格点置信度 [M]
        """
        if self.grid_points is None:
            self.build_grid(borehole_coords)

        print(f"插值方法: {self.interpolation_method}")
        print(f"钻孔点数: {len(borehole_coords)}, 网格点数: {len(self.grid_points)}")

        if self.interpolation_method == 'knn':
            # KNN加权插值
            self.grid_lithology, self.grid_confidence = self._knn_interpolate(
                borehole_coords, borehole_lithology, borehole_confidence
            )

        elif self.interpolation_method == 'nearest':
            # 最近邻插值
            interpolator = NearestNDInterpolator(borehole_coords, borehole_lithology)
            self.grid_lithology = interpolator(self.grid_points).astype(int)

            if borehole_confidence is not None:
                conf_interpolator = NearestNDInterpolator(borehole_coords, borehole_confidence)
                self.grid_confidence = conf_interpolator(self.grid_points)
            else:
                # 基于距离计算置信度
                tree = KDTree(borehole_coords)
                distances, _ = tree.query(self.grid_points, k=1)
                max_dist = distances.max()
                self.grid_confidence = 1.0 - (distances / max_dist)

        elif self.interpolation_method == 'idw':
            # 反距离加权插值
            self.grid_lithology, self.grid_confidence = self._idw_interpolate(
                borehole_coords, borehole_lithology, borehole_confidence
            )

        else:
            raise ValueError(f"未知插值方法: {self.interpolation_method}")

        return self.grid_lithology, self.grid_confidence

    def _knn_interpolate(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        power: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KNN加权投票插值

        对于每个网格点，找K个最近的钻孔点，
        根据距离加权投票决定岩性
        """
        tree = KDTree(coords)
        distances, indices = tree.query(self.grid_points, k=self.k_neighbors)

        num_classes = int(labels.max()) + 1
        grid_labels = np.zeros(len(self.grid_points), dtype=int)
        grid_conf = np.zeros(len(self.grid_points))

        for i in range(len(self.grid_points)):
            # 距离权重 (反距离)
            dists = distances[i]
            dists = np.maximum(dists, 1e-10)  # 避免除零
            weights = 1.0 / (dists ** power)
            weights /= weights.sum()

            # 加权投票
            neighbor_labels = labels[indices[i]]
            votes = np.zeros(num_classes)
            for j, lbl in enumerate(neighbor_labels):
                votes[lbl] += weights[j]

            grid_labels[i] = np.argmax(votes)
            grid_conf[i] = votes[grid_labels[i]]  # 置信度 = 最高票数

        return grid_labels, grid_conf

    def _idw_interpolate(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        power: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        反距离加权插值 (IDW)
        """
        tree = KDTree(coords)

        # 使用更多邻居进行IDW
        k = min(15, len(coords))
        distances, indices = tree.query(self.grid_points, k=k)

        num_classes = int(labels.max()) + 1
        grid_labels = np.zeros(len(self.grid_points), dtype=int)
        grid_conf = np.zeros(len(self.grid_points))

        for i in range(len(self.grid_points)):
            dists = distances[i]
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / (dists ** power)
            weights /= weights.sum()

            neighbor_labels = labels[indices[i]]
            votes = np.zeros(num_classes)
            for j, lbl in enumerate(neighbor_labels):
                votes[lbl] += weights[j]

            grid_labels[i] = np.argmax(votes)
            grid_conf[i] = votes[grid_labels[i]]

        return grid_labels, grid_conf

    def get_voxel_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取体素模型 (3D数组形式)

        Returns:
            lithology_3d: 岩性3D数组 [nx, ny, nz]
            confidence_3d: 置信度3D数组 [nx, ny, nz]
        """
        if self.grid_lithology is None:
            raise ValueError("请先运行interpolate_lithology()")

        nx, ny, nz = self.grid_shape
        lithology_3d = self.grid_lithology.reshape(nx, ny, nz)
        confidence_3d = self.grid_confidence.reshape(nx, ny, nz)

        return lithology_3d, confidence_3d

    def get_slice(
        self,
        axis: str = 'z',
        index: Optional[int] = None,
        position: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        获取切片

        Args:
            axis: 切片方向 ('x', 'y', 'z')
            index: 切片索引 (优先使用)
            position: 切片位置坐标

        Returns:
            slice_data: 切片岩性数据
            slice_coords: 切片坐标网格
            slice_info: 切片信息
        """
        lithology_3d, _ = self.get_voxel_model()
        nx, ny, nz = self.grid_shape

        if axis == 'x':
            if index is None:
                if position is not None:
                    x_grid = self.grid_info['x_grid']
                    index = np.argmin(np.abs(x_grid - position))
                else:
                    index = nx // 2

            slice_data = lithology_3d[index, :, :]
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
            x_coords, z_coords = np.meshgrid(
                self.grid_info['x_grid'],
                self.grid_info['z_grid'],
                indexing='ij'
            )
            slice_coords = {'x': x_coords, 'z': z_coords}
            slice_info = {'axis': 'y', 'index': index, 'position': self.grid_info['y_grid'][index]}

        elif axis == 'z':
            if index is None:
                if position is not None:
                    z_grid = self.grid_info['z_grid']
                    index = np.argmin(np.abs(z_grid - position))
                else:
                    index = nz // 2

            slice_data = lithology_3d[:, :, index]
            x_coords, y_coords = np.meshgrid(
                self.grid_info['x_grid'],
                self.grid_info['y_grid'],
                indexing='ij'
            )
            slice_coords = {'x': x_coords, 'y': y_coords}
            slice_info = {'axis': 'z', 'index': index, 'position': self.grid_info['z_grid'][index]}

        else:
            raise ValueError(f"无效的axis: {axis}")

        return slice_data, slice_coords, slice_info

    def export_vtk(self, filepath: str, lithology_names: Optional[List[str]] = None):
        """
        导出为VTK格式 (可用于ParaView等软件)

        Args:
            filepath: 输出文件路径 (.vtk)
            lithology_names: 岩性名称列表
        """
        if self.grid_lithology is None:
            raise ValueError("请先运行interpolate_lithology()")

        nx, ny, nz = self.grid_shape
        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']

        lithology_3d, confidence_3d = self.get_voxel_model()

        with open(filepath, 'w') as f:
            # VTK头部
            f.write("# vtk DataFile Version 3.0\n")
            f.write("3D Geological Model\n")
            f.write("ASCII\n")
            f.write("DATASET RECTILINEAR_GRID\n")

            # 网格维度
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")

            # X坐标
            f.write(f"X_COORDINATES {nx} float\n")
            f.write(" ".join(f"{x:.6f}" for x in x_grid) + "\n")

            # Y坐标
            f.write(f"Y_COORDINATES {ny} float\n")
            f.write(" ".join(f"{y:.6f}" for y in y_grid) + "\n")

            # Z坐标
            f.write(f"Z_COORDINATES {nz} float\n")
            f.write(" ".join(f"{z:.6f}" for z in z_grid) + "\n")

            # 数据
            f.write(f"POINT_DATA {nx * ny * nz}\n")

            # 岩性标签
            f.write("SCALARS lithology int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{lithology_3d[i, j, k]}\n")

            # 置信度
            f.write("SCALARS confidence float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{confidence_3d[i, j, k]:.6f}\n")

        print(f"VTK文件已导出: {filepath}")

        # 导出岩性名称映射
        if lithology_names:
            json_path = filepath.replace('.vtk', '_lithology_names.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({i: name for i, name in enumerate(lithology_names)}, f, ensure_ascii=False, indent=2)
            print(f"岩性名称映射已导出: {json_path}")

    def export_numpy(self, filepath: str):
        """
        导出为NumPy格式 (.npz)

        Args:
            filepath: 输出文件路径 (.npz)
        """
        if self.grid_lithology is None:
            raise ValueError("请先运行interpolate_lithology()")

        lithology_3d, confidence_3d = self.get_voxel_model()

        np.savez_compressed(
            filepath,
            lithology=lithology_3d,
            confidence=confidence_3d,
            x_grid=self.grid_info['x_grid'],
            y_grid=self.grid_info['y_grid'],
            z_grid=self.grid_info['z_grid'],
            bounds=np.array([
                self.bounds['x'],
                self.bounds['y'],
                self.bounds['z']
            ])
        )
        print(f"NumPy文件已导出: {filepath}")

    def export_csv(self, filepath: str, lithology_names: Optional[List[str]] = None):
        """
        导出为CSV格式

        Args:
            filepath: 输出文件路径 (.csv)
            lithology_names: 岩性名称列表
        """
        if self.grid_lithology is None:
            raise ValueError("请先运行interpolate_lithology()")

        df = pd.DataFrame({
            'x': self.grid_points[:, 0],
            'y': self.grid_points[:, 1],
            'z': self.grid_points[:, 2],
            'lithology_code': self.grid_lithology,
            'confidence': self.grid_confidence
        })

        if lithology_names:
            df['lithology'] = df['lithology_code'].map(
                {i: name for i, name in enumerate(lithology_names)}
            )

        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"CSV文件已导出: {filepath}")

    def get_statistics(self, lithology_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取模型统计信息

        Returns:
            stats_df: 统计信息DataFrame
        """
        if self.grid_lithology is None:
            raise ValueError("请先运行interpolate_lithology()")

        lithology_3d, confidence_3d = self.get_voxel_model()
        cell_volume = (
            self.grid_info['cell_size']['x'] *
            self.grid_info['cell_size']['y'] *
            self.grid_info['cell_size']['z']
        )

        stats = []
        for i in range(int(self.grid_lithology.max()) + 1):
            mask = self.grid_lithology == i
            count = mask.sum()
            volume = count * cell_volume

            name = lithology_names[i] if lithology_names and i < len(lithology_names) else f"类别{i}"

            stats.append({
                '岩性': name,
                '体素数': count,
                '体积 (m³)': volume,
                '占比 (%)': 100.0 * count / len(self.grid_lithology),
                '平均置信度': self.grid_confidence[mask].mean() if count > 0 else 0
            })

        return pd.DataFrame(stats)


def build_geological_model(
    trainer,
    data,
    result: Dict,
    resolution: Tuple[int, int, int] = (50, 50, 50),
    interpolation_method: str = 'knn',
    output_dir: str = 'output'
) -> GeoModel3D:
    """
    构建完整的三维地质模型

    Args:
        trainer: 训练好的GeoModelTrainer
        data: PyG Data对象
        result: 数据处理结果字典
        resolution: 网格分辨率 (nx, ny, nz)
        interpolation_method: 插值方法
        output_dir: 输出目录

    Returns:
        model: GeoModel3D对象
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("构建三维地质模型")
    print("=" * 60)

    # 1. 获取钻孔点预测结果
    print("\n[1/4] 获取钻孔点预测...")
    predictions, probabilities = trainer.predict(data, return_probs=True)
    coords = data.coords.numpy()
    confidence = probabilities.max(axis=1)

    print(f"  钻孔采样点数: {len(coords)}")
    print(f"  预测类别数: {probabilities.shape[1]}")

    # 2. 创建3D模型
    print("\n[2/4] 创建三维网格...")
    model = GeoModel3D(
        resolution=resolution,
        interpolation_method=interpolation_method,
        k_neighbors=8
    )
    model.build_grid(coords)
    model.lithology_classes = result['lithology_classes']

    # 3. 插值到网格
    print("\n[3/4] 插值岩性到网格...")
    model.interpolate_lithology(coords, predictions, confidence)

    # 4. 统计和导出
    print("\n[4/4] 生成统计和导出...")
    stats = model.get_statistics(result['lithology_classes'])
    print("\n岩性体积统计:")
    print(stats.to_string(index=False))

    # 导出
    model.export_vtk(os.path.join(output_dir, 'geological_model.vtk'), result['lithology_classes'])
    model.export_numpy(os.path.join(output_dir, 'geological_model.npz'))
    model.export_csv(os.path.join(output_dir, 'geological_model.csv'), result['lithology_classes'])

    # 保存统计信息
    stats.to_csv(os.path.join(output_dir, 'model_statistics.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("三维地质模型构建完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    return model


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("测试三维地质建模模块...")

    # 模拟数据
    np.random.seed(42)
    n_points = 500

    # 模拟钻孔点
    coords = np.column_stack([
        np.random.uniform(0, 1000, n_points),
        np.random.uniform(0, 1000, n_points),
        np.random.uniform(-500, 0, n_points)
    ])

    # 模拟岩性 (基于深度的简单规则)
    lithology = np.zeros(n_points, dtype=int)
    lithology[coords[:, 2] > -100] = 0  # 表层
    lithology[(coords[:, 2] <= -100) & (coords[:, 2] > -250)] = 1  # 中层
    lithology[(coords[:, 2] <= -250) & (coords[:, 2] > -400)] = 2  # 深层
    lithology[coords[:, 2] <= -400] = 3  # 底层

    confidence = np.random.uniform(0.7, 1.0, n_points)

    # 构建模型
    model = GeoModel3D(
        resolution=(30, 30, 30),
        interpolation_method='knn'
    )

    model.build_grid(coords)
    model.interpolate_lithology(coords, lithology, confidence)

    # 获取统计
    stats = model.get_statistics(['表层土', '砂岩层', '泥岩层', '基岩'])
    print("\n统计信息:")
    print(stats)

    # 获取切片
    slice_data, slice_coords, slice_info = model.get_slice('z', position=-200)
    print(f"\n水平切片 (z=-200m): {slice_data.shape}")

    print("\n测试完成!")
