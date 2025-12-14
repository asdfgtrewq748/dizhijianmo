"""
三维地质建模模块
基于层面插值方法构建真实的层状地质模型
"""

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, List, Optional
import os
import json
import warnings

# 尝试导入pykrige
try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False
    print("警告: 未安装pykrige，克里金插值将不可用")


class StratigraphicModel3D:
    """
    层状三维地质模型
    基于层面插值方法，正确模拟真实的层状地质结构

    原理：
    1. 从钻孔数据中提取每个地层的顶底界面深度
    2. 对每个地层界面进行空间插值（RBF/IDW）
    3. 根据界面位置判断三维网格中每个点属于哪一层
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (50, 50, 50),
        bounds: Optional[Dict] = None,
        interpolation_method: str = 'rbf',  # 'rbf', 'idw', 'linear'
        smoothing: float = 0.1  # RBF平滑参数
    ):
        self.resolution = resolution
        self.bounds = bounds
        self.interpolation_method = interpolation_method
        self.smoothing = smoothing

        # 模型数据
        self.grid_points = None
        self.grid_lithology = None
        self.grid_confidence = None
        self.grid_shape = None
        self.lithology_classes = []
        self.layer_interfaces = {}  # 存储每层的顶底界面插值结果
        self.grid_info = {}

    def extract_layer_interfaces(self, df: pd.DataFrame) -> Dict:
        """
        从钻孔数据中提取地层界面信息

        Args:
            df: 包含钻孔数据的DataFrame

        Returns:
            layer_data: 字典，包含每层的界面数据
        """
        layer_data = {}

        for bh_id in df['borehole_id'].unique():
            bh_data = df[df['borehole_id'] == bh_id]
            x = bh_data['x'].iloc[0]
            y = bh_data['y'].iloc[0]

            # 获取该钻孔的所有地层（按层序排列）
            if 'layer_order' in bh_data.columns:
                layers = bh_data.groupby('layer_order').agg({
                    'lithology': 'first',
                    'top_depth': 'first',
                    'bottom_depth': 'first'
                }).reset_index().sort_values('layer_order')
            else:
                continue

            # 记录每层的顶底深度
            for _, layer in layers.iterrows():
                layer_order = int(layer['layer_order'])
                lithology = layer['lithology']
                top_depth = layer['top_depth']
                bottom_depth = layer['bottom_depth']

                if layer_order not in layer_data:
                    layer_data[layer_order] = {
                        'lithology': lithology,
                        'points': [],
                    }

                layer_data[layer_order]['points'].append({
                    'x': x,
                    'y': y,
                    'top_depth': top_depth,
                    'bottom_depth': bottom_depth,
                    'borehole_id': bh_id
                })

        return layer_data

    def build_grid(self, df: pd.DataFrame, padding: float = 0.05):
        """根据钻孔数据范围构建三维网格"""
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        z_min = -df['bottom_depth'].max()
        z_max = -df['top_depth'].min()

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

    def interpolate_interface(self, points: List[Dict], grid_x: np.ndarray,
                              grid_y: np.ndarray, depth_key: str) -> np.ndarray:
        """对单个地层界面进行空间插值"""
        xy = np.array([[p['x'], p['y']] for p in points])
        z = np.array([p[depth_key] for p in points])

        # 移除重复点 (坐标相同但深度不同的点取平均)
        df_points = pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1], 'z': z})
        df_unique = df_points.groupby(['x', 'y']).mean().reset_index()
        xy = df_unique[['x', 'y']].values
        z = df_unique['z'].values

        XX, YY = np.meshgrid(grid_x, grid_y, indexing='ij')
        grid_points = np.column_stack([XX.ravel(), YY.ravel()])

        # 1. 克里金插值 (优先)
        if self.interpolation_method == 'kriging' and HAS_PYKRIGE:
            try:
                # 如果点数过多，进行降采样以避免内存溢出
                MAX_KRIGING_POINTS = 500
                if len(xy) > MAX_KRIGING_POINTS:
                    indices = np.random.choice(len(xy), MAX_KRIGING_POINTS, replace=False)
                    xy_k, z_k = xy[indices], z[indices]
                else:
                    xy_k, z_k = xy, z

                OK = OrdinaryKriging(
                    xy_k[:, 0], xy_k[:, 1], z_k,
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False
                )
                
                # 执行插值
                z_pred, ss = OK.execute('grid', grid_x, grid_y)
                # pykrige返回的形状可能是 (ny, nx)，需要转置为 (nx, ny)
                if z_pred.shape != XX.shape:
                    z_pred = z_pred.T
                
                interpolated = z_pred
                
            except Exception as e:
                print(f"克里金插值失败，回退到RBF: {e}")
                # 回退到RBF
                try:
                    interpolator = RBFInterpolator(xy, z, smoothing=self.smoothing, kernel='thin_plate_spline')
                    interpolated = interpolator(grid_points).reshape(XX.shape)
                except:
                    interpolator = LinearNDInterpolator(xy, z, fill_value=np.mean(z))
                    interpolated = interpolator(grid_points).reshape(XX.shape)

        # 2. RBF插值
        elif self.interpolation_method == 'rbf' or (self.interpolation_method == 'kriging' and not HAS_PYKRIGE):
            try:
                interpolator = RBFInterpolator(xy, z, smoothing=self.smoothing, kernel='thin_plate_spline')
                interpolated = interpolator(grid_points).reshape(XX.shape)
            except Exception as e:
                print(f"RBF插值失败，使用线性插值: {e}")
                interpolator = LinearNDInterpolator(xy, z, fill_value=np.mean(z))
                interpolated = interpolator(grid_points).reshape(XX.shape)

        # 3. IDW插值
        elif self.interpolation_method == 'idw':
            tree = KDTree(xy)
            distances, indices = tree.query(grid_points, k=min(len(points), 8))
            distances = np.maximum(distances, 1e-10)
            weights = 1.0 / distances ** 2
            weights /= weights.sum(axis=1, keepdims=True)
            interpolated = (weights * z[indices]).sum(axis=1).reshape(XX.shape)

        # 4. 线性插值
        elif self.interpolation_method == 'linear':
            interpolator = LinearNDInterpolator(xy, z, fill_value=np.mean(z))
            interpolated = interpolator(grid_points).reshape(XX.shape)

        # 5. 最近邻插值 (默认)
        else:
            interpolator = NearestNDInterpolator(xy, z)
            interpolated = interpolator(grid_points).reshape(XX.shape)
            
        # 后处理：异常值裁剪 (防止插值结果超出合理范围)
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min
        if z_range > 0:
            # 允许超出一定范围 (例如 50%)
            safe_min = z_min - z_range * 0.5
            safe_max = z_max + z_range * 0.5
            interpolated = np.clip(interpolated, safe_min, safe_max)

        return interpolated

    def build_stratigraphic_model(self, df: pd.DataFrame, lithology_classes: List[str]):
        """构建层状地质模型"""
        print("\n" + "=" * 60)
        print("构建层状三维地质模型")
        print("=" * 60)

        self.lithology_classes = lithology_classes

        # 1. 构建网格
        print("\n[1/4] 构建三维网格...")
        self.build_grid(df)

        # 2. 提取地层界面
        print("\n[2/4] 提取地层界面...")
        layer_data = self.extract_layer_interfaces(df)
        print(f"  共识别 {len(layer_data)} 个地层")

        # 3. 对每个界面进行插值
        print("\n[3/4] 插值地层界面...")
        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']
        nx, ny, nz = self.grid_shape

        sorted_layers = sorted(layer_data.keys())
        interface_tops = {}
        interface_bottoms = {}
        layer_lithology = {}

        for layer_order in sorted_layers:
            data = layer_data[layer_order]
            points = data['points']
            lithology = data['lithology']

            if len(points) < 3:
                print(f"  第 {layer_order} 层 ({lithology}) 控制点不足，跳过")
                continue

            top_interface = self.interpolate_interface(points, x_grid, y_grid, 'top_depth')
            bottom_interface = self.interpolate_interface(points, x_grid, y_grid, 'bottom_depth')

            interface_tops[layer_order] = top_interface
            interface_bottoms[layer_order] = bottom_interface
            layer_lithology[layer_order] = lithology

            print(f"  第 {layer_order} 层: {lithology} (控制点: {len(points)})")

        # 4. 根据界面位置确定每个体素的岩性
        print("\n[4/4] 填充三维模型...")

        lithology_3d = np.full((nx, ny, nz), -1, dtype=int)
        confidence_3d = np.zeros((nx, ny, nz))

        litho_to_idx = {litho: idx for idx, litho in enumerate(lithology_classes)}

        for i in range(nx):
            for j in range(ny):
                for layer_order in sorted_layers:
                    if layer_order not in interface_tops:
                        continue

                    top_depth = interface_tops[layer_order][i, j]
                    bottom_depth = interface_bottoms[layer_order][i, j]
                    lithology = layer_lithology[layer_order]

                    if lithology not in litho_to_idx:
                        continue

                    litho_idx = litho_to_idx[lithology]
                    z_top = -top_depth
                    z_bottom = -bottom_depth

                    for k in range(nz):
                        z_val = z_grid[k]
                        if z_bottom <= z_val <= z_top:
                            if lithology_3d[i, j, k] == -1:
                                lithology_3d[i, j, k] = litho_idx
                                confidence_3d[i, j, k] = 1.0

        # 处理未覆盖的区域
        undefined_mask = lithology_3d == -1
        if undefined_mask.any():
            print(f"  填充未定义区域: {undefined_mask.sum()} 个体素")
            defined_indices = np.array(np.where(~undefined_mask)).T
            undefined_indices = np.array(np.where(undefined_mask)).T

            if len(defined_indices) > 0 and len(undefined_indices) > 0:
                tree = KDTree(defined_indices)
                _, nearest_idx = tree.query(undefined_indices, k=1)
                for idx, ui in enumerate(undefined_indices):
                    di = defined_indices[nearest_idx[idx]]
                    lithology_3d[ui[0], ui[1], ui[2]] = lithology_3d[di[0], di[1], di[2]]
                    confidence_3d[ui[0], ui[1], ui[2]] = 0.5

        self.grid_lithology = lithology_3d.ravel()
        self.grid_confidence = confidence_3d.ravel()
        self.layer_interfaces = {
            'tops': interface_tops,
            'bottoms': interface_bottoms,
            'lithology': layer_lithology
        }

        print(f"\n模型构建完成!")
        print(f"  总体素数: {nx * ny * nz}")
        print(f"  有效体素: {(lithology_3d >= 0).sum()}")

    def get_voxel_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取体素模型"""
        nx, ny, nz = self.grid_shape
        lithology_3d = self.grid_lithology.reshape(nx, ny, nz)
        confidence_3d = self.grid_confidence.reshape(nx, ny, nz)
        return lithology_3d, confidence_3d

    def get_slice(self, axis: str = 'z', index: Optional[int] = None,
                  position: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """获取切片"""
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
        """导出为VTK格式"""
        if self.grid_lithology is None:
            raise ValueError("请先构建模型")

        nx, ny, nz = self.grid_shape
        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']

        lithology_3d, confidence_3d = self.get_voxel_model()

        with open(filepath, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("3D Stratigraphic Model\n")
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
                        f.write(f"{lithology_3d[i, j, k]}\n")

        print(f"VTK文件已导出: {filepath}")

    def export_numpy(self, filepath: str):
        """导出为NumPy格式"""
        lithology_3d, confidence_3d = self.get_voxel_model()
        np.savez_compressed(
            filepath,
            lithology=lithology_3d,
            confidence=confidence_3d,
            x_grid=self.grid_info['x_grid'],
            y_grid=self.grid_info['y_grid'],
            z_grid=self.grid_info['z_grid'],
            lithology_classes=self.lithology_classes
        )
        print(f"NumPy文件已导出: {filepath}")

    def export_csv(self, filepath: str, lithology_names: Optional[List[str]] = None):
        """导出为CSV格式"""
        if self.grid_lithology is None:
            raise ValueError("请先构建模型")

        nx, ny, nz = self.grid_shape
        x_grid = self.grid_info['x_grid']
        y_grid = self.grid_info['y_grid']
        z_grid = self.grid_info['z_grid']

        lithology_3d, confidence_3d = self.get_voxel_model()

        rows = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    rows.append({
                        'x': x_grid[i],
                        'y': y_grid[j],
                        'z': z_grid[k],
                        'lithology_code': lithology_3d[i, j, k],
                        'confidence': confidence_3d[i, j, k]
                    })

        df_out = pd.DataFrame(rows)
        if lithology_names:
            df_out['lithology'] = df_out['lithology_code'].map(
                {i: name for i, name in enumerate(lithology_names)}
            )
        df_out.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"CSV文件已导出: {filepath}")

    def get_statistics(self, lithology_names: Optional[List[str]] = None) -> pd.DataFrame:
        """获取模型统计信息"""
        if self.grid_lithology is None:
            raise ValueError("请先构建模型")

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


# 为了向后兼容，保留 GeoModel3D 的别名
GeoModel3D = StratigraphicModel3D


def build_stratigraphic_model_from_df(
    df: pd.DataFrame,
    lithology_classes: List[str],
    resolution: Tuple[int, int, int] = (50, 50, 50),
    output_dir: str = 'output'
) -> StratigraphicModel3D:
    """
    从钻孔数据DataFrame直接构建层状地质模型
    """
    os.makedirs(output_dir, exist_ok=True)

    model = StratigraphicModel3D(
        resolution=resolution,
        interpolation_method='rbf',
        smoothing=0.1
    )

    model.build_stratigraphic_model(df, lithology_classes)

    model.export_vtk(os.path.join(output_dir, 'stratigraphic_model.vtk'), lithology_classes)
    model.export_numpy(os.path.join(output_dir, 'stratigraphic_model.npz'))

    stats = model.get_statistics(lithology_classes)
    print("\n岩性体积统计:")
    print(stats.to_string(index=False))
    stats.to_csv(os.path.join(output_dir, 'model_statistics.csv'), index=False, encoding='utf-8-sig')

    return model


# ==================== 兼容性代码 ====================

def build_geological_model(
    trainer,
    data,
    result,
    resolution=(50, 50, 50),
    interpolation_method='rbf',
    output_dir='output'
):
    """
    构建三维地质模型 (兼容性包装器)
    """
    print("正在构建地质模型...")
    
    # 从result中获取原始DataFrame
    if 'raw_df' in result:
        df = result['raw_df']
    else:
        print("警告: 无法获取原始DataFrame，建模可能失败")
        return None
        
    # 使用StratigraphicModel3D
    model = StratigraphicModel3D(
        resolution=resolution,
        interpolation_method=interpolation_method
    )
    
    model.build_stratigraphic_model(df, result['lithology_classes'])
    
    # 保存统计信息
    os.makedirs(output_dir, exist_ok=True)
    stats = model.get_statistics(result['lithology_classes'])
    stats.to_csv(os.path.join(output_dir, 'model_statistics.csv'), index=False, encoding='utf-8-sig')
    
    # 保存模型数据
    lithology_3d, confidence_3d = model.get_voxel_model()
    np.savez(
        os.path.join(output_dir, 'geological_model.npz'),
        lithology=lithology_3d,
        confidence=confidence_3d,
        info=model.grid_info
    )
    
    return model

# 别名
GeoModel3D = StratigraphicModel3D


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("测试层状地质建模模块...")

    # 模拟数据
    np.random.seed(42)

    # 模拟3个钻孔的数据
    test_data = []
    for bh_id, (x, y) in enumerate([('BH1', (100, 100)), ('BH2', (500, 100)), ('BH3', (300, 400))]):
        bh_name, (bh_x, bh_y) = (x, y) if isinstance(x, str) else (f'BH{bh_id}', (x, y))

        # 每个钻孔有5层
        layers = [
            ('表土', 0, 10 + np.random.rand() * 5),
            ('砂岩', 10 + np.random.rand() * 5, 50 + np.random.rand() * 10),
            ('泥岩', 50 + np.random.rand() * 10, 100 + np.random.rand() * 20),
            ('煤', 100 + np.random.rand() * 20, 110 + np.random.rand() * 5),
            ('砂岩', 110 + np.random.rand() * 5, 200),
        ]

        for order, (litho, top, bottom) in enumerate(layers):
            test_data.append({
                'borehole_id': bh_name,
                'x': bh_x,
                'y': bh_y,
                'layer_order': order,
                'lithology': litho,
                'top_depth': top,
                'bottom_depth': bottom,
                'layer_thickness': bottom - top
            })

    df = pd.DataFrame(test_data)
    print("\n测试数据:")
    print(df)

    # 构建模型
    model = StratigraphicModel3D(
        resolution=(20, 20, 20),
        interpolation_method='rbf'
    )

    lithology_classes = ['表土', '砂岩', '泥岩', '煤']
    model.build_stratigraphic_model(df, lithology_classes)

    # 获取统计
    stats = model.get_statistics(lithology_classes)
    print("\n统计信息:")
    print(stats)

    print("\n测试完成!")
