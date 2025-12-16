"""
地质厚度建模 - 基于传统地质统计学方法

核心理念：
1. 放弃深度学习，使用经典地质统计学方法
2. 每层独立建模，自动选择最佳插值方法
3. 层序累加法构建三维模型
4. 专为小样本地质数据优化

为什么不用GNN:
- 地质厚度预测本质是空间插值，不是图神经网络任务
- 28样本 vs 46层输出 = 严重欠定问题
- 传统方法在地质领域有几十年成功经验
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import griddata, Rbf
from typing import Dict, List, Tuple, Optional
import warnings


# =============================================================================
# 一、BlockModel数据结构
# =============================================================================

class BlockModel:
    """
    地层块体模型

    表示一个地层的三维几何形态
    """

    def __init__(
        self,
        name: str,
        points: int,
        top_surface: np.ndarray,
        bottom_surface: np.ndarray
    ):
        self.name = name
        self.points = points
        self.top_surface = np.asarray(top_surface, dtype=float)
        self.bottom_surface = np.asarray(bottom_surface, dtype=float)

        # 计算厚度
        thickness = self.top_surface - self.bottom_surface
        thickness = np.clip(thickness, 0.0, None)
        self.thickness_grid = thickness

        # 统计信息
        self.avg_thickness = float(np.nanmean(thickness))
        self.max_thickness = float(np.nanmax(thickness))
        self.avg_height = float(np.nanmean(self.top_surface))
        self.max_height = float(np.nanmax(self.top_surface))
        self.min_height = float(np.nanmin(self.top_surface))
        self.avg_bottom = float(np.nanmean(self.bottom_surface))
        self.min_bottom = float(np.nanmin(self.bottom_surface))
        self.base = self.avg_bottom


# =============================================================================
# 二、逐层厚度预测器（核心）
# =============================================================================

class PerLayerThicknessPredictor:
    """
    逐层厚度预测器 - 为每层选择最佳插值方法

    自适应策略：
    - 数据点 < 3: 常数法（用中位数）
    - 数据点 3-10: IDW（反距离加权，稳定可靠）
    - 数据点 > 10: Kriging/RBF（捕捉空间相关性）

    这比GNN更适合地质数据：
    1. 不需要大量训练样本
    2. 保证空间连续性
    3. 可解释性强
    4. 计算效率高
    """

    def __init__(
        self,
        layer_order: List[str],
        default_method: str = 'idw',
        idw_power: float = 2.0,
        n_neighbors: int = 8,
        min_thickness: float = 0.5
    ):
        """
        初始化

        Args:
            layer_order: 地层顺序（从底到顶）
            default_method: 默认插值方法
            idw_power: IDW幂次
            n_neighbors: 最近邻数量
            min_thickness: 最小厚度
        """
        self.layer_order = layer_order
        self.default_method = default_method
        self.idw_power = idw_power
        self.n_neighbors = n_neighbors
        self.min_thickness = min_thickness

        self.layer_data = {}
        self.layer_stats = {}
        self.is_fitted = False

    def fit(
        self,
        borehole_data: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        layer_col: str = 'lithology',
        thickness_col: str = 'layer_thickness'
    ):
        """
        拟合模型（实际上是准备每层的插值数据）
        """
        print("\n" + "="*60)
        print("逐层厚度预测器 - 数据准备")
        print("="*60)

        for layer_name in self.layer_order:
            layer_df = borehole_data[borehole_data[layer_col] == layer_name]
            n_points = len(layer_df)

            if n_points == 0:
                print(f"  {layer_name}: 无数据 → 使用默认厚度 {self.min_thickness}m")
                self.layer_data[layer_name] = {
                    'method': 'constant',
                    'value': self.min_thickness,
                    'n_points': 0
                }
                continue

            coords = layer_df[[x_col, y_col]].values
            thickness = layer_df[thickness_col].values

            # 统计信息
            stats = {
                'mean': np.mean(thickness),
                'median': np.median(thickness),
                'std': np.std(thickness),
                'min': np.min(thickness),
                'max': np.max(thickness),
                'n_points': n_points
            }
            self.layer_stats[layer_name] = stats

            # 根据数据点数量自动选择方法
            if n_points < 3:
                method = 'constant'
                print(f"  {layer_name}: {n_points}点 → 常数法 (中位数={stats['median']:.2f}m)")
            elif n_points < 10:
                method = 'idw'
                print(f"  {layer_name}: {n_points}点 → IDW法 (均值={stats['mean']:.2f}±{stats['std']:.2f}m)")
            else:
                method = self.default_method
                print(f"  {layer_name}: {n_points}点 → {method.upper()}法 "
                      f"(均值={stats['mean']:.2f}±{stats['std']:.2f}m, 范围={stats['min']:.2f}~{stats['max']:.2f}m)")

            self.layer_data[layer_name] = {
                'method': method,
                'coords': coords,
                'thickness': thickness,
                'stats': stats,
                'n_points': n_points
            }

        self.is_fitted = True
        print(f"\n✓ 数据准备完成: {len(self.layer_order)}层")

    def predict_grid(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        预测网格厚度

        Args:
            grid_x: X网格坐标 (nx,)
            grid_y: Y网格坐标 (ny,)
            verbose: 是否打印进度

        Returns:
            thickness_grids: {layer_name: (ny, nx)}
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit()方法")

        XI, YI = np.meshgrid(grid_x, grid_y)
        xi_flat = XI.flatten()
        yi_flat = YI.flatten()
        query_points = np.column_stack([xi_flat, yi_flat])

        thickness_grids = {}

        if verbose:
            print(f"\n预测网格厚度: {XI.shape[1]}×{XI.shape[0]} = {len(query_points)}个点")

        for layer_name in self.layer_order:
            data = self.layer_data.get(layer_name)
            if data is None:
                thickness_grids[layer_name] = np.full(XI.shape, self.min_thickness)
                continue

            method = data['method']

            if method == 'constant':
                grid_thick = np.full(len(xi_flat), data.get('value', data['stats']['median']))

            elif method == 'idw':
                grid_thick = self._idw_interpolate(
                    data['coords'], data['thickness'], query_points
                )

            elif method == 'rbf':
                grid_thick = self._rbf_interpolate(
                    data['coords'], data['thickness'], query_points
                )

            elif method == 'linear':
                grid_thick = griddata(
                    data['coords'], data['thickness'], query_points,
                    method='linear', fill_value=data['stats']['median']
                )

            else:  # nearest
                grid_thick = griddata(
                    data['coords'], data['thickness'], query_points,
                    method='nearest'
                )

            # 后处理
            grid_thick = np.nan_to_num(grid_thick, nan=data['stats']['median'])
            grid_thick = np.clip(grid_thick, self.min_thickness, None)

            thickness_grids[layer_name] = grid_thick.reshape(XI.shape)

        if verbose:
            print("✓ 网格预测完成")

        return thickness_grids

    def _idw_interpolate(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray
    ) -> np.ndarray:
        """
        反距离加权插值（IDW）

        公式: v(x) = Σ w_i * v_i, w_i = 1/d_i^p / Σ 1/d_j^p
        """
        tree = KDTree(coords)
        k = min(self.n_neighbors, len(coords))

        distances, indices = tree.query(query_points, k=k)

        # 处理距离为0的情况（查询点恰好在数据点上）
        distances = np.maximum(distances, 1e-10)

        # 计算权重
        weights = 1.0 / (distances ** self.idw_power)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights = weights / weights_sum

        # 加权平均
        result = np.sum(weights * values[indices], axis=1)

        return result

    def _rbf_interpolate(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray
    ) -> np.ndarray:
        """
        径向基函数插值（RBF）
        """
        try:
            rbf = Rbf(coords[:, 0], coords[:, 1], values,
                     function='thin_plate', smooth=0.5)
            result = rbf(query_points[:, 0], query_points[:, 1])
        except Exception as e:
            warnings.warn(f"RBF插值失败: {e}, 回退到IDW")
            result = self._idw_interpolate(coords, values, query_points)

        return result

    def get_layer_summary(self) -> pd.DataFrame:
        """获取每层的摘要信息"""
        rows = []
        for layer_name in self.layer_order:
            data = self.layer_data.get(layer_name, {})
            stats = data.get('stats', {})
            rows.append({
                '地层': layer_name,
                '数据点数': data.get('n_points', 0),
                '插值方法': data.get('method', 'N/A'),
                '均值(m)': stats.get('mean', 0),
                '中位数(m)': stats.get('median', 0),
                '标准差(m)': stats.get('std', 0),
                '最小(m)': stats.get('min', 0),
                '最大(m)': stats.get('max', 0)
            })
        return pd.DataFrame(rows)

    def evaluate(
        self,
        test_data: pd.DataFrame,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        x_col: str = 'x',
        y_col: str = 'y',
        layer_col: str = 'lithology',
        thickness_col: str = 'layer_thickness'
    ) -> Dict[str, float]:
        """
        评估预测性能（在测试集上）
        """
        # 预测网格
        thickness_grids = self.predict_grid(grid_x, grid_y, verbose=False)

        predictions = []
        actuals = []

        XI, YI = np.meshgrid(grid_x, grid_y)

        for _, row in test_data.iterrows():
            layer = row[layer_col]
            if layer not in thickness_grids:
                continue

            actual = row[thickness_col]

            # 找最近的网格点
            x, y = row[x_col], row[y_col]
            i = np.argmin(np.abs(grid_x - x))
            j = np.argmin(np.abs(grid_y - y))

            pred = thickness_grids[layer][j, i]

            predictions.append(pred)
            actuals.append(actual)

        if len(predictions) == 0:
            return {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0, 'n_samples': 0}

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 计算指标
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))

        ss_res = np.sum((actuals - predictions)**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r2 = max(0, 1 - ss_res / (ss_tot + 1e-8))

        # 相对误差
        mape = np.mean(np.abs((predictions - actuals) / (actuals + 0.1))) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_samples': len(predictions)
        }


# =============================================================================
# 三、三维地质模型构建器
# =============================================================================

class GeologicalModelBuilder:
    """
    三维地质模型构建器 - 层序累加法

    核心算法：
    1. 从底层开始，逐层累加厚度
    2. 数学上保证层间无重叠无空缺
    3. 自动修正垂向顺序冲突
    """

    def __init__(
        self,
        layer_order: List[str],
        resolution: int = 50,
        base_level: float = 0.0,
        gap_value: float = 0.0,
        min_thickness: float = 0.5
    ):
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
            thickness_grids: 每层的厚度网格
            x_range, y_range: 坐标范围

        Returns:
            block_models: BlockModel列表
            XI, YI: 坐标网格
        """
        # 创建坐标网格
        x_grid = np.linspace(x_range[0], x_range[1], self.resolution)
        y_grid = np.linspace(y_range[0], y_range[1], self.resolution)
        XI, YI = np.meshgrid(x_grid, y_grid)

        block_models = []
        current_base_surface = np.full(XI.shape, self.base_level, dtype=float)

        print(f"\n{'='*60}")
        print(f"构建三维地质模型 - 层序累加法")
        print(f"{'='*60}")
        print(f"网格分辨率: {self.resolution}×{self.resolution}")
        print(f"基准面高程: {self.base_level}m")
        print(f"层间间隙: {self.gap_value}m")
        print(f"{'='*60}\n")

        for layer_name in self.layer_order:
            if layer_name not in thickness_grids:
                print(f"  ⚠️  {layer_name}: 无厚度数据，跳过")
                continue

            thickness_grid = thickness_grids[layer_name]

            # 确保尺寸匹配
            if thickness_grid.shape != XI.shape:
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

            print(f"  ✓ {layer_name:<20} "
                  f"底={bottom_surface.mean():>6.2f}m, "
                  f"厚={thickness_grid.mean():>5.2f}m "
                  f"(σ={thickness_grid.std():>4.2f}), "
                  f"顶={top_surface.mean():>6.2f}m")

            # 更新下一层的基准面
            current_base_surface = top_surface + self.gap_value

        print(f"\n{'='*60}")
        print(f"✓ 模型构建完成: {len(block_models)}层")
        print(f"  总高度: {current_base_surface.mean() - self.base_level:.2f}m")
        print(f"{'='*60}\n")

        # 强制垂向顺序修正
        self._enforce_columnwise_order(block_models)

        return block_models, XI, YI

    def _enforce_columnwise_order(
        self,
        block_models: List[BlockModel],
        min_gap: float = 0.5
    ):
        """逐列强制垂向顺序修正"""
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
        if fixed_count > 0:
            print(f"[垂向修正] 修复了 {fixed_count}/{total_cells} 个柱子 ({100*fixed_count/total_cells:.1f}%)")

    def build_voxel_model(
        self,
        block_models: List[BlockModel],
        XI: np.ndarray,
        YI: np.ndarray,
        nz: int = 50
    ) -> Tuple[np.ndarray, Dict]:
        """构建体素模型"""
        z_min = min(bm.bottom_surface.min() for bm in block_models)
        z_max = max(bm.top_surface.max() for bm in block_models)

        z_range = z_max - z_min
        z_min -= z_range * 0.05
        z_max += z_range * 0.05

        ny, nx = XI.shape
        z_grid = np.linspace(z_min, z_max, nz)

        layer_to_idx = {bm.name: i+1 for i, bm in enumerate(block_models)}
        voxel_grid = np.zeros((nx, ny, nz), dtype=np.int32)

        for k, z in enumerate(z_grid):
            for bm in block_models:
                mask = (z >= bm.bottom_surface) & (z < bm.top_surface)
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
# 四、统一接口
# =============================================================================

class GeologicalThicknessModeling:
    """
    地质厚度建模统一接口

    工作流程：
    1. 准备每层的插值数据
    2. 预测厚度网格
    3. 层序累加法构建三维模型
    4. 输出体素模型
    """

    def __init__(
        self,
        layer_order: List[str],
        resolution: int = 50,
        base_level: float = 0.0,
        gap_value: float = 0.0,
        min_thickness: float = 0.5,
        idw_power: float = 2.0
    ):
        self.layer_order = layer_order
        self.resolution = resolution
        self.base_level = base_level
        self.gap_value = gap_value
        self.min_thickness = min_thickness

        self.predictor = PerLayerThicknessPredictor(
            layer_order=layer_order,
            idw_power=idw_power,
            min_thickness=min_thickness
        )

        self.builder = GeologicalModelBuilder(
            layer_order=layer_order,
            resolution=resolution,
            base_level=base_level,
            gap_value=gap_value,
            min_thickness=min_thickness
        )

        self.block_models = None
        self.grid_info = None

    def fit(
        self,
        borehole_data: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        layer_col: str = 'lithology',
        thickness_col: str = 'layer_thickness'
    ):
        """准备插值数据"""
        self.predictor.fit(
            borehole_data=borehole_data,
            x_col=x_col,
            y_col=y_col,
            layer_col=layer_col,
            thickness_col=thickness_col
        )

    def build(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> List[BlockModel]:
        """构建三维地质模型"""
        # 创建网格
        grid_x = np.linspace(x_range[0], x_range[1], self.resolution)
        grid_y = np.linspace(y_range[0], y_range[1], self.resolution)

        # 预测厚度网格
        thickness_grids = self.predictor.predict_grid(grid_x, grid_y)

        # 构建模型
        self.block_models, XI, YI = self.builder.build_model(
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
        """获取体素模型"""
        if self.block_models is None:
            raise RuntimeError("请先调用build()构建模型")

        return self.builder.build_voxel_model(
            block_models=self.block_models,
            XI=self.grid_info['XI'],
            YI=self.grid_info['YI'],
            nz=nz
        )

    def get_layer_summary(self) -> pd.DataFrame:
        """获取每层摘要"""
        return self.predictor.get_layer_summary()

    def evaluate(
        self,
        test_data: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        layer_col: str = 'lithology',
        thickness_col: str = 'layer_thickness'
    ) -> Dict[str, float]:
        """评估模型性能"""
        grid_x = np.linspace(
            self.grid_info['x_range'][0],
            self.grid_info['x_range'][1],
            self.resolution
        )
        grid_y = np.linspace(
            self.grid_info['y_range'][0],
            self.grid_info['y_range'][1],
            self.resolution
        )

        return self.predictor.evaluate(
            test_data=test_data,
            grid_x=grid_x,
            grid_y=grid_y,
            x_col=x_col,
            y_col=y_col,
            layer_col=layer_col,
            thickness_col=thickness_col
        )


# =============================================================================
# 五、便捷函数
# =============================================================================

def quick_build_model(
    borehole_data: pd.DataFrame,
    layer_order: List[str],
    resolution: int = 50,
    base_level: float = 0.0,
    **kwargs
) -> GeologicalThicknessModeling:
    """
    快速构建地质模型的便捷函数

    Example:
        >>> modeling = quick_build_model(
        ...     borehole_data=df,
        ...     layer_order=['Layer1', 'Layer2', ...],
        ...     resolution=50
        ... )
        >>> block_models = modeling.block_models
    """
    # 确定坐标范围
    x_range = (borehole_data['x'].min(), borehole_data['x'].max())
    y_range = (borehole_data['y'].min(), borehole_data['y'].max())

    # 创建建模器
    modeling = GeologicalThicknessModeling(
        layer_order=layer_order,
        resolution=resolution,
        base_level=base_level,
        **kwargs
    )

    # 拟合和构建
    modeling.fit(borehole_data)
    modeling.build(x_range, y_range)

    return modeling


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("地质厚度建模模块测试")
    print("=" * 60)

    # 模拟数据
    np.random.seed(42)

    layers = ['Layer_A', 'Layer_B', 'Layer_C', 'Layer_D']
    n_boreholes = 15

    data = []
    for bh in range(n_boreholes):
        x = np.random.rand() * 1000
        y = np.random.rand() * 1000
        for layer in layers:
            if np.random.rand() > 0.2:  # 80%概率存在
                thickness = np.random.rand() * 10 + 2
                data.append({
                    'borehole': f'BH{bh}',
                    'x': x,
                    'y': y,
                    'lithology': layer,
                    'layer_thickness': thickness
                })

    df = pd.DataFrame(data)

    print(f"\n模拟钻孔数据: {n_boreholes}个钻孔, {len(df)}条记录")

    # 快速构建模型
    modeling = quick_build_model(
        borehole_data=df,
        layer_order=layers,
        resolution=30,
        base_level=0.0
    )

    # 打印摘要
    print("\n" + "="*60)
    print("每层数据摘要:")
    print("="*60)
    summary = modeling.get_layer_summary()
    print(summary.to_string(index=False))

    # 打印模型
    print("\n" + "="*60)
    print("构建的BlockModel:")
    print("="*60)
    for bm in modeling.block_models:
        print(f"{bm.name}: "
              f"平均厚度={bm.avg_thickness:.2f}m, "
              f"底面={bm.avg_bottom:.2f}m, "
              f"顶面={bm.avg_height:.2f}m")

    print("\n✓ 测试完成!")
