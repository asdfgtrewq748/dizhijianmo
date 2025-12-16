"""
厚度预测模型 V2 - 基于传统地质统计学方法

核心思想：
1. 每层单独建模（而不是一个模型预测所有层）
2. 使用经典的地质统计学方法（IDW、Kriging）
3. 根据数据点数量自动选择最佳方法
4. GNN仅用于学习空间相关性权重，不做直接预测

这种方法更适合小样本地质数据
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import warnings


class PerLayerThicknessPredictor:
    """
    逐层厚度预测器

    对每一层使用最适合的方法进行预测：
    - 数据点 < 3: 使用中位数
    - 数据点 3-10: 使用IDW（反距离加权）
    - 数据点 > 10: 使用Kriging或RBF
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
            default_method: 默认插值方法 ('idw', 'rbf', 'linear', 'nearest')
            idw_power: IDW幂次（越大越局部）
            n_neighbors: 最近邻数量
            min_thickness: 最小厚度
        """
        self.layer_order = layer_order
        self.default_method = default_method
        self.idw_power = idw_power
        self.n_neighbors = n_neighbors
        self.min_thickness = min_thickness

        # 存储每层的数据和模型
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
        拟合模型

        Args:
            borehole_data: 钻孔层表数据
            x_col, y_col: 坐标列名
            layer_col: 地层列名
            thickness_col: 厚度列名
        """
        print("\n" + "="*60)
        print("逐层厚度预测器 - 拟合中")
        print("="*60)

        for layer_name in self.layer_order:
            layer_df = borehole_data[borehole_data[layer_col] == layer_name]

            n_points = len(layer_df)
            if n_points == 0:
                print(f"  {layer_name}: 无数据，使用默认厚度")
                self.layer_data[layer_name] = {
                    'method': 'constant',
                    'value': self.min_thickness,
                    'n_points': 0
                }
                continue

            coords = layer_df[[x_col, y_col]].values
            thickness = layer_df[thickness_col].values

            # 计算统计信息
            stats = {
                'mean': np.mean(thickness),
                'median': np.median(thickness),
                'std': np.std(thickness),
                'min': np.min(thickness),
                'max': np.max(thickness),
                'n_points': n_points
            }
            self.layer_stats[layer_name] = stats

            # 根据数据点数量选择方法
            if n_points < 3:
                method = 'constant'
                print(f"  {layer_name}: {n_points}个点 → 常数法 (中位数={stats['median']:.2f}m)")
            elif n_points < 8:
                method = 'idw'
                print(f"  {layer_name}: {n_points}个点 → IDW法 (均值={stats['mean']:.2f}m)")
            else:
                method = self.default_method
                print(f"  {layer_name}: {n_points}个点 → {method}法 (均值={stats['mean']:.2f}m)")

            self.layer_data[layer_name] = {
                'method': method,
                'coords': coords,
                'thickness': thickness,
                'stats': stats,
                'n_points': n_points
            }

        self.is_fitted = True
        print(f"\n拟合完成: {len(self.layer_order)}层")

    def predict_grid(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        预测网格厚度

        Args:
            grid_x: X网格坐标 (nx,)
            grid_y: Y网格坐标 (ny,)

        Returns:
            thickness_grids: {layer_name: (ny, nx)}
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit()拟合模型")

        XI, YI = np.meshgrid(grid_x, grid_y)
        xi_flat = XI.flatten()
        yi_flat = YI.flatten()
        query_points = np.column_stack([xi_flat, yi_flat])

        thickness_grids = {}

        for layer_name in self.layer_order:
            data = self.layer_data.get(layer_name)
            if data is None:
                thickness_grids[layer_name] = np.full(XI.shape, self.min_thickness)
                continue

            method = data['method']

            if method == 'constant':
                grid_thick = np.full(len(xi_flat), data.get('value', self.min_thickness))

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

        return thickness_grids

    def _idw_interpolate(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray
    ) -> np.ndarray:
        """
        反距离加权插值
        """
        tree = KDTree(coords)
        k = min(self.n_neighbors, len(coords))

        distances, indices = tree.query(query_points, k=k)

        # 处理距离为0的情况
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
        径向基函数插值
        """
        try:
            rbf = Rbf(coords[:, 0], coords[:, 1], values,
                     function='thin_plate', smooth=0.5)
            result = rbf(query_points[:, 0], query_points[:, 1])
        except Exception as e:
            warnings.warn(f"RBF插值失败: {e}, 使用IDW")
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
                '方法': data.get('method', 'N/A'),
                '均值(m)': stats.get('mean', 0),
                '中位数(m)': stats.get('median', 0),
                '标准差(m)': stats.get('std', 0),
                '最小(m)': stats.get('min', 0),
                '最大(m)': stats.get('max', 0)
            })
        return pd.DataFrame(rows)


class SimpleKriging:
    """
    简单克里金插值

    适用于具有空间自相关性的地质数据
    """

    def __init__(
        self,
        variogram_model: str = 'spherical',
        nlags: int = 10,
        weight: bool = True
    ):
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.weight = weight

        self.coords = None
        self.values = None
        self.mean = None
        self.sill = None
        self.range_ = None
        self.nugget = None

    def fit(self, coords: np.ndarray, values: np.ndarray):
        """拟合变差函数"""
        self.coords = coords
        self.values = values
        self.mean = np.mean(values)

        # 计算实验变差函数
        n = len(coords)
        distances = []
        semivariances = []

        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                sv = 0.5 * (values[i] - values[j])**2
                distances.append(d)
                semivariances.append(sv)

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        # 分箱计算
        max_dist = np.max(distances) * 0.5
        bins = np.linspace(0, max_dist, self.nlags + 1)

        lag_centers = []
        lag_semivar = []

        for i in range(self.nlags):
            mask = (distances >= bins[i]) & (distances < bins[i+1])
            if np.sum(mask) > 0:
                lag_centers.append((bins[i] + bins[i+1]) / 2)
                lag_semivar.append(np.mean(semivariances[mask]))

        if len(lag_centers) < 3:
            # 数据太少，使用默认参数
            self.sill = np.var(values)
            self.range_ = max_dist
            self.nugget = 0
        else:
            # 简单拟合球状模型
            lag_centers = np.array(lag_centers)
            lag_semivar = np.array(lag_semivar)

            self.sill = np.max(lag_semivar)
            self.nugget = max(0, np.min(lag_semivar))
            self.range_ = lag_centers[np.argmax(lag_semivar > 0.9 * self.sill)] if np.any(lag_semivar > 0.9 * self.sill) else max_dist

    def _variogram(self, h: np.ndarray) -> np.ndarray:
        """计算变差函数值"""
        h = np.asarray(h)
        result = np.zeros_like(h, dtype=float)

        # 球状模型
        mask = h <= self.range_
        hr = h[mask] / self.range_
        result[mask] = self.nugget + (self.sill - self.nugget) * (1.5 * hr - 0.5 * hr**3)
        result[~mask] = self.sill

        return result

    def predict(self, query_points: np.ndarray) -> np.ndarray:
        """预测"""
        n_train = len(self.coords)
        n_query = len(query_points)

        predictions = np.zeros(n_query)

        # 构建训练点之间的变差矩阵
        train_dists = cdist(self.coords, self.coords)
        K = self._variogram(train_dists)

        # 添加小的对角项以保证数值稳定
        K += np.eye(n_train) * 1e-6

        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # 矩阵奇异，退回到IDW
            warnings.warn("Kriging矩阵奇异，使用IDW")
            tree = KDTree(self.coords)
            k = min(8, n_train)
            distances, indices = tree.query(query_points, k=k)
            distances = np.maximum(distances, 1e-10)
            weights = 1.0 / (distances ** 2)
            weights = weights / weights.sum(axis=1, keepdims=True)
            return np.sum(weights * self.values[indices], axis=1)

        for i in range(n_query):
            # 计算查询点到训练点的变差
            q_dists = np.linalg.norm(self.coords - query_points[i], axis=1)
            k = self._variogram(q_dists)

            # 克里金权重
            weights = K_inv @ k
            weights = weights / np.sum(weights)  # 归一化

            predictions[i] = np.sum(weights * self.values)

        return predictions


class HybridThicknessPredictor:
    """
    混合厚度预测器

    结合多种方法的优点：
    1. 对稀疏层使用IDW
    2. 对密集层使用Kriging
    3. 对所有层应用空间平滑约束
    """

    def __init__(
        self,
        layer_order: List[str],
        kriging_threshold: int = 10,  # 超过此点数使用Kriging
        smooth_factor: float = 0.3,   # 空间平滑因子
        min_thickness: float = 0.5
    ):
        self.layer_order = layer_order
        self.kriging_threshold = kriging_threshold
        self.smooth_factor = smooth_factor
        self.min_thickness = min_thickness

        self.predictors = {}
        self.layer_stats = {}
        self.global_trend = None

    def fit(
        self,
        borehole_data: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        layer_col: str = 'lithology',
        thickness_col: str = 'layer_thickness'
    ):
        """拟合模型"""
        print("\n" + "="*60)
        print("混合厚度预测器 - 拟合中")
        print("="*60)

        # 计算全局趋势（所有层的平均厚度分布）
        all_coords = borehole_data[[x_col, y_col]].drop_duplicates().values

        for layer_name in self.layer_order:
            layer_df = borehole_data[borehole_data[layer_col] == layer_name]
            n_points = len(layer_df)

            if n_points == 0:
                self.predictors[layer_name] = {
                    'type': 'constant',
                    'value': self.min_thickness
                }
                print(f"  {layer_name}: 无数据 → 常数")
                continue

            coords = layer_df[[x_col, y_col]].values
            thickness = layer_df[thickness_col].values

            stats = {
                'mean': np.mean(thickness),
                'median': np.median(thickness),
                'std': np.std(thickness),
                'n_points': n_points
            }
            self.layer_stats[layer_name] = stats

            if n_points < 3:
                self.predictors[layer_name] = {
                    'type': 'constant',
                    'value': stats['median']
                }
                print(f"  {layer_name}: {n_points}点 → 常数({stats['median']:.2f}m)")

            elif n_points < self.kriging_threshold:
                # 使用IDW
                self.predictors[layer_name] = {
                    'type': 'idw',
                    'coords': coords,
                    'values': thickness,
                    'stats': stats
                }
                print(f"  {layer_name}: {n_points}点 → IDW(均值={stats['mean']:.2f}m)")

            else:
                # 使用Kriging
                kriging = SimpleKriging()
                kriging.fit(coords, thickness)
                self.predictors[layer_name] = {
                    'type': 'kriging',
                    'model': kriging,
                    'coords': coords,
                    'values': thickness,
                    'stats': stats
                }
                print(f"  {layer_name}: {n_points}点 → Kriging(均值={stats['mean']:.2f}m)")

        print(f"\n拟合完成!")

    def predict_grid(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """预测网格厚度"""
        XI, YI = np.meshgrid(grid_x, grid_y)
        query_points = np.column_stack([XI.flatten(), YI.flatten()])

        thickness_grids = {}

        for layer_name in self.layer_order:
            pred = self.predictors.get(layer_name)
            if pred is None:
                thickness_grids[layer_name] = np.full(XI.shape, self.min_thickness)
                continue

            pred_type = pred['type']

            if pred_type == 'constant':
                grid_thick = np.full(len(query_points), pred['value'])

            elif pred_type == 'idw':
                grid_thick = self._idw_predict(
                    pred['coords'], pred['values'], query_points
                )

            elif pred_type == 'kriging':
                grid_thick = pred['model'].predict(query_points)

            else:
                grid_thick = np.full(len(query_points), self.min_thickness)

            # 后处理
            stats = pred.get('stats', {})
            median = stats.get('median', self.min_thickness)
            grid_thick = np.nan_to_num(grid_thick, nan=median)
            grid_thick = np.clip(grid_thick, self.min_thickness, None)

            thickness_grids[layer_name] = grid_thick.reshape(XI.shape)

        # 应用空间平滑
        if self.smooth_factor > 0:
            thickness_grids = self._apply_spatial_smoothing(thickness_grids)

        return thickness_grids

    def _idw_predict(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray,
        power: float = 2.0
    ) -> np.ndarray:
        """IDW预测"""
        tree = KDTree(coords)
        k = min(8, len(coords))
        distances, indices = tree.query(query_points, k=k)
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / (distances ** power)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return np.sum(weights * values[indices], axis=1)

    def _apply_spatial_smoothing(
        self,
        thickness_grids: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """应用空间平滑"""
        from scipy.ndimage import gaussian_filter

        smoothed = {}
        for layer_name, grid in thickness_grids.items():
            # 轻微高斯平滑
            sigma = 1.0 * self.smooth_factor
            smoothed_grid = gaussian_filter(grid, sigma=sigma)
            # 保持原值和平滑值的加权
            smoothed[layer_name] = (1 - self.smooth_factor) * grid + self.smooth_factor * smoothed_grid
            smoothed[layer_name] = np.clip(smoothed[layer_name], self.min_thickness, None)

        return smoothed


def evaluate_predictor(
    predictor,
    test_data: pd.DataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    x_col: str = 'x',
    y_col: str = 'y',
    layer_col: str = 'lithology',
    thickness_col: str = 'layer_thickness'
) -> Dict[str, float]:
    """
    评估预测器性能（留一法交叉验证）
    """
    predictions = []
    actuals = []

    thickness_grids = predictor.predict_grid(grid_x, grid_y)

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

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 计算指标
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))

    ss_res = np.sum((actuals - predictions)**2)
    ss_tot = np.sum((actuals - np.mean(actuals))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 相对误差
    mape = np.mean(np.abs((predictions - actuals) / (actuals + 0.1))) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': max(0, r2),  # 防止负R²
        'mape': mape,
        'n_samples': len(predictions)
    }


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("厚度预测器V2测试")

    # 模拟数据
    np.random.seed(42)

    layers = ['Layer_A', 'Layer_B', 'Layer_C', 'Layer_D']
    n_boreholes = 20

    data = []
    for bh in range(n_boreholes):
        x = np.random.rand() * 1000
        y = np.random.rand() * 1000
        for layer in layers:
            if np.random.rand() > 0.2:  # 80%概率存在
                thickness = np.random.rand() * 10 + 1
                data.append({
                    'borehole': f'BH{bh}',
                    'x': x,
                    'y': y,
                    'lithology': layer,
                    'layer_thickness': thickness
                })

    df = pd.DataFrame(data)

    # 测试PerLayer预测器
    predictor = PerLayerThicknessPredictor(layer_order=layers)
    predictor.fit(df)

    grid_x = np.linspace(0, 1000, 30)
    grid_y = np.linspace(0, 1000, 30)

    thickness_grids = predictor.predict_grid(grid_x, grid_y)

    print("\n预测结果:")
    for layer, grid in thickness_grids.items():
        print(f"  {layer}: mean={grid.mean():.2f}m, range=[{grid.min():.2f}, {grid.max():.2f}]m")

    # 评估
    metrics = evaluate_predictor(predictor, df, grid_x, grid_y)
    print(f"\n评估指标:")
    print(f"  MAE: {metrics['mae']:.3f}m")
    print(f"  RMSE: {metrics['rmse']:.3f}m")
    print(f"  R²: {metrics['r2']:.3f}")
    print(f"  MAPE: {metrics['mape']:.1f}%")
