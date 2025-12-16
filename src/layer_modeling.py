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
    """
    层序数据处理器

    核心设计思路：
    1. 煤层作为标志层，保持独立（15-4煤、16-1煤等各自独立）
    2. 非煤岩层填充在煤层之间
    3. 按钻孔中的实际层序构建地层模型
    4. 对于稀疏数据的层，使用插值保持连续性
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        normalize_coords: bool = True,
        normalize_thickness: bool = True,
        min_layer_occurrence: int = 2  # 层至少出现在多少钻孔中才纳入建模
    ):
        self.k_neighbors = k_neighbors
        self.normalize_coords = normalize_coords
        self.normalize_thickness = normalize_thickness
        self.min_layer_occurrence = min_layer_occurrence

        # 归一化参数
        self.coord_mean = None
        self.coord_std = None
        self.thickness_mean = None
        self.thickness_std = None

        # 层序信息
        self.layer_order = None
        self.coal_layers = []  # 煤层列表
        self.non_coal_layers = []  # 非煤层列表
        self.num_layers = 0

        # 层统计信息
        self.layer_stats = None

    def _extract_coal_number(self, name: str) -> tuple:
        """
        从煤层名称中提取编号用于排序
        如 '15-4煤' -> (15, 4, 0)
           '16-1上煤' -> (16, 1, 1)
           '16-1下煤' -> (16, 1, -1)
        """
        import re

        # 处理 "上煤"、"下煤"、"中煤" 后缀
        suffix_order = 0
        if '上' in name:
            suffix_order = 1
        elif '中' in name:
            suffix_order = 0
        elif '下' in name:
            suffix_order = -1

        # 提取数字
        numbers = re.findall(r'\d+', name)
        if len(numbers) >= 2:
            return (int(numbers[0]), int(numbers[1]), suffix_order)
        elif len(numbers) == 1:
            return (int(numbers[0]), 0, suffix_order)
        else:
            return (999, 0, suffix_order)  # 无法解析的排在后面

    def standardize_lithology(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化岩性名称（层序建模专用）

        煤层保留独立编号，非煤岩性统一命名
        """
        df = df.copy()

        def clean_name(name):
            if pd.isna(name):
                return '未知'
            name = str(name).strip()

            # 修复乱码
            garbled_fixes = {
                'ú': '煤', 'ϸɰ': '细砂', '��ɰ': '粉砂',
                'ɰ��': '砂质', '̿��': '炭质', '��ֳ': '腐殖',
                '����': '泥岩', '������': '砾岩',
            }
            for g, f in garbled_fixes.items():
                if g in name:
                    name = name.replace(g, f)

            # 煤层保留原始编号
            if '煤' in name:
                return name.strip()

            # 非煤岩性标准化
            if '砂岩' in name or ('砂' in name and '砾' not in name and '泥' not in name):
                if '粉' in name: return '粉砂岩'
                elif '细' in name: return '细砂岩'
                elif '中' in name: return '中砂岩'
                elif '粗' in name: return '粗砂岩'
                elif '含砾' in name: return '含砾砂岩'
                else: return '砂岩'
            if '砾岩' in name or '砾' in name:
                if '砂' in name: return '砂砾岩'
                elif '细' in name: return '细砾岩'
                elif '中' in name: return '中砾岩'
                elif '粗' in name: return '粗砾岩'
                else: return '砾岩'
            if '泥岩' in name or '泥' in name:
                if '炭' in name or '碳' in name: return '炭质泥岩'
                elif '砂' in name: return '砂质泥岩'
                elif '粉砂' in name: return '粉砂质泥岩'
                else: return '泥岩'
            if '腐殖' in name or '表土' in name: return '表土'
            if '黄土' in name: return '黄土'
            if '黏土' in name or '粘土' in name: return '黏土'

            return name

        df['lithology'] = df['lithology'].apply(clean_name)
        return df

    def infer_layer_order(self, df: pd.DataFrame) -> List[str]:
        """
        从钻孔数据推断岩层顺序（从深到浅）

        改进逻辑：
        1. 分离煤层和非煤层
        2. 煤层按编号排序（15-4, 15-5, 16-1, 16-2...）
        3. 非煤层按平均深度插入
        4. 最终合并成完整层序
        """
        # 检查输入数据有效性
        if df is None or df.empty:
            raise ValueError("输入数据为空，无法推断岩层顺序")

        required_cols = ['lithology', 'z', 'borehole_id', 'layer_thickness']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # 计算每种岩性的统计信息
        layer_stats = df.groupby('lithology').agg({
            'z': ['mean', 'min', 'max'],
            'borehole_id': 'nunique',
            'layer_thickness': ['sum', 'mean']
        }).reset_index()

        layer_stats.columns = ['lithology', 'z_mean', 'z_min', 'z_max',
                               'borehole_count', 'total_thickness', 'mean_thickness']

        self.layer_stats = layer_stats

        if layer_stats.empty:
            raise ValueError("数据中没有有效的岩性信息")

        # ========== 分离煤层和非煤层 ==========
        all_layers = layer_stats['lithology'].tolist()
        self.coal_layers = [l for l in all_layers if '煤' in str(l)]
        self.non_coal_layers = [l for l in all_layers if '煤' not in str(l)]

        print(f"\n识别到的岩层类型:")
        print(f"  煤层: {len(self.coal_layers)}种")
        print(f"  非煤层: {len(self.non_coal_layers)}种")

        # ========== 煤层排序（按编号，从深到浅）==========
        # 煤层编号越大通常越深（如16-3煤比15-4煤深）
        self.coal_layers.sort(key=self._extract_coal_number, reverse=True)

        print(f"\n煤层顺序（从深到浅）:")
        for i, coal in enumerate(self.coal_layers):
            stats = layer_stats[layer_stats['lithology'] == coal].iloc[0]
            print(f"  {i+1}. {coal}: 平均高程={stats['z_mean']:.1f}m, "
                  f"出现在{int(stats['borehole_count'])}个钻孔, "
                  f"平均厚度={stats['mean_thickness']:.2f}m")

        # ========== 构建完整层序 ==========
        # 策略：煤层作为骨架，非煤层根据平均深度插入到合适位置

        # 先按深度排序非煤层
        non_coal_stats = layer_stats[layer_stats['lithology'].isin(self.non_coal_layers)]
        non_coal_sorted = non_coal_stats.sort_values('z_mean')['lithology'].tolist()

        # 获取煤层的深度信息
        coal_depths = {}
        for coal in self.coal_layers:
            coal_stats = layer_stats[layer_stats['lithology'] == coal].iloc[0]
            coal_depths[coal] = coal_stats['z_mean']

        # 合并层序：按深度从深到浅
        all_with_depth = []
        for layer in self.coal_layers:
            stats = layer_stats[layer_stats['lithology'] == layer].iloc[0]
            all_with_depth.append((layer, stats['z_mean'], 'coal'))
        for layer in self.non_coal_layers:
            stats = layer_stats[layer_stats['lithology'] == layer].iloc[0]
            all_with_depth.append((layer, stats['z_mean'], 'non_coal'))

        # 按深度排序（z值从小到大 = 从深到浅）
        all_with_depth.sort(key=lambda x: x[1])

        self.layer_order = [item[0] for item in all_with_depth]
        self.num_layers = len(self.layer_order)

        print(f"\n完整层序（从深到浅，共{self.num_layers}层）:")
        for i, (layer, z, ltype) in enumerate(all_with_depth):
            stats = layer_stats[layer_stats['lithology'] == layer].iloc[0]
            type_mark = "[煤]" if ltype == 'coal' else ""
            print(f"  {i+1}. {layer}{type_mark}: z={z:.1f}m, "
                  f"钻孔数={int(stats['borehole_count'])}")

        return self.layer_order

    def extract_thickness_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取每个钻孔各层的厚度数据

        改进：
        1. 对于同一钻孔中同一岩性出现多次的情况，分别记录
        2. 使用层序号来区分不同层位
        3. 对缺失层进行标记（而非简单设为0）

        返回: DataFrame，每行一个钻孔，列为各层厚度
        """
        if self.layer_order is None:
            self.infer_layer_order(df)

        thickness_data = []
        layer_presence = {layer: 0 for layer in self.layer_order}  # 统计每层出现次数

        for bh_id in df['borehole_id'].unique():
            bh_data = df[df['borehole_id'] == bh_id].copy()

            # 按深度排序（z从大到小 = 从浅到深）
            bh_data = bh_data.sort_values('z', ascending=False)

            record = {
                'borehole_id': bh_id,
                'x': bh_data['x'].iloc[0],
                'y': bh_data['y'].iloc[0],
                'total_depth': bh_data['z'].min(),
                'surface_elevation': bh_data['z'].max()
            }

            # 提取各层厚度
            for layer_name in self.layer_order:
                layer_data = bh_data[bh_data['lithology'] == layer_name]
                if len(layer_data) > 0:
                    # 取该岩性的总厚度
                    thickness = layer_data['layer_thickness'].sum()
                    layer_presence[layer_name] += 1
                else:
                    thickness = np.nan  # 用NaN标记缺失，而非0
                record[f'thickness_{layer_name}'] = thickness

            thickness_data.append(record)

        result_df = pd.DataFrame(thickness_data)

        # 打印层出现统计
        print(f"\n各层在钻孔中的出现情况:")
        total_boreholes = len(result_df)
        for layer in self.layer_order:
            count = layer_presence[layer]
            pct = 100 * count / total_boreholes
            status = "[OK]" if count >= self.min_layer_occurrence else "[SPARSE]"
            print(f"  {layer}: {count}/{total_boreholes} ({pct:.1f}%) {status}")

        return result_df

    def fill_missing_thickness(self, thickness_df: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失的厚度数据（改进：使用中位数而非0）

        策略（参考geological_modeling_algorithms）：
        1. 对于有效数据的层：用IDW插值填充缺失位置
        2. 对于全部缺失的层：用经验最小厚度0.5m填充（不是0！）
        3. 煤层：如果插值结果太小，用中位数保护
        """
        df = thickness_df.copy()
        coords = df[['x', 'y']].values

        for layer in self.layer_order:
            col = f'thickness_{layer}'
            if col not in df.columns:
                continue

            # 找出缺失的行
            missing_mask = df[col].isna()
            if not missing_mask.any():
                continue

            # 找出有数据的行
            valid_mask = ~missing_mask
            n_valid = valid_mask.sum()
            n_missing = missing_mask.sum()

            if n_valid == 0:
                # 关键修复：全部缺失时，用经验最小厚度（不是0！）
                fill_value = 0.5  # 默认0.5米
                print(f"  警告: {layer} 在所有钻孔中都缺失，使用经验最小厚度{fill_value}m")
                df.loc[missing_mask, col] = fill_value
                continue

            # 使用IDW插值填充
            valid_coords = coords[valid_mask]
            valid_values = df.loc[valid_mask, col].values
            missing_coords = coords[missing_mask]

            # 计算中位数用于保护
            median_thickness = np.median(valid_values)
            min_thickness = np.min(valid_values)

            # KNN + IDW
            tree = KDTree(valid_coords)
            k = min(3, n_valid)  # 使用最近的3个点
            distances, indices = tree.query(missing_coords, k=k)

            if k == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)

            # IDW权重
            weights = 1.0 / (distances + 1e-8)
            weights = weights / weights.sum(axis=1, keepdims=True)

            # 插值
            interpolated = np.sum(weights * valid_values[indices], axis=1)

            # 关键修复：使用中位数保护（不是平均值）
            if '煤' in layer:
                # 煤层：如果插值结果太小，用中位数
                min_threshold = max(median_thickness * 0.5, min_thickness * 0.5, 0.5)
                interpolated = np.maximum(interpolated, min_threshold)
            else:
                # 非煤层：至少保证最小厚度0.5m
                interpolated = np.maximum(interpolated, 0.5)

            df.loc[missing_mask, col] = interpolated
            print(f"  填充 {layer}: {n_missing}个缺失值 (使用{n_valid}个有效点插值，中位数={median_thickness:.2f}m)")

        return df

    def build_graph_data(
        self,
        thickness_df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42,
        fill_missing: bool = True
    ) -> Tuple[Data, Dict]:
        """
        构建用于GNN训练的图数据

        特征: 坐标 + 其他地质特征
        标签: 各层厚度

        Args:
            thickness_df: 厚度数据DataFrame
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子，确保结果可复现
            fill_missing: 是否填充缺失的厚度数据
        """
        n_samples = len(thickness_df)

        if n_samples == 0:
            raise ValueError("厚度数据为空")

        # 填充缺失值
        if fill_missing:
            print("\n填充缺失的厚度数据...")
            thickness_df = self.fill_missing_thickness(thickness_df)

        # 设置随机种子确保可复现
        np.random.seed(random_seed)

        # 1. 提取坐标
        coords = thickness_df[['x', 'y']].values.astype(np.float32)

        # 2. 提取厚度标签
        thickness_cols = [f'thickness_{layer}' for layer in self.layer_order]
        thicknesses = thickness_df[thickness_cols].fillna(0).values.astype(np.float32)

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
            self.optimizer, mode='min', factor=0.5, patience=20
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
        smooth_sigma: float = 1.0,
        min_thickness_floor: float = 0.5  # 最小厚度下限，避免被挤成0体素
    ):
        self.resolution = resolution
        self.use_gnn = use_gnn
        self.interpolation_method = interpolation_method
        self.smooth_surfaces = smooth_surfaces
        self.smooth_sigma = smooth_sigma
        self.min_thickness_floor = max(min_thickness_floor, 0.0)

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
        
        # 关键修复：预估累加后的z范围，防止顶面超出网格
        # 累加所有层的平均厚度来估算顶面高程
        if 'layer_thickness' in df.columns and 'lithology' in df.columns:
            # 估算总厚度：所有层的平均厚度之和
            total_avg_thickness = df.groupby('lithology')['layer_thickness'].mean().sum()
            # 留出足够的空间：底面向下10%，顶面向上累加厚度+20%余量
            z_min = z_min - abs(z_max - z_min) * 0.1
            z_max = z_min + total_avg_thickness * 1.2
        else:
            # 回退：如果没有厚度信息，假设顶面在地表附近
            z_max = max(z_max, 0) + abs(z_max - z_min) * 0.5

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

    def _enforce_columnwise_order(
        self,
        all_bottoms: List[np.ndarray],
        all_tops: List[np.ndarray],
        all_thickness: List[np.ndarray],
        layer_order: List[str],
        min_gap: float = 0.5,
        min_thickness: float = 0.5,
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        逐列强制重排序层序（参考geological_modeling_algorithms）

        对每个(i,j)垂直柱子，按底面深度从小到大排序，然后自下而上重新码放，
        保证相邻层之间有min_gap，每层厚度不小于min_thickness。

        Args:
            all_bottoms: 所有层的底面列表
            all_tops: 所有层的顶面列表
            all_thickness: 所有层的厚度列表
            layer_order: 层名顺序
            min_gap: 最小层间间隙（米）
            min_thickness: 最小层厚（米）
            verbose: 是否打印信息

        Returns:
            重排序后的 (all_bottoms, all_tops)
        """
        nlay = len(all_bottoms)
        if nlay < 2:
            return all_bottoms, all_tops

        if verbose:
            print(f"  最小间隙: {min_gap}m, 最小厚度: {min_thickness}m")

        # 转为numpy数组 (nlay, nx, ny)
        bottoms = np.stack(all_bottoms)
        tops = np.stack(all_tops)

        nx, ny = bottoms.shape[1:]
        total_cells = nx * ny
        fixed_count = 0

        # 逐列处理
        for i in range(nx):
            for j in range(ny):
                # 提取这一列的所有层
                bcol = bottoms[:, i, j]
                tcol = tops[:, i, j]

                # 找出有效的层（bottom和top都是有限值）
                valid_idx = np.where(np.isfinite(bcol) & np.isfinite(tcol))[0]
                if valid_idx.size == 0:
                    continue

                # 按原始bottom深度排序（从浅到深，即从下到上）
                order = valid_idx[np.argsort(bcol[valid_idx])]

                # 检查是否需要修复
                needs_fix = False
                for ii in range(len(order) - 1):
                    if tops[order[ii], i, j] + min_gap > bottoms[order[ii+1], i, j]:
                        needs_fix = True
                        break

                if not needs_fix:
                    continue

                fixed_count += 1

                # 这一列最底部的起始深度
                z_cur = float(np.min(bcol[valid_idx]))

                # 自下而上重新码放
                for idx in order:
                    # 计算厚度
                    thick = float(tcol[idx] - bcol[idx])
                    if not np.isfinite(thick) or thick < min_thickness:
                        thick = min_thickness

                    # 重新设置底面和顶面
                    bottoms[idx, i, j] = z_cur
                    tops[idx, i, j] = z_cur + thick

                    # 更新下一层的起始位置
                    z_cur = tops[idx, i, j] + float(min_gap)

        if verbose:
            print(f"  修复了 {fixed_count}/{total_cells} 个垂直柱 ({fixed_count/total_cells*100:.1f}%)")

        # 转回列表
        all_bottoms_new = [bottoms[k] for k in range(nlay)]
        all_tops_new = [tops[k] for k in range(nlay)]

        return all_bottoms_new, all_tops_new

    def _check_vertical_order(
        self,
        all_bottoms: List[np.ndarray],
        all_tops: List[np.ndarray],
        layer_order: List[str]
    ) -> Dict[str, int]:
        """
        检查相邻层在每个网格点的垂向顺序（参考geological_modeling_algorithms）

        检查是否存在 upper.bottom < lower.top 的情况（即重叠）
        """
        nlay = len(all_bottoms)
        if nlay < 2:
            print("  只有1层，无需检查")
            return {}

        bottoms = np.stack(all_bottoms)
        tops = np.stack(all_tops)

        ny, nx = bottoms.shape[1:]
        total_cells = ny * nx

        print(f"\n[垂向顺序检查] 共 {nlay} 层，{total_cells} 个网格点")
        print(f"{'状态':>4} {'层号':>4} {'下层':>15} {'上层':>15} {'重叠点':>10} {'重叠率':>10} {'最大重叠':>12}")
        print("-" * 82)

        total_bad = 0
        results = {}

        for k in range(nlay - 1):
            lower_top = tops[k]
            upper_bottom = bottoms[k + 1]

            # 只在有效点检查
            valid = np.isfinite(lower_top) & np.isfinite(upper_bottom)
            bad = valid & (upper_bottom < lower_top)

            bad_count = int(bad.sum())
            valid_count = int(valid.sum())

            lower_name = layer_order[k]
            upper_name = layer_order[k + 1]

            if valid_count > 0:
                bad_percent = (bad_count / valid_count) * 100

                # 计算最大重叠量
                overlap = np.where(bad, lower_top - upper_bottom, 0.0)
                max_overlap = float(np.max(overlap)) if bad_count > 0 else 0.0

                status = "[X]" if bad_count > 0 else "[OK]"
                print(f"{status} {k:>4} {lower_name:>15} {upper_name:>15} {bad_count:>10} {bad_percent:>9.1f}% {max_overlap:>11.2f}m")

                total_bad += bad_count
                results[f"{lower_name}→{upper_name}"] = {
                    'bad_count': bad_count,
                    'total_count': valid_count,
                    'max_overlap': max_overlap
                }
            else:
                print(f"[WARN] {k:>4} {lower_name:>15} {upper_name:>15} {'无有效点':>10}")

        print("-" * 82)
        if total_bad == 0:
            print(f"[OK] 检查通过: 所有相邻层在所有网格点都满足垂向顺序\n")
        else:
            print(f"[ERROR] 检查失败: 共 {total_bad} 个网格点存在层间重叠\n")

        return results

    def _interpolate_thickness(
        self,
        thickness_df: pd.DataFrame,
        layer_name: str,
        min_thickness: float = 0.5,
        verbose: bool = False
    ) -> np.ndarray:
        """
        使用传统插值预测某层厚度

        改进（参考geological_modeling_algorithms）：
        1. NaN填充使用中位数而非0（避免零厚度导致层重叠）
        2. 强制最小厚度0.5m（防止薄层被挤压消失）
        3. 对于煤层，使用更保守的插值策略
        """
        col_name = f'thickness_{layer_name}'

        # 获取有效数据点（非NaN且非0）
        valid_mask = thickness_df[col_name].notna() & (thickness_df[col_name] > 0)
        n_valid = valid_mask.sum()

        nx, ny, _ = self.resolution

        if n_valid == 0:
            # 该层没有有效数据，返回最小厚度（不是0）
            print(f"    {layer_name}: 无有效数据，使用最小厚度{min_thickness:.2f}m")
            return np.full((nx, ny), min_thickness)

        points = thickness_df.loc[valid_mask, ['x', 'y']].values
        values = thickness_df.loc[valid_mask, col_name].values

        # 计算统计量用于填充和限制
        mean_thickness = values.mean()
        median_thickness = np.median(values)
        min_valid_thickness = values.min()
        max_valid_thickness = values.max()
        
        # 关键修复：根据岩性类型设置合理上限（防止插值外推爆炸）
        if '煤' in layer_name:
            # 煤层一般较薄，单层通常<30m
            max_reasonable_thickness = min(max_valid_thickness * 2, 30.0)
        elif any(x in layer_name for x in ['砾岩', '砂岩', '泥岩', '粉砂岩', '砂质泥岩', '炭质泥岩']):
            # 碎屑岩层厚度适中，一般<50m
            max_reasonable_thickness = min(max_valid_thickness * 2, 50.0)
        elif any(x in layer_name for x in ['黄土', '黏土', '表土']):
            # 覆盖层较薄，一般<20m
            max_reasonable_thickness = min(max_valid_thickness * 2, 20.0)
        else:
            # 其他岩性，保守上限50m
            max_reasonable_thickness = min(max_valid_thickness * 2, 50.0)

        # 根据数据点数量选择插值方法
        if n_valid < 3:
            # 数据太少，使用中位数填充（更稳健）
            fill_value = max(median_thickness, min_thickness)
            thickness = np.full((nx, ny), fill_value)
            print(f"    {layer_name}: 数据点仅{n_valid}个，使用中位数{fill_value:.2f}m填充")
        else:
            # RBF插值
            if self.interpolation_method == 'rbf':
                # 根据数据点密度调整平滑参数
                smoothing = 0.1 if n_valid > 10 else 0.5
                interpolator = RBFInterpolator(
                    points, values,
                    kernel='thin_plate_spline',
                    smoothing=smoothing
                )
            else:
                # 线性插值：NaN区域使用中位数填充
                interpolator = LinearNDInterpolator(points, values, fill_value=median_thickness)

            thickness = interpolator(self.grid_xy).reshape(nx, ny)

            # 关键修复：将插值产生的NaN用中位数填充（不是0！）
            nan_mask = ~np.isfinite(thickness)
            if nan_mask.any():
                fill_value = max(median_thickness, min_thickness)
                thickness[nan_mask] = fill_value
                nan_count = nan_mask.sum()
                print(f"    {layer_name}: {nan_count}个位置插值失败，用中位数{fill_value:.2f}m填充")

        # 确保非负且不小于最小厚度
        thickness = np.maximum(thickness, min_thickness)
        
        # 关键修复：使用合理上限防止外推产生极端值
        thickness = np.minimum(thickness, max_reasonable_thickness)
        
        if verbose and n_valid >= 3:
            print(f"    {layer_name}: 数据点{n_valid}个，中位数={median_thickness:.2f}m，上限={max_reasonable_thickness:.2f}m")

        # 平滑
        if self.smooth_surfaces:
            thickness = gaussian_filter(thickness, sigma=self.smooth_sigma)
            # 平滑后再次保证范围
            thickness = np.clip(thickness, min_thickness, max_reasonable_thickness)

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
            if self.min_thickness_floor > 0:
                layer_thickness = np.where(layer_thickness > 0, np.maximum(layer_thickness, self.min_thickness_floor), 0)
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
                thickness_dict[layer_name] = self._interpolate_thickness(
                    thickness_df, layer_name, min_thickness=0.5, verbose=verbose
                )

        # 4. 逐层累加构建曲面
        if verbose:
            print("\n[4/4] 逐层累加构建曲面...")

        current_surface = bottom_surface.copy()

        # 保存所有层的底面和顶面用于后续重排序
        all_bottoms = []
        all_tops = []
        all_thickness = []

        for layer_name in processor.layer_order:
            thickness = thickness_dict[layer_name]
            top_surface = current_surface + thickness

            all_bottoms.append(current_surface.copy())
            all_tops.append(top_surface.copy())
            all_thickness.append(thickness)

            self.surfaces[layer_name] = {
                'bottom': current_surface.copy(),
                'top': top_surface.copy(),
                'thickness': thickness
            }

            if verbose:
                mean_thickness = thickness.mean()
                print(f"  {layer_name}: 平均厚度 {mean_thickness:.2f} m")

            current_surface = top_surface

        # 关键修复：逐列强制重排序（参考geological_modeling_algorithms）
        if verbose:
            print("\n[列重排序] 强制垂向顺序...")

        all_bottoms, all_tops = self._enforce_columnwise_order(
            all_bottoms, all_tops, all_thickness,
            processor.layer_order, verbose=verbose
        )

        # 更新surfaces
        for i, layer_name in enumerate(processor.layer_order):
            self.surfaces[layer_name] = {
                'bottom': all_bottoms[i],
                'top': all_tops[i],
                'thickness': all_tops[i] - all_bottoms[i]
            }

        # 验证垂向顺序
        if verbose:
            self._check_vertical_order(all_bottoms, all_tops, processor.layer_order)

        # 5. 体素化 - 使用向量化优化
        if verbose:
            print("\n体素化填充...")
            print(f"  Z网格范围: {z_grid[0]:.1f} ~ {z_grid[-1]:.1f} m")

        self.lithology_3d = np.zeros((nx, ny, nz), dtype=np.int32)

        # 预计算z网格的索引映射
        z_min_grid, z_max_grid = z_grid[0], z_grid[-1]
        
        # 诊断：统计每层实际占用的z范围
        if verbose:
            print(f"\n  各层实际z范围（前10层）:")
            for idx, layer_name in enumerate(processor.layer_order[:10]):
                surface_info = self.surfaces[layer_name]
                b_min, b_max = surface_info['bottom'].min(), surface_info['bottom'].max()
                t_min, t_max = surface_info['top'].min(), surface_info['top'].max()
                print(f"    {idx}. {layer_name}: bottom[{b_min:.1f}, {b_max:.1f}], top[{t_min:.1f}, {t_max:.1f}]")

        for layer_idx, layer_name in enumerate(processor.layer_order):
            surface_info = self.surfaces[layer_name]
            bottom = surface_info['bottom']
            top = surface_info['top']

            # 向量化：计算所有网格点的k索引
            # 将底面和顶面的z值映射到z_grid索引
            k_start_all = np.searchsorted(z_grid, bottom.ravel())
            k_end_all = np.searchsorted(z_grid, top.ravel())

            # 边界裁剪
            k_start_all = np.clip(k_start_all, 0, nz)
            k_end_all = np.clip(k_end_all, 0, nz)

            # 重塑回2D
            k_start_2d = k_start_all.reshape(nx, ny)
            k_end_2d = k_end_all.reshape(nx, ny)

            # 填充体素 - 使用循环但每次填充一整列
            for i in range(nx):
                for j in range(ny):
                    k_start = k_start_2d[i, j]
                    k_end = k_end_2d[i, j]
                    # 索引越界直接跳过
                    if k_start >= nz:
                        continue

                    # 若厚度小于z分辨率导致同一索引，强制占至少1个体素
                    if k_end <= k_start and thickness[i, j] > 0:
                        k_end = min(k_start + 1, nz)

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

    processor = LayerDataProcessor(k_neighbors=10, min_layer_occurrence=1)

    # 先标准化岩性名称
    df = processor.standardize_lithology(df)
    if verbose:
        print(f"\n标准化后岩性类别: {df['lithology'].nunique()}种")

    processor.infer_layer_order(df)
    thickness_df = processor.extract_thickness_data(df)
    # 填充缺失厚度，防止稀疏层被当成0厚度
    thickness_df = processor.fill_missing_thickness(thickness_df)

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
