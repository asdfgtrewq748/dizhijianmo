"""
数据加载与预处理模块
将钻孔数据转换为PyTorch Geometric图结构

针对敏东矿区钻孔数据格式:
- 坐标文件: 钻孔名,坐标x,坐标y
- 钻孔文件: 序号,名称,厚度/m,弹性模量/Gpa,容重/kN*m-3,抗拉强度/MPa
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree, Delaunay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List, Dict, Union
import os
import json
import glob


class BoreholeDataProcessor:
    """
    钻孔数据处理器
    负责将原始钻孔数据转换为图结构
    """

    def __init__(
        self,
        k_neighbors: int = 10,              # KNN图的邻居数
        max_distance: Optional[float] = None,  # 最大连接距离（超过此距离不连边）
        graph_type: str = 'knn',            # 图构建方式: 'knn', 'radius', 'delaunay'
        normalize_coords: bool = True,       # 是否标准化坐标
        normalize_features: bool = True,     # 是否标准化特征
        sample_interval: float = 1.0         # 沿深度方向的采样间隔(米)
    ):
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance
        self.graph_type = graph_type
        self.normalize_coords = normalize_coords
        self.normalize_features = normalize_features
        self.sample_interval = sample_interval

        # 数据预处理器
        self.coord_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # 元数据
        self.lithology_classes: List[str] = []
        self.feature_names: List[str] = []

    def load_coordinates(self, coord_file: str) -> pd.DataFrame:
        """
        加载钻孔坐标文件

        Args:
            coord_file: 坐标文件路径

        Returns:
            df: 包含钻孔名和坐标的DataFrame
        """
        # 尝试不同编码读取
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
            try:
                df = pd.read_csv(coord_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        # 标准化列名
        df.columns = df.columns.str.strip()

        # 重命名列
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if '钻孔' in col or 'name' in col_lower or 'id' in col_lower or '名' in col:
                col_mapping[col] = 'borehole_id'
            elif 'x' in col_lower or ('坐标' in col and 'x' in col.lower()):
                col_mapping[col] = 'x'
            elif 'y' in col_lower or ('坐标' in col and 'y' in col.lower()):
                col_mapping[col] = 'y'

        df = df.rename(columns=col_mapping)

        print(f"加载坐标文件: {len(df)} 个钻孔")
        return df

    def load_single_borehole(self, borehole_file: str, borehole_id: str,
                             x: float, y: float, surface_z: float = 0.0) -> pd.DataFrame:
        """
        加载单个钻孔数据文件

        Args:
            borehole_file: 钻孔数据文件路径
            borehole_id: 钻孔编号
            x, y: 钻孔平面坐标
            surface_z: 地表高程 (默认0, 向下为负)

        Returns:
            df: 包含采样点的DataFrame
        """
        # 尝试不同编码读取
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
            try:
                df = pd.read_csv(borehole_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        # 清理列名
        df.columns = df.columns.str.strip()

        # 识别列
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if '序号' in col or 'id' in col_lower or 'no' in col_lower:
                col_mapping[col] = 'layer_id'
            elif '名称' in col or '岩性' in col or 'name' in col_lower or 'lith' in col_lower:
                col_mapping[col] = 'lithology'
            elif '厚度' in col or 'thick' in col_lower:
                col_mapping[col] = 'thickness'
            elif '弹性模量' in col or 'elastic' in col_lower:
                col_mapping[col] = 'elastic_modulus'
            elif '容重' in col or 'density' in col_lower or '密度' in col:
                col_mapping[col] = 'density'
            elif '抗拉强度' in col or 'tensile' in col_lower:
                col_mapping[col] = 'tensile_strength'

        df = df.rename(columns=col_mapping)

        # 确保必要的列存在
        if 'lithology' not in df.columns or 'thickness' not in df.columns:
            print(f"警告: {borehole_file} 缺少必要列, 跳过")
            return pd.DataFrame()

        # 按序号排序 (如果有的话), 确保从地表向下
        # CSV中序号1是地表层（如腐殖土），序号越大越深
        if 'layer_id' in df.columns:
            df = df.sort_values('layer_id', ascending=True).reset_index(drop=True)

        # 为每一层添加序号（从地表向下），用于保留层序信息
        df['layer_order'] = np.arange(len(df))

        # 计算每层的顶底深度
        # 从地表(z=0)开始，累加厚度得到深度
        df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce').fillna(0)

        # 计算累积深度
        cumulative_depth = df['thickness'].cumsum()
        df['top_depth'] = cumulative_depth.shift(1).fillna(0)  # 顶部深度
        df['bottom_depth'] = cumulative_depth  # 底部深度
        df['center_depth'] = (df['top_depth'] + df['bottom_depth']) / 2  # 层中心深度

        # 生成采样点
        samples = []
        for _, row in df.iterrows():
            top = row['top_depth']
            bottom = row['bottom_depth']
            lithology = row['lithology']

            # 获取可选属性
            elastic_modulus = row.get('elastic_modulus', np.nan)
            density = row.get('density', np.nan)
            tensile_strength = row.get('tensile_strength', np.nan)

            # 沿深度方向采样
            if self.sample_interval > 0:
                # 在每层内按间隔采样
                depth_points = np.arange(top, bottom, self.sample_interval)
                if len(depth_points) == 0:
                    depth_points = [row['center_depth']]  # 至少取层中心点
            else:
                # 只取层中心点
                depth_points = [row['center_depth']]

            for z_depth in depth_points:
                samples.append({
                    'borehole_id': borehole_id,
                    'x': x,
                    'y': y,
                    'z': surface_z - z_depth,  # 深度为负值
                    'lithology': lithology,
                    'layer_order': row['layer_order'],  # 层序编号，保留同岩性不同层位的差异
                    'top_depth': row['top_depth'],
                    'bottom_depth': row['bottom_depth'],
                    'center_depth': row['center_depth'],
                    'layer_thickness': row['thickness'],
                    'relative_depth': z_depth / (row['bottom_depth'] + 0.01),  # 相对深度位置
                    'depth_ratio': (z_depth - row['top_depth']) / (row['thickness'] + 0.01),  # 层内位置
                    'elastic_modulus': elastic_modulus,
                    'density': density,
                    'tensile_strength': tensile_strength
                })

        return pd.DataFrame(samples)

    def load_all_boreholes(
        self,
        data_dir: str,
        coord_file: str = None,
        surface_elevation: float = 0.0
    ) -> pd.DataFrame:
        """
        加载目录中所有钻孔数据

        Args:
            data_dir: 数据目录路径
            coord_file: 坐标文件路径 (如果为None, 则在data_dir中查找)
            surface_elevation: 地表高程

        Returns:
            df: 合并后的所有采样点DataFrame
        """
        # 查找坐标文件
        if coord_file is None:
            coord_files = glob.glob(os.path.join(data_dir, '*坐标*.csv'))
            if coord_files:
                coord_file = coord_files[0]
            else:
                raise FileNotFoundError("未找到坐标文件，请指定coord_file参数")

        # 加载坐标
        coord_df = self.load_coordinates(coord_file)

        # 创建钻孔ID到坐标的映射
        coord_map = {}
        for _, row in coord_df.iterrows():
            bh_id = str(row['borehole_id']).strip()
            coord_map[bh_id] = (row['x'], row['y'])

        print(f"坐标映射: {list(coord_map.keys())}")

        # 查找所有钻孔文件
        all_csv = glob.glob(os.path.join(data_dir, '*.csv'))

        # 排除坐标文件
        borehole_files = [f for f in all_csv if '坐标' not in os.path.basename(f)]

        print(f"找到 {len(borehole_files)} 个钻孔文件")

        # 加载所有钻孔
        all_samples = []
        loaded_count = 0

        for bh_file in borehole_files:
            # 从文件名提取钻孔ID
            bh_id = os.path.splitext(os.path.basename(bh_file))[0]

            # 查找坐标
            if bh_id in coord_map:
                x, y = coord_map[bh_id]
            else:
                print(f"警告: 钻孔 {bh_id} 无坐标信息，跳过")
                continue

            # 加载钻孔数据
            df = self.load_single_borehole(bh_file, bh_id, x, y, surface_elevation)

            if not df.empty:
                all_samples.append(df)
                loaded_count += 1

        if not all_samples:
            raise ValueError("未加载任何钻孔数据")

        # 合并所有数据
        combined_df = pd.concat(all_samples, ignore_index=True)

        print(f"\n数据加载完成:")
        print(f"  - 加载钻孔数: {loaded_count}")
        print(f"  - 总采样点数: {len(combined_df)}")
        print(f"  - 岩性种类: {combined_df['lithology'].nunique()}")
        print(f"  - 深度范围: {combined_df['z'].min():.1f} ~ {combined_df['z'].max():.1f} m")

        return combined_df

    def standardize_lithology(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化岩性名称 (处理同义词、别名等)

        Args:
            df: 包含lithology列的DataFrame

        Returns:
            df: 标准化后的DataFrame
        """
        # 岩性标准化映射表
        lithology_mapping = {
            # 煤层
            '煤': '煤',
            'ú': '煤',  # 乱码修复
            # 砂岩类
            '细砂岩': '细砂岩',
            'ϸɰ��': '细砂岩',
            '中砂岩': '中砂岩',
            '粗砂岩': '粗砂岩',
            '粉砂岩': '粉砂岩',
            '��ɰ��': '粉砂岩',
            # 砾岩类
            '中砾岩': '中砾岩',
            '粗砾岩': '粗砾岩',
            '细砾岩': '细砾岩',
            '砾岩': '砾岩',
            # 泥岩类
            '泥�ite': '泥岩',
            '泥岩': '泥岩',
            '砂质泥岩': '砂质泥岩',
            'ɰ������': '砂质泥岩',
            '炭质泥岩': '炭质泥岩',
            '̿������': '炭质泥岩',
            # 其他
            '腐殖土': '腐殖土',
            '��ֳ��': '腐殖土',
            '砂砾岩': '砂砾岩',
            'ɰ����': '砂砾岩',
            '页岩': '页岩',
            '灰岩': '灰岩',
        }

        # 创建一个清理后的岩性列
        def clean_lithology(name):
            if pd.isna(name):
                return '未知'

            name = str(name).strip()

            # 先尝试直接映射
            if name in lithology_mapping:
                return lithology_mapping[name]

            # 尝试包含关系匹配
            for key, value in lithology_mapping.items():
                if key in name or name in key:
                    return value

            # 基于关键字匹配
            if '煤' in name or 'ú' in name:
                return '煤'
            elif '泥' in name or '��' in name:
                if '炭' in name or '̿' in name:
                    return '炭质泥岩'
                elif '砂' in name or 'ɰ' in name:
                    return '砂质泥岩'
                else:
                    return '泥岩'
            elif '砂' in name or 'ɰ' in name:
                if '粉' in name or '��' in name:
                    return '粉砂岩'
                elif '细' in name or 'ϸ' in name:
                    return '细砂岩'
                elif '中' in name:
                    return '中砂岩'
                elif '粗' in name:
                    return '粗砂岩'
                elif '砾' in name or '��' in name:
                    return '砂砾岩'
                else:
                    return '砂岩'
            elif '砾' in name or '��' in name:
                return '砾岩'
            elif '腐' in name or 'ֳ' in name:
                return '腐殖土'
            elif '页' in name:
                return '页岩'
            elif '灰' in name:
                return '灰岩'
            else:
                return name  # 保留原名

        df = df.copy()
        df['lithology'] = df['lithology'].apply(clean_lithology)

        print(f"\n标准化后的岩性类别:")
        print(df['lithology'].value_counts())

        return df

    def build_graph(
        self,
        coords: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        根据坐标构建图的边索引

        Args:
            coords: 节点坐标 [N, 3]
            features: 节点特征 [N, F] (可选)

        Returns:
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E] (基于距离)
        """
        N = len(coords)

        # 动态调整k值,确保不超过节点数
        k = min(self.k_neighbors, N - 1)

        if self.graph_type == 'knn':
            # KNN图: 每个点连接最近的K个邻居
            tree = KDTree(coords)
            distances, indices = tree.query(coords, k=k + 1)

            edges = []
            weights = []
            for i in range(N):
                for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                    # 如果设置了最大距离，则过滤
                    if self.max_distance is None or dist <= self.max_distance:
                        edges.append([i, j])
                        edges.append([j, i])  # 无向图，添加反向边
                        weights.extend([dist, dist])

            if not edges:
                raise ValueError("图构建失败：没有边生成。请检查max_distance参数。")

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(weights, dtype=torch.float32)

            # 将距离转换为相似度权重 (距离越近权重越大)
            edge_weight = torch.exp(-edge_weight / edge_weight.mean())

        elif self.graph_type == 'radius':
            # 半径图: 连接半径范围内的所有点
            if self.max_distance is None:
                raise ValueError("radius图类型需要指定max_distance")

            tree = KDTree(coords)
            pairs = tree.query_pairs(self.max_distance, output_type='ndarray')

            if len(pairs) == 0:
                print("警告: radius图没有边，回退到KNN")
                self.graph_type = 'knn'
                return self.build_graph(coords, features)

            edges = []
            weights = []
            for i, j in pairs:
                dist = np.linalg.norm(coords[i] - coords[j])
                edges.append([i, j])
                edges.append([j, i])
                weights.extend([dist, dist])

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(weights, dtype=torch.float32)
            edge_weight = torch.exp(-edge_weight / edge_weight.mean())

        elif self.graph_type == 'delaunay':
            # Delaunay三角剖分图 (适合不规则分布的数据)
            try:
                tri = Delaunay(coords)
                edges = set()
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            edges.add((simplex[i], simplex[j]))
                            edges.add((simplex[j], simplex[i]))

                edge_list = list(edges)
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                # 计算边权重
                weights = []
                for src, dst in zip(edge_index[0], edge_index[1]):
                    dist = np.linalg.norm(coords[src.item()] - coords[dst.item()])
                    weights.append(dist)
                edge_weight = torch.tensor(weights, dtype=torch.float32)
                edge_weight = torch.exp(-edge_weight / edge_weight.mean())

            except Exception as e:
                print(f"Delaunay三角剖分失败，回退到KNN: {e}")
                self.graph_type = 'knn'
                return self.build_graph(coords, features)
        else:
            raise ValueError(f"未知的图类型: {self.graph_type}")

        return edge_index, edge_weight

    def process(
        self,
        df: pd.DataFrame,
        label_col: str = 'lithology',
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        处理钻孔数据并创建图数据集

        Args:
            df: 钻孔数据DataFrame
            label_col: 岩性标签列名
            feature_cols: 要使用的特征列 (None则使用所有可用的数值特征)
            test_size: 测试集比例
            val_size: 验证集比例

        Returns:
            data_dict: 包含训练/验证/测试数据的字典
        """
        # 标准化岩性名称
        df = self.standardize_lithology(df)

        # 提取坐标
        coords = df[['x', 'y', 'z']].values.astype(np.float32)

        # 确定特征列
        exclude_cols = {'x', 'y', 'z', label_col, 'borehole_id', 'id', 'layer_id',
                        'top_depth', 'bottom_depth', 'center_depth'}

        if feature_cols is None:
            # 自动检测数值特征列
            feature_cols = []
            for c in df.columns:
                if c not in exclude_cols:
                    if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        # 检查是否有有效值
                        if df[c].notna().sum() > len(df) * 0.1:  # 至少10%有效
                            feature_cols.append(c)

        self.feature_names = feature_cols
        print(f"使用的特征列: {feature_cols}")

        # 提取特征
        if feature_cols:
            features = df[feature_cols].values.astype(np.float32)
            # 填充缺失值
            features = np.nan_to_num(features, nan=0.0)
        else:
            features = np.zeros((len(df), 1), dtype=np.float32)

        # 提取并编码标签
        labels = df[label_col].values
        self.label_encoder.fit(labels)
        labels_encoded = self.label_encoder.transform(labels)
        self.lithology_classes = list(self.label_encoder.classes_)

        print(f"\n岩性类别 ({len(self.lithology_classes)}类):")
        for i, cls in enumerate(self.lithology_classes):
            count = (labels_encoded == i).sum()
            print(f"  {i}: {cls} ({count}个)")

        # 标准化坐标
        if self.normalize_coords:
            coords_normalized = self.coord_scaler.fit_transform(coords)
        else:
            coords_normalized = coords

        # 标准化特征
        if self.normalize_features and feature_cols:
            features_normalized = self.feature_scaler.fit_transform(features)
        else:
            features_normalized = features

        # 合并坐标和特征作为节点特征
        node_features = np.concatenate([coords_normalized, features_normalized], axis=1)
        x = torch.tensor(node_features, dtype=torch.float32)
        y = torch.tensor(labels_encoded, dtype=torch.long)

        # 构建图
        print(f"\n构建{self.graph_type}图 (k={self.k_neighbors})...")
        edge_index, edge_weight = self.build_graph(coords)
        print(f"图构建完成: {x.shape[0]} 节点, {edge_index.shape[1]} 边")
        print(f"平均每节点边数: {edge_index.shape[1] / x.shape[0]:.1f}")

        # 创建数据划分掩码
        n_nodes = len(df)
        indices = np.arange(n_nodes)

        # 先划分出测试集
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels_encoded
        )

        # 再从训练+验证集中划分验证集
        val_ratio = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio, random_state=42,
            stratify=labels_encoded[train_val_idx]
        )

        # 创建掩码
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # 创建PyG Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            coords=torch.tensor(coords, dtype=torch.float32),  # 保留原始坐标用于可视化
        )

        # 返回结果
        result = {
            'data': data,
            'num_features': x.shape[1],
            'num_classes': len(self.lithology_classes),
            'lithology_classes': self.lithology_classes,
            'feature_names': ['x', 'y', 'z'] + feature_cols,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'raw_df': df  # 保留原始数据用于可视化
        }

        print(f"\n数据集划分:")
        print(f"  训练集: {train_mask.sum().item()} ({100*train_mask.sum().item()/n_nodes:.1f}%)")
        print(f"  验证集: {val_mask.sum().item()} ({100*val_mask.sum().item()/n_nodes:.1f}%)")
        print(f"  测试集: {test_mask.sum().item()} ({100*test_mask.sum().item()/n_nodes:.1f}%)")

        return result

    def save_preprocessor(self, path: str):
        """保存预处理器状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        state = {
            'coord_scaler_mean': self.coord_scaler.mean_.tolist() if hasattr(self.coord_scaler, 'mean_') else None,
            'coord_scaler_scale': self.coord_scaler.scale_.tolist() if hasattr(self.coord_scaler, 'scale_') else None,
            'feature_scaler_mean': self.feature_scaler.mean_.tolist() if hasattr(self.feature_scaler, 'mean_') else None,
            'feature_scaler_scale': self.feature_scaler.scale_.tolist() if hasattr(self.feature_scaler, 'scale_') else None,
            'lithology_classes': self.lithology_classes,
            'feature_names': self.feature_names,
            'k_neighbors': self.k_neighbors,
            'max_distance': self.max_distance,
            'graph_type': self.graph_type,
            'sample_interval': self.sample_interval
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_preprocessor(self, path: str):
        """加载预处理器状态"""
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        if state['coord_scaler_mean']:
            self.coord_scaler.mean_ = np.array(state['coord_scaler_mean'])
            self.coord_scaler.scale_ = np.array(state['coord_scaler_scale'])
        if state['feature_scaler_mean']:
            self.feature_scaler.mean_ = np.array(state['feature_scaler_mean'])
            self.feature_scaler.scale_ = np.array(state['feature_scaler_scale'])

        self.lithology_classes = state['lithology_classes']
        self.label_encoder.classes_ = np.array(self.lithology_classes)
        self.feature_names = state['feature_names']
        self.k_neighbors = state['k_neighbors']
        self.max_distance = state['max_distance']
        self.graph_type = state['graph_type']
        self.sample_interval = state.get('sample_interval', 1.0)


class GridInterpolator:
    """
    网格插值器
    将稀疏的钻孔预测结果插值到规则三维网格上
    """

    def __init__(
        self,
        grid_resolution: Tuple[int, int, int] = (50, 50, 50),  # x, y, z方向的网格数
        bounds: Optional[Dict] = None  # {'x': (min, max), 'y': (min, max), 'z': (min, max)}
    ):
        self.grid_resolution = grid_resolution
        self.bounds = bounds

    def create_grid_points(
        self,
        coords: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        创建三维规则网格点

        Args:
            coords: 原始数据坐标 [N, 3]

        Returns:
            grid_points: 网格点坐标 [M, 3]
            grid_info: 网格信息字典
        """
        # 确定边界
        if self.bounds is None:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        else:
            x_min, x_max = self.bounds['x']
            y_min, y_max = self.bounds['y']
            z_min, z_max = self.bounds['z']

        # 创建网格
        nx, ny, nz = self.grid_resolution
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)

        # 生成所有网格点
        xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        grid_info = {
            'resolution': self.grid_resolution,
            'bounds': {
                'x': (x_min, x_max),
                'y': (y_min, y_max),
                'z': (z_min, z_max)
            },
            'x_grid': x_grid,
            'y_grid': y_grid,
            'z_grid': z_grid
        }

        return grid_points, grid_info


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("测试数据处理模块 - 加载真实钻孔数据...")

    # 数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    # 创建处理器
    processor = BoreholeDataProcessor(
        k_neighbors=8,
        graph_type='knn',
        sample_interval=2.0  # 每2米采样一个点
    )

    # 加载所有钻孔数据
    df = processor.load_all_boreholes(data_dir)

    print(f"\n数据预览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")

    # 处理数据
    result = processor.process(df)

    data = result['data']
    print(f"\n图数据信息:")
    print(f"  节点数: {data.num_nodes}")
    print(f"  边数: {data.num_edges}")
    print(f"  特征维度: {data.x.shape[1]}")
    print(f"  类别数: {result['num_classes']}")

    # 保存预处理器
    processor.save_preprocessor(os.path.join(data_dir, 'preprocessor.json'))
    print("\n预处理器状态已保存")
