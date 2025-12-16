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
        if not os.path.exists(coord_file):
            raise FileNotFoundError(f"坐标文件不存在: {coord_file}")

        # 尝试不同编码读取
        df = None
        last_error = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                df = pd.read_csv(coord_file, encoding=encoding)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise IOError(f"读取坐标文件失败: {coord_file}, 错误: {e}")

        if df is None:
            raise IOError(f"无法以任何编码读取文件 {coord_file}: {last_error}")

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

        # 验证必需列
        required_cols = ['borehole_id', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"坐标文件缺少必需列: {missing_cols}, 现有列: {list(df.columns)}")

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
        if not os.path.exists(borehole_file):
            print(f"警告: 钻孔文件不存在: {borehole_file}")
            return pd.DataFrame()

        # 尝试不同编码读取
        df = None
        last_error = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                df = pd.read_csv(borehole_file, encoding=encoding)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                print(f"警告: 读取 {borehole_file} 失败: {e}")
                return pd.DataFrame()

        if df is None:
            print(f"警告: 无法以任何编码读取 {borehole_file}: {last_error}")
            return pd.DataFrame()

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

    # ========== 厚度任务专用：直接读取“层表”（每层一行），不做层内采样 ==========
    def load_single_borehole_layers(
        self,
        borehole_file: str,
        borehole_id: str,
        x: float,
        y: float,
        surface_z: float = 0.0
    ) -> pd.DataFrame:
        """读取单个钻孔的层表（每层一行）。保持文件原始行序作为层序。

        Args:
            borehole_file: 钻孔数据文件路径
            borehole_id: 钻孔编号
            x, y: 钻孔平面坐标
            surface_z: 地表高程（默认0，向下为负）

        Returns:
            df_layers: 包含层级信息的DataFrame，列包括
                borehole_id, x, y, lithology, layer_order, layer_thickness,
                top_depth, bottom_depth, center_depth
        """

        if not os.path.exists(borehole_file):
            print(f"警告: 钻孔文件不存在: {borehole_file}")
            return pd.DataFrame()

        df = None
        last_error = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                df = pd.read_csv(borehole_file, encoding=encoding)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                print(f"警告: 读取 {borehole_file} 失败: {e}")
                return pd.DataFrame()

        if df is None:
            print(f"警告: 无法以任何编码读取 {borehole_file}: {last_error}")
            return pd.DataFrame()

        df.columns = df.columns.str.strip()

        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if '序号' in col or 'id' in col_lower or 'no' in col_lower:
                col_mapping[col] = 'layer_id'
            elif '名称' in col or '岩性' in col or 'name' in col_lower or 'lith' in col_lower:
                col_mapping[col] = 'lithology'
            elif '厚度' in col or 'thick' in col_lower:
                col_mapping[col] = 'layer_thickness'
            elif '弹性模量' in col or 'elastic' in col_lower:
                col_mapping[col] = 'elastic_modulus'
            elif '容重' in col or 'density' in col_lower or '密度' in col:
                col_mapping[col] = 'density'
            elif '抗拉强度' in col or 'tensile' in col_lower:
                col_mapping[col] = 'tensile_strength'

        df = df.rename(columns=col_mapping)

        if 'lithology' not in df.columns or 'layer_thickness' not in df.columns:
            print(f"警告: {borehole_file} 缺少必要列, 跳过")
            return pd.DataFrame()

        # 按文件原始行序定义层序（顶部到最底部）
        df = df.reset_index(drop=True)
        df['layer_order'] = np.arange(len(df), dtype=np.int32)

        df['layer_thickness'] = pd.to_numeric(df['layer_thickness'], errors='coerce').fillna(0)
        cumulative_depth = df['layer_thickness'].cumsum()
        df['top_depth'] = cumulative_depth.shift(1).fillna(0)
        df['bottom_depth'] = cumulative_depth
        df['center_depth'] = (df['top_depth'] + df['bottom_depth']) / 2

        df['borehole_id'] = borehole_id
        df['x'] = x
        df['y'] = y
        df['surface_z'] = surface_z

        return df

    def load_all_borehole_layers(
        self,
        data_dir: str,
        coord_file: str = None,
        surface_elevation: float = 0.0
    ) -> pd.DataFrame:
        """加载目录内所有钻孔的“层表”，保留每层一行。

        返回合并后的层表 DataFrame。
        """

        if coord_file is None:
            coord_files = glob.glob(os.path.join(data_dir, '*坐标*.csv'))
            if coord_files:
                coord_file = coord_files[0]
            else:
                raise FileNotFoundError("未找到坐标文件，请指定coord_file参数")

        coord_df = self.load_coordinates(coord_file)
        coord_map = {str(r['borehole_id']).strip(): (r['x'], r['y']) for _, r in coord_df.iterrows()}

        all_csv = glob.glob(os.path.join(data_dir, '*.csv'))
        borehole_files = [f for f in all_csv if '坐标' not in os.path.basename(f)]

        all_layers = []
        loaded = 0
        for bh_file in borehole_files:
            bh_id = os.path.splitext(os.path.basename(bh_file))[0]
            if bh_id not in coord_map:
                print(f"警告: 钻孔 {bh_id} 无坐标信息，跳过")
                continue
            x, y = coord_map[bh_id]
            df_layers = self.load_single_borehole_layers(bh_file, bh_id, x, y, surface_elevation)
            if not df_layers.empty:
                all_layers.append(df_layers)
                loaded += 1

        if not all_layers:
            raise ValueError("未加载任何钻孔层表")

        combined = pd.concat(all_layers, ignore_index=True)
        print(f"层表加载完成: 钻孔数 {loaded}, 总层数 {len(combined)}")
        return combined

    # ========== 厚度任务：构建节点级厚度/存在/掩码张量 ==========
    def _infer_global_layer_order(self, df_layers: pd.DataFrame) -> List[str]:
        """依据各钻孔的层位次序推断全局 layer_order。
        思路：按每孔的行序赋位置，计算每个岩性的出现位置中位数，按中位数升序排序。
        """
        positions = {}
        for bh_id, sub in df_layers.groupby('borehole_id'):
            for idx, lith in zip(sub['layer_order'].values, sub['lithology'].values):
                positions.setdefault(lith, []).append(idx)

        layer_order = sorted(positions.keys(), key=lambda k: np.median(positions[k]))
        print("自动推断的 layer_order:")
        for i, name in enumerate(layer_order):
            print(f"  {i}: {name} (样本数 {len(positions[name])})")
        return layer_order

    def process_thickness(
        self,
        df_layers: pd.DataFrame,
        layer_order: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        merge_coal: bool = False,
        split_mode: str = 'random',  # 'random' | 'kfold'
        n_splits: int = 5,
        random_seed: int = 42
    ) -> Dict:
        """厚度任务数据处理：每个钻孔一个节点，标签为各层厚度向量。

        返回字典包含 PyG Data 与统计信息。
        """

        # 1) 岩性标准化，明确禁用煤层合并
        df_layers = self.standardize_lithology(df_layers, merge_coal=merge_coal)

        # 2) layer_order 确定
        if layer_order is None:
            layer_order = self._infer_global_layer_order(df_layers)
        layer_to_idx = {name: i for i, name in enumerate(layer_order)}
        num_layers = len(layer_order)

        # 3) 按钻孔聚合，构造 y_thick / y_exist / y_mask
        borehole_groups = list(df_layers.groupby('borehole_id'))
        num_bh = len(borehole_groups)

        y_thick = np.zeros((num_bh, num_layers), dtype=np.float32)
        y_exist = np.zeros((num_bh, num_layers), dtype=np.float32)
        y_mask = np.zeros((num_bh, num_layers), dtype=np.float32)

        node_features = []
        coords = []
        bh_ids = []

        for idx, (bh_id, sub) in enumerate(borehole_groups):
            # 使用原始层序（行序）
            sub = sub.sort_values('layer_order', ascending=True)
            total_depth = sub['layer_thickness'].sum()
            x = sub['x'].iloc[0]
            y = sub['y'].iloc[0]
            surface_z = sub.get('surface_z', pd.Series([0])).iloc[0]

            coords.append([x, y, 0.0])  # 仅平面坐标，z=0
            bh_ids.append(bh_id)

            # 逐层填充
            for _, row in sub.iterrows():
                lith = row['lithology']
                if lith not in layer_to_idx:
                    continue
                li = layer_to_idx[lith]
                thickness = float(row['layer_thickness'])
                y_thick[idx, li] = thickness
                y_exist[idx, li] = 1.0
                y_mask[idx, li] = 1.0

            # 钻孔级特征：x, y, total_depth, surface_z
            feat = [x, y, total_depth, surface_z]
            if feature_cols:
                for c in feature_cols:
                    feat.append(sub[c].iloc[0] if c in sub.columns else np.nan)
            node_features.append(feat)

        node_features = np.array(node_features, dtype=np.float32)
        coords = np.array(coords, dtype=np.float32)

        # 4) 统计信息（存在率）
        exist_counts = y_exist.sum(axis=0)
        exist_rate = exist_counts / num_bh
        print("层存在统计：")
        for i, name in enumerate(layer_order):
            print(f"  {name}: count={exist_counts[i]:.0f}, rate={exist_rate[i]:.2f}")
        print("y_mask 每层非零计数:", y_mask.sum(axis=0))

        # 5) 标准化特征与坐标
        if self.normalize_coords:
            coords_norm = self.coord_scaler.fit_transform(coords)
        else:
            coords_norm = coords

        if self.normalize_features:
            node_features = np.nan_to_num(node_features, nan=0.0)
            node_features_norm = self.feature_scaler.fit_transform(node_features)
        else:
            node_features_norm = node_features

        x_tensor = torch.tensor(node_features_norm, dtype=torch.float32)

        # 6) 构图（KNN 默认使用 x,y 平面坐标）
        edge_index, edge_weight = self.build_graph(coords)
        edge_attr = edge_weight.view(-1, 1)

        y_thick_t = torch.tensor(y_thick, dtype=torch.float32)
        y_exist_t = torch.tensor(y_exist, dtype=torch.float32)
        y_mask_t = torch.tensor(y_mask, dtype=torch.float32)

        n_nodes = num_bh
        indices = np.arange(n_nodes)
        fold_indices = []
        if split_mode == 'kfold':
            # 仅划分训练+验证，保留 test_size 比例作为独立测试
            train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            for train_sub, val_sub in kf.split(train_val_idx):
                fold_indices.append((train_val_idx[train_sub], train_val_idx[val_sub]))
            # 默认使用第一折作为当前 data 的 mask，便于单次训练
            train_idx, val_idx = fold_indices[0]
        else:
            train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
            val_ratio = val_size / (1 - test_size)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, random_state=random_seed)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            y_thick=y_thick_t,
            y_exist=y_exist_t,
            y_mask=y_mask_t,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            coords=torch.tensor(coords, dtype=torch.float32),
            borehole_id=bh_ids,
            layer_order=layer_order
        )

        return {
            'data': data,
            'num_features': x_tensor.shape[1],
            'num_layers': num_layers,
            'layer_order': layer_order,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'fold_indices': fold_indices,
            'exist_rate': exist_rate,
            'raw_df': df_layers
        }

    def standardize_lithology(self, df: pd.DataFrame, merge_coal: bool = True) -> pd.DataFrame:
        """
        标准化岩性名称 (处理乱码、统一命名格式)

        Args:
            df: 包含lithology列的DataFrame
            merge_coal: 是否合并煤层（默认True，用于GNN分类训练以减少类别数）
                        设为False时保留独立煤层名称（用于层序建模）

        Returns:
            df: 标准化后的DataFrame
        """
        # 保存merge_coal标志供内部函数使用
        _merge_coal = merge_coal

        # 创建一个清理后的岩性列
        def clean_lithology(name):
            if pd.isna(name):
                return '未知'

            name = str(name).strip()

            # ========== 修复乱码 ==========
            # 常见乱码映射（GBK读取UTF-8或反过来导致的乱码）
            garbled_fixes = {
                'ú': '煤',
                'ϸɰ��': '细砂岩',
                '��ɰ��': '粉砂岩',
                'ɰ������': '砂质泥岩',
                '̿������': '炭质泥岩',
                '��ֳ��': '腐殖土',
                'ɰ����': '砂砾岩',
                '����': '泥岩',
                '������': '砾岩',
                '��ɰ': '粉砂',
                'ϸɰ': '细砂',
                '̿��': '炭质',
            }

            # 尝试修复乱码
            for garbled, fix in garbled_fixes.items():
                if garbled in name:
                    name = name.replace(garbled, fix)

            # ========== 煤层处理 ==========
            if '煤' in name:
                if _merge_coal:
                    # 合并所有煤层为单一类别（用于GNN分类，减少类别数提高准确率）
                    return '煤'
                else:
                    # 保留独立煤层编号（用于层序建模）
                    return name.strip()

            # ========== 非煤岩性标准化 ==========
            # 砂岩类 - 保留粒度区分
            if '砂岩' in name or ('砂' in name and '砾' not in name and '泥' not in name):
                if '粉' in name:
                    return '粉砂岩'
                elif '细' in name:
                    return '细砂岩'
                elif '中' in name:
                    return '中砂岩'
                elif '粗' in name:
                    return '粗砂岩'
                elif '质' in name and '泥' in name:
                    return '砂质泥岩'
                else:
                    return '砂岩'

            # 砾岩类 - 保留粒度区分
            if '砾岩' in name or '砾' in name:
                if '砂' in name:
                    return '砂砾岩'
                elif '细' in name:
                    return '细砾岩'
                elif '中' in name:
                    return '中砾岩'
                elif '粗' in name:
                    return '粗砾岩'
                else:
                    return '砾岩'

            # 泥岩类
            if '泥岩' in name or '泥' in name:
                if '炭' in name or '碳' in name:
                    return '炭质泥岩'
                elif '砂' in name:
                    return '砂质泥岩'
                else:
                    return '泥岩'

            # 其他岩性
            if '腐殖' in name or '表土' in name:
                return '腐殖土'
            elif '粘土' in name or '黏土' in name or '黄土' in name:
                return '粘土'
            elif '页岩' in name:
                return '页岩'
            elif '灰岩' in name:
                return '灰岩'

            # 无法识别的保留原名
            return name

        df = df.copy()
        df['lithology'] = df['lithology'].apply(clean_lithology)

        print(f"\n标准化后的岩性类别 ({df['lithology'].nunique()}种):")
        value_counts = df['lithology'].value_counts()
        # 分类显示：煤层和非煤层
        coal_types = [v for v in value_counts.index if '煤' in str(v)]
        non_coal_types = [v for v in value_counts.index if '煤' not in str(v)]

        if coal_types:
            print(f"  煤层 ({len(coal_types)}种):")
            for ct in coal_types:
                print(f"    {ct}: {value_counts[ct]}个采样点")

        if non_coal_types:
            print(f"  非煤岩层 ({len(non_coal_types)}种):")
            for nct in non_coal_types:
                print(f"    {nct}: {value_counts[nct]}个采样点")

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
        val_size: float = 0.1,
        merge_coal: bool = True
    ) -> Dict:
        """
        处理钻孔数据并创建图数据集

        Args:
            df: 钻孔数据DataFrame
            label_col: 岩性标签列名
            feature_cols: 要使用的特征列 (None则使用所有可用的数值特征)
            test_size: 测试集比例
            val_size: 验证集比例

        Args:
            merge_coal: 是否合并所有煤层为单一类别；为 False 时保留独立煤层编号

        Returns:
            data_dict: 包含训练/验证/测试数据的字典
        """
        # 标准化岩性名称
        df = self.standardize_lithology(df, merge_coal=merge_coal)

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
        
        # ========== 特征工程增强 ==========
        # 添加层序特征（如果存在）
        engineered_features = []
        
        if 'layer_order' in df.columns:
            # 归一化层序号
            layer_order = df['layer_order'].values.astype(np.float32)
            layer_order_norm = (layer_order - layer_order.min()) / (layer_order.max() - layer_order.min() + 1e-8)
            engineered_features.append(layer_order_norm.reshape(-1, 1))
        
        if 'thickness' in df.columns:
            # 厚度特征（已包含在feature_cols中，但添加log变换）
            thickness = df['thickness'].fillna(df['thickness'].median()).values
            log_thickness = np.log1p(thickness).reshape(-1, 1)  # log(1+x)变换
            engineered_features.append(log_thickness)
        
        # 空间衍生特征
        # 1. 到质心的距离
        centroid = coords.mean(axis=0)
        dist_to_center = np.linalg.norm(coords - centroid, axis=1).reshape(-1, 1)
        engineered_features.append(dist_to_center)
        
        # 2. 深度分段特征 (z轴离散化)
        z_coords = coords[:, 2]
        z_bins = np.percentile(z_coords, [0, 25, 50, 75, 100])
        z_binned = np.digitize(z_coords, z_bins).astype(np.float32).reshape(-1, 1)
        engineered_features.append(z_binned)
        
        # 合并所有特征
        if engineered_features:
            features = np.concatenate([features] + engineered_features, axis=1)
            print(f"添加工程特征后，特征维度: {features.shape[1]}")

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

        # 检查是否可以使用分层采样（每个类至少需要2个样本）
        unique_labels, label_counts = np.unique(labels_encoded, return_counts=True)
        min_count = label_counts.min()
        can_stratify = min_count >= 2

        if not can_stratify:
            # 有些类别样本太少，不使用分层采样
            rare_classes = [self.lithology_classes[i] for i, c in enumerate(label_counts) if c < 2]
            print(f"\n警告: 以下岩性样本太少，无法分层采样: {rare_classes}")
            print("使用随机采样代替分层采样")
            stratify_labels = None
        else:
            stratify_labels = labels_encoded

        # 先划分出测试集
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=stratify_labels
        )

        # 再从训练+验证集中划分验证集
        val_ratio = val_size / (1 - test_size)
        if can_stratify:
            # 检查训练+验证集中是否仍可分层
            train_val_labels = labels_encoded[train_val_idx]
            _, tv_counts = np.unique(train_val_labels, return_counts=True)
            can_stratify_tv = tv_counts.min() >= 2
            stratify_tv = train_val_labels if can_stratify_tv else None
        else:
            stratify_tv = None

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio, random_state=42,
            stratify=stratify_tv
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
