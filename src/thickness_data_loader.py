"""
钻孔数据处理模块 (重构版)

核心职责：
1. 加载原始钻孔CSV数据
2. 构建层表数据（每层一行，保留层序信息）
3. 为GNN厚度预测任务准备图数据

重要改变：
- 不再对层内进行密集采样（那是分类任务的做法）
- 保留完整的层序信息用于厚度回归
- 支持传统插值和GNN两种建模方式
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree, Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List, Dict
import os
import glob
import json


class BoreholeDataLoader:
    """
    钻孔数据加载器

    负责从CSV文件加载原始钻孔数据
    """

    @staticmethod
    def load_csv(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
        """加载CSV文件，自动检测编码"""
        encodings_to_try = [encoding] if encoding else ['utf-8-sig', 'gbk', 'utf-8', 'gb2312', 'latin-1']
        last_error = None

        for enc in encodings_to_try:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise RuntimeError(f"读取文件失败: {file_path} -> {e}")

        raise RuntimeError(f"无法解析文件编码: {file_path} -> {last_error}")

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df = df.copy()
        df.columns = df.columns.str.strip()

        rename_map = {
            '序号(从下到上)': '序号',
            '厚度/m': '厚度',
            '弹性模量/Gpa': '弹性模量',
            '弹性模量/GPa': '弹性模量',
            '容重/kN*m-3': '容重',
            '容重/kN·m-3': '容重',
            '抗拉强度/MPa': '抗拉强度',
            'Unnamed: 6': '备注',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return df

    @staticmethod
    def load_coordinates(coord_file: str) -> pd.DataFrame:
        """加载钻孔坐标文件"""
        df = BoreholeDataLoader.load_csv(coord_file)
        df.columns = df.columns.str.strip()

        # 识别并重命名列
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if '钻孔' in col or 'name' in col_lower or 'id' in col_lower or '名' in col:
                col_mapping[col] = 'borehole_id'
            elif 'x' in col_lower or ('坐标' in col and 'x' in col.lower()):
                col_mapping[col] = 'x'
            elif 'y' in col_lower or ('坐标' in col and 'y' in col.lower()):
                col_mapping[col] = 'y'
            elif '高程' in col or 'elevation' in col_lower or 'z' in col_lower:
                col_mapping[col] = 'surface_z'

        df = df.rename(columns=col_mapping)

        required_cols = ['borehole_id', 'x', 'y']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"坐标文件缺少必需列: {missing_cols}")

        return df


class LayerTableProcessor:
    """
    层表处理器

    将原始钻孔数据转换为层表格式（每层一行）
    这是厚度预测任务的标准数据格式
    """

    def __init__(self, merge_coal: bool = False):
        """
        初始化

        Args:
            merge_coal: 是否合并所有煤层为单一类别
                - True: 用于GNN分类训练（减少类别数）
                - False: 用于层序建模（保留独立煤层）
        """
        self.merge_coal = merge_coal

    def load_single_borehole(
        self,
        borehole_file: str,
        borehole_id: str,
        x: float,
        y: float,
        surface_z: float = 0.0
    ) -> pd.DataFrame:
        """
        加载单个钻孔的层表

        Args:
            borehole_file: 钻孔数据文件路径
            borehole_id: 钻孔编号
            x, y: 平面坐标
            surface_z: 地表高程

        Returns:
            层表DataFrame，每行代表一层
        """
        if not os.path.exists(borehole_file):
            return pd.DataFrame()

        df = BoreholeDataLoader.load_csv(borehole_file)
        df = BoreholeDataLoader.standardize_columns(df)
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
            elif '容重' in col or 'density' in col_lower:
                col_mapping[col] = 'density'
            elif '抗拉强度' in col or 'tensile' in col_lower:
                col_mapping[col] = 'tensile_strength'

        df = df.rename(columns=col_mapping)

        # 检查必需列
        if 'lithology' not in df.columns or 'thickness' not in df.columns:
            print(f"警告: {borehole_file} 缺少必要列(lithology/thickness), 跳过")
            return pd.DataFrame()

        # 保持原始行序作为层序（从顶到底或从底到顶）
        df = df.reset_index(drop=True)
        df['layer_order'] = np.arange(len(df), dtype=np.int32)

        # 计算深度
        df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce').fillna(0)
        cumulative_depth = df['thickness'].cumsum()
        df['top_depth'] = cumulative_depth.shift(1).fillna(0)
        df['bottom_depth'] = cumulative_depth
        df['center_depth'] = (df['top_depth'] + df['bottom_depth']) / 2

        # 添加钻孔信息
        df['borehole_id'] = borehole_id
        df['x'] = x
        df['y'] = y
        df['surface_z'] = surface_z

        return df

    def load_all_boreholes(
        self,
        data_dir: str,
        coord_file: Optional[str] = None,
        surface_elevation: float = 0.0
    ) -> pd.DataFrame:
        """
        加载目录内所有钻孔的层表

        Returns:
            合并后的层表DataFrame
        """
        # 查找坐标文件
        if coord_file is None:
            coord_files = glob.glob(os.path.join(data_dir, '*坐标*.csv'))
            if coord_files:
                coord_file = coord_files[0]
            else:
                raise FileNotFoundError("未找到坐标文件")

        coord_df = BoreholeDataLoader.load_coordinates(coord_file)
        coord_map = {
            str(row['borehole_id']).strip(): (row['x'], row['y'])
            for _, row in coord_df.iterrows()
        }

        # 查找钻孔文件
        all_csv = glob.glob(os.path.join(data_dir, '*.csv'))
        borehole_files = [f for f in all_csv if '坐标' not in os.path.basename(f)]

        all_layers = []
        loaded_count = 0

        for bh_file in borehole_files:
            bh_id = os.path.splitext(os.path.basename(bh_file))[0]

            if bh_id not in coord_map:
                print(f"警告: 钻孔 {bh_id} 无坐标信息，跳过")
                continue

            x, y = coord_map[bh_id]
            df_layers = self.load_single_borehole(bh_file, bh_id, x, y, surface_elevation)

            if not df_layers.empty:
                all_layers.append(df_layers)
                loaded_count += 1

        if not all_layers:
            raise ValueError("未加载任何钻孔层表")

        combined = pd.concat(all_layers, ignore_index=True)
        print(f"层表加载完成: {loaded_count} 个钻孔, {len(combined)} 层")

        return combined

    def standardize_lithology(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化岩性名称

        处理乱码、统一命名格式、可选合并煤层
        """
        _merge_coal = self.merge_coal

        def clean_lithology(name):
            if pd.isna(name):
                return '未知'

            name = str(name).strip()

            # 修复乱码
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
            }
            for garbled, fix in garbled_fixes.items():
                if garbled in name:
                    name = name.replace(garbled, fix)

            # 煤层处理
            if '煤' in name:
                if _merge_coal:
                    return '煤'
                else:
                    return name.strip()

            # 砂岩类
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

            # 砾岩类
            if '砾岩' in name or '砾' in name:
                if '砂' in name:
                    return '砂砾岩'
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

            # 其他
            if '腐殖' in name or '表土' in name:
                return '腐殖土'
            elif '粘土' in name or '黏土' in name:
                return '粘土'
            elif '页岩' in name:
                return '页岩'
            elif '灰岩' in name:
                return '灰岩'

            return name

        df = df.copy()
        df['lithology'] = df['lithology'].apply(clean_lithology)

        print(f"标准化后的岩性类别: {df['lithology'].nunique()} 种")
        for lith, count in df['lithology'].value_counts().items():
            print(f"  {lith}: {count} 层")

        return df


class ThicknessDataBuilder:
    """
    厚度预测数据构建器

    将层表数据转换为GNN训练所需的图数据格式
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        graph_type: str = 'knn',
        normalize: bool = True
    ):
        self.k_neighbors = k_neighbors
        self.graph_type = graph_type
        self.normalize = normalize

        self.coord_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

    def infer_layer_order(self, df_layers: pd.DataFrame, method: str = 'position_based', min_occurrence_rate: float = 0.3) -> List[str]:
        """
        自动推断全局层序

        Args:
            df_layers: 层表数据
            method: 推断方法
                - 'simple': 按岩性名称合并（旧方法，会丢失重复层）
                - 'position_based': 按位置区分同名层（推荐）
                - 'marker_based': 以煤层为标志层对齐（最准确，但需要煤层数据）
            min_occurrence_rate: 最小出现率阈值 (仅对position_based有效)

        Returns:
            层序列表（从底到顶）
        """
        if method == 'simple':
            return self._infer_layer_order_simple(df_layers)
        elif method == 'position_based':
            return self._infer_layer_order_position_based(df_layers, min_occurrence_rate)
        elif method == 'marker_based':
            return self._infer_layer_order_marker_based(df_layers)
        else:
            raise ValueError(f"未知方法: {method}")

    def _infer_layer_order_simple(self, df_layers: pd.DataFrame) -> List[str]:
        """简单方法：按岩性名称合并（会丢失重复层）"""
        positions = {}
        for bh_id, sub in df_layers.groupby('borehole_id'):
            for idx, lith in zip(sub['layer_order'].values, sub['lithology'].values):
                positions.setdefault(lith, []).append(idx)

        layer_order = sorted(positions.keys(), key=lambda k: np.median(positions[k]))

        print("简单层序推断（从底到顶）- 警告：同种岩性被合并!")
        for i, name in enumerate(layer_order):
            print(f"  {i}: {name} (出现{len(positions[name])}次)")

        return layer_order

    def _infer_layer_order_position_based(self, df_layers: pd.DataFrame, min_occurrence_rate: float = 0.3) -> List[str]:
        """
        按位置区分同名层的方法

        Args:
            df_layers: 层表数据
            min_occurrence_rate: 最小出现率阈值 (0-1)，低于此值的层将被忽略

        逻辑：
        1. 统计每种岩性在各钻孔中的最大出现次数
        2. 为每次出现创建独立的层标识
        3. 按相对深度位置排序
        """
        print(f"\n[层序推断] 使用位置区分方法 (最小出现率={min_occurrence_rate*100:.0f}%)...")

        # 统计每种岩性在各钻孔中的出现次数
        lith_max_occurrences = {}  # {岩性: 最大出现次数}
        lith_positions = {}  # {岩性: {出现序号: [相对位置列表]}}

        total_boreholes = df_layers['borehole_id'].nunique()

        for bh_id, sub in df_layers.groupby('borehole_id'):
            sub = sub.sort_values('layer_order', ascending=True)
            total_layers = len(sub)

            lith_count = {}  # 当前钻孔中每种岩性的出现计数

            for idx, (_, row) in enumerate(sub.iterrows()):
                lith = row['lithology']
                relative_pos = idx / max(total_layers - 1, 1)  # 相对位置 0-1

                # 计算该岩性的出现序号
                if lith not in lith_count:
                    lith_count[lith] = 0
                occurrence = lith_count[lith]
                lith_count[lith] += 1

                # 记录位置
                if lith not in lith_positions:
                    lith_positions[lith] = {}
                if occurrence not in lith_positions[lith]:
                    lith_positions[lith][occurrence] = []
                lith_positions[lith][occurrence].append(relative_pos)

            # 更新最大出现次数
            for lith, count in lith_count.items():
                lith_max_occurrences[lith] = max(lith_max_occurrences.get(lith, 0), count)

        # 创建层列表，每个(岩性, 出现序号)为一个独立层
        layer_list = []  # [(层名, 平均相对位置, 出现次数)]
        min_count = max(2, int(total_boreholes * min_occurrence_rate))

        for lith, max_occ in lith_max_occurrences.items():
            for occ in range(max_occ):
                if occ in lith_positions.get(lith, {}):
                    positions = lith_positions[lith][occ]
                    avg_pos = np.median(positions)
                    occurrence_count = len(positions)

                    # 只有当出现次数足够多时才作为独立层
                    if occurrence_count >= min_count:
                        if max_occ > 1:
                            layer_name = f"{lith}_{occ+1}"  # 粉砂岩_1, 粉砂岩_2, ...
                        else:
                            layer_name = lith  # 只出现一次的不加后缀
                        layer_list.append((layer_name, avg_pos, occurrence_count))

        # 按相对位置排序（从顶到底 -> 从底到顶）
        layer_list.sort(key=lambda x: x[1])  # 按位置升序（顶部=0，底部=1）

        layer_order = [item[0] for item in layer_list]

        print(f"\n[层序推断] 推断出 {len(layer_order)} 层（从顶到底）:")
        for i, (name, pos, count) in enumerate(layer_list):
            print(f"  {i}: {name} (相对位置={pos:.2f}, 出现{count}次, {count*100/total_boreholes:.0f}%)")

        return layer_order

    def _infer_layer_order_marker_based(self, df_layers: pd.DataFrame) -> List[str]:
        """
        以煤层为标志层的对齐方法

        逻辑：
        1. 识别所有煤层作为标志层
        2. 以煤层为锚点，计算其他层相对于最近煤层的位置
        3. 构建相对于标志层的层序
        """
        print("\n[层序推断] 使用煤层标志法...")

        # 提取所有唯一的煤层名称
        all_coal_layers = df_layers[df_layers['lithology'].str.contains('煤', na=False)]['lithology'].unique()
        coal_layers = sorted(set(all_coal_layers))

        if len(coal_layers) == 0:
            print("警告：未找到煤层，回退到位置法")
            return self._infer_layer_order_position_based(df_layers)

        print(f"  识别到 {len(coal_layers)} 个煤层标志: {coal_layers[:10]}...")

        # 分析煤层的相对位置
        coal_positions = {}  # {煤层名: [相对位置列表]}

        for bh_id, sub in df_layers.groupby('borehole_id'):
            sub = sub.sort_values('layer_order', ascending=True)
            total_depth = sub['bottom_depth'].max()

            for _, row in sub.iterrows():
                lith = row['lithology']
                if '煤' in lith:
                    rel_pos = row['center_depth'] / max(total_depth, 1)
                    if lith not in coal_positions:
                        coal_positions[lith] = []
                    coal_positions[lith].append(rel_pos)

        # 按中位深度排序煤层
        coal_order = sorted(coal_positions.keys(),
                          key=lambda k: np.median(coal_positions[k]))

        print(f"  煤层顺序（从浅到深）: {coal_order[:10]}...")

        # 分析非煤层相对于煤层的位置
        non_coal_layers = {}  # {岩性: {位置类型: 计数}}

        for bh_id, sub in df_layers.groupby('borehole_id'):
            sub = sub.sort_values('layer_order', ascending=True)
            layers_list = sub['lithology'].tolist()
            depths = sub['center_depth'].tolist()

            for i, lith in enumerate(layers_list):
                if '煤' not in lith:
                    # 找最近的煤层
                    for j, coal in enumerate(layers_list):
                        if '煤' in coal:
                            if i < j:
                                key = f"上_{coal}"  # 在煤层上方
                            else:
                                key = f"下_{coal}"  # 在煤层下方
                            if lith not in non_coal_layers:
                                non_coal_layers[lith] = {}
                            non_coal_layers[lith][key] = non_coal_layers[lith].get(key, 0) + 1
                            break

        # 构建完整层序：使用位置法作为基础，但保留煤层的精确位置
        return self._infer_layer_order_position_based(df_layers)

    def build_graph(self, coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建图结构

        Args:
            coords: 节点坐标 [N, 2] 或 [N, 3]

        Returns:
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E]
        """
        N = len(coords)
        k = min(self.k_neighbors, N - 1)

        if self.graph_type == 'knn':
            tree = KDTree(coords)
            distances, indices = tree.query(coords, k=k + 1)

            edges = []
            weights = []
            for i in range(N):
                for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                    edges.append([i, j])
                    edges.append([j, i])
                    weights.extend([dist, dist])

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(weights, dtype=torch.float32)
            edge_weight = torch.exp(-edge_weight / (edge_weight.mean() + 1e-8))

        elif self.graph_type == 'delaunay':
            try:
                tri = Delaunay(coords[:, :2])
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            edges.add((simplex[i], simplex[j]))
                            edges.add((simplex[j], simplex[i]))

                edge_list = list(edges)
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                weights = []
                for src, dst in zip(edge_index[0], edge_index[1]):
                    dist = np.linalg.norm(coords[src.item()] - coords[dst.item()])
                    weights.append(dist)
                edge_weight = torch.tensor(weights, dtype=torch.float32)
                edge_weight = torch.exp(-edge_weight / (edge_weight.mean() + 1e-8))

            except Exception as e:
                print(f"Delaunay失败，回退到KNN: {e}")
                return self.build_graph(coords)
        else:
            raise ValueError(f"未知图类型: {self.graph_type}")

        return edge_index, edge_weight

    def build_thickness_data(
        self,
        df_layers: pd.DataFrame,
        layer_order: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42
    ) -> Dict:
        """
        构建厚度预测任务的图数据

        Args:
            df_layers: 层表数据
            layer_order: 地层顺序（None则自动推断）
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子

        Returns:
            包含PyG Data和元数据的字典
        """
        # 确定层序
        if layer_order is None:
            layer_order = self.infer_layer_order(df_layers)

        layer_to_idx = {name: i for i, name in enumerate(layer_order)}
        num_layers = len(layer_order)

        # 按钻孔聚合
        borehole_groups = list(df_layers.groupby('borehole_id'))
        num_bh = len(borehole_groups)

        # 初始化目标张量
        y_thick = np.zeros((num_bh, num_layers), dtype=np.float32)  # 厚度
        y_exist = np.zeros((num_bh, num_layers), dtype=np.float32)  # 存在性
        y_mask = np.ones((num_bh, num_layers), dtype=np.float32)    # 有效掩码

        node_features = []
        coords = []
        bh_ids = []

        # 检查层名是否有编号后缀（如 粉砂岩_1）
        has_numbered_layers = any('_' in name and name.split('_')[-1].isdigit() for name in layer_order)

        # 为df_layers添加位置标记的层名列
        layer_name_list = []
        df_layers_indexed = df_layers.copy()

        for idx, (bh_id, sub) in enumerate(borehole_groups):
            sub = sub.sort_values('layer_order', ascending=True)
            total_depth = sub['thickness'].sum()
            x = sub['x'].iloc[0]
            y = sub['y'].iloc[0]
            surface_z = sub.get('surface_z', pd.Series([0])).iloc[0]

            coords.append([x, y])
            bh_ids.append(bh_id)

            # 填充各层厚度 + 记录位置标记的层名
            if has_numbered_layers:
                # 带编号的层：需要跟踪每种岩性的出现次数
                lith_occurrence = {}  # {岩性: 当前出现次数}

                for row_idx, row in sub.iterrows():
                    lith = row['lithology']

                    # 计算当前出现序号
                    if lith not in lith_occurrence:
                        lith_occurrence[lith] = 0
                    occ = lith_occurrence[lith]
                    lith_occurrence[lith] += 1

                    # 尝试匹配带编号的层名
                    layer_name_with_num = f"{lith}_{occ+1}"
                    layer_name_without_num = lith

                    if layer_name_with_num in layer_to_idx:
                        li = layer_to_idx[layer_name_with_num]
                        y_thick[idx, li] = float(row['thickness'])
                        y_exist[idx, li] = 1.0
                        layer_name_list.append((row_idx, layer_name_with_num))
                    elif layer_name_without_num in layer_to_idx:
                        li = layer_to_idx[layer_name_without_num]
                        y_thick[idx, li] = float(row['thickness'])
                        y_exist[idx, li] = 1.0
                        layer_name_list.append((row_idx, layer_name_without_num))
                    else:
                        layer_name_list.append((row_idx, layer_name_with_num))
            else:
                # 无编号的层：直接匹配岩性名称
                for row_idx, row in sub.iterrows():
                    lith = row['lithology']
                    layer_name_list.append((row_idx, lith))
                    if lith not in layer_to_idx:
                        continue
                    li = layer_to_idx[lith]
                    y_thick[idx, li] = float(row['thickness'])
                    y_exist[idx, li] = 1.0

            # 增强特征：钻孔统计信息
            num_layers_in_bh = len(sub)
            avg_thickness = sub['thickness'].mean()
            std_thickness = sub['thickness'].std() if len(sub) > 1 else 0.0
            max_thickness = sub['thickness'].max()
            min_thickness = sub['thickness'].min()

            # 岩性多样性
            lith_diversity = sub['lithology'].nunique() / num_layers

            # 各类岩石的厚度占比
            coal_ratio = sub[sub['lithology'].str.contains('煤', na=False)]['thickness'].sum() / max(total_depth, 1)
            sandstone_ratio = sub[sub['lithology'].str.contains('砂岩', na=False)]['thickness'].sum() / max(total_depth, 1)
            mudstone_ratio = sub[sub['lithology'].str.contains('泥岩', na=False)]['thickness'].sum() / max(total_depth, 1)

            # 钻孔特征：增强版（15个特征）
            node_features.append([
                x, y,  # 位置
                total_depth,  # 总深度
                surface_z,  # 地表高程
                num_layers_in_bh,  # 层数
                avg_thickness,  # 平均厚度
                std_thickness,  # 厚度标准差
                max_thickness,  # 最大厚度
                min_thickness,  # 最小厚度
                lith_diversity,  # 岩性多样性
                coal_ratio,  # 煤层厚度占比
                sandstone_ratio,  # 砂岩厚度占比
                mudstone_ratio,  # 泥岩厚度占比
                total_depth / max(num_layers_in_bh, 1),  # 平均层厚
                max_thickness / max(min_thickness, 0.1)  # 厚度变异系数
            ])

        # 将位置标记的层名添加到df_layers
        layer_name_df = pd.DataFrame(layer_name_list, columns=['_idx', 'layer_name'])
        layer_name_df = layer_name_df.set_index('_idx')
        df_layers_indexed['layer_name'] = df_layers_indexed.index.map(layer_name_df['layer_name'])

        node_features = np.array(node_features, dtype=np.float32)
        coords = np.array(coords, dtype=np.float32)

        # 统计
        exist_counts = y_exist.sum(axis=0)
        exist_rate = exist_counts / num_bh
        print("\n各层存在统计:")
        for i, name in enumerate(layer_order):
            print(f"  {name}: {int(exist_counts[i])}/{num_bh} ({exist_rate[i]*100:.1f}%)")

        # 标准化
        if self.normalize:
            coords_norm = self.coord_scaler.fit_transform(coords)
            node_features = np.nan_to_num(node_features, nan=0.0)
            features_norm = self.feature_scaler.fit_transform(node_features)
        else:
            coords_norm = coords
            features_norm = node_features

        x_tensor = torch.tensor(features_norm, dtype=torch.float32)

        # 构图
        edge_index, edge_weight = self.build_graph(coords)
        edge_attr = edge_weight.view(-1, 1)

        # 转换目标为张量
        y_thick_t = torch.tensor(y_thick, dtype=torch.float32)
        y_exist_t = torch.tensor(y_exist, dtype=torch.float32)
        y_mask_t = torch.tensor(y_mask, dtype=torch.float32)

        # 数据划分
        n_nodes = num_bh
        indices = np.arange(n_nodes)

        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_seed
        )
        val_ratio = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio, random_state=random_seed
        )

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # 创建PyG Data
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

        print(f"\n数据集构建完成:")
        print(f"  钻孔数: {num_bh}")
        print(f"  地层数: {num_layers}")
        print(f"  特征维度: {x_tensor.shape[1]}")
        print(f"  训练集: {train_mask.sum().item()}")
        print(f"  验证集: {val_mask.sum().item()}")
        print(f"  测试集: {test_mask.sum().item()}")

        return {
            'data': data,
            'num_features': x_tensor.shape[1],
            'num_layers': num_layers,
            'layer_order': layer_order,
            'layer_to_idx': layer_to_idx,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'exist_rate': exist_rate,
            'raw_df': df_layers_indexed,  # 包含 layer_name 列
            'borehole_coords': coords,
            'borehole_ids': bh_ids
        }


class ThicknessDataProcessor:
    """
    厚度预测数据处理器（统一接口）

    整合数据加载、预处理、图构建的完整流程
    """

    def __init__(
        self,
        merge_coal: bool = False,
        k_neighbors: int = 8,
        graph_type: str = 'knn',
        normalize: bool = True
    ):
        """
        初始化

        Args:
            merge_coal: 是否合并煤层
            k_neighbors: KNN邻居数
            graph_type: 图类型 ('knn' 或 'delaunay')
            normalize: 是否标准化特征
        """
        self.layer_processor = LayerTableProcessor(merge_coal=merge_coal)
        self.data_builder = ThicknessDataBuilder(
            k_neighbors=k_neighbors,
            graph_type=graph_type,
            normalize=normalize
        )

    def process_directory(
        self,
        data_dir: str,
        coord_file: Optional[str] = None,
        layer_order: Optional[List[str]] = None,
        layer_method: str = 'position_based',
        min_occurrence_rate: float = 0.3,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        处理数据目录

        Args:
            data_dir: 数据目录
            coord_file: 坐标文件路径
            layer_order: 地层顺序（None则自动推断）
            layer_method: 层序推断方法 ('simple', 'position_based', 'marker_based')
            min_occurrence_rate: 最小出现率阈值 (仅对position_based有效)
            test_size: 测试集比例
            val_size: 验证集比例

        Returns:
            处理结果字典
        """
        # 加载层表
        df_layers = self.layer_processor.load_all_boreholes(data_dir, coord_file)

        # 标准化岩性
        df_layers = self.layer_processor.standardize_lithology(df_layers)

        # 如果未指定层序，使用指定方法推断
        if layer_order is None:
            layer_order = self.data_builder.infer_layer_order(
                df_layers,
                method=layer_method,
                min_occurrence_rate=min_occurrence_rate
            )

        # 构建图数据
        result = self.data_builder.build_thickness_data(
            df_layers=df_layers,
            layer_order=layer_order,
            test_size=test_size,
            val_size=val_size
        )

        return result

    def save_metadata(self, result: Dict, path: str):
        """保存元数据"""
        metadata = {
            'num_features': result['num_features'],
            'num_layers': result['num_layers'],
            'layer_order': result['layer_order'],
            'exist_rate': result['exist_rate'].tolist(),
            'num_boreholes': len(result['borehole_ids']),
            'borehole_ids': result['borehole_ids']
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_metadata(self, path: str) -> Dict:
        """加载元数据"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("测试重构后的数据处理模块...")

    # 数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    if os.path.exists(data_dir):
        # 创建处理器
        processor = ThicknessDataProcessor(
            merge_coal=False,  # 保留独立煤层用于层序建模
            k_neighbors=8,
            graph_type='knn'
        )

        # 处理数据
        result = processor.process_directory(data_dir)

        print("\n处理结果:")
        print(f"  数据对象类型: {type(result['data'])}")
        print(f"  节点数: {result['data'].x.shape[0]}")
        print(f"  特征维度: {result['num_features']}")
        print(f"  地层数: {result['num_layers']}")
        print(f"  层序: {result['layer_order']}")
    else:
        print(f"数据目录不存在: {data_dir}")
        print("请创建data目录并放入钻孔数据文件")
