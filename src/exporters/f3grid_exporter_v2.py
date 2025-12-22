"""
FLAC3D f3grid 导出器 v2 - 正确的网格格式

根据 FLAC3D 官方文档实现的 f3grid ASCII 格式导出器。

格式规范 (来自 https://docs.itascacg.com):
- 注释行以 * 开头
- G <id> <x> <y> <z>  - 节点定义
- Z B8 <id> <gp1> ... <gp8>  - 八节点六面体单元定义
- ZGROUP '<name>'  - 单元分组定义
  <zone_id> <zone_id> ...

FLAC3D B8 节点顺序 (官方标准，见 "Orientation of Nodes and Faces within a Zone"):

    节点编号 1-8 的标准顺序：
    1=SW_bot, 2=SE_bot, 3=NW_bot, 4=SW_top, 5=NE_bot, 6=NW_top, 7=SE_top, 8=NE_top

    这是一个交织模式（不是先底面后顶面）：
    [1:SW_bot, 2:SE_bot, 3:NW_bot, 4:SW_top, 5:NE_bot, 6:NW_top, 7:SE_top, 8:NE_top]

    从顶视图看（Z轴向上）:

        NW(3,6) ---- NE(5,8)        y (北)
         |            |             ^
         |            |             |
        SW(1,4) ---- SE(2,7)        +---> x (东)
                                   /
                                  z (向上)

    注意：
    - 节点 1,2,5,3 构成底面 (Z较小)
    - 节点 4,7,8,6 构成顶面 (Z较大)
    - 底面顶点 1 对应顶面顶点 4 (SW)
    - 底面顶点 2 对应顶面顶点 7 (SE)
    - 底面顶点 3 对应顶面顶点 6 (NW)
    - 底面顶点 5 对应顶面顶点 8 (NE)

关键实现要点:
1. 层间节点共享 - 上层底面 = 下层顶面，保证应力连续传递
2. 正确的 B8 节点顺序 - 按照 FLAC3D 官方标准 (交织模式)
3. 单元质量检查 - 过滤退化单元
4. 坐标精度控制 - 避免浮点误差
"""

import os
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# 几何检查辅助函数
# ============================================================

def tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """
    计算四面体体积

    Args:
        a, b, c, d: 四面体的4个顶点坐标 (3D向量)

    Returns:
        四面体的有向体积（正值表示正确方向）
    """
    return np.dot(np.cross(b - a, c - a), d - a) / 6.0


def check_hex_geometry(coords: np.ndarray, tolerance: float = 1e-9) -> Tuple[bool, float]:
    """
    检查六面体（B8）几何是否有效

    使用简化的厚度检查：验证顶面在底面之上，每个角点厚度为正。
    这比四面体分解更宽松，避免过度过滤有效单元。

    Args:
        coords: 8x3 数组，B8 单元的 8 个节点坐标（按 FLAC3D 标准顺序）
        tolerance: 厚度容差（默认 1e-9）

    Returns:
        (is_valid, min_thickness): 是否有效，最小厚度
    """
    # FLAC3D B8 节点顺序 (0-indexed):
    # 0=SW_bot, 1=SE_bot, 2=NW_bot, 3=SW_top, 4=NE_bot, 5=NW_top, 6=SE_top, 7=NE_top
    # 底面节点: 0, 1, 2, 4 (z值较小)
    # 顶面节点: 3, 5, 6, 7 (z值较大)
    # 对应关系: 0<->3(SW), 1<->6(SE), 2<->5(NW), 4<->7(NE)

    # 检查每个角点的厚度（顶面z - 底面z）
    thickness_sw = coords[3, 2] - coords[0, 2]  # SW: top(3) - bottom(0)
    thickness_se = coords[6, 2] - coords[1, 2]  # SE: top(6) - bottom(1)
    thickness_nw = coords[5, 2] - coords[2, 2]  # NW: top(5) - bottom(2)
    thickness_ne = coords[7, 2] - coords[4, 2]  # NE: top(7) - bottom(4)

    min_thickness = min(thickness_sw, thickness_se, thickness_nw, thickness_ne)
    is_valid = min_thickness > tolerance

    return is_valid, min_thickness


@dataclass
class GridPoint:
    """网格节点"""
    id: int
    x: float
    y: float
    z: float


@dataclass
class BrickZone:
    """B8 六面体单元"""
    id: int
    gp_ids: List[int]  # 8个节点 ID，按 FLAC3D 标准顺序
    group: str


@dataclass
class ExportStats:
    """导出统计"""
    total_gridpoints: int = 0
    total_zones: int = 0
    shared_nodes: int = 0
    degenerate_zones_removed: int = 0
    groups: int = 0
    min_thickness: float = float('inf')
    max_thickness: float = 0.0
    file_size_kb: float = 0.0
    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 坐标范围（原始坐标）
    coord_range_x: Tuple[float, float] = (0.0, 0.0)
    coord_range_y: Tuple[float, float] = (0.0, 0.0)
    coord_range_z: Tuple[float, float] = (0.0, 0.0)
    # 模型尺寸
    model_size: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class F3GridExporterV2:
    """
    FLAC3D f3grid 导出器 v2

    正确实现层间节点共享和 B8 单元顺序。

    使用方法:
        exporter = F3GridExporterV2()
        exporter.export(data, "model.f3grid", options)

    在 FLAC3D 中导入:
        zone import f3grid "model.f3grid"
    """

    # 坐标精度（小数位数），用于节点合并判断
    COORD_PRECISION = 6

    # 最小单元厚度（米），低于此值视为退化单元
    MIN_ZONE_THICKNESS = 0.001

    def __init__(self):
        self.gridpoints: List[GridPoint] = []
        self.zones: List[BrickZone] = []
        self.groups: Dict[str, List[int]] = {}  # group_name -> [zone_ids]

        self._next_gp_id = 1
        self._next_zone_id = 1

        # 坐标 -> 节点ID 映射，用于节点共享
        self._coord_to_gp: Dict[Tuple[float, float, float], int] = {}

        self.stats = ExportStats()

    def export(self, data: Dict[str, Any], output_path: str,
               options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出地质模型为 FLAC3D f3grid 格式

        Args:
            data: 地层数据字典
                {
                    "layers": [
                        {
                            "name": str,
                            "grid_x": 1D array (nx,) 或 2D array (ny, nx),
                            "grid_y": 1D array (ny,) 或 2D array (ny, nx),
                            "top_surface_z": 2D array (ny, nx),
                            "bottom_surface_z": 2D array (ny, nx)
                        },
                        ...  # 从下到上排序
                    ]
                }
            output_path: 输出文件路径 (.f3grid)
            options: 导出选项
                - downsample_factor: int, 降采样倍数 (默认 1)
                - coal_downsample_factor: int, 煤层降采样倍数 (默认同downsample_factor)
                - coal_adjacent_layers: int, 煤层上下相邻层数使用高密度 (默认 1)
                - selected_coal_layers: list, 需要高密度的煤层索引列表 (默认None表示所有煤层)
                - min_zone_thickness: float, 最小单元厚度 (默认 0.001)
                - coord_precision: int, 坐标精度 (默认 6)
                - check_overlap: bool, 检查层间重叠 (默认 True)
                - create_interfaces: bool, 创建接触面模式 (默认 False，层间不共享节点)
                - uniform_downsample: bool, 使用统一降采样 (默认 False，建议启用以避免层间非共形)
                - merge_thin_layers: bool, 合并极薄层 (默认 False，建议启用以提高稳定性)
                - merge_thickness_threshold: float, 薄层合并阈值(米) (默认 0.5)
                - merge_same_lithology_only: bool, 仅合并同岩性层 (默认 True)
                - force_layer_continuity: bool, 强制层间贴合 (默认 True，确保层间无缝隙/重叠)
                - enforce_minimum_thickness: bool, 节点级最小厚度强制 (默认 True，确保每个位置厚度>=阈值)

        Returns:
            输出文件路径
        """
        options = options or {}
        downsample = max(1, int(options.get('downsample_factor', 1)))
        coal_downsample = max(1, int(options.get('coal_downsample_factor', downsample)))
        coal_adjacent = int(options.get('coal_adjacent_layers', 1))
        selected_coal_layers = options.get('selected_coal_layers', None)  # None表示所有煤层
        check_overlap = bool(options.get('check_overlap', True))
        create_interfaces = bool(options.get('create_interfaces', False))  # 接触面模式
        uniform_downsample = bool(options.get('uniform_downsample', False))  # 统一降采样模式

        # 新增：薄层合并选项
        merge_thin_layers = bool(options.get('merge_thin_layers', False))
        merge_threshold = float(options.get('merge_thickness_threshold', 0.5))
        merge_same_lithology = bool(options.get('merge_same_lithology_only', True))

        # 新增：层间强制贴合选项
        force_continuity = bool(options.get('force_layer_continuity', True))

        # 新增：节点级最小厚度强制选项
        enforce_min_thickness = bool(options.get('enforce_minimum_thickness', True))

        self.MIN_ZONE_THICKNESS = float(options.get('min_zone_thickness', 0.001))
        self.COORD_PRECISION = int(options.get('coord_precision', 6))
        self.create_interfaces = create_interfaces
        self.uniform_downsample = uniform_downsample

        # 重置状态
        self._reset()

        layers = data.get('layers', [])
        if not layers:
            raise ValueError("没有可导出的地层数据")

        print(f"\n{'='*60}")
        print(f"FLAC3D f3grid Exporter v2")
        print(f"{'='*60}")
        print(f"地层数量: {len(layers)} (原始)")

        # 1. 薄层合并（可选）
        if merge_thin_layers:
            print(f"\n--- 薄层合并 ---")
            print(f"  合并阈值: {merge_threshold}m")
            print(f"  仅合并同岩性: {'是' if merge_same_lithology else '否'}")
            layers = self._merge_thin_layers(
                layers, merge_threshold, merge_same_lithology
            )
            print(f"  合并后地层数: {len(layers)}")

        # 2. 层间强制贴合（可选，建议启用）
        if force_continuity:
            print(f"\n--- 层间强制贴合 ---")
            layers = self._enforce_layer_continuity(layers)
            print(f"  已确保层间无缝隙/重叠")

        # 3. 节点级最小厚度强制（可选，建议启用）
        if enforce_min_thickness:
            print(f"\n--- 节点级最小厚度强制 ---")
            layers = self._enforce_minimum_thickness_at_nodes(layers, self.MIN_ZONE_THICKNESS)
            print(f"  已确保所有节点位置厚度 >= {self.MIN_ZONE_THICKNESS}m")

        if uniform_downsample:
            # 统一降采样模式：所有层使用相同降采样率
            print(f"降采样: 统一 {downsample}x (所有地层)")
            print(f"  [强烈建议] 统一降采样确保层间完全共形，避免几何不连续")
        else:
            # 自适应降采样模式：煤层使用不同降采样率
            print(f"降采样: 常规地层 {downsample}x, 煤层区域 {coal_downsample}x")
            print(f"  ⚠️  警告: 层间不同分辨率可能导致非共形、缝隙或重叠")
        print(f"煤层相邻层范围: ±{coal_adjacent} 层")
        print(f"接触面模式: {'启用 (层间不共享节点)' if create_interfaces else '禁用 (层间共享节点)'}")
        print(f"输出文件: {output_path}")

        # 识别煤层及相邻层
        coal_indices, high_density_indices = self._identify_coal_layers(
            layers, coal_adjacent, selected_coal_layers
        )

        # 1. 生成所有层的网格，实现层间节点共享（或分离以创建接触面）
        layer_names = self._generate_all_layers(layers, downsample, coal_downsample, high_density_indices)

        # 2. 转换为相对坐标（以最小角点为原点）
        self._convert_to_relative_coordinates()

        # 3. 写入 f3grid 文件
        self._write_f3grid(output_path)

        # 4. 如果启用接触面模式，生成接触面脚本
        if create_interfaces:
            interface_script_path = output_path.replace('.f3grid', '_interfaces.fis')
            self._write_interface_script(interface_script_path, layer_names)

        # 5. 计算文件大小
        try:
            self.stats.file_size_kb = os.path.getsize(output_path) / 1024
        except:
            pass

        # 打印统计
        self._print_stats()

        print(f"\n[OK] 导出完成!")
        print(f"\n在 FLAC3D 中使用:")
        print(f'  zone import f3grid "{os.path.basename(output_path)}"')

        return output_path

    def _reset(self):
        """重置导出器状态"""
        self.gridpoints = []
        self.zones = []
        self.groups = {}
        self._next_gp_id = 1
        self._next_zone_id = 1
        self._coord_to_gp = {}
        self.stats = ExportStats()

    def _merge_thin_layers(self, layers: List[Dict], threshold: float,
                          same_lithology_only: bool = True) -> List[Dict]:
        """
        合并极薄层以提高模型稳定性

        策略：
        1. 计算每层的平均厚度
        2. 将连续的薄层（< threshold）合并为一层
        3. 如果 same_lithology_only=True，仅合并同岩性的薄层
        4. 合并后的层名为：原始层名（如"砂岩+泥岩+粉砂岩"）或相同岩性名
        5. 合并后的层保留最底部的 bottom_z 和最顶部的 top_z

        Args:
            layers: 原始地层列表（从下到上排序）
            threshold: 合并阈值（米），小于此值的层可能被合并
            same_lithology_only: 是否仅合并同岩性层

        Returns:
            合并后的地层列表
        """
        if len(layers) <= 1:
            return layers

        print(f"  开始分析地层厚度...")

        # 1. 计算每层的平均厚度
        layer_thicknesses = []
        for i, layer in enumerate(layers):
            top_z = np.asarray(layer['top_surface_z'], dtype=float)
            bottom_z = np.asarray(layer['bottom_surface_z'], dtype=float)
            valid_mask = ~(np.isnan(top_z) | np.isnan(bottom_z))
            if np.any(valid_mask):
                avg_thickness = np.mean((top_z - bottom_z)[valid_mask])
            else:
                avg_thickness = 0.0
            layer_thicknesses.append(avg_thickness)
            print(f"    [{i}] {layer.get('name', 'Unknown')}: 平均厚度 {avg_thickness:.3f}m")

        # 2. 识别需要合并的层组
        merged_layers = []
        i = 0
        merge_count = 0

        while i < len(layers):
            current_layer = layers[i]
            current_thickness = layer_thicknesses[i]
            current_name = current_layer.get('name', f'Layer_{i}')

            # 提取岩性名（去掉数字序号）
            current_lithology = self._sanitize_name(current_name)

            # 如果当前层厚度足够，直接保留
            if current_thickness >= threshold:
                merged_layers.append(current_layer.copy())
                i += 1
                continue

            # 当前层是薄层，尝试向上合并
            merge_group = [i]
            j = i + 1

            # 向上找连续的薄层
            while j < len(layers):
                next_layer = layers[j]
                next_thickness = layer_thicknesses[j]
                next_name = next_layer.get('name', f'Layer_{j}')
                next_lithology = self._sanitize_name(next_name)

                # 检查是否可以合并
                can_merge = next_thickness < threshold

                # 如果要求同岩性，还需检查岩性是否相同
                if same_lithology_only:
                    can_merge = can_merge and (next_lithology == current_lithology)

                if can_merge:
                    merge_group.append(j)
                    j += 1
                else:
                    break

            # 合并这一组薄层
            if len(merge_group) > 1:
                # 合并多层
                base_layer = layers[merge_group[0]].copy()
                top_layer = layers[merge_group[-1]]

                # 合并后的顶面 = 最上层的顶面
                base_layer['top_surface_z'] = top_layer['top_surface_z'].copy()

                # 合并后的层名
                if same_lithology_only:
                    # 同岩性合并，保留岩性名
                    base_layer['name'] = current_name
                else:
                    # 不同岩性合并，组合名称
                    merged_names = [layers[k].get('name', f'L{k}') for k in merge_group]
                    base_layer['name'] = '+'.join(merged_names[:3])  # 最多显示3个
                    if len(merge_group) > 3:
                        base_layer['name'] += f'+...({len(merge_group)}层)'

                merged_layers.append(base_layer)

                layer_names = [layers[k].get('name', f'L{k}') for k in merge_group]
                total_thickness = sum(layer_thicknesses[k] for k in merge_group)
                print(f"  ✓ 合并 {len(merge_group)} 层: {' + '.join(layer_names)} → 总厚度 {total_thickness:.3f}m")
                merge_count += len(merge_group) - 1

                i = j  # 跳到下一组
            else:
                # 单层，但太薄 - 仍然保留（避免空洞）
                merged_layers.append(current_layer.copy())
                print(f"  ⚠ 薄层 {current_name} ({current_thickness:.3f}m) 无法合并，保留")
                i += 1

        print(f"  合并统计: 原始 {len(layers)} 层 → 合并后 {len(merged_layers)} 层（减少 {merge_count} 层）")

        return merged_layers

    def _enforce_layer_continuity(self, layers: List[Dict]) -> List[Dict]:
        """
        强制层间连续性，确保上层的底面 = 下层的顶面

        这样可以避免：
        1. 层间缝隙（下层顶面低于上层底面）
        2. 层间重叠（下层顶面高于上层底面）

        策略：
        - 从下往上处理
        - 对于每一层（除了最底层），强制其 bottom_z = 下层的 top_z

        Args:
            layers: 地层列表（从下到上排序）

        Returns:
            修正后的地层列表
        """
        if len(layers) <= 1:
            return layers

        print(f"  处理 {len(layers)} 个地层的层间贴合...")

        fixed_layers = []
        gap_count = 0
        overlap_count = 0

        for i, layer in enumerate(layers):
            layer_copy = layer.copy()
            layer_copy['top_surface_z'] = np.asarray(layer['top_surface_z'], dtype=float).copy()
            layer_copy['bottom_surface_z'] = np.asarray(layer['bottom_surface_z'], dtype=float).copy()

            if i == 0:
                # 最底层，无需修正
                fixed_layers.append(layer_copy)
            else:
                # 上层：强制底面 = 下层顶面
                prev_top_z = fixed_layers[i-1]['top_surface_z']
                current_bottom_z = layer_copy['bottom_surface_z']

                # 检查原始状态
                valid_mask = ~(np.isnan(prev_top_z) | np.isnan(current_bottom_z))
                if np.any(valid_mask):
                    diff = current_bottom_z[valid_mask] - prev_top_z[valid_mask]
                    max_gap = np.max(diff[diff > 0]) if np.any(diff > 0) else 0
                    max_overlap = -np.min(diff[diff < 0]) if np.any(diff < 0) else 0

                    if max_gap > 0.01:
                        print(f"    [{i}] {layer.get('name', 'Unknown')}: 检测到缝隙，最大 {max_gap:.3f}m")
                        gap_count += 1
                    if max_overlap > 0.01:
                        print(f"    [{i}] {layer.get('name', 'Unknown')}: 检测到重叠，最大 {max_overlap:.3f}m")
                        overlap_count += 1

                # 强制贴合
                layer_copy['bottom_surface_z'] = prev_top_z.copy()

                fixed_layers.append(layer_copy)

        if gap_count > 0 or overlap_count > 0:
            print(f"  修正统计: {gap_count} 处缝隙, {overlap_count} 处重叠 → 已全部修正")
        else:
            print(f"  检查完成: 层间已连续，无需修正")

        return fixed_layers

    def _enforce_minimum_thickness_at_nodes(self, layers: List[Dict], min_thickness: float) -> List[Dict]:
        """
        强制节点级最小厚度，确保每个位置的厚度都不小于阈值

        策略：
        - 对每一层，确保 top_z >= bottom_z + min_thickness
        - 这会在节点层面强制每个位置厚度为正
        - 必须在层间贴合之后运行，以保持层间连续性的同时确保最小厚度
        - 只修改top_z，不修改bottom_z，因此不会破坏层间连续性

        Args:
            layers: 地层列表（从下到上排序）
            min_thickness: 最小厚度（米）

        Returns:
            修正后的地层列表
        """
        if len(layers) == 0:
            return layers

        print(f"  对每个节点位置强制最小厚度 {min_thickness}m...")

        fixed_layers = []
        adjustment_count = 0
        total_adjusted_nodes = 0

        for i, layer in enumerate(layers):
            layer_copy = layer.copy()
            layer_copy['top_surface_z'] = np.asarray(layer['top_surface_z'], dtype=float).copy()
            layer_copy['bottom_surface_z'] = np.asarray(layer['bottom_surface_z'], dtype=float).copy()

            top_z = layer_copy['top_surface_z']
            bottom_z = layer_copy['bottom_surface_z']

            # 计算当前厚度
            thickness = top_z - bottom_z

            # 找出厚度不足的位置
            valid_mask = ~(np.isnan(top_z) | np.isnan(bottom_z))
            thin_mask = valid_mask & (thickness < min_thickness)

            if np.any(thin_mask):
                # 抬高顶面以满足最小厚度
                layer_copy['top_surface_z'] = np.where(
                    valid_mask,
                    np.maximum(top_z, bottom_z + min_thickness),
                    top_z
                )

                num_adjusted = np.sum(thin_mask)
                min_original_thickness = np.min(thickness[thin_mask]) if np.any(thin_mask) else 0
                max_adjustment = np.max((bottom_z + min_thickness - top_z)[thin_mask]) if np.any(thin_mask) else 0

                adjustment_count += 1
                total_adjusted_nodes += num_adjusted

                print(f"    [{i}] {layer.get('name', 'Unknown')}: 修正 {num_adjusted} 个节点 "
                      f"(最小原厚度: {min_original_thickness:.4f}m, 最大抬升: {max_adjustment:.4f}m)")

            fixed_layers.append(layer_copy)

        if adjustment_count > 0:
            print(f"  修正统计: {adjustment_count} 层需要调整, 共 {total_adjusted_nodes} 个节点")
        else:
            print(f"  检查完成: 所有位置厚度已满足要求")

        return fixed_layers

    def _identify_coal_layers(self, layers: List[Dict], adjacent_range: int,
                             selected_indices: Optional[List[int]] = None) -> Tuple[set, set]:
        """
        识别煤层及其相邻层，用于自适应网格密度

        Args:
            layers: 地层列表
            adjacent_range: 煤层上下相邻层的范围
            selected_indices: 用户选择的煤层索引列表，None表示所有煤层

        Returns:
            coal_indices: 所有煤层索引集合
            high_density_indices: 需要高密度的层索引集合（选中的煤层+相邻层）
        """
        # 识别所有煤层
        coal_indices = set()
        for i, layer in enumerate(layers):
            name = layer.get('name', '')
            if '煤' in name or 'coal' in name.lower():
                coal_indices.add(i)

        print(f"\n--- 煤层识别 ---")
        print(f"  识别到 {len(coal_indices)} 个煤层:")
        for i in sorted(coal_indices):
            print(f"    [{i}] {layers[i].get('name', 'Unknown')}")

        # 确定需要高密度的煤层
        if selected_indices is None:
            # 没有指定，使用所有煤层
            selected_coal_indices = coal_indices
            print(f"  使用全部 {len(selected_coal_indices)} 个煤层的高密度网格")
        else:
            # 使用用户选择的煤层
            selected_coal_indices = set(selected_indices) & coal_indices
            print(f"  用户选择 {len(selected_coal_indices)} 个煤层使用高密度网格:")
            for i in sorted(selected_coal_indices):
                print(f"    [{i}] {layers[i].get('name', 'Unknown')}")

        # 扩展到相邻层
        high_density_indices = set()
        for coal_idx in selected_coal_indices:
            for offset in range(-adjacent_range, adjacent_range + 1):
                idx = coal_idx + offset
                if 0 <= idx < len(layers):
                    high_density_indices.add(idx)

        print(f"  高密度区域包含 {len(high_density_indices)} 个地层 (选中煤层±{adjacent_range}层)")

        return coal_indices, high_density_indices

    def _generate_all_layers(self, layers: List[Dict], default_downsample: int,
                            coal_downsample: int, high_density_indices: set) -> List[str]:
        """
        生成所有层的网格，关键是实现层间节点共享（或分离以创建接触面）

        策略:
        1. 从最底层开始，逐层向上处理
        2. 每层创建时，如果不是接触面模式，检查底面节点是否已存在（与下层顶面共享）
        3. 如果是接触面模式，层间不共享节点，便于后续创建接触面
        4. 根据层类型使用不同的降采样率（煤层区域 vs 普通区域）

        Returns:
            层名列表（sanitized后的名称）
        """
        print(f"\n--- 生成网格 ---")

        # 存储每层的顶面节点 ID 网格，用于下一层共享
        # last_top_gp_ids[j][i] = 节点ID
        last_top_gp_ids: Optional[List[List[int]]] = None
        last_downsample = None  # 上一层使用的降采样率

        layer_names = []  # 记录所有层的名称

        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get('name', f'Layer_{layer_idx}')
            safe_name = self._sanitize_name(layer_name)
            layer_names.append(safe_name)  # 记录层名

            # 根据是否在高密度区域选择降采样率
            if self.uniform_downsample:
                # 统一降采样模式：所有层使用相同的降采样率
                current_downsample = default_downsample
                density_label = "统一"
            elif layer_idx in high_density_indices:
                current_downsample = coal_downsample
                density_label = "高密度"
            else:
                current_downsample = default_downsample
                density_label = "常规"

            print(f"\n  处理地层 {layer_idx+1}/{len(layers)}: {layer_name} [{density_label}, {current_downsample}x]")

            # 获取网格数据
            grid_x = np.asarray(layer['grid_x'], dtype=float)
            grid_y = np.asarray(layer['grid_y'], dtype=float)
            top_z = np.asarray(layer['top_surface_z'], dtype=float)
            bottom_z = np.asarray(layer['bottom_surface_z'], dtype=float)

            # 确保是 2D 网格，并强制 x、y 升序排列（修复节点顺序问题）
            if grid_x.ndim == 1 and grid_y.ndim == 1:
                # 1D 输入：先排序，再重排 z 数组
                x1 = grid_x.copy()
                y1 = grid_y.copy()

                # 获取排序索引
                ix = np.argsort(x1)
                iy = np.argsort(y1)

                # 应用排序
                x1 = x1[ix]
                y1 = y1[iy]

                # 重排 z 数组（使用 ix_ 进行索引网格）
                top_z = top_z[np.ix_(iy, ix)]
                bottom_z = bottom_z[np.ix_(iy, ix)]

                # 创建 meshgrid
                grid_x, grid_y = np.meshgrid(x1, y1)

                print(f"    坐标排序: X={'升序' if np.all(np.diff(x1) >= 0) else '降序修正'}, "
                      f"Y={'升序' if np.all(np.diff(y1) >= 0) else '降序修正'}")
            elif grid_x.ndim == 2:
                # 2D 输入：检查是否需要转置或翻转
                # 检查第一行 x 是否单调递增
                x_increasing = np.all(np.diff(grid_x[0, :]) >= 0) if grid_x.shape[1] > 1 else True
                # 检查第一列 y 是否单调递增
                y_increasing = np.all(np.diff(grid_y[:, 0]) >= 0) if grid_y.shape[0] > 1 else True

                if not x_increasing:
                    print(f"    警告: X方向降序，翻转数组")
                    grid_x = np.flip(grid_x, axis=1)
                    grid_y = np.flip(grid_y, axis=1)
                    top_z = np.flip(top_z, axis=1)
                    bottom_z = np.flip(bottom_z, axis=1)

                if not y_increasing:
                    print(f"    警告: Y方向降序，翻转数组")
                    grid_x = np.flip(grid_x, axis=0)
                    grid_y = np.flip(grid_y, axis=0)
                    top_z = np.flip(top_z, axis=0)
                    bottom_z = np.flip(bottom_z, axis=0)

            # 降采样
            if current_downsample > 1:
                grid_x = grid_x[::current_downsample, ::current_downsample]
                grid_y = grid_y[::current_downsample, ::current_downsample]
                top_z = top_z[::current_downsample, ::current_downsample]
                bottom_z = bottom_z[::current_downsample, ::current_downsample]

            ny, nx = grid_x.shape
            print(f"    网格大小: {ny} x {nx}")

            # 如果降采样率改变，无法共享上一层的节点
            # 如果启用接触面模式，也不共享节点
            can_share_nodes = (not self.create_interfaces and
                             last_top_gp_ids is not None and
                             last_downsample == current_downsample)

            # 创建节点网格
            # bottom_gp_ids[j][i] = 底面节点ID
            # top_gp_ids[j][i] = 顶面节点ID
            bottom_gp_ids = [[0] * nx for _ in range(ny)]
            top_gp_ids = [[0] * nx for _ in range(ny)]

            shared_count = 0
            new_bottom_count = 0

            # 创建底面节点
            for j in range(ny):
                for i in range(nx):
                    x = float(grid_x[j, i])
                    y = float(grid_y[j, i])
                    z = float(bottom_z[j, i])

                    if np.isnan(z):
                        continue

                    # 检查是否可以复用上一层的顶面节点
                    if can_share_nodes and j < len(last_top_gp_ids) and i < len(last_top_gp_ids[j]):
                        existing_gp_id = last_top_gp_ids[j][i]
                        if existing_gp_id > 0:
                            # 找到上一层的顶面节点
                            existing_gp = self._get_gridpoint(existing_gp_id)
                            if existing_gp is not None:
                                # 检查坐标是否匹配
                                if (abs(existing_gp.x - x) < 1e-6 and
                                    abs(existing_gp.y - y) < 1e-6 and
                                    abs(existing_gp.z - z) < 1e-6):
                                    # 完全匹配，复用节点
                                    bottom_gp_ids[j][i] = existing_gp_id
                                    shared_count += 1
                                    continue

                    # 需要创建新节点
                    gp_id = self._get_or_create_gridpoint(x, y, z)
                    bottom_gp_ids[j][i] = gp_id
                    new_bottom_count += 1

            # 创建顶面节点（总是新创建）
            for j in range(ny):
                for i in range(nx):
                    x = float(grid_x[j, i])
                    y = float(grid_y[j, i])
                    z = float(top_z[j, i])

                    if np.isnan(z):
                        continue

                    gp_id = self._get_or_create_gridpoint(x, y, z)
                    top_gp_ids[j][i] = gp_id

            print(f"    底面节点: {new_bottom_count} 新建, {shared_count} 共享")
            self.stats.shared_nodes += shared_count

            # 创建单元
            zone_count = 0
            degenerate_count = 0

            if safe_name not in self.groups:
                self.groups[safe_name] = []

            for j in range(ny - 1):
                for i in range(nx - 1):
                    # 获取 8 个角点的节点 ID
                    # 底面 4 个角点
                    gp_b_00 = bottom_gp_ids[j][i]         # (i, j) - 左下
                    gp_b_10 = bottom_gp_ids[j][i+1]       # (i+1, j) - 右下
                    gp_b_01 = bottom_gp_ids[j+1][i]       # (i, j+1) - 左上
                    gp_b_11 = bottom_gp_ids[j+1][i+1]     # (i+1, j+1) - 右上

                    # 顶面 4 个角点
                    gp_t_00 = top_gp_ids[j][i]
                    gp_t_10 = top_gp_ids[j][i+1]
                    gp_t_01 = top_gp_ids[j+1][i]
                    gp_t_11 = top_gp_ids[j+1][i+1]

                    # 检查所有节点是否有效
                    gp_all = [gp_b_00, gp_b_10, gp_b_01, gp_b_11,
                              gp_t_00, gp_t_10, gp_t_01, gp_t_11]

                    if any(gp == 0 for gp in gp_all):
                        continue  # 有无效节点，跳过

                    # 检查是否为退化单元（所有节点不能重复）
                    if len(set(gp_all)) < 8:
                        degenerate_count += 1
                        continue

                    # 检查单元厚度（改用逐角点检查，而非平均厚度）
                    z_bottom = [bottom_z[j, i], bottom_z[j, i+1],
                                bottom_z[j+1, i], bottom_z[j+1, i+1]]
                    z_top = [top_z[j, i], top_z[j, i+1],
                             top_z[j+1, i], top_z[j+1, i+1]]

                    if any(np.isnan(z) for z in z_bottom + z_top):
                        continue

                    # 逐角点厚度检查：所有角点厚度必须大于阈值
                    corner_thicknesses = [zt - zb for zt, zb in zip(z_top, z_bottom)]
                    min_corner_thickness = min(corner_thicknesses)
                    max_corner_thickness = max(corner_thicknesses)
                    avg_thickness = sum(corner_thicknesses) / 4

                    # 如果任何角点厚度小于阈值，说明存在局部翻转
                    if min_corner_thickness < self.MIN_ZONE_THICKNESS:
                        degenerate_count += 1
                        continue

                    # 更新厚度统计（使用最小角点厚度）
                    self.stats.min_thickness = min(self.stats.min_thickness, min_corner_thickness)
                    self.stats.max_thickness = max(self.stats.max_thickness, max_corner_thickness)

                    # 创建 B8 单元
                    # FLAC3D / Itasca B8 节点顺序 (官方标准，见 "Orientation of Nodes and Faces within a Zone")
                    # Node: 1=SW_bot, 2=SE_bot, 3=NW_bot, 4=SW_top, 5=NE_bot, 6=NW_top, 7=SE_top, 8=NE_top
                    # 注意：这是交织模式，不是先底面后顶面！
                    gp_ids = [
                        gp_b_00,  # 1: SW_bot (底面西南, x_min, y_min)
                        gp_b_10,  # 2: SE_bot (底面东南, x_max, y_min)
                        gp_b_01,  # 3: NW_bot (底面西北, x_min, y_max)  ← 关键修正
                        gp_t_00,  # 4: SW_top (顶面西南)  ← 关键修正
                        gp_b_11,  # 5: NE_bot (底面东北, x_max, y_max)  ← 关键修正
                        gp_t_01,  # 6: NW_top (顶面西北)  ← 关键修正
                        gp_t_10,  # 7: SE_top (顶面东南)  ← 关键修正
                        gp_t_11,  # 8: NE_top (顶面东北)
                    ]

                    # 四面体体积检查：确保单元没有翻转
                    # 构造 8x3 坐标数组（必须与 gp_ids 顺序完全一致）
                    coords = np.array([
                        [grid_x[j, i], grid_y[j, i], bottom_z[j, i]],           # 0: SW_bot (1)
                        [grid_x[j, i+1], grid_y[j, i+1], bottom_z[j, i+1]],     # 1: SE_bot (2)
                        [grid_x[j+1, i], grid_y[j+1, i], bottom_z[j+1, i]],     # 2: NW_bot (3) ← 修正
                        [grid_x[j, i], grid_y[j, i], top_z[j, i]],              # 3: SW_top (4) ← 修正
                        [grid_x[j+1, i+1], grid_y[j+1, i+1], bottom_z[j+1, i+1]], # 4: NE_bot (5) ← 修正
                        [grid_x[j+1, i], grid_y[j+1, i], top_z[j+1, i]],        # 5: NW_top (6) ← 修正
                        [grid_x[j, i+1], grid_y[j, i+1], top_z[j, i+1]],        # 6: SE_top (7) ← 修正
                        [grid_x[j+1, i+1], grid_y[j+1, i+1], top_z[j+1, i+1]],  # 7: NE_top (8)
                    ])

                    is_valid, min_tet_vol = check_hex_geometry(coords, tolerance=1e-9)
                    if not is_valid:
                        degenerate_count += 1
                        continue  # 跳过几何无效的单元

                    zone = BrickZone(
                        id=self._next_zone_id,
                        gp_ids=gp_ids,
                        group=safe_name
                    )
                    self.zones.append(zone)
                    self.groups[safe_name].append(zone.id)
                    self._next_zone_id += 1
                    zone_count += 1

            print(f"    单元数: {zone_count}")
            if degenerate_count > 0:
                print(f"    过滤退化单元: {degenerate_count}")
                self.stats.degenerate_zones_removed += degenerate_count

            # 保存顶面节点 ID，供下一层使用
            last_top_gp_ids = top_gp_ids
            last_downsample = current_downsample

            # 检查层间重叠（当前层的底面应该不低于上一层的顶面）
            if layer_idx > 0:
                # 只有在降采样率相同时才能准确比较
                overlap_count = 0
                if can_share_nodes:
                    prev_layer = layers[layer_idx - 1]
                    prev_top_z = np.asarray(prev_layer['top_surface_z'], dtype=float)
                    if current_downsample > 1:
                        prev_top_z = prev_top_z[::current_downsample, ::current_downsample]

                    # 检查底面是否低于上一层顶面
                    valid_mask = ~(np.isnan(bottom_z) | np.isnan(prev_top_z))
                    overlap_mask = valid_mask & (bottom_z < prev_top_z - 0.01)  # 0.01m容差
                    overlap_count = np.sum(overlap_mask)

                    if overlap_count > 0:
                        max_overlap = np.max(prev_top_z[overlap_mask] - bottom_z[overlap_mask])
                        print(f"    ⚠️  检测到 {overlap_count} 处层间重叠，最大重叠 {max_overlap:.3f}m")

        # 更新统计
        self.stats.total_gridpoints = len(self.gridpoints)
        self.stats.total_zones = len(self.zones)
        self.stats.groups = len(self.groups)

        return layer_names

    def _convert_to_relative_coordinates(self):
        """
        将所有节点坐标转换为相对坐标
        以最小角点（min_x, min_y, min_z）作为原点(0, 0, 0)
        """
        if not self.gridpoints:
            return

        print(f"\n--- 转换为相对坐标 ---")

        # 找到坐标范围
        min_x = min(gp.x for gp in self.gridpoints)
        min_y = min(gp.y for gp in self.gridpoints)
        min_z = min(gp.z for gp in self.gridpoints)
        max_x_orig = max(gp.x for gp in self.gridpoints)
        max_y_orig = max(gp.y for gp in self.gridpoints)
        max_z_orig = max(gp.z for gp in self.gridpoints)

        # 记录原始坐标范围
        self.stats.coord_range_x = (min_x, max_x_orig)
        self.stats.coord_range_y = (min_y, max_y_orig)
        self.stats.coord_range_z = (min_z, max_z_orig)
        self.stats.model_size = (max_x_orig - min_x, max_y_orig - min_y, max_z_orig - min_z)

        print(f"  原始坐标范围:")
        print(f"    X: [{min_x:.2f}, {max_x_orig:.2f}] (尺寸: {max_x_orig - min_x:.2f}m)")
        print(f"    Y: [{min_y:.2f}, {max_y_orig:.2f}] (尺寸: {max_y_orig - min_y:.2f}m)")
        print(f"    Z: [{min_z:.2f}, {max_z_orig:.2f}] (尺寸: {max_z_orig - min_z:.2f}m)")
        print(f"  原点偏移: X={min_x:.2f}, Y={min_y:.2f}, Z={min_z:.2f}")

        # 记录偏移量（用于后续坐标还原）
        self.stats.origin_offset = (min_x, min_y, min_z)

        # 转换所有节点为相对坐标
        for gp in self.gridpoints:
            gp.x -= min_x
            gp.y -= min_y
            gp.z -= min_z

        # 更新坐标映射字典
        self._coord_to_gp.clear()
        for gp in self.gridpoints:
            key = (
                round(gp.x, self.COORD_PRECISION),
                round(gp.y, self.COORD_PRECISION),
                round(gp.z, self.COORD_PRECISION)
            )
            self._coord_to_gp[key] = gp.id

        # 计算新的坐标范围
        max_x = max(gp.x for gp in self.gridpoints)
        max_y = max(gp.y for gp in self.gridpoints)
        max_z = max(gp.z for gp in self.gridpoints)

        print(f"  新坐标范围: X=[0, {max_x:.2f}], Y=[0, {max_y:.2f}], Z=[0, {max_z:.2f}]")

    def _get_or_create_gridpoint(self, x: float, y: float, z: float) -> int:
        """获取或创建节点，自动处理坐标合并"""
        # 四舍五入到指定精度
        key = (
            round(x, self.COORD_PRECISION),
            round(y, self.COORD_PRECISION),
            round(z, self.COORD_PRECISION)
        )

        if key in self._coord_to_gp:
            return self._coord_to_gp[key]

        gp = GridPoint(
            id=self._next_gp_id,
            x=key[0],
            y=key[1],
            z=key[2]
        )
        self.gridpoints.append(gp)
        self._coord_to_gp[key] = gp.id
        self._next_gp_id += 1

        return gp.id

    def _get_gridpoint(self, gp_id: int) -> Optional[GridPoint]:
        """根据 ID 获取节点"""
        for gp in self.gridpoints:
            if gp.id == gp_id:
                return gp
        return None

    def _write_f3grid(self, output_path: str):
        """写入 f3grid ASCII 文件"""
        print(f"\n--- 写入文件 ---")

        ox, oy, oz = self.stats.origin_offset

        with open(output_path, 'w', encoding='utf-8') as f:
            # 文件头
            f.write("* ============================================================\n")
            f.write("* FLAC3D Native Grid File (f3grid ASCII format)\n")
            f.write("* Generated by F3GridExporterV2\n")
            f.write("* ============================================================\n")
            f.write(f"* Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"* Total Gridpoints: {len(self.gridpoints)}\n")
            f.write(f"* Total Zones: {len(self.zones)}\n")
            f.write(f"* Total Groups: {len(self.groups)}\n")
            f.write("* ============================================================\n")
            f.write("* COORDINATE SYSTEM: Relative coordinates\n")
            f.write(f"* Origin Offset: X={ox:.6f}, Y={oy:.6f}, Z={oz:.6f}\n")
            f.write("* To convert back to absolute coordinates:\n")
            f.write(f"*   X_abs = X_rel + {ox:.6f}\n")
            f.write(f"*   Y_abs = Y_rel + {oy:.6f}\n")
            f.write(f"*   Z_abs = Z_rel + {oz:.6f}\n")
            f.write("* ============================================================\n")
            f.write("*\n")
            f.write("* Usage in FLAC3D:\n")
            f.write(f'*   zone import f3grid "{os.path.basename(output_path)}"\n')
            f.write("*\n")
            f.write("* ============================================================\n\n")

            # 节点定义
            f.write("* GRIDPOINTS\n")
            f.write("* G <id> <x> <y> <z>\n")
            for gp in self.gridpoints:
                f.write(f"G {gp.id} {gp.x:.6f} {gp.y:.6f} {gp.z:.6f}\n")
            f.write("\n")

            # 单元定义
            f.write("* ZONES (B8 = 8-node brick)\n")
            f.write("* Z B8 <id> <gp1> <gp2> <gp3> <gp4> <gp5> <gp6> <gp7> <gp8>\n")
            for zone in self.zones:
                gp_str = ' '.join(str(gp_id) for gp_id in zone.gp_ids)
                f.write(f"Z B8 {zone.id} {gp_str}\n")
            f.write("\n")

            # 分组定义
            f.write("* ZONE GROUPS\n")
            f.write("* ZGROUP '<name>'\n")
            f.write("* <zone_id> <zone_id> ...\n")
            for group_name, zone_ids in self.groups.items():
                if not zone_ids:
                    continue
                f.write(f"ZGROUP '{group_name}'\n")
                # 每行最多 15 个 ID
                for i in range(0, len(zone_ids), 15):
                    batch = zone_ids[i:i+15]
                    f.write(' '.join(str(zid) for zid in batch) + '\n')
                f.write("\n")

            # 文件尾
            f.write("* ============================================================\n")
            f.write("* End of Grid File\n")
            f.write("* ============================================================\n")

        print(f"  写入完成: {output_path}")

    def _sanitize_name(self, name: str) -> str:
        """将名称转换为 FLAC3D 兼容的 ASCII 形式，同岩性合并到同一组"""
        # 中文到英文的映射
        replacements = {
            '煤': 'coal',
            '砂质泥岩': 'sandy_mudstone',
            '炭质泥岩': 'carbonaceous_mudstone',
            '高岭质泥岩': 'kaolinite_mudstone',
            '高岭岩': 'kaolinite_rock',
            '风化煤': 'weathered_coal',
            '含砾': 'conglomeratic',
            '泥岩': 'mudstone',
            '砂岩': 'sandstone',
            '灰岩': 'limestone',
            '石灰岩': 'limestone',
            '页岩': 'shale',
            '粉砂岩': 'siltstone',
        }

        result = name or 'group'

        # 先去掉名称中的数字序号（如 "砂岩_25" -> "砂岩", "16-4煤" -> "煤"）
        # 去掉开头的数字和分隔符（如 "16-4煤" -> "煤", "16_3_煤" -> "煤"）
        result = re.sub(r'^[\d_\-\.]+', '', result)
        # 去掉末尾的数字和分隔符（如 "砂岩_25" -> "砂岩", "泥岩-3" -> "泥岩"）
        result = re.sub(r'[\d_\-\.]+$', '', result)

        # 中文转英文
        for cn, en in replacements.items():
            result = result.replace(cn, en)

        # 替换非法字符
        result = re.sub(r'[^0-9A-Za-z_]', '_', result)
        result = result.strip('_')

        # 再次清理可能残留的数字后缀（英文转换后可能还有，如 "sandstone_25"）
        result = re.sub(r'_*\d+$', '', result)
        result = result.strip('_')

        return result if result else 'group'

    def _write_interface_script(self, script_path: str, layer_names: List[str]):
        """
        生成 FLAC3D 接触面定义脚本

        当启用接触面模式时，层间节点不共享，需要创建 interface 将相邻层连接起来。
        这样可以模拟层间的滑动、分离等接触行为。

        Args:
            script_path: 脚本输出路径 (.fis)
            layer_names: 层名列表（按从下到上排序）
        """
        print(f"\n--- 生成接触面脚本 ---")

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("; ============================================================\n")
            f.write("; FLAC3D Interface Creation Script\n")
            f.write("; ============================================================\n")
            f.write("; This script creates interface elements between all layers\n")
            f.write("; Run this script AFTER importing the f3grid file\n")
            f.write("; ============================================================\n")
            f.write(f"; Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"; Total Layers: {len(layer_names)}\n")
            f.write(f"; Total Interfaces: {len(layer_names) - 1}\n")
            f.write("; ============================================================\n\n")

            f.write("; Usage:\n")
            f.write(";   1. Import the grid file:\n")
            f.write(";      zone import f3grid \"geological_model.f3grid\"\n")
            f.write(";   2. Run this script:\n")
            f.write(f";      program call \"{os.path.basename(script_path)}\"\n")
            f.write(";   3. Assign interface properties (manual):\n")
            f.write(";      zone interface node property stiffness-normal 1e9 stiffness-shear 1e9\n\n")

            f.write("; ============================================================\n")
            f.write("; Create interfaces between adjacent layers\n")
            f.write("; ============================================================\n\n")

            # 为每对相邻层创建接触面
            for i in range(len(layer_names) - 1):
                lower_layer = layer_names[i]
                upper_layer = layer_names[i + 1]
                interface_name = f"interface_{lower_layer}_{upper_layer}"

                f.write(f"; --- Interface {i+1}: {lower_layer} <-> {upper_layer} ---\n")
                f.write(f"zone interface create by-face separate ...\n")
                f.write(f"    range group '{lower_layer}' group '{upper_layer}' ...\n")
                f.write(f"    name '{interface_name}'\n")
                f.write(f"zone interface node group '{interface_name}' range interface '{interface_name}'\n")
                f.write("\n")

            f.write("; ============================================================\n")
            f.write("; List all interfaces\n")
            f.write("; ============================================================\n")
            f.write("zone interface list\n")
            f.write("zone interface node list information\n\n")

            f.write("; ============================================================\n")
            f.write("; IMPORTANT: Assign interface properties\n")
            f.write("; ============================================================\n")
            f.write("; The interfaces have been created but need material properties.\n")
            f.write("; You need to manually assign stiffness and friction properties:\n\n")

            for i in range(len(layer_names) - 1):
                lower_layer = layer_names[i]
                upper_layer = layer_names[i + 1]
                interface_name = f"interface_{lower_layer}_{upper_layer}"
                f.write(f"; Interface {i+1}: {interface_name}\n")
                f.write(f"zone interface node property stiffness-normal 1e9 stiffness-shear 1e9 ...\n")
                f.write(f"    friction 30.0 cohesion 0.0 ...\n")
                f.write(f"    range interface '{interface_name}'\n\n")

            f.write("; ============================================================\n")
            f.write("; End of Interface Script\n")
            f.write("; ============================================================\n")

        print(f"  接触面脚本已生成: {script_path}")
        print(f"  包含 {len(layer_names) - 1} 个层间接触面")
        print(f"\n  使用方法:")
        print(f"    1. 先导入网格: zone import f3grid \"geological_model.f3grid\"")
        print(f"    2. 运行脚本: program call \"{os.path.basename(script_path)}\"")
        print(f"    3. 根据需要修改接触面参数（刚度、摩擦角、粘聚力等）")


    def _print_stats(self):
        """打印导出统计"""
        print(f"\n{'='*60}")
        print(f"导出统计")
        print(f"{'='*60}")
        print(f"总节点数: {self.stats.total_gridpoints:,}")
        print(f"总单元数: {self.stats.total_zones:,}")
        print(f"共享节点数: {self.stats.shared_nodes:,}")
        print(f"过滤退化单元: {self.stats.degenerate_zones_removed:,}")
        print(f"分组数: {self.stats.groups}")

        if self.stats.min_thickness < float('inf'):
            print(f"厚度范围: {self.stats.min_thickness:.3f}m - {self.stats.max_thickness:.3f}m")

        ox, oy, oz = self.stats.origin_offset
        if ox != 0 or oy != 0 or oz != 0:
            print(f"\n坐标系统: 相对坐标")
            print(f"原点偏移: X={ox:.2f}m, Y={oy:.2f}m, Z={oz:.2f}m")

        print(f"\n分组详情:")
        for name, zone_ids in self.groups.items():
            print(f"  {name}: {len(zone_ids):,} 单元")


def export_f3grid(data: Dict[str, Any], output_path: str,
                  options: Optional[Dict[str, Any]] = None) -> str:
    """便捷导出函数"""
    exporter = F3GridExporterV2()
    return exporter.export(data, output_path, options)


# 测试代码
if __name__ == '__main__':
    import numpy as np

    # 创建测试数据
    nx, ny = 10, 10
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    XI, YI = np.meshgrid(x, y)

    # 创建两个层
    base_z = -100 + 0.1 * XI + 0.05 * YI

    layers = [
        {
            'name': '底板泥岩',
            'grid_x': x,
            'grid_y': y,
            'top_surface_z': base_z + 5,
            'bottom_surface_z': base_z,
        },
        {
            'name': '煤层',
            'grid_x': x,
            'grid_y': y,
            'top_surface_z': base_z + 8,
            'bottom_surface_z': base_z + 5,  # 与底板泥岩顶面一致
        },
        {
            'name': '顶板砂岩',
            'grid_x': x,
            'grid_y': y,
            'top_surface_z': base_z + 20,
            'bottom_surface_z': base_z + 8,  # 与煤层顶面一致
        },
    ]

    data = {'layers': layers}

    exporter = F3GridExporterV2()
    exporter.export(data, 'test_model.f3grid')
