"""
FLAC3D 增强版导出器 - 确保应力/位移传导和正确分组

核心特性:
1. 层间节点精确共享 - 上层底面 = 下层顶面 (同一节点ID)
2. 正确的FLAC3D命令语法
3. 网格质量验证
4. 支持多种导入方式

输出格式:
- .f3dat - FLAC3D原生命令脚本 (推荐)
- .flac3d - FLAC3D命令脚本 (旧版兼容)

在FLAC3D中导入:
    program call 'model.f3dat'
"""

import os
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class FLAC3DNode:
    """FLAC3D节点"""
    id: int
    x: float
    y: float
    z: float


@dataclass
class FLAC3DZone:
    """FLAC3D单元(Brick8)"""
    id: int
    node_ids: List[int]  # 8个节点ID
    group: str


@dataclass
class FLAC3DGroup:
    """FLAC3D单元组"""
    name: str
    zone_ids: List[int] = field(default_factory=list)


class EnhancedFLAC3DExporter:
    """
    增强版FLAC3D导出器

    关键特性:
    1. 层间节点共享 - 确保应力/位移连续传导
    2. 正确的B8单元节点顺序
    3. 完整的FLAC3D命令语法
    4. 网格质量检查
    5. 详细的验证报告
    """

    def __init__(self):
        self.nodes: List[FLAC3DNode] = []
        self.zones: List[FLAC3DZone] = []
        self.groups: Dict[str, FLAC3DGroup] = {}

        self._next_node_id = 1
        self._next_zone_id = 1
        self._node_lookup: Dict[int, FLAC3DNode] = {}
        self._coord_to_node: Dict[Tuple[float, float, float], int] = {}  # 坐标到节点ID的映射

        # 统计信息
        self.stats = {
            'total_nodes': 0,
            'shared_nodes': 0,
            'total_zones': 0,
            'min_thickness': float('inf'),
            'max_thickness': 0,
            'negative_volume_zones': 0,
            'negative_zone_ids': [],
            'max_node_usage': 0,
            'avg_node_usage': 0.0,
            'z_layer_count': 0,
            'fixed_negative_zones': 0,
            'skipped_zones': 0,
            'max_aspect_ratio': 0.0,
        }

        self._last_coord_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_coord_precision: int = 6

        # 默认岩性材料参数（可被调用方覆盖）
        self.rock_properties: Dict[str, Dict[str, float]] = {
            'coal': {'dens': 1400, 'bulk': 2.5e9, 'shear': 1.2e9, 'cohesion': 1.5e6, 'friction': 30},
            'mudstone': {'dens': 2400, 'bulk': 8e9, 'shear': 4e9, 'cohesion': 3e6, 'friction': 35},
            'sandstone': {'dens': 2600, 'bulk': 15e9, 'shear': 10e9, 'cohesion': 8e6, 'friction': 40},
            'limestone': {'dens': 2700, 'bulk': 25e9, 'shear': 15e9, 'cohesion': 12e6, 'friction': 45},
        }

    def export(self, data: Dict[str, Any], output_path: str,
               options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为FLAC3D格式

        Args:
            data: 地层数据
            output_path: 输出路径
            options: 导出选项
                - downsample_factor: 降采样(默认1)
                - normalize_coords: 坐标归一化(默认True)
                - validate_mesh: 验证网格质量(默认True)
                - min_thickness: 最小厚度阈值(默认0.01m)

        Returns:
            输出文件路径
        """
        options = options or {}

        downsample = int(options.get('downsample_factor', 1))
        normalize_coords = options.get('normalize_coords', True)
        validate_mesh = options.get('validate_mesh', True)
        min_thickness = float(options.get('min_thickness', 0.01))
        coord_precision = int(options.get('coord_precision', 8))
        self._last_coord_precision = coord_precision

        layers = data.get('layers', [])
        if not layers:
            raise ValueError("没有可导出的地层数据")

        print(f"\n{'='*60}")
        print(f"FLAC3D Enhanced Exporter")
        print(f"{'='*60}")
        print(f"地层数量: {len(layers)}")
        print(f"降采样: {downsample}x")
        print(f"坐标归一化: {normalize_coords}")
        print(f"最小厚度阈值: {min_thickness}m")

        # 重置状态
        self._reset()

        # 计算坐标偏移
        coord_offset = self._calculate_offset(layers, normalize_coords)
        self._last_coord_offset = coord_offset
        print(f"坐标偏移: X={coord_offset[0]:.2f}, Y={coord_offset[1]:.2f}, Z={coord_offset[2]:.2f}")

        # 强制层间连续性
        self._enforce_layer_continuity(layers)

        # 生成网格
        print(f"\n--- 生成网格 ---")
        self._generate_mesh(layers, downsample, coord_offset, coord_precision, min_thickness)

        # 验证网格
        quality_report_path = None
        if validate_mesh:
            print(f"\n--- 网格验证 ---")
            quality_report_path = self._validate_mesh(output_path)

        # 写入文件
        output_path = str(Path(output_path).with_suffix('.f3dat'))
        self._write_flac3d_script(output_path)

        # 输出质量报告路径（如有）
        if quality_report_path:
            print(f"质量报告: {quality_report_path}")

        # 打印统计
        self._print_statistics()

        print(f"\n导出完成: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        print(f"\n在FLAC3D中导入:")
        print(f"  program call '{output_path}'")

        return output_path

    def _reset(self):
        """重置状态"""
        self.nodes = []
        self.zones = []
        self.groups = {}
        self._next_node_id = 1
        self._next_zone_id = 1
        self._node_lookup = {}
        self._coord_to_node = {}
        self.stats = {
            'total_nodes': 0,
            'shared_nodes': 0,
            'total_zones': 0,
            'min_thickness': float('inf'),
            'max_thickness': 0,
            'negative_volume_zones': 0,
            'negative_zone_ids': [],
            'max_node_usage': 0,
            'avg_node_usage': 0.0,
            'z_layer_count': 0,
            'fixed_negative_zones': 0,
            'skipped_zones': 0,
            'max_aspect_ratio': 0.0,
        }

    def _calculate_offset(self, layers: List[Dict], normalize: bool) -> Tuple[float, float, float]:
        """计算坐标偏移"""
        if not normalize:
            return (0.0, 0.0, 0.0)

        all_x, all_y, all_z = [], [], []

        for layer in layers:
            grid_x = np.asarray(layer.get('grid_x', []), dtype=float)
            grid_y = np.asarray(layer.get('grid_y', []), dtype=float)
            top_z = np.asarray(layer.get('top_surface_z', []), dtype=float)
            bottom_z = np.asarray(layer.get('bottom_surface_z', []), dtype=float)

            if grid_x.size > 0:
                valid_x = grid_x[np.isfinite(grid_x)]
                all_x.extend(valid_x.flatten())
            if grid_y.size > 0:
                valid_y = grid_y[np.isfinite(grid_y)]
                all_y.extend(valid_y.flatten())
            if top_z.size > 0:
                valid_z = top_z[np.isfinite(top_z)]
                all_z.extend(valid_z.flatten())
            if bottom_z.size > 0:
                valid_z = bottom_z[np.isfinite(bottom_z)]
                all_z.extend(valid_z.flatten())

        if all_x and all_y and all_z:
            return (float(np.median(all_x)), float(np.median(all_y)), float(np.min(all_z)))
        return (0.0, 0.0, 0.0)

    def _enforce_layer_continuity(self, layers: List[Dict]):
        """
        强制层间连续性: 上层底面 = 下层顶面

        这是确保应力/位移传导的关键!
        """
        print(f"\n--- 强制层间连续性 ---")

        for i in range(1, len(layers)):
            lower_layer = layers[i - 1]
            upper_layer = layers[i]

            lower_top = np.asarray(lower_layer['top_surface_z'], dtype=float)
            upper_bottom = np.asarray(upper_layer['bottom_surface_z'], dtype=float)

            # 检查差异
            diff = np.abs(upper_bottom - lower_top)
            max_diff = np.nanmax(diff)
            mean_diff = np.nanmean(diff)

            if max_diff > 1e-6:
                print(f"  层 {i-1} -> 层 {i}: 最大差异 {max_diff:.6f}m, 平均差异 {mean_diff:.6f}m")
                # 强制上层底面 = 下层顶面
                upper_layer['bottom_surface_z'] = lower_top.copy()
                print(f"    -> 已强制上层底面 = 下层顶面")
            else:
                print(f"  层 {i-1} -> 层 {i}: 已连续 (差异 < 1e-6m)")

    def _get_or_create_node(self, x: float, y: float, z: float, precision: int) -> int:
        """
        获取或创建节点 (实现节点共享)

        使用坐标的精确匹配来共享节点
        """
        # 四舍五入到指定精度
        key = (round(x, precision), round(y, precision), round(z, precision))

        if key in self._coord_to_node:
            self.stats['shared_nodes'] += 1
            return self._coord_to_node[key]

        # 创建新节点
        node = FLAC3DNode(
            id=self._next_node_id,
            x=key[0],
            y=key[1],
            z=key[2]
        )
        self.nodes.append(node)
        self._node_lookup[node.id] = node
        self._coord_to_node[key] = node.id
        self._next_node_id += 1

        return node.id

    def _generate_mesh(self, layers: List[Dict], downsample: int,
                      coord_offset: Tuple[float, float, float],
                      precision: int, min_thickness: float):
        """生成完整的网格"""

        x_off, y_off, z_off = coord_offset

        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get('name', f'Layer_{layer_idx}')
            safe_name = self._sanitize_name(layer_name)

            print(f"\n  处理地层 {layer_idx+1}/{len(layers)}: {layer_name}")

            # 获取网格数据
            grid_x = np.asarray(layer.get('grid_x', []), dtype=float)
            grid_y = np.asarray(layer.get('grid_y', []), dtype=float)
            top_z = np.asarray(layer.get('top_surface_z', []), dtype=float)
            bottom_z = np.asarray(layer.get('bottom_surface_z', []), dtype=float)

            if grid_x.size == 0:
                print(f"    跳过: 数据为空")
                continue

            # 确保2D
            if grid_x.ndim == 1:
                grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # 降采样
            if downsample > 1:
                grid_x = grid_x[::downsample, ::downsample]
                grid_y = grid_y[::downsample, ::downsample]
                top_z = top_z[::downsample, ::downsample]
                bottom_z = bottom_z[::downsample, ::downsample]

            rows, cols = grid_x.shape
            print(f"    网格尺寸: {rows}x{cols}")

            # 应用坐标偏移
            grid_x = grid_x - x_off
            grid_y = grid_y - y_off
            top_z = top_z - z_off
            bottom_z = bottom_z - z_off

            # 创建分组
            if safe_name not in self.groups:
                self.groups[safe_name] = FLAC3DGroup(name=safe_name)

            # 生成单元
            zone_count = 0
            skipped_count = 0

            for i in range(rows - 1):
                for j in range(cols - 1):
                    # 检查有效性
                    if (np.isnan(top_z[i, j]) or np.isnan(bottom_z[i, j]) or
                        np.isnan(top_z[i, j+1]) or np.isnan(bottom_z[i, j+1]) or
                        np.isnan(top_z[i+1, j+1]) or np.isnan(bottom_z[i+1, j+1]) or
                        np.isnan(top_z[i+1, j]) or np.isnan(bottom_z[i+1, j])):
                        skipped_count += 1
                        continue

                    # 计算厚度
                    thickness = np.mean([
                        top_z[i, j] - bottom_z[i, j],
                        top_z[i, j+1] - bottom_z[i, j+1],
                        top_z[i+1, j+1] - bottom_z[i+1, j+1],
                        top_z[i+1, j] - bottom_z[i+1, j]
                    ])

                    # 更新统计
                    self.stats['min_thickness'] = min(self.stats['min_thickness'], thickness)
                    self.stats['max_thickness'] = max(self.stats['max_thickness'], thickness)

                    if thickness < min_thickness:
                        skipped_count += 1
                        continue

                    # 获取/创建8个节点 (使用节点共享!)
                    # 底面4个节点 (逆时针从SW开始)
                    n0 = self._get_or_create_node(grid_x[i, j], grid_y[i, j], bottom_z[i, j], precision)
                    n1 = self._get_or_create_node(grid_x[i, j+1], grid_y[i, j+1], bottom_z[i, j+1], precision)
                    n2 = self._get_or_create_node(grid_x[i+1, j+1], grid_y[i+1, j+1], bottom_z[i+1, j+1], precision)
                    n3 = self._get_or_create_node(grid_x[i+1, j], grid_y[i+1, j], bottom_z[i+1, j], precision)

                    # 顶面4个节点 (逆时针从SW开始)
                    n4 = self._get_or_create_node(grid_x[i, j], grid_y[i, j], top_z[i, j], precision)
                    n5 = self._get_or_create_node(grid_x[i, j+1], grid_y[i, j+1], top_z[i, j+1], precision)
                    n6 = self._get_or_create_node(grid_x[i+1, j+1], grid_y[i+1, j+1], top_z[i+1, j+1], precision)
                    n7 = self._get_or_create_node(grid_x[i+1, j], grid_y[i+1, j], top_z[i+1, j], precision)

                    # FLAC3D B8节点顺序: 底面(1-2-3-4) + 顶面(5-6-7-8)
                    # 注意: FLAC3D使用1-based索引,但我们的node_ids已经是从1开始
                    node_ids = [n0, n1, n3, n2, n4, n5, n7, n6]  # 调整顺序以匹配FLAC3D

                    # 检查体积 (确保正体积)，尝试修复
                    volume = self._calculate_hex_volume(node_ids)
                    if volume < 0:
                        self.stats['negative_volume_zones'] += 1
                        self.stats['negative_zone_ids'].append(self._next_zone_id)

                        # 尝试翻转节点顺序修复
                        alt_node_ids = [n0, n3, n1, n2, n4, n7, n5, n6]
                        alt_vol = self._calculate_hex_volume(alt_node_ids)
                        if alt_vol > 0:
                            node_ids = alt_node_ids
                            volume = alt_vol
                            self.stats['fixed_negative_zones'] += 1
                        else:
                            # 无法修复则跳过该单元
                            self.stats['skipped_zones'] += 1
                            continue

                    # 估算长细比，记录最大值
                    aspect = self._estimate_aspect_ratio(node_ids)
                    if aspect > self.stats['max_aspect_ratio']:
                        self.stats['max_aspect_ratio'] = aspect

                    # 创建单元
                    zone = FLAC3DZone(
                        id=self._next_zone_id,
                        node_ids=node_ids,
                        group=safe_name
                    )
                    self.zones.append(zone)
                    self.groups[safe_name].zone_ids.append(zone.id)
                    self._next_zone_id += 1
                    zone_count += 1

            print(f"    生成单元: {zone_count}, 跳过: {skipped_count}")
            print(f"    节点共享: {self.stats['shared_nodes']}")

        self.stats['total_nodes'] = len(self.nodes)
        self.stats['total_zones'] = len(self.zones)

    def _calculate_hex_volume(self, node_ids: List[int]) -> float:
        """计算六面体体积 (使用分解为5个四面体的方法)"""
        # 获取8个顶点坐标
        coords = []
        for nid in node_ids:
            node = self._node_lookup[nid]
            coords.append(np.array([node.x, node.y, node.z]))

        # 分解为5个四面体计算体积
        tet_indices = [
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
            (1, 3, 4, 6)
        ]

        total_volume = 0
        for i0, i1, i2, i3 in tet_indices:
            v1 = coords[i1] - coords[i0]
            v2 = coords[i2] - coords[i0]
            v3 = coords[i3] - coords[i0]
            vol = np.dot(v1, np.cross(v2, v3)) / 6.0
            total_volume += vol

        return total_volume

    def _estimate_aspect_ratio(self, node_ids: List[int]) -> float:
        """粗略估算单元长细比，避免极端扁薄单元"""
        coords = []
        for nid in node_ids:
            node = self._node_lookup[nid]
            coords.append(np.array([node.x, node.y, node.z]))
        coords = np.vstack(coords)

        span = coords.max(axis=0) - coords.min(axis=0)
        max_len = float(span.max())
        min_len = float(span.min())
        min_len = min_len if min_len > 1e-6 else 1e-6
        return max_len / min_len

    def _validate_mesh(self, output_path: str) -> Optional[str]:
        """验证网格质量并输出报告"""
        print(f"  检查节点连续性...")

        # 检查每个节点被多少个单元共享
        node_usage = {}
        for zone in self.zones:
            for nid in zone.node_ids:
                node_usage[nid] = node_usage.get(nid, 0) + 1

        # 统计
        usage_counts = list(node_usage.values())
        max_usage = max(usage_counts) if usage_counts else 0
        avg_usage = np.mean(usage_counts) if usage_counts else 0

        print(f"    节点最大共享数: {max_usage}")
        print(f"    节点平均共享数: {avg_usage:.2f}")
        self.stats['max_node_usage'] = max_usage
        self.stats['avg_node_usage'] = float(avg_usage)

        # 检查层间节点共享
        print(f"  验证层间节点共享...")

        # 按Z坐标分组节点
        z_layers = {}
        for node in self.nodes:
            z_key = round(node.z, 4)
            if z_key not in z_layers:
                z_layers[z_key] = []
            z_layers[z_key].append(node.id)

        z_values = sorted(z_layers.keys())
        print(f"    发现 {len(z_values)} 个Z层面")
        self.stats['z_layer_count'] = len(z_values)

        # 检查相邻层是否共享节点
        for i in range(1, len(z_values)):
            lower_z = z_values[i-1]
            upper_z = z_values[i]
            gap = upper_z - lower_z
            lower_nodes = set(z_layers[lower_z])
            upper_nodes = set(z_layers[upper_z])

            # 在层间应该有共享节点 (通过坐标匹配)
            print(f"    层面 Z={lower_z:.2f} -> Z={upper_z:.2f}: 间距 {gap:.4f}m, 节点数 {len(lower_nodes)}/{len(upper_nodes)}")

        # 生成质量报告
        report_path = str(Path(output_path).with_suffix('.quality.txt'))
        try:
            with open(report_path, 'w', encoding='utf-8') as rpt:
                rpt.write("FLAC3D Mesh Quality Report\n")
                rpt.write("==========================\n\n")
                rpt.write(f"Total nodes: {self.stats['total_nodes']}\n")
                rpt.write(f"Shared nodes: {self.stats['shared_nodes']}\n")
                rpt.write(f"Total zones: {self.stats['total_zones']}\n")
                rpt.write(f"Groups: {len(self.groups)}\n")
                rpt.write(f"Z layers: {self.stats['z_layer_count']}\n")
                rpt.write(f"Thickness range: {self.stats['min_thickness']:.4f} - {self.stats['max_thickness']:.4f} m\n\n")

                rpt.write("Node usage\n")
                rpt.write(f"  Max shared: {self.stats['max_node_usage']}\n")
                rpt.write(f"  Avg shared: {self.stats['avg_node_usage']:.2f}\n\n")

                rpt.write("Volumes\n")
                rpt.write(f"  Negative volume zones: {self.stats['negative_volume_zones']}\n")
                rpt.write(f"  Fixed negative zones: {self.stats['fixed_negative_zones']}\n")
                rpt.write(f"  Skipped zones (unfixed): {self.stats['skipped_zones']}\n")
                if self.stats['negative_zone_ids']:
                    sample_ids = self.stats['negative_zone_ids'][:20]
                    rpt.write(f"  Sample zone ids: {sample_ids}\n")
                rpt.write("\n")

                rpt.write("Geometry\n")
                rpt.write(f"  Max aspect ratio: {self.stats['max_aspect_ratio']:.3f}\n")

                rpt.write("Coordinates\n")
                rpt.write(f"  Offset: {self._last_coord_offset}\n")
                rpt.write(f"  Precision: {self._last_coord_precision}\n")

            return report_path
        except Exception:
            return None

    def _sanitize_name(self, name: str) -> str:
        """清理组名"""
        # 中文到英文映射
        mapping = {
            '煤': 'coal',
            '煤层': 'coal',
            '砂岩': 'sandstone',
            '细砂岩': 'fine_sandstone',
            '中砂岩': 'medium_sandstone',
            '粗砂岩': 'coarse_sandstone',
            '泥岩': 'mudstone',
            '砂质泥岩': 'sandy_mudstone',
            '炭质泥岩': 'carbonaceous_mudstone',
            '页岩': 'shale',
            '炭质页岩': 'carbonaceous_shale',
            '粉砂岩': 'siltstone',
            '灰岩': 'limestone',
            '石灰岩': 'limestone',
            '砾岩': 'conglomerate',
        }

        result = name or 'group'
        for cn, en in mapping.items():
            result = result.replace(cn, en)

        # 移除非法字符
        result = re.sub(r'[^0-9A-Za-z_]', '_', result)
        result = result.strip('_') or 'group'

        return result

    def _write_flac3d_script(self, output_path: str):
        """写入FLAC3D命令脚本 (使用缓冲写入优化)"""
        print(f"\n--- 写入FLAC3D脚本 ---")

        # 缓冲区配置
        BUFFER_SIZE = 50000  # 50k lines buffer
        buffer = []

        def flush_buffer(file_handle):
            if buffer:
                file_handle.writelines(buffer)
                buffer.clear()

        with open(output_path, 'w', encoding='utf-8') as f:
            # 辅助写入函数
            def write_line(line):
                buffer.append(line + '\n')
                if len(buffer) >= BUFFER_SIZE:
                    flush_buffer(f)

            # 文件头
            write_line("; ============================================")
            write_line("; FLAC3D Mesh Import Script")
            write_line("; Generated by Enhanced FLAC3D Exporter")
            write_line("; ============================================")
            write_line(f"; Total Gridpoints: {len(self.nodes)}")
            write_line(f"; Total Zones: {len(self.zones)}")
            write_line(f"; Total Groups: {len(self.groups)}")
            write_line(f"; Shared Nodes: {self.stats['shared_nodes']}")
            write_line("; ============================================\n")

            # 初始化模型
            write_line("; Initialize model")
            write_line("model new")
            write_line("model largestrain off\n")

            # 创建节点
            write_line("; ============================================")
            write_line("; Create gridpoints")
            write_line("; ============================================")
            
            # 批量处理节点
            for i, node in enumerate(self.nodes):
                write_line(f"zone gridpoint create ({node.x:.6f},{node.y:.6f},{node.z:.6f})")
                if i % 10000 == 0 and i > 0:
                    print(f"  写入节点: {i}/{len(self.nodes)}", end='\r')
            write_line("")
            print(f"  写入节点: 完成        ")

            # 创建单元
            write_line("; ============================================")
            write_line("; Create zones (brick)")
            write_line("; ============================================")
            
            # 批量处理单元
            for i, zone in enumerate(self.zones):
                gp_str = ' '.join(str(nid) for nid in zone.node_ids)
                write_line(f"zone create brick point-id {gp_str}")
                if i % 5000 == 0 and i > 0:
                    print(f"  写入单元: {i}/{len(self.zones)}", end='\r')
            write_line("")
            print(f"  写入单元: 完成        ")

            # 创建分组
            write_line("; ============================================")
            write_line("; Assign zone groups")
            write_line("; ============================================")
            for group_name, group in self.groups.items():
                if not group.zone_ids:
                    continue

                # 使用范围选择
                write_line(f"; Group: {group_name} ({len(group.zone_ids)} zones)")

                # 分批处理,每批100个单元
                batch_size = 100
                for i in range(0, len(group.zone_ids), batch_size):
                    batch = group.zone_ids[i:i+batch_size]
                    zone_list = ' '.join(str(zid) for zid in batch)
                    write_line(f"zone group '{group_name}' range id {zone_list}")

                write_line("")

            # 材料属性
            write_line("; ============================================")
            write_line("; Assign material properties (auto-mapped)")
            write_line("; ============================================")
            for group_name, group in self.groups.items():
                props = self._get_material_props(group_name)
                if not props:
                    continue

                batch_size = 200
                for i in range(0, len(group.zone_ids), batch_size):
                    batch = group.zone_ids[i:i+batch_size]
                    zone_list = ' '.join(str(zid) for zid in batch)
                    write_line(
                        "zone property "
                        f"dens={props['dens']} bulk={props['bulk']} shear={props['shear']} "
                        f"cohesion={props['cohesion']} friction={props['friction']} "
                        f"range id {zone_list}"
                    )
            write_line("")

            # 验证信息
            write_line("; ============================================")
            write_line("; Verify mesh")
            write_line("; ============================================")
            write_line("zone list")
            write_line("zone group list\n")

            # 文件结束
            write_line("; ============================================")
            write_line("; End of mesh definition")
            write_line("; ============================================")
            write_line("; Next steps:")
            write_line("; 1. Assign constitutive models: zone cmodel assign ...")
            write_line("; 2. Assign material properties: zone property ...")
            write_line("; 3. Set boundary conditions: zone face apply ...")
            write_line("; 4. Initialize stresses: zone initialize ...")
            write_line("; 5. Solve: model solve")
            
            # 刷新剩余缓冲区
            flush_buffer(f)

    def _print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"网格统计")
        print(f"{'='*60}")
        print(f"总节点数: {self.stats['total_nodes']}")
        print(f"共享节点数: {self.stats['shared_nodes']}")
        print(f"总单元数: {self.stats['total_zones']}")
        print(f"分组数: {len(self.groups)}")
        print(f"厚度范围: {self.stats['min_thickness']:.4f}m - {self.stats['max_thickness']:.4f}m")
        if self.stats['negative_volume_zones'] > 0:
            print(f"警告: 检测到 {self.stats['negative_volume_zones']} 个负体积单元，修复 {self.stats['fixed_negative_zones']}，跳过 {self.stats['skipped_zones']}")
        print(f"最大长细比: {self.stats['max_aspect_ratio']:.3f}")

        # 按分组统计
        print(f"\n分组详情:")
        for name, group in self.groups.items():
            print(f"  {name}: {len(group.zone_ids)} 单元")

    def _get_material_props(self, group_name: str) -> Optional[Dict[str, float]]:
        """根据组名选择材料参数"""
        key = group_name.lower()
        if key in self.rock_properties:
            return self.rock_properties[key]

        # 宽松匹配
        for rock_key in self.rock_properties.keys():
            if rock_key in key:
                return self.rock_properties[rock_key]

        # 默认使用砂岩参数
        return self.rock_properties.get('sandstone')


def export_for_flac3d(data: Dict[str, Any], output_path: str,
                      options: Optional[Dict[str, Any]] = None) -> str:
    """
    便捷导出函数

    Args:
        data: 地层数据
        output_path: 输出路径
        options: 选项

    Returns:
        输出文件路径
    """
    exporter = EnhancedFLAC3DExporter()
    return exporter.export(data, output_path, options)


if __name__ == '__main__':
    print("Enhanced FLAC3D Exporter")
    print("确保层间节点共享,保证应力/位移正常传导")
