"""
Midas GTS NX FPN 格式导出器

FPN (Finite element Program Neutral file) 是 Midas GTS NX 的标准网格交换格式。
可以被多种转换工具读取，包括转换为 FLAC3D f3grid 格式。

格式规范:
- NODE, node_id, x, y, z, coord_sys, ...
- HEXA, elem_id, mat_id, n1, n2, n3, n4, n5, n6
       , n7, n8, ...

参考: mx.fpn 文件
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FPNNode:
    """FPN 节点"""
    id: int
    x: float
    y: float
    z: float


@dataclass
class FPNHexa:
    """FPN 六面体单元 (8节点)"""
    id: int
    mat_id: int
    node_ids: List[int]  # 8个节点


class FPNExporter:
    """
    Midas GTS NX FPN 格式导出器

    导出为中间格式，方便转换为其他格式（如 FLAC3D f3grid）
    """

    COORD_PRECISION = 6  # 坐标精度

    def __init__(self):
        self.nodes: List[FPNNode] = []
        self.elements: List[FPNHexa] = []
        self.materials: Dict[str, int] = {}  # group_name -> mat_id

        self._next_node_id = 1
        self._next_elem_id = 1
        self._next_mat_id = 1

        self._coord_to_node: Dict[Tuple[float, float, float], int] = {}

    def export(self, data: Dict[str, Any], output_path: str,
               options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出地质模型为 FPN 格式

        Args:
            data: 地层数据字典 (与 f3grid_exporter_v2 相同格式)
            output_path: 输出文件路径 (.fpn)
            options: 导出选项
                - downsample_factor: int, 降采样倍数
                - coal_downsample_factor: int, 煤层降采样倍数
                - coal_adjacent_layers: int, 煤层上下相邻层数使用高密度
                - selected_coal_layers: list, 需要高密度的煤层索引列表
                - create_interfaces: bool, 创建接触面模式 (默认 False，层间不共享节点)

        Returns:
            输出文件路径
        """
        options = options or {}
        downsample = max(1, int(options.get('downsample_factor', 1)))
        coal_downsample = max(1, int(options.get('coal_downsample_factor', downsample)))
        coal_adjacent = int(options.get('coal_adjacent_layers', 1))
        selected_coal_layers = options.get('selected_coal_layers', None)
        create_interfaces = bool(options.get('create_interfaces', False))  # 接触面模式
        self.create_interfaces = create_interfaces

        # 重置状态
        self._reset()

        layers = data.get('layers', [])
        if not layers:
            raise ValueError("没有可导出的地层数据")

        print(f"\n{'='*60}")
        print(f"Midas GTS NX FPN Exporter")
        print(f"{'='*60}")
        print(f"地层数量: {len(layers)}")
        print(f"降采样: 常规地层 {downsample}x, 煤层区域 {coal_downsample}x")
        print(f"接触面模式: {'启用 (层间不共享节点)' if create_interfaces else '禁用 (层间共享节点)'}")
        print(f"输出文件: {output_path}")

        # 识别煤层
        coal_indices, high_density_indices = self._identify_coal_layers(
            layers, coal_adjacent, selected_coal_layers
        )

        # 生成网格
        self._generate_all_layers(layers, downsample, coal_downsample, high_density_indices)

        # 写入文件
        self._write_fpn(output_path)

        print(f"\n[OK] 导出完成!")
        print(f"  总节点数: {len(self.nodes):,}")
        print(f"  总单元数: {len(self.elements):,}")
        print(f"  材料分组: {len(self.materials)}")

        return output_path

    def _reset(self):
        """重置导出器状态"""
        self.nodes = []
        self.elements = []
        self.materials = {}
        self._next_node_id = 1
        self._next_elem_id = 1
        self._next_mat_id = 1
        self._coord_to_node = {}

    def _identify_coal_layers(self, layers: List[Dict], adjacent_range: int,
                             selected_indices: Optional[List[int]] = None) -> Tuple[set, set]:
        """识别煤层及相邻层"""
        coal_indices = set()
        for i, layer in enumerate(layers):
            name = layer.get('name', '')
            if '煤' in name or 'coal' in name.lower():
                coal_indices.add(i)

        print(f"\n--- 煤层识别 ---")
        print(f"  识别到 {len(coal_indices)} 个煤层")

        if selected_indices is None:
            selected_coal_indices = coal_indices
            print(f"  使用全部 {len(selected_coal_indices)} 个煤层的高密度网格")
        else:
            selected_coal_indices = set(selected_indices) & coal_indices
            print(f"  用户选择 {len(selected_coal_indices)} 个煤层使用高密度网格")

        # 扩展到相邻层
        high_density_indices = set()
        for coal_idx in selected_coal_indices:
            for offset in range(-adjacent_range, adjacent_range + 1):
                idx = coal_idx + offset
                if 0 <= idx < len(layers):
                    high_density_indices.add(idx)

        print(f"  高密度区域包含 {len(high_density_indices)} 个地层")

        return coal_indices, high_density_indices

    def _generate_all_layers(self, layers: List[Dict], default_downsample: int,
                            coal_downsample: int, high_density_indices: set):
        """生成所有层的网格"""
        print(f"\n--- 生成网格 ---")

        last_top_node_ids: Optional[List[List[int]]] = None
        last_downsample = None

        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get('name', f'Layer_{layer_idx}')
            safe_name = self._sanitize_name(layer_name)

            # 获取材料ID
            if safe_name not in self.materials:
                self.materials[safe_name] = self._next_mat_id
                self._next_mat_id += 1
            mat_id = self.materials[safe_name]

            # 选择降采样率
            if layer_idx in high_density_indices:
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

            # 确保是 2D 网格
            if grid_x.ndim == 1 and grid_y.ndim == 1:
                grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # 降采样
            if current_downsample > 1:
                grid_x = grid_x[::current_downsample, ::current_downsample]
                grid_y = grid_y[::current_downsample, ::current_downsample]
                top_z = top_z[::current_downsample, ::current_downsample]
                bottom_z = bottom_z[::current_downsample, ::current_downsample]

            ny, nx = grid_x.shape
            print(f"    网格大小: {ny} x {nx}")

            # 节点共享判断（如果启用接触面模式，则不共享节点）
            can_share_nodes = (not self.create_interfaces and
                             last_top_node_ids is not None and
                             last_downsample == current_downsample)

            # 创建节点网格
            bottom_node_ids = [[0] * nx for _ in range(ny)]
            top_node_ids = [[0] * nx for _ in range(ny)]

            shared_count = 0

            # 创建底面节点
            for j in range(ny):
                for i in range(nx):
                    x = float(grid_x[j, i])
                    y = float(grid_y[j, i])
                    z = float(bottom_z[j, i])

                    if np.isnan(z):
                        continue

                    # 尝试共享节点
                    if can_share_nodes and j < len(last_top_node_ids) and i < len(last_top_node_ids[j]):
                        existing_id = last_top_node_ids[j][i]
                        if existing_id > 0:
                            # 检查坐标是否匹配
                            existing_node = self._get_node(existing_id)
                            if existing_node and abs(existing_node.z - z) < 1e-6:
                                bottom_node_ids[j][i] = existing_id
                                shared_count += 1
                                continue

                    # 创建新节点
                    node_id = self._get_or_create_node(x, y, z)
                    bottom_node_ids[j][i] = node_id

            # 创建顶面节点
            for j in range(ny):
                for i in range(nx):
                    x = float(grid_x[j, i])
                    y = float(grid_y[j, i])
                    z = float(top_z[j, i])

                    if np.isnan(z):
                        continue

                    node_id = self._get_or_create_node(x, y, z)
                    top_node_ids[j][i] = node_id

            print(f"    共享节点: {shared_count}")

            # 创建单元
            elem_count = 0

            for j in range(ny - 1):
                for i in range(nx - 1):
                    # 8个角点
                    n_b_00 = bottom_node_ids[j][i]
                    n_b_10 = bottom_node_ids[j][i+1]
                    n_b_01 = bottom_node_ids[j+1][i]
                    n_b_11 = bottom_node_ids[j+1][i+1]

                    n_t_00 = top_node_ids[j][i]
                    n_t_10 = top_node_ids[j][i+1]
                    n_t_01 = top_node_ids[j+1][i]
                    n_t_11 = top_node_ids[j+1][i+1]

                    all_nodes = [n_b_00, n_b_10, n_b_01, n_b_11,
                                n_t_00, n_t_10, n_t_01, n_t_11]

                    if any(n == 0 for n in all_nodes):
                        continue

                    if len(set(all_nodes)) < 8:
                        continue

                    # FPN 六面体节点顺序（参考 mx.fpn）
                    # 基于观察，Midas 使用标准的 [SW,SE,NE,NW] + [SW,SE,NE,NW] 顺序
                    node_ids = [
                        n_b_00,  # 底面 SW
                        n_b_10,  # 底面 SE
                        n_b_11,  # 底面 NE
                        n_b_01,  # 底面 NW
                        n_t_00,  # 顶面 SW
                        n_t_10,  # 顶面 SE
                        n_t_11,  # 顶面 NE
                        n_t_01,  # 顶面 NW
                    ]

                    elem = FPNHexa(
                        id=self._next_elem_id,
                        mat_id=mat_id,
                        node_ids=node_ids
                    )
                    self.elements.append(elem)
                    self._next_elem_id += 1
                    elem_count += 1

            print(f"    单元数: {elem_count}")

            # 保存顶面节点供下一层使用
            last_top_node_ids = top_node_ids
            last_downsample = current_downsample

    def _get_or_create_node(self, x: float, y: float, z: float) -> int:
        """获取或创建节点"""
        key = (
            round(x, self.COORD_PRECISION),
            round(y, self.COORD_PRECISION),
            round(z, self.COORD_PRECISION)
        )

        if key in self._coord_to_node:
            return self._coord_to_node[key]

        node = FPNNode(
            id=self._next_node_id,
            x=key[0],
            y=key[1],
            z=key[2]
        )
        self.nodes.append(node)
        self._coord_to_node[key] = node.id
        self._next_node_id += 1

        return node.id

    def _get_node(self, node_id: int) -> Optional[FPNNode]:
        """根据ID获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def _sanitize_name(self, name: str) -> str:
        """将名称转换为安全的材料名"""
        import re

        replacements = {
            '煤': 'coal',
            '泥岩': 'mudstone',
            '砂岩': 'sandstone',
            '灰岩': 'limestone',
            '页岩': 'shale',
            '粉砂岩': 'siltstone',
        }

        result = name or 'group'
        result = re.sub(r'^[\d_\-\.]+', '', result)
        result = re.sub(r'[\d_\-\.]+$', '', result)

        for cn, en in replacements.items():
            result = result.replace(cn, en)

        result = re.sub(r'[^0-9A-Za-z_]', '_', result)
        result = result.strip('_')
        result = re.sub(r'_*\d+$', '', result)
        result = result.strip('_')

        return result if result else 'group'

    def _write_fpn(self, output_path: str):
        """写入 FPN 文件"""
        print(f"\n--- 写入 FPN 文件 ---")

        with open(output_path, 'w', encoding='utf-8') as f:
            # 文件头
            f.write("$$ *********************************************\n")
            f.write("$$      Neutral File Created from Python\n")
            f.write("$$ *********************************************\n")
            f.write("\n")

            # 版本信息
            f.write("$$ Version information.\n")
            f.write("VER, 2.0.0\n")
            f.write("\n")

            # 项目设置
            f.write("$$ Project Setting Data.\n")
            f.write("PROJ   , , 0, 1,          9.80665,          9.80665,               0.,               1.,\n")
            f.write("$$ Unit system.\n")
            f.write("UNIT, KN,M,SEC\n")
            f.write("\n")

            # 坐标系
            f.write("$$ Coordinate data.\n")
            f.write("CRECT  , 1, Global, 0,               0.,               0.,               0., ,\n")
            f.write("       ,               1.,               0.,               0.,               0.,               1.,               0., ,\n")
            f.write("\n")

            # 节点定义
            f.write("$$      Node\n")
            for node in self.nodes:
                f.write(f"NODE   , {node.id}, {node.x:.6f}, {node.y:.6f}, {node.z:.6f}, 1, , ,\n")

            f.write("\n")

            # 单元定义
            f.write("$$      Element\n")
            for elem in self.elements:
                # FPN 格式：第一行 6 个节点，第二行 2 个节点
                n = elem.node_ids
                f.write(f"HEXA   , {elem.id}, {elem.mat_id}, {n[0]}, {n[1]}, {n[2]}, {n[3]}, {n[4]}, {n[5]}\n")
                f.write(f"       , {n[6]}, {n[7]}, , , , , ,\n")

            f.write("\n")

            # 材料分组
            f.write("$$ Material groups\n")
            for mat_name, mat_id in self.materials.items():
                f.write(f"$$ Material {mat_id}: {mat_name}\n")

        print(f"  写入完成: {output_path}")


def export_fpn(data: Dict[str, Any], output_path: str,
               options: Optional[Dict[str, Any]] = None) -> str:
    """便捷导出函数"""
    exporter = FPNExporter()
    return exporter.export(data, output_path, options)


if __name__ == '__main__':
    print("FPN Exporter - Use from app_qt.py or other scripts")
