"""
MIDAS 专用导出器 (土木工程有限元分析软件)

支持软件:
1. MIDAS GTS NX - 岩土工程分析
2. MIDAS Civil - 桥梁结构分析
3. MIDAS Gen - 建筑结构分析

导出格式:
- .mct (MIDAS Command Text) - 文本命令格式
- .msh (Mesh格式) - 网格数据
- .txt (通用数据格式)

特点:
- 节点和单元定义
- 材质属性
- 边界条件支持
- 地层分组
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MidasNode:
    """MIDAS节点"""
    id: int
    x: float
    y: float
    z: float


@dataclass
class MidasElement:
    """MIDAS单元"""
    id: int
    type: str  # 'SOLID' (六面体) 或 'SHELL' (壳单元)
    nodes: List[int]  # 节点ID列表
    material: str  # 材料名称


class MIDASExporter:
    """
    MIDAS专用导出器

    支持的单元类型:
    - SOLID (六面体) - 用于3D岩土分析
    - SHELL (四边形) - 用于表面分析

    材料属性:
    - 弹性模量
    - 泊松比
    - 密度
    - 内摩擦角 (岩土)
    - 粘聚力 (岩土)
    """

    # 岩石材料属性 (示例值)
    ROCK_MATERIALS = {
        '煤': {
            'E': 2.0e9,      # 弹性模量 (Pa)
            'nu': 0.30,      # 泊松比
            'rho': 1400,     # 密度 (kg/m³)
            'phi': 25,       # 内摩擦角 (度)
            'c': 1.0e6       # 粘聚力 (Pa)
        },
        '砂岩': {
            'E': 15.0e9,
            'nu': 0.25,
            'rho': 2300,
            'phi': 35,
            'c': 3.0e6
        },
        '泥岩': {
            'E': 8.0e9,
            'nu': 0.28,
            'rho': 2100,
            'phi': 28,
            'c': 2.0e6
        },
        '页岩': {
            'E': 10.0e9,
            'nu': 0.27,
            'rho': 2200,
            'phi': 30,
            'c': 2.5e6
        },
        '灰岩': {
            'E': 25.0e9,
            'nu': 0.22,
            'rho': 2500,
            'phi': 38,
            'c': 4.0e6
        },
    }

    def __init__(self):
        self.nodes: List[MidasNode] = []
        self.elements: List[MidasElement] = []
        self.materials: Dict[str, Dict] = {}
        self.next_node_id = 1
        self.next_elem_id = 1

    def export_mct(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为MIDAS命令文本 (.mct)格式

        Args:
            data: 地层数据
            output_path: 输出路径
            options: 选项
                - downsample_factor: 降采样(默认5)
                - normalize_coords: 归一化(默认True)
                - element_type: 'SOLID' 或 'SHELL' (默认'SOLID')
                - include_materials: 是否包含材料属性(默认True)

        Returns:
            输出文件路径
        """
        if options is None:
            options = {}

        downsample = options.get('downsample_factor', 5)
        normalize_coords = options.get('normalize_coords', True)
        element_type = options.get('element_type', 'SOLID')
        include_materials = options.get('include_materials', True)

        layers_data = data.get('layers', [])
        if not layers_data:
            raise ValueError("没有可导出的地层数据")

        print(f"[MIDAS MCT Export] 开始导出 {len(layers_data)} 个地层")
        print(f"  降采样: {downsample}x")
        print(f"  单元类型: {element_type}")
        print(f"  材料属性: {include_materials}")

        # 重置数据
        self.nodes = []
        self.elements = []
        self.materials = {}
        self.next_node_id = 1
        self.next_elem_id = 1

        # 计算坐标偏移
        coord_offset = self._calculate_offset(layers_data, normalize_coords)

        # 处理每个地层
        for idx, layer in enumerate(layers_data):
            layer_name = self._sanitize_name(layer.get('name', f'Layer_{idx}'))

            print(f"  处理地层 {idx+1}/{len(layers_data)}: {layer_name}")

            # 生成节点和单元
            if element_type == 'SOLID':
                self._generate_solid_elements(layer, layer_name, downsample, coord_offset)
            else:
                self._generate_shell_elements(layer, layer_name, downsample, coord_offset)

            # 添加材料
            if include_materials:
                self._add_material(layer_name)

        # 写入MCT文件
        output_path = str(Path(output_path).with_suffix('.mct'))
        self._write_mct_file(output_path, include_materials)

        print(f"[MIDAS MCT Export] 导出完成: {output_path}")
        print(f"  节点数: {len(self.nodes)}")
        print(f"  单元数: {len(self.elements)}")
        print(f"  提示: 在MIDAS中使用 文件 > 导入 > 文本文件")

        return output_path

    def export_mesh(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为MIDAS网格文件 (.msh)

        格式更紧凑，适合大模型
        """
        if options is None:
            options = {}

        # 先生成节点和单元
        self.export_mct(data, output_path.replace('.msh', '.mct'), options)

        # 写入MSH格式
        output_path = str(Path(output_path).with_suffix('.msh'))
        self._write_msh_file(output_path)

        print(f"[MIDAS MSH Export] 导出完成: {output_path}")

        return output_path

    def _generate_solid_elements(self, layer: Dict, layer_name: str, downsample: int, coord_offset: Tuple):
        """生成六面体单元 (SOLID)"""
        grid_x = np.array(layer.get('grid_x', []))
        grid_y = np.array(layer.get('grid_y', []))
        top_z = np.array(layer.get('top_surface_z', []))
        bottom_z = np.array(layer.get('bottom_surface_z', []))

        if grid_x.size == 0 or top_z.size == 0 or bottom_z.size == 0:
            return

        # 降采样
        if downsample > 1 and grid_x.ndim == 2:
            grid_x = grid_x[::downsample, ::downsample]
            grid_y = grid_y[::downsample, ::downsample]
            top_z = top_z[::downsample, ::downsample]
            bottom_z = bottom_z[::downsample, ::downsample]

        # 应用偏移
        grid_x = grid_x - coord_offset[0]
        grid_y = grid_y - coord_offset[1]
        top_z = top_z - coord_offset[2]
        bottom_z = bottom_z - coord_offset[2]

        rows, cols = grid_x.shape
        if rows < 2 or cols < 2:
            return

        # 创建节点映射
        node_map = {}

        # 生成底面节点
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(bottom_z[i, j]):
                    node = MidasNode(
                        id=self.next_node_id,
                        x=float(grid_x[i, j]),
                        y=float(grid_y[i, j]),
                        z=float(bottom_z[i, j])
                    )
                    self.nodes.append(node)
                    node_map[(i, j, 'bottom')] = self.next_node_id
                    self.next_node_id += 1

        # 生成顶面节点
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(top_z[i, j]):
                    node = MidasNode(
                        id=self.next_node_id,
                        x=float(grid_x[i, j]),
                        y=float(grid_y[i, j]),
                        z=float(top_z[i, j])
                    )
                    self.nodes.append(node)
                    node_map[(i, j, 'top')] = self.next_node_id
                    self.next_node_id += 1

        # 生成六面体单元
        elem_count = 0
        for i in range(rows - 1):
            for j in range(cols - 1):
                # 8个节点: 底面4个 + 顶面4个
                # MIDAS节点顺序: bottom(1-2-3-4) + top(5-6-7-8)
                try:
                    nodes = [
                        node_map[(i, j, 'bottom')],
                        node_map[(i, j+1, 'bottom')],
                        node_map[(i+1, j+1, 'bottom')],
                        node_map[(i+1, j, 'bottom')],
                        node_map[(i, j, 'top')],
                        node_map[(i, j+1, 'top')],
                        node_map[(i+1, j+1, 'top')],
                        node_map[(i+1, j, 'top')]
                    ]

                    element = MidasElement(
                        id=self.next_elem_id,
                        type='SOLID',
                        nodes=nodes,
                        material=layer_name
                    )
                    self.elements.append(element)
                    self.next_elem_id += 1
                    elem_count += 1

                except KeyError:
                    continue

        print(f"    生成 {elem_count} 个SOLID单元")

    def _generate_shell_elements(self, layer: Dict, layer_name: str, downsample: int, coord_offset: Tuple):
        """生成壳单元 (SHELL) - 仅表面"""
        grid_x = np.array(layer.get('grid_x', []))
        grid_y = np.array(layer.get('grid_y', []))
        top_z = np.array(layer.get('top_surface_z', []))

        if grid_x.size == 0 or top_z.size == 0:
            return

        # 降采样
        if downsample > 1 and grid_x.ndim == 2:
            grid_x = grid_x[::downsample, ::downsample]
            grid_y = grid_y[::downsample, ::downsample]
            top_z = top_z[::downsample, ::downsample]

        # 应用偏移
        grid_x = grid_x - coord_offset[0]
        grid_y = grid_y - coord_offset[1]
        top_z = top_z - coord_offset[2]

        rows, cols = grid_x.shape
        if rows < 2 or cols < 2:
            return

        # 创建节点
        node_map = {}
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(top_z[i, j]):
                    node = MidasNode(
                        id=self.next_node_id,
                        x=float(grid_x[i, j]),
                        y=float(grid_y[i, j]),
                        z=float(top_z[i, j])
                    )
                    self.nodes.append(node)
                    node_map[(i, j)] = self.next_node_id
                    self.next_node_id += 1

        # 生成四边形壳单元
        elem_count = 0
        for i in range(rows - 1):
            for j in range(cols - 1):
                try:
                    nodes = [
                        node_map[(i, j)],
                        node_map[(i, j+1)],
                        node_map[(i+1, j+1)],
                        node_map[(i+1, j)]
                    ]

                    element = MidasElement(
                        id=self.next_elem_id,
                        type='SHELL',
                        nodes=nodes,
                        material=layer_name
                    )
                    self.elements.append(element)
                    self.next_elem_id += 1
                    elem_count += 1

                except KeyError:
                    continue

        print(f"    生成 {elem_count} 个SHELL单元")

    def _add_material(self, layer_name: str):
        """添加材料属性"""
        # 查找匹配的材料
        material_props = None
        for rock_name, props in self.ROCK_MATERIALS.items():
            if rock_name in layer_name:
                material_props = props.copy()
                break

        # 使用默认材料
        if material_props is None:
            material_props = {
                'E': 10.0e9,
                'nu': 0.25,
                'rho': 2200,
                'phi': 30,
                'c': 2.0e6
            }

        self.materials[layer_name] = material_props

    def _write_mct_file(self, output_path: str, include_materials: bool):
        """写入MIDAS命令文本文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # 文件头
            f.write("*MIDAS GTS\n")
            f.write("*UNIT\n")
            f.write("LENGTH=m, FORCE=N\n")
            f.write("\n")

            # 节点
            f.write("*NODE\n")
            f.write("; iNO, X, Y, Z\n")
            for node in self.nodes:
                f.write(f"  {node.id}, {node.x:.6f}, {node.y:.6f}, {node.z:.6f}\n")
            f.write("\n")

            # 材料
            if include_materials and self.materials:
                f.write("*MATERIAL\n")
                for mat_name, props in self.materials.items():
                    safe_name = self._sanitize_name(mat_name)
                    f.write(f"; Material: {safe_name}\n")
                    f.write(f"*MAT, {safe_name}, ELASTIC\n")
                    f.write(f"  E={props['E']:.3e}, NU={props['nu']:.3f}, RHO={props['rho']:.1f}\n")
                    if 'phi' in props and 'c' in props:
                        f.write(f"*MAT, {safe_name}, MOHR-COULOMB\n")
                        f.write(f"  PHI={props['phi']:.1f}, C={props['c']:.3e}\n")
                f.write("\n")

            # 单元
            f.write("*ELEMENT\n")
            for elem in self.elements:
                nodes_str = ', '.join(str(n) for n in elem.nodes)
                mat_name = self._sanitize_name(elem.material)
                f.write(f"  {elem.id}, TYPE={elem.type}, MAT={mat_name}, NODE={nodes_str}\n")
            f.write("\n")

            # 分组
            f.write("*GROUP\n")
            groups = {}
            for elem in self.elements:
                mat = elem.material
                if mat not in groups:
                    groups[mat] = []
                groups[mat].append(elem.id)

            for group_name, elem_ids in groups.items():
                safe_name = self._sanitize_name(group_name)
                f.write(f"*GRP, {safe_name}\n")
                # 每行最多15个单元ID
                for i in range(0, len(elem_ids), 15):
                    ids = elem_ids[i:i+15]
                    f.write(f"  {', '.join(str(x) for x in ids)}\n")
            f.write("\n")

            # 文件尾
            f.write("*END\n")

    def _write_msh_file(self, output_path: str):
        """写入MIDAS网格文件 (紧凑格式)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # 头部
            f.write("$MeshFormat\n")
            f.write("2.2 0 8\n")
            f.write("$EndMeshFormat\n")

            # 节点
            f.write("$Nodes\n")
            f.write(f"{len(self.nodes)}\n")
            for node in self.nodes:
                f.write(f"{node.id} {node.x:.6f} {node.y:.6f} {node.z:.6f}\n")
            f.write("$EndNodes\n")

            # 单元
            f.write("$Elements\n")
            f.write(f"{len(self.elements)}\n")
            for elem in self.elements:
                elem_type = 5 if elem.type == 'SOLID' else 3  # 5=六面体, 3=四边形
                nodes_str = ' '.join(str(n) for n in elem.nodes)
                f.write(f"{elem.id} {elem_type} 0 {nodes_str}\n")
            f.write("$EndElements\n")

    def _calculate_offset(self, layers: List[Dict], normalize: bool) -> Tuple[float, float, float]:
        """计算坐标偏移"""
        if not normalize:
            return (0.0, 0.0, 0.0)

        all_x, all_y, all_z = [], [], []

        for layer in layers:
            for key in ['grid_x', 'grid_y', 'top_surface_z', 'bottom_surface_z']:
                arr = layer.get(key)
                if arr is not None:
                    arr = np.array(arr)
                    valid = arr[~np.isnan(arr)]
                    if key == 'grid_x':
                        all_x.extend(valid.flatten())
                    elif key == 'grid_y':
                        all_y.extend(valid.flatten())
                    else:
                        all_z.extend(valid.flatten())

        if all_x and all_y and all_z:
            return (float(np.median(all_x)), float(np.median(all_y)), float(np.min(all_z)))
        return (0.0, 0.0, 0.0)

    def _sanitize_name(self, name: str) -> str:
        """清理名称"""
        sanitized = name.replace(' ', '_').replace('/', '_')
        return ''.join(c for c in sanitized if c.isalnum() or c == '_') or 'unnamed'


def export_for_midas(data: Dict[str, Any], output_path: str,
                     format_type: str = 'mct', options: Optional[Dict[str, Any]] = None) -> str:
    """
    便捷导出函数

    Args:
        data: 地层数据
        output_path: 输出路径
        format_type: 格式类型 ('mct' 或 'msh')
        options: 导出选项

    Returns:
        导出文件路径
    """
    exporter = MIDASExporter()

    if format_type.lower() == 'mct':
        return exporter.export_mct(data, output_path, options)
    elif format_type.lower() == 'msh':
        return exporter.export_mesh(data, output_path, options)
    else:
        raise ValueError(f"不支持的格式: {format_type}，请使用 'mct' 或 'msh'")


if __name__ == '__main__':
    print("MIDAS Exporter - 测试模块")
    print("支持格式: .mct (命令文本), .msh (网格)")
    print("支持软件: MIDAS GTS NX, MIDAS Civil, MIDAS Gen")
