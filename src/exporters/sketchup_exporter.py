"""
SketchUp 专用导出器

支持格式:
1. COLLADA (.dae) - SketchUp原生支持，推荐格式
2. Enhanced OBJ - 包含材质和图层信息的OBJ
3. 3D Tiles (.json) - 用于大规模地质模型

特点:
- 自动创建图层/组
- 地层材质和颜色
- 保持模型可编辑性
- 优化文件大小
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path


class SketchUpExporter:
    """
    SketchUp专用导出器

    主要功能:
    1. COLLADA格式导出(推荐) - .dae
    2. Enhanced OBJ导出 - 包含完整材质和图层
    3. 自动优化网格
    """

    # SketchUp友好的岩石颜色配置 (RGB 0-1)
    ROCK_COLORS = {
        '煤': (0.10, 0.10, 0.10),
        '煤层': (0.10, 0.10, 0.10),
        '砂岩': (0.82, 0.73, 0.55),
        '细砂岩': (0.88, 0.80, 0.62),
        '中砂岩': (0.82, 0.73, 0.55),
        '粗砂岩': (0.76, 0.67, 0.48),
        '泥岩': (0.55, 0.60, 0.55),
        '砂质泥岩': (0.62, 0.67, 0.60),
        '页岩': (0.40, 0.45, 0.52),
        '炭质页岩': (0.32, 0.37, 0.42),
        '粉砂岩': (0.72, 0.66, 0.58),
        '灰岩': (0.70, 0.75, 0.80),
        '石灰岩': (0.70, 0.75, 0.80),
        '砾岩': (0.65, 0.50, 0.38),
    }

    def __init__(self):
        self.default_colors = [
            (0.90, 0.40, 0.30),  # 红褐
            (0.40, 0.70, 0.90),  # 天蓝
            (0.30, 0.75, 0.60),  # 青绿
            (0.80, 0.70, 0.40),  # 沙黄
            (0.60, 0.50, 0.70),  # 紫灰
        ]

    def export_collada(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为COLLADA (.dae)格式 - SketchUp推荐格式

        Args:
            data: 地层数据
            output_path: 输出路径
            options: 选项
                - downsample_factor: 降采样倍数(默认3)
                - normalize_coords: 是否归一化坐标(默认True)
                - export_textures: 是否导出纹理(默认False)

        Returns:
            输出文件路径
        """
        if options is None:
            options = {}

        downsample = options.get('downsample_factor', 3)
        normalize_coords = options.get('normalize_coords', True)

        layers = data.get('layers', [])
        if not layers:
            raise ValueError("没有可导出的地层数据")

        print(f"[SketchUp DAE Export] 开始导出 {len(layers)} 个地层")
        print(f"  降采样: {downsample}x")
        print(f"  坐标归一化: {normalize_coords}")

        # 计算坐标偏移
        coord_offset = self._calculate_offset(layers, normalize_coords)

        # 构建COLLADA XML
        root = self._create_collada_root()

        # 添加资源库
        library_geometries = ET.SubElement(root, 'library_geometries')
        library_materials = ET.SubElement(root, 'library_materials')
        library_effects = ET.SubElement(root, 'library_effects')
        library_visual_scenes = ET.SubElement(root, 'library_visual_scenes')

        visual_scene = ET.SubElement(library_visual_scenes, 'visual_scene', id='Scene', name='Scene')

        # 处理每个地层
        for idx, layer in enumerate(layers):
            layer_name = self._sanitize_name(layer.get('name', f'Layer_{idx}'))
            material_id = f'mat_{layer_name}'

            print(f"  处理地层 {idx+1}/{len(layers)}: {layer_name}")

            # 获取颜色
            color = self._get_layer_color(layer.get('name', ''), idx)

            # 创建材质和效果
            self._create_material(library_materials, library_effects, material_id, layer_name, color)

            # 创建几何体
            geometry_id = f'geom_{layer_name}'
            vertices, faces = self._extract_layer_geometry(layer, downsample, coord_offset)

            if not faces:
                print(f"    跳过 (无有效几何)")
                continue

            self._create_geometry(library_geometries, geometry_id, layer_name, vertices, faces)

            # 添加到场景
            self._add_node_to_scene(visual_scene, layer_name, geometry_id, material_id)

            print(f"    顶点: {len(vertices)}, 面: {len(faces)}")

        # 设置场景
        scene = ET.SubElement(root, 'scene')
        ET.SubElement(scene, 'instance_visual_scene', url='#Scene')

        # 写入文件
        output_path = str(Path(output_path).with_suffix('.dae'))
        self._write_xml(root, output_path)

        print(f"[SketchUp DAE Export] 导出完成: {output_path}")
        print(f"  提示: 在SketchUp中使用 文件 > 导入 > COLLADA (.dae)")

        return output_path

    def export_enhanced_obj(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出增强OBJ格式

        特点:
        - 每个地层作为独立group
        - MTL材质文件包含地层颜色
        - 优化的面片顺序
        """
        if options is None:
            options = {}

        downsample = options.get('downsample_factor', 3)
        normalize_coords = options.get('normalize_coords', True)

        layers = data.get('layers', [])
        if not layers:
            raise ValueError("没有可导出的地层数据")

        print(f"[SketchUp OBJ Export] 开始导出 {len(layers)} 个地层")

        coord_offset = self._calculate_offset(layers, normalize_coords)

        # 确保输出路径正确
        output_path = str(Path(output_path).with_suffix('.obj'))
        mtl_path = output_path.replace('.obj', '.mtl')
        mtl_filename = os.path.basename(mtl_path)

        # 收集所有几何
        all_vertices = []
        all_groups = []
        vertex_offset = 1

        for idx, layer in enumerate(layers):
            layer_name = self._sanitize_name(layer.get('name', f'Layer_{idx}'))
            material_name = f'mat_{layer_name}'

            print(f"  处理地层 {idx+1}/{len(layers)}: {layer_name}")

            vertices, faces = self._extract_layer_geometry(layer, downsample, coord_offset)

            if not faces:
                print(f"    跳过 (无有效几何)")
                continue

            all_vertices.extend(vertices)
            all_groups.append({
                'name': layer_name,
                'material': material_name,
                'faces': [(f[0] + vertex_offset - 1, f[1] + vertex_offset - 1, f[2] + vertex_offset - 1) for f in faces],
                'color': self._get_layer_color(layer.get('name', ''), idx)
            })

            vertex_offset += len(vertices)

            print(f"    顶点: {len(vertices)}, 面: {len(faces)}")

        # 写入OBJ文件
        self._write_obj_file(output_path, mtl_filename, all_vertices, all_groups)

        # 写入MTL文件
        self._write_mtl_file(mtl_path, all_groups)

        print(f"[SketchUp OBJ Export] 导出完成: {output_path}")
        print(f"  材质文件: {mtl_path}")
        print(f"  提示: 导入时确保OBJ和MTL文件在同一目录")

        return output_path

    def _create_collada_root(self) -> ET.Element:
        """创建COLLADA根元素"""
        root = ET.Element('COLLADA', {
            'xmlns': 'http://www.collada.org/2005/11/COLLADASchema',
            'version': '1.4.1'
        })

        # Asset信息
        asset = ET.SubElement(root, 'asset')
        contributor = ET.SubElement(asset, 'contributor')
        ET.SubElement(contributor, 'author').text = 'Geological Modeling System'
        ET.SubElement(contributor, 'authoring_tool').text = 'Python COLLADA Exporter'

        ET.SubElement(asset, 'created').text = '2025-01-01T00:00:00'
        ET.SubElement(asset, 'modified').text = '2025-01-01T00:00:00'

        unit = ET.SubElement(asset, 'unit', name='meter', meter='1')
        ET.SubElement(asset, 'up_axis').text = 'Z_UP'  # SketchUp使用Z轴向上

        return root

    def _create_material(self, library_materials: ET.Element, library_effects: ET.Element,
                        material_id: str, name: str, color: Tuple[float, float, float]):
        """创建材质和效果"""
        # 效果
        effect_id = f'effect_{material_id}'
        effect = ET.SubElement(library_effects, 'effect', id=effect_id)
        profile = ET.SubElement(effect, 'profile_COMMON')
        technique = ET.SubElement(profile, 'technique', sid='common')
        phong = ET.SubElement(technique, 'phong')

        # 漫反射颜色
        diffuse = ET.SubElement(phong, 'diffuse')
        color_elem = ET.SubElement(diffuse, 'color', sid='diffuse')
        color_elem.text = f'{color[0]} {color[1]} {color[2]} 1'

        # 镜面反射
        specular = ET.SubElement(phong, 'specular')
        spec_color = ET.SubElement(specular, 'color', sid='specular')
        spec_color.text = '0.2 0.2 0.2 1'

        # 光泽度
        shininess = ET.SubElement(phong, 'shininess')
        shininess_float = ET.SubElement(shininess, 'float', sid='shininess')
        shininess_float.text = '20.0'

        # 材质
        material = ET.SubElement(library_materials, 'material', id=material_id, name=name)
        ET.SubElement(material, 'instance_effect', url=f'#{effect_id}')

    def _create_geometry(self, library_geometries: ET.Element, geometry_id: str,
                        name: str, vertices: List[Tuple[float, float, float]],
                        faces: List[Tuple[int, int, int]]):
        """创建几何体"""
        geometry = ET.SubElement(library_geometries, 'geometry', id=geometry_id, name=name)
        mesh = ET.SubElement(geometry, 'mesh')

        # 顶点位置
        positions_id = f'{geometry_id}_positions'
        source_positions = ET.SubElement(mesh, 'source', id=positions_id)
        float_array = ET.SubElement(source_positions, 'float_array',
                                   id=f'{positions_id}_array',
                                   count=str(len(vertices) * 3))

        # 填充顶点数据
        position_data = []
        for v in vertices:
            position_data.extend([str(v[0]), str(v[1]), str(v[2])])
        float_array.text = ' '.join(position_data)

        # 访问器
        technique_common = ET.SubElement(source_positions, 'technique_common')
        accessor = ET.SubElement(technique_common, 'accessor',
                                source=f'#{positions_id}_array',
                                count=str(len(vertices)),
                                stride='3')
        ET.SubElement(accessor, 'param', name='X', type='float')
        ET.SubElement(accessor, 'param', name='Y', type='float')
        ET.SubElement(accessor, 'param', name='Z', type='float')

        # 顶点
        vertices_elem = ET.SubElement(mesh, 'vertices', id=f'{geometry_id}_vertices')
        ET.SubElement(vertices_elem, 'input', semantic='POSITION', source=f'#{positions_id}')

        # 三角形
        triangles = ET.SubElement(mesh, 'triangles', count=str(len(faces)))
        ET.SubElement(triangles, 'input', semantic='VERTEX', source=f'#{geometry_id}_vertices', offset='0')

        p_elem = ET.SubElement(triangles, 'p')
        face_data = []
        for face in faces:
            face_data.extend([str(face[0]), str(face[1]), str(face[2])])
        p_elem.text = ' '.join(face_data)

    def _add_node_to_scene(self, visual_scene: ET.Element, name: str, geometry_id: str, material_id: str):
        """添加节点到场景"""
        node = ET.SubElement(visual_scene, 'node', id=f'node_{name}', name=name, type='NODE')

        # 变换矩阵(单位矩阵)
        matrix = ET.SubElement(node, 'matrix', sid='transform')
        matrix.text = '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'

        # 几何实例
        instance_geometry = ET.SubElement(node, 'instance_geometry', url=f'#{geometry_id}', name=name)
        bind_material = ET.SubElement(instance_geometry, 'bind_material')
        technique_common = ET.SubElement(bind_material, 'technique_common')
        ET.SubElement(technique_common, 'instance_material', symbol='material', target=f'#{material_id}')

    def _extract_layer_geometry(self, layer: Dict, downsample: int,
                                coord_offset: Tuple[float, float, float]) -> Tuple[List, List]:
        """提取地层几何数据"""
        grid_x = np.array(layer.get('grid_x', []))
        grid_y = np.array(layer.get('grid_y', []))
        top_z = np.array(layer.get('top_surface_z', []))
        bottom_z = np.array(layer.get('bottom_surface_z', []))

        if grid_x.size == 0 or top_z.size == 0:
            return [], []

        # 降采样
        if downsample > 1:
            if grid_x.ndim == 2:
                grid_x = grid_x[::downsample, ::downsample]
                grid_y = grid_y[::downsample, ::downsample]
                top_z = top_z[::downsample, ::downsample]
                bottom_z = bottom_z[::downsample, ::downsample]

        # 应用偏移
        grid_x = grid_x - coord_offset[0]
        grid_y = grid_y - coord_offset[1]
        top_z = top_z - coord_offset[2]
        bottom_z = bottom_z - coord_offset[2]

        # 生成网格
        vertices, faces = self._generate_solid_mesh(grid_x, grid_y, top_z, bottom_z)

        return vertices, faces

    def _generate_solid_mesh(self, grid_x, grid_y, top_z, bottom_z):
        """生成封闭的实体网格"""
        if grid_x.ndim != 2:
            return [], []

        rows, cols = grid_x.shape
        if rows < 2 or cols < 2:
            return [], []

        vertices = []
        faces = []
        vertex_map = {}

        # 创建顶点
        for surface_name, z_grid in [('top', top_z), ('bottom', bottom_z)]:
            for i in range(rows):
                for j in range(cols):
                    if not np.isnan(z_grid[i, j]):
                        idx = len(vertices)
                        vertices.append((
                            float(grid_x[i, j]),
                            float(grid_y[i, j]),
                            float(z_grid[i, j])
                        ))
                        vertex_map[(i, j, surface_name)] = idx

        # 生成面片
        # 顶面和底面
        for surface, z_grid in [('top', top_z), ('bottom', bottom_z)]:
            for i in range(rows - 1):
                for j in range(cols - 1):
                    v00 = vertex_map.get((i, j, surface))
                    v01 = vertex_map.get((i, j+1, surface))
                    v11 = vertex_map.get((i+1, j+1, surface))
                    v10 = vertex_map.get((i+1, j, surface))

                    if all(v is not None for v in [v00, v01, v11, v10]):
                        if surface == 'top':
                            # 顶面朝上
                            faces.append((v00, v01, v11))
                            faces.append((v00, v11, v10))
                        else:
                            # 底面朝下
                            faces.append((v00, v10, v11))
                            faces.append((v00, v11, v01))

        # 侧面
        for i in range(rows - 1):
            for j in range(cols - 1):
                # 检查是否是边界
                is_boundary = (i == 0 or i == rows - 2 or j == 0 or j == cols - 2)

                if is_boundary:
                    # 四个边
                    edges = [
                        ((i, j), (i, j+1)),      # 下边
                        ((i, j+1), (i+1, j+1)),  # 右边
                        ((i+1, j+1), (i+1, j)),  # 上边
                        ((i+1, j), (i, j))       # 左边
                    ]

                    for (p1, p2) in edges:
                        v1_top = vertex_map.get((*p1, 'top'))
                        v2_top = vertex_map.get((*p2, 'top'))
                        v1_bot = vertex_map.get((*p1, 'bottom'))
                        v2_bot = vertex_map.get((*p2, 'bottom'))

                        if all(v is not None for v in [v1_top, v2_top, v1_bot, v2_bot]):
                            faces.append((v1_top, v2_top, v2_bot))
                            faces.append((v1_top, v2_bot, v1_bot))

        return vertices, faces

    def _write_obj_file(self, path: str, mtl_filename: str, vertices: List, groups: List):
        """写入OBJ文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Geological Model - SketchUp Enhanced OBJ\n")
            f.write(f"# Layers: {len(groups)}\n")
            f.write(f"# Vertices: {len(vertices)}\n\n")
            f.write(f"mtllib {mtl_filename}\n\n")

            # 顶点
            f.write("# Vertices\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")

            # 分组
            for group in groups:
                f.write(f"# Layer: {group['name']}\n")
                f.write(f"g {group['name']}\n")
                f.write(f"usemtl {group['material']}\n")
                for face in group['faces']:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                f.write("\n")

    def _write_mtl_file(self, path: str, groups: List):
        """写入MTL文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Geological Model Materials\n\n")

            for group in groups:
                color = group['color']
                f.write(f"newmtl {group['material']}\n")
                f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
                f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
                f.write(f"Ks 0.2 0.2 0.2\n")
                f.write(f"Ns 50.0\n")
                f.write(f"d 1.0\n")
                f.write(f"illum 2\n\n")

    def _write_xml(self, root: ET.Element, path: str):
        """写入格式化的XML文件"""
        xml_str = ET.tostring(root, encoding='utf-8')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ', encoding='utf-8')

        with open(path, 'wb') as f:
            f.write(pretty_xml)

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

    def _get_layer_color(self, name: str, idx: int) -> Tuple[float, float, float]:
        """获取地层颜色"""
        # 精确匹配
        if name in self.ROCK_COLORS:
            return self.ROCK_COLORS[name]

        # 模糊匹配
        for key, color in self.ROCK_COLORS.items():
            if key in name:
                return color

        # 默认颜色
        return self.default_colors[idx % len(self.default_colors)]

    def _sanitize_name(self, name: str) -> str:
        """清理名称"""
        sanitized = name.replace(' ', '_').replace('/', '_')
        return ''.join(c for c in sanitized if c.isalnum() or c == '_') or 'unnamed'


def export_for_sketchup(data: Dict[str, Any], output_path: str,
                        format_type: str = 'dae', options: Optional[Dict[str, Any]] = None) -> str:
    """
    便捷导出函数

    Args:
        data: 地层数据
        output_path: 输出路径
        format_type: 格式类型 ('dae' 或 'obj')
        options: 导出选项

    Returns:
        导出文件路径
    """
    exporter = SketchUpExporter()

    if format_type.lower() == 'dae':
        return exporter.export_collada(data, output_path, options)
    elif format_type.lower() == 'obj':
        return exporter.export_enhanced_obj(data, output_path, options)
    else:
        raise ValueError(f"不支持的格式: {format_type}，请使用 'dae' 或 'obj'")


if __name__ == '__main__':
    print("SketchUp Exporter - 测试模块")
    print("支持格式: COLLADA (.dae), Enhanced OBJ (.obj)")
