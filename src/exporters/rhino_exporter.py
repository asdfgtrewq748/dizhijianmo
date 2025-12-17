"""
Rhino 专用导出器

支持格式:
1. .3dm - Rhino原生格式 (通过rhino3dm库)
2. STEP (.step/.stp) - 工业标准交换格式
3. IGES (.iges/.igs) - 通用CAD交换格式

特点:
- NURBS曲面支持
- 图层和组管理
- 材质和颜色
- 保持参数化编辑能力
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class RhinoExporter:
    """
    Rhino专用导出器

    依赖:
    - rhino3dm: pip install rhino3dm

    功能:
    1. .3dm原生格式 (推荐)
    2. STEP格式 (用于与其他CAD软件交换)
    3. IGES格式 (传统CAD交换格式)
    """

    # Rhino友好的岩石颜色 (RGB 0-255)
    ROCK_COLORS_RGB = {
        '煤': (26, 26, 26),
        '煤层': (26, 26, 26),
        '砂岩': (209, 186, 140),
        '细砂岩': (224, 204, 158),
        '中砂岩': (209, 186, 140),
        '粗砂岩': (194, 171, 122),
        '泥岩': (140, 153, 140),
        '砂质泥岩': (158, 171, 153),
        '页岩': (102, 115, 133),
        '炭质页岩': (82, 95, 107),
        '粉砂岩': (184, 168, 148),
        '灰岩': (179, 191, 204),
        '石灰岩': (179, 191, 204),
        '砾岩': (166, 128, 97),
    }

    def __init__(self):
        self.rhino3dm_available = False
        try:
            import rhino3dm as r3d
            self.r3d = r3d
            self.rhino3dm_available = True
            print("[Rhino Exporter] rhino3dm 库已加载")
        except ImportError:
            print("[Rhino Exporter] 警告: rhino3dm未安装，仅支持STEP/IGES导出")
            print("  安装方法: pip install rhino3dm")

    def export_3dm(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为Rhino .3dm格式

        Args:
            data: 地层数据
            output_path: 输出路径
            options: 选项
                - downsample_factor: 降采样(默认2)
                - normalize_coords: 是否归一化(默认True)
                - create_nurbs: 是否创建NURBS曲面(默认False，使用Mesh)

        Returns:
            输出文件路径
        """
        if not self.rhino3dm_available:
            raise ImportError("需要安装 rhino3dm: pip install rhino3dm")

        if options is None:
            options = {}

        downsample = options.get('downsample_factor', 2)
        normalize_coords = options.get('normalize_coords', True)
        create_nurbs = options.get('create_nurbs', False)

        layers_data = data.get('layers', [])
        if not layers_data:
            raise ValueError("没有可导出的地层数据")

        print(f"[Rhino 3DM Export] 开始导出 {len(layers_data)} 个地层")
        print(f"  降采样: {downsample}x")
        print(f"  坐标归一化: {normalize_coords}")
        print(f"  NURBS曲面: {create_nurbs}")

        # 计算坐标偏移
        coord_offset = self._calculate_offset(layers_data, normalize_coords)

        # 创建Rhino文档
        model = self.r3d.File3dm()
        model.Settings.ModelUnitSystem = self.r3d.UnitSystem.Meters

        # 处理每个地层
        for idx, layer in enumerate(layers_data):
            layer_name = layer.get('name', f'Layer_{idx}')

            print(f"  处理地层 {idx+1}/{len(layers_data)}: {layer_name}")

            # 创建图层
            rhino_layer = self._create_layer(model, layer_name, idx)

            # 获取几何
            vertices, faces = self._extract_layer_geometry(layer, downsample, coord_offset)

            if not faces:
                print(f"    跳过 (无有效几何)")
                continue

            if create_nurbs:
                # 创建NURBS曲面 (更适合Rhino)
                self._add_nurbs_surfaces(model, layer, downsample, coord_offset, rhino_layer.Index)
            else:
                # 创建Mesh (更简单快速)
                mesh = self._create_rhino_mesh(vertices, faces)
                attrs = self.r3d.ObjectAttributes()
                attrs.LayerIndex = rhino_layer.Index
                model.Objects.AddMesh(mesh, attrs)

            print(f"    顶点: {len(vertices)}, 面: {len(faces)}")

        # 保存文件
        output_path = str(Path(output_path).with_suffix('.3dm'))
        model.Write(output_path, 7)  # 版本7 (Rhino 7)

        print(f"[Rhino 3DM Export] 导出完成: {output_path}")
        print(f"  提示: 在Rhino中直接打开此文件")

        return output_path

    def export_step(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为STEP格式 (工业标准)

        STEP (Standard for the Exchange of Product model data)
        - ISO 10303标准
        - 支持精确的NURBS几何
        - 广泛用于CAD/CAM/CAE软件交换
        """
        if options is None:
            options = {}

        downsample = options.get('downsample_factor', 2)
        normalize_coords = options.get('normalize_coords', True)

        layers_data = data.get('layers', [])
        if not layers_data:
            raise ValueError("没有可导出的地层数据")

        print(f"[Rhino STEP Export] 开始导出 {len(layers_data)} 个地层")

        # 如果有rhino3dm，使用它导出STEP
        if self.rhino3dm_available:
            # 先生成3dm，然后让用户在Rhino中另存为STEP
            temp_3dm = output_path.replace('.step', '_temp.3dm').replace('.stp', '_temp.3dm')
            self.export_3dm(data, temp_3dm, options)

            print(f"[Rhino STEP Export] 已生成临时3DM文件: {temp_3dm}")
            print(f"  请在Rhino中打开此文件，然后使用 文件 > 另存为 > STEP (.step)")
            print(f"  或者使用Rhino命令: _Export '{Path(output_path).absolute()}' _Enter")

            return temp_3dm
        else:
            # 导出为简单的STEP文本格式 (基础支持)
            output_path = str(Path(output_path).with_suffix('.step'))
            self._export_step_text(data, output_path, options)
            return output_path

    def export_iges(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出为IGES格式 (传统CAD交换格式)

        IGES (Initial Graphics Exchange Specification)
        - 传统CAD交换标准
        - 广泛支持但精度略低于STEP
        """
        if options is None:
            options = {}

        layers_data = data.get('layers', [])
        if not layers_data:
            raise ValueError("没有可导出的地层数据")

        print(f"[Rhino IGES Export] 开始导出 {len(layers_data)} 个地层")

        if self.rhino3dm_available:
            # 先生成3dm
            temp_3dm = output_path.replace('.iges', '_temp.3dm').replace('.igs', '_temp.3dm')
            self.export_3dm(data, temp_3dm, options)

            print(f"[Rhino IGES Export] 已生成临时3DM文件: {temp_3dm}")
            print(f"  请在Rhino中打开此文件，然后使用 文件 > 另存为 > IGES (.iges)")

            return temp_3dm
        else:
            output_path = str(Path(output_path).with_suffix('.iges'))
            self._export_iges_text(data, output_path, options)
            return output_path

    def _create_layer(self, model, name: str, idx: int):
        """创建Rhino图层"""
        layer = self.r3d.Layer()
        layer.Name = name

        # 设置颜色
        color = self._get_layer_color_rgb(name, idx)
        layer.Color = color

        # 添加到模型
        layer_index = model.Layers.Add(layer)
        return model.Layers[layer_index]

    def _create_rhino_mesh(self, vertices: List[Tuple], faces: List[Tuple]):
        """创建Rhino Mesh对象"""
        mesh = self.r3d.Mesh()

        # 添加顶点
        for v in vertices:
            mesh.Vertices.Add(v[0], v[1], v[2])

        # 添加面
        for f in faces:
            mesh.Faces.AddFace(f[0], f[1], f[2])

        return mesh

    def _add_nurbs_surfaces(self, model, layer: Dict, downsample: int,
                           coord_offset: Tuple, layer_index: int):
        """添加NURBS曲面 (更适合Rhino编辑)"""
        grid_x = np.array(layer.get('grid_x', []))
        grid_y = np.array(layer.get('grid_y', []))
        top_z = np.array(layer.get('top_surface_z', []))
        bottom_z = np.array(layer.get('bottom_surface_z', []))

        if grid_x.size == 0 or top_z.size == 0:
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

        # 创建顶面NURBS
        top_surface = self.r3d.NurbsSurface.Create(3, False, 2, 2, cols, rows)

        for i in range(rows):
            for j in range(cols):
                if not np.isnan(top_z[i, j]):
                    point = self.r3d.Point3d(grid_x[i, j], grid_y[i, j], top_z[i, j])
                    top_surface.Points.SetPoint(j, i, point)

        attrs = self.r3d.ObjectAttributes()
        attrs.LayerIndex = layer_index
        model.Objects.AddSurface(top_surface, attrs)

        # 创建底面NURBS
        bottom_surface = self.r3d.NurbsSurface.Create(3, False, 2, 2, cols, rows)

        for i in range(rows):
            for j in range(cols):
                if not np.isnan(bottom_z[i, j]):
                    point = self.r3d.Point3d(grid_x[i, j], grid_y[i, j], bottom_z[i, j])
                    bottom_surface.Points.SetPoint(j, i, point)

        model.Objects.AddSurface(bottom_surface, attrs)

    def _export_step_text(self, data: Dict, output_path: str, options: Dict):
        """导出简单的STEP文本格式"""
        layers_data = data.get('layers', [])
        downsample = options.get('downsample_factor', 2)
        coord_offset = self._calculate_offset(layers_data, options.get('normalize_coords', True))

        with open(output_path, 'w', encoding='utf-8') as f:
            # STEP文件头
            f.write("ISO-10303-21;\n")
            f.write("HEADER;\n")
            f.write("FILE_DESCRIPTION(('Geological Model'), '2;1');\n")
            f.write("FILE_NAME('geological_model.step', '2025-01-01', ('Author'), ('Organization'), '', '', '');\n")
            f.write("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n")
            f.write("ENDSEC;\n\n")
            f.write("DATA;\n")

            entity_id = 1

            # 简化的几何输出
            for idx, layer in enumerate(layers_data):
                layer_name = layer.get('name', f'Layer_{idx}')
                vertices, faces = self._extract_layer_geometry(layer, downsample, coord_offset)

                if not faces:
                    continue

                f.write(f"/* Layer: {layer_name} */\n")

                # 这里可以添加更详细的STEP实体定义
                # 为简化，仅添加注释
                f.write(f"/* {len(vertices)} vertices, {len(faces)} faces */\n\n")

            f.write("ENDSEC;\n")
            f.write("END-ISO-10303-21;\n")

        print(f"[STEP Export] 基础STEP文件已导出: {output_path}")
        print(f"  注意: 这是简化的STEP格式，建议使用Rhino导出完整STEP")

    def _export_iges_text(self, data: Dict, output_path: str, options: Dict):
        """导出简单的IGES文本格式"""
        layers_data = data.get('layers', [])

        with open(output_path, 'w', encoding='utf-8') as f:
            # IGES文件头
            f.write("                                                                        S      1\n")
            f.write("1H,,1H;,10HGeological,13HModel Export,,,,,,,,,,,                       G      1\n")

            # 简化的几何输出
            f.write("# Simplified IGES export - recommend using Rhino for full IGES       D      1\n")

            f.write("S      1G      1D      1P      0                                        T      1\n")

        print(f"[IGES Export] 基础IGES文件已导出: {output_path}")
        print(f"  注意: 这是简化的IGES格式，建议使用Rhino导出完整IGES")

    def _extract_layer_geometry(self, layer: Dict, downsample: int,
                                coord_offset: Tuple) -> Tuple[List, List]:
        """提取地层几何"""
        grid_x = np.array(layer.get('grid_x', []))
        grid_y = np.array(layer.get('grid_y', []))
        top_z = np.array(layer.get('top_surface_z', []))
        bottom_z = np.array(layer.get('bottom_surface_z', []))

        if grid_x.size == 0 or top_z.size == 0:
            return [], []

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

        # 生成封闭网格
        vertices, faces = self._generate_solid_mesh(grid_x, grid_y, top_z, bottom_z)

        return vertices, faces

    def _generate_solid_mesh(self, grid_x, grid_y, top_z, bottom_z):
        """生成封闭实体网格"""
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

        # 生成面片 (顶面、底面、侧面)
        for surface, z_grid in [('top', top_z), ('bottom', bottom_z)]:
            for i in range(rows - 1):
                for j in range(cols - 1):
                    v00 = vertex_map.get((i, j, surface))
                    v01 = vertex_map.get((i, j+1, surface))
                    v11 = vertex_map.get((i+1, j+1, surface))
                    v10 = vertex_map.get((i+1, j, surface))

                    if all(v is not None for v in [v00, v01, v11, v10]):
                        if surface == 'top':
                            faces.append((v00, v01, v11))
                            faces.append((v00, v11, v10))
                        else:
                            faces.append((v00, v10, v11))
                            faces.append((v00, v11, v01))

        # 侧面 (简化版，仅边界)
        # ...省略侧面生成逻辑，与SketchUp导出器类似

        return vertices, faces

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

    def _get_layer_color_rgb(self, name: str, idx: int) -> tuple:
        """获取RGB颜色 (0-255)"""
        # 精确匹配
        if name in self.ROCK_COLORS_RGB:
            color = self.ROCK_COLORS_RGB[name]
            if self.rhino3dm_available:
                return self.r3d.Color4f(color[0]/255, color[1]/255, color[2]/255, 1.0)
            return color

        # 模糊匹配
        for key, color in self.ROCK_COLORS_RGB.items():
            if key in name:
                if self.rhino3dm_available:
                    return self.r3d.Color4f(color[0]/255, color[1]/255, color[2]/255, 1.0)
                return color

        # 默认颜色
        default_colors = [
            (230, 100, 80), (100, 180, 230), (80, 190, 150),
            (200, 180, 100), (150, 130, 180)
        ]
        color = default_colors[idx % len(default_colors)]

        if self.rhino3dm_available:
            return self.r3d.Color4f(color[0]/255, color[1]/255, color[2]/255, 1.0)
        return color


def export_for_rhino(data: Dict[str, Any], output_path: str,
                     format_type: str = '3dm', options: Optional[Dict[str, Any]] = None) -> str:
    """
    便捷导出函数

    Args:
        data: 地层数据
        output_path: 输出路径
        format_type: 格式类型 ('3dm', 'step', 'iges')
        options: 导出选项

    Returns:
        导出文件路径
    """
    exporter = RhinoExporter()

    format_type = format_type.lower()

    if format_type == '3dm':
        return exporter.export_3dm(data, output_path, options)
    elif format_type in ['step', 'stp']:
        return exporter.export_step(data, output_path, options)
    elif format_type in ['iges', 'igs']:
        return exporter.export_iges(data, output_path, options)
    else:
        raise ValueError(f"不支持的格式: {format_type}，请使用 '3dm', 'step' 或 'iges'")


if __name__ == '__main__':
    print("Rhino Exporter - 测试模块")
    print("支持格式: .3dm (Rhino), STEP (.step), IGES (.iges)")
    print("安装rhino3dm: pip install rhino3dm")
