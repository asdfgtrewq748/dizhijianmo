"""
OBJ格式导出器 - 用于Blender、3ds Max等第三方软件

OBJ (Wavefront OBJ) 格式特点：
1. 通用3D格式，几乎所有3D软件都支持
2. 纯文本格式，易于调试
3. 支持MTL材质文件，可定义地层颜色
4. 支持多个对象分组（每个地层一个group）
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .base_exporter import BaseExporter


class OBJExporter(BaseExporter):
    """
    导出地质模型为 OBJ 格式 (适用于 Blender、3ds Max 等)
    
    导出策略：
    1. 每个地层作为独立的group (g LayerName)
    2. 自动生成MTL材质文件
    3. 支持坐标归一化
    4. 支持降采样以减少面片数
    """
    
    # 预定义的地层颜色（RGB 0-1）
    LAYER_COLORS = [
        (0.2, 0.2, 0.2),    # 深灰 - 煤层
        (0.76, 0.70, 0.50),  # 砂黄 - 砂岩
        (0.55, 0.55, 0.55),  # 灰色 - 泥岩
        (0.65, 0.60, 0.55),  # 棕灰 - 砂质泥岩
        (0.70, 0.65, 0.50),  # 黄棕 - 粉砂岩
        (0.50, 0.50, 0.55),  # 蓝灰 - 页岩
        (0.80, 0.75, 0.65),  # 浅黄 - 石灰�ite
        (0.45, 0.40, 0.35),  # 深棕 - 炭质泥岩
    ]
    
    def __init__(self):
        super().__init__()
    
    def export(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出地质模型为OBJ格式
        
        Args:
            data: 包含地层数据的字典
            output_path: 输出文件路径 (.obj)
            options: 导出选项
                - downsample_factor: 降采样倍数（默认5）
                - normalize_coords: 是否坐标归一化（默认True）
                - export_mtl: 是否导出材质文件（默认True）
                - y_up: 是否使用Y轴朝上（默认False，Z轴朝上）
            
        Returns:
            str: 导出文件的路径
        """
        if options is None:
            options = {}
        
        downsample_factor = options.get("downsample_factor", 5)
        normalize_coords = options.get("normalize_coords", True)
        export_mtl = options.get("export_mtl", True)
        y_up = options.get("y_up", False)  # Blender默认Z-up，3ds Max默认Y-up
        
        layers = data.get("layers", [])
        if not layers:
            raise ValueError("没有可导出的地层数据")
        
        print(f"[OBJ Export] 开始导出 {len(layers)} 个地层")
        print(f"  降采样倍数: {downsample_factor}")
        print(f"  坐标归一化: {normalize_coords}")
        print(f"  导出材质: {export_mtl}")
        
        # 计算坐标偏移量
        coord_offset = self._calculate_coord_offset(layers, normalize_coords)
        
        # 确保输出路径正确
        if not output_path.endswith('.obj'):
            output_path = output_path + '.obj'
        
        # MTL文件路径
        mtl_path = output_path.replace('.obj', '.mtl')
        mtl_filename = os.path.basename(mtl_path)
        
        # 收集所有顶点和面
        all_vertices = []
        all_faces = []  # [(group_name, face_indices, material_name), ...]
        vertex_offset = 1  # OBJ索引从1开始
        
        for layer_idx, layer in enumerate(layers):
            layer_name = self._sanitize_name(layer.get("name", f"Layer_{layer_idx}"))
            material_name = f"mat_{layer_name}"
            
            print(f"\n  [处理 {layer_idx+1}/{len(layers)}] {layer_name}")
            
            # 获取网格数据
            grid_x = np.array(layer.get("grid_x", []))
            grid_y = np.array(layer.get("grid_y", []))
            top_z = np.array(layer.get("top_surface_z", []))
            bottom_z = np.array(layer.get("bottom_surface_z", []))
            
            if grid_x.size == 0 or top_z.size == 0:
                print(f"    跳过：数据为空")
                continue
            
            # 降采样
            if downsample_factor > 1:
                grid_x, grid_y, top_z, bottom_z = self._downsample(
                    grid_x, grid_y, top_z, bottom_z, downsample_factor
                )
            
            # 应用坐标偏移
            grid_x = grid_x - coord_offset[0]
            grid_y = grid_y - coord_offset[1]
            top_z = top_z - coord_offset[2]
            bottom_z = bottom_z - coord_offset[2]
            
            # 生成封闭六面体的顶点和面
            vertices, faces = self._generate_block_mesh(
                grid_x, grid_y, top_z, bottom_z, 
                vertex_offset, y_up
            )
            
            all_vertices.extend(vertices)
            all_faces.append((layer_name, faces, material_name))
            vertex_offset += len(vertices)
            
            print(f"    顶点数: {len(vertices)}, 面数: {len(faces)}")
        
        # 写入OBJ文件
        print(f"\n[OBJ Export] 写入文件: {output_path}")
        self._write_obj_file(output_path, mtl_filename, all_vertices, all_faces, export_mtl)
        
        # 写入MTL文件
        if export_mtl:
            print(f"[OBJ Export] 写入材质文件: {mtl_path}")
            self._write_mtl_file(mtl_path, all_faces, layers)
        
        print(f"[OBJ Export] 导出完成!")
        return output_path
    
    def _calculate_coord_offset(self, layers: List[Dict], normalize: bool) -> Tuple[float, float, float]:
        """计算坐标偏移量用于归一化"""
        if not normalize:
            return (0.0, 0.0, 0.0)
        
        all_x, all_y, all_z = [], [], []
        
        for layer in layers:
            grid_x = np.array(layer.get("grid_x", []))
            grid_y = np.array(layer.get("grid_y", []))
            top_z = np.array(layer.get("top_surface_z", []))
            bottom_z = np.array(layer.get("bottom_surface_z", []))
            
            if grid_x.size > 0:
                all_x.extend(grid_x.flatten())
            if grid_y.size > 0:
                all_y.extend(grid_y.flatten())
            if top_z.size > 0:
                valid_top = top_z[~np.isnan(top_z)]
                all_z.extend(valid_top.flatten())
            if bottom_z.size > 0:
                valid_bottom = bottom_z[~np.isnan(bottom_z)]
                all_z.extend(valid_bottom.flatten())
        
        if all_x and all_y and all_z:
            return (min(all_x), min(all_y), min(all_z))
        return (0.0, 0.0, 0.0)
    
    def _downsample(self, grid_x, grid_y, top_z, bottom_z, factor):
        """降采样网格数据"""
        if factor <= 1:
            return grid_x, grid_y, top_z, bottom_z
        
        # 对2D网格进行降采样
        if len(grid_x.shape) == 2:
            grid_x = grid_x[::factor, ::factor]
            grid_y = grid_y[::factor, ::factor]
            top_z = top_z[::factor, ::factor]
            if bottom_z.size > 0 and len(bottom_z.shape) == 2:
                bottom_z = bottom_z[::factor, ::factor]
        
        return grid_x, grid_y, top_z, bottom_z
    
    def _sanitize_name(self, name: str) -> str:
        """清理名称，移除OBJ不支持的字符"""
        # 替换空格和特殊字符
        sanitized = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        return sanitized or "unnamed"
    
    def _generate_block_mesh(self, grid_x, grid_y, top_z, bottom_z, 
                             vertex_offset: int, y_up: bool) -> Tuple[List, List]:
        """
        生成封闭六面体网格
        
        Returns:
            vertices: 顶点列表 [(x, y, z), ...]
            faces: 面列表 [(v1, v2, v3, v4), ...] 或 [(v1, v2, v3), ...]
        """
        vertices = []
        faces = []
        
        if len(grid_x.shape) != 2:
            return vertices, faces
        
        rows, cols = grid_x.shape
        if rows < 2 or cols < 2:
            return vertices, faces
        
        # 创建顶点映射 (row, col, is_top) -> vertex_index
        vertex_map = {}
        
        # 添加顶面和底面顶点
        for i in range(rows):
            for j in range(cols):
                x = grid_x[i, j]
                y = grid_y[i, j]
                z_top = top_z[i, j]
                z_bottom = bottom_z[i, j] if bottom_z.size > 0 else z_top - 10
                
                if np.isnan(z_top) or np.isnan(z_bottom):
                    continue
                
                # 顶面顶点
                if y_up:
                    vertices.append((x, z_top, -y))  # Y-up坐标系
                else:
                    vertices.append((x, y, z_top))   # Z-up坐标系
                vertex_map[(i, j, True)] = len(vertices) - 1 + vertex_offset
                
                # 底面顶点
                if y_up:
                    vertices.append((x, z_bottom, -y))
                else:
                    vertices.append((x, y, z_bottom))
                vertex_map[(i, j, False)] = len(vertices) - 1 + vertex_offset
        
        # 生成顶面和底面的四边形
        for i in range(rows - 1):
            for j in range(cols - 1):
                # 检查四个角点是否都有效
                corners_top = [(i, j, True), (i, j+1, True), (i+1, j+1, True), (i+1, j, True)]
                corners_bottom = [(i, j, False), (i, j+1, False), (i+1, j+1, False), (i+1, j, False)]
                
                if all(c in vertex_map for c in corners_top):
                    # 顶面（逆时针，法向量朝上）
                    faces.append(tuple(vertex_map[c] for c in corners_top))
                
                if all(c in vertex_map for c in corners_bottom):
                    # 底面（顺时针，法向量朝下）
                    faces.append(tuple(vertex_map[c] for c in reversed(corners_bottom)))
        
        # 生成四个侧面
        # 前侧面 (i=0)
        for j in range(cols - 1):
            if all(k in vertex_map for k in [(0, j, True), (0, j+1, True), (0, j, False), (0, j+1, False)]):
                faces.append((
                    vertex_map[(0, j, True)],
                    vertex_map[(0, j, False)],
                    vertex_map[(0, j+1, False)],
                    vertex_map[(0, j+1, True)]
                ))
        
        # 后侧面 (i=rows-1)
        for j in range(cols - 1):
            i = rows - 1
            if all(k in vertex_map for k in [(i, j, True), (i, j+1, True), (i, j, False), (i, j+1, False)]):
                faces.append((
                    vertex_map[(i, j+1, True)],
                    vertex_map[(i, j+1, False)],
                    vertex_map[(i, j, False)],
                    vertex_map[(i, j, True)]
                ))
        
        # 左侧面 (j=0)
        for i in range(rows - 1):
            if all(k in vertex_map for k in [(i, 0, True), (i+1, 0, True), (i, 0, False), (i+1, 0, False)]):
                faces.append((
                    vertex_map[(i+1, 0, True)],
                    vertex_map[(i+1, 0, False)],
                    vertex_map[(i, 0, False)],
                    vertex_map[(i, 0, True)]
                ))
        
        # 右侧面 (j=cols-1)
        for i in range(rows - 1):
            j = cols - 1
            if all(k in vertex_map for k in [(i, j, True), (i+1, j, True), (i, j, False), (i+1, j, False)]):
                faces.append((
                    vertex_map[(i, j, True)],
                    vertex_map[(i, j, False)],
                    vertex_map[(i+1, j, False)],
                    vertex_map[(i+1, j, True)]
                ))
        
        return vertices, faces
    
    def _write_obj_file(self, path: str, mtl_filename: str, 
                        vertices: List, faces_groups: List, use_mtl: bool):
        """写入OBJ文件"""
        with open(path, 'w', encoding='utf-8') as f:
            # 文件头
            f.write("# Geological Model OBJ Export\n")
            f.write("# Generated by Geological Modeling System\n")
            f.write(f"# Layers: {len(faces_groups)}\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Total Faces: {sum(len(fg[1]) for fg in faces_groups)}\n")
            f.write("\n")
            
            # 引用材质文件
            if use_mtl:
                f.write(f"mtllib {mtl_filename}\n\n")
            
            # 写入所有顶点
            f.write("# Vertices\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            
            # 写入每个地层的面（按组分类）
            for group_name, faces, material_name in faces_groups:
                f.write(f"# Layer: {group_name}\n")
                f.write(f"g {group_name}\n")
                if use_mtl:
                    f.write(f"usemtl {material_name}\n")
                
                for face in faces:
                    if len(face) == 4:
                        f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")
                    elif len(face) == 3:
                        f.write(f"f {face[0]} {face[1]} {face[2]}\n")
                f.write("\n")
    
    def _write_mtl_file(self, path: str, faces_groups: List, layers: List):
        """写入MTL材质文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Geological Model Material Library\n")
            f.write("# Generated by Geological Modeling System\n\n")
            
            for idx, (group_name, _, material_name) in enumerate(faces_groups):
                # 根据地层名称选择颜色
                color = self._get_layer_color(layers[idx] if idx < len(layers) else {}, idx)
                
                f.write(f"# Material for {group_name}\n")
                f.write(f"newmtl {material_name}\n")
                f.write(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")  # 环境光
                f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")  # 漫反射
                f.write(f"Ks 0.200 0.200 0.200\n")  # 镜面反射
                f.write(f"Ns 100.0\n")  # 高光指数
                f.write(f"d 1.0\n")  # 不透明度
                f.write(f"illum 2\n")  # 光照模型
                f.write("\n")
    
    def _get_layer_color(self, layer: Dict, idx: int) -> Tuple[float, float, float]:
        """根据地层类型获取颜色"""
        name = layer.get("name", "").lower()
        
        # 根据地层名称判断类型
        if "煤" in name or "coal" in name:
            return (0.15, 0.15, 0.15)  # 深黑色 - 煤层
        elif "砂岩" in name or "sandstone" in name:
            return (0.82, 0.73, 0.55)  # 砂黄色
        elif "泥岩" in name or "mudstone" in name:
            return (0.55, 0.55, 0.55)  # 灰色
        elif "砂质" in name:
            return (0.65, 0.60, 0.50)  # 棕灰
        elif "粉砂" in name or "siltstone" in name:
            return (0.70, 0.65, 0.52)  # 黄棕
        elif "页岩" in name or "shale" in name:
            return (0.50, 0.52, 0.58)  # 蓝灰
        elif "灰�ite" in name or "limestone" in name:
            return (0.78, 0.75, 0.68)  # 浅黄
        elif "炭质" in name:
            return (0.35, 0.32, 0.28)  # 深棕
        else:
            # 使用预定义颜色循环
            return self.LAYER_COLORS[idx % len(self.LAYER_COLORS)]
