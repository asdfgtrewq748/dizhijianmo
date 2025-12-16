import struct
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .base_exporter import BaseExporter


class STLExporter(BaseExporter):
    """
    导出地质模型为 STL 格式 (适用于 FLAC3D 数值模拟)
    
    STL格式特点：
    1. 三角面片网格格式，FLAC3D原生支持
    2. 支持二进制和ASCII两种格式
    3. 文件小，导入快
    4. 适合单一封闭体块
    
    导出策略：
    1. 分层导出：每个地层导出为独立的STL文件
    2. 封闭体块：每层包含完整的六面体（顶、底、四个侧面）
    3. 三角化：将四边形面分解为两个三角形
    4. 法向量：自动计算并确保朝外
    """
    
    def __init__(self):
        super().__init__()
        self.format = 'binary'  # 默认使用二进制格式
    
    def export(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出单层或多层地质模型为STL格式
        
        Args:
            data: 包含地层数据的字典，格式如下:
                  {
                      "layers": [
                          {
                              "name": "LayerName",
                              "grid_x": np.ndarray,
                              "grid_y": np.ndarray,
                              "top_surface_z": np.ndarray,
                              "bottom_surface_z": np.ndarray
                          },
                          ...
                      ]
                  }
            output_path: 输出文件路径 (.stl)
            options: 导出选项
                - format: 'binary' 或 'ascii' (默认binary)
                - downsample_factor: 降采样倍数（默认5）
                - normalize_coords: 是否坐标归一化（默认True）
                - single_layer_index: 仅导出指定索引的单层（可选）
            
        Returns:
            str: 导出文件的路径
        """
        if options is None:
            options = {}
        
        self.format = options.get("format", "binary")
        downsample_factor = options.get("downsample_factor", 5)
        normalize_coords = options.get("normalize_coords", True)
        single_layer_index = options.get("single_layer_index", None)
        global_coord_offset = options.get("global_coord_offset", None)  # 全局偏移量（用于分层导出）
        
        all_layers = data.get("layers", [])
        if not all_layers:
            raise ValueError("没有可导出的地层数据")
        
        # 如果指定了单层索引，只导出该层
        if single_layer_index is not None:
            if 0 <= single_layer_index < len(all_layers):
                layers = [all_layers[single_layer_index]]
                print(f"[STL Export] 单层导出模式: {layers[0].get('name')}")
            else:
                raise ValueError(f"无效的图层索引: {single_layer_index}")
        else:
            layers = all_layers
            print(f"[STL Export] 多层导出模式: {len(layers)} 个地层")
        
        # 坐标归一化：分层导出时使用全局偏移量，否则基于当前层计算
        if global_coord_offset is not None:
            # 分层导出模式：使用预计算的全局偏移量
            coord_offset = global_coord_offset
            print(f"  [分层模式] 使用全局偏移量: X={coord_offset[0]:.2f}, Y={coord_offset[1]:.2f}, Z={coord_offset[2]:.2f}")
        elif normalize_coords:
            # 整体导出模式：基于所有层计算偏移量
            coord_offset = self._calculate_coord_offset(all_layers, normalize_coords)
        else:
            coord_offset = (0, 0, 0)
        
        # 收集所有三角面片
        all_triangles = []
        
        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get("name", f"Layer_{layer_idx}")
            print(f"\n  [处理 {layer_idx+1}/{len(layers)}] {layer_name}")
            print(f"    可用字段: {list(layer.keys())}")
            
            # 获取该层的顶面和底面数据
            print(f"    准备顶面数据...")
            top_grids = self._prepare_grid_data(layer, downsample_factor, coord_offset, use_bottom=False)
            
            if top_grids is None:
                print(f"    [ERROR] 顶面数据无效,跳过此层")
                continue
            
            print(f"    准备底面数据...")
            bottom_grids = self._prepare_grid_data(layer, downsample_factor, coord_offset, use_bottom=True)
            
            if bottom_grids is None:
                print(f"    [ERROR] 底面数据无效,跳过此层")
                continue
            
            top_x, top_y, top_z = top_grids
            bottom_x, bottom_y, bottom_z = bottom_grids
            
            # 🔧 信任建模阶段的逐列排序,不再在导出时修改Z值
            # (之前的"导出阶段修复"会破坏精确的层间对齐)
            top_z_min = float(np.nanmin(top_z))
            top_z_max = float(np.nanmax(top_z))
            bottom_z_min = float(np.nanmin(bottom_z))
            bottom_z_max = float(np.nanmax(bottom_z))
            
            print(f"    [Z范围] 顶面: [{top_z_min:.2f}, {top_z_max:.2f}]m")
            print(f"    [Z范围] 底面: [{bottom_z_min:.2f}, {bottom_z_max:.2f}]m")
            
            # 仅检查但不修复(修复应该在建模阶段完成)
            if top_z_min < bottom_z_max:
                print(f"    [WARNING] {layer_name} 顶底面存在交错!")
                print(f"              顶面最小({top_z_min:.2f}m) < 底面最大({bottom_z_max:.2f}m)")
                print(f"              请检查建模阶段的逐列排序是否正确执行")
                # 不再修改Z值,信任建模阶段的数据
            
            # 验证厚度
            thickness = top_z - bottom_z
            avg_thickness = np.nanmean(thickness)
            min_thickness = np.nanmin(thickness)
            
            if avg_thickness < 1e-6:
                print(f"    [WARNING] 跳过此层(厚度过小: {avg_thickness:.6f}m)")
                continue
            
            if min_thickness < 0:
                print(f"    [ERROR] 检测到负厚度! 最小厚度: {min_thickness:.2f}m")
                print(f"            这说明建模阶段的修复未生效,请检查日志")
            
            print(f"    [厚度] 范围: [{min_thickness:.2f}, {np.nanmax(thickness):.2f}]m (平均: {avg_thickness:.2f}m)")
            
            # 生成该层的三角面片
            layer_triangles = self._build_triangulated_block(
                top_x, top_y, top_z,
                bottom_x, bottom_y, bottom_z
            )
            
            all_triangles.extend(layer_triangles)
            print(f"    [OK] 生成 {len(layer_triangles)} 个三角面片")
        
        if not all_triangles:
            raise ValueError("未能生成任何有效的三角面片")
        
        print(f"[STL Export] 总共 {len(all_triangles)} 个三角面片")
        
        # 写入STL文件
        if self.format == 'binary':
            self._write_binary_stl(output_path, all_triangles)
        else:
            self._write_ascii_stl(output_path, all_triangles)
        
        print(f"[STL Export] 文件已保存: {output_path}")
        return output_path
    
    def _calculate_coord_offset(self, layers: List[Dict], normalize_coords: bool = True) -> Tuple[float, float, float]:
        """计算坐标偏移量"""
        all_x, all_y, all_z = [], [], []
        
        for layer in layers:
            for z_field in ['top_surface_z', 'grid_z', 'bottom_surface_z', 'grid_z_bottom']:
                grid_x = layer.get("grid_x")
                grid_y = layer.get("grid_y")
                grid_z = layer.get(z_field)
                
                if grid_x is not None and grid_y is not None and grid_z is not None:
                    grid_x = np.array(grid_x).flatten()
                    grid_y = np.array(grid_y).flatten()
                    grid_z = np.array(grid_z).flatten()
                    
                    valid = ~(np.isnan(grid_x) | np.isnan(grid_y) | np.isnan(grid_z))
                    all_x.extend(grid_x[valid])
                    all_y.extend(grid_y[valid])
                    all_z.extend(grid_z[valid])
        
        if not all_x:
            print(f"  [警告] 无法计算坐标偏移量，使用(0,0,0)")
            return (0, 0, 0)
        
        # 使用中位数作为XY偏移，使用最小值作为Z偏移
        offset_x = float(np.median(all_x))
        offset_y = float(np.median(all_y))
        offset_z = float(np.min(all_z))
        
        # 输出详细的坐标统计信息
        print(f"  [坐标统计] 收集了 {len(all_x)} 个坐标点")
        print(f"  [原始坐标] X范围: [{np.min(all_x):.2f}, {np.max(all_x):.2f}], 中位数: {offset_x:.2f}")
        print(f"  [原始坐标] Y范围: [{np.min(all_y):.2f}, {np.max(all_y):.2f}], 中位数: {offset_y:.2f}")
        print(f"  [原始坐标] Z范围: [{np.min(all_z):.2f}, {np.max(all_z):.2f}], 最小值: {offset_z:.2f}")
        
        # 显示归一化状态和效果
        if not normalize_coords:
            print(f"  [归一化] [DISABLED] 未启用 - 将使用原始坐标导出")
            if max(abs(offset_x), abs(offset_y)) > 1e6:
                print(f"  [警告] [WARNING] 检测到超大坐标值(百万级别),FLAC3D可能出现精度问题!")
                print(f"  [建议] 强烈建议启用坐标归一化(normalize_coords=True)")
        else:
            print(f"  [归一化] [OK] 已启用 - 偏移量: X={offset_x:.2f}, Y={offset_y:.2f}, Z={offset_z:.2f}")
            # 计算归一化后的范围
            norm_x_min = np.min(all_x) - offset_x
            norm_x_max = np.max(all_x) - offset_x
            norm_y_min = np.min(all_y) - offset_y
            norm_y_max = np.max(all_y) - offset_y
            norm_z_min = np.min(all_z) - offset_z
            norm_z_max = np.max(all_z) - offset_z
            print(f"  [归一化后] X范围: [{norm_x_min:.2f}, {norm_x_max:.2f}] (跨度: {norm_x_max-norm_x_min:.2f}m)")
            print(f"  [归一化后] Y范围: [{norm_y_min:.2f}, {norm_y_max:.2f}] (跨度: {norm_y_max-norm_y_min:.2f}m)")
            print(f"  [归一化后] Z范围: [{norm_z_min:.2f}, {norm_z_max:.2f}] (跨度: {norm_z_max-norm_z_min:.2f}m)")
        
        return (offset_x, offset_y, offset_z)
    
    def _prepare_grid_data(self, layer: Dict, downsample: int, offset: Tuple[float, float, float], 
                           use_bottom: bool = False) -> Optional[Tuple]:
        """准备并降采样网格数据"""
        layer_name = layer.get("name", "Unknown")
        
        # 获取X、Y网格
        grid_x = layer.get("grid_x")
        grid_y = layer.get("grid_y")
        
        # 调试：打印可用的键
        if grid_x is None or grid_y is None:
            print(f"      [调试] {layer_name} 可用键: {list(layer.keys())}")
            print(f"      [错误] 缺少grid_x或grid_y")
            return None
        
        # 获取Z值（顶面或底面）
        if use_bottom:
            grid_z = layer.get("bottom_surface_z")
            if grid_z is None:
                grid_z = layer.get("grid_z_bottom")
            if grid_z is None:
                top_z = layer.get("top_surface_z")
                if top_z is None:
                    top_z = layer.get("grid_z")
                thickness = layer.get("thickness")
                if top_z is not None and thickness is not None:
                    grid_z = np.array(top_z) - np.array(thickness)
                    print(f"      [计算] 底面 = 顶面 - 厚度")
        else:
            grid_z = layer.get("top_surface_z")
            if grid_z is None:
                grid_z = layer.get("grid_z")
        
        if grid_x is None or grid_y is None or grid_z is None:
            print(f"      [错误] {layer_name} 数据不完整: X={grid_x is not None}, Y={grid_y is not None}, Z={grid_z is not None}")
            return None
        
        # 转换为numpy数组
        grid_x = np.array(grid_x)
        grid_y = np.array(grid_y)
        grid_z = np.array(grid_z)
        
        # 检查数据形状
        print(f"      [数据] X shape: {grid_x.shape}, Y shape: {grid_y.shape}, Z shape: {grid_z.shape}")
        
        # 如果X、Y是一维数组，创建网格
        if grid_x.ndim == 1 and grid_y.ndim == 1:
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            print(f"      [网格] 创建meshgrid: {grid_x.shape}")
        
        # Z必须是2维
        if grid_z.ndim != 2:
            print(f"      [错误] Z维度错误: {grid_z.ndim}, 期望2")
            return None
        
        # 检查形状匹配
        if grid_x.shape != grid_z.shape or grid_y.shape != grid_z.shape:
            print(f"      [错误] 形状不匹配: X{grid_x.shape}, Y{grid_y.shape}, Z{grid_z.shape}")
            return None
        
        # 降采样前检查数据范围
        print(f"      [原始] X范围: [{np.nanmin(grid_x):.2f}, {np.nanmax(grid_x):.2f}]")
        print(f"      [原始] Y范围: [{np.nanmin(grid_y):.2f}, {np.nanmax(grid_y):.2f}]")
        print(f"      [原始] Z范围: [{np.nanmin(grid_z):.2f}, {np.nanmax(grid_z):.2f}]")
        
        # 填充NaN值以避免孔洞（处理所有坐标轴）
        nan_x = np.isnan(grid_x).sum()
        nan_y = np.isnan(grid_y).sum()
        nan_z = np.isnan(grid_z).sum()
        total_nan = nan_x + nan_y + nan_z
        
        if total_nan > 0:
            print(f"      [填充] 检测到NaN值 - X:{nan_x}, Y:{nan_y}, Z:{nan_z}")
            from scipy.interpolate import griddata
            
            # 创建完全有效的掩码（X、Y、Z都有效）
            valid_mask = ~(np.isnan(grid_x) | np.isnan(grid_y) | np.isnan(grid_z))
            valid_count = valid_mask.sum()
            
            print(f"      [填充] 完全有效点数: {valid_count}/{grid_z.size}")
            
            if valid_count >= 3:
                # 获取有效点的索引
                rows, cols = grid_z.shape
                row_idx, col_idx = np.mgrid[0:rows, 0:cols]
                
                valid_row = row_idx[valid_mask]
                valid_col = col_idx[valid_mask]
                
                # 分别填充X、Y、Z
                for name, grid in [('X', grid_x), ('Y', grid_y), ('Z', grid_z)]:
                    if np.isnan(grid).sum() > 0:
                        # 使用行列索引作为坐标进行插值
                        valid_positions = np.column_stack([valid_row, valid_col])
                        valid_values = grid[valid_mask]
                        
                        # 需要填充的点
                        nan_mask = np.isnan(grid)
                        if nan_mask.sum() > 0:
                            nan_row = row_idx[nan_mask]
                            nan_col = col_idx[nan_mask]
                            nan_positions = np.column_stack([nan_row, nan_col])
                            
                            # 最近邻插值
                            filled = griddata(valid_positions, valid_values, 
                                            nan_positions, method='nearest')
                            grid[nan_mask] = filled
                            print(f"      [填充] {name}坐标: {nan_mask.sum()} 个NaN已填充")
            else:
                # 有效点不足，使用简单填充
                print(f"      [填充] 有效点不足，使用均值填充")
                grid_x = np.nan_to_num(grid_x, nan=np.nanmean(grid_x))
                grid_y = np.nan_to_num(grid_y, nan=np.nanmean(grid_y))
                grid_z = np.nan_to_num(grid_z, nan=np.nanmean(grid_z))
        
        # 最终检查：确保没有残留NaN
        final_nan = np.isnan(grid_x).sum() + np.isnan(grid_y).sum() + np.isnan(grid_z).sum()
        if final_nan > 0:
            print(f"      [警告] 仍有 {final_nan} 个NaN，用有效值填充")
            # WARNING: 不能用0填充,Z坐标=0会影响厚度计算
            # 用最近邻有效值填充
            if np.isnan(grid_x).any():
                valid_x = grid_x[~np.isnan(grid_x)]
                grid_x = np.nan_to_num(grid_x, nan=float(np.mean(valid_x)) if len(valid_x) > 0 else 0.0)
            if np.isnan(grid_y).any():
                valid_y = grid_y[~np.isnan(grid_y)]
                grid_y = np.nan_to_num(grid_y, nan=float(np.mean(valid_y)) if len(valid_y) > 0 else 0.0)
            if np.isnan(grid_z).any():
                valid_z = grid_z[~np.isnan(grid_z)]
                grid_z = np.nan_to_num(grid_z, nan=float(np.mean(valid_z)) if len(valid_z) > 0 else 0.0)
            print(f"      [OK] NaN已用有效值均值填充")
        
        # 检测异常值（如坐标过大）
        if np.nanmax(np.abs(grid_x)) > 1e7 or np.nanmax(np.abs(grid_y)) > 1e7:
            print(f"      [警告] 检测到超大坐标值，强烈建议启用坐标归一化！")
        
        # 降采样
        grid_x = grid_x[::downsample, ::downsample]
        grid_y = grid_y[::downsample, ::downsample]
        grid_z = grid_z[::downsample, ::downsample]
        
        print(f"      [降采样] 新形状: {grid_x.shape}, 降采样率: {downsample}x")
        
        # 坐标归一化
        grid_x = grid_x - offset[0]
        grid_y = grid_y - offset[1]
        grid_z = grid_z - offset[2]
        
        # 归一化后的范围
        print(f"      [归一化] X范围: [{np.nanmin(grid_x):.2f}, {np.nanmax(grid_x):.2f}]")
        print(f"      [归一化] Y范围: [{np.nanmin(grid_y):.2f}, {np.nanmax(grid_y):.2f}]")
        print(f"      [归一化] Z范围: [{np.nanmin(grid_z):.2f}, {np.nanmax(grid_z):.2f}]")
        
        # 检查有效数据点数量
        valid_count = np.sum(~np.isnan(grid_z))
        total_count = grid_z.size
        valid_ratio = valid_count / total_count if total_count > 0 else 0
        
        print(f"      [有效性] {valid_count}/{total_count} 点有效 ({valid_ratio*100:.1f}%)")
        
        if valid_count < 4:
            print(f"      [错误] 有效数据点不足（至少需要4个点）")
            return None
        
        return (grid_x, grid_y, grid_z)
    
    def _ensure_closed_boundary(self, grid_x, grid_y, grid_z):
        """
        确保网格边界闭合：处理边界上的所有NaN值
        """
        rows, cols = grid_z.shape
        
        # 对三个坐标分别处理
        for grid_name, grid in [('Z', grid_z), ('X', grid_x), ('Y', grid_y)]:
            # 检查并修复四条边界
            for i in range(rows):
                # 左边界
                if np.isnan(grid[i, 0]):
                    for j in range(1, cols):
                        if not np.isnan(grid[i, j]):
                            grid[i, 0] = grid[i, j]
                            break
                # 右边界
                if np.isnan(grid[i, -1]):
                    for j in range(cols-2, -1, -1):
                        if not np.isnan(grid[i, j]):
                            grid[i, -1] = grid[i, j]
                            break
            
            for j in range(cols):
                # 上边界
                if np.isnan(grid[0, j]):
                    for i in range(1, rows):
                        if not np.isnan(grid[i, j]):
                            grid[0, j] = grid[i, j]
                            break
                # 下边界
                if np.isnan(grid[-1, j]):
                    for i in range(rows-2, -1, -1):
                        if not np.isnan(grid[i, j]):
                            grid[-1, j] = grid[i, j]
                            break
            
            # 四个角点必须有效
            corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
            for r, c in corners:
                if np.isnan(grid[r, c]):
                    # 使用最近的有效邻居
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = (0 if r == 0 else rows-1) + di, (0 if c == 0 else cols-1) + dj
                            if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(grid[ni, nj]):
                                grid[r, c] = grid[ni, nj]
                                break
                        if not np.isnan(grid[r, c]):
                            break
        
        return grid_x, grid_y, grid_z
    
    def _build_triangulated_block(self, top_x, top_y, top_z, bottom_x, bottom_y, bottom_z) -> List[Dict]:
        """
        构建流形三角网格 (Manifold Mesh)
        
        核心改进:
        1. 使用顶点索引表,避免重复顶点
        2. 确保每条边恰好被2个三角形共享
        3. 生成完全闭合的网格
        
        返回格式: [{"vertices": [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)], "normal": (nx,ny,nz)}, ...]
        """
        rows, cols = top_z.shape
        
        # 确保边界闭合
        top_x, top_y, top_z = self._ensure_closed_boundary(top_x, top_y, top_z)
        bottom_x, bottom_y, bottom_z = self._ensure_closed_boundary(bottom_x, bottom_y, bottom_z)
        
        # Step 1: 构建唯一顶点索引表
        vertex_dict = {}  # {(x,y,z): index}
        vertex_list = []  # [(x,y,z), ...]
        vertex_counter = 0
        
        def add_vertex(x, y, z):
            """添加顶点到索引表,如果已存在则返回现有索引"""
            nonlocal vertex_counter
            # 使用浮点数容差来判断顶点是否相同
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key not in vertex_dict:
                vertex_dict[key] = vertex_counter
                vertex_list.append((float(x), float(y), float(z)))
                vertex_counter += 1
            return vertex_dict[key]
        
        # Step 2: 为所有有效网格点创建顶点索引
        top_indices = np.full((rows, cols), -1, dtype=int)
        bottom_indices = np.full((rows, cols), -1, dtype=int)
        
        valid_top = ~(np.isnan(top_x) | np.isnan(top_y) | np.isnan(top_z))
        valid_bottom = ~(np.isnan(bottom_x) | np.isnan(bottom_y) | np.isnan(bottom_z))
        
        for r in range(rows):
            for c in range(cols):
                if valid_top[r, c]:
                    top_indices[r, c] = add_vertex(top_x[r, c], top_y[r, c], top_z[r, c])
                if valid_bottom[r, c]:
                    bottom_indices[r, c] = add_vertex(bottom_x[r, c], bottom_y[r, c], bottom_z[r, c])
        
        print(f"    [Manifold] 创建了 {len(vertex_list)} 个唯一顶点 (原始网格: {rows}x{cols}x2 = {rows*cols*2})")
        
        # Step 3: 生成表面三角形(只生成外表面,不生成内部重复的面)
        triangles = []
        
        def add_triangle_by_indices(idx1, idx2, idx3, expected_normal):
            """通过顶点索引添加三角形"""
            v1, v2, v3 = vertex_list[idx1], vertex_list[idx2], vertex_list[idx3]
            tri = self._create_triangle([v1, v2, v3], expected_normal)
            if tri:
                triangles.append(tri)
        
        # Step 4a: 生成顶面和底面三角形
        # 策略: 遍历所有网格四边形,为顶面和底面各生成2个三角形
        cell_count = 0
        for r in range(rows - 1):
            for c in range(cols - 1):
                t_tl = top_indices[r, c]
                t_tr = top_indices[r, c+1]
                t_br = top_indices[r+1, c+1]
                t_bl = top_indices[r+1, c]
                
                b_tl = bottom_indices[r, c]
                b_tr = bottom_indices[r, c+1]
                b_br = bottom_indices[r+1, c+1]
                b_bl = bottom_indices[r+1, c]
                
                # 检查顶点有效性
                if any(idx < 0 for idx in [t_tl, t_tr, t_br, t_bl, b_tl, b_tr, b_br, b_bl]):
                    continue
                
                # 厚度检查
                thickness_corners = [
                    vertex_list[t_tl][2] - vertex_list[b_tl][2],
                    vertex_list[t_tr][2] - vertex_list[b_tr][2],
                    vertex_list[t_br][2] - vertex_list[b_br][2],
                    vertex_list[t_bl][2] - vertex_list[b_bl][2]
                ]
                avg_thickness = sum(thickness_corners) / 4.0
                min_thickness = min(thickness_corners)
                
                if avg_thickness < 0.1 or min_thickness < 0:
                    continue
                
                cell_count += 1
                
                # 顶面 (2个三角形)
                add_triangle_by_indices(t_tl, t_tr, t_br, (0, 0, 1))
                add_triangle_by_indices(t_tl, t_br, t_bl, (0, 0, 1))
                
                # 底面 (2个三角形)
                add_triangle_by_indices(b_tl, b_bl, b_br, (0, 0, -1))
                add_triangle_by_indices(b_tl, b_br, b_tr, (0, 0, -1))
        
        # Step 4b: 生成四周侧面(只在边界处生成)
        # 前侧面 (row=0)
        for c in range(cols - 1):
            t_tl, t_tr = top_indices[0, c], top_indices[0, c+1]
            b_tl, b_tr = bottom_indices[0, c], bottom_indices[0, c+1]
            if all(idx >= 0 for idx in [t_tl, t_tr, b_tl, b_tr]):
                add_triangle_by_indices(t_tl, b_tl, b_tr, (0, -1, 0))
                add_triangle_by_indices(t_tl, b_tr, t_tr, (0, -1, 0))
        
        # 后侧面 (row=rows-1)
        for c in range(cols - 1):
            t_tl, t_tr = top_indices[rows-1, c], top_indices[rows-1, c+1]
            b_tl, b_tr = bottom_indices[rows-1, c], bottom_indices[rows-1, c+1]
            if all(idx >= 0 for idx in [t_tl, t_tr, b_tl, b_tr]):
                add_triangle_by_indices(t_tr, b_tr, b_tl, (0, 1, 0))
                add_triangle_by_indices(t_tr, b_tl, t_tl, (0, 1, 0))
        
        # 左侧面 (col=0)
        for r in range(rows - 1):
            t_tl, t_bl = top_indices[r, 0], top_indices[r+1, 0]
            b_tl, b_bl = bottom_indices[r, 0], bottom_indices[r+1, 0]
            if all(idx >= 0 for idx in [t_tl, t_bl, b_tl, b_bl]):
                add_triangle_by_indices(t_bl, b_bl, b_tl, (-1, 0, 0))
                add_triangle_by_indices(t_bl, b_tl, t_tl, (-1, 0, 0))
        
        # 右侧面 (col=cols-1)
        for r in range(rows - 1):
            t_tr, t_br = top_indices[r, cols-1], top_indices[r+1, cols-1]
            b_tr, b_br = bottom_indices[r, cols-1], bottom_indices[r+1, cols-1]
            if all(idx >= 0 for idx in [t_tr, t_br, b_tr, b_br]):
                add_triangle_by_indices(t_tr, b_tr, b_br, (1, 0, 0))
                add_triangle_by_indices(t_tr, b_br, t_br, (1, 0, 0))
        
        print(f"    [Manifold] 处理了 {cell_count} 个网格单元, 生成 {len(triangles)} 个三角形")
        
        # 验证流形性
        manifold_check = self._check_manifold_quality(triangles)
        if manifold_check['is_manifold']:
            print(f"    [Manifold] [OK] 网格为流形 (所有边被2个三角形共享)")
        else:
            print(f"    [Manifold] [WARNING] 非流形边: {manifold_check['non_manifold_edges']}")
        
        return triangles
    
    def _check_manifold_quality(self, triangles: List[Dict]) -> Dict[str, Any]:
        """
        检查网格流形性质量
        
        Returns:
            {
                'is_manifold': bool,
                'non_manifold_edges': int,
                'edge_stats': {share_count: edge_count}
            }
        """
        from collections import defaultdict
        
        edge_count = defaultdict(int)
        
        for tri in triangles:
            vertices = tri['vertices']
            # 对每条边计数(使用顶点坐标的有序对)
            edges = [
                tuple(sorted([vertices[0], vertices[1]], key=lambda v: (v[0], v[1], v[2]))),
                tuple(sorted([vertices[1], vertices[2]], key=lambda v: (v[0], v[1], v[2]))),
                tuple(sorted([vertices[2], vertices[0]], key=lambda v: (v[0], v[1], v[2])))
            ]
            for edge in edges:
                edge_count[edge] += 1
        
        # 统计边的共享情况
        edge_stats = defaultdict(int)
        for count in edge_count.values():
            edge_stats[count] += 1
        
        non_manifold = sum(cnt for share_count, cnt in edge_stats.items() if share_count != 2)
        
        return {
            'is_manifold': non_manifold == 0,
            'non_manifold_edges': non_manifold,
            'edge_stats': dict(edge_stats),
            'total_edges': len(edge_count)
        }
    
    def _add_perimeter_walls(self, top_x, top_y, top_z, bottom_x, bottom_y, bottom_z) -> List[Dict]:
        """
        [已废弃] 为整个网格添加外围侧面
        
        此方法已被新的流形网格生成算法取代,不再需要单独添加外围墙壁。
        新算法通过顶点索引表自动处理所有面,确保流形性。
        """
        return []  # 返回空列表,不再生成重复的外围墙壁
    
    def _quad_to_triangles(self, quad: List[Tuple], expected_normal: Tuple) -> List[Dict]:
        """
        将四边形分解为两个三角形
        quad: [p1, p2, p3, p4] 按逆时针或顺时针顺序
        """
        if len(quad) != 4:
            return []
        
        p1, p2, p3, p4 = quad
        
        # 分解为两个三角形: (p1, p2, p3) 和 (p1, p3, p4)
        tri1 = self._create_triangle([p1, p2, p3], expected_normal)
        tri2 = self._create_triangle([p1, p3, p4], expected_normal)
        
        result = []
        if tri1:
            result.append(tri1)
        if tri2:
            result.append(tri2)
        
        return result
    
    def _create_triangle(self, vertices: List[Tuple], expected_normal: Tuple) -> Optional[Dict]:
        """
        创建三角形并计算法向量
        vertices: [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
        """
        if len(vertices) != 3:
            return None
        
        p1, p2, p3 = vertices
        
        # 检查是否退化
        epsilon = 1e-8
        for i in range(3):
            for j in range(i + 1, 3):
                dist = sum((vertices[i][k] - vertices[j][k])**2 for k in range(3))**0.5
                if dist < epsilon:
                    return None
        
        # 计算法向量: (p2-p1) × (p3-p1)
        v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
        
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        )
        
        # 归一化
        length = sum(n**2 for n in normal)**0.5
        if length < epsilon:
            return None
        
        normal = tuple(n / length for n in normal)
        
        # 检查法向量方向
        dot_product = sum(normal[i] * expected_normal[i] for i in range(3))
        
        # 如果法向量反向，交换顶点顺序
        if dot_product < 0:
            vertices = [p1, p3, p2]
            normal = tuple(-n for n in normal)
        
        return {
            "vertices": vertices,
            "normal": normal
        }
    
    def _validate_mesh_quality(self, triangles: List[Dict]) -> Dict[str, Any]:
        """
        验证网格质量，检测几何问题
        
        Returns:
            字典包含：is_valid, issues, statistics
        """
        issues = []
        stats = {
            "total_triangles": len(triangles),
            "min_edge_length": float('inf'),
            "max_edge_length": 0,
            "degenerate_count": 0,
            "inverted_normal_count": 0
        }
        
        for i, tri in enumerate(triangles):
            vertices = tri['vertices']
            
            # 检查边长
            for j in range(3):
                v1 = vertices[j]
                v2 = vertices[(j+1) % 3]
                edge_length = sum((v1[k] - v2[k])**2 for k in range(3))**0.5
                stats["min_edge_length"] = min(stats["min_edge_length"], edge_length)
                stats["max_edge_length"] = max(stats["max_edge_length"], edge_length)
                
                if edge_length < 1e-6:
                    stats["degenerate_count"] += 1
                    if len(issues) < 10:
                        issues.append(f"三角形 {i} 有退化边（长度<1e-6）")
            
            # 检查法向量
            normal = tri['normal']
            normal_length = sum(n**2 for n in normal)**0.5
            if abs(normal_length - 1.0) > 0.01:
                stats["inverted_normal_count"] += 1
                if len(issues) < 10:
                    issues.append(f"三角形 {i} 法向量异常（长度={normal_length:.3f}）")
        
        is_valid = len(issues) == 0
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "statistics": stats
        }
    
    def _write_binary_stl(self, filepath: str, triangles: List[Dict]):
        """写入二进制STL文件"""
        print(f"  [二进制写入] 准备写入 {len(triangles)} 个三角面片到 {os.path.basename(filepath)}")
        
        # 验证网格质量
        quality = self._validate_mesh_quality(triangles)
        if not quality["is_valid"]:
            print(f"  [警告] 网格质量问题：")
            for issue in quality["issues"][:5]:
                print(f"    - {issue}")
        
        stats = quality["statistics"]
        print(f"  [质量] 边长范围: [{stats['min_edge_length']:.3f}, {stats['max_edge_length']:.3f}]m")
        print(f"  [质量] 退化三角形: {stats['degenerate_count']}, 异常法向量: {stats['inverted_normal_count']}")
        
        with open(filepath, 'wb') as f:
            # Header (80 bytes)
            header = b'Binary STL exported from Geological Modeling System' + b' ' * 29
            header_written = f.write(header[:80])
            print(f"  [二进制写入] 头部写入: {header_written} 字节")
            print(f"  [二进制写入] 当前文件位置: {f.tell()}")
            
            # Number of triangles (4 bytes, unsigned int)
            num_triangles = len(triangles)
            num_bytes = struct.pack('<I', num_triangles)
            print(f"  [二进制写入] 三角形数量: {num_triangles}, 字节: {num_bytes.hex()}")
            num_written = f.write(num_bytes)
            print(f"  [二进制写入] 数量字节写入: {num_written} 字节")
            print(f"  [二进制写入] 当前文件位置: {f.tell()}")
            
            # Triangle data
            for tri in triangles:
                normal = tri['normal']
                vertices = tri['vertices']
                
                # Normal vector (3 floats)
                f.write(struct.pack('<fff', *normal))
                
                # 3 vertices (3 floats each)
                for vertex in vertices:
                    f.write(struct.pack('<fff', *vertex))
                
                # Attribute byte count (2 bytes, usually 0)
                f.write(struct.pack('<H', 0))
        
        # 验证写入
        with open(filepath, 'rb') as f:
            f.seek(80)
            verify_num = struct.unpack('<I', f.read(4))[0]
            print(f"  [验证] 读取到三角形数量: {verify_num}")
            if verify_num != len(triangles):
                print(f"  [警告] [WARNING] 写入验证失败!期望 {len(triangles)}, 读取 {verify_num}")
    
    def _write_ascii_stl(self, filepath: str, triangles: List[Dict]):
        """写入ASCII STL文件"""
        with open(filepath, 'w') as f:
            f.write('solid GeologicalModel\n')
            
            for tri in triangles:
                normal = tri['normal']
                vertices = tri['vertices']
                
                f.write(f'  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n')
                f.write('    outer loop\n')
                
                for vertex in vertices:
                    f.write(f'      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}\n')
                
                f.write('    endloop\n')
                f.write('  endfacet\n')
            
            f.write('endsolid GeologicalModel\n')
