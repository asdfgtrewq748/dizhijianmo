import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .base_exporter import BaseExporter

try:
    import ezdxf
    from ezdxf.math import Vec3
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    ezdxf = None
    Vec3 = None

class DXFExporter(BaseExporter):
    """
    导出地质模型为 DXF 格式 (FLAC3D/SketchUp/CAD)
    
    优化策略：
    1. 网格抽稀：可配置的降采样率，减少面片数量
    2. 体块构建：导出封闭的六面体（上表面+下表面+侧面），而非单层面
    3. 坐标归一化：将大地坐标转换为相对坐标，避免精度丢失
    """
    
    def export(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        if not EZDXF_AVAILABLE or ezdxf is None:
            raise ImportError(
                "ezdxf library is not installed. Please run: pip install ezdxf==1.3.0"
            )
        
        # 解析导出选项
        if options is None:
            options = {}
        
        downsample_factor = options.get("downsample_factor", 5)  # 默认降采样5倍
        export_as_blocks = options.get("export_as_blocks", True)  # 默认导出为体块
        normalize_coords = options.get("normalize_coords", True)  # 默认坐标归一化
        
        """
        导出 DXF 文件（优化版 - 适用于 FLAC3D）
        
        Args:
            data: 包含地层数据的字典，格式如下:
                  {
                      "layers": [
                          {
                              "name": "LayerName",
                              "grid_x": np.ndarray, # 2D array
                              "grid_y": np.ndarray, # 2D array
                              "grid_z": np.ndarray  # 2D array
                          },
                          ...
                      ]
                  }
            output_path: 输出文件路径 (.dxf)
            options: 导出选项
                - downsample_factor: 降采样倍数（默认5，即每5个点取1个）
                - export_as_blocks: 是否导出为封闭体块（默认True）
                - normalize_coords: 是否坐标归一化（默认True）
            
        Returns:
            str: 导出文件的路径
        """
        # 验证数据
        layers = data.get("layers", [])
        if not layers:
            raise ValueError("没有可导出的地层数据")
        
        print(f"[DXF Export] 开始导出 {len(layers)} 个地层 (降采样={downsample_factor}x, 体块模式={export_as_blocks})")
        
        # 坐标归一化准备
        coord_offset = self._calculate_coord_offset(layers) if normalize_coords else (0, 0, 0)
        
        doc = ezdxf.new('R2010')  # 使用兼容性更好的版本
        msp = doc.modelspace()
        
        total_faces = 0
        total_blocks = 0
        
        # 为每个地层构建独立的封闭体块（使用自身的顶面和底面）
        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get("name", f"Layer_{layer_idx}")
            safe_layer_name = self._sanitize_layer_name(layer_name)
            
            # 确保图层存在
            if safe_layer_name not in doc.layers:
                doc.layers.add(name=safe_layer_name)
            
            print(f"  [体块 {layer_idx + 1}] {safe_layer_name}")
            
            # 获取该层自己的顶面和底面数据
            top_grids = self._prepare_grid_data(layer, downsample_factor, coord_offset, use_bottom=False)
            bottom_grids = self._prepare_grid_data(layer, downsample_factor, coord_offset, use_bottom=True)
            
            if top_grids is None or bottom_grids is None:
                print(f"    ⚠️  跳过此体块（数据不完整）")
                continue
            
            top_x, top_y, top_z = top_grids
            bottom_x, bottom_y, bottom_z = bottom_grids
            
            # 验证厚度
            avg_thickness = np.nanmean(top_z - bottom_z)
            if avg_thickness < 1e-6:
                print(f"    ⚠️  跳过此体块（厚度过小: {avg_thickness:.6f}m）")
                continue
            
            print(f"    厚度范围: {np.nanmin(top_z - bottom_z):.2f}m ~ {np.nanmax(top_z - bottom_z):.2f}m (平均: {avg_thickness:.2f}m)")
            
            rows, cols = top_z.shape
            print(f"    网格尺寸: {rows}x{cols} (降采样后)")
            
            if export_as_blocks:
                # 构建封闭的六面体体块
                block_faces = self._build_closed_blocks(
                    top_x, top_y, top_z,
                    bottom_x, bottom_y, bottom_z,
                    safe_layer_name
                )
                total_blocks += 1
            else:
                # 仅导出顶面（传统模式）
                block_faces = self._build_top_surface(
                    top_x, top_y, top_z,
                    safe_layer_name
                )
            
            # 批量添加面片到DXF
            for face in block_faces:
                msp.add_3dface(face['points'], dxfattribs={'layer': face['layer']})
                total_faces += 1
            
            print(f"    ✅ 生成 {len(block_faces)} 个面片")
        
        print(f"[DXF Export] 总共生成 {total_blocks} 个体块, {total_faces} 个面片")
        print(f"[DXF Export] 导出模式: {'体块模式(6面体)' if export_as_blocks else '表面模式(单面)'}")
        print(f"[DXF Export] 采样因子: {downsample_factor}x (原始尺寸缩减 {downsample_factor**2} 倍)")
        print(f"[DXF Export] 坐标归一化: {'启用' if normalize_coords else '禁用'}")
        
        if total_faces == 0:
            raise ValueError("未能生成任何有效的3D面片，请检查数据")
        
        # 添加元数据注释
        if normalize_coords and coord_offset != (0, 0, 0):
            msp.add_text(
                f"Coordinate Offset: X={coord_offset[0]:.2f}, Y={coord_offset[1]:.2f}, Z={coord_offset[2]:.2f}",
                dxfattribs={
                    'layer': '0',
                    'insert': (0, 0, 0),
                    'height': 1.0,
                    'style': 'Standard'
                }
            )
        
        doc.saveas(output_path)
        print(f"[DXF Export] 文件已保存: {output_path}")
        return output_path
    
    def _sanitize_layer_name(self, name: str) -> str:
        """清理图层名称，移除非法字符"""
        return "".join(c for c in name if c.isalnum() or c in "_- ")
    
    def _calculate_coord_offset(self, layers: List[Dict]) -> Tuple[float, float, float]:
        """计算坐标偏移量，用于将大地坐标归一化到(0,0,0)附近"""
        all_x, all_y, all_z = [], [], []
        
        for layer in layers:
            grid_x = layer.get("grid_x")
            grid_y = layer.get("grid_y")
            grid_z = layer.get("grid_z")
            
            if grid_x is not None and grid_y is not None and grid_z is not None:
                grid_x = np.array(grid_x).flatten()
                grid_y = np.array(grid_y).flatten()
                grid_z = np.array(grid_z).flatten()
                
                valid = ~(np.isnan(grid_x) | np.isnan(grid_y) | np.isnan(grid_z))
                all_x.extend(grid_x[valid])
                all_y.extend(grid_y[valid])
                all_z.extend(grid_z[valid])
        
        if not all_x:
            return (0, 0, 0)
        
        offset_x = float(np.median(all_x))
        offset_y = float(np.median(all_y))
        offset_z = float(np.min(all_z))  # Z使用最小值，保持模型在地面以上
        
        print(f"  [坐标归一化] 偏移量: X={offset_x:.2f}, Y={offset_y:.2f}, Z={offset_z:.2f}")
        return (offset_x, offset_y, offset_z)
    
    def _prepare_grid_data(self, layer: Dict, downsample: int, offset: Tuple[float, float, float], use_bottom: bool = False) -> Optional[Tuple]:
        """准备并降采样网格数据
        
        Args:
            layer: 地层数据
            downsample: 降采样倍数
            offset: 坐标偏移量
            use_bottom: True=使用底面(bottom_surface_z), False=使用顶面(top_surface_z)
        """
        grid_x = layer.get("grid_x")
        grid_y = layer.get("grid_y")
        
        # 根据参数选择顶面或底面
        if use_bottom:
            grid_z = layer.get("bottom_surface_z") or layer.get("grid_z_bottom")
            if grid_z is None:
                # 尝试从顶面和厚度计算
                top_z = layer.get("top_surface_z") or layer.get("grid_z")
                thickness = layer.get("thickness")
                if top_z is not None and thickness is not None:
                    grid_z = np.array(top_z) - np.array(thickness)
                else:
                    return None
        else:
            grid_z = layer.get("top_surface_z") or layer.get("grid_z")
        
        if grid_x is None or grid_y is None or grid_z is None:
            return None
        
        # 转换为numpy数组
        grid_x = np.array(grid_x)
        grid_y = np.array(grid_y)
        grid_z = np.array(grid_z)
        
        # 如果是1D，扩展为2D
        if grid_x.ndim == 1 and grid_y.ndim == 1:
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        if grid_z.ndim != 2:
            return None
        
        # 降采样
        grid_x = grid_x[::downsample, ::downsample]
        grid_y = grid_y[::downsample, ::downsample]
        grid_z = grid_z[::downsample, ::downsample]
        
        # 坐标归一化
        grid_x = grid_x - offset[0]
        grid_y = grid_y - offset[1]
        grid_z = grid_z - offset[2]
        
        return (grid_x, grid_y, grid_z)
    
    def _build_closed_blocks(self, top_x, top_y, top_z, bottom_x, bottom_y, bottom_z, layer_name: str) -> List[Dict]:
        """构建封闭的六面体体块（上表面+下表面+四个侧面），避免自相交"""
        faces = []
        rows, cols = top_z.shape
        
        valid_top = ~(np.isnan(top_x) | np.isnan(top_y) | np.isnan(top_z))
        valid_bottom = ~(np.isnan(bottom_x) | np.isnan(bottom_y) | np.isnan(bottom_z))
        
        # 添加小间隙以避免完全重合
        epsilon = 1e-6
        
        for r in range(rows - 1):
            for c in range(cols - 1):
                # 检查单元的8个顶点是否都有效
                corners_valid = (
                    valid_top[r, c] and valid_top[r, c+1] and 
                    valid_top[r+1, c+1] and valid_top[r+1, c] and
                    valid_bottom[r, c] and valid_bottom[r, c+1] and 
                    valid_bottom[r+1, c+1] and valid_bottom[r+1, c]
                )
                
                if not corners_valid:
                    continue
                
                # 上表面的4个顶点（逆时针，法向朝上）
                t1 = (float(top_x[r, c]), float(top_y[r, c]), float(top_z[r, c]))
                t2 = (float(top_x[r, c+1]), float(top_y[r, c+1]), float(top_z[r, c+1]))
                t3 = (float(top_x[r+1, c+1]), float(top_y[r+1, c+1]), float(top_z[r+1, c+1]))
                t4 = (float(top_x[r+1, c]), float(top_y[r+1, c]), float(top_z[r+1, c]))
                
                # 下表面的4个顶点
                b1 = (float(bottom_x[r, c]), float(bottom_y[r, c]), float(bottom_z[r, c]))
                b2 = (float(bottom_x[r, c+1]), float(bottom_y[r, c+1]), float(bottom_z[r, c+1]))
                b3 = (float(bottom_x[r+1, c+1]), float(bottom_y[r+1, c+1]), float(bottom_z[r+1, c+1]))
                b4 = (float(bottom_x[r+1, c]), float(bottom_y[r+1, c]), float(bottom_z[r+1, c]))
                
                # 检查顶面和底面是否退化（厚度太小）
                min_thickness = min(
                    abs(t1[2] - b1[2]), abs(t2[2] - b2[2]),
                    abs(t3[2] - b3[2]), abs(t4[2] - b4[2])
                )
                
                if min_thickness < epsilon:
                    continue  # 跳过退化的体块
                
                # 确保底面在顶面下方
                if any(b[2] > t[2] for b, t in zip([b1,b2,b3,b4], [t1,t2,t3,t4])):
                    # 交换顶面和底面
                    t1, t2, t3, t4, b1, b2, b3, b4 = b1, b2, b3, b4, t1, t2, t3, t4
                
                # 构建六个面，确保法向量一致朝外
                # 顶面（法向朝上 +Z）
                top_face = self._create_face_with_normal([t1, t2, t3, t4], (0, 0, 1))
                if top_face:
                    faces.append({'points': top_face, 'layer': layer_name})
                
                # 底面（法向朝下 -Z）
                bottom_face = self._create_face_with_normal([b1, b4, b3, b2], (0, 0, -1))
                if bottom_face:
                    faces.append({'points': bottom_face, 'layer': layer_name})
                
                # 四个侧面（法向朝外）
                # 前侧面（-Y方向）
                front_face = self._create_face_with_normal([t1, b1, b2, t2], (0, -1, 0))
                if front_face:
                    faces.append({'points': front_face, 'layer': layer_name})
                
                # 右侧面（+X方向）
                right_face = self._create_face_with_normal([t2, b2, b3, t3], (1, 0, 0))
                if right_face:
                    faces.append({'points': right_face, 'layer': layer_name})
                
                # 后侧面（+Y方向）
                back_face = self._create_face_with_normal([t3, b3, b4, t4], (0, 1, 0))
                if back_face:
                    faces.append({'points': back_face, 'layer': layer_name})
                
                # 左侧面（-X方向）
                left_face = self._create_face_with_normal([t4, b4, b1, t1], (-1, 0, 0))
                if left_face:
                    faces.append({'points': left_face, 'layer': layer_name})
        
        return faces
    
    def _create_face_with_normal(self, points: List[Tuple[float, float, float]], 
                                  expected_normal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """创建面片并确保法向量方向正确"""
        if len(points) < 3:
            return None
        
        # 检查点是否退化（过于接近）
        epsilon = 1e-8
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = sum((points[i][k] - points[j][k])**2 for k in range(3))**0.5
                if dist < epsilon:
                    return None  # 退化的面
        
        # 计算面的法向量（使用前3个点）
        p1, p2, p3 = points[0], points[1], points[2]
        
        # 向量 v1 = p2 - p1
        v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        # 向量 v2 = p3 - p1
        v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
        
        # 叉积 normal = v1 × v2
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        )
        
        # 归一化
        length = sum(n**2 for n in normal)**0.5
        if length < epsilon:
            return None  # 退化的面（三点共线）
        
        normal = tuple(n / length for n in normal)
        
        # 计算与期望法向量的点积
        dot_product = sum(normal[i] * expected_normal[i] for i in range(3))
        
        # 如果法向量反向，翻转点的顺序
        if dot_product < 0:
            points = list(reversed(points))
        
        return points
    
    def _build_top_surface(self, grid_x, grid_y, grid_z, layer_name: str) -> List[Dict]:
        """仅构建顶面（传统模式），带法向量检查"""
        faces = []
        rows, cols = grid_z.shape
        valid = ~(np.isnan(grid_x) | np.isnan(grid_y) | np.isnan(grid_z))
        
        for r in range(rows - 1):
            for c in range(cols - 1):
                if not (valid[r, c] and valid[r, c+1] and valid[r+1, c+1] and valid[r+1, c]):
                    continue
                
                p1 = (float(grid_x[r, c]), float(grid_y[r, c]), float(grid_z[r, c]))
                p2 = (float(grid_x[r, c+1]), float(grid_y[r, c+1]), float(grid_z[r, c+1]))
                p3 = (float(grid_x[r+1, c+1]), float(grid_y[r+1, c+1]), float(grid_z[r+1, c+1]))
                p4 = (float(grid_x[r+1, c]), float(grid_y[r+1, c]), float(grid_z[r+1, c]))
                
                # 确保法向量朝上
                face = self._create_face_with_normal([p1, p2, p3, p4], (0, 0, 1))
                if face:
                    faces.append({'points': face, 'layer': layer_name})
        
        return faces
