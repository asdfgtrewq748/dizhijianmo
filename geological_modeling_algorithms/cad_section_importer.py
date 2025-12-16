"""
CAD剖面图导入与验证模块
用于解析DXF格式的剖面图，并与地质模型生成的剖面进行对比验证。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from dataclasses import dataclass

try:
    import ezdxf
    from ezdxf.document import Drawing
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

@dataclass
class CADLayerData:
    name: str
    points: List[Tuple[float, float]]  # (x, z) 或 (y, z) 坐标点列表
    color: str = "#000000"

@dataclass
class DeviationMetrics:
    mae: float  # 平均绝对误差
    rmse: float # 均方根误差
    max_error: float # 最大误差
    max_error_x: float # 最大误差发生的位置
    bias: float # 平均偏差 (正值表示模型偏高)
    count: int # 参与计算的点数

class CADSectionImporter:
    """CAD剖面图导入器"""
    
    def __init__(self):
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf library is not installed. Please run: pip install ezdxf")
        self.doc: Optional[Drawing] = None
        self.filename: str = ""

    def load_dxf(self, file_path: str) -> Dict[str, Any]:
        """
        加载DXF文件并返回基本信息
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            self.doc = ezdxf.readfile(file_path)
            self.filename = os.path.basename(file_path)
            
            # 获取所有图层
            layers = []
            for layer in self.doc.layers:
                layers.append({
                    "name": layer.dxf.name,
                    "color": layer.dxf.color,
                    "is_frozen": layer.is_frozen(),
                    "is_off": layer.is_off()
                })
                
            return {
                "filename": self.filename,
                "dxf_version": self.doc.dxfversion,
                "layers": layers,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def extract_layer_data(self, layer_names: List[str], axis: str = 'x') -> Dict[str, CADLayerData]:
        """
        提取指定图层的线条数据
        
        Args:
            layer_names: 需要提取的图层名称列表
            axis: 剖面方向 'x' 或 'y' (决定提取哪两个坐标作为距离和高程)
                  如果是 'x' 方向剖面，通常使用 (x, z)
                  如果是 'y' 方向剖面，通常使用 (y, z)
        
        Returns:
            Dict[layer_name, CADLayerData]
        """
        if not self.doc:
            raise ValueError("DXF document not loaded")
            
        msp = self.doc.modelspace()
        result = {}
        
        for layer_name in layer_names:
            points = []
            # 查找该图层的所有实体
            # 目前主要支持 LINE, POLYLINE, LWPOLYLINE, SPLINE
            # 简化处理：提取所有顶点，按X排序
            
            entities = msp.query(f'*[layer=="{layer_name}"]')
            
            for entity in entities:
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    points.append(self._extract_point(start, axis))
                    points.append(self._extract_point(end, axis))
                    
                elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                    # 获取多段线的所有点
                    if entity.dxftype() == 'LWPOLYLINE':
                        with entity.points() as pts:
                            for p in pts:
                                # LWPolyline points are (x, y, [start_width, [end_width, [bulge]]])
                                # Need to handle 2D points correctly
                                points.append(self._extract_point(p, axis, is_2d=True))
                    else:
                        for p in entity.points():
                            points.append(self._extract_point(p, axis))
                            
                elif entity.dxftype() == 'SPLINE':
                    # 样条曲线需要离散化
                    try:
                        # ezdxf 0.16+ 支持 flattening
                        for p in entity.flattening(distance=0.1):
                            points.append(self._extract_point(p, axis))
                    except:
                        # 降级处理：只取控制点
                        for p in entity.control_points:
                            points.append(self._extract_point(p, axis))
            
            # 排序并去重
            if points:
                # 按横坐标排序
                points.sort(key=lambda p: p[0])
                
                # 简单的去重 (如果坐标非常接近)
                unique_points = []
                if len(points) > 0:
                    unique_points.append(points[0])
                    for i in range(1, len(points)):
                        last = unique_points[-1]
                        curr = points[i]
                        # 如果距离大于阈值则添加
                        if abs(curr[0] - last[0]) > 0.001 or abs(curr[1] - last[1]) > 0.001:
                            unique_points.append(curr)
                
                result[layer_name] = CADLayerData(
                    name=layer_name,
                    points=unique_points
                )
                
        return result

    def _extract_point(self, point, axis, is_2d=False) -> Tuple[float, float]:
        """根据剖面方向提取 (distance, elevation)"""
        # point 可能是 (x, y, z) 或 (x, y)
        x = point[0]
        y = point[1]
        z = point[2] if len(point) > 2 else 0.0
        
        if axis.lower() == 'x':
            # X方向剖面：X轴为距离，Z轴为高程
            return (float(x), float(z))
        elif axis.lower() == 'y':
            # Y方向剖面：Y轴为距离，Z轴为高程
            return (float(y), float(z))
        else:
            # 默认 X
            return (float(x), float(z))


class SectionComparator:
    """剖面对比分析器"""
    
    @staticmethod
    def compare(cad_data: List[Tuple[float, float]], 
                model_x: np.ndarray, 
                model_z: np.ndarray) -> Dict[str, Any]:
        """
        对比CAD数据和模型数据
        
        Args:
            cad_data: CAD提取的点列表 [(dist, z), ...]
            model_x: 模型剖面的横坐标数组
            model_z: 模型剖面的高程数组
            
        Returns:
            包含偏差指标和对齐后的数据的字典
        """
        if not cad_data or len(model_x) == 0:
            return {"error": "Empty data"}
            
        # 1. 将CAD数据转换为numpy数组
        cad_arr = np.array(cad_data)
        cad_x_raw = cad_arr[:, 0]
        cad_z_raw = cad_arr[:, 1]
        
        # 2. 确定对比范围 (取交集)
        min_x = max(cad_x_raw.min(), model_x.min())
        max_x = min(cad_x_raw.max(), model_x.max())
        
        if min_x >= max_x:
            return {"error": "No overlap between CAD and Model data"}
            
        # 3. 在重叠范围内，将CAD数据插值到模型的X坐标上
        # 筛选出在范围内的模型点
        mask = (model_x >= min_x) & (model_x <= max_x)
        eval_x = model_x[mask]
        eval_model_z = model_z[mask]
        
        # 线性插值 CAD Z值
        eval_cad_z = np.interp(eval_x, cad_x_raw, cad_z_raw)
        
        # 4. 计算偏差 (Model - CAD)
        diff = eval_model_z - eval_cad_z
        abs_diff = np.abs(diff)
        
        # 5. 计算统计指标
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)
        
        max_error_idx = np.argmax(abs_diff)
        max_error = abs_diff[max_error_idx]
        max_error_x = eval_x[max_error_idx]
        
        metrics = DeviationMetrics(
            mae=float(mae),
            rmse=float(rmse),
            max_error=float(max_error),
            max_error_x=float(max_error_x),
            bias=float(bias),
            count=len(diff)
        )
        
        return {
            "metrics": metrics,
            "aligned_data": {
                "x": eval_x.tolist(),
                "cad_z": eval_cad_z.tolist(),
                "model_z": eval_model_z.tolist(),
                "diff": diff.tolist()
            }
        }
