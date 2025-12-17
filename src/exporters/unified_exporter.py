"""
统一导出管理器 - 集成所有导出功能

支持的格式和软件:
1. SketchUp: .dae, .obj
2. Rhino: .3dm, .step, .iges
3. MIDAS: .mct, .msh
4. CAD: .dxf, .dwg
5. FLAC3D: .f3grid, .stl
6. 通用格式: .obj, .stl, .vtk, .ply
"""

import os
from typing import Any, Dict, List, Optional
from pathlib import Path


class UnifiedExportManager:
    """
    统一导出管理器

    提供简单的接口,自动选择合适的导出器
    """

    # 格式到软件的映射
    FORMAT_SOFTWARE_MAP = {
        # SketchUp
        'dae': 'SketchUp (COLLADA)',
        'collada': 'SketchUp (COLLADA)',

        # Rhino
        '3dm': 'Rhino',
        'step': 'Rhino/CAD (STEP)',
        'stp': 'Rhino/CAD (STEP)',
        'iges': 'Rhino/CAD (IGES)',
        'igs': 'Rhino/CAD (IGES)',

        # MIDAS
        'mct': 'MIDAS (命令文本)',
        'msh': 'MIDAS (网格)',

        # CAD
        'dxf': 'AutoCAD/SketchUp',
        'dwg': 'AutoCAD',

        # FLAC3D
        'f3grid': 'FLAC3D (原生网格)',
        'dat': 'FLAC3D (命令脚本)',

        # 通用格式
        'obj': '通用3D (OBJ)',
        'stl': '3D打印/FLAC3D (STL)',
        'vtk': 'ParaView/VTK',
        'ply': 'MeshLab',
    }

    def __init__(self):
        """初始化导出管理器"""
        self._init_exporters()

    def _init_exporters(self):
        """初始化所有导出器"""
        # SketchUp导出器
        try:
            from .sketchup_exporter import SketchUpExporter
            self.sketchup_exporter = SketchUpExporter()
            print("[Export Manager] SketchUp exporter loaded")
        except Exception as e:
            print(f"[Export Manager] SketchUp exporter not available: {e}")
            self.sketchup_exporter = None

        # Rhino导出器
        try:
            from .rhino_exporter import RhinoExporter
            self.rhino_exporter = RhinoExporter()
            print("[Export Manager] Rhino exporter loaded")
        except Exception as e:
            print(f"[Export Manager] Rhino exporter not available: {e}")
            self.rhino_exporter = None

        # MIDAS导出器
        try:
            from .midas_exporter import MIDASExporter
            self.midas_exporter = MIDASExporter()
            print("[Export Manager] MIDAS exporter loaded")
        except Exception as e:
            print(f"[Export Manager] MIDAS exporter not available: {e}")
            self.midas_exporter = None

        # 使用geological_modeling_algorithms中的导出器
        try:
            import sys
            geo_path = Path(__file__).parent.parent.parent / 'geological_modeling_algorithms' / 'exporters'
            if str(geo_path) not in sys.path:
                sys.path.insert(0, str(geo_path))

            from dxf_exporter import DXFExporter
            from flac3d_exporter import FLAC3DExporter
            from f3grid_exporter import F3GridExporter
            from obj_exporter import OBJExporter
            from stl_exporter import STLExporter

            self.dxf_exporter = DXFExporter()
            self.flac3d_exporter = FLAC3DExporter()
            self.f3grid_exporter = F3GridExporter()
            self.obj_exporter = OBJExporter()
            self.stl_exporter = STLExporter()

            print("[Export Manager] Geological exporters loaded")
        except Exception as e:
            print(f"[Export Manager] Geological exporters not available: {e}")
            self.dxf_exporter = None
            self.flac3d_exporter = None
            self.f3grid_exporter = None
            self.obj_exporter = None
            self.stl_exporter = None

    def export(self, data: Dict[str, Any], output_path: str,
               format_type: Optional[str] = None,
               options: Optional[Dict[str, Any]] = None) -> str:
        """
        统一导出接口

        Args:
            data: 地层数据
            output_path: 输出路径
            format_type: 格式类型 (可选，从文件扩展名自动推断)
            options: 导出选项

        Returns:
            导出文件路径

        示例:
            # 自动推断格式
            export(data, "model.dae")

            # 指定格式
            export(data, "model.ext", format_type='dae')
        """
        # 推断格式
        if format_type is None:
            format_type = Path(output_path).suffix.lstrip('.')

        format_type = format_type.lower()

        print(f"[Export Manager] 导出格式: {format_type}")
        print(f"  目标软件: {self.FORMAT_SOFTWARE_MAP.get(format_type, '未知')}")

        # 根据格式选择导出器
        if format_type in ['dae', 'collada']:
            return self._export_sketchup_dae(data, output_path, options)

        elif format_type == '3dm':
            return self._export_rhino_3dm(data, output_path, options)

        elif format_type in ['step', 'stp']:
            return self._export_rhino_step(data, output_path, options)

        elif format_type in ['iges', 'igs']:
            return self._export_rhino_iges(data, output_path, options)

        elif format_type == 'mct':
            return self._export_midas_mct(data, output_path, options)

        elif format_type == 'msh':
            return self._export_midas_msh(data, output_path, options)

        elif format_type == 'dxf':
            return self._export_dxf(data, output_path, options)

        elif format_type == 'f3grid':
            return self._export_f3grid(data, output_path, options)

        elif format_type == 'dat':
            return self._export_flac3d_dat(data, output_path, options)

        elif format_type == 'obj':
            return self._export_obj(data, output_path, options)

        elif format_type == 'stl':
            return self._export_stl(data, output_path, options)

        else:
            raise ValueError(f"不支持的格式: {format_type}")

    def export_multiple(self, data: Dict[str, Any], output_dir: str,
                       formats: List[str], options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        批量导出多种格式

        Args:
            data: 地层数据
            output_dir: 输出目录
            formats: 格式列表 ['dae', 'dxf', 'f3grid']
            options: 导出选项

        Returns:
            格式到文件路径的映射
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        print(f"[Export Manager] 批量导出 {len(formats)} 种格式")

        for fmt in formats:
            try:
                output_path = os.path.join(output_dir, f'model.{fmt}')
                exported_path = self.export(data, output_path, fmt, options)
                results[fmt] = exported_path
                print(f"  ✓ {fmt}: {exported_path}")
            except Exception as e:
                print(f"  ✗ {fmt}: {e}")
                results[fmt] = None

        return results

    def get_supported_formats(self) -> List[str]:
        """获取所有支持的格式"""
        return sorted(self.FORMAT_SOFTWARE_MAP.keys())

    def get_format_info(self, format_type: str) -> Dict[str, Any]:
        """获取格式信息"""
        format_type = format_type.lower()

        info = {
            'format': format_type,
            'software': self.FORMAT_SOFTWARE_MAP.get(format_type, '未知'),
            'available': True,
            'description': ''
        }

        # 检查可用性
        if format_type in ['dae', 'collada']:
            info['available'] = self.sketchup_exporter is not None
            info['description'] = 'COLLADA格式，SketchUp原生支持'

        elif format_type in ['3dm', 'step', 'stp', 'iges', 'igs']:
            info['available'] = self.rhino_exporter is not None
            if format_type == '3dm':
                info['description'] = 'Rhino原生格式，支持NURBS'
            else:
                info['description'] = '工业标准CAD交换格式'

        elif format_type in ['mct', 'msh']:
            info['available'] = self.midas_exporter is not None
            info['description'] = 'MIDAS有限元分析格式'

        elif format_type == 'dxf':
            info['available'] = self.dxf_exporter is not None
            info['description'] = 'AutoCAD交换格式，广泛支持'

        elif format_type in ['f3grid', 'dat']:
            info['available'] = (self.f3grid_exporter is not None or
                               self.flac3d_exporter is not None)
            info['description'] = 'FLAC3D地质力学分析格式'

        elif format_type in ['obj', 'stl']:
            info['available'] = True
            info['description'] = '通用3D格式'

        return info

    # ======== 私有导出方法 ========

    def _export_sketchup_dae(self, data, output_path, options):
        if self.sketchup_exporter is None:
            raise ImportError("SketchUp导出器不可用")
        return self.sketchup_exporter.export_collada(data, output_path, options)

    def _export_rhino_3dm(self, data, output_path, options):
        if self.rhino_exporter is None:
            raise ImportError("Rhino导出器不可用")
        return self.rhino_exporter.export_3dm(data, output_path, options)

    def _export_rhino_step(self, data, output_path, options):
        if self.rhino_exporter is None:
            raise ImportError("Rhino导出器不可用")
        return self.rhino_exporter.export_step(data, output_path, options)

    def _export_rhino_iges(self, data, output_path, options):
        if self.rhino_exporter is None:
            raise ImportError("Rhino导出器不可用")
        return self.rhino_exporter.export_iges(data, output_path, options)

    def _export_midas_mct(self, data, output_path, options):
        if self.midas_exporter is None:
            raise ImportError("MIDAS导出器不可用")
        return self.midas_exporter.export_mct(data, output_path, options)

    def _export_midas_msh(self, data, output_path, options):
        if self.midas_exporter is None:
            raise ImportError("MIDAS导出器不可用")
        return self.midas_exporter.export_mesh(data, output_path, options)

    def _export_dxf(self, data, output_path, options):
        if self.dxf_exporter is None:
            raise ImportError("DXF导出器不可用")
        return self.dxf_exporter.export(data, output_path, options)

    def _export_f3grid(self, data, output_path, options):
        if self.f3grid_exporter is None:
            raise ImportError("F3Grid导出器不可用")
        return self.f3grid_exporter.export(data, output_path, options)

    def _export_flac3d_dat(self, data, output_path, options):
        if self.flac3d_exporter is None:
            raise ImportError("FLAC3D导出器不可用")
        return self.flac3d_exporter.export(data, output_path, options)

    def _export_obj(self, data, output_path, options):
        # 优先使用SketchUp增强OBJ
        if self.sketchup_exporter is not None:
            return self.sketchup_exporter.export_enhanced_obj(data, output_path, options)
        elif self.obj_exporter is not None:
            return self.obj_exporter.export(data, output_path, options)
        else:
            raise ImportError("OBJ导出器不可用")

    def _export_stl(self, data, output_path, options):
        if self.stl_exporter is None:
            raise ImportError("STL导出器不可用")
        return self.stl_exporter.export(data, output_path, options)


# ======== 全局便捷函数 ========

_global_manager = None


def get_export_manager() -> UnifiedExportManager:
    """获取全局导出管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedExportManager()
    return _global_manager


def export_model(data: Dict[str, Any], output_path: str,
                format_type: Optional[str] = None,
                options: Optional[Dict[str, Any]] = None) -> str:
    """
    便捷导出函数

    示例:
        export_model(data, "model.dae")
        export_model(data, "model.dxf", options={'downsample_factor': 5})
    """
    manager = get_export_manager()
    return manager.export(data, output_path, format_type, options)


def export_model_multiple(data: Dict[str, Any], output_dir: str,
                         formats: List[str], options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    批量导出

    示例:
        export_model_multiple(data, "output/", ['dae', 'dxf', 'f3grid'])
    """
    manager = get_export_manager()
    return manager.export_multiple(data, output_dir, formats, options)


def list_supported_formats() -> List[str]:
    """列出所有支持的格式"""
    manager = get_export_manager()
    return manager.get_supported_formats()


def get_format_info(format_type: str) -> Dict[str, Any]:
    """获取格式信息"""
    manager = get_export_manager()
    return manager.get_format_info(format_type)


if __name__ == '__main__':
    print("=" * 60)
    print("统一导出管理器 - Unified Export Manager")
    print("=" * 60)

    manager = UnifiedExportManager()

    print("\n支持的格式:")
    for fmt in manager.get_supported_formats():
        info = manager.get_format_info(fmt)
        status = "✓" if info['available'] else "✗"
        print(f"  {status} .{fmt:8s} - {info['software']:25s} {info['description']}")

    print("\n使用示例:")
    print("  from src.exporters.unified_exporter import export_model")
    print("  export_model(data, 'model.dae')  # SketchUp")
    print("  export_model(data, 'model.dxf')  # AutoCAD")
    print("  export_model(data, 'model.f3grid')  # FLAC3D")
