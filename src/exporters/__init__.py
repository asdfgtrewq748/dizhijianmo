"""
导出器模块 - 支持多种CAD/BIM/FEA软件格式

使用方法:
    from src.exporters import export_model, export_model_multiple

    # 单一格式导出
    export_model(data, "model.dae")  # SketchUp
    export_model(data, "model.dxf")  # AutoCAD
    export_model(data, "model.f3grid")  # FLAC3D

    # 批量导出
    export_model_multiple(data, "output/", ['dae', 'dxf', 'f3grid'])
"""

from .unified_exporter import (
    UnifiedExportManager,
    get_export_manager,
    export_model,
    export_model_multiple,
    list_supported_formats,
    get_format_info
)

__all__ = [
    'UnifiedExportManager',
    'get_export_manager',
    'export_model',
    'export_model_multiple',
    'list_supported_formats',
    'get_format_info',
]

__version__ = '1.0.0'
