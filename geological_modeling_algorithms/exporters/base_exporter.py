from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseExporter(ABC):
    """
    地质模型导出基类
    """
    
    @abstractmethod
    def export(self, data: Any, output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出数据到指定格式
        
        Args:
            data: 要导出的数据 (通常是网格数据或体素数据)
            output_path: 输出文件路径
            options: 导出选项
            
        Returns:
            str: 导出文件的路径
        """
        pass
