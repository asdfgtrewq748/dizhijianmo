"""
相机视角预设管理器
用于论文出图的标准视角配置
"""
from typing import Dict, Tuple, Optional
import numpy as np


class CameraPreset:
    """相机预设配置"""

    def __init__(self, name: str, description: str,
                 position: Tuple[float, float, float] = None,
                 focal_point: Tuple[float, float, float] = None,
                 view_up: Tuple[float, float, float] = None,
                 view_angle: float = None,
                 use_method: str = None):
        """
        Args:
            name: 预设名称
            description: 描述信息
            position: 相机位置 (x, y, z)
            focal_point: 焦点位置 (x, y, z)
            view_up: 向上方向 (x, y, z)
            view_angle: 视角角度
            use_method: PyVista视角方法名称 (如 'view_isometric', 'view_xy' 等)
        """
        self.name = name
        self.description = description
        self.position = position
        self.focal_point = focal_point
        self.view_up = view_up
        self.view_angle = view_angle
        self.use_method = use_method


class CameraPresetManager:
    """相机预设管理器"""

    # 标准论文视角预设
    PRESETS = {
        "立体全景": CameraPreset(
            name="立体全景",
            description="西南向等轴测视角，适合展示整体三维结构",
            use_method="view_isometric"
        ),
        "正北剖面": CameraPreset(
            name="正北剖面",
            description="从南向北看的垂直剖面",
            view_up=(0, 0, 1),
            use_method="view_xz"
        ),
        "正东剖面": CameraPreset(
            name="正东剖面",
            description="从西向东看的垂直剖面",
            view_up=(0, 0, 1),
            use_method="view_yz"
        ),
        "纯俯视图": CameraPreset(
            name="纯俯视图",
            description="正上方俯视，适合平面图和等值线",
            use_method="view_xy"
        ),
        "正南剖面": CameraPreset(
            name="正南剖面",
            description="从北向南看的垂直剖面",
            view_up=(0, 0, 1),
            use_method="view_xz"  # 将反转方向
        ),
        "正西剖面": CameraPreset(
            name="正西剖面",
            description="从东向西看的垂直剖面",
            view_up=(0, 0, 1),
            use_method="view_yz"  # 将反转方向
        ),
        "东北视角": CameraPreset(
            name="东北视角",
            description="从西南向东北的立体视角",
            view_up=(0, 0, 1),
            use_method=None  # 自定义角度
        ),
        "西北视角": CameraPreset(
            name="西北视角",
            description="从东南向西北的立体视角",
            view_up=(0, 0, 1),
            use_method=None
        ),
    }

    @classmethod
    def apply_preset(cls, plotter, preset_name: str, bounds: Optional[Tuple] = None) -> bool:
        """
        应用预设视角到PyVista plotter

        Args:
            plotter: PyVista QtInteractor对象
            preset_name: 预设名称
            bounds: 模型边界 (xmin, xmax, ymin, ymax, zmin, zmax)，用于计算相机距离

        Returns:
            bool: 成功返回True，失败返回False
        """
        if preset_name not in cls.PRESETS:
            return False

        preset = cls.PRESETS[preset_name]

        # 如果有use_method，使用PyVista内置方法
        if preset.use_method:
            method = getattr(plotter, preset.use_method, None)
            if method:
                method()

                # 特殊处理：反转视角
                if preset_name == "正南剖面":
                    # 旋转180度
                    plotter.camera.azimuth = 180
                elif preset_name == "正西剖面":
                    plotter.camera.azimuth = 180

                return True

        # 自定义视角
        if bounds:
            center = (
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            )

            # 计算合适的相机距离
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            max_dim = max(dx, dy, dz)
            distance = max_dim * 2.5  # 距离因子

            if preset_name == "东北视角":
                # 从西南向东北看 (azimuth=45°, elevation=30°)
                plotter.camera.focal_point = center
                angle_h = np.radians(45)  # 水平角
                angle_v = np.radians(30)  # 垂直角

                pos_x = center[0] - distance * np.cos(angle_v) * np.cos(angle_h)
                pos_y = center[1] - distance * np.cos(angle_v) * np.sin(angle_h)
                pos_z = center[2] + distance * np.sin(angle_v)

                plotter.camera.position = (pos_x, pos_y, pos_z)
                plotter.camera.up = (0, 0, 1)

            elif preset_name == "西北视角":
                # 从东南向西北看 (azimuth=-45°, elevation=30°)
                plotter.camera.focal_point = center
                angle_h = np.radians(-45)
                angle_v = np.radians(30)

                pos_x = center[0] - distance * np.cos(angle_v) * np.cos(angle_h)
                pos_y = center[1] - distance * np.cos(angle_v) * np.sin(angle_h)
                pos_z = center[2] + distance * np.sin(angle_v)

                plotter.camera.position = (pos_x, pos_y, pos_z)
                plotter.camera.up = (0, 0, 1)

        # 应用自定义相机参数
        if preset.position:
            plotter.camera.position = preset.position
        if preset.focal_point:
            plotter.camera.focal_point = preset.focal_point
        if preset.view_up:
            plotter.camera.up = preset.view_up
        if preset.view_angle:
            plotter.camera.view_angle = preset.view_angle

        return True

    @classmethod
    def get_preset_names(cls) -> list:
        """获取所有预设名称列表"""
        return list(cls.PRESETS.keys())

    @classmethod
    def get_preset_description(cls, preset_name: str) -> str:
        """获取预设描述"""
        if preset_name in cls.PRESETS:
            return cls.PRESETS[preset_name].description
        return ""

    @classmethod
    def save_current_as_preset(cls, plotter, name: str, description: str = "") -> CameraPreset:
        """
        保存当前视角为新预设

        Args:
            plotter: PyVista QtInteractor对象
            name: 预设名称
            description: 描述

        Returns:
            CameraPreset: 新创建的预设对象
        """
        preset = CameraPreset(
            name=name,
            description=description,
            position=plotter.camera.position,
            focal_point=plotter.camera.focal_point,
            view_up=plotter.camera.up,
            view_angle=plotter.camera.view_angle
        )
        cls.PRESETS[name] = preset
        return preset
