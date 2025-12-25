"""
学术图表注释工具
添加比例尺、指北针、坐标网格等科研论文必需元素
"""
import numpy as np
import pyvista as pv
from typing import Tuple, Optional


class ScientificAnnotations:
    """科研图表注释工具类"""

    @staticmethod
    def add_scale_bar(plotter, length: float = None, position: str = 'lower_right',
                      color: str = 'black', n_labels: int = 5,
                      font_size: int = 12, title: str = None) -> None:
        """
        添加比例尺

        Args:
            plotter: PyVista plotter对象
            length: 比例尺长度（米），None则自动计算
            position: 位置 ('lower_right', 'lower_left', 'upper_right', 'upper_left')
            color: 颜色
            n_labels: 标签数量
            font_size: 字体大小
            title: 标题（如"比例尺"、"Scale Bar"等）
        """
        # 获取模型边界
        bounds = plotter.bounds
        if not bounds or len(bounds) < 6:
            return

        # 计算模型尺寸
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        max_range = max(x_range, y_range)

        # 自动计算合适的比例尺长度
        if length is None:
            # 选择一个"整齐"的长度（约为模型尺寸的20-30%）
            target = max_range * 0.25
            magnitude = 10 ** np.floor(np.log10(target))
            length = magnitude * round(target / magnitude)

            # 确保是1, 2, 5, 10的倍数
            nice_values = [1, 2, 5, 10]
            factor = length / magnitude
            nearest = min(nice_values, key=lambda x: abs(x - factor))
            length = magnitude * nearest

        # PyVista内置的比例尺
        try:
            plotter.add_scalar_bar(
                title=title or f"比例尺 (m)",
                n_labels=n_labels,
                position_x=0.85 if 'right' in position else 0.05,
                position_y=0.05 if 'lower' in position else 0.85,
                width=0.1,
                height=0.25,
                title_font_size=font_size,
                label_font_size=font_size - 2,
                color=color,
                fmt='%.0f'
            )
        except:
            # 备用方案：手动绘制比例尺
            ScientificAnnotations._draw_custom_scale_bar(
                plotter, length, position, color, font_size
            )

    @staticmethod
    def _draw_custom_scale_bar(plotter, length: float, position: str,
                                color: str, font_size: int) -> None:
        """手动绘制比例尺（备用方案）"""
        bounds = plotter.bounds
        x_center = (bounds[0] + bounds[1]) / 2
        y_min = bounds[2]
        z_min = bounds[4]

        # 确定比例尺位置
        if 'right' in position:
            x_start = bounds[1] - length * 1.2
        else:
            x_start = bounds[0] + length * 0.2

        if 'upper' in position:
            y_pos = bounds[3] - (bounds[3] - bounds[2]) * 0.1
        else:
            y_pos = y_min + (bounds[3] - bounds[2]) * 0.1

        z_pos = z_min - (bounds[5] - bounds[4]) * 0.05

        # 绘制水平线
        line_points = np.array([
            [x_start, y_pos, z_pos],
            [x_start + length, y_pos, z_pos]
        ])
        line = pv.Line(line_points[0], line_points[1])
        plotter.add_mesh(line, color=color, line_width=4, name='scale_bar_line')

        # 添加刻度
        for i in range(3):  # 起点、中点、终点
            tick_x = x_start + length * i / 2
            tick_line = pv.Line(
                [tick_x, y_pos, z_pos],
                [tick_x, y_pos, z_pos + (bounds[5] - bounds[4]) * 0.02]
            )
            plotter.add_mesh(tick_line, color=color, line_width=3,
                            name=f'scale_bar_tick_{i}')

        # 添加文字标签
        label_pos = [x_start + length / 2, y_pos, z_pos - (bounds[5] - bounds[4]) * 0.03]
        plotter.add_point_labels(
            [label_pos],
            [f"{length:.0f}m"],
            font_size=font_size,
            text_color=color,
            point_size=0,
            name='scale_bar_label',
            always_visible=True,
            shape_opacity=0
        )

    @staticmethod
    def add_north_arrow(plotter, position: Tuple[float, float] = (0.9, 0.9),
                       size: float = 50, color: str = 'red',
                       show_text: bool = True, text_size: int = 16) -> None:
        """
        添加指北针

        Args:
            plotter: PyVista plotter对象
            position: 屏幕位置 (0-1范围，相对位置)
            size: 箭头大小
            color: 箭头颜色
            show_text: 是否显示"N"字
            text_size: 文字大小
        """
        bounds = plotter.bounds
        if not bounds or len(bounds) < 6:
            return

        # 计算3D空间中的位置
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_max = bounds[5]

        # 将屏幕相对位置转为3D坐标
        x_pos = bounds[0] + x_range * position[0]
        y_pos = bounds[2] + y_range * position[1]
        z_pos = z_max + (bounds[5] - bounds[4]) * 0.1

        # 创建指向北方(+Y方向)的箭头
        arrow_start = [x_pos, y_pos, z_pos]
        arrow_direction = [0, 1, 0]  # 北方向

        arrow = pv.Arrow(
            start=arrow_start,
            direction=arrow_direction,
            scale=size,
            tip_length=0.25,
            tip_radius=0.1,
            shaft_radius=0.05
        )

        plotter.add_mesh(
            arrow,
            color=color,
            lighting=False,
            name='north_arrow'
        )

        # 添加"N"字标签
        if show_text:
            text_pos = [x_pos, y_pos + size * 1.2, z_pos]
            plotter.add_point_labels(
                [text_pos],
                ['N'],
                font_size=text_size,
                text_color=color,
                point_size=0,
                bold=True,
                name='north_arrow_label',
                always_visible=True,
                shape_opacity=0,
                shadow=True
            )

    @staticmethod
    def add_coordinate_info(plotter, position: str = 'lower_left',
                           coord_system: str = "WGS84 / UTM Zone 50N",
                           z_exaggeration: float = 1.0,
                           font_size: int = 10, color: str = 'black') -> None:
        """
        添加坐标系统和垂直夸张信息

        Args:
            plotter: PyVista plotter对象
            position: 位置
            coord_system: 坐标系统名称
            z_exaggeration: 垂直夸张倍数
            font_size: 字体大小
            color: 文字颜色
        """
        text_lines = [
            f"坐标系: {coord_system}",
        ]

        if z_exaggeration != 1.0:
            text_lines.append(f"垂直夸张: {z_exaggeration:.1f}x")

        text = "\n".join(text_lines)

        # 确定位置
        pos_map = {
            'lower_left': (0.02, 0.02),
            'lower_right': (0.75, 0.02),
            'upper_left': (0.02, 0.95),
            'upper_right': (0.75, 0.95)
        }

        screen_pos = pos_map.get(position, (0.02, 0.02))

        plotter.add_text(
            text,
            position=screen_pos,
            font_size=font_size,
            color=color,
            name='coordinate_info',
            viewport=True  # 使用屏幕坐标
        )

    @staticmethod
    def add_figure_caption(plotter, caption: str, position: str = 'upper_left',
                          font_size: int = 14, color: str = 'black') -> None:
        """
        添加图题/标题

        Args:
            plotter: PyVista plotter对象
            caption: 图题文字
            position: 位置
            font_size: 字体大小
            color: 颜色
        """
        pos_map = {
            'upper_left': (0.02, 0.95),
            'upper_center': (0.35, 0.95),
            'upper_right': (0.7, 0.95)
        }

        screen_pos = pos_map.get(position, (0.02, 0.95))

        plotter.add_text(
            caption,
            position=screen_pos,
            font_size=font_size,
            color=color,
            font='arial',
            name='figure_caption',
            viewport=True
        )

    @staticmethod
    def add_grid_overlay(plotter, spacing: float = None, color: str = 'gray',
                        opacity: float = 0.3, show_xy: bool = True,
                        show_xz: bool = False, show_yz: bool = False) -> None:
        """
        添加网格覆盖层（用于显示坐标网格）

        Args:
            plotter: PyVista plotter对象
            spacing: 网格间距（米），None则自动计算
            color: 网格颜色
            opacity: 不透明度
            show_xy: 显示XY平面网格
            show_xz: 显示XZ平面网格
            show_yz: 显示YZ平面网格
        """
        bounds = plotter.bounds
        if not bounds or len(bounds) < 6:
            return

        if spacing is None:
            max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
            spacing = max_dim / 10

        # XY平面网格（底部）
        if show_xy:
            grid_xy = pv.Plane(
                center=((bounds[0] + bounds[1]) / 2,
                       (bounds[2] + bounds[3]) / 2,
                       bounds[4]),
                direction=(0, 0, 1),
                i_size=bounds[1] - bounds[0],
                j_size=bounds[3] - bounds[2],
                i_resolution=int((bounds[1] - bounds[0]) / spacing),
                j_resolution=int((bounds[3] - bounds[2]) / spacing)
            )
            plotter.add_mesh(
                grid_xy, color=color, style='wireframe',
                opacity=opacity, line_width=1, name='grid_xy'
            )

        # XZ平面网格
        if show_xz:
            grid_xz = pv.Plane(
                center=((bounds[0] + bounds[1]) / 2,
                       bounds[2],
                       (bounds[4] + bounds[5]) / 2),
                direction=(0, 1, 0),
                i_size=bounds[1] - bounds[0],
                j_size=bounds[5] - bounds[4],
                i_resolution=int((bounds[1] - bounds[0]) / spacing),
                j_resolution=int((bounds[5] - bounds[4]) / spacing)
            )
            plotter.add_mesh(
                grid_xz, color=color, style='wireframe',
                opacity=opacity, line_width=1, name='grid_xz'
            )

        # YZ平面网格
        if show_yz:
            grid_yz = pv.Plane(
                center=(bounds[0],
                       (bounds[2] + bounds[3]) / 2,
                       (bounds[4] + bounds[5]) / 2),
                direction=(1, 0, 0),
                i_size=bounds[3] - bounds[2],
                j_size=bounds[5] - bounds[4],
                i_resolution=int((bounds[3] - bounds[2]) / spacing),
                j_resolution=int((bounds[5] - bounds[4]) / spacing)
            )
            plotter.add_mesh(
                grid_yz, color=color, style='wireframe',
                opacity=opacity, line_width=1, name='grid_yz'
            )

    @staticmethod
    def remove_all_annotations(plotter) -> None:
        """移除所有注释元素"""
        annotation_names = [
            'scale_bar_line', 'scale_bar_label',
            'north_arrow', 'north_arrow_label',
            'coordinate_info', 'figure_caption',
            'grid_xy', 'grid_xz', 'grid_yz'
        ]

        # 也移除带索引的元素
        for i in range(10):
            annotation_names.append(f'scale_bar_tick_{i}')

        for name in annotation_names:
            try:
                plotter.remove_actor(name)
            except:
                pass
