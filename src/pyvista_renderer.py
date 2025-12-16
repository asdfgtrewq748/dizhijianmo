"""
PyVista 专业地质模型渲染模块

特性：
- 真实纹理贴图支持
- PBR (物理渲染) 材质
- 专业光照系统
- 3D 体块渲染（带侧面）
- 多种导出格式（HTML, PNG, OBJ, STL, VTK）
- 剖面切割
- 动画导出

依赖：
- pyvista
- numpy
- scipy (用于纹理生成)
"""

import numpy as np
import pyvista as pv
from typing import List, Dict, Optional, Tuple, Union
import os
from pathlib import Path


# =============================================================================
# 一、岩石材质配置
# =============================================================================

class RockMaterial:
    """岩石材质定义"""

    # 专业地质配色 (RGB 0-1)
    ROCK_COLORS = {
        # 煤层 - 黑色系
        '煤': (0.10, 0.10, 0.10),
        '煤层': (0.10, 0.10, 0.10),

        # 砂岩 - 黄褐色系
        '砂岩': (0.77, 0.64, 0.35),
        '细砂岩': (0.83, 0.72, 0.47),
        '中砂岩': (0.77, 0.64, 0.35),
        '粗砂岩': (0.71, 0.58, 0.28),

        # 泥岩 - 灰绿色系
        '泥岩': (0.42, 0.48, 0.42),
        '砂质泥岩': (0.48, 0.54, 0.45),

        # 页岩 - 深灰色系
        '页岩': (0.29, 0.33, 0.41),
        '炭质页岩': (0.24, 0.28, 0.32),

        # 粉砂岩 - 浅褐色系
        '粉砂岩': (0.66, 0.56, 0.47),

        # 灰岩 - 灰蓝色系
        '灰岩': (0.55, 0.62, 0.67),
        '石灰岩': (0.55, 0.62, 0.67),

        # 砾岩 - 棕红色系
        '砾岩': (0.55, 0.35, 0.24),

        # 表土/黏土 - 土黄色系
        '表土': (0.71, 0.58, 0.42),
        '黏土': (0.60, 0.50, 0.38),
        '土层': (0.71, 0.58, 0.42),
    }

    # PBR 材质参数
    PBR_PARAMS = {
        '煤': {'metallic': 0.1, 'roughness': 0.6, 'ambient': 0.3},
        '砂岩': {'metallic': 0.0, 'roughness': 0.8, 'ambient': 0.4},
        '泥岩': {'metallic': 0.0, 'roughness': 0.7, 'ambient': 0.35},
        '页岩': {'metallic': 0.05, 'roughness': 0.5, 'ambient': 0.3},
        '灰岩': {'metallic': 0.1, 'roughness': 0.4, 'ambient': 0.4},
        '砾岩': {'metallic': 0.0, 'roughness': 0.9, 'ambient': 0.35},
        '表土': {'metallic': 0.0, 'roughness': 0.95, 'ambient': 0.4},
    }

    # 默认颜色列表
    DEFAULT_COLORS = [
        (0.90, 0.29, 0.21),  # 红
        (0.30, 0.73, 0.84),  # 青
        (0.00, 0.63, 0.53),  # 绿
        (0.24, 0.33, 0.53),  # 蓝
        (0.95, 0.61, 0.50),  # 橙
        (0.52, 0.57, 0.71),  # 灰蓝
        (0.57, 0.82, 0.76),  # 浅绿
        (0.49, 0.38, 0.28),  # 棕
    ]

    @classmethod
    def get_color(cls, rock_name: str, index: int = 0) -> Tuple[float, float, float]:
        """获取岩石颜色"""
        # 精确匹配
        if rock_name in cls.ROCK_COLORS:
            return cls.ROCK_COLORS[rock_name]

        # 模糊匹配
        for key, color in cls.ROCK_COLORS.items():
            if key in rock_name or rock_name in key:
                return color

        # 使用默认颜色
        return cls.DEFAULT_COLORS[index % len(cls.DEFAULT_COLORS)]

    @classmethod
    def get_pbr_params(cls, rock_name: str) -> Dict:
        """获取 PBR 材质参数"""
        for key, params in cls.PBR_PARAMS.items():
            if key in rock_name or rock_name in key:
                return params
        return {'metallic': 0.0, 'roughness': 0.7, 'ambient': 0.35}


# =============================================================================
# 二、纹理生成器
# =============================================================================

class TextureGenerator:
    """程序化纹理生成器"""

    @staticmethod
    def generate_noise_texture(
        size: Tuple[int, int] = (512, 512),
        scale: float = 50.0,
        octaves: int = 4,
        seed: int = 42
    ) -> np.ndarray:
        """
        生成 Perlin 风格噪声纹理

        Args:
            size: 纹理尺寸 (height, width)
            scale: 噪声尺度
            octaves: 八度数（细节层级）
            seed: 随机种子

        Returns:
            纹理数组 (height, width, 3) RGB
        """
        np.random.seed(seed)
        h, w = size

        texture = np.zeros((h, w))

        for octave in range(octaves):
            freq = 2 ** octave
            amp = 0.5 ** octave

            # 生成低分辨率噪声
            low_h = max(2, h // (scale * freq))
            low_w = max(2, w // (scale * freq))
            noise = np.random.randn(int(low_h), int(low_w))

            # 上采样到目标分辨率
            from scipy.ndimage import zoom
            zoom_h = h / noise.shape[0]
            zoom_w = w / noise.shape[1]
            upsampled = zoom(noise, (zoom_h, zoom_w), order=1)

            texture += upsampled * amp

        # 归一化到 0-1
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

        # 转为 RGB
        texture_rgb = np.stack([texture] * 3, axis=-1)
        return texture_rgb

    @staticmethod
    def generate_rock_texture(
        rock_type: str,
        size: Tuple[int, int] = (512, 512),
        base_color: Tuple[float, float, float] = None,
        variation: float = 0.15
    ) -> np.ndarray:
        """
        生成特定岩石类型的纹理

        Args:
            rock_type: 岩石类型
            size: 纹理尺寸
            base_color: 基础颜色 (RGB 0-1)
            variation: 颜色变化强度

        Returns:
            纹理数组 (height, width, 3) RGB 0-255
        """
        h, w = size

        if base_color is None:
            base_color = RockMaterial.get_color(rock_type)

        # 根据岩石类型生成不同纹理
        if rock_type in ['煤', '煤层']:
            # 煤层 - 层状纹理
            texture = TextureGenerator._layered_texture(size, layers=20, orientation='horizontal')

        elif '砂' in rock_type:
            # 砂岩 - 颗粒状纹理
            texture = TextureGenerator._granular_texture(size, grain_size=3)

        elif rock_type in ['泥岩', '砂质泥岩']:
            # 泥岩 - 平滑层理
            texture = TextureGenerator._smooth_layered_texture(size)

        elif '页岩' in rock_type:
            # 页岩 - 薄片状纹理
            texture = TextureGenerator._layered_texture(size, layers=50, orientation='horizontal')

        elif '灰岩' in rock_type or '石灰岩' in rock_type:
            # 灰岩 - 块状纹理
            texture = TextureGenerator._blocky_texture(size)

        elif rock_type in ['砾岩']:
            # 砾岩 - 粗颗粒纹理
            texture = TextureGenerator._granular_texture(size, grain_size=15)

        else:
            # 默认 - 轻微噪声
            texture = TextureGenerator.generate_noise_texture(size, scale=30)[:, :, 0]

        # 应用颜色
        base = np.array(base_color)
        texture_3d = texture if texture.ndim == 3 else texture[:, :, np.newaxis]
        if texture_3d.shape[-1] == 1:
            texture_3d = np.repeat(texture_3d, 3, axis=-1)

        # 颜色调制
        colored = base * (1 - variation) + texture_3d * variation * base
        colored = np.clip(colored, 0, 1)

        # 转换为 0-255
        return (colored * 255).astype(np.uint8)

    @staticmethod
    def _layered_texture(size, layers=20, orientation='horizontal'):
        """层状纹理"""
        h, w = size
        if orientation == 'horizontal':
            base = np.linspace(0, layers * 2 * np.pi, h)
            texture = (np.sin(base) * 0.5 + 0.5)[:, np.newaxis]
            texture = np.tile(texture, (1, w))
        else:
            base = np.linspace(0, layers * 2 * np.pi, w)
            texture = (np.sin(base) * 0.5 + 0.5)[np.newaxis, :]
            texture = np.tile(texture, (h, 1))

        # 添加随机扰动
        np.random.seed(42)
        noise = np.random.randn(h, w) * 0.1
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=2)
        texture = texture + noise
        texture = np.clip(texture, 0, 1)
        return texture

    @staticmethod
    def _granular_texture(size, grain_size=3):
        """颗粒状纹理"""
        h, w = size
        np.random.seed(42)

        # 生成低分辨率噪声
        low_h = h // grain_size
        low_w = w // grain_size
        noise = np.random.rand(low_h, low_w)

        # 上采样
        from scipy.ndimage import zoom
        texture = zoom(noise, (grain_size, grain_size), order=0)[:h, :w]

        # 平滑
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(texture, sigma=1)
        return texture

    @staticmethod
    def _smooth_layered_texture(size):
        """平滑层理纹理"""
        h, w = size
        y = np.linspace(0, 4 * np.pi, h)
        x = np.linspace(0, 8 * np.pi, w)
        X, Y = np.meshgrid(x, y)
        texture = np.sin(Y) * 0.3 + np.sin(X * 0.2) * 0.2 + 0.5

        np.random.seed(42)
        noise = np.random.randn(h, w) * 0.05
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(noise, sigma=3)
        texture = texture + noise
        texture = np.clip(texture, 0, 1)
        return texture

    @staticmethod
    def _blocky_texture(size):
        """块状纹理"""
        h, w = size
        np.random.seed(42)

        # Voronoi 风格
        n_points = 50
        points_y = np.random.randint(0, h, n_points)
        points_x = np.random.randint(0, w, n_points)
        values = np.random.rand(n_points)

        Y, X = np.mgrid[0:h, 0:w]
        texture = np.zeros((h, w))

        for i in range(n_points):
            dist = np.sqrt((Y - points_y[i])**2 + (X - points_x[i])**2)
            mask = dist < (h + w) / (n_points * 0.5)
            texture[mask] = values[i]

        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(texture, sigma=5)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        return texture


# =============================================================================
# 三、PyVista 地质模型渲染器
# =============================================================================

class GeologicalModelRenderer:
    """
    PyVista 地质模型渲染器

    支持：
    - 表面渲染（顶面/底面）
    - 体块渲染（带侧面的完整3D块体）
    - 纹理贴图
    - PBR 材质
    - 多种导出格式
    """

    def __init__(
        self,
        background: str = 'white',
        window_size: Tuple[int, int] = (1920, 1080),
        use_pbr: bool = True
    ):
        """
        初始化渲染器

        Args:
            background: 背景颜色
            window_size: 窗口大小
            use_pbr: 是否使用 PBR 渲染
        """
        self.background = background
        self.window_size = window_size
        self.use_pbr = use_pbr
        self.meshes = []
        self.plotter = None

    def create_layer_mesh(
        self,
        XI: np.ndarray,
        YI: np.ndarray,
        top_surface: np.ndarray,
        bottom_surface: np.ndarray,
        name: str,
        color: Tuple[float, float, float] = None,
        texture: np.ndarray = None,
        add_sides: bool = True
    ) -> pv.PolyData:
        """
        创建单个地层的完整 3D 网格（包含顶面、底面和侧面）

        Args:
            XI, YI: 坐标网格
            top_surface: 顶面高程网格
            bottom_surface: 底面高程网格
            name: 地层名称
            color: 颜色 (RGB 0-1)
            texture: 纹理数组
            add_sides: 是否添加侧面

        Returns:
            PyVista PolyData 网格
        """
        ny, nx = XI.shape

        # 创建顶面网格
        top_grid = pv.StructuredGrid(XI, YI, top_surface)
        top_surface_mesh = top_grid.extract_surface()

        # 创建底面网格
        bottom_grid = pv.StructuredGrid(XI, YI, bottom_surface)
        bottom_surface_mesh = bottom_grid.extract_surface()

        meshes = [top_surface_mesh, bottom_surface_mesh]

        # 添加侧面
        if add_sides:
            side_meshes = self._create_side_meshes(XI, YI, top_surface, bottom_surface)
            meshes.extend(side_meshes)

        # 合并所有网格
        combined = meshes[0]
        for mesh in meshes[1:]:
            combined = combined.merge(mesh)

        combined['layer_name'] = np.array([name] * combined.n_points, dtype=object)

        return combined

    def _create_side_meshes(
        self,
        XI: np.ndarray,
        YI: np.ndarray,
        top_surface: np.ndarray,
        bottom_surface: np.ndarray
    ) -> List[pv.PolyData]:
        """创建四个侧面网格"""
        ny, nx = XI.shape
        side_meshes = []

        # 前侧面 (j=0)
        side_meshes.append(self._create_single_side(
            XI[0, :], YI[0, :], top_surface[0, :], bottom_surface[0, :]
        ))

        # 后侧面 (j=ny-1)
        side_meshes.append(self._create_single_side(
            XI[-1, :], YI[-1, :], top_surface[-1, :], bottom_surface[-1, :]
        ))

        # 左侧面 (i=0)
        side_meshes.append(self._create_single_side(
            XI[:, 0], YI[:, 0], top_surface[:, 0], bottom_surface[:, 0]
        ))

        # 右侧面 (i=nx-1)
        side_meshes.append(self._create_single_side(
            XI[:, -1], YI[:, -1], top_surface[:, -1], bottom_surface[:, -1]
        ))

        return side_meshes

    def _create_single_side(
        self,
        x_line: np.ndarray,
        y_line: np.ndarray,
        top_line: np.ndarray,
        bottom_line: np.ndarray
    ) -> pv.PolyData:
        """创建单个侧面"""
        n = len(x_line)

        # 创建顶点 (上边 + 下边)
        points = np.zeros((2 * n, 3))
        points[:n, 0] = x_line
        points[:n, 1] = y_line
        points[:n, 2] = top_line
        points[n:, 0] = x_line
        points[n:, 1] = y_line
        points[n:, 2] = bottom_line

        # 创建面 (每两个相邻顶点形成一个四边形)
        faces = []
        for i in range(n - 1):
            # 四边形: 上左, 上右, 下右, 下左
            quad = [4, i, i + 1, n + i + 1, n + i]
            faces.extend(quad)

        faces = np.array(faces)
        return pv.PolyData(points, faces)

    def render_model(
        self,
        block_models: List,
        XI: np.ndarray,
        YI: np.ndarray,
        show_layers: List[str] = None,
        add_sides: bool = True,
        use_textures: bool = True,
        opacity: float = 1.0,
        show_edges: bool = False,
        edge_color: str = 'black',
        lighting: str = 'three_lights',
        camera_position: str = 'iso'
    ) -> pv.Plotter:
        """
        渲染完整的地质模型

        Args:
            block_models: BlockModel 列表
            XI, YI: 坐标网格
            show_layers: 要显示的地层名称列表
            add_sides: 是否添加侧面
            use_textures: 是否使用纹理
            opacity: 透明度
            show_edges: 是否显示边缘
            edge_color: 边缘颜色
            lighting: 光照模式
            camera_position: 相机位置

        Returns:
            PyVista Plotter 对象
        """
        # 创建 plotter
        self.plotter = pv.Plotter(
            window_size=self.window_size,
            lighting=lighting
        )
        self.plotter.set_background(self.background)

        # 添加光源
        self._setup_lighting()

        # 渲染每个地层
        for i, bm in enumerate(block_models):
            if show_layers is not None and bm.name not in show_layers:
                continue

            # 获取颜色
            color = RockMaterial.get_color(bm.name, i)

            # 创建网格
            mesh = self.create_layer_mesh(
                XI, YI,
                bm.top_surface, bm.bottom_surface,
                bm.name,
                color=color,
                add_sides=add_sides
            )

            # 生成纹理
            texture = None
            if use_textures:
                texture_array = TextureGenerator.generate_rock_texture(
                    bm.name, size=(256, 256), base_color=color
                )
                texture = pv.numpy_to_texture(texture_array)

            # 获取 PBR 参数
            pbr_params = RockMaterial.get_pbr_params(bm.name)

            # 添加到场景
            if self.use_pbr and texture is not None:
                self.plotter.add_mesh(
                    mesh,
                    texture=texture,
                    pbr=True,
                    metallic=pbr_params['metallic'],
                    roughness=pbr_params['roughness'],
                    opacity=opacity,
                    show_edges=show_edges,
                    edge_color=edge_color,
                    name=bm.name,
                    label=bm.name
                )
            else:
                self.plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    show_edges=show_edges,
                    edge_color=edge_color,
                    ambient=pbr_params['ambient'],
                    name=bm.name,
                    label=bm.name
                )

            self.meshes.append(mesh)

        # 添加坐标轴
        self.plotter.add_axes(
            xlabel='X (m)',
            ylabel='Y (m)',
            zlabel='Z (m)',
            line_width=2
        )

        # 添加比例尺
        self.plotter.add_scalar_bar(title='地层')

        # 设置相机位置
        self.plotter.camera_position = camera_position

        return self.plotter

    def _setup_lighting(self):
        """设置专业光照"""
        if self.plotter is None:
            return

        # 移除默认光源
        self.plotter.remove_all_lights()

        # 添加主光源 (模拟太阳光)
        main_light = pv.Light(
            position=(1, 1, 2),
            focal_point=(0, 0, 0),
            color='white',
            intensity=1.0
        )
        self.plotter.add_light(main_light)

        # 添加填充光
        fill_light = pv.Light(
            position=(-1, -1, 1),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.3
        )
        self.plotter.add_light(fill_light)

        # 添加背光
        back_light = pv.Light(
            position=(0, -2, 0.5),
            focal_point=(0, 0, 0),
            color='white',
            intensity=0.2
        )
        self.plotter.add_light(back_light)

    def add_borehole_markers(
        self,
        coords: np.ndarray,
        borehole_ids: List[str],
        z_top: float = None,
        marker_size: float = 50,
        color: str = 'red'
    ):
        """
        添加钻孔位置标记

        Args:
            coords: 钻孔坐标 (N, 2)
            borehole_ids: 钻孔 ID 列表
            z_top: 标记的 Z 坐标
            marker_size: 标记大小
            color: 标记颜色
        """
        if self.plotter is None:
            return

        if z_top is None:
            # 自动计算 Z 坐标
            z_top = max(m.bounds[5] for m in self.meshes) if self.meshes else 100

        # 创建点云
        points = np.column_stack([
            coords[:, 0],
            coords[:, 1],
            np.full(len(coords), z_top + 10)
        ])

        point_cloud = pv.PolyData(points)

        self.plotter.add_mesh(
            point_cloud,
            color=color,
            point_size=marker_size,
            render_points_as_spheres=True,
            name='boreholes'
        )

        # 添加标签
        for i, (x, y) in enumerate(coords):
            self.plotter.add_point_labels(
                [[x, y, z_top + 15]],
                [borehole_ids[i]],
                font_size=12,
                text_color='black',
                shape_color='white',
                shape_opacity=0.7
            )

    def add_cross_section(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        normal: str = 'auto'
    ):
        """
        添加剖面切割

        Args:
            start_point: 剖面起点 (x, y)
            end_point: 剖面终点 (x, y)
            normal: 法向量方向
        """
        if self.plotter is None or not self.meshes:
            return

        # 计算剖面法向量
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        normal_vec = np.array([-dy, dx, 0])
        normal_vec = normal_vec / np.linalg.norm(normal_vec)

        # 剖面中心点
        center = np.array([
            (start_point[0] + end_point[0]) / 2,
            (start_point[1] + end_point[1]) / 2,
            0
        ])

        # 对每个网格应用剖面切割
        for mesh in self.meshes:
            clipped = mesh.clip(normal=normal_vec, origin=center)
            if clipped.n_points > 0:
                self.plotter.add_mesh(
                    clipped,
                    style='surface',
                    show_edges=True,
                    edge_color='black',
                    opacity=0.9
                )

    def show(self, interactive: bool = True):
        """显示渲染结果"""
        if self.plotter is not None:
            if interactive:
                self.plotter.show()
            else:
                self.plotter.show(interactive=False, auto_close=False)

    def export_screenshot(
        self,
        filename: str,
        scale: int = 2,
        transparent_background: bool = False
    ):
        """
        导出截图

        Args:
            filename: 输出文件名
            scale: 缩放倍数
            transparent_background: 是否透明背景
        """
        if self.plotter is None:
            return

        self.plotter.screenshot(
            filename,
            scale=scale,
            transparent_background=transparent_background
        )
        print(f"截图已保存: {filename}")

    def export_html(self, filename: str):
        """
        导出为交互式 HTML

        Args:
            filename: 输出文件名
        """
        if self.plotter is None:
            return

        self.plotter.export_html(filename)
        print(f"HTML 已保存: {filename}")

    def export_mesh(
        self,
        filename: str,
        file_format: str = 'obj'
    ):
        """
        导出网格文件

        Args:
            filename: 输出文件名
            file_format: 文件格式 ('obj', 'stl', 'vtk', 'ply')
        """
        if not self.meshes:
            return

        # 合并所有网格
        combined = self.meshes[0]
        for mesh in self.meshes[1:]:
            combined = combined.merge(mesh)

        # 根据格式保存
        if file_format == 'obj':
            combined.save(filename)
        elif file_format == 'stl':
            combined.save(filename)
        elif file_format == 'vtk':
            combined.save(filename)
        elif file_format == 'ply':
            combined.save(filename)
        else:
            combined.save(filename)

        print(f"网格已保存: {filename}")

    def create_animation(
        self,
        filename: str,
        n_frames: int = 120,
        viewup: Tuple[float, float, float] = (0, 0, 1)
    ):
        """
        创建 360 度旋转动画

        Args:
            filename: 输出 GIF 文件名
            n_frames: 帧数
            viewup: 上方向向量
        """
        if self.plotter is None:
            return

        # 打开 GIF 写入
        self.plotter.open_gif(filename)

        # 获取场景中心
        center = self.plotter.center

        # 旋转动画
        for i in range(n_frames):
            angle = i * 360 / n_frames
            self.plotter.camera.azimuth = angle
            self.plotter.write_frame()

        self.plotter.close()
        print(f"动画已保存: {filename}")


# =============================================================================
# 四、便捷函数
# =============================================================================

def render_geological_model(
    block_models: List,
    XI: np.ndarray,
    YI: np.ndarray,
    output_dir: str = 'output/pyvista',
    show_interactive: bool = True,
    export_formats: List[str] = None,
    **kwargs
):
    """
    一键渲染地质模型

    Args:
        block_models: BlockModel 列表
        XI, YI: 坐标网格
        output_dir: 输出目录
        show_interactive: 是否显示交互窗口
        export_formats: 导出格式列表 ['png', 'html', 'obj']
        **kwargs: 传递给 render_model 的参数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建渲染器
    renderer = GeologicalModelRenderer()

    # 渲染模型
    renderer.render_model(block_models, XI, YI, **kwargs)

    # 导出
    if export_formats:
        for fmt in export_formats:
            if fmt == 'png':
                renderer.export_screenshot(
                    os.path.join(output_dir, 'model.png'),
                    scale=2
                )
            elif fmt == 'html':
                renderer.export_html(
                    os.path.join(output_dir, 'model.html')
                )
            elif fmt in ['obj', 'stl', 'vtk', 'ply']:
                renderer.export_mesh(
                    os.path.join(output_dir, f'model.{fmt}'),
                    file_format=fmt
                )

    # 显示
    if show_interactive:
        renderer.show()

    return renderer


def create_model_comparison(
    block_models_list: List[List],
    XI: np.ndarray,
    YI: np.ndarray,
    titles: List[str],
    output_file: str = 'comparison.png'
):
    """
    创建多模型对比图

    Args:
        block_models_list: 多组 BlockModel 列表
        XI, YI: 坐标网格
        titles: 每组的标题
        output_file: 输出文件
    """
    n_models = len(block_models_list)

    # 创建子图
    plotter = pv.Plotter(
        shape=(1, n_models),
        window_size=(600 * n_models, 600)
    )

    for i, (block_models, title) in enumerate(zip(block_models_list, titles)):
        plotter.subplot(0, i)
        plotter.add_title(title, font_size=12)

        renderer = GeologicalModelRenderer()
        for j, bm in enumerate(block_models):
            color = RockMaterial.get_color(bm.name, j)
            mesh = renderer.create_layer_mesh(
                XI, YI,
                bm.top_surface, bm.bottom_surface,
                bm.name,
                add_sides=True
            )
            plotter.add_mesh(mesh, color=color, opacity=0.9)

        plotter.add_axes()

    plotter.link_views()
    plotter.screenshot(output_file, scale=2)
    print(f"对比图已保存: {output_file}")


# =============================================================================
# 五、测试代码
# =============================================================================

if __name__ == "__main__":
    print("PyVista 地质模型渲染器测试...")

    # 创建测试数据
    resolution = 30
    x = np.linspace(0, 1000, resolution)
    y = np.linspace(0, 1000, resolution)
    XI, YI = np.meshgrid(x, y)

    # 模拟地层数据
    class MockBlockModel:
        def __init__(self, name, base, thickness_func):
            self.name = name
            noise = np.random.randn(*XI.shape) * 2
            thickness = thickness_func(XI, YI) + noise
            thickness = np.clip(thickness, 1, None)
            self.bottom_surface = np.full(XI.shape, base)
            self.top_surface = self.bottom_surface + thickness
            self.thickness_grid = thickness

    np.random.seed(42)
    block_models = [
        MockBlockModel('砂岩', 0, lambda x, y: 10 + np.sin(x/200) * 3),
        MockBlockModel('泥岩', 12, lambda x, y: 8 + np.cos(y/150) * 2),
        MockBlockModel('煤层', 22, lambda x, y: 3 + np.random.rand(*x.shape)),
        MockBlockModel('砂岩', 27, lambda x, y: 15 + np.sin((x+y)/300) * 5),
        MockBlockModel('页岩', 44, lambda x, y: 6 + np.cos(x/250) * 2),
    ]

    # 测试渲染
    print("\n渲染测试模型...")
    renderer = render_geological_model(
        block_models, XI, YI,
        output_dir='output/pyvista_test',
        show_interactive=True,
        export_formats=['png', 'html'],
        use_textures=True,
        add_sides=True,
        opacity=0.95
    )
