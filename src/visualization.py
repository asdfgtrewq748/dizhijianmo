"""
SCI论文级可视化模块
提供高质量的地质和机器学习可视化功能

支持双引擎架构:
- Matplotlib: 静态高分辨率图 (300-600 DPI)，适合论文投稿
- Plotly: 交互式Web图，适合Streamlit展示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional, Union
import os
import warnings
warnings.filterwarnings('ignore')


# ==================== SCI样式配置 ====================

class SCIFigureStyle:
    """SCI论文图表样式管理类"""

    # 标准SCI配色方案 - Nature/Science风格
    GEOLOGY_COLORS = [
        '#E64B35',  # 红色 - 煤层
        '#4DBBD5',  # 青色 - 砂岩
        '#00A087',  # 绿色 - 泥岩
        '#3C5488',  # 深蓝 - 砾岩
        '#F39B7F',  # 橙色 - 粉砂岩
        '#8491B4',  # 灰蓝 - 页岩
        '#91D1C2',  # 浅绿 - 灰岩
        '#DC0000',  # 深红
        '#7E6148',  # 棕色 - 土层
        '#B09C85',  # 米色
        '#00468B',  # 海军蓝
        '#ED0000',  # 亮红
        '#42B540',  # 草绿
        '#0099B4',  # 湖蓝
        '#925E9F',  # 紫色
        '#FDAF91',  # 浅橙
    ]

    # ML图件配色 - ColorBrewer色盲友好
    ML_COLORS = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿
        '#17becf',  # 青色
    ]

    # 地质填充图案 (matplotlib hatch patterns)
    GEOLOGY_HATCHES = {
        '煤': '...',
        '砂岩': '///',
        '泥岩': '---',
        '砾岩': 'ooo',
        '粉砂岩': '\\\\\\',
        '页岩': '|||',
        '灰岩': '+++',
        '表土': '',
        '黏土': 'xxx',
    }

    # 字体配置
    FONT_CONFIG = {
        'family': 'Arial',
        'title_size': 14,
        'label_size': 12,
        'tick_size': 10,
        'legend_size': 10,
        'annotation_size': 9,
    }

    # 图像尺寸 (单位: 英寸)
    FIGURE_SIZES = {
        'single_column': (3.35, 3.35),   # 85mm
        'double_column': (7.09, 4.72),   # 180mm x 120mm
        'full_page': (7.09, 9.45),       # 180mm x 240mm
        'square': (4.72, 4.72),          # 120mm
    }

    # 导出配置
    EXPORT_CONFIG = {
        'dpi': 300,
        'formats': ['png', 'pdf', 'svg'],
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'transparent': False,
    }

    def __init__(self):
        """初始化SCI样式"""
        self._setup_matplotlib_style()

    def _setup_matplotlib_style(self):
        """设置matplotlib全局样式"""
        plt.rcParams.update({
            'font.family': self.FONT_CONFIG['family'],
            'font.size': self.FONT_CONFIG['label_size'],
            'axes.titlesize': self.FONT_CONFIG['title_size'],
            'axes.labelsize': self.FONT_CONFIG['label_size'],
            'xtick.labelsize': self.FONT_CONFIG['tick_size'],
            'ytick.labelsize': self.FONT_CONFIG['tick_size'],
            'legend.fontsize': self.FONT_CONFIG['legend_size'],
            'figure.titlesize': self.FONT_CONFIG['title_size'],
            'axes.linewidth': 1.0,
            'axes.edgecolor': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'axes.spines.top': True,
            'axes.spines.right': True,
            'legend.frameon': True,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'white',
            'savefig.dpi': self.EXPORT_CONFIG['dpi'],
        })

    @classmethod
    def get_color_palette(cls, n_colors: int, palette_type: str = 'geology') -> List[str]:
        """获取颜色调色板"""
        colors = cls.GEOLOGY_COLORS if palette_type == 'geology' else cls.ML_COLORS
        if n_colors <= len(colors):
            return colors[:n_colors]
        return (colors * (n_colors // len(colors) + 1))[:n_colors]

    @classmethod
    def get_hatch(cls, lithology: str) -> str:
        """获取岩性填充图案"""
        return cls.GEOLOGY_HATCHES.get(lithology, '')

    @classmethod
    def create_figure(cls, size_type: str = 'single_column',
                      nrows: int = 1, ncols: int = 1,
                      **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """创建标准尺寸的图形"""
        figsize = cls.FIGURE_SIZES.get(size_type, cls.FIGURE_SIZES['single_column'])
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes

    @classmethod
    def style_axis(cls, ax: plt.Axes,
                   xlabel: str = '', ylabel: str = '',
                   title: str = '', grid: bool = True):
        """美化坐标轴"""
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontweight='bold')
        if title:
            ax.set_title(title, fontweight='bold', pad=10)

        if grid:
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

        # 确保刻度在内侧
        ax.tick_params(direction='in', top=True, right=True)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')

    @classmethod
    def add_scalebar(cls, ax: plt.Axes, length: float,
                     label: str = '', location: str = 'lower right'):
        """添加比例尺"""
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm

        fontprops = fm.FontProperties(size=cls.FONT_CONFIG['annotation_size'])
        scalebar = AnchoredSizeBar(
            ax.transData,
            length, label if label else f'{length:.0f} m',
            location,
            pad=0.5,
            color='black',
            frameon=True,
            size_vertical=length * 0.02,
            fontproperties=fontprops,
            sep=5,
            fill_bar=True,
        )
        ax.add_artist(scalebar)

    @classmethod
    def add_north_arrow(cls, ax: plt.Axes, x: float = 0.95, y: float = 0.95,
                        arrow_length: float = 0.08):
        """添加指北针"""
        ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                    fontsize=cls.FONT_CONFIG['annotation_size'],
                    fontweight='bold', ha='center', va='bottom')
        ax.annotate('', xy=(x, y - 0.01), xycoords='axes fraction',
                    xytext=(x, y - arrow_length - 0.01),
                    textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


# ==================== 地质专业图件类 ====================

class GeologyPlots:
    """地质专业图件绑定类"""

    def __init__(self, style: SCIFigureStyle = None):
        self.style = style or SCIFigureStyle()

    def plot_borehole_layout(self, df: pd.DataFrame,
                             show_labels: bool = True,
                             show_convex_hull: bool = True,
                             show_scalebar: bool = True,
                             show_north_arrow: bool = True,
                             figsize: str = 'single_column',
                             return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制钻孔布置平面图

        Args:
            df: 钻孔数据DataFrame
            show_labels: 是否显示钻孔编号
            show_convex_hull: 是否显示凸包边界
            show_scalebar: 是否显示比例尺
            show_north_arrow: 是否显示指北针
            figsize: 图像尺寸类型
            return_plotly: 是否返回Plotly图形
        """
        # 获取钻孔位置
        bh_coords = df.groupby('borehole_id')[['x', 'y']].first().reset_index()

        if return_plotly:
            return self._plot_borehole_layout_plotly(bh_coords, df, show_labels, show_convex_hull)

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        # 绘制凸包边界
        if show_convex_hull and len(bh_coords) >= 3:
            points = bh_coords[['x', 'y']].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])  # 闭合
            ax.fill(hull_points[:, 0], hull_points[:, 1],
                    alpha=0.1, color='#3C5488', zorder=1)
            ax.plot(hull_points[:, 0], hull_points[:, 1],
                    '--', color='#3C5488', linewidth=1.5,
                    label='Study Area Boundary', zorder=2)

        # 绘制钻孔位置
        ax.scatter(bh_coords['x'], bh_coords['y'],
                   s=80, c='#E64B35', edgecolors='black',
                   linewidths=1.0, zorder=3, label='Borehole')

        # 添加钻孔编号
        if show_labels:
            for _, row in bh_coords.iterrows():
                ax.annotate(row['borehole_id'],
                            (row['x'], row['y']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=SCIFigureStyle.FONT_CONFIG['annotation_size'],
                            fontweight='bold')

        # 设置坐标轴
        SCIFigureStyle.style_axis(ax,
                                   xlabel='X Coordinate (m)',
                                   ylabel='Y Coordinate (m)',
                                   title='Borehole Layout Map')

        # 添加比例尺
        if show_scalebar:
            x_range = bh_coords['x'].max() - bh_coords['x'].min()
            scale_length = round(x_range / 5, -1)  # 取整
            if scale_length > 0:
                SCIFigureStyle.add_scalebar(ax, scale_length, f'{scale_length:.0f} m')

        # 添加指北针
        if show_north_arrow:
            SCIFigureStyle.add_north_arrow(ax)

        # 设置等比例
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower left', framealpha=0.9)

        plt.tight_layout()
        return fig

    def _plot_borehole_layout_plotly(self, bh_coords: pd.DataFrame,
                                      df: pd.DataFrame,
                                      show_labels: bool,
                                      show_convex_hull: bool) -> go.Figure:
        """Plotly版本的钻孔布置图"""
        fig = go.Figure()

        # 凸包边界
        if show_convex_hull and len(bh_coords) >= 3:
            points = bh_coords[['x', 'y']].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])

            fig.add_trace(go.Scatter(
                x=hull_points[:, 0], y=hull_points[:, 1],
                mode='lines',
                line=dict(color='#3C5488', width=2, dash='dash'),
                fill='toself',
                fillcolor='rgba(60, 84, 136, 0.1)',
                name='Study Area Boundary'
            ))

        # 钻孔位置
        fig.add_trace(go.Scatter(
            x=bh_coords['x'], y=bh_coords['y'],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=15, color='#E64B35',
                       line=dict(width=2, color='black')),
            text=bh_coords['borehole_id'] if show_labels else None,
            textposition='top right',
            textfont=dict(size=10, family='Arial'),
            name='Borehole'
        ))

        fig.update_layout(
            title=dict(text='<b>Borehole Layout Map</b>',
                      font=dict(size=16, family='Arial'),
                      x=0.5, xanchor='center'),
            xaxis_title='<b>X Coordinate (m)</b>',
            yaxis_title='<b>Y Coordinate (m)</b>',
            xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E5E5E5',
                      zeroline=False, showline=True, linewidth=1.5, linecolor='black',
                      mirror=True, ticks='outside', tickwidth=1.5),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E5E5E5',
                      zeroline=False, showline=True, linewidth=1.5, linecolor='black',
                      mirror=True, ticks='outside', tickwidth=1.5, scaleanchor='x'),
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
            height=600
        )

        return fig

    def plot_stratigraphic_correlation(self, df: pd.DataFrame,
                                       borehole_ids: List[str] = None,
                                       max_boreholes: int = 8,
                                       connect_layers: bool = True,
                                       figsize: str = 'double_column',
                                       return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制地层对比图（栅栏图）

        Args:
            df: 钻孔数据DataFrame
            borehole_ids: 指定钻孔ID列表
            max_boreholes: 最大显示钻孔数
            connect_layers: 是否连接同层位
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        # 选择钻孔
        if borehole_ids is None:
            borehole_ids = df['borehole_id'].unique()[:max_boreholes]

        # 获取岩性和颜色
        lithologies = sorted(df['lithology'].unique())
        colors = SCIFigureStyle.get_color_palette(len(lithologies))
        color_map = {litho: colors[i] for i, litho in enumerate(lithologies)}

        if return_plotly:
            return self._plot_correlation_plotly(df, borehole_ids, color_map, connect_layers)

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        column_width = 0.6
        spacing = 1.5

        # 存储层位信息用于连接
        layer_positions = {litho: [] for litho in lithologies}

        for i, bh_id in enumerate(borehole_ids):
            bh_data = df[df['borehole_id'] == bh_id].copy()
            if bh_data.empty:
                continue

            x_pos = i * spacing

            # 按层序排列
            if 'layer_order' in bh_data.columns:
                layers = bh_data.groupby('layer_order').agg({
                    'lithology': 'first',
                    'top_depth': 'first',
                    'bottom_depth': 'first',
                }).reset_index().sort_values('layer_order')
            else:
                continue

            # 绘制每层
            for _, layer in layers.iterrows():
                litho = layer['lithology']
                top = -layer['top_depth']
                bottom = -layer['bottom_depth']

                rect = mpatches.FancyBboxPatch(
                    (x_pos - column_width/2, bottom),
                    column_width, top - bottom,
                    boxstyle="square,pad=0",
                    facecolor=color_map[litho],
                    edgecolor='black',
                    linewidth=0.8,
                    hatch=SCIFigureStyle.get_hatch(litho)
                )
                ax.add_patch(rect)

                # 记录层位位置
                layer_positions[litho].append({
                    'x': x_pos,
                    'top': top,
                    'bottom': bottom
                })

            # 钻孔标签
            ax.text(x_pos, 5, bh_id, ha='center', va='bottom',
                   fontsize=SCIFigureStyle.FONT_CONFIG['annotation_size'],
                   fontweight='bold', rotation=45)

        # 连接同层位
        if connect_layers:
            for litho, positions in layer_positions.items():
                if len(positions) < 2:
                    continue
                positions.sort(key=lambda x: x['x'])
                for j in range(len(positions) - 1):
                    p1, p2 = positions[j], positions[j + 1]
                    # 连接顶部
                    ax.plot([p1['x'] + column_width/2, p2['x'] - column_width/2],
                           [p1['top'], p2['top']],
                           '--', color=color_map[litho], linewidth=0.8, alpha=0.6)
                    # 连接底部
                    ax.plot([p1['x'] + column_width/2, p2['x'] - column_width/2],
                           [p1['bottom'], p2['bottom']],
                           '--', color=color_map[litho], linewidth=0.8, alpha=0.6)

        # 设置坐标轴
        ax.set_xlim(-spacing, len(borehole_ids) * spacing)
        ax.autoscale(axis='y')

        SCIFigureStyle.style_axis(ax,
                                   xlabel='',
                                   ylabel='Elevation (m)',
                                   title='Stratigraphic Correlation Diagram')
        ax.set_xticks([])

        # 图例
        legend_elements = [mpatches.Patch(facecolor=color_map[litho],
                                          edgecolor='black',
                                          hatch=SCIFigureStyle.get_hatch(litho),
                                          label=litho)
                          for litho in lithologies]
        ax.legend(handles=legend_elements, loc='upper right',
                 ncol=2, framealpha=0.9)

        plt.tight_layout()
        return fig

    def _plot_correlation_plotly(self, df: pd.DataFrame,
                                  borehole_ids: List[str],
                                  color_map: Dict,
                                  connect_layers: bool) -> go.Figure:
        """Plotly版本的地层对比图"""
        fig = go.Figure()

        column_width = 0.4
        spacing = 1.0
        lithologies = list(color_map.keys())
        layer_positions = {litho: [] for litho in lithologies}

        for i, bh_id in enumerate(borehole_ids):
            bh_data = df[df['borehole_id'] == bh_id].copy()
            if bh_data.empty:
                continue

            x_pos = i * spacing

            if 'layer_order' in bh_data.columns:
                layers = bh_data.groupby('layer_order').agg({
                    'lithology': 'first',
                    'top_depth': 'first',
                    'bottom_depth': 'first',
                }).reset_index().sort_values('layer_order')
            else:
                continue

            for _, layer in layers.iterrows():
                litho = layer['lithology']
                top = -layer['top_depth']
                bottom = -layer['bottom_depth']

                # 绘制矩形
                fig.add_trace(go.Scatter(
                    x=[x_pos - column_width/2, x_pos + column_width/2,
                       x_pos + column_width/2, x_pos - column_width/2,
                       x_pos - column_width/2],
                    y=[bottom, bottom, top, top, bottom],
                    fill='toself',
                    fillcolor=color_map[litho],
                    line=dict(color='black', width=1),
                    name=litho,
                    showlegend=(i == 0),
                    legendgroup=litho,
                    hovertemplate=f'{litho}<br>Top: {-top:.1f}m<br>Bottom: {-bottom:.1f}m<extra></extra>'
                ))

                layer_positions[litho].append({
                    'x': x_pos, 'top': top, 'bottom': bottom
                })

        # 添加钻孔标签
        for i, bh_id in enumerate(borehole_ids):
            fig.add_annotation(
                x=i * spacing, y=5,
                text=f'<b>{bh_id}</b>',
                showarrow=False,
                font=dict(size=10, family='Arial'),
                textangle=-45
            )

        fig.update_layout(
            title=dict(text='<b>Stratigraphic Correlation Diagram</b>',
                      font=dict(size=16, family='Arial'),
                      x=0.5, xanchor='center'),
            xaxis_title='',
            yaxis_title='<b>Elevation (m)</b>',
            xaxis=dict(showticklabels=False, showgrid=False,
                      showline=True, linewidth=1.5, linecolor='black', mirror=True),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E5E5E5',
                      showline=True, linewidth=1.5, linecolor='black', mirror=True,
                      ticks='outside', tickwidth=1.5),
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='black',
                       borderwidth=1, orientation='h', yanchor='bottom', y=1.02),
            height=700
        )

        return fig

    def plot_fence_diagram(self, df: pd.DataFrame,
                          geo_model=None,
                          section_lines: List[Tuple] = None,
                          return_plotly: bool = True) -> go.Figure:
        """
        绘制三维栅栏剖面图 (Fence Diagram)

        Args:
            df: 钻孔数据DataFrame
            geo_model: 三维地质模型 (StratigraphicModel3D)
            section_lines: 剖面线列表 [(x1,y1,x2,y2), ...]
            return_plotly: 返回Plotly图形
        """
        fig = go.Figure()

        # 获取颜色映射
        lithologies = sorted(df['lithology'].unique())
        colors = SCIFigureStyle.get_color_palette(len(lithologies))
        color_map = {litho: colors[i] for i, litho in enumerate(lithologies)}

        # 自动生成剖面线
        if section_lines is None:
            bh_coords = df.groupby('borehole_id')[['x', 'y']].first().values
            x_min, x_max = bh_coords[:, 0].min(), bh_coords[:, 0].max()
            y_min, y_max = bh_coords[:, 1].min(), bh_coords[:, 1].max()
            y_mid = (y_min + y_max) / 2
            x_mid = (x_min + x_max) / 2
            section_lines = [
                (x_min, y_mid, x_max, y_mid),  # 东西向
                (x_mid, y_min, x_mid, y_max),  # 南北向
            ]

        # 绘制钻孔柱状体
        boreholes = df['borehole_id'].unique()
        for bh_id in boreholes:
            bh_data = df[df['borehole_id'] == bh_id].copy()
            x_center = bh_data['x'].iloc[0]
            y_center = bh_data['y'].iloc[0]

            if 'layer_order' in bh_data.columns:
                layers = bh_data.groupby('layer_order').agg({
                    'lithology': 'first',
                    'top_depth': 'first',
                    'bottom_depth': 'first',
                }).reset_index().sort_values('layer_order')
            else:
                continue

            for _, layer in layers.iterrows():
                litho = layer['lithology']
                z_top = -layer['top_depth']
                z_bottom = -layer['bottom_depth']

                # 简化为线段
                fig.add_trace(go.Scatter3d(
                    x=[x_center, x_center],
                    y=[y_center, y_center],
                    z=[z_bottom, z_top],
                    mode='lines',
                    line=dict(color=color_map[litho], width=8),
                    name=litho,
                    showlegend=False,
                    legendgroup=litho,
                    hovertemplate=f'{bh_id}<br>{litho}<br>Top: {-z_top:.1f}m<br>Bottom: {-z_bottom:.1f}m<extra></extra>'
                ))

        # 绘制剖面面片
        if geo_model is not None:
            for i, (x1, y1, x2, y2) in enumerate(section_lines):
                # 生成剖面点
                n_points = 50
                xs = np.linspace(x1, x2, n_points)
                ys = np.linspace(y1, y2, n_points)

                z_grid = geo_model.grid_info['z_grid']
                lithology_3d, _ = geo_model.get_voxel_model()

                # 提取剖面数据
                for j in range(n_points - 1):
                    # 找到最近的网格点
                    x_idx = np.argmin(np.abs(geo_model.grid_info['x_grid'] - xs[j]))
                    y_idx = np.argmin(np.abs(geo_model.grid_info['y_grid'] - ys[j]))

                    for k in range(len(z_grid) - 1):
                        litho_idx = lithology_3d[x_idx, y_idx, k]
                        if litho_idx >= 0 and litho_idx < len(lithologies):
                            litho = lithologies[litho_idx]

                            # 绘制小面片
                            fig.add_trace(go.Mesh3d(
                                x=[xs[j], xs[j+1], xs[j+1], xs[j]],
                                y=[ys[j], ys[j+1], ys[j+1], ys[j]],
                                z=[z_grid[k], z_grid[k], z_grid[k+1], z_grid[k+1]],
                                color=color_map[litho],
                                opacity=0.8,
                                showlegend=False,
                                hoverinfo='skip'
                            ))

        # 添加图例
        for litho in lithologies:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[litho]),
                name=litho,
                showlegend=True
            ))

        # 设置布局
        fig.update_layout(
            title=dict(text='<b>3D Fence Diagram</b>',
                      font=dict(size=16, family='Arial'),
                      x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(title='<b>X (m)</b>', backgroundcolor='white',
                          gridcolor='#E5E5E5', showbackground=True),
                yaxis=dict(title='<b>Y (m)</b>', backgroundcolor='white',
                          gridcolor='#E5E5E5', showbackground=True),
                zaxis=dict(title='<b>Elevation (m)</b>', backgroundcolor='white',
                          gridcolor='#E5E5E5', showbackground=True),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            paper_bgcolor='white',
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
            height=700
        )

        return fig

    def plot_thickness_contour(self, df: pd.DataFrame,
                               lithology: str = None,
                               resolution: int = 50,
                               figsize: str = 'single_column',
                               return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制地层厚度等值线图

        Args:
            df: 钻孔数据DataFrame
            lithology: 目标岩性，None则显示总厚度
            resolution: 插值分辨率
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        # 计算每个钻孔的厚度
        thickness_data = []
        for bh_id in df['borehole_id'].unique():
            bh_data = df[df['borehole_id'] == bh_id]
            x = bh_data['x'].iloc[0]
            y = bh_data['y'].iloc[0]

            if lithology:
                thickness = bh_data[bh_data['lithology'] == lithology]['layer_thickness'].sum()
            else:
                thickness = bh_data['layer_thickness'].sum()

            thickness_data.append({'x': x, 'y': y, 'thickness': thickness})

        thickness_df = pd.DataFrame(thickness_data)

        # 插值
        xi = np.linspace(thickness_df['x'].min(), thickness_df['x'].max(), resolution)
        yi = np.linspace(thickness_df['y'].min(), thickness_df['y'].max(), resolution)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata(
            (thickness_df['x'].values, thickness_df['y'].values),
            thickness_df['thickness'].values,
            (xi, yi),
            method='cubic'
        )

        title = f'{lithology} Thickness Contour Map' if lithology else 'Total Thickness Contour Map'

        if return_plotly:
            fig = go.Figure()

            # 等值线填充
            fig.add_trace(go.Contour(
                x=np.linspace(thickness_df['x'].min(), thickness_df['x'].max(), resolution),
                y=np.linspace(thickness_df['y'].min(), thickness_df['y'].max(), resolution),
                z=zi,
                colorscale='Viridis',
                contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
                colorbar=dict(title='Thickness (m)', titlefont=dict(size=12)),
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Thickness: %{z:.1f}m<extra></extra>'
            ))

            # 钻孔位置
            fig.add_trace(go.Scatter(
                x=thickness_df['x'], y=thickness_df['y'],
                mode='markers+text',
                marker=dict(size=10, color='white', line=dict(width=2, color='black')),
                text=[f'{t:.1f}' for t in thickness_df['thickness']],
                textposition='top center',
                textfont=dict(size=9),
                name='Borehole'
            ))

            fig.update_layout(
                title=dict(text=f'<b>{title}</b>', font=dict(size=16), x=0.5, xanchor='center'),
                xaxis_title='<b>X (m)</b>',
                yaxis_title='<b>Y (m)</b>',
                xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E5E5E5',
                          showline=True, linewidth=1.5, linecolor='black', mirror=True),
                yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E5E5E5',
                          showline=True, linewidth=1.5, linecolor='black', mirror=True,
                          scaleanchor='x'),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=600
            )
            return fig

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis', alpha=0.9)
        ax.contour(xi, yi, zi, levels=15, colors='black', linewidths=0.5)

        # 钻孔位置
        ax.scatter(thickness_df['x'], thickness_df['y'],
                   c='white', s=50, edgecolors='black', linewidths=1.5, zorder=5)

        # 标注厚度值
        for _, row in thickness_df.iterrows():
            ax.annotate(f'{row["thickness"]:.1f}',
                       (row['x'], row['y']),
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=8, ha='center', fontweight='bold')

        # 颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Thickness (m)', fontweight='bold')

        SCIFigureStyle.style_axis(ax, xlabel='X (m)', ylabel='Y (m)', title=title)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        return fig

    def plot_stratigraphic_column(self, df: pd.DataFrame,
                                  borehole_id: str,
                                  show_depth_scale: bool = True,
                                  show_pattern: bool = True,
                                  figsize: str = 'single_column',
                                  return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制综合地层柱状图 (带填充图案)

        Args:
            df: 钻孔数据DataFrame
            borehole_id: 钻孔ID
            show_depth_scale: 显示深度刻度
            show_pattern: 显示岩性填充图案
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        bh_data = df[df['borehole_id'] == borehole_id].copy()

        if bh_data.empty:
            raise ValueError(f"Borehole {borehole_id} not found")

        # 获取层序
        if 'layer_order' in bh_data.columns:
            layers = bh_data.groupby('layer_order').agg({
                'lithology': 'first',
                'top_depth': 'first',
                'bottom_depth': 'first',
                'layer_thickness': 'first',
            }).reset_index().sort_values('layer_order')
        else:
            layers = bh_data.sort_values('top_depth')

        lithologies = sorted(df['lithology'].unique())
        colors = SCIFigureStyle.get_color_palette(len(lithologies))
        color_map = {litho: colors[i] for i, litho in enumerate(lithologies)}

        if return_plotly:
            return self._plot_column_plotly(layers, color_map, borehole_id)

        # Matplotlib版本 - 垂直柱状图
        fig, ax = SCIFigureStyle.create_figure(figsize)

        column_width = 0.3

        for _, layer in layers.iterrows():
            litho = layer['lithology']
            top = layer['top_depth']
            bottom = layer['bottom_depth']
            thickness = layer['layer_thickness']

            rect = mpatches.FancyBboxPatch(
                (0.35, top),
                column_width, thickness,
                boxstyle="square,pad=0",
                facecolor=color_map[litho],
                edgecolor='black',
                linewidth=1.0,
                hatch=SCIFigureStyle.get_hatch(litho) if show_pattern else ''
            )
            ax.add_patch(rect)

            # 岩性标注
            ax.text(0.75, top + thickness/2, litho,
                   va='center', ha='left',
                   fontsize=SCIFigureStyle.FONT_CONFIG['annotation_size'])

            # 厚度标注
            ax.text(0.25, top + thickness/2, f'{thickness:.1f}m',
                   va='center', ha='right',
                   fontsize=SCIFigureStyle.FONT_CONFIG['annotation_size'])

        # 深度刻度
        if show_depth_scale:
            max_depth = layers['bottom_depth'].max()
            ax.set_ylim(max_depth * 1.05, -max_depth * 0.02)

            # 左侧深度刻度
            depth_ticks = np.arange(0, max_depth + 10, 10)
            ax.set_yticks(depth_ticks)
            ax.set_yticklabels([f'{d:.0f}' for d in depth_ticks])

        ax.set_xlim(0, 1.0)
        ax.set_xlabel('')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_title(f'Borehole {borehole_id}\nStratigraphic Column', fontweight='bold')
        ax.set_xticks([])

        # 只显示左边框和底边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.tight_layout()
        return fig

    def _plot_column_plotly(self, layers: pd.DataFrame,
                            color_map: Dict,
                            borehole_id: str) -> go.Figure:
        """Plotly版本的柱状图"""
        fig = go.Figure()

        for _, layer in layers.iterrows():
            litho = layer['lithology']
            top = layer['top_depth']
            bottom = layer['bottom_depth']
            thickness = layer['layer_thickness']

            fig.add_trace(go.Bar(
                x=[thickness],
                y=[f"{top:.1f}-{bottom:.1f}m"],
                orientation='h',
                marker=dict(color=color_map[litho], line=dict(color='black', width=1)),
                text=f'{litho} ({thickness:.1f}m)',
                textposition='inside',
                name=litho,
                showlegend=False,
                hovertemplate=f'{litho}<br>Depth: {top:.1f}-{bottom:.1f}m<br>Thickness: {thickness:.1f}m<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=f'<b>Borehole {borehole_id} Stratigraphic Column</b>',
                      font=dict(size=14), x=0.5, xanchor='center'),
            xaxis_title='<b>Thickness (m)</b>',
            yaxis_title='<b>Depth Range</b>',
            barmode='stack',
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=600,
            yaxis=dict(autorange='reversed')
        )

        return fig


# ==================== 机器学习图件类 ====================

class MLPlots:
    """机器学习专业图件类"""

    def __init__(self, style: SCIFigureStyle = None):
        self.style = style or SCIFigureStyle()

    def plot_model_architecture(self, model_config: Dict,
                                figsize: str = 'double_column',
                                return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制GNN模型架构图

        Args:
            model_config: 模型配置字典
                - input_dim: 输入特征维度
                - hidden_dim: 隐藏层维度
                - output_dim: 输出类别数
                - num_layers: GNN层数
                - model_type: 模型类型 (GCN/GAT/GraphSAGE等)
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        input_dim = model_config.get('input_dim', 16)
        hidden_dim = model_config.get('hidden_dim', 128)
        output_dim = model_config.get('output_dim', 5)
        num_layers = model_config.get('num_layers', 4)
        model_type = model_config.get('model_type', 'GNN')

        fig, ax = SCIFigureStyle.create_figure(figsize)

        # 层位置
        layer_positions = np.linspace(0.1, 0.9, num_layers + 2)
        node_sizes = [input_dim, *[hidden_dim]*num_layers, output_dim]
        node_sizes_scaled = [min(s/10 + 3, 8) for s in node_sizes]

        colors = ['#4DBBD5', *['#3C5488']*num_layers, '#E64B35']

        # 绘制节点层
        for i, (x_pos, n_nodes, color) in enumerate(zip(layer_positions, node_sizes_scaled, colors)):
            y_positions = np.linspace(0.2, 0.8, int(n_nodes))

            for y in y_positions:
                circle = Circle((x_pos, y), 0.02, facecolor=color,
                               edgecolor='black', linewidth=1.0, zorder=3)
                ax.add_patch(circle)

            # 层标签
            if i == 0:
                label = f'Input\n({input_dim}D)'
            elif i == len(layer_positions) - 1:
                label = f'Output\n({output_dim} classes)'
            else:
                label = f'{model_type}\nLayer {i}\n({hidden_dim}D)'

            ax.text(x_pos, 0.05, label, ha='center', va='top',
                   fontsize=SCIFigureStyle.FONT_CONFIG['annotation_size'],
                   fontweight='bold')

        # 绘制连接线
        for i in range(len(layer_positions) - 1):
            x1, x2 = layer_positions[i], layer_positions[i + 1]
            n1, n2 = node_sizes_scaled[i], node_sizes_scaled[i + 1]
            y1_positions = np.linspace(0.2, 0.8, int(n1))
            y2_positions = np.linspace(0.2, 0.8, int(n2))

            # 只绘制部分连接线避免混乱
            for y1 in y1_positions[::2]:
                for y2 in y2_positions[::2]:
                    ax.plot([x1, x2], [y1, y2], '-', color='#CCCCCC',
                           linewidth=0.3, alpha=0.5, zorder=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{model_type} Model Architecture', fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def plot_graph_structure(self, data,
                             sample_size: int = 200,
                             figsize: str = 'single_column',
                             return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制图结构可视化

        Args:
            data: PyG Data对象
            sample_size: 采样节点数
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        import torch

        # 获取坐标和边
        coords = data.coords.cpu().numpy() if hasattr(data, 'coords') else data.x[:, :3].cpu().numpy()
        edge_index = data.edge_index.cpu().numpy()
        labels = data.y.cpu().numpy()

        # 采样
        n_nodes = len(coords)
        if n_nodes > sample_size:
            sample_idx = np.random.choice(n_nodes, sample_size, replace=False)
        else:
            sample_idx = np.arange(n_nodes)

        if return_plotly:
            return self._plot_graph_plotly(coords, edge_index, labels, sample_idx)

        fig = plt.figure(figsize=SCIFigureStyle.FIGURE_SIZES['double_column'])
        ax = fig.add_subplot(111, projection='3d')

        # 绘制边
        for i, j in edge_index.T:
            if i in sample_idx and j in sample_idx:
                ax.plot([coords[i, 0], coords[j, 0]],
                       [coords[i, 1], coords[j, 1]],
                       [coords[i, 2], coords[j, 2]],
                       color='#CCCCCC', linewidth=0.3, alpha=0.3)

        # 绘制节点
        unique_labels = np.unique(labels)
        colors = SCIFigureStyle.get_color_palette(len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = (labels == label) & np.isin(np.arange(len(labels)), sample_idx)
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                      c=colors[i], s=20, alpha=0.8, label=f'Class {label}')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Graph Structure Visualization', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_graph_plotly(self, coords, edge_index, labels, sample_idx) -> go.Figure:
        """Plotly版本的图结构可视化"""
        fig = go.Figure()

        # 边
        edge_x, edge_y, edge_z = [], [], []
        for i, j in edge_index.T[:1000]:  # 限制边数
            if i in sample_idx and j in sample_idx:
                edge_x.extend([coords[i, 0], coords[j, 0], None])
                edge_y.extend([coords[i, 1], coords[j, 1], None])
                edge_z.extend([coords[i, 2], coords[j, 2], None])

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='#CCCCCC', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

        # 节点
        unique_labels = np.unique(labels)
        colors = SCIFigureStyle.get_color_palette(len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = (labels == label) & np.isin(np.arange(len(labels)), sample_idx)
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode='markers',
                marker=dict(size=4, color=colors[i]),
                name=f'Class {label}'
            ))

        fig.update_layout(
            title=dict(text='<b>Graph Structure Visualization</b>',
                      font=dict(size=16), x=0.5, xanchor='center'),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            paper_bgcolor='white',
            height=600
        )

        return fig

    def plot_feature_embedding(self, features: np.ndarray,
                               labels: np.ndarray,
                               class_names: List[str] = None,
                               method: str = 'tsne',
                               figsize: str = 'single_column',
                               return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制特征降维可视化 (t-SNE/UMAP)

        Args:
            features: 特征矩阵 [N, D]
            labels: 标签数组 [N]
            class_names: 类别名称列表
            method: 降维方法 ('tsne' or 'umap')
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        # 降维
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
            except ImportError:
                reducer = TSNE(n_components=2, random_state=42)
                method = 'tsne'

        # 采样大数据集
        max_samples = 5000
        if len(features) > max_samples:
            idx = np.random.choice(len(features), max_samples, replace=False)
            features = features[idx]
            labels = labels[idx]

        embedding = reducer.fit_transform(features)

        unique_labels = np.unique(labels)
        colors = SCIFigureStyle.get_color_palette(len(unique_labels))

        if class_names is None:
            class_names = [f'Class {i}' for i in unique_labels]

        if return_plotly:
            fig = go.Figure()

            for i, label in enumerate(unique_labels):
                mask = labels == label
                name = class_names[label] if label < len(class_names) else f'Class {label}'
                fig.add_trace(go.Scatter(
                    x=embedding[mask, 0], y=embedding[mask, 1],
                    mode='markers',
                    marker=dict(size=6, color=colors[i], opacity=0.7,
                               line=dict(width=0.5, color='white')),
                    name=name
                ))

            fig.update_layout(
                title=dict(text=f'<b>Feature Space ({method.upper()} Projection)</b>',
                          font=dict(size=16), x=0.5, xanchor='center'),
                xaxis_title=f'<b>{method.upper()} 1</b>',
                yaxis_title=f'<b>{method.upper()} 2</b>',
                paper_bgcolor='white',
                plot_bgcolor='white',
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
                height=500
            )
            return fig

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = class_names[label] if label < len(class_names) else f'Class {label}'
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=colors[i], s=20, alpha=0.7, label=name,
                      edgecolors='white', linewidths=0.3)

        SCIFigureStyle.style_axis(ax,
                                   xlabel=f'{method.upper()} 1',
                                   ylabel=f'{method.upper()} 2',
                                   title=f'Feature Space ({method.upper()} Projection)')
        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        return fig

    def plot_learning_curves(self, histories: Dict[str, Dict],
                             metrics: List[str] = ['loss', 'accuracy'],
                             figsize: str = 'double_column',
                             return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制学习曲线对比图

        Args:
            histories: 训练历史字典 {model_name: history_dict}
            metrics: 要绘制的指标
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        n_metrics = len(metrics)
        colors = SCIFigureStyle.get_color_palette(len(histories), 'ml')

        if return_plotly:
            fig = make_subplots(rows=1, cols=n_metrics,
                               subplot_titles=[m.title() for m in metrics])

            for i, (model_name, history) in enumerate(histories.items()):
                epochs = list(range(1, len(history.get('train_loss', [])) + 1))

                for j, metric in enumerate(metrics):
                    train_key = f'train_{metric}' if metric != 'loss' else 'train_loss'
                    val_key = f'val_{metric}' if metric != 'loss' else 'val_loss'

                    if train_key in history:
                        fig.add_trace(go.Scatter(
                            x=epochs, y=history[train_key],
                            mode='lines',
                            line=dict(color=colors[i], width=2),
                            name=f'{model_name} (Train)',
                            legendgroup=model_name,
                            showlegend=(j == 0)
                        ), row=1, col=j+1)

                    if val_key in history:
                        fig.add_trace(go.Scatter(
                            x=epochs, y=history[val_key],
                            mode='lines',
                            line=dict(color=colors[i], width=2, dash='dash'),
                            name=f'{model_name} (Val)',
                            legendgroup=model_name,
                            showlegend=(j == 0)
                        ), row=1, col=j+1)

            fig.update_layout(
                title=dict(text='<b>Learning Curves Comparison</b>',
                          font=dict(size=16), x=0.5, xanchor='center'),
                paper_bgcolor='white',
                height=400
            )

            for j in range(n_metrics):
                fig.update_xaxes(title_text='<b>Epoch</b>', row=1, col=j+1)
                fig.update_yaxes(title_text=f'<b>{metrics[j].title()}</b>', row=1, col=j+1)

            return fig

        # Matplotlib版本
        fig, axes = plt.subplots(1, n_metrics, figsize=SCIFigureStyle.FIGURE_SIZES[figsize])
        if n_metrics == 1:
            axes = [axes]

        for i, (model_name, history) in enumerate(histories.items()):
            epochs = list(range(1, len(history.get('train_loss', [])) + 1))

            for j, metric in enumerate(metrics):
                ax = axes[j]
                train_key = f'train_{metric}' if metric != 'loss' else 'train_loss'
                val_key = f'val_{metric}' if metric != 'loss' else 'val_loss'

                if train_key in history:
                    ax.plot(epochs, history[train_key], '-', color=colors[i],
                           linewidth=2, label=f'{model_name} (Train)')

                if val_key in history:
                    ax.plot(epochs, history[val_key], '--', color=colors[i],
                           linewidth=2, label=f'{model_name} (Val)')

                SCIFigureStyle.style_axis(ax,
                                           xlabel='Epoch',
                                           ylabel=metric.title(),
                                           title=f'{metric.title()} Curve')
                ax.legend(loc='best', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_roc_curves(self, y_true: np.ndarray,
                        y_proba: np.ndarray,
                        class_names: List[str] = None,
                        figsize: str = 'single_column',
                        return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制多类别ROC曲线

        Args:
            y_true: 真实标签 [N]
            y_proba: 预测概率 [N, C]
            class_names: 类别名称
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        n_classes = y_proba.shape[1]

        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]

        # 二值化标签
        y_bin = label_binarize(y_true, classes=range(n_classes))

        colors = SCIFigureStyle.get_color_palette(n_classes)

        if return_plotly:
            fig = go.Figure()

            # 每个类别的ROC
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)

                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    line=dict(color=colors[i], width=2),
                    name=f'{class_names[i]} (AUC={roc_auc:.3f})'
                ))

            # 对角线
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Random',
                showlegend=False
            ))

            fig.update_layout(
                title=dict(text='<b>ROC Curves (Multi-class)</b>',
                          font=dict(size=16), x=0.5, xanchor='center'),
                xaxis_title='<b>False Positive Rate</b>',
                yaxis_title='<b>True Positive Rate</b>',
                paper_bgcolor='white',
                plot_bgcolor='white',
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
                height=500
            )
            return fig

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                   label=f'{class_names[i]} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)

        SCIFigureStyle.style_axis(ax,
                                   xlabel='False Positive Rate',
                                   ylabel='True Positive Rate',
                                   title='ROC Curves (Multi-class)')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        return fig

    def plot_classification_heatmap(self, report_dict: Dict,
                                    class_names: List[str] = None,
                                    figsize: str = 'single_column',
                                    return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制分类报告热力图

        Args:
            report_dict: sklearn classification_report(output_dict=True) 的输出
            class_names: 类别名称
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        # 提取数据
        metrics = ['precision', 'recall', 'f1-score']

        if class_names is None:
            class_names = [k for k in report_dict.keys()
                          if k not in ['accuracy', 'macro avg', 'weighted avg']]

        data = []
        for cls in class_names:
            if cls in report_dict:
                row = [report_dict[cls][m] for m in metrics]
                data.append(row)

        data = np.array(data)

        if return_plotly:
            fig = go.Figure(data=go.Heatmap(
                z=data,
                x=metrics,
                y=class_names,
                colorscale='RdYlGn',
                zmin=0, zmax=1,
                text=np.round(data, 3),
                texttemplate='%{text:.3f}',
                textfont=dict(size=10),
                hoverongaps=False,
                colorbar=dict(title='Score')
            ))

            fig.update_layout(
                title=dict(text='<b>Classification Performance Heatmap</b>',
                          font=dict(size=16), x=0.5, xanchor='center'),
                xaxis_title='<b>Metric</b>',
                yaxis_title='<b>Class</b>',
                paper_bgcolor='white',
                height=400
            )
            return fig

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.title() for m in metrics])
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)

        # 添加数值
        for i in range(len(class_names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                              ha='center', va='center',
                              color='black' if data[i, j] > 0.5 else 'white',
                              fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8, label='Score')

        ax.set_title('Classification Performance Heatmap', fontweight='bold')

        plt.tight_layout()
        return fig


# ==================== 结果分析图件类 ====================

class ResultPlots:
    """结果分析图件类"""

    def __init__(self, style: SCIFigureStyle = None):
        self.style = style or SCIFigureStyle()

    def plot_error_distribution_3d(self, coords: np.ndarray,
                                    predictions: np.ndarray,
                                    true_labels: np.ndarray,
                                    class_names: List[str] = None,
                                    return_plotly: bool = True) -> go.Figure:
        """
        绘制预测误差空间分布图

        Args:
            coords: 坐标数组 [N, 3]
            predictions: 预测标签 [N]
            true_labels: 真实标签 [N]
            class_names: 类别名称
            return_plotly: 是否返回Plotly图形
        """
        # 识别错误预测
        errors = predictions != true_labels
        correct = ~errors

        colors = SCIFigureStyle.get_color_palette(len(np.unique(true_labels)))

        fig = go.Figure()

        # 正确预测点
        fig.add_trace(go.Scatter3d(
            x=coords[correct, 0],
            y=coords[correct, 1],
            z=coords[correct, 2],
            mode='markers',
            marker=dict(size=3, color='#00A087', opacity=0.3),
            name='Correct Predictions',
            hovertemplate='Correct<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))

        # 错误预测点 - 按错误类型分组
        unique_errors = np.unique(np.column_stack([true_labels[errors], predictions[errors]]), axis=0)

        for true_label, pred_label in unique_errors:
            mask = errors & (true_labels == true_label) & (predictions == pred_label)

            true_name = class_names[true_label] if class_names else f'Class {true_label}'
            pred_name = class_names[pred_label] if class_names else f'Class {pred_label}'

            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode='markers',
                marker=dict(size=6, color='#E64B35', symbol='x',
                           line=dict(width=2, color='black')),
                name=f'{true_name} → {pred_name}',
                hovertemplate=f'Error: {true_name} → {pred_name}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Z: %{{z:.1f}}<extra></extra>'
            ))

        # 统计信息
        accuracy = correct.sum() / len(correct) * 100

        fig.update_layout(
            title=dict(text=f'<b>Prediction Error Distribution (Accuracy: {accuracy:.1f}%)</b>',
                      font=dict(size=16), x=0.5, xanchor='center'),
            scene=dict(
                xaxis_title='<b>X (m)</b>',
                yaxis_title='<b>Y (m)</b>',
                zaxis_title='<b>Elevation (m)</b>',
                aspectmode='data'
            ),
            paper_bgcolor='white',
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
            height=700
        )

        return fig

    def plot_uncertainty_map(self, coords: np.ndarray,
                             confidence: np.ndarray,
                             threshold: float = 0.8,
                             return_plotly: bool = True) -> go.Figure:
        """
        绘制预测不确定性分布图

        Args:
            coords: 坐标数组 [N, 3]
            confidence: 置信度数组 [N]
            threshold: 高置信度阈值
            return_plotly: 是否返回Plotly图形
        """
        fig = go.Figure()

        # 颜色映射 - 低置信度为红色，高置信度为绿色
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=confidence,
                colorscale='RdYlGn',
                cmin=0, cmax=1,
                colorbar=dict(title='Confidence', thickness=15),
                opacity=0.7
            ),
            hovertemplate='Confidence: %{marker.color:.3f}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))

        # 统计
        low_conf = (confidence < threshold).sum()
        high_conf = (confidence >= threshold).sum()

        fig.update_layout(
            title=dict(text=f'<b>Prediction Uncertainty Distribution</b><br>'
                           f'<sub>High confidence (≥{threshold}): {high_conf} | '
                           f'Low confidence (<{threshold}): {low_conf}</sub>',
                      font=dict(size=14), x=0.5, xanchor='center'),
            scene=dict(
                xaxis_title='<b>X (m)</b>',
                yaxis_title='<b>Y (m)</b>',
                zaxis_title='<b>Elevation (m)</b>',
                aspectmode='data'
            ),
            paper_bgcolor='white',
            height=700
        )

        return fig

    def plot_prediction_comparison(self, df: pd.DataFrame,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   class_names: List[str],
                                   figsize: str = 'double_column',
                                   return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制预测vs真实对比柱状图 (按钻孔分组)

        Args:
            df: 原始数据DataFrame
            predictions: 预测标签
            true_labels: 真实标签
            class_names: 类别名称
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        # 按钻孔计算准确率
        borehole_ids = df['borehole_id'].values
        unique_boreholes = np.unique(borehole_ids)

        borehole_stats = []
        for bh_id in unique_boreholes:
            mask = borehole_ids == bh_id
            acc = (predictions[mask] == true_labels[mask]).mean()
            borehole_stats.append({'borehole': bh_id, 'accuracy': acc})

        stats_df = pd.DataFrame(borehole_stats).sort_values('accuracy', ascending=False)

        if return_plotly:
            colors = ['#00A087' if acc >= 0.8 else '#F39B7F' if acc >= 0.6 else '#E64B35'
                     for acc in stats_df['accuracy']]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=stats_df['borehole'],
                y=stats_df['accuracy'],
                marker=dict(color=colors, line=dict(color='black', width=1)),
                text=[f'{acc:.1%}' for acc in stats_df['accuracy']],
                textposition='outside',
                hovertemplate='%{x}<br>Accuracy: %{y:.1%}<extra></extra>'
            ))

            # 参考线
            fig.add_hline(y=0.8, line_dash='dash', line_color='green',
                         annotation_text='80% threshold')

            fig.update_layout(
                title=dict(text='<b>Prediction Accuracy by Borehole</b>',
                          font=dict(size=16), x=0.5, xanchor='center'),
                xaxis_title='<b>Borehole ID</b>',
                yaxis_title='<b>Accuracy</b>',
                yaxis=dict(range=[0, 1.1]),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=500
            )
            return fig

        # Matplotlib版本
        fig, ax = SCIFigureStyle.create_figure(figsize)

        colors = ['#00A087' if acc >= 0.8 else '#F39B7F' if acc >= 0.6 else '#E64B35'
                 for acc in stats_df['accuracy']]

        bars = ax.bar(range(len(stats_df)), stats_df['accuracy'], color=colors,
                     edgecolor='black', linewidth=1)

        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1, label='80% threshold')

        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels(stats_df['borehole'], rotation=45, ha='right')

        SCIFigureStyle.style_axis(ax,
                                   xlabel='Borehole ID',
                                   ylabel='Accuracy',
                                   title='Prediction Accuracy by Borehole')
        ax.set_ylim([0, 1.1])
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig

    def plot_volume_statistics(self, stats_df: pd.DataFrame,
                               figsize: str = 'double_column',
                               return_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制岩性体积统计图 (饼图+柱状图组合)

        Args:
            stats_df: 统计DataFrame，包含'岩性', '体积 (m³)', '占比 (%)'列
            figsize: 图像尺寸
            return_plotly: 是否返回Plotly图形
        """
        colors = SCIFigureStyle.get_color_palette(len(stats_df))

        if return_plotly:
            fig = make_subplots(rows=1, cols=2,
                               specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                               subplot_titles=['Volume Distribution', 'Volume by Lithology'])

            # 饼图
            fig.add_trace(go.Pie(
                labels=stats_df['岩性'],
                values=stats_df['体积 (m³)'],
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textinfo='label+percent',
                textfont=dict(size=10),
                hole=0.3
            ), row=1, col=1)

            # 柱状图
            fig.add_trace(go.Bar(
                x=stats_df['岩性'],
                y=stats_df['体积 (m³)'],
                marker=dict(color=colors, line=dict(color='black', width=1)),
                text=[f'{v:.0f}' for v in stats_df['体积 (m³)']],
                textposition='outside'
            ), row=1, col=2)

            fig.update_layout(
                title=dict(text='<b>Lithology Volume Statistics</b>',
                          font=dict(size=16), x=0.5, xanchor='center'),
                paper_bgcolor='white',
                height=450,
                showlegend=False
            )

            fig.update_xaxes(title_text='<b>Lithology</b>', row=1, col=2)
            fig.update_yaxes(title_text='<b>Volume (m³)</b>', row=1, col=2)

            return fig

        # Matplotlib版本
        fig, axes = plt.subplots(1, 2, figsize=SCIFigureStyle.FIGURE_SIZES[figsize])

        # 饼图
        axes[0].pie(stats_df['体积 (m³)'], labels=stats_df['岩性'],
                   colors=colors, autopct='%1.1f%%',
                   wedgeprops=dict(linewidth=1, edgecolor='white'))
        axes[0].set_title('Volume Distribution', fontweight='bold')

        # 柱状图
        bars = axes[1].bar(stats_df['岩性'], stats_df['体积 (m³)'],
                          color=colors, edgecolor='black', linewidth=1)

        SCIFigureStyle.style_axis(axes[1],
                                   xlabel='Lithology',
                                   ylabel='Volume (m³)',
                                   title='Volume by Lithology')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig


# ==================== 图像导出类 ====================

class FigureExporter:
    """高分辨率图像导出器"""

    def __init__(self, output_dir: str = 'output/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_figure(self, fig: Union[plt.Figure, go.Figure],
                      filename: str,
                      formats: List[str] = None,
                      dpi: int = 300,
                      **kwargs) -> List[str]:
        """
        导出图形到多种格式

        Args:
            fig: Matplotlib或Plotly图形
            filename: 文件名(不含扩展名)
            formats: 导出格式列表 ['png', 'pdf', 'svg']
            dpi: 分辨率
            **kwargs: 额外参数

        Returns:
            导出的文件路径列表
        """
        if formats is None:
            formats = ['png', 'pdf']

        exported_files = []

        if isinstance(fig, plt.Figure):
            for fmt in formats:
                filepath = os.path.join(self.output_dir, f'{filename}.{fmt}')
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                           pad_inches=0.1, facecolor='white', **kwargs)
                exported_files.append(filepath)
                print(f"Exported: {filepath}")

        elif isinstance(fig, go.Figure):
            for fmt in formats:
                filepath = os.path.join(self.output_dir, f'{filename}.{fmt}')
                if fmt == 'html':
                    fig.write_html(filepath)
                else:
                    fig.write_image(filepath, scale=dpi/72)
                exported_files.append(filepath)
                print(f"Exported: {filepath}")

        return exported_files

    def export_batch(self, figures: Dict[str, Union[plt.Figure, go.Figure]],
                     formats: List[str] = None,
                     dpi: int = 300) -> Dict[str, List[str]]:
        """
        批量导出图形

        Args:
            figures: {filename: figure} 字典
            formats: 导出格式列表
            dpi: 分辨率

        Returns:
            {filename: [exported_paths]} 字典
        """
        results = {}
        for name, fig in figures.items():
            results[name] = self.export_figure(fig, name, formats, dpi)
        return results

    def create_figure_panel(self, figures: List[Union[plt.Figure, go.Figure]],
                            layout: Tuple[int, int],
                            titles: List[str] = None,
                            figsize: Tuple[float, float] = (12, 10)) -> plt.Figure:
        """
        创建组合图板

        Args:
            figures: 图形列表
            layout: 布局 (rows, cols)
            titles: 子图标题
            figsize: 图板尺寸

        Returns:
            组合图形
        """
        rows, cols = layout
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for i, (subfig, ax) in enumerate(zip(figures, axes)):
            if isinstance(subfig, plt.Figure):
                # 复制matplotlib图形内容
                for orig_ax in subfig.axes:
                    for line in orig_ax.lines:
                        ax.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(), linewidth=line.get_linewidth())

            if titles and i < len(titles):
                ax.set_title(f'({chr(97+i)}) {titles[i]}', fontweight='bold')

        # 隐藏多余的子图
        for i in range(len(figures), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig


# ==================== 便捷接口 ====================

def create_all_figures(df: pd.DataFrame,
                       result: Dict,
                       trainer=None,
                       geo_model=None,
                       output_dir: str = 'output/figures') -> Dict[str, go.Figure]:
    """
    一键生成所有SCI论文图件

    Args:
        df: 钻孔数据DataFrame
        result: 数据处理结果字典
        trainer: 训练好的模型
        geo_model: 地质模型
        output_dir: 输出目录

    Returns:
        生成的图件字典
    """
    figures = {}

    style = SCIFigureStyle()
    geo_plots = GeologyPlots(style)
    ml_plots = MLPlots(style)
    result_plots = ResultPlots(style)
    exporter = FigureExporter(output_dir)

    # 地质图件
    print("Generating geology figures...")
    figures['borehole_layout'] = geo_plots.plot_borehole_layout(df, return_plotly=True)
    figures['stratigraphic_correlation'] = geo_plots.plot_stratigraphic_correlation(df, return_plotly=True)

    lithologies = sorted(df['lithology'].unique())
    for litho in lithologies[:3]:  # 主要岩性
        figures[f'thickness_contour_{litho}'] = geo_plots.plot_thickness_contour(
            df, lithology=litho, return_plotly=True)

    # 如果有模型数据
    if trainer is not None and hasattr(trainer, 'history') and trainer.history:
        print("Generating ML figures...")
        figures['learning_curves'] = ml_plots.plot_learning_curves(
            {'Model': trainer.history}, return_plotly=True)

    # 如果有地质模型
    if geo_model is not None:
        print("Generating 3D model figures...")
        figures['fence_diagram'] = geo_plots.plot_fence_diagram(df, geo_model, return_plotly=True)

        stats = geo_model.get_statistics(result.get('lithology_classes', []))
        figures['volume_statistics'] = result_plots.plot_volume_statistics(stats, return_plotly=True)

    print(f"Generated {len(figures)} figures")
    return figures


# 快捷访问
style = SCIFigureStyle()
geology = GeologyPlots(style)
ml = MLPlots(style)
results = ResultPlots(style)
exporter = FigureExporter()
