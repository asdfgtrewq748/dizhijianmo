"""
可视化前端模块 - 基于Streamlit和Plotly
提供交互式三维地质模型可视化界面

针对敏东矿区钻孔数据优化
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import torch
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.data_loader import BoreholeDataProcessor, GridInterpolator
from src.trainer import GeoModelTrainer, compute_class_weights
from src.modeling import StratigraphicModel3D, build_stratigraphic_model_from_df


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="GNN三维地质建模系统 - 敏东矿区",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式 ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SCI论文配图样式配置 ====================
# 专业配色方案 - 适合地质图
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
    '#AD002A',  # 酒红
    '#ADB6B6',  # 银灰
]

# SCI论文图表通用配置
SCI_LAYOUT = dict(
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color='#333333'
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=60, r=20, t=50, b=60),
)

# 坐标轴通用配置
SCI_AXIS = dict(
    showline=True,
    linewidth=1.5,
    linecolor='#333333',
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E5E5E5',
    zeroline=False,
    ticks='outside',
    tickwidth=1.5,
    tickcolor='#333333',
    title_font=dict(size=12, family="Arial, sans-serif"),
    mirror=True,
)

# 图例通用配置
SCI_LEGEND = dict(
    font=dict(size=10, family="Arial, sans-serif"),
    bgcolor='rgba(255,255,255,0.9)',
    bordercolor='#CCCCCC',
    borderwidth=1,
)


def get_color_palette(n: int) -> list:
    """Return a palette with at least n distinct colors for geological data."""
    if n <= len(GEOLOGY_COLORS):
        return GEOLOGY_COLORS[:n]

    # 如果需要更多颜色，扩展调色板
    extended = GEOLOGY_COLORS.copy()
    additional = (
        px.colors.qualitative.Set2
        + px.colors.qualitative.Pastel1
        + px.colors.qualitative.Dark2
    )
    extended.extend(additional)

    if n <= len(extended):
        return extended[:n]

    repeats = (n + len(extended) - 1) // len(extended)
    return (extended * repeats)[:n]


def apply_sci_style(fig: go.Figure, height: int = 500) -> go.Figure:
    """应用SCI论文样式到图表"""
    fig.update_layout(
        **SCI_LAYOUT,
        height=height,
    )
    fig.update_xaxes(**SCI_AXIS)
    fig.update_yaxes(**SCI_AXIS)
    return fig


# ==================== 可视化函数 ====================
def create_cylinder_mesh(x_center, y_center, z_top, z_bottom, radius, n_sides=16):
    """
    创建圆柱体的网格数据
    返回用于绘制圆柱体侧面的坐标
    """
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)

    # 圆柱体侧面的坐标
    x_circle = x_center + radius * np.cos(theta)
    y_circle = y_center + radius * np.sin(theta)

    # 创建侧面网格
    x_surf = np.array([x_circle, x_circle])
    y_surf = np.array([y_circle, y_circle])
    z_surf = np.array([[z_top] * len(theta), [z_bottom] * len(theta)])

    return x_surf, y_surf, z_surf


def plot_borehole_cylinders_3d(df: pd.DataFrame, cylinder_radius: float = None) -> go.Figure:
    """
    绘制三维钻孔圆柱体图 - 优化版本，使用Mesh3d批量渲染
    """
    fig = go.Figure()

    # 获取岩性类别和颜色
    lithology_categories = sorted(df['lithology'].unique())
    colors = get_color_palette(len(lithology_categories))
    color_map = {category: colors[idx] for idx, category in enumerate(lithology_categories)}

    # 颜色转RGB数值 - 支持多种格式
    def color_to_rgb(color_str):
        """将颜色字符串转换为RGB元组，支持hex和rgb()格式"""
        if color_str.startswith('#'):
            hex_color = color_str.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        elif color_str.startswith('rgb'):
            # 处理 rgb(r, g, b) 格式
            import re
            match = re.search(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
            if match:
                return tuple(int(x) for x in match.groups())
        # 默认返回灰色
        return (128, 128, 128)

    color_map_rgb = {k: color_to_rgb(v) for k, v in color_map.items()}

    # 自动计算圆柱体半径
    if cylinder_radius is None:
        borehole_coords = df.groupby('borehole_id')[['x', 'y']].first().values
        if len(borehole_coords) > 1:
            from scipy.spatial import distance
            dists = distance.pdist(borehole_coords)
            min_dist = np.min(dists) if len(dists) > 0 else 100
            cylinder_radius = min_dist * 0.06
        else:
            cylinder_radius = 50

    # 按岩性分组收集所有圆柱体数据
    lithology_meshes = {litho: {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': [], 'hover': []}
                        for litho in lithology_categories}

    n_sides = 12  # 减少面数提高性能
    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    boreholes = df['borehole_id'].unique()

    for bh_id in boreholes:
        bh_data = df[df['borehole_id'] == bh_id].copy()
        x_center = bh_data['x'].iloc[0]
        y_center = bh_data['y'].iloc[0]

        # 按层序获取每层信息
        if 'layer_order' in bh_data.columns:
            layers = bh_data.groupby('layer_order').agg({
                'lithology': 'first',
                'top_depth': 'first',
                'bottom_depth': 'first',
                'layer_thickness': 'first'
            }).reset_index().sort_values('layer_order')
        else:
            continue

        # 合并相邻同岩性层以减少对象数
        merged_layers = []
        current_layer = None
        for _, layer in layers.iterrows():
            if current_layer is None:
                current_layer = {
                    'lithology': layer['lithology'],
                    'top_depth': layer['top_depth'],
                    'bottom_depth': layer['bottom_depth'],
                    'layer_thickness': layer['layer_thickness']
                }
            elif current_layer['lithology'] == layer['lithology']:
                # 合并相邻同岩性层
                current_layer['bottom_depth'] = layer['bottom_depth']
                current_layer['layer_thickness'] += layer['layer_thickness']
            else:
                merged_layers.append(current_layer)
                current_layer = {
                    'lithology': layer['lithology'],
                    'top_depth': layer['top_depth'],
                    'bottom_depth': layer['bottom_depth'],
                    'layer_thickness': layer['layer_thickness']
                }
        if current_layer:
            merged_layers.append(current_layer)

        # 为每层添加圆柱体网格数据
        for layer in merged_layers:
            lithology = layer['lithology']
            z_top = -layer['top_depth']
            z_bottom = -layer['bottom_depth']

            mesh_data = lithology_meshes[lithology]
            base_idx = len(mesh_data['x'])

            # 添加顶部和底部圆的顶点
            for z_val in [z_top, z_bottom]:
                for ci, si in zip(cos_theta, sin_theta):
                    mesh_data['x'].append(x_center + cylinder_radius * ci)
                    mesh_data['y'].append(y_center + cylinder_radius * si)
                    mesh_data['z'].append(z_val)

            # 添加侧面三角形
            for idx in range(n_sides):
                next_idx = (idx + 1) % n_sides
                # 顶部索引
                t1, t2 = base_idx + idx, base_idx + next_idx
                # 底部索引
                b1, b2 = base_idx + n_sides + idx, base_idx + n_sides + next_idx
                # 两个三角形组成一个侧面
                mesh_data['i'].extend([t1, t1])
                mesh_data['j'].extend([t2, b1])
                mesh_data['k'].extend([b1, b2])

    # 为每种岩性创建一个Mesh3d
    for lithology in lithology_categories:
        mesh_data = lithology_meshes[lithology]
        if not mesh_data['x']:
            continue

        rgb = color_map_rgb[lithology]
        fig.add_trace(go.Mesh3d(
            x=mesh_data['x'],
            y=mesh_data['y'],
            z=mesh_data['z'],
            i=mesh_data['i'],
            j=mesh_data['j'],
            k=mesh_data['k'],
            color=f'rgb({rgb[0]},{rgb[1]},{rgb[2]})',
            opacity=0.9,
            name=lithology,
            showlegend=True,
            flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.2, roughness=0.5),
            lightposition=dict(x=1000, y=1000, z=1000),
            hoverinfo='name'
        ))

    # 3D场景配置
    scene_axis = dict(
        backgroundcolor='#F8F9FA',
        gridcolor='#DEE2E6',
        gridwidth=1,
        showbackground=True,
        linecolor='#495057',
        linewidth=2,
        tickfont=dict(size=10, family="Arial"),
        title_font=dict(size=12, family="Arial", color='#212529'),
    )

    fig.update_layout(
        title=dict(
            text="<b>3D Borehole Stratigraphic Model</b>",
            font=dict(size=16, family="Arial", color='#212529'),
            x=0.5, xanchor='center'
        ),
        scene=dict(
            xaxis=dict(**scene_axis, title="<b>X (m)</b>"),
            yaxis=dict(**scene_axis, title="<b>Y (m)</b>"),
            zaxis=dict(**scene_axis, title="<b>Elevation (m)</b>"),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0), up=dict(x=0, y=0, z=1))
        ),
        legend=dict(
            **SCI_LEGEND,
            title=dict(text="<b>Lithology</b>", font=dict(size=12)),
            yanchor="top", y=0.98, xanchor="left", x=0.02,
            itemsizing='constant'
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=60, b=0),
        height=700
    )

    return fig


def plot_borehole_3d(df: pd.DataFrame, color_col: str = 'lithology') -> go.Figure:
    """
    绘制三维钻孔散点图 - SCI论文质量
    """
    fig = go.Figure()

    categories = sorted(df[color_col].unique())
    colors = get_color_palette(len(categories))
    color_map = {category: colors[idx] for idx, category in enumerate(categories)}

    for category in categories:
        mask = df[color_col] == category
        subset = df[mask]

        fig.add_trace(go.Scatter3d(
            x=subset['x'],
            y=subset['y'],
            z=subset['z'],
            mode='markers',
            name=str(category),
            marker=dict(
                size=4,
                color=color_map[category],
                opacity=0.85,
                line=dict(width=0.5, color='#333333')
            ),
            hovertemplate=(
                f"<b>{category}</b><br>"
                "X: %{x:.1f} m<br>"
                "Y: %{y:.1f} m<br>"
                "Z: %{z:.1f} m<br>"
                "<extra></extra>"
            )
        ))

    # SCI风格的3D场景配置
    scene_axis = dict(
        backgroundcolor='#FAFAFA',
        gridcolor='#E0E0E0',
        gridwidth=1,
        showbackground=True,
        linecolor='#333333',
        linewidth=2,
        tickfont=dict(size=10, family="Arial"),
        title_font=dict(size=12, family="Arial"),
    )

    fig.update_layout(
        title=dict(
            text="<b>3D Borehole Data Visualization</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(**scene_axis, title="X (m)"),
            yaxis=dict(**scene_axis, title="Y (m)"),
            zaxis=dict(**scene_axis, title="Depth (m)"),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        legend=dict(
            **SCI_LEGEND,
            title=dict(text="<b>Lithology</b>", font=dict(size=11)),
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            itemsizing='constant'
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )

    return fig


def plot_predictions_3d(
    coords: np.ndarray,
    predictions: np.ndarray,
    lithology_classes: list,
    true_labels: np.ndarray = None,
    show_errors: bool = False
) -> go.Figure:
    """绘制预测结果的三维可视化 - SCI论文质量"""
    fig = go.Figure()

    colors = get_color_palette(len(lithology_classes))

    for i, class_name in enumerate(lithology_classes):
        mask = predictions == i

        if show_errors and true_labels is not None:
            correct_mask = mask & (predictions == true_labels)
            error_mask = mask & (predictions != true_labels)

            if correct_mask.any():
                fig.add_trace(go.Scatter3d(
                    x=coords[correct_mask, 0],
                    y=coords[correct_mask, 1],
                    z=coords[correct_mask, 2],
                    mode='markers',
                    name=f"{class_name} (Correct)",
                    marker=dict(
                        size=4,
                        color=colors[i],
                        opacity=0.85,
                        line=dict(width=0.5, color='#333333')
                    ),
                ))

            if error_mask.any():
                fig.add_trace(go.Scatter3d(
                    x=coords[error_mask, 0],
                    y=coords[error_mask, 1],
                    z=coords[error_mask, 2],
                    mode='markers',
                    name=f"{class_name} (Error)",
                    marker=dict(
                        size=6,
                        color=colors[i],
                        opacity=1.0,
                        symbol='x',
                        line=dict(width=2, color='#DC0000')
                    ),
                ))
        else:
            if mask.any():
                fig.add_trace(go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode='markers',
                    name=class_name,
                    marker=dict(
                        size=4,
                        color=colors[i],
                        opacity=0.85,
                        line=dict(width=0.5, color='#333333')
                    ),
                ))

    # SCI风格的3D场景配置
    scene_axis = dict(
        backgroundcolor='#FAFAFA',
        gridcolor='#E0E0E0',
        gridwidth=1,
        showbackground=True,
        linecolor='#333333',
        linewidth=2,
        tickfont=dict(size=10, family="Arial"),
        title_font=dict(size=12, family="Arial"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Model Prediction Results</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(**scene_axis, title="X (m)"),
            yaxis=dict(**scene_axis, title="Y (m)"),
            zaxis=dict(**scene_axis, title="Depth (m)"),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        legend=dict(
            **SCI_LEGEND,
            title=dict(text="<b>Lithology</b>", font=dict(size=11)),
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            itemsizing='constant'
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )

    return fig


def plot_borehole_column(df: pd.DataFrame, borehole_id: str) -> go.Figure:
    """绘制单个钻孔柱状图，保持层序，不合并同名岩层。"""
    bh_data = df[df['borehole_id'] == borehole_id].copy()

    if 'layer_order' in bh_data.columns:
        layers = (bh_data
                  .sort_values('layer_order')
                  .drop_duplicates('layer_order'))
    else:
        layers = bh_data.sort_values('z', ascending=False)

    if layers.empty:
        return go.Figure()

    lithologies = sorted(bh_data['lithology'].unique())
    colors = get_color_palette(len(lithologies))
    color_map = {lithology: colors[idx] for idx, lithology in enumerate(lithologies)}

    fig = go.Figure()

    for _, row in layers.iterrows():
        top_depth = row.get('top_depth', None)
        bottom_depth = row.get('bottom_depth', None)
        depth_range = None
        if top_depth is not None and bottom_depth is not None:
            depth_range = f"{top_depth:.1f} ~ {bottom_depth:.1f} m"

        fig.add_trace(go.Bar(
            x=[row['layer_thickness']],
            y=[row['lithology']],
            orientation='h',
            marker=dict(
                color=color_map[row['lithology']],
                line=dict(color='#333333', width=1)
            ),
            text=f"厚度: {row['layer_thickness']:.1f}m" + (f" | 深度: {depth_range}" if depth_range else ""),
            textposition='inside',
            showlegend=False,
            hovertemplate=(
                f"岩性: {row['lithology']}<br>"
                + (f"深度范围: {depth_range}<br>" if depth_range else "")
                + f"厚度: {row['layer_thickness']:.1f}m<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>Borehole {borehole_id} Stratigraphic Column</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>Thickness (m)</b>",
        yaxis_title="<b>Lithology</b>",
        barmode='stack',
        height=600,
        yaxis=dict(autorange='reversed'),
        **SCI_LAYOUT
    )
    fig.update_xaxes(**SCI_AXIS)
    fig.update_yaxes(**SCI_AXIS, tickfont=dict(size=10))

    return fig


def plot_cross_section(
    coords: np.ndarray,
    predictions: np.ndarray,
    lithology_classes: list,
    axis: str = 'x',
    position: float = None,
    thickness: float = 100
) -> go.Figure:
    """绘制剖面图 - SCI论文质量"""
    axis_idx = {'x': 0, 'y': 1}[axis]
    other_axes = [1, 2] if axis == 'x' else [0, 2]

    if position is None:
        position = coords[:, axis_idx].mean()

    mask = np.abs(coords[:, axis_idx] - position) <= thickness / 2

    fig = go.Figure()
    colors = get_color_palette(len(lithology_classes))

    for i, class_name in enumerate(lithology_classes):
        class_mask = mask & (predictions == i)
        if class_mask.any():
            fig.add_trace(go.Scatter(
                x=coords[class_mask, other_axes[0]],
                y=coords[class_mask, other_axes[1]],
                mode='markers',
                name=class_name,
                marker=dict(
                    size=8,
                    color=colors[i],
                    opacity=0.85,
                    line=dict(width=0.5, color='#333333'),
                    symbol='circle'
                )
            ))

    xlabel = '<b>Y (m)</b>' if axis == 'x' else '<b>X (m)</b>'
    fig.update_layout(
        title=dict(
            text=f"<b>Cross Section ({axis.upper()}={position:.1f} m, Width: ±{thickness/2:.0f} m)</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=xlabel,
        yaxis_title="<b>Depth (m)</b>",
        legend=dict(
            **SCI_LEGEND,
            title=dict(text="<b>Lithology</b>", font=dict(size=11)),
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        ),
        height=600,
        **SCI_LAYOUT
    )
    fig.update_xaxes(**SCI_AXIS)
    fig.update_yaxes(**SCI_AXIS)

    return fig


def plot_training_history(history: dict) -> go.Figure:
    """绘制训练历史曲线 - SCI论文质量"""
    fig = go.Figure()

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#3C5488', width=2),
        marker=dict(size=4, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#E64B35', width=2),
        marker=dict(size=4, symbol='square')
    ))

    fig.update_layout(
        title=dict(
            text="<b>Training Progress - Loss Curve</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Loss</b>",
        legend=dict(
            **SCI_LEGEND,
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        ),
        height=400,
        **SCI_LAYOUT
    )
    fig.update_xaxes(**SCI_AXIS)
    fig.update_yaxes(**SCI_AXIS)

    return fig


def plot_accuracy_history(history: dict) -> go.Figure:
    """绘制准确率曲线 - SCI论文质量"""
    fig = go.Figure()

    epochs = list(range(1, len(history['train_acc']) + 1))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_acc'],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='#3C5488', width=2),
        marker=dict(size=4, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_acc'],
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='#E64B35', width=2),
        marker=dict(size=4, symbol='square')
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_f1'],
        mode='lines+markers',
        name='Validation F1-Score',
        line=dict(color='#00A087', width=2, dash='dash'),
        marker=dict(size=4, symbol='diamond')
    ))

    fig.update_layout(
        title=dict(
            text="<b>Training Progress - Accuracy Curve</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Score</b>",
        legend=dict(
            **SCI_LEGEND,
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        ),
        height=400,
        **SCI_LAYOUT
    )
    fig.update_xaxes(**SCI_AXIS)
    fig.update_yaxes(**SCI_AXIS, range=[0, 1.05])

    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> go.Figure:
    """绘制混淆矩阵 - SCI论文质量"""
    # 计算归一化混淆矩阵（按行归一化，显示召回率）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零

    # 创建注释文本：显示数量和百分比
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(f"{cm[i, j]}<br>({cm_normalized[i, j]*100:.1f}%)")

    annotations = np.array(annotations).reshape(cm.shape)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=[
            [0, '#FFFFFF'],
            [0.2, '#C6DBEF'],
            [0.4, '#6BAED6'],
            [0.6, '#2171B5'],
            [0.8, '#08519C'],
            [1.0, '#08306B']
        ],
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=10, family="Arial"),
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="<b>Count</b>", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=15
        )
    ))

    fig.update_layout(
        title=dict(
            text="<b>Confusion Matrix</b>",
            font=dict(size=14, family="Arial", color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>Predicted Class</b>",
        yaxis_title="<b>True Class</b>",
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10, family="Arial"),
            title_font=dict(size=12, family="Arial"),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=10, family="Arial"),
            title_font=dict(size=12, family="Arial"),
            autorange='reversed'  # 使对角线从左上到右下
        ),
        height=550,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=20, t=60, b=100)
    )

    return fig


# ==================== 主应用 ====================
def main():
    st.markdown('<h1 class="main-header">🏔️ GNN三维地质建模系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">敏东矿区钻孔数据分析</p>', unsafe_allow_html=True)

    # 获取项目路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')

    # 侧边栏 - 参数设置
    with st.sidebar:
        st.header("⚙️ 参数设置")

        # 数据设置
        st.subheader("📊 数据配置")
        sample_interval = st.slider("采样间隔 (米)", 0.5, 5.0, 2.0, 0.5)

        # 图构建设置
        st.subheader("🔗 图构建")
        graph_type = st.selectbox("图类型", ['knn', 'radius', 'delaunay'])
        k_neighbors = st.slider("K邻居数", 5, 25, 15)

        # 模型设置
        st.subheader("🧠 模型配置")
        model_type = st.selectbox("模型类型", ['enhanced', 'graphsage', 'gcn', 'gat', 'geo3d'])
        hidden_dim = st.selectbox("隐藏层维度", [64, 128, 256], index=1)
        num_layers = st.slider("GNN层数", 2, 6, 4)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3)

        # 训练设置
        st.subheader("🎯 训练配置")
        learning_rate = st.select_slider(
            "学习率",
            options=[0.001, 0.005, 0.01, 0.02],
            value=0.005
        )
        epochs = st.slider("训练轮数", 100, 500, 300)
        patience = st.slider("早停耐心值", 20, 80, 50)

    # 主区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 数据探索", "🚀 模型训练", "📈 结果分析", "🗺️ 三维可视化", "🏗️ 地质建模"])

    # 初始化session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'probs' not in st.session_state:
        st.session_state.probs = None

    # Tab 1: 数据探索
    with tab1:
        st.header("数据探索与预处理")

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("🔄 加载敏东矿区数据", type="primary"):
                with st.spinner("正在加载钻孔数据..."):
                    try:
                        # 创建处理器并加载数据
                        processor = BoreholeDataProcessor(
                            k_neighbors=k_neighbors,
                            graph_type=graph_type,
                            sample_interval=sample_interval
                        )

                        df = processor.load_all_boreholes(data_dir)

                        # 处理数据
                        result = processor.process(
                            df,
                            feature_cols=['layer_thickness'],
                            test_size=0.2,
                            val_size=0.1
                        )

                        st.session_state.df = df
                        st.session_state.data = result['data']
                        st.session_state.processor = processor
                        st.session_state.result = result

                        st.success(f"数据加载成功! 共 {len(df)} 个采样点")

                    except Exception as e:
                        st.error(f"数据加载失败: {str(e)}")

        with col2:
            if st.session_state.data is not None:
                data = st.session_state.data
                result = st.session_state.result
                df = st.session_state.df

                # 显示统计信息
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("采样点数", data.num_nodes)
                col_b.metric("图边数", data.num_edges)
                col_c.metric("钻孔数", df['borehole_id'].nunique())
                col_d.metric("岩性类别", result['num_classes'])

        if st.session_state.data is not None:
            df = st.session_state.df
            result = st.session_state.result

            # 数据预览
            st.subheader("数据预览")
            st.dataframe(
                df[['borehole_id', 'x', 'y', 'z', 'lithology', 'layer_thickness']].head(20),
                width="stretch"
            )

            # 三维可视化
            st.subheader("钻孔分布可视化")

            # 可视化方式选择
            vis_col1, vis_col2 = st.columns([1, 3])
            with vis_col1:
                vis_mode = st.radio(
                    "显示模式",
                    ["🔘 散点模式", "🧱 圆柱体模式"],
                    index=1,
                    help="圆柱体模式更直观地展示每个钻孔的地层结构"
                )
                if "圆柱体" in vis_mode:
                    cylinder_scale = st.slider("圆柱体大小", 0.5, 2.0, 1.0, 0.1,
                                               help="调整圆柱体的相对大小")

            with vis_col2:
                if "圆柱体" in vis_mode:
                    # 计算基础半径
                    borehole_coords = df.groupby('borehole_id')[['x', 'y']].first().values
                    if len(borehole_coords) > 1:
                        from scipy.spatial import distance
                        dists = distance.pdist(borehole_coords)
                        min_dist = np.min(dists) if len(dists) > 0 else 100
                        base_radius = min_dist * 0.08
                    else:
                        base_radius = 50
                    adjusted_radius = base_radius * cylinder_scale

                    fig = plot_borehole_cylinders_3d(df, cylinder_radius=adjusted_radius)
                else:
                    fig = plot_borehole_3d(df)

                st.plotly_chart(fig, use_container_width=True)

            # 统计图
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("岩性分布")
                litho_counts = df['lithology'].value_counts().sort_values(ascending=True)
                colors = get_color_palette(len(litho_counts))

                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=litho_counts.values,
                        y=litho_counts.index,
                        orientation='h',
                        marker=dict(
                            color=colors[:len(litho_counts)],
                            line=dict(color='#333333', width=1)
                        ),
                        text=litho_counts.values,
                        textposition='outside',
                        textfont=dict(size=10, family="Arial")
                    )
                ])
                fig_bar.update_layout(
                    title=dict(
                        text="<b>Lithology Distribution</b>",
                        font=dict(size=14, family="Arial", color='#333333'),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis_title="<b>Sample Count</b>",
                    yaxis_title="<b>Lithology</b>",
                    height=400,
                    showlegend=False,
                    **SCI_LAYOUT
                )
                fig_bar.update_xaxes(**SCI_AXIS)
                fig_bar.update_yaxes(**SCI_AXIS, tickfont=dict(size=10))
                st.plotly_chart(fig_bar, width="stretch")

            with col2:
                st.subheader("深度分布")
                fig_hist = go.Figure(data=[
                    go.Histogram(
                        x=df['z'],
                        nbinsx=50,
                        marker=dict(
                            color='#3C5488',
                            line=dict(color='#333333', width=0.5)
                        ),
                        opacity=0.85
                    )
                ])
                fig_hist.update_layout(
                    title=dict(
                        text="<b>Depth Distribution</b>",
                        font=dict(size=14, family="Arial", color='#333333'),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis_title="<b>Depth (m)</b>",
                    yaxis_title="<b>Frequency</b>",
                    height=400,
                    bargap=0.05,
                    **SCI_LAYOUT
                )
                fig_hist.update_xaxes(**SCI_AXIS)
                fig_hist.update_yaxes(**SCI_AXIS)
                st.plotly_chart(fig_hist, width="stretch")

            # 单钻孔柱状图
            st.subheader("钻孔柱状图")
            borehole_ids = df['borehole_id'].unique().tolist()
            selected_bh = st.selectbox("选择钻孔", borehole_ids)
            if selected_bh:
                fig_col = plot_borehole_column(df, selected_bh)
                st.plotly_chart(fig_col, width="stretch")

    # Tab 2: 模型训练
    with tab2:
        st.header("模型训练")

        if st.session_state.data is None:
            st.warning("⚠️ 请先在'数据探索'标签页加载数据")
            st.stop()

        data = st.session_state.data
        result = st.session_state.result

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("模型信息")
            st.write(f"**模型类型:** {model_type.upper()}")
            st.write(f"**输入特征:** {result['num_features']}")
            st.write(f"**输出类别:** {result['num_classes']}")
            st.write(f"**隐藏维度:** {hidden_dim}")
            st.write(f"**GNN层数:** {num_layers}")

            st.subheader("数据集划分")
            st.write(f"训练集: {data.train_mask.sum().item()}")
            st.write(f"验证集: {data.val_mask.sum().item()}")
            st.write(f"测试集: {data.test_mask.sum().item()}")

            use_class_weights = st.checkbox("使用类别权重", value=True)

        with col2:
            if st.button("🚀 开始训练", type="primary"):
                # 创建模型
                model = get_model(
                    model_type,
                    in_channels=result['num_features'],
                    hidden_channels=hidden_dim,
                    out_channels=result['num_classes'],
                    num_layers=num_layers,
                    dropout=dropout
                )

                # 类别权重
                class_weights = compute_class_weights(data.y) if use_class_weights else None

                # 创建训练器 - 使用Focal Loss
                trainer = GeoModelTrainer(
                    model=model,
                    learning_rate=learning_rate,
                    class_weights=class_weights,
                    loss_type='focal',
                    num_classes=result['num_classes'],
                    focal_gamma=2.0
                )

                # 训练进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()

                def update_progress(epoch, train_loss, val_loss, val_acc):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}")
                    metrics_placeholder.write(
                        f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}"
                    )

                # 训练
                with st.spinner("训练中..."):
                    history = trainer.train(
                        data,
                        epochs=epochs,
                        patience=patience,
                        verbose=False,
                        callback=update_progress
                    )

                # 保存状态
                st.session_state.model = model
                st.session_state.trainer = trainer
                st.session_state.history = history

                st.success(f"✅ 训练完成! 最佳验证准确率: {trainer.best_val_acc:.4f}")

        # 显示训练曲线
        if st.session_state.history is not None:
            st.subheader("训练曲线")
            col1, col2 = st.columns(2)
            with col1:
                fig_loss = plot_training_history(st.session_state.history)
                st.plotly_chart(fig_loss, width="stretch")
            with col2:
                fig_acc = plot_accuracy_history(st.session_state.history)
                st.plotly_chart(fig_acc, width="stretch")

    # Tab 3: 结果分析
    with tab3:
        st.header("结果分析")

        if st.session_state.trainer is None:
            st.warning("⚠️ 请先训练模型")
            st.stop()

        trainer = st.session_state.trainer
        data = st.session_state.data
        result = st.session_state.result

        if st.button("📊 评估模型", type="primary"):
            with st.spinner("评估中..."):
                eval_results = trainer.evaluate(data, result['lithology_classes'])
                predictions, probs = trainer.predict(data, return_probs=True)

                st.session_state.eval_results = eval_results
                st.session_state.predictions = predictions
                st.session_state.probs = probs

        if 'eval_results' in st.session_state and st.session_state.eval_results is not None:
            eval_results = st.session_state.eval_results

            # 关键指标
            col1, col2, col3 = st.columns(3)
            col1.metric("测试准确率", f"{eval_results['accuracy']:.4f}")
            col2.metric("F1 (Macro)", f"{eval_results['f1_macro']:.4f}")
            col3.metric("F1 (Weighted)", f"{eval_results['f1_weighted']:.4f}")

            # 混淆矩阵
            st.subheader("混淆矩阵")
            fig_cm = plot_confusion_matrix(
                eval_results['confusion_matrix'],
                result['lithology_classes']
            )
            st.plotly_chart(fig_cm, width="stretch")

            # 分类报告
            st.subheader("详细分类报告")
            report_df = pd.DataFrame(eval_results['classification_report']).transpose()
            st.dataframe(report_df, width="stretch")
        else:
            st.info("请点击上方“评估模型”获取测试集指标")

    # Tab 4: 三维可视化
    with tab4:
        st.header("三维模型可视化")

        if st.session_state.predictions is None:
            st.warning("⚠️ 请先在'结果分析'标签页进行模型评估")
            st.stop()

        data = st.session_state.data
        result = st.session_state.result
        predictions = st.session_state.predictions
        coords = data.coords.cpu().numpy()

        # 可视化选项
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("显示选项")
            show_type = st.radio("显示内容", ["预测结果", "真实标签", "对比"])
            show_errors = st.checkbox("高亮错误预测", value=False)

            st.subheader("剖面设置")
            section_axis = st.selectbox("剖面方向", ['x', 'y'])
            axis_range = coords[:, 0 if section_axis == 'x' else 1]
            section_pos = st.slider(
                "剖面位置",
                float(axis_range.min()),
                float(axis_range.max()),
                float(axis_range.mean())
            )
            section_thickness = st.slider("剖面厚度 (m)", 50, 500, 200)

        with col2:
            # 三维散点图
            if show_type == "预测结果":
                fig_3d = plot_predictions_3d(
                    coords, predictions,
                    result['lithology_classes'],
                    data.y.cpu().numpy() if show_errors else None,
                    show_errors
                )
            elif show_type == "真实标签":
                fig_3d = plot_predictions_3d(
                    coords, data.y.cpu().numpy(),
                    result['lithology_classes']
                )
            else:
                fig_3d = plot_predictions_3d(
                    coords, predictions,
                    result['lithology_classes'],
                    data.y.cpu().numpy(),
                    show_errors=True
                )

            st.plotly_chart(fig_3d, width="stretch")

        # 剖面图
        st.subheader("剖面视图")
        col1, col2 = st.columns(2)

        with col1:
            fig_section_pred = plot_cross_section(
                coords, predictions,
                result['lithology_classes'],
                axis=section_axis,
                position=section_pos,
                thickness=section_thickness
            )
            fig_section_pred.update_layout(title="预测剖面")
            st.plotly_chart(fig_section_pred, width="stretch")

        with col2:
            fig_section_true = plot_cross_section(
                coords, data.y.cpu().numpy(),
                result['lithology_classes'],
                axis=section_axis,
                position=section_pos,
                thickness=section_thickness
            )
            fig_section_true.update_layout(title="真实剖面")
            st.plotly_chart(fig_section_true, width="stretch")

    # Tab 5: 地质建模
    with tab5:
        st.header("三维地质体建模")

        if st.session_state.predictions is None:
            st.warning("⚠️ 请先在'结果分析'标签页进行模型评估")
            st.stop()

        data = st.session_state.data
        result = st.session_state.result
        trainer = st.session_state.trainer
        predictions = st.session_state.predictions
        probs = st.session_state.probs

        # 建模参数
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("网格参数")
            nx = st.slider("X方向网格数", 20, 100, 50)
            ny = st.slider("Y方向网格数", 20, 100, 50)
            nz = st.slider("Z方向网格数", 20, 100, 40)

            interp_method = st.selectbox("插值方法", ['rbf', 'idw', 'linear'],
                                          help="RBF(径向基函数)插值效果最好")

        with col2:
            if st.button("🏗️ 构建三维地质模型", type="primary"):
                with st.spinner("正在构建层状三维地质模型..."):
                    # 创建层状地质模型
                    geo_model = StratigraphicModel3D(
                        resolution=(nx, ny, nz),
                        interpolation_method=interp_method,
                        smoothing=0.1
                    )

                    # 使用原始钻孔数据构建层状模型
                    geo_model.build_stratigraphic_model(st.session_state.df, result['lithology_classes'])

                    st.session_state.geo_model = geo_model

                    # 获取统计
                    stats = geo_model.get_statistics(result['lithology_classes'])
                    st.session_state.model_stats = stats

                    st.success(f"✅ 层状地质模型构建完成! 共 {nx*ny*nz:,} 个体素")

        # 显示模型信息和统计
        if 'geo_model' in st.session_state:
            geo_model = st.session_state.geo_model
            stats = st.session_state.model_stats

            st.subheader("岩性体积统计")
            st.dataframe(stats, width="stretch")

            # 可视化切片
            st.subheader("模型切片可视化")

            slice_col1, slice_col2 = st.columns([1, 3])

            with slice_col1:
                slice_axis = st.selectbox("切片方向", ['z', 'x', 'y'], key='slice_axis')
                grid_info = geo_model.grid_info

                if slice_axis == 'z':
                    z_range = grid_info['z_grid']
                    slice_pos = st.slider("切片位置 (深度)", float(z_range.min()), float(z_range.max()), float(z_range.mean()))
                elif slice_axis == 'x':
                    x_range = grid_info['x_grid']
                    slice_pos = st.slider("切片位置 (X)", float(x_range.min()), float(x_range.max()), float(x_range.mean()))
                else:
                    y_range = grid_info['y_grid']
                    slice_pos = st.slider("切片位置 (Y)", float(y_range.min()), float(y_range.max()), float(y_range.mean()))

            with slice_col2:
                # 获取切片
                slice_data, slice_coords, slice_info = geo_model.get_slice(slice_axis, position=slice_pos)

                # 绘制切片 - SCI论文质量
                fig_slice = go.Figure()

                colors = get_color_palette(len(result['lithology_classes']))

                if slice_axis == 'z':
                    for i, class_name in enumerate(result['lithology_classes']):
                        mask = slice_data == i
                        if mask.any():
                            fig_slice.add_trace(go.Scatter(
                                x=slice_coords['x'][mask].flatten(),
                                y=slice_coords['y'][mask].flatten(),
                                mode='markers',
                                name=class_name,
                                marker=dict(
                                    size=6,
                                    color=colors[i],
                                    opacity=0.85,
                                    line=dict(width=0.3, color='#333333')
                                )
                            ))
                    fig_slice.update_layout(
                        title=dict(
                            text=f"<b>Horizontal Slice (Z = {slice_pos:.1f} m)</b>",
                            font=dict(size=14, family="Arial", color='#333333'),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="<b>X (m)</b>",
                        yaxis_title="<b>Y (m)</b>"
                    )
                elif slice_axis == 'x':
                    for i, class_name in enumerate(result['lithology_classes']):
                        mask = slice_data == i
                        if mask.any():
                            fig_slice.add_trace(go.Scatter(
                                x=slice_coords['y'][mask].flatten(),
                                y=slice_coords['z'][mask].flatten(),
                                mode='markers',
                                name=class_name,
                                marker=dict(
                                    size=6,
                                    color=colors[i],
                                    opacity=0.85,
                                    line=dict(width=0.3, color='#333333')
                                )
                            ))
                    fig_slice.update_layout(
                        title=dict(
                            text=f"<b>X Cross Section (X = {slice_pos:.1f} m)</b>",
                            font=dict(size=14, family="Arial", color='#333333'),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="<b>Y (m)</b>",
                        yaxis_title="<b>Z (m)</b>"
                    )
                else:
                    for i, class_name in enumerate(result['lithology_classes']):
                        mask = slice_data == i
                        if mask.any():
                            fig_slice.add_trace(go.Scatter(
                                x=slice_coords['x'][mask].flatten(),
                                y=slice_coords['z'][mask].flatten(),
                                mode='markers',
                                name=class_name,
                                marker=dict(
                                    size=6,
                                    color=colors[i],
                                    opacity=0.85,
                                    line=dict(width=0.3, color='#333333')
                                )
                            ))
                    fig_slice.update_layout(
                        title=dict(
                            text=f"<b>Y Cross Section (Y = {slice_pos:.1f} m)</b>",
                            font=dict(size=14, family="Arial", color='#333333'),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="<b>X (m)</b>",
                        yaxis_title="<b>Z (m)</b>"
                    )

                # 应用SCI样式
                fig_slice.update_layout(
                    legend=dict(
                        **SCI_LEGEND,
                        title=dict(text="<b>Lithology</b>", font=dict(size=11)),
                        yanchor="top",
                        y=0.98,
                        xanchor="right",
                        x=0.98
                    ),
                    height=500,
                    **SCI_LAYOUT
                )
                fig_slice.update_xaxes(**SCI_AXIS)
                fig_slice.update_yaxes(**SCI_AXIS)

                st.plotly_chart(fig_slice, width="stretch")

            # ==================== 三维地质体模型可视化 ====================
            st.subheader("三维地质体模型")

            vis_col1, vis_col2 = st.columns([1, 3])

            with vis_col1:
                st.write("**显示设置**")
                opacity_3d = st.slider("透明度", 0.1, 1.0, 0.8, key='opacity_3d')
                show_all_layers = st.checkbox("显示所有岩层", value=True)

                if not show_all_layers:
                    selected_lithologies = st.multiselect(
                        "选择显示的岩性",
                        result['lithology_classes'],
                        default=result['lithology_classes'][:3] if len(result['lithology_classes']) > 3 else result['lithology_classes']
                    )
                else:
                    selected_lithologies = result['lithology_classes']

                surface_count = st.slider("曲面精细度", 1, 3, 2, help="值越大曲面越精细，但渲染越慢")

            with vis_col2:
                # 创建三维等值面可视化
                fig_3d_model = go.Figure()

                lithology_3d, confidence_3d = geo_model.get_voxel_model()
                colors = get_color_palette(len(result['lithology_classes']))

                # 获取网格信息
                x_grid = geo_model.grid_info['x_grid']
                y_grid = geo_model.grid_info['y_grid']
                z_grid = geo_model.grid_info['z_grid']

                # 颜色转换函数
                def color_to_rgb(color_str):
                    if color_str.startswith('#'):
                        hex_color = color_str.lstrip('#')
                        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    elif color_str.startswith('rgb'):
                        import re
                        match = re.search(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
                        if match:
                            return tuple(int(x) for x in match.groups())
                    return (128, 128, 128)

                # 使用 Isosurface 为每种岩性创建连续曲面
                # 先构建正确的坐标网格
                nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
                X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

                for i, class_name in enumerate(result['lithology_classes']):
                    if class_name not in selected_lithologies:
                        continue

                    # 创建该岩性的二值场（1表示该岩性，0表示其他）
                    binary_field = (lithology_3d == i).astype(float)

                    # 如果该岩性不存在，跳过
                    if binary_field.sum() == 0:
                        continue

                    # 对二值场进行轻微平滑以获得更好的等值面
                    from scipy.ndimage import gaussian_filter
                    smoothed_field = gaussian_filter(binary_field, sigma=0.8)

                    rgb = color_to_rgb(colors[i])

                    # 使用Isosurface绘制等值面
                    fig_3d_model.add_trace(go.Isosurface(
                        x=X.flatten(),
                        y=Y.flatten(),
                        z=Z.flatten(),
                        value=smoothed_field.flatten(),
                        isomin=0.3,
                        isomax=0.7,
                        surface_count=surface_count,
                        colorscale=[[0, f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'],
                                   [1, f'rgb({rgb[0]},{rgb[1]},{rgb[2]})']],
                        showscale=False,
                        opacity=opacity_3d,
                        name=class_name,
                        showlegend=True,
                        caps=dict(x_show=True, y_show=True, z_show=True),
                        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
                        lightposition=dict(x=1000, y=1000, z=500)
                    ))

                # 设置3D场景
                scene_axis = dict(
                    backgroundcolor='#FAFAFA',
                    gridcolor='#E0E0E0',
                    gridwidth=1,
                    showbackground=True,
                    linecolor='#333333',
                    linewidth=2,
                    tickfont=dict(size=10, family="Arial"),
                    title_font=dict(size=12, family="Arial"),
                )

                fig_3d_model.update_layout(
                    title=dict(
                        text="<b>3D Geological Model (Voxel Visualization)</b>",
                        font=dict(size=14, family="Arial", color='#333333'),
                        x=0.5,
                        xanchor='center'
                    ),
                    scene=dict(
                        xaxis=dict(**scene_axis, title="X (m)"),
                        yaxis=dict(**scene_axis, title="Y (m)"),
                        zaxis=dict(**scene_axis, title="Depth (m)"),
                        aspectmode='data',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    legend=dict(
                        **SCI_LEGEND,
                        title=dict(text="<b>Lithology</b>", font=dict(size=11)),
                        yanchor="top",
                        y=0.95,
                        xanchor="left",
                        x=0.02,
                        itemsizing='constant'
                    ),
                    paper_bgcolor='white',
                    margin=dict(l=0, r=0, t=50, b=0),
                    height=700
                )

                st.plotly_chart(fig_3d_model, use_container_width=True)

            # 导出按钮
            st.subheader("导出模型")
            col1, col2, col3 = st.columns(3)

            project_root = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(project_root, 'output')
            os.makedirs(output_dir, exist_ok=True)

            with col1:
                if st.button("📥 导出 VTK"):
                    vtk_path = os.path.join(output_dir, 'geological_model.vtk')
                    geo_model.export_vtk(vtk_path, result['lithology_classes'])
                    st.success(f"VTK文件已保存至:\n{vtk_path}")
                    st.info("提示: 使用 ParaView 打开 VTK 文件进行三维可视化")

            with col2:
                if st.button("📥 导出 CSV"):
                    csv_path = os.path.join(output_dir, 'geological_model.csv')
                    geo_model.export_csv(csv_path, result['lithology_classes'])
                    st.success(f"CSV文件已保存至:\n{csv_path}")

            with col3:
                if st.button("📥 导出 NumPy"):
                    npz_path = os.path.join(output_dir, 'geological_model.npz')
                    geo_model.export_numpy(npz_path)
                    st.success(f"NumPy文件已保存至:\n{npz_path}")


if __name__ == "__main__":
    main()
