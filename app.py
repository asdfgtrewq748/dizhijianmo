"""
GNN厚度预测三维地质建模 - Streamlit可视化前端 (新版)

使用正确的建模逻辑：GNN预测厚度 → 层序累加 → 三维模型
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import torch
import os
import sys
import base64
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入新版模块
from src.thickness_data_loader import ThicknessDataProcessor, LayerTableProcessor
from src.gnn_thickness_modeling import (
    GNNThicknessPredictor, GeologicalModelBuilder,
    GNNGeologicalModeling, TraditionalThicknessInterpolator
)
from src.thickness_trainer import (
    create_trainer, ThicknessTrainer, ThicknessEvaluator,
    k_fold_cross_validation, get_optimized_config_for_small_dataset
)
from src.thickness_predictor_v2 import (
    PerLayerThicknessPredictor, HybridThicknessPredictor, evaluate_predictor
)

# 尝试导入 PyVista 渲染器
PYVISTA_AVAILABLE = False
try:
    import pyvista as pv
    from src.pyvista_renderer import (
        GeologicalModelRenderer, RockMaterial, TextureGenerator,
        render_geological_model
    )
    PYVISTA_AVAILABLE = True
    # 设置 PyVista 为离屏渲染模式
    pv.OFF_SCREEN = True
except ImportError as e:
    print(f"PyVista 未安装或导入失败: {e}")

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="GNN三维地质建模系统 (新版)",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 地质岩石配色与材质系统 ====================

# 基于真实岩石的专业配色方案
ROCK_COLORS = {
    # 煤层 - 黑色系
    '煤': {'base': '#1a1a1a', 'highlight': '#333333', 'shadow': '#0d0d0d'},
    '煤层': {'base': '#1a1a1a', 'highlight': '#333333', 'shadow': '#0d0d0d'},

    # 砂岩 - 黄褐色系
    '砂岩': {'base': '#c4a35a', 'highlight': '#d4b76a', 'shadow': '#a48940'},
    '细砂岩': {'base': '#d4b878', 'highlight': '#e4c888', 'shadow': '#b49858'},
    '中砂岩': {'base': '#c4a35a', 'highlight': '#d4b76a', 'shadow': '#a48940'},
    '粗砂岩': {'base': '#b49347', 'highlight': '#c4a357', 'shadow': '#947327'},

    # 泥岩 - 灰绿色系
    '泥岩': {'base': '#6b7b6b', 'highlight': '#7b8b7b', 'shadow': '#5b6b5b'},
    '砂质泥岩': {'base': '#7a8a72', 'highlight': '#8a9a82', 'shadow': '#6a7a62'},

    # 页岩 - 深灰色系
    '页岩': {'base': '#4a5568', 'highlight': '#5a6578', 'shadow': '#3a4558'},
    '炭质页岩': {'base': '#3d4852', 'highlight': '#4d5862', 'shadow': '#2d3842'},

    # 粉砂岩 - 浅褐色系
    '粉砂岩': {'base': '#a89078', 'highlight': '#b8a088', 'shadow': '#988068'},

    # 灰�ite - 灰蓝色系
    '灰岩': {'base': '#8b9daa', 'highlight': '#9badba', 'shadow': '#7b8d9a'},
    '石灰岩': {'base': '#8b9daa', 'highlight': '#9badba', 'shadow': '#7b8d9a'},

    # 砾岩 - 棕红色系
    '砾岩': {'base': '#8b5a3c', 'highlight': '#9b6a4c', 'shadow': '#7b4a2c'},

    # 表土/黏土 - 土黄色系
    '表土': {'base': '#b5956c', 'highlight': '#c5a57c', 'shadow': '#a5855c'},
    '黏土': {'base': '#9a8060', 'highlight': '#aa9070', 'shadow': '#8a7050'},
    '土层': {'base': '#b5956c', 'highlight': '#c5a57c', 'shadow': '#a5855c'},
}

# 默认配色（用于未指定岩性）
DEFAULT_COLORS = [
    {'base': '#E64B35', 'highlight': '#F65B45', 'shadow': '#D63B25'},
    {'base': '#4DBBD5', 'highlight': '#5DCBE5', 'shadow': '#3DABC5'},
    {'base': '#00A087', 'highlight': '#10B097', 'shadow': '#009077'},
    {'base': '#3C5488', 'highlight': '#4C6498', 'shadow': '#2C4478'},
    {'base': '#F39B7F', 'highlight': '#FFAB8F', 'shadow': '#E38B6F'},
    {'base': '#8491B4', 'highlight': '#94A1C4', 'shadow': '#7481A4'},
    {'base': '#91D1C2', 'highlight': '#A1E1D2', 'shadow': '#81C1B2'},
    {'base': '#7E6148', 'highlight': '#8E7158', 'shadow': '#6E5138'},
]

def get_rock_color(rock_name):
    """获取岩石的专业配色"""
    # 尝试精确匹配
    if rock_name in ROCK_COLORS:
        return ROCK_COLORS[rock_name]

    # 尝试模糊匹配
    for key in ROCK_COLORS:
        if key in rock_name or rock_name in key:
            return ROCK_COLORS[key]

    return None

def get_color_map(layer_order):
    """获取岩层颜色映射（增强版）"""
    color_map = {}
    default_idx = 0

    for name in layer_order:
        rock_color = get_rock_color(name)
        if rock_color:
            color_map[name] = rock_color['base']
        else:
            color_map[name] = DEFAULT_COLORS[default_idx % len(DEFAULT_COLORS)]['base']
            default_idx += 1

    return color_map

def get_full_color_map(layer_order):
    """获取完整的岩层颜色映射（包含高光和阴影色）"""
    color_map = {}
    default_idx = 0

    for name in layer_order:
        rock_color = get_rock_color(name)
        if rock_color:
            color_map[name] = rock_color
        else:
            color_map[name] = DEFAULT_COLORS[default_idx % len(DEFAULT_COLORS)]
            default_idx += 1

    return color_map

def generate_rock_texture(shape, rock_type='sandstone', intensity=0.15):
    """
    生成程序化岩石纹理

    Args:
        shape: 网格形状 (rows, cols)
        rock_type: 岩石类型
        intensity: 纹理强度 (0-1)

    Returns:
        纹理数组，值域 [-intensity, intensity]
    """
    rows, cols = shape
    texture = np.zeros((rows, cols))

    if rock_type in ['coal', '煤', '煤层']:
        # 煤层 - 层状纹理
        for i in range(3):
            freq = 0.1 * (i + 1)
            texture += np.sin(np.linspace(0, 4*np.pi*freq, cols)) * (0.5 ** i)
        texture = np.tile(texture, (rows, 1))

    elif rock_type in ['sandstone', '砂岩', '细砂岩', '中砂岩', '粗砂岩']:
        # 砂岩 - 颗粒状纹理
        np.random.seed(42)
        noise = np.random.randn(rows, cols)
        # 高斯平滑
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(noise, sigma=1.5)

    elif rock_type in ['mudstone', '泥岩', '砂质泥岩']:
        # 泥岩 - 平滑层理
        y = np.linspace(0, 2*np.pi, rows)
        x = np.linspace(0, 4*np.pi, cols)
        X, Y = np.meshgrid(x, y)
        texture = np.sin(Y * 2) * 0.5 + np.sin(X * 0.5) * 0.3

    elif rock_type in ['shale', '页岩', '炭质页岩']:
        # 页岩 - 薄层状纹理
        for i in range(rows):
            texture[i, :] = np.sin(i * 0.5) * 0.7 + np.random.randn(cols) * 0.3

    elif rock_type in ['limestone', '灰岩', '石灰岩']:
        # 灰岩 - 不规则块状
        np.random.seed(42)
        texture = np.random.randn(rows, cols)
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(texture, sigma=3)

    elif rock_type in ['conglomerate', '砾岩']:
        # 砾岩 - 大颗粒纹理
        np.random.seed(42)
        texture = np.random.randn(rows // 3 + 1, cols // 3 + 1)
        texture = np.repeat(np.repeat(texture, 3, axis=0), 3, axis=1)[:rows, :cols]

    else:
        # 默认 - 轻微噪声
        np.random.seed(42)
        texture = np.random.randn(rows, cols) * 0.5
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(texture, sigma=2)

    # 归一化到指定强度范围
    if texture.max() != texture.min():
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        texture = texture * 2 * intensity - intensity

    return texture

def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """将RGB元组转换为十六进制颜色"""
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, min(255, int(rgb[0]))),
        max(0, min(255, int(rgb[1]))),
        max(0, min(255, int(rgb[2])))
    )

def create_textured_colorscale(base_color, texture_intensity=0.15):
    """
    创建带纹理效果的颜色比例尺

    Args:
        base_color: 基础颜色（十六进制）
        texture_intensity: 纹理强度

    Returns:
        Plotly colorscale列表
    """
    base_rgb = hex_to_rgb(base_color)

    # 创建颜色渐变，模拟光照效果
    colorscale = []
    steps = 10

    for i in range(steps + 1):
        t = i / steps
        # 从阴影到高光
        factor = 0.7 + 0.6 * t  # 从0.7到1.3
        adjusted_rgb = tuple(min(255, int(c * factor)) for c in base_rgb)
        colorscale.append([t, rgb_to_hex(adjusted_rgb)])

    return colorscale


# ==================== 主应用 ====================
def main():
    st.markdown('<h1 class="main-header">🏔️ GNN厚度预测三维地质建模</h1>', unsafe_allow_html=True)
    st.markdown('''
    <p style="text-align: center; color: gray;">
    使用正确的建模逻辑：<b>GNN预测厚度(回归)</b> → <b>层序累加</b> → <b>三维模型</b>
    </p>
    ''', unsafe_allow_html=True)

    # 项目路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')

    # 侧边栏参数
    with st.sidebar:
        st.header("⚙️ 参数设置")

        st.subheader("📊 数据配置")
        merge_coal = st.checkbox("合并煤层", value=False,
                                  help="是否将所有煤层合并为单一类别")
        layer_method = st.selectbox(
            "层序推断方法",
            ['position_based', 'simple', 'marker_based'],
            index=0,
            help="""
            - position_based: 按位置区分重复层（推荐，如粉砂岩_1、粉砂岩_2）
            - simple: 简单合并同名层（会丢失重复层）
            - marker_based: 以煤层为标志层对齐
            """
        )
        if layer_method == 'position_based':
            min_occurrence_rate = st.slider(
                "最小出现率",
                0.0, 0.5, 0.05,
                step=0.05,
                help="只保留出现率高于此值的地层。0.05表示至少在5%的钻孔中出现（约2个钻孔）。设为0可保留所有层。"
            )
        else:
            min_occurrence_rate = 0.05
        k_neighbors = st.slider("K邻居数", 4, 20, 10, help="增加邻居数可提高空间关联性")

        st.subheader("🔧 预测方法")
        prediction_method = st.radio(
            "选择厚度预测方法",
            ["传统方法（IDW/Kriging）", "GNN深度学习"],
            index=0,  # 默认传统方法
            help="传统方法更适合小样本数据（<50个钻孔），GNN适合大数据集"
        )
        use_traditional = prediction_method == "传统方法（IDW/Kriging）"

        if use_traditional:
            st.info("✅ 传统地质统计学方法：IDW + Kriging，对小样本数据效果好")
            interp_method = st.selectbox(
                "插值方法",
                ["idw", "kriging", "hybrid"],
                format_func=lambda x: {"idw": "反距离加权(IDW)", "kriging": "克里金", "hybrid": "混合方法"}[x],
                help="IDW简单稳定，Kriging考虑空间相关性，混合自动选择"
            )

        st.subheader("🧠 模型配置")

        if use_traditional:
            # 传统方法不需要复杂配置
            st.info("传统方法无需配置神经网络参数")
            # 设置默认值（不会用到，但避免变量未定义）
            use_auto_config = False
            hidden_dim = 128
            gnn_layers = 3
            dropout = 0.2
            conv_type = 'gatv2'
            epochs = 1
            learning_rate = 0.001
            patience = 10
            use_kfold = False
            n_splits = 3
        else:
            # GNN方法需要配置
            # 添加自动配置选项
            use_auto_config = st.checkbox(
                "🤖 自动优化配置（推荐）",
                value=True,
                help="根据数据规模自动选择最优超参数，特别适合小样本数据"
            )

            if use_auto_config:
                st.info("✅ 将根据数据规模自动优化所有参数")
                # 用于显示，实际值在训练时计算
                hidden_dim = 128
                gnn_layers = 3
                dropout = 0.2
                conv_type = 'gatv2'
                epochs = 200
                learning_rate = 0.001
                patience = 30
            else:
                hidden_dim = st.selectbox("隐藏层维度", [64, 96, 128, 160, 256], index=2, help="更大的维度可提高表达能力")
                gnn_layers = st.slider("GNN层数", 2, 4, 3, help="更深的网络可捕获更远距离的空间关系")
                conv_type = st.selectbox("卷积类型", ['gatv2', 'transformer', 'sage'], help="GATv2通常效果最好")
                dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05, help="防止过拟合")

                st.subheader("🎯 训练配置")
                epochs = st.slider("训练轮数", 100, 500, 200, help="更多轮数通常效果更好")
                learning_rate = st.select_slider("学习率",
                                                  options=[0.0001, 0.0005, 0.001, 0.002, 0.005],
                                                  value=0.001, help="较小的学习率更稳定")
                patience = st.slider("早停耐心值", 15, 50, 30, help="更大的耐心值避免过早停止")

            # K-fold选项仅在GNN模式下显示
            use_kfold = st.checkbox(
                "使用K-fold交叉验证",
                value=False,
                help="交叉验证可以更准确评估模型性能"
            )
            if use_kfold:
                n_splits = st.slider("折数", 3, 5, 3)
            else:
                n_splits = 3

        st.subheader("🗺️ 建模配置")
        resolution = st.slider("网格分辨率", 20, 100, 50)
        base_level = st.number_input("基准面高程(m)", value=0.0)
        gap_value = st.number_input("层间间隙(m)", value=0.0, min_value=0.0)

    # 初始化session state
    if 'data_result' not in st.session_state:
        st.session_state.data_result = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'block_models' not in st.session_state:
        st.session_state.block_models = None
    # 传统方法相关
    if 'traditional_predictor' not in st.session_state:
        st.session_state.traditional_predictor = None
    if 'use_traditional_method' not in st.session_state:
        st.session_state.use_traditional_method = False
    if 'traditional_metrics' not in st.session_state:
        st.session_state.traditional_metrics = None

    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 数据加载", "🚀 模型训练", "🗺️ 三维建模", "📈 结果分析"
    ])

    # ==================== Tab 1: 数据加载 ====================
    with tab1:
        st.header("数据加载与预处理")

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("🔄 加载钻孔数据", type="primary"):
                with st.spinner("正在加载数据..."):
                    try:
                        processor = ThicknessDataProcessor(
                            merge_coal=merge_coal,
                            k_neighbors=k_neighbors,
                            graph_type='knn'
                        )
                        result = processor.process_directory(
                            data_dir,
                            layer_method=layer_method,
                            min_occurrence_rate=min_occurrence_rate
                        )
                        st.session_state.data_result = result
                        st.success(f"✅ 数据加载成功!")
                    except Exception as e:
                        st.error(f"❌ 加载失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        with col2:
            if st.session_state.data_result is not None:
                result = st.session_state.data_result

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("钻孔数", len(result['borehole_ids']))
                col_b.metric("地层数", result['num_layers'])
                col_c.metric("特征维度", result['num_features'])

        if st.session_state.data_result is not None:
            result = st.session_state.data_result

            # 层序显示
            st.subheader("地层序列（从底到顶）")
            layer_order = result['layer_order']
            color_map = get_color_map(layer_order)

            cols = st.columns(min(len(layer_order), 6))
            for i, layer in enumerate(layer_order):
                with cols[i % len(cols)]:
                    color = color_map[layer]
                    st.markdown(f'''
                    <div style="background-color:{color}; padding:10px; border-radius:5px;
                                text-align:center; color:white; margin:5px 0;">
                        <b>{i+1}. {layer}</b><br>
                        存在率: {result['exist_rate'][i]*100:.0f}%
                    </div>
                    ''', unsafe_allow_html=True)

            # 数据预览
            st.subheader("原始数据预览")
            df = result['raw_df']
            st.dataframe(
                df[['borehole_id', 'x', 'y', 'lithology', 'thickness']].head(20),
                width="stretch"
            )

            # 钻孔分布图
            st.subheader("钻孔平面分布")
            coords = result['borehole_coords']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode='markers+text',
                marker=dict(size=12, color='#3C5488'),
                text=result['borehole_ids'],
                textposition='top center',
                name='钻孔位置'
            ))
            fig.update_layout(
                xaxis_title='X坐标',
                yaxis_title='Y坐标',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")

    # ==================== Tab 2: 模型训练 ====================
    with tab2:
        # 根据选择的方法显示不同标题
        if use_traditional:
            st.header("传统地质统计学厚度预测")
        else:
            st.header("GNN厚度预测模型训练")

        if st.session_state.data_result is None:
            st.warning("⚠️ 请先在【数据加载】页面加载数据")
            st.stop()

        result = st.session_state.data_result

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("模型信息")
            st.write(f"**任务类型:** 厚度回归")
            st.write(f"**输入特征:** {result['num_features']}")
            st.write(f"**输出层数:** {result['num_layers']}")
            st.write(f"**钻孔数量:** {result['data'].x.shape[0]}")

            if use_traditional:
                # 传统方法说明
                st.info(f"""
                **传统方法优势:**
                - 适合小样本（<50钻孔）
                - 逐层独立建模
                - 自动选择最佳插值法
                """)
            else:
                # K-fold选项（仅GNN）
                use_kfold = st.checkbox(
                    "使用K-fold交叉验证",
                    value=False,
                    help="对于小样本数据，K-fold交叉验证可以更准确地评估模型性能"
                )

                if use_kfold:
                    n_splits = st.selectbox("Fold数量", [3, 5, 10], index=1)
                    st.info(f"将进行{n_splits}-fold交叉验证，评估更可靠但耗时更长")

            if st.button("🚀 开始训练", type="primary"):
                with st.spinner("正在训练模型..." if not use_traditional else "正在拟合模型..."):
                    try:
                        n_samples = result['data'].x.shape[0]
                        n_layers_out = result['num_layers']
                        n_features = result['num_features']

                        if use_traditional:
                            # ===== 传统方法训练 =====
                            st.write("### 使用传统地质统计学方法...")

                            # 获取原始数据
                            raw_df = result['raw_df']
                            layer_order = result['layer_order']

                            # 选择预测器类型
                            if interp_method == 'hybrid':
                                predictor = HybridThicknessPredictor(
                                    layer_order=layer_order,
                                    kriging_threshold=10,
                                    smooth_factor=0.3,
                                    min_thickness=0.5
                                )
                            else:
                                predictor = PerLayerThicknessPredictor(
                                    layer_order=layer_order,
                                    default_method=interp_method,
                                    idw_power=2.0,
                                    n_neighbors=8,
                                    min_thickness=0.5
                                )

                            # 拟合模型 - 使用 layer_name 列（包含位置标记）
                            predictor.fit(
                                raw_df,
                                x_col='x',
                                y_col='y',
                                layer_col='layer_name',  # 使用位置标记的层名
                                thickness_col='thickness'
                            )

                            # 保存到session state
                            st.session_state.traditional_predictor = predictor
                            st.session_state.model = None  # 清除GNN模型
                            st.session_state.use_traditional_method = True
                            st.session_state.history = None

                            # 计算评估指标（在训练数据上）
                            coords = result['borehole_coords']
                            x_range = (coords[:, 0].min(), coords[:, 0].max())
                            y_range = (coords[:, 1].min(), coords[:, 1].max())
                            grid_x = np.linspace(x_range[0], x_range[1], 30)
                            grid_y = np.linspace(y_range[0], y_range[1], 30)

                            eval_metrics = evaluate_predictor(
                                predictor, raw_df, grid_x, grid_y,
                                x_col='x', y_col='y',
                                layer_col='layer_name',  # 使用位置标记的层名
                                thickness_col='thickness'
                            )
                            st.session_state.traditional_metrics = eval_metrics

                            st.success("✅ 传统方法拟合完成!")

                        else:
                            # ===== GNN方法训练 =====
                            # 获取优化配置（如果启用自动配置）
                            if use_auto_config:
                                opt_config = get_optimized_config_for_small_dataset(
                                    n_samples=n_samples,
                                    n_layers=n_layers_out,
                                    n_features=n_features
                                )

                                st.info("📊 自动配置分析:")
                                st.json({
                                    "样本数": n_samples,
                                    "层数": n_layers_out,
                                    "推荐模型": opt_config['model'],
                                    "推荐训练": opt_config['training'],
                                    "K-fold建议": opt_config['kfold']['reason'] if 'kfold' in opt_config else "N/A"
                                })

                                # 使用优化的配置
                                hidden_dim = opt_config['model']['hidden_channels']
                                gnn_layers = opt_config['model']['num_layers']
                                dropout = opt_config['model']['dropout']
                                learning_rate = opt_config['trainer']['learning_rate']
                                epochs = opt_config['training']['epochs']
                                patience = opt_config['training']['patience']
                                use_augmentation = opt_config['trainer']['use_augmentation']
                                warmup_epochs = opt_config['training']['warmup_epochs']

                                # 如果不建议使用K-fold但用户选了，给出警告
                                if use_kfold and not opt_config['kfold']['use_kfold']:
                                    st.warning(f"⚠️ {opt_config['kfold']['reason']}")
                            else:
                                use_augmentation = False
                                warmup_epochs = 0

                            st.session_state.use_traditional_method = False

                            if use_kfold:
                                # K-fold交叉验证
                                st.write("### K-fold交叉验证训练中...")

                                # 创建模型类和参数
                                heads = opt_config['model'].get('heads', 4) if use_auto_config else 4
                                model_kwargs = {
                                    'in_channels': n_features,
                                    'hidden_channels': hidden_dim,
                                    'num_layers': gnn_layers,
                                    'num_output_layers': n_layers_out,
                                    'dropout': dropout,
                                    'conv_type': conv_type,
                                    'heads': heads
                                }

                                trainer_kwargs = {
                                    'learning_rate': learning_rate,
                                    'use_augmentation': use_augmentation,
                                    'scheduler_type': 'plateau'
                                }

                                # 执行K-fold交叉验证
                                cv_results = k_fold_cross_validation(
                                    model_class=GNNThicknessPredictor,
                                    model_kwargs=model_kwargs,
                                    data=result['data'],
                                    n_splits=n_splits,
                                    epochs=epochs,
                                    patience=patience,
                                    trainer_kwargs=trainer_kwargs,
                                    verbose=True
                                )

                                st.session_state.cv_results = cv_results
                                st.session_state.use_kfold = True

                                # 使用最佳fold的模型
                                best_fold_idx = np.argmin([r['test_metrics']['mae'] for r in cv_results['fold_results']])
                                st.info(f"✅ 交叉验证完成! 最佳fold: {best_fold_idx + 1}")

                            else:
                                # 普通训练
                                # 创建模型和训练器
                                heads = opt_config['model'].get('heads', 4) if use_auto_config else 4
                                model, trainer = create_trainer(
                                    num_features=n_features,
                                    num_layers=n_layers_out,
                                    hidden_channels=hidden_dim,
                                    gnn_layers=gnn_layers,
                                    dropout=dropout,
                                    conv_type=conv_type,
                                    learning_rate=learning_rate,
                                    use_augmentation=use_augmentation,
                                    scheduler_type='plateau',
                                    heads=heads
                                )

                                # 训练
                                history = trainer.train(
                                    data=result['data'],
                                    epochs=epochs,
                                    patience=patience,
                                    warmup_epochs=warmup_epochs,
                                    verbose=False
                                )

                                st.session_state.model = model
                                st.session_state.trainer = trainer
                                st.session_state.history = history
                                st.session_state.use_kfold = False

                            st.success("✅ 训练完成!")

                    except Exception as e:
                        st.error(f"❌ 训练失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        with col2:
            # 传统方法结果显示
            if st.session_state.get('use_traditional_method', False) and st.session_state.get('traditional_predictor') is not None:
                predictor = st.session_state.traditional_predictor
                metrics = st.session_state.get('traditional_metrics') or {}

                st.subheader("传统方法拟合结果")

                # 显示评估指标
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("MAE", f"{metrics.get('mae', 0):.3f} m")
                col_b.metric("RMSE", f"{metrics.get('rmse', 0):.3f} m")
                col_c.metric("R²", f"{metrics.get('r2', 0):.3f}")
                col_d.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")

                # 显示每层的统计信息
                st.subheader("各层拟合详情")
                if hasattr(predictor, 'get_layer_summary'):
                    summary_df = predictor.get_layer_summary()
                    st.dataframe(summary_df, width="stretch")
                elif hasattr(predictor, 'layer_stats'):
                    stats_data = []
                    for layer_name in predictor.layer_order:
                        pred_info = predictor.predictors.get(layer_name, {})
                        stats = pred_info.get('stats', {})
                        stats_data.append({
                            '地层': layer_name,
                            '方法': pred_info.get('type', 'N/A'),
                            '数据点': stats.get('n_points', 0),
                            '均值(m)': f"{stats.get('mean', 0):.2f}",
                            '中位数(m)': f"{stats.get('median', 0):.2f}",
                            '标准差(m)': f"{stats.get('std', 0):.2f}"
                        })
                    if stats_data:
                        st.dataframe(pd.DataFrame(stats_data), width="stretch")

                # 方法分布图
                st.subheader("各层插值方法分布")
                method_counts = {}
                if hasattr(predictor, 'layer_data'):
                    for layer_name, data in predictor.layer_data.items():
                        method = data.get('method', 'unknown')
                        method_counts[method] = method_counts.get(method, 0) + 1
                elif hasattr(predictor, 'predictors'):
                    for layer_name, pred_info in predictor.predictors.items():
                        method = pred_info.get('type', 'unknown')
                        method_counts[method] = method_counts.get(method, 0) + 1

                if method_counts:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(method_counts.keys()),
                        values=list(method_counts.values()),
                        hole=0.4,
                        marker_colors=['#E64B35', '#4DBBD5', '#00A087', '#3C5488']
                    )])
                    fig.update_layout(
                        title="各层使用的插值方法",
                        height=350
                    )
                    st.plotly_chart(fig, width="stretch")

                st.success("✅ 传统方法无需迭代训练，可直接进行三维建模!")

            # K-fold结果显示
            elif st.session_state.get('use_kfold', False) and st.session_state.get('cv_results') is not None:
                cv_results = st.session_state.cv_results

                st.subheader("K-Fold交叉验证结果")

                # 汇总指标
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("平均MAE", f"{cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f} m")
                col_b.metric("平均RMSE", f"{cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f} m")
                col_c.metric("平均R²", f"{cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")

                # 每个fold的结果
                st.subheader("各Fold结果")
                fold_df = pd.DataFrame([
                    {
                        'Fold': r['fold'],
                        'MAE (m)': f"{r['test_metrics']['mae']:.3f}",
                        'RMSE (m)': f"{r['test_metrics']['rmse']:.3f}",
                        'R²': f"{r['test_metrics']['r2']:.3f}"
                    }
                    for r in cv_results['fold_results']
                ])
                st.dataframe(fold_df, width="stretch")

                # 可视化各fold的性能
                fig = go.Figure()
                folds = [r['fold'] for r in cv_results['fold_results']]
                maes = [r['test_metrics']['mae'] for r in cv_results['fold_results']]
                rmses = [r['test_metrics']['rmse'] for r in cv_results['fold_results']]

                fig.add_trace(go.Bar(x=folds, y=maes, name='MAE', marker_color='#E64B35'))
                fig.add_trace(go.Bar(x=folds, y=rmses, name='RMSE', marker_color='#4DBBD5'))
                fig.update_layout(
                    xaxis_title='Fold',
                    yaxis_title='误差 (m)',
                    height=400,
                    barmode='group'
                )
                st.plotly_chart(fig, width="stretch")

            # 普通训练结果显示
            elif st.session_state.history is not None:
                history = st.session_state.history

                # 训练曲线
                st.subheader("训练曲线")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['train_loss'], name='训练损失',
                    line=dict(color='#E64B35')
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_loss'], name='验证损失',
                    line=dict(color='#4DBBD5')
                ))
                fig.update_layout(
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

                # MAE曲线
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=history['train_mae'], name='训练MAE',
                    line=dict(color='#00A087')
                ))
                fig2.add_trace(go.Scatter(
                    y=history['val_mae'], name='验证MAE',
                    line=dict(color='#3C5488')
                ))
                fig2.update_layout(
                    xaxis_title='Epoch',
                    yaxis_title='MAE (m)',
                    height=400
                )
                st.plotly_chart(fig2, width="stretch")

                # 测试指标
                if 'test_metrics' in history:
                    metrics = history['test_metrics']
                    st.subheader("测试集评估")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("MAE", f"{metrics['mae']:.3f} m")
                    col_b.metric("RMSE", f"{metrics['rmse']:.3f} m")
                    col_c.metric("R²", f"{metrics['r2']:.3f}")

    # ==================== Tab 3: 三维建模 ====================
    with tab3:
        st.header("三维地质模型构建")

        # 检查是否有可用的预测模型（传统或GNN）
        has_traditional = st.session_state.get('use_traditional_method', False) and st.session_state.get('traditional_predictor') is not None
        has_gnn = st.session_state.model is not None

        if not has_traditional and not has_gnn:
            st.warning("⚠️ 请先在【模型训练】页面训练模型")
            st.stop()

        result = st.session_state.data_result

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("建模参数")
            st.write(f"**网格分辨率:** {resolution}×{resolution}")
            st.write(f"**基准面高程:** {base_level} m")
            st.write(f"**层间间隙:** {gap_value} m")

            if has_traditional:
                st.info("使用传统地质统计学方法预测厚度")
            else:
                st.info("使用GNN深度学习方法预测厚度")

            if st.button("🏗️ 构建三维模型", type="primary"):
                with st.spinner("正在构建模型..."):
                    try:
                        # 获取坐标范围
                        coords = result['borehole_coords']
                        x_range = (coords[:, 0].min(), coords[:, 0].max())
                        y_range = (coords[:, 1].min(), coords[:, 1].max())

                        # 创建网格
                        grid_x = np.linspace(x_range[0], x_range[1], resolution)
                        grid_y = np.linspace(y_range[0], y_range[1], resolution)

                        if has_traditional:
                            # ===== 使用传统方法预测厚度 =====
                            predictor = st.session_state.traditional_predictor
                            thickness_grids = predictor.predict_grid(grid_x, grid_y)
                            XI, YI = np.meshgrid(grid_x, grid_y)
                        else:
                            # ===== 使用GNN预测厚度 =====
                            model = st.session_state.model
                            device = next(model.parameters()).device
                            model.eval()
                            data = result['data'].to(device)

                            with torch.no_grad():
                                pred_thick, pred_exist = model(
                                    data.x, data.edge_index,
                                    data.edge_attr if hasattr(data, 'edge_attr') else None
                                )
                                pred_thick = pred_thick.cpu().numpy()
                                pred_exist = torch.sigmoid(pred_exist).cpu().numpy()

                            # 插值到网格
                            from scipy.interpolate import griddata
                            XI, YI = np.meshgrid(grid_x, grid_y)
                            xi_flat, yi_flat = XI.flatten(), YI.flatten()

                            thickness_grids = {}
                            for i, layer_name in enumerate(result['layer_order']):
                                layer_thick = pred_thick[:, i]
                                exist_mask = pred_exist[:, i] > 0.5
                                if exist_mask.sum() < 3:
                                    exist_mask = np.ones(len(layer_thick), dtype=bool)

                                x_valid = coords[exist_mask, 0]
                                y_valid = coords[exist_mask, 1]
                                z_valid = layer_thick[exist_mask]

                                grid_thick = griddata(
                                    (x_valid, y_valid), z_valid, (xi_flat, yi_flat),
                                    method='linear'
                                )
                                if np.any(np.isnan(grid_thick)):
                                    nearest = griddata(
                                        (x_valid, y_valid), z_valid, (xi_flat, yi_flat),
                                        method='nearest'
                                    )
                                    grid_thick = np.where(np.isnan(grid_thick), nearest, grid_thick)

                                grid_thick = np.clip(grid_thick, 0.5, None)
                                thickness_grids[layer_name] = grid_thick.reshape(XI.shape)

                        # 层序累加构建模型
                        builder = GeologicalModelBuilder(
                            layer_order=result['layer_order'],
                            resolution=resolution,
                            base_level=base_level,
                            gap_value=gap_value
                        )

                        block_models, XI, YI = builder.build_model(
                            thickness_grids=thickness_grids,
                            x_range=x_range,
                            y_range=y_range
                        )

                        st.session_state.block_models = block_models
                        st.session_state.grid_XI = XI
                        st.session_state.grid_YI = YI

                        st.success("✅ 三维模型构建完成!")

                    except Exception as e:
                        st.error(f"❌ 构建失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        with col2:
            if st.session_state.block_models is not None:
                block_models = st.session_state.block_models

                # 显示各层信息
                st.subheader("各层统计")
                layer_info = []
                for bm in block_models:
                    layer_info.append({
                        '地层': bm.name,
                        '平均厚度(m)': f"{bm.avg_thickness:.2f}",
                        '最大厚度(m)': f"{bm.max_thickness:.2f}",
                        '底面高程(m)': f"{bm.avg_bottom:.2f}",
                        '顶面高程(m)': f"{bm.avg_height:.2f}"
                    })
                st.dataframe(pd.DataFrame(layer_info), width="stretch")

                # 三维可视化
                st.subheader("三维模型可视化")

                XI = st.session_state.grid_XI
                YI = st.session_state.grid_YI
                color_map = get_color_map(result['layer_order'])

                # 可视化设置
                st.subheader("显示设置")
                col_opt1, col_opt2, col_opt3 = st.columns(3)

                with col_opt1:
                    show_layers = st.multiselect(
                        "选择显示的地层",
                        result['layer_order'],
                        default=result['layer_order']  # 默认显示所有岩层
                    )

                with col_opt2:
                    render_mode = st.selectbox(
                        "渲染模式",
                        ['增强材质', '基础渲染', '线框模式'],
                        index=0,
                        help="增强材质：带纹理和光照效果"
                    )

                with col_opt3:
                    show_sides = st.checkbox("显示侧面", value=True, help="显示地层侧面轮廓")
                    surface_opacity = st.slider("透明度", 0.3, 1.0, 0.9)

                # 高级性能选项（默认最高质量，可调整以适应不同硬件）
                with st.expander("⚙️ 高级性能选项", expanded=False):
                    col_perf1, col_perf2 = st.columns(2)
                    with col_perf1:
                        preview_quality = st.selectbox(
                            "预览质量",
                            ['高质量', '平衡', '高性能'],
                            index=0,
                            help="高质量：完整分辨率，最佳效果；高性能：降采样显示"
                        )
                    with col_perf2:
                        skip_bottom = st.checkbox("隐藏底面", value=False, help="隐藏底面可减少渲染量")

                # 根据预览质量调整分辨率（默认高质量，无降采样）
                if preview_quality == '高质量':
                    downsample = 1  # 完整分辨率
                elif preview_quality == '平衡':
                    downsample = 2
                else:  # 高性能
                    downsample = 4

                # 降采样网格
                if downsample > 1:
                    XI_display = XI[::downsample, ::downsample]
                    YI_display = YI[::downsample, ::downsample]
                else:
                    XI_display = XI
                    YI_display = YI

                # 获取完整颜色映射（包含高光和阴影）
                full_color_map = get_full_color_map(result['layer_order'])

                fig = go.Figure()

                for bm in block_models:
                    if bm.name not in show_layers:
                        continue

                    colors = full_color_map[bm.name]
                    base_color = colors['base']

                    # 降采样曲面数据
                    if downsample > 1:
                        top_display = bm.top_surface[::downsample, ::downsample]
                        bottom_display = bm.bottom_surface[::downsample, ::downsample]
                    else:
                        top_display = bm.top_surface
                        bottom_display = bm.bottom_surface

                    if render_mode == '增强材质':
                        # 生成纹理（使用降采样后的尺寸）
                        texture = generate_rock_texture(
                            XI_display.shape, rock_type=bm.name, intensity=0.12
                        )
                        surface_color = texture
                        colorscale = create_textured_colorscale(base_color)

                        # 顶面
                        fig.add_trace(go.Surface(
                            x=XI_display, y=YI_display, z=top_display,
                            surfacecolor=surface_color,
                            colorscale=colorscale,
                            showscale=False,
                            opacity=surface_opacity,
                            name=f"{bm.name}",
                            lighting=dict(
                                ambient=0.6,
                                diffuse=0.8,
                                specular=0.3,
                                roughness=0.8,
                                fresnel=0.2
                            ),
                            lightposition=dict(x=1000, y=1000, z=2000),
                            hovertemplate=f"<b>{bm.name}</b><br>Z: %{{z:.1f}}m<extra></extra>"
                        ))

                        # 底面（可选）
                        if not skip_bottom:
                            fig.add_trace(go.Surface(
                                x=XI_display, y=YI_display, z=bottom_display,
                                surfacecolor=surface_color * 0.8,
                                colorscale=colorscale,
                                showscale=False,
                                opacity=surface_opacity * 0.7,
                                name=f"{bm.name} (底)",
                                hoverinfo='skip'
                            ))

                    elif render_mode == '线框模式':
                        fig.add_trace(go.Surface(
                            x=XI_display, y=YI_display, z=top_display,
                            colorscale=[[0, base_color], [1, base_color]],
                            showscale=False,
                            opacity=0.3,
                            name=f"{bm.name}",
                            contours=dict(
                                x=dict(show=True, color='black', width=1),
                                y=dict(show=True, color='black', width=1),
                                z=dict(show=True, color='black', width=2)
                            )
                        ))

                    else:
                        # 基础渲染
                        fig.add_trace(go.Surface(
                            x=XI_display, y=YI_display, z=top_display,
                            colorscale=[[0, base_color], [1, base_color]],
                            showscale=False,
                            opacity=surface_opacity,
                            name=f"{bm.name}"
                        ))

                    # 添加侧面（优化：合并为单个trace）
                    if show_sides and render_mode != '线框模式':
                        shadow_color = colors.get('shadow', base_color)
                        n = XI_display.shape[0]

                        # 合并所有侧面线条为一个trace（使用None分隔）
                        x_all, y_all, z_all = [], [], []

                        # 前侧面 (y=0) - 简化为边界线
                        x_all.extend(XI_display[0, :].tolist() + [None])
                        y_all.extend(YI_display[0, :].tolist() + [None])
                        z_all.extend(top_display[0, :].tolist() + [None])
                        x_all.extend(XI_display[0, :].tolist() + [None])
                        y_all.extend(YI_display[0, :].tolist() + [None])
                        z_all.extend(bottom_display[0, :].tolist() + [None])

                        # 后侧面 (y=max)
                        x_all.extend(XI_display[-1, :].tolist() + [None])
                        y_all.extend(YI_display[-1, :].tolist() + [None])
                        z_all.extend(top_display[-1, :].tolist() + [None])
                        x_all.extend(XI_display[-1, :].tolist() + [None])
                        y_all.extend(YI_display[-1, :].tolist() + [None])
                        z_all.extend(bottom_display[-1, :].tolist() + [None])

                        # 左侧面 (x=0)
                        x_all.extend(XI_display[:, 0].tolist() + [None])
                        y_all.extend(YI_display[:, 0].tolist() + [None])
                        z_all.extend(top_display[:, 0].tolist() + [None])
                        x_all.extend(XI_display[:, 0].tolist() + [None])
                        y_all.extend(YI_display[:, 0].tolist() + [None])
                        z_all.extend(bottom_display[:, 0].tolist() + [None])

                        # 右侧面 (x=max)
                        x_all.extend(XI_display[:, -1].tolist() + [None])
                        y_all.extend(YI_display[:, -1].tolist() + [None])
                        z_all.extend(top_display[:, -1].tolist() + [None])
                        x_all.extend(XI_display[:, -1].tolist() + [None])
                        y_all.extend(YI_display[:, -1].tolist() + [None])
                        z_all.extend(bottom_display[:, -1].tolist() + [None])

                        # 添加垂直连接线（只在角落）
                        corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
                        for j, i in corners:
                            x_all.extend([XI_display[j, i], XI_display[j, i], None])
                            y_all.extend([YI_display[j, i], YI_display[j, i], None])
                            z_all.extend([bottom_display[j, i], top_display[j, i], None])

                        fig.add_trace(go.Scatter3d(
                            x=x_all, y=y_all, z=z_all,
                            mode='lines',
                            line=dict(color=shadow_color, width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                # 添加图例
                for layer_name in show_layers:
                    colors = full_color_map[layer_name]
                    fig.add_trace(go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode='markers',
                        marker=dict(size=10, color=colors['base']),
                        name=layer_name,
                        showlegend=True
                    ))

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            title='X (m)',
                            backgroundcolor='rgb(240, 240, 240)',
                            gridcolor='white',
                            showbackground=True,
                            zerolinecolor='gray'
                        ),
                        yaxis=dict(
                            title='Y (m)',
                            backgroundcolor='rgb(240, 240, 240)',
                            gridcolor='white',
                            showbackground=True,
                            zerolinecolor='gray'
                        ),
                        zaxis=dict(
                            title='Z 高程 (m)',
                            backgroundcolor='rgb(230, 230, 230)',
                            gridcolor='white',
                            showbackground=True,
                            zerolinecolor='gray'
                        ),
                        aspectmode='data',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.0),
                            up=dict(x=0, y=0, z=1)
                        )
                    ),
                    height=750,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='black',
                        borderwidth=1
                    )
                )

                # 使用WebGL配置提升性能
                config = {
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'responsive': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                }
                st.plotly_chart(fig, width="stretch", config=config)

                # 显示颜色图例说明
                st.markdown("---")
                st.subheader("岩层颜色图例")
                legend_cols = st.columns(min(len(show_layers), 4))
                for i, layer_name in enumerate(show_layers):
                    with legend_cols[i % len(legend_cols)]:
                        colors = full_color_map[layer_name]
                        st.markdown(f'''
                        <div style="display:flex; align-items:center; margin:5px 0;">
                            <div style="width:30px; height:20px; background:{colors['base']};
                                        border:1px solid black; margin-right:8px;"></div>
                            <span><b>{layer_name}</b></span>
                        </div>
                        ''', unsafe_allow_html=True)

                # ==================== PyVista 高级渲染 ====================
                st.markdown("---")
                st.subheader("PyVista 高级渲染")

                if PYVISTA_AVAILABLE:
                    st.info("PyVista 已启用，支持 PBR 材质、纹理贴图和多种导出格式")

                    pv_col1, pv_col2 = st.columns(2)

                    with pv_col1:
                        pv_use_textures = st.checkbox("启用程序化纹理", value=True,
                                                       help="为不同岩石类型生成专业纹理")
                        pv_use_pbr = st.checkbox("启用 PBR 渲染", value=True,
                                                  help="物理渲染，更真实的材质效果")
                        pv_add_sides = st.checkbox("渲染侧面", value=True,
                                                    help="显示地层的侧面轮廓")

                    with pv_col2:
                        pv_export_format = st.multiselect(
                            "导出格式",
                            ['PNG (高清截图)', 'HTML (交互式)', 'OBJ (3D模型)', 'STL (3D打印)'],
                            default=['PNG (高清截图)', 'HTML (交互式)']
                        )
                        pv_opacity = st.slider("PyVista 透明度", 0.5, 1.0, 0.95)

                    if st.button("🎨 PyVista 渲染导出", type="primary"):
                        with st.spinner("正在使用 PyVista 渲染..."):
                            try:
                                # 创建输出目录
                                output_dir = os.path.join(project_root, 'output', 'pyvista')
                                os.makedirs(output_dir, exist_ok=True)

                                # 创建渲染器
                                renderer = GeologicalModelRenderer(
                                    background='white',
                                    window_size=(1920, 1080),
                                    use_pbr=pv_use_pbr
                                )

                                # 渲染模型
                                plotter = renderer.render_model(
                                    block_models,
                                    XI, YI,
                                    show_layers=show_layers,
                                    add_sides=pv_add_sides,
                                    use_textures=pv_use_textures,
                                    opacity=pv_opacity,
                                    show_edges=False,
                                    lighting='three_lights'
                                )

                                # 添加钻孔标记
                                if 'borehole_coords' in result and 'borehole_ids' in result:
                                    renderer.add_borehole_markers(
                                        result['borehole_coords'],
                                        result['borehole_ids']
                                    )

                                exported_files = []

                                # 导出文件
                                if 'PNG (高清截图)' in pv_export_format:
                                    png_path = os.path.join(output_dir, 'geological_model.png')
                                    renderer.export_screenshot(png_path, scale=2)
                                    exported_files.append(png_path)

                                if 'HTML (交互式)' in pv_export_format:
                                    html_path = os.path.join(output_dir, 'geological_model.html')
                                    renderer.export_html(html_path)
                                    exported_files.append(html_path)

                                if 'OBJ (3D模型)' in pv_export_format:
                                    obj_path = os.path.join(output_dir, 'geological_model.obj')
                                    renderer.export_mesh(obj_path, file_format='obj')
                                    exported_files.append(obj_path)

                                if 'STL (3D打印)' in pv_export_format:
                                    stl_path = os.path.join(output_dir, 'geological_model.stl')
                                    renderer.export_mesh(stl_path, file_format='stl')
                                    exported_files.append(stl_path)

                                st.success(f"✅ PyVista 渲染完成!")

                                # 显示导出文件列表
                                st.write("**导出文件：**")
                                for f in exported_files:
                                    st.write(f"- `{f}`")

                                # 显示 PNG 预览
                                png_path = os.path.join(output_dir, 'geological_model.png')
                                if os.path.exists(png_path):
                                    st.image(png_path, caption="PyVista 渲染结果", width="stretch")

                                # 提供 HTML 下载链接
                                html_path = os.path.join(output_dir, 'geological_model.html')
                                if os.path.exists(html_path):
                                    with open(html_path, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    st.download_button(
                                        label="📥 下载交互式 HTML",
                                        data=html_content,
                                        file_name="geological_model.html",
                                        mime="text/html"
                                    )

                            except Exception as e:
                                st.error(f"❌ PyVista 渲染失败: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

                    # 提供独立脚本说明
                    with st.expander("使用独立 PyVista 脚本（交互式窗口）"):
                        st.markdown("""
                        如果需要交互式 3D 窗口，可以运行以下脚本：

                        ```python
                        # render_pyvista.py
                        from src.pyvista_renderer import render_geological_model
                        import numpy as np

                        # 加载你的模型数据
                        # block_models, XI, YI = ...

                        # 渲染（会打开交互窗口）
                        renderer = render_geological_model(
                            block_models, XI, YI,
                            show_interactive=True,
                            use_textures=True,
                            add_sides=True,
                            export_formats=['png', 'html', 'obj']
                        )
                        ```

                        **交互操作：**
                        - 左键拖动：旋转
                        - 右键拖动：平移
                        - 滚轮：缩放
                        - `r` 键：重置视角
                        - `q` 键：退出
                        """)

                else:
                    st.warning("⚠️ PyVista 未安装。请运行 `pip install pyvista` 安装后重启应用。")
                    st.code("pip install pyvista pyvistaqt", language="bash")

    # ==================== Tab 4: 结果分析 ====================
    with tab4:
        st.header("结果分析与导出")

        if st.session_state.block_models is None:
            st.warning("⚠️ 请先在【三维建模】页面构建模型")
            st.stop()

        block_models = st.session_state.block_models
        result = st.session_state.data_result

        # 厚度分布图
        st.subheader("各层厚度分布")

        fig = go.Figure()
        for bm in block_models:
            thickness_flat = bm.thickness_grid.flatten()
            fig.add_trace(go.Box(
                y=thickness_flat,
                name=bm.name,
                boxpoints='outliers'
            ))

        fig.update_layout(
            yaxis_title='厚度 (m)',
            height=500
        )
        st.plotly_chart(fig, width="stretch")

        # 导出选项
        st.subheader("模型导出")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 保存模型信息"):
                import json
                output_dir = os.path.join(project_root, 'output')
                os.makedirs(output_dir, exist_ok=True)

                model_info = {
                    'layer_order': result['layer_order'],
                    'resolution': resolution,
                    'layers': []
                }
                for bm in block_models:
                    model_info['layers'].append({
                        'name': bm.name,
                        'avg_thickness': float(bm.avg_thickness),
                        'max_thickness': float(bm.max_thickness),
                        'avg_bottom': float(bm.avg_bottom),
                        'avg_top': float(bm.avg_height)
                    })

                with open(os.path.join(output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=2)

                st.success(f"✅ 已保存到 output/model_info.json")

        with col2:
            if st.button("💾 保存网格数据"):
                output_dir = os.path.join(project_root, 'output')
                os.makedirs(output_dir, exist_ok=True)

                for bm in block_models:
                    np.savez(
                        os.path.join(output_dir, f'layer_{bm.name}.npz'),
                        top_surface=bm.top_surface,
                        bottom_surface=bm.bottom_surface,
                        thickness_grid=bm.thickness_grid
                    )

                st.success(f"✅ 已保存到 output/layer_*.npz")


if __name__ == "__main__":
    main()
