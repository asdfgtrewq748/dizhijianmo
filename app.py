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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入新版模块
from src.thickness_data_loader import ThicknessDataProcessor, LayerTableProcessor
from src.gnn_thickness_modeling import (
    GNNThicknessPredictor, GeologicalModelBuilder,
    GNNGeologicalModeling, TraditionalThicknessInterpolator
)
from src.thickness_trainer import create_trainer, ThicknessTrainer, ThicknessEvaluator

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

# 配色方案
GEOLOGY_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
]

def get_color_map(layer_order):
    """获取岩层颜色映射"""
    colors = GEOLOGY_COLORS * (len(layer_order) // len(GEOLOGY_COLORS) + 1)
    return {name: colors[i] for i, name in enumerate(layer_order)}


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
        k_neighbors = st.slider("K邻居数", 4, 20, 10, help="增加邻居数可提高空间关联性")

        st.subheader("🧠 模型配置")
        hidden_dim = st.selectbox("隐藏层维度", [128, 256, 512], index=1, help="更大的维度可提高表达能力")
        gnn_layers = st.slider("GNN层数", 2, 6, 4, help="更深的网络可捕获更远距离的空间关系")
        conv_type = st.selectbox("卷积类型", ['gatv2', 'transformer', 'sage'], help="GATv2通常效果最好")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, help="较小的dropout避免欠拟合")

        st.subheader("🎯 训练配置")
        epochs = st.slider("训练轮数", 100, 1000, 500, help="更多轮数通常效果更好")
        learning_rate = st.select_slider("学习率",
                                          options=[0.0001, 0.0005, 0.001, 0.002],
                                          value=0.0005, help="较小的学习率更稳定")
        patience = st.slider("早停耐心值", 20, 100, 50, help="更大的耐心值避免过早停止")

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
                        result = processor.process_directory(data_dir)
                        st.session_state.data_result = result
                        st.success(f"✅ 数据加载成功!")
                    except Exception as e:
                        st.error(f"❌ 加载失败: {str(e)}")

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
                use_container_width=True
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
            st.plotly_chart(fig, use_container_width=True)

    # ==================== Tab 2: 模型训练 ====================
    with tab2:
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
            st.write(f"**卷积类型:** {conv_type.upper()}")

            if st.button("🚀 开始训练", type="primary"):
                with st.spinner("正在训练模型..."):
                    try:
                        # 创建模型和训练器
                        model, trainer = create_trainer(
                            num_features=result['num_features'],
                            num_layers=result['num_layers'],
                            hidden_channels=hidden_dim,
                            gnn_layers=gnn_layers,
                            dropout=dropout,
                            conv_type=conv_type,
                            learning_rate=learning_rate
                        )

                        # 训练
                        history = trainer.train(
                            data=result['data'],
                            epochs=epochs,
                            patience=patience,
                            verbose=False
                        )

                        st.session_state.model = model
                        st.session_state.trainer = trainer
                        st.session_state.history = history

                        st.success("✅ 训练完成!")

                    except Exception as e:
                        st.error(f"❌ 训练失败: {str(e)}")

        with col2:
            if st.session_state.history is not None:
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
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig2, use_container_width=True)

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

        if st.session_state.model is None:
            st.warning("⚠️ 请先在【模型训练】页面训练模型")
            st.stop()

        result = st.session_state.data_result
        model = st.session_state.model

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("建模参数")
            st.write(f"**网格分辨率:** {resolution}×{resolution}")
            st.write(f"**基准面高程:** {base_level} m")
            st.write(f"**层间间隙:** {gap_value} m")

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

                        # GNN预测厚度
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
                st.dataframe(pd.DataFrame(layer_info), use_container_width=True)

                # 三维可视化
                st.subheader("三维模型可视化")

                XI = st.session_state.grid_XI
                YI = st.session_state.grid_YI
                color_map = get_color_map(result['layer_order'])

                # 选择显示的层
                show_layers = st.multiselect(
                    "选择显示的地层",
                    result['layer_order'],
                    default=result['layer_order']
                )

                fig = go.Figure()
                for bm in block_models:
                    if bm.name not in show_layers:
                        continue

                    color = color_map[bm.name]

                    # 顶面
                    fig.add_trace(go.Surface(
                        x=XI, y=YI, z=bm.top_surface,
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        opacity=0.8,
                        name=f"{bm.name} (顶)"
                    ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title='X (m)',
                        yaxis_title='Y (m)',
                        zaxis_title='Z (m)',
                        aspectmode='data'
                    ),
                    height=700,
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
