"""
PyVista 独立渲染脚本

使用方法：
1. 先运行 Streamlit 应用完成模型构建
2. 运行此脚本进行高质量渲染和导出

python render_pyvista.py
"""

import numpy as np
import os
import sys
import pickle
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pyvista_renderer import (
    GeologicalModelRenderer, RockMaterial, TextureGenerator,
    render_geological_model, create_model_comparison
)


def load_model_from_npz(output_dir='output'):
    """
    从 npz 文件加载模型数据

    Returns:
        block_models: BlockModel 列表
        XI, YI: 坐标网格
    """
    from src.gnn_thickness_modeling import BlockModel

    # 加载模型信息
    import json
    info_path = os.path.join(output_dir, 'model_info.json')

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"找不到模型信息文件: {info_path}\n请先在 Streamlit 应用中构建并保存模型。")

    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)

    layer_order = model_info['layer_order']
    resolution = model_info['resolution']

    block_models = []

    for layer_name in layer_order:
        npz_path = os.path.join(output_dir, f'layer_{layer_name}.npz')

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            bm = type('BlockModel', (), {
                'name': layer_name,
                'top_surface': data['top_surface'],
                'bottom_surface': data['bottom_surface'],
                'thickness_grid': data['thickness_grid'],
                'avg_thickness': float(np.nanmean(data['thickness_grid'])),
                'max_thickness': float(np.nanmax(data['thickness_grid'])),
                'avg_height': float(np.nanmean(data['top_surface'])),
                'avg_bottom': float(np.nanmean(data['bottom_surface']))
            })()
            block_models.append(bm)
        else:
            print(f"警告: 找不到 {layer_name} 的数据文件")

    if not block_models:
        raise ValueError("没有加载到任何地层数据")

    # 生成网格坐标
    ny, nx = block_models[0].top_surface.shape
    # 假设坐标范围（如果没有保存的话）
    x = np.linspace(0, 1000, nx)
    y = np.linspace(0, 1000, ny)
    XI, YI = np.meshgrid(x, y)

    return block_models, XI, YI


def demo_render():
    """
    演示渲染 - 使用模拟数据
    """
    print("=" * 60)
    print("PyVista 地质模型渲染演示")
    print("=" * 60)

    # 创建模拟数据
    resolution = 50
    x = np.linspace(0, 1000, resolution)
    y = np.linspace(0, 1000, resolution)
    XI, YI = np.meshgrid(x, y)

    np.random.seed(42)

    # 定义地层
    layer_configs = [
        ('表土', 0, lambda x, y: 5 + np.sin(x/300) * 2),
        ('砂岩', 6, lambda x, y: 12 + np.cos(y/200) * 4 + np.random.randn(*x.shape) * 0.5),
        ('泥岩', 20, lambda x, y: 8 + np.sin((x+y)/400) * 3),
        ('煤层', 30, lambda x, y: 4 + np.random.rand(*x.shape) * 2),
        ('砂岩', 36, lambda x, y: 18 + np.cos(x/250) * 5),
        ('页岩', 56, lambda x, y: 10 + np.sin(y/300) * 4),
    ]

    block_models = []
    current_base = 0

    for name, _, thickness_func in layer_configs:
        thickness = thickness_func(XI, YI)
        thickness = np.clip(thickness, 1, None)

        bm = type('BlockModel', (), {
            'name': name,
            'bottom_surface': np.full(XI.shape, current_base),
            'top_surface': np.full(XI.shape, current_base) + thickness,
            'thickness_grid': thickness,
            'avg_thickness': float(np.mean(thickness)),
            'max_thickness': float(np.max(thickness)),
            'avg_height': float(current_base + np.mean(thickness)),
            'avg_bottom': float(current_base)
        })()
        block_models.append(bm)
        current_base = current_base + np.mean(thickness) + 1

    print(f"\n创建了 {len(block_models)} 个地层:")
    for bm in block_models:
        print(f"  - {bm.name}: 厚度={bm.avg_thickness:.1f}m, "
              f"底面={bm.avg_bottom:.1f}m, 顶面={bm.avg_height:.1f}m")

    # 渲染
    print("\n正在渲染...")
    output_dir = 'output/pyvista_demo'

    renderer = render_geological_model(
        block_models, XI, YI,
        output_dir=output_dir,
        show_interactive=True,  # 打开交互窗口
        export_formats=['png', 'html'],
        use_textures=True,
        add_sides=True,
        opacity=0.95
    )

    print(f"\n渲染完成! 文件保存在: {output_dir}/")


def render_from_saved_model():
    """
    从保存的模型文件渲染
    """
    print("=" * 60)
    print("PyVista 渲染 - 从保存的模型")
    print("=" * 60)

    try:
        block_models, XI, YI = load_model_from_npz('output')

        print(f"\n加载了 {len(block_models)} 个地层:")
        for bm in block_models:
            print(f"  - {bm.name}: 厚度={bm.avg_thickness:.1f}m")

        # 渲染
        print("\n正在渲染...")
        output_dir = 'output/pyvista'

        renderer = render_geological_model(
            block_models, XI, YI,
            output_dir=output_dir,
            show_interactive=True,
            export_formats=['png', 'html', 'obj'],
            use_textures=True,
            add_sides=True,
            opacity=0.95
        )

        print(f"\n渲染完成! 文件保存在: {output_dir}/")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示: 请先运行 Streamlit 应用构建模型，并点击【保存网格数据】按钮。")
        print("      或者运行演示模式: python render_pyvista.py --demo")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PyVista 地质模型渲染')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')
    parser.add_argument('--output', default='output', help='输出目录')
    args = parser.parse_args()

    if args.demo:
        demo_render()
    else:
        render_from_saved_model()


if __name__ == "__main__":
    main()
