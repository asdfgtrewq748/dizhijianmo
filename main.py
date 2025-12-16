"""
主入口脚本
提供命令行接口：训练、建模、可视化

针对敏东矿区钻孔数据格式优化

建模方法：
- traditional: 传统层面插值建模
- gnn: GNN直接预测建模（分类）
- layer: 层序累加建模（推荐，GNN预测厚度回归）
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.data_loader import BoreholeDataProcessor, GridInterpolator
from src.trainer import GeoModelTrainer, compute_class_weights
from src.modeling import GeoModel3D, build_geological_model
from src.gnn_modeling import DirectPredictionModeling, build_gnn_geological_model
from src.layer_modeling import (
    LayerDataProcessor, GNNThicknessPredictor, ThicknessTrainer,
    LayerBasedGeologicalModeling, build_layer_based_model
)


def run_full_pipeline(
    data_dir: str = None,
    model_type: str = 'graphsage',    # 使用更稳定的GraphSAGE
    hidden_dim: int = 128,            # 适中的隐藏层大小
    num_layers: int = 4,              # 适中的深度
    epochs: int = 500,
    sample_interval: float = 1.0,     # 稳定采样间隔
    k_neighbors: int = 12,            # 适中的k值，避免过平滑
    grid_resolution: tuple = (50, 50, 50),
    output_dir: str = 'output',
    modeling_method: str = 'layer',   # 'traditional', 'gnn', 'layer', 'all'
    merge_coal: bool = True
):
    """
    完整流程: 数据加载 → 训练 → 建模 → 导出
    """
    print("=" * 70)
    print("GNN三维地质建模系统 - 完整流程")
    print("=" * 70)

    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    # ==================== 1. 数据加载 ====================
    print("\n" + "=" * 70)
    print("阶段 1/4: 数据加载与预处理")
    print("=" * 70)

    processor = BoreholeDataProcessor(
        k_neighbors=k_neighbors,
        graph_type='knn',
        normalize_coords=True,
        normalize_features=True,
        sample_interval=sample_interval
    )

    df = processor.load_all_boreholes(
        data_dir=data_dir,
        surface_elevation=0.0
    )

    print(f"\n数据概览:")
    print(f"  钻孔数量: {df['borehole_id'].nunique()}")
    print(f"  采样点数: {len(df)}")
    print(f"  X范围: {df['x'].min():.1f} ~ {df['x'].max():.1f} m")
    print(f"  Y范围: {df['y'].min():.1f} ~ {df['y'].max():.1f} m")
    print(f"  Z范围: {df['z'].min():.1f} ~ {df['z'].max():.1f} m")

    # ==================== 2. 图构建与训练 ====================
    print("\n" + "=" * 70)
    print("阶段 2/4: 图构建与GNN训练")
    print("=" * 70)

    result = processor.process(
        df,
        label_col='lithology',
        feature_cols=['layer_thickness', 'relative_depth', 'depth_ratio'],
        test_size=0.2,
        val_size=0.1,
        merge_coal=merge_coal
    )

    data = result['data']

    # 创建模型 - 根据模型类型传递不同参数
    model_kwargs = {
        'model_name': model_type,
        'in_channels': result['num_features'],
        'hidden_channels': hidden_dim,
        'out_channels': result['num_classes'],
        'num_layers': num_layers,
        'dropout': 0.3,
    }
    # heads 参数只对 GAT 类模型有效
    if model_type.lower() in ['gat', 'enhanced', 'transformer']:
        model_kwargs['heads'] = 4

    model = get_model(**model_kwargs)

    print(f"\n模型: {model_type.upper()}")
    print(f"  输入特征: {result['num_features']}")
    print(f"  输出类别: {result['num_classes']}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练 - 使用稳定配置
    class_weights = compute_class_weights(data.y, method='effective')

    trainer = GeoModelTrainer(
        model=model,
        device='auto',
        learning_rate=0.001,       # 稳定的学习率
        weight_decay=5e-4,         # 适中的权重衰减
        optimizer_type='adamw',
        scheduler_type='plateau',  # 稳定的调度器
        class_weights=class_weights,
        loss_type='focal',
        num_classes=result['num_classes'],
        focal_gamma=2.0,           # 标准gamma值
        use_augmentation=True,     # 启用数据增强
        augment_noise_std=0.02,    # 低噪声
        augment_edge_drop=0.03,    # 低边丢弃
        use_mixup=False,           # 关闭Mixup
        use_ema=True,              # 启用EMA平滑训练
        ema_decay=0.995            # EMA衰减率
    )

    model_path = os.path.join(output_dir, 'models', 'best_model.pt')
    history = trainer.train(
        data,
        epochs=epochs,
        patience=100,
        verbose=True,
        save_path=model_path
    )

    # ==================== 3. 模型评估 ====================
    print("\n" + "=" * 70)
    print("阶段 3/4: 模型评估")
    print("=" * 70)

    eval_results = trainer.evaluate(data, result['lithology_classes'])

    # 保存预测结果
    predictions, probabilities = trainer.predict(data, return_probs=True)
    output_df = result['raw_df'].copy()
    output_df['predicted_lithology'] = [result['lithology_classes'][p] for p in predictions]
    output_df['confidence'] = probabilities.max(axis=1)
    output_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')

    # ==================== 4. 三维地质建模 ====================
    print("\n" + "=" * 70)
    print("阶段 4/4: 三维地质建模")
    print("=" * 70)

    geo_model = None
    gnn_model = None
    layer_model = None
    layer_processor = None

    # 传统插值建模
    if modeling_method in ['traditional', 'all']:
        print("\n[方法1] 传统层面插值建模...")
        geo_model = build_geological_model(
            trainer=trainer,
            data=data,
            result=result,
            resolution=grid_resolution,
            interpolation_method='kriging',
            output_dir=output_dir
        )

    # GNN直接预测建模
    if modeling_method in ['gnn', 'all']:
        print("\n[方法2] GNN直接预测建模...")
        gnn_model = build_gnn_geological_model(
            trainer=trainer,
            data=data,
            result=result,
            method='direct',
            resolution=grid_resolution,
            output_dir=output_dir,
            k_neighbors=8,
            smooth_output=True,
            smooth_sigma=0.5
        )

    # 层序累加建模（推荐）
    if modeling_method in ['layer', 'all']:
        print("\n[方法3] 层序累加建模（推荐）...")
        print("  核心思想: 逐层累加厚度构建曲面，GNN预测厚度（回归）")
        print("  优势: 无岩体冲突、无空缺区域、符合地质沉积规律")

        layer_model, layer_processor, thickness_trainer = build_layer_based_model(
            df=df,
            resolution=grid_resolution,
            use_gnn=False,  # 使用传统插值，不训练厚度GNN
            epochs=min(epochs, 300),  # 厚度预测训练轮数
            hidden_dim=hidden_dim,
            num_gnn_layers=num_layers,
            output_dir=output_dir,
            verbose=True
        )

    # 保存预处理器
    processor.save_preprocessor(os.path.join(output_dir, 'preprocessor.json'))

    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("全部完成!")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  模型权重: {output_dir}/models/best_model.pt")
    print(f"  钻孔预测: {output_dir}/predictions.csv")
    if modeling_method in ['traditional', 'all']:
        print(f"\n  [传统方法]")
        print(f"    三维模型(VTK): {output_dir}/geological_model.vtk (如有)")
        print(f"    三维模型(NumPy): {output_dir}/geological_model.npz")
        print(f"    体积统计: {output_dir}/model_statistics.csv")
    if modeling_method in ['gnn', 'all']:
        print(f"\n  [GNN直接预测]")
        print(f"    三维模型(VTK): {output_dir}/gnn_model_direct.vtk")
        print(f"    三维模型(NumPy): {output_dir}/gnn_model_direct.npz")
        print(f"    体积统计: {output_dir}/gnn_model_direct_stats.csv")
    if modeling_method in ['layer', 'all']:
        print(f"\n  [层序累加建模] (推荐)")
        print(f"    三维模型(VTK): {output_dir}/layer_model.vtk")
        print(f"    三维模型(NumPy): {output_dir}/layer_model.npz")
        print(f"    体积统计: {output_dir}/layer_model_stats.csv")
        print(f"    厚度预测模型: {output_dir}/thickness_model.pt")
    print(f"\n提示: VTK文件可用ParaView打开进行三维可视化")

    return trainer, data, result, geo_model, gnn_model, layer_model


def run_demo():
    """运行演示 (使用默认参数)"""
    return run_full_pipeline(
        grid_resolution=(40, 40, 40),
        epochs=200,
        modeling_method='layer'  # 使用推荐的层序累加建模
    )


def run_layer_based_modeling(
    data_dir: str = None,
    hidden_dim: int = 128,
    num_layers: int = 4,
    epochs: int = 300,
    sample_interval: float = 1.0,
    k_neighbors: int = 10,
    grid_resolution: tuple = (50, 50, 50),
    use_gnn: bool = True,
    smooth_surfaces: bool = True,
    smooth_sigma: float = 1.0,
    output_dir: str = 'output'
):
    """
    层序累加地质建模（独立流程）

    核心思想：
    1. 确定底面作为基准
    2. 从最深层开始，逐层累加厚度构建曲面
    3. GNN用于预测每层厚度（回归问题）
    4. 曲面之间的空间即为该层岩体

    优势：
    - 无岩体冲突（数学上不可能）
    - 无空缺区域（完全填充）
    - 符合地质沉积规律
    - 曲面连续光滑

    Args:
        data_dir: 钻孔数据目录
        hidden_dim: GNN隐藏层维度
        num_layers: GNN层数
        epochs: 厚度预测模型训练轮数
        sample_interval: 钻孔采样间隔
        k_neighbors: 图构建的K邻居数
        grid_resolution: 网格分辨率 (nx, ny, nz)
        use_gnn: 是否使用GNN预测厚度
        smooth_surfaces: 是否平滑曲面
        smooth_sigma: 平滑系数
        output_dir: 输出目录

    Returns:
        layer_model: 层序累加地质模型
        layer_processor: 层序数据处理器
        thickness_trainer: 厚度预测训练器
    """
    print("=" * 70)
    print("层序累加地质建模")
    print("=" * 70)
    print("\n核心思想:")
    print("  1. 确定底面作为基准")
    print("  2. 从最深层开始，逐层累加厚度构建曲面")
    print("  3. GNN预测每层厚度（回归问题，非分类）")
    print("  4. 曲面之间的空间即为该层岩体")
    print("\n优势:")
    print("  - 无岩体冲突（数学上保证）")
    print("  - 无空缺区域（完全填充）")
    print("  - 符合地质沉积规律")

    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data')

    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 数据加载 ====================
    print("\n" + "=" * 70)
    print("阶段 1/3: 数据加载")
    print("=" * 70)

    processor = BoreholeDataProcessor(
        k_neighbors=k_neighbors,
        sample_interval=sample_interval
    )

    df = processor.load_all_boreholes(
        data_dir=data_dir,
        surface_elevation=0.0
    )

    print(f"\n数据概览:")
    print(f"  钻孔数量: {df['borehole_id'].nunique()}")
    print(f"  采样点数: {len(df)}")
    print(f"  岩性种类: {df['lithology'].nunique()}")
    print(f"  X范围: {df['x'].min():.1f} ~ {df['x'].max():.1f} m")
    print(f"  Y范围: {df['y'].min():.1f} ~ {df['y'].max():.1f} m")
    print(f"  Z范围: {df['z'].min():.1f} ~ {df['z'].max():.1f} m")

    # ==================== 2. 层序累加建模 ====================
    print("\n" + "=" * 70)
    print("阶段 2/3: 层序累加建模")
    print("=" * 70)

    layer_model, layer_processor, thickness_trainer = build_layer_based_model(
        df=df,
        resolution=grid_resolution,
        use_gnn=use_gnn,
        epochs=epochs,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_layers,
        output_dir=output_dir,
        verbose=True
    )

    # ==================== 3. 统计与输出 ====================
    print("\n" + "=" * 70)
    print("阶段 3/3: 统计与输出")
    print("=" * 70)

    stats = layer_model.get_statistics(layer_processor.layer_order)
    print("\n模型统计:")
    print(stats.to_string(index=False))

    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("层序累加建模完成!")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  三维模型(VTK): {output_dir}/layer_model.vtk")
    print(f"  三维模型(NumPy): {output_dir}/layer_model.npz")
    print(f"  体积统计: {output_dir}/layer_model_stats.csv")
    if use_gnn:
        print(f"  厚度预测模型: {output_dir}/thickness_model.pt")
    print(f"\n提示: VTK文件可用ParaView打开进行三维可视化")

    return layer_model, layer_processor, thickness_trainer


def run_training_only(
    data_dir: str,
    model_type: str = 'graphsage',
    hidden_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    sample_interval: float = 2.0,
    k_neighbors: int = 10,
    output_dir: str = 'output',
    merge_coal: bool = True
):
    """仅训练模型，不进行建模"""
    print("=" * 60)
    print("GNN模型训练")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    # 加载数据
    processor = BoreholeDataProcessor(
        k_neighbors=k_neighbors,
        sample_interval=sample_interval
    )
    df = processor.load_all_boreholes(data_dir)
    result = processor.process(df, feature_cols=['layer_thickness'], merge_coal=merge_coal)
    data = result['data']

    # 创建并训练模型
    model = get_model(
        model_name=model_type,
        in_channels=result['num_features'],
        hidden_channels=hidden_dim,
        out_channels=result['num_classes'],
        num_layers=num_layers
    )

    class_weights = compute_class_weights(data.y)
    trainer = GeoModelTrainer(model=model, class_weights=class_weights)

    model_path = os.path.join(output_dir, 'models', 'best_model.pt')
    trainer.train(data, epochs=epochs, save_path=model_path)
    trainer.evaluate(data, result['lithology_classes'])

    processor.save_preprocessor(os.path.join(output_dir, 'preprocessor.json'))

    return trainer, data, result


def run_modeling_only(
    model_path: str,
    data_dir: str,
    preprocessor_path: str,
    resolution: tuple = (50, 50, 50),
    output_dir: str = 'output',
    merge_coal: bool = True
):
    """仅进行建模 (使用已训练的模型)"""
    print("=" * 60)
    print("三维地质建模 (使用已有模型)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 加载预处理器
    processor = BoreholeDataProcessor()
    processor.load_preprocessor(preprocessor_path)

    # 加载数据
    df = processor.load_all_boreholes(data_dir)
    result = processor.process(df, feature_cols=['layer_thickness'], merge_coal=merge_coal)
    data = result['data']

    # 加载模型
    model = get_model(
        model_name='graphsage',
        in_channels=result['num_features'],
        hidden_channels=64,
        out_channels=result['num_classes']
    )

    trainer = GeoModelTrainer(model=model)
    trainer.load_model(model_path)

    # 建模
    geo_model = build_geological_model(
        trainer=trainer,
        data=data,
        result=result,
        resolution=resolution,
        output_dir=output_dir
    )

    return geo_model


def run_webapp():
    """启动Streamlit可视化界面"""
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    subprocess.run(['streamlit', 'run', app_path])


def main():
    parser = argparse.ArgumentParser(
        description='GNN三维地质建模系统 - 敏东矿区',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
=========

  完整流程 (训练 + 建模):
    python main.py run --data ./data --resolution 50 50 50

  快速演示:
    python main.py demo

  层序累加建模 (推荐):
    python main.py layer --data ./data --resolution 50 50 50 --epochs 300

  仅训练模型:
    python main.py train --data ./data --model graphsage --epochs 300

  仅进行建模 (使用已有模型):
    python main.py model --model-path output/models/best_model.pt --data ./data

  启动可视化界面:
    python main.py webapp

建模方法说明:
============
  traditional - 传统层面插值建模
  gnn         - GNN直接预测岩性（分类）
  layer       - 层序累加建模（推荐，GNN预测厚度回归）
  all         - 运行所有建模方法

输出文件说明:
============
  geological_model.vtk  - VTK格式，可用ParaView打开
  geological_model.npz  - NumPy格式，可用Python读取
  layer_model.vtk       - 层序累加模型VTK格式
  layer_model.npz       - 层序累加模型NumPy格式
  model_statistics.csv  - 岩性体积统计
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # demo命令
    subparsers.add_parser('demo', help='运行完整演示 (训练+层序累加建模)')

    # webapp命令
    subparsers.add_parser('webapp', help='启动可视化Web界面')

    # run命令 (完整流程)
    run_parser = subparsers.add_parser('run', help='完整流程: 训练 + 建模')
    run_parser.add_argument('--data', type=str, default='./data', help='数据目录')
    run_parser.add_argument('--model', type=str, default='graphsage',
                           choices=['gcn', 'graphsage', 'gat', 'geo3d', 'enhanced', 'transformer'])
    run_parser.add_argument('--hidden', type=int, default=128)
    run_parser.add_argument('--layers', type=int, default=4)
    run_parser.add_argument('--epochs', type=int, default=500)
    run_parser.add_argument('--sample-interval', type=float, default=1.0)
    run_parser.add_argument('--k-neighbors', type=int, default=12)
    run_parser.add_argument('--resolution', type=int, nargs=3, default=[50, 50, 50],
                           help='网格分辨率 (nx ny nz)')
    run_parser.add_argument('--output', type=str, default='output')
    run_parser.add_argument('--modeling-method', type=str, default='layer',
                           choices=['traditional', 'gnn', 'layer', 'all'],
                           help='建模方法: traditional=传统插值, gnn=GNN直接预测, layer=层序累加(推荐), all=所有方法')
    run_parser.add_argument('--no-merge-coal', action='store_true', help='不合并煤层，保留各煤层编号')

    # layer命令 (层序累加建模)
    layer_parser = subparsers.add_parser('layer', help='层序累加地质建模（推荐）')
    layer_parser.add_argument('--data', type=str, default='./data', help='数据目录')
    layer_parser.add_argument('--hidden', type=int, default=128, help='GNN隐藏层维度')
    layer_parser.add_argument('--layers', type=int, default=4, help='GNN层数')
    layer_parser.add_argument('--epochs', type=int, default=300, help='厚度预测模型训练轮数')
    layer_parser.add_argument('--sample-interval', type=float, default=1.0, help='采样间隔')
    layer_parser.add_argument('--k-neighbors', type=int, default=10, help='K邻居数')
    layer_parser.add_argument('--resolution', type=int, nargs=3, default=[50, 50, 50],
                             help='网格分辨率 (nx ny nz)')
    layer_parser.add_argument('--no-gnn', action='store_true', help='不使用GNN，使用传统插值')
    layer_parser.add_argument('--smooth', type=float, default=1.0, help='曲面平滑系数')
    layer_parser.add_argument('--output', type=str, default='output')

    # train命令 (仅训练)
    train_parser = subparsers.add_parser('train', help='仅训练模型')
    train_parser.add_argument('--data', type=str, default='./data')
    train_parser.add_argument('--model', type=str, default='graphsage',
                             choices=['gcn', 'graphsage', 'gat', 'geo3d', 'enhanced', 'transformer'])
    train_parser.add_argument('--hidden', type=int, default=128)
    train_parser.add_argument('--layers', type=int, default=4)
    train_parser.add_argument('--epochs', type=int, default=500)
    train_parser.add_argument('--sample-interval', type=float, default=1.0)
    train_parser.add_argument('--k-neighbors', type=int, default=12)
    train_parser.add_argument('--output', type=str, default='output')
    train_parser.add_argument('--no-merge-coal', action='store_true', help='不合并煤层，保留各煤层编号')

    # model命令 (仅建模)
    model_parser = subparsers.add_parser('model', help='仅进行建模 (使用已有模型)')
    model_parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    model_parser.add_argument('--data', type=str, default='./data')
    model_parser.add_argument('--preprocessor', type=str, default='output/preprocessor.json')
    model_parser.add_argument('--resolution', type=int, nargs=3, default=[50, 50, 50])
    model_parser.add_argument('--output', type=str, default='output')
    model_parser.add_argument('--no-merge-coal', action='store_true', help='不合并煤层，保留各煤层编号 (需与训练时一致)')

    args = parser.parse_args()

    if args.command == 'demo':
        run_demo()

    elif args.command == 'webapp':
        run_webapp()

    elif args.command == 'run':
        run_full_pipeline(
            data_dir=args.data,
            model_type=args.model,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            sample_interval=args.sample_interval,
            k_neighbors=args.k_neighbors,
            grid_resolution=tuple(args.resolution),
            output_dir=args.output,
            modeling_method=args.modeling_method,
            merge_coal=not args.no_merge_coal
        )

    elif args.command == 'train':
        run_training_only(
            data_dir=args.data,
            model_type=args.model,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            sample_interval=args.sample_interval,
            k_neighbors=args.k_neighbors,
            output_dir=args.output,
            merge_coal=not args.no_merge_coal
        )

    elif args.command == 'layer':
        run_layer_based_modeling(
            data_dir=args.data,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            sample_interval=args.sample_interval,
            k_neighbors=args.k_neighbors,
            grid_resolution=tuple(args.resolution),
            use_gnn=not args.no_gnn,
            smooth_sigma=args.smooth,
            output_dir=args.output
        )

    elif args.command == 'model':
        run_modeling_only(
            model_path=args.model_path,
            data_dir=args.data,
            preprocessor_path=args.preprocessor,
            resolution=tuple(args.resolution),
            output_dir=args.output,
            merge_coal=not args.no_merge_coal
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
