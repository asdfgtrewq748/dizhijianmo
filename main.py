"""
主入口脚本
提供命令行接口：训练、建模、可视化

针对敏东矿区钻孔数据格式优化
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


def run_full_pipeline(
    data_dir: str = None,
    model_type: str = 'enhanced',
    hidden_dim: int = 128,
    num_layers: int = 4,
    epochs: int = 300,
    sample_interval: float = 1.0,
    k_neighbors: int = 15,
    grid_resolution: tuple = (50, 50, 50),
    output_dir: str = 'output'
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
        feature_cols=['layer_thickness'],
        test_size=0.2,
        val_size=0.1
    )

    data = result['data']

    # 创建模型
    model = get_model(
        model_name=model_type,
        in_channels=result['num_features'],
        hidden_channels=hidden_dim,
        out_channels=result['num_classes'],
        num_layers=num_layers,
        dropout=0.5
    )

    print(f"\n模型: {model_type.upper()}")
    print(f"  输入特征: {result['num_features']}")
    print(f"  输出类别: {result['num_classes']}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    class_weights = compute_class_weights(data.y)

    trainer = GeoModelTrainer(
        model=model,
        device='auto',
        learning_rate=0.005,
        weight_decay=1e-4,
        class_weights=class_weights,
        loss_type='focal',
        num_classes=result['num_classes'],
        focal_gamma=2.0
    )

    model_path = os.path.join(output_dir, 'models', 'best_model.pt')
    history = trainer.train(
        data,
        epochs=epochs,
        patience=50,
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

    geo_model = build_geological_model(
        trainer=trainer,
        data=data,
        result=result,
        resolution=grid_resolution,
        interpolation_method='knn',
        output_dir=output_dir
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
    print(f"  三维模型(VTK): {output_dir}/geological_model.vtk")
    print(f"  三维模型(NumPy): {output_dir}/geological_model.npz")
    print(f"  三维模型(CSV): {output_dir}/geological_model.csv")
    print(f"  体积统计: {output_dir}/model_statistics.csv")
    print(f"\n提示: VTK文件可用ParaView打开进行三维可视化")

    return trainer, data, result, geo_model


def run_demo():
    """运行演示 (使用默认参数)"""
    return run_full_pipeline(
        grid_resolution=(40, 40, 40),
        epochs=200
    )


def run_training_only(
    data_dir: str,
    model_type: str = 'graphsage',
    hidden_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    sample_interval: float = 2.0,
    k_neighbors: int = 10,
    output_dir: str = 'output'
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
    result = processor.process(df, feature_cols=['layer_thickness'])
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
    output_dir: str = 'output'
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
    result = processor.process(df, feature_cols=['layer_thickness'])
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

  仅训练模型:
    python main.py train --data ./data --model graphsage --epochs 300

  仅进行建模 (使用已有模型):
    python main.py model --model-path output/models/best_model.pt --data ./data

  启动可视化界面:
    python main.py webapp

输出文件说明:
============
  geological_model.vtk  - VTK格式，可用ParaView打开
  geological_model.npz  - NumPy格式，可用Python读取
  geological_model.csv  - CSV格式，通用表格格式
  model_statistics.csv  - 岩性体积统计
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # demo命令
    subparsers.add_parser('demo', help='运行完整演示 (训练+建模)')

    # webapp命令
    subparsers.add_parser('webapp', help='启动可视化Web界面')

    # run命令 (完整流程)
    run_parser = subparsers.add_parser('run', help='完整流程: 训练 + 建模')
    run_parser.add_argument('--data', type=str, default='./data', help='数据目录')
    run_parser.add_argument('--model', type=str, default='enhanced',
                           choices=['gcn', 'graphsage', 'gat', 'geo3d', 'enhanced'])
    run_parser.add_argument('--hidden', type=int, default=128)
    run_parser.add_argument('--layers', type=int, default=4)
    run_parser.add_argument('--epochs', type=int, default=300)
    run_parser.add_argument('--sample-interval', type=float, default=1.0)
    run_parser.add_argument('--k-neighbors', type=int, default=15)
    run_parser.add_argument('--resolution', type=int, nargs=3, default=[50, 50, 50],
                           help='网格分辨率 (nx ny nz)')
    run_parser.add_argument('--output', type=str, default='output')

    # train命令 (仅训练)
    train_parser = subparsers.add_parser('train', help='仅训练模型')
    train_parser.add_argument('--data', type=str, default='./data')
    train_parser.add_argument('--model', type=str, default='enhanced',
                             choices=['gcn', 'graphsage', 'gat', 'geo3d', 'enhanced'])
    train_parser.add_argument('--hidden', type=int, default=128)
    train_parser.add_argument('--layers', type=int, default=4)
    train_parser.add_argument('--epochs', type=int, default=300)
    train_parser.add_argument('--sample-interval', type=float, default=1.0)
    train_parser.add_argument('--k-neighbors', type=int, default=15)
    train_parser.add_argument('--output', type=str, default='output')

    # model命令 (仅建模)
    model_parser = subparsers.add_parser('model', help='仅进行建模 (使用已有模型)')
    model_parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    model_parser.add_argument('--data', type=str, default='./data')
    model_parser.add_argument('--preprocessor', type=str, default='output/preprocessor.json')
    model_parser.add_argument('--resolution', type=int, nargs=3, default=[50, 50, 50])
    model_parser.add_argument('--output', type=str, default='output')

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
            output_dir=args.output
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
            output_dir=args.output
        )

    elif args.command == 'model':
        run_modeling_only(
            model_path=args.model_path,
            data_dir=args.data,
            preprocessor_path=args.preprocessor,
            resolution=tuple(args.resolution),
            output_dir=args.output
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
