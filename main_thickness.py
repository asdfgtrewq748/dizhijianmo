"""
GNN厚度预测三维地质建模 - 命令行主入口

使用重构后的正确逻辑：
1. GNN预测厚度（回归）
2. 层序累加构建三维模型

使用方法：
    python main_thickness.py train --epochs 200
    python main_thickness.py build --output model_output
    python main_thickness.py all --epochs 200 --output model_output
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.thickness_data_loader import ThicknessDataProcessor
from src.gnn_thickness_modeling import (
    GNNThicknessPredictor,
    GeologicalModelBuilder,
    GNNGeologicalModeling,
    TraditionalThicknessInterpolator
)
from src.thickness_trainer import create_trainer, ThicknessEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GNN厚度预测三维地质建模系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 训练GNN厚度预测模型
    python main_thickness.py train --epochs 300

    # 构建三维地质模型
    python main_thickness.py build --resolution 50

    # 完整流程（训练+构建）
    python main_thickness.py all --epochs 300 --resolution 50

    # 使用传统插值方法（对比基准）
    python main_thickness.py baseline --method linear
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 训练子命令
    train_parser = subparsers.add_parser('train', help='训练GNN厚度预测模型')
    train_parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    train_parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    train_parser.add_argument('--gnn-layers', type=int, default=3, help='GNN层数')
    train_parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率')
    train_parser.add_argument('--conv-type', type=str, default='gatv2',
                              choices=['gatv2', 'transformer', 'sage'], help='卷积类型')
    train_parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    train_parser.add_argument('--output', type=str, default='output', help='输出目录')

    # 构建子命令
    build_parser = subparsers.add_parser('build', help='构建三维地质模型')
    build_parser.add_argument('--model-path', type=str, default='output/best_model.pt', help='模型路径')
    build_parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    build_parser.add_argument('--resolution', type=int, default=50, help='网格分辨率')
    build_parser.add_argument('--base-level', type=float, default=0.0, help='基准面高程')
    build_parser.add_argument('--gap', type=float, default=0.0, help='层间间隙')
    build_parser.add_argument('--output', type=str, default='output', help='输出目录')

    # 完整流程子命令
    all_parser = subparsers.add_parser('all', help='完整流程：训练+构建')
    all_parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    all_parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    all_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    all_parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    all_parser.add_argument('--gnn-layers', type=int, default=3, help='GNN层数')
    all_parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率')
    all_parser.add_argument('--conv-type', type=str, default='gatv2', help='卷积类型')
    all_parser.add_argument('--resolution', type=int, default=50, help='网格分辨率')
    all_parser.add_argument('--base-level', type=float, default=0.0, help='基准面高程')
    all_parser.add_argument('--gap', type=float, default=0.0, help='层间间隙')
    all_parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    all_parser.add_argument('--output', type=str, default='output', help='输出目录')

    # 基准方法子命令
    baseline_parser = subparsers.add_parser('baseline', help='使用传统插值方法（对比基准）')
    baseline_parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    baseline_parser.add_argument('--method', type=str, default='linear',
                                  choices=['linear', 'cubic', 'nearest', 'rbf'], help='插值方法')
    baseline_parser.add_argument('--resolution', type=int, default=50, help='网格分辨率')
    baseline_parser.add_argument('--output', type=str, default='output', help='输出目录')

    return parser.parse_args()


def load_data(data_dir: str, merge_coal: bool = False) -> dict:
    """加载并处理数据"""
    print(f"\n{'='*60}")
    print("数据加载与预处理")
    print(f"{'='*60}")
    print(f"数据目录: {data_dir}")

    processor = ThicknessDataProcessor(
        merge_coal=merge_coal,
        k_neighbors=8,
        graph_type='knn',
        normalize=True
    )

    result = processor.process_directory(data_dir)

    print(f"\n数据处理完成:")
    print(f"  钻孔数: {len(result['borehole_ids'])}")
    print(f"  地层数: {result['num_layers']}")
    print(f"  层序: {result['layer_order']}")

    return result


def train_model(args, data_result: dict) -> tuple:
    """训练GNN厚度预测模型"""
    print(f"\n{'='*60}")
    print("GNN厚度预测模型训练")
    print(f"{'='*60}")

    data = data_result['data']

    # 创建模型和训练器
    model, trainer = create_trainer(
        num_features=data_result['num_features'],
        num_layers=data_result['num_layers'],
        hidden_channels=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        learning_rate=args.lr
    )

    print(f"\n模型配置:")
    print(f"  输入特征: {data_result['num_features']}")
    print(f"  输出层数: {data_result['num_layers']}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  GNN层数: {args.gnn_layers}")
    print(f"  卷积类型: {args.conv_type}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    history = trainer.train(
        data=data,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
        log_interval=20,
        save_dir=args.output
    )

    return model, trainer, history


def build_geological_model(
    model,
    data_result: dict,
    resolution: int = 50,
    base_level: float = 0.0,
    gap_value: float = 0.0,
    output_dir: str = 'output'
):
    """构建三维地质模型"""
    print(f"\n{'='*60}")
    print("三维地质模型构建")
    print(f"{'='*60}")

    data = data_result['data']
    layer_order = data_result['layer_order']
    borehole_coords = data_result['borehole_coords']

    # 确定坐标范围
    x_range = (borehole_coords[:, 0].min(), borehole_coords[:, 0].max())
    y_range = (borehole_coords[:, 1].min(), borehole_coords[:, 1].max())

    print(f"\n构建参数:")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"  X范围: {x_range}")
    print(f"  Y范围: {y_range}")
    print(f"  基准面: {base_level}m")
    print(f"  层间隙: {gap_value}m")

    # 创建网格
    grid_x = np.linspace(x_range[0], x_range[1], resolution)
    grid_y = np.linspace(y_range[0], y_range[1], resolution)

    # 使用GNN预测厚度
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        pred_thick, pred_exist = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
        )
        pred_thick = pred_thick.cpu().numpy()
        pred_exist = torch.sigmoid(pred_exist).cpu().numpy()

    # 将点预测插值到网格
    from scipy.interpolate import griddata

    XI, YI = np.meshgrid(grid_x, grid_y)
    xi_flat = XI.flatten()
    yi_flat = YI.flatten()

    thickness_grids = {}
    for i, layer_name in enumerate(layer_order):
        layer_thick = pred_thick[:, i]
        layer_exist = pred_exist[:, i]

        # 使用存在该层的钻孔进行插值
        exist_mask = layer_exist > 0.5
        if exist_mask.sum() < 3:
            exist_mask = np.ones(len(layer_thick), dtype=bool)

        x_valid = borehole_coords[exist_mask, 0]
        y_valid = borehole_coords[exist_mask, 1]
        z_valid = layer_thick[exist_mask]

        try:
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
        except Exception:
            grid_thick = griddata(
                (x_valid, y_valid), z_valid, (xi_flat, yi_flat),
                method='nearest'
            )

        grid_thick = np.clip(grid_thick, 0.5, None)
        thickness_grids[layer_name] = grid_thick.reshape(XI.shape)

    # 使用层序累加构建模型
    builder = GeologicalModelBuilder(
        layer_order=layer_order,
        resolution=resolution,
        base_level=base_level,
        gap_value=gap_value
    )

    block_models, XI, YI = builder.build_model(
        thickness_grids=thickness_grids,
        x_range=x_range,
        y_range=y_range
    )

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型信息
    model_info = {
        'layer_order': layer_order,
        'resolution': resolution,
        'x_range': list(x_range),
        'y_range': list(y_range),
        'base_level': base_level,
        'gap_value': gap_value,
        'layers': []
    }

    for bm in block_models:
        model_info['layers'].append({
            'name': bm.name,
            'avg_thickness': bm.avg_thickness,
            'max_thickness': bm.max_thickness,
            'avg_bottom': bm.avg_bottom,
            'avg_top': bm.avg_height
        })

        # 保存每层的网格数据
        np.savez(
            os.path.join(output_dir, f'layer_{bm.name}.npz'),
            top_surface=bm.top_surface,
            bottom_surface=bm.bottom_surface,
            thickness_grid=bm.thickness_grid
        )

    with open(os.path.join(output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\n构建完成! 输出目录: {output_dir}")
    print(f"\n各层信息:")
    for bm in block_models:
        print(f"  {bm.name}: 平均厚度={bm.avg_thickness:.2f}m, "
              f"底面={bm.avg_bottom:.2f}m, 顶面={bm.avg_height:.2f}m")

    return block_models, XI, YI


def run_baseline(args):
    """运行传统插值基准方法"""
    print(f"\n{'='*60}")
    print("传统插值方法（基准对比）")
    print(f"{'='*60}")

    # 加载数据
    from src.thickness_data_loader import LayerTableProcessor

    processor = LayerTableProcessor(merge_coal=False)
    df_layers = processor.load_all_boreholes(args.data_dir)
    df_layers = processor.standardize_lithology(df_layers)

    # 推断层序
    from src.thickness_data_loader import ThicknessDataBuilder
    builder = ThicknessDataBuilder()
    layer_order = builder.infer_layer_order(df_layers)

    # 确定坐标范围
    x_min, x_max = df_layers['x'].min(), df_layers['x'].max()
    y_min, y_max = df_layers['y'].min(), df_layers['y'].max()

    grid_x = np.linspace(x_min, x_max, args.resolution)
    grid_y = np.linspace(y_min, y_max, args.resolution)

    # 使用传统插值
    interpolator = TraditionalThicknessInterpolator(
        method=args.method,
        layer_order=layer_order
    )

    thickness_grids = interpolator.interpolate_thickness(
        borehole_data=df_layers,
        grid_x=grid_x,
        grid_y=grid_y,
        layer_col='lithology',
        thickness_col='thickness',
        x_col='x',
        y_col='y'
    )

    # 构建模型
    model_builder = GeologicalModelBuilder(
        layer_order=layer_order,
        resolution=args.resolution,
        base_level=0.0,
        gap_value=0.0
    )

    block_models, XI, YI = model_builder.build_model(
        thickness_grids=thickness_grids,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max)
    )

    # 保存结果
    os.makedirs(args.output, exist_ok=True)

    model_info = {
        'method': f'baseline_{args.method}',
        'layer_order': layer_order,
        'resolution': args.resolution,
        'layers': []
    }

    for bm in block_models:
        model_info['layers'].append({
            'name': bm.name,
            'avg_thickness': bm.avg_thickness,
            'max_thickness': bm.max_thickness,
            'avg_bottom': bm.avg_bottom,
            'avg_top': bm.avg_height
        })

    with open(os.path.join(args.output, f'baseline_{args.method}_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\n基准模型构建完成!")
    print(f"插值方法: {args.method}")
    print(f"\n各层信息:")
    for bm in block_models:
        print(f"  {bm.name}: 平均厚度={bm.avg_thickness:.2f}m")


def main():
    args = parse_args()

    if args.command is None:
        print("请指定子命令。使用 --help 查看帮助。")
        return

    print(f"\n{'#'*60}")
    print(f"# GNN厚度预测三维地质建模系统")
    print(f"# 命令: {args.command}")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    if args.command == 'train':
        # 训练模式
        data_result = load_data(args.data_dir)
        model, trainer, history = train_model(args, data_result)

        # 保存元数据
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'training_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'layer_order': data_result['layer_order'],
                'num_features': data_result['num_features'],
                'num_layers': data_result['num_layers'],
                'args': vars(args)
            }, f, ensure_ascii=False, indent=2)

        print(f"\n训练完成! 模型保存到: {args.output}")

    elif args.command == 'build':
        # 构建模式
        data_result = load_data(args.data_dir)

        # 加载模型
        from src.gnn_thickness_modeling import GNNThicknessPredictor

        checkpoint = torch.load(args.model_path, map_location='cpu')

        # 从元数据恢复模型配置
        metadata_path = os.path.join(os.path.dirname(args.model_path), 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            model_args = metadata.get('args', {})
        else:
            model_args = {'hidden_dim': 128, 'gnn_layers': 3, 'dropout': 0.2, 'conv_type': 'gatv2'}

        model = GNNThicknessPredictor(
            in_channels=data_result['num_features'],
            hidden_channels=model_args.get('hidden_dim', 128),
            num_layers=model_args.get('gnn_layers', 3),
            num_output_layers=data_result['num_layers'],
            dropout=model_args.get('dropout', 0.2),
            conv_type=model_args.get('conv_type', 'gatv2')
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # 构建模型
        build_geological_model(
            model=model,
            data_result=data_result,
            resolution=args.resolution,
            base_level=args.base_level,
            gap_value=args.gap,
            output_dir=args.output
        )

    elif args.command == 'all':
        # 完整流程
        data_result = load_data(args.data_dir)
        model, trainer, history = train_model(args, data_result)

        # 保存元数据
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'training_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'layer_order': data_result['layer_order'],
                'num_features': data_result['num_features'],
                'num_layers': data_result['num_layers'],
                'args': vars(args)
            }, f, ensure_ascii=False, indent=2)

        # 构建模型
        build_geological_model(
            model=model,
            data_result=data_result,
            resolution=args.resolution,
            base_level=args.base_level,
            gap_value=args.gap,
            output_dir=args.output
        )

        # 生成评估报告
        evaluator = ThicknessEvaluator(model)
        report = evaluator.generate_report(
            data=data_result['data'],
            layer_order=data_result['layer_order'],
            save_path=os.path.join(args.output, 'evaluation_report.txt')
        )
        print(report)

    elif args.command == 'baseline':
        run_baseline(args)

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
