import sys
import os
import traceback
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

# Project modules
from src.thickness_data_loader import ThicknessDataProcessor
from src.gnn_thickness_modeling import (
    GNNThicknessPredictor, GeologicalModelBuilder
)
from src.thickness_trainer import create_trainer, get_optimized_config_for_small_dataset
from src.thickness_predictor_v2 import (
    PerLayerThicknessPredictor, HybridThicknessPredictor, evaluate_predictor
)

class DataLoaderThread(QThread):
    """数据加载线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir, merge_coal, k_neighbors, layer_method, min_occurrence_rate):
        super().__init__()
        self.data_dir = data_dir
        self.merge_coal = merge_coal
        self.k_neighbors = k_neighbors
        self.layer_method = layer_method
        self.min_occurrence_rate = min_occurrence_rate

    def run(self):
        try:
            self.progress.emit("正在加载钻孔数据...")
            processor = ThicknessDataProcessor(
                merge_coal=self.merge_coal,
                k_neighbors=self.k_neighbors,
                graph_type='knn'
            )
            result = processor.process_directory(
                self.data_dir,
                layer_method=self.layer_method,
                min_occurrence_rate=self.min_occurrence_rate
            )
            self.progress.emit(f"✓ 数据加载完成: {len(result['borehole_ids'])} 个钻孔, {result['num_layers']} 个地层")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"数据加载失败: {str(e)}")


class TrainingThread(QThread):
    """模型训练线程"""
    progress = pyqtSignal(str)
    epoch_update = pyqtSignal(int, float, float)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, data_result, config):
        super().__init__()
        self.data_result = data_result
        self.config = config

    def run(self):
        try:
            self.progress.emit("正在初始化模型...")

            n_features = self.config['num_features']
            n_layers = self.config['num_layers']

            model, trainer = create_trainer(
                num_features=n_features,
                num_layers=n_layers,
                hidden_channels=self.config['hidden_dim'],
                gnn_layers=self.config['gnn_layers'],
                dropout=self.config['dropout'],
                conv_type=self.config['conv_type'],
                learning_rate=self.config['learning_rate'],
                use_augmentation=self.config.get('use_augmentation', False),
                scheduler_type='plateau',
                heads=self.config.get('heads', 4)
            )

            self.progress.emit("开始训练...")

            history = trainer.train(
                data=self.data_result['data'],
                epochs=self.config['epochs'],
                patience=self.config['patience'],
                warmup_epochs=self.config.get('warmup_epochs', 0),
                verbose=False
            )

            self.progress.emit("✓ 训练完成!")
            self.finished.emit(model, history)

        except Exception as e:
            self.error.emit(f"训练失败: {str(e)}\n{traceback.format_exc()}")


class TraditionalPredictorThread(QThread):
    """传统方法拟合线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, dict)
    error = pyqtSignal(str)

    def __init__(self, data_result, interp_method):
        super().__init__()
        self.data_result = data_result
        self.interp_method = interp_method

    def run(self):
        try:
            self.progress.emit("正在拟合传统模型...")

            raw_df = self.data_result['raw_df']
            layer_order = self.data_result['layer_order']

            if self.interp_method == 'hybrid':
                predictor = HybridThicknessPredictor(
                    layer_order=layer_order,
                    kriging_threshold=10,
                    smooth_factor=0.3,
                    min_thickness=0.5
                )
            else:
                predictor = PerLayerThicknessPredictor(
                    layer_order=layer_order,
                    default_method=self.interp_method,
                    idw_power=2.0,
                    n_neighbors=8,
                    min_thickness=0.5
                )

            predictor.fit(
                raw_df,
                x_col='x',
                y_col='y',
                layer_col='layer_name',
                thickness_col='thickness'
            )

            coords = self.data_result['borehole_coords']
            x_range = (coords[:, 0].min(), coords[:, 0].max())
            y_range = (coords[:, 1].min(), coords[:, 1].max())
            grid_x = np.linspace(x_range[0], x_range[1], 30)
            grid_y = np.linspace(y_range[0], y_range[1], 30)

            eval_metrics = evaluate_predictor(
                predictor, raw_df, grid_x, grid_y,
                x_col='x', y_col='y',
                layer_col='layer_name',
                thickness_col='thickness'
            )

            self.progress.emit("✓ 传统方法拟合完成!")
            self.finished.emit(predictor, eval_metrics)

        except Exception as e:
            self.error.emit(f"拟合失败: {str(e)}\n{traceback.format_exc()}")


class ModelingThread(QThread):
    """三维建模线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, object, object)
    error = pyqtSignal(str)

    def __init__(self, data_result, predictor, resolution, base_level, gap_value, use_traditional):
        super().__init__()
        self.data_result = data_result
        self.predictor = predictor
        self.resolution = resolution
        self.base_level = base_level
        self.gap_value = gap_value
        self.use_traditional = use_traditional

    def run(self):
        try:
            self.progress.emit("正在生成网格...")

            coords = self.data_result['borehole_coords']
            x_range = (coords[:, 0].min(), coords[:, 0].max())
            y_range = (coords[:, 1].min(), coords[:, 1].max())

            grid_x = np.linspace(x_range[0], x_range[1], self.resolution)
            grid_y = np.linspace(y_range[0], y_range[1], self.resolution)

            if self.use_traditional:
                thickness_grids = self.predictor.predict_grid(grid_x, grid_y)
                XI, YI = np.meshgrid(grid_x, grid_y)
            else:
                model = self.predictor
                device = next(model.parameters()).device
                model.eval()
                data = self.data_result['data'].to(device)

                with torch.no_grad():
                    pred_thick, pred_exist = model(
                        data.x, data.edge_index,
                        data.edge_attr if hasattr(data, 'edge_attr') else None
                    )
                    pred_thick = pred_thick.cpu().numpy()
                    pred_exist = torch.sigmoid(pred_exist).cpu().numpy()

                from scipy.interpolate import griddata
                XI, YI = np.meshgrid(grid_x, grid_y)
                xi_flat, yi_flat = XI.flatten(), YI.flatten()

                thickness_grids = {}
                for i, layer_name in enumerate(self.data_result['layer_order']):
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

            self.progress.emit("正在构建三维模型...")

            builder = GeologicalModelBuilder(
                layer_order=self.data_result['layer_order'],
                resolution=self.resolution,
                base_level=self.base_level,
                gap_value=self.gap_value
            )

            block_models, XI, YI = builder.build_model(
                thickness_grids=thickness_grids,
                x_range=x_range,
                y_range=y_range
            )

            self.progress.emit("✓ 三维模型构建完成!")
            self.finished.emit(block_models, XI, YI)

        except Exception as e:
            self.error.emit(f"建模失败: {str(e)}\n{traceback.format_exc()}")
