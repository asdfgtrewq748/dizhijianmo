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
    progress_percent = pyqtSignal(int)  # 新增：百分比进度信号
    progress_detail = pyqtSignal(str)  # 新增：详细信息信号
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir, merge_coal, k_neighbors, layer_method, min_occurrence_rate):
        super().__init__()
        self.data_dir = data_dir
        self.merge_coal = merge_coal
        self.k_neighbors = k_neighbors
        self.layer_method = layer_method
        self.min_occurrence_rate = min_occurrence_rate
        self._stop_requested = False

    def request_stop(self):
        """请求停止线程"""
        self._stop_requested = True

    def run(self):
        try:
            self.progress.emit("正在扫描钻孔文件...")
            self.progress_percent.emit(5)

            if self._stop_requested:
                return

            self.progress.emit("正在加载钻孔数据...")
            self.progress_percent.emit(10)

            processor = ThicknessDataProcessor(
                merge_coal=self.merge_coal,
                k_neighbors=self.k_neighbors,
                graph_type='knn'
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(30)
            self.progress_detail.emit("解析地层数据...")

            result = processor.process_directory(
                self.data_dir,
                layer_method=self.layer_method,
                min_occurrence_rate=self.min_occurrence_rate
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(70)
            self.progress_detail.emit("构建图网络...")

            # 模拟图网络构建过程
            import time
            time.sleep(0.5)

            if self._stop_requested:
                return

            self.progress_percent.emit(90)
            self.progress_detail.emit(f"完成: {len(result['borehole_ids'])} 个钻孔, {result['num_layers']} 个地层")

            self.progress.emit(f"✓ 数据加载完成: {len(result['borehole_ids'])} 个钻孔, {result['num_layers']} 个地层")
            self.progress_percent.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"数据加载失败: {str(e)}")


class TrainingThread(QThread):
    """模型训练线程"""
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)  # 新增：百分比进度信号
    progress_detail = pyqtSignal(str)  # 新增：详细信息信号
    epoch_update = pyqtSignal(int, float, float)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, data_result, config):
        super().__init__()
        self.data_result = data_result
        self.config = config
        self._stop_requested = False

    def request_stop(self):
        """请求停止训练"""
        self._stop_requested = True

    def run(self):
        try:
            self.progress.emit("正在初始化模型...")
            self.progress_percent.emit(5)
            self.progress_detail.emit("初始化GNN网络架构...")

            n_features = self.config['num_features']
            n_layers = self.config['num_layers']

            if self._stop_requested:
                return

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

            if self._stop_requested:
                return

            self.progress.emit("开始训练...")
            self.progress_percent.emit(10)

            total_epochs = self.config['epochs']
            self.progress_detail.emit(f"总轮数: {total_epochs}")

            history = trainer.train(
                data=self.data_result['data'],
                epochs=total_epochs,
                patience=self.config['patience'],
                warmup_epochs=self.config.get('warmup_epochs', 0),
                verbose=False
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(90)
            self.progress_detail.emit("保存训练结果...")

            self.progress.emit("✓ 训练完成!")
            self.progress_percent.emit(100)
            self.finished.emit(model, history)

        except Exception as e:
            self.error.emit(f"训练失败: {str(e)}\n{traceback.format_exc()}")


class TraditionalPredictorThread(QThread):
    """传统方法拟合线程"""
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)  # 新增：百分比进度信号
    progress_detail = pyqtSignal(str)  # 新增：详细信息信号
    finished = pyqtSignal(object, dict)
    error = pyqtSignal(str)

    def __init__(self, data_result, interp_method):
        super().__init__()
        self.data_result = data_result
        self.interp_method = interp_method
        self._stop_requested = False

    def request_stop(self):
        """请求停止"""
        self._stop_requested = True

    def run(self):
        try:
            self.progress.emit("正在拟合传统模型...")
            self.progress_percent.emit(10)
            self.progress_detail.emit(f"方法: {self.interp_method}")

            if self._stop_requested:
                return

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

            if self._stop_requested:
                return

            self.progress_percent.emit(30)
            self.progress_detail.emit(f"拟合 {len(layer_order)} 个地层...")

            predictor.fit(
                raw_df,
                x_col='x',
                y_col='y',
                layer_col='layer_name',
                thickness_col='thickness'
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(60)
            self.progress_detail.emit("生成评估网格...")

            coords = self.data_result['borehole_coords']
            x_range = (coords[:, 0].min(), coords[:, 0].max())
            y_range = (coords[:, 1].min(), coords[:, 1].max())
            grid_x = np.linspace(x_range[0], x_range[1], 30)
            grid_y = np.linspace(y_range[0], y_range[1], 30)

            if self._stop_requested:
                return

            self.progress_percent.emit(80)
            self.progress_detail.emit("评估预测性能...")

            eval_metrics = evaluate_predictor(
                predictor, raw_df, grid_x, grid_y,
                x_col='x', y_col='y',
                layer_col='layer_name',
                thickness_col='thickness'
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(95)
            self.progress_detail.emit(f"R²: {eval_metrics.get('r2', 0):.3f}")

            self.progress.emit("✓ 传统方法拟合完成!")
            self.progress_percent.emit(100)
            self.finished.emit(predictor, eval_metrics)

        except Exception as e:
            self.error.emit(f"拟合失败: {str(e)}\n{traceback.format_exc()}")


class ModelingThread(QThread):
    """三维建模线程"""
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)  # 新增：百分比进度信号
    progress_detail = pyqtSignal(str)  # 新增：详细信息信号
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
        self._stop_requested = False

    def request_stop(self):
        """请求停止建模"""
        self._stop_requested = True

    def run(self):
        try:
            self.progress.emit("正在生成网格...")
            self.progress_percent.emit(5)
            self.progress_detail.emit(f"分辨率: {self.resolution}×{self.resolution}")

            if self._stop_requested:
                return

            coords = self.data_result['borehole_coords']
            x_range = (coords[:, 0].min(), coords[:, 0].max())
            y_range = (coords[:, 1].min(), coords[:, 1].max())

            grid_x = np.linspace(x_range[0], x_range[1], self.resolution)
            grid_y = np.linspace(y_range[0], y_range[1], self.resolution)

            self.progress_percent.emit(15)

            if self.use_traditional:
                self.progress.emit("正在使用传统方法预测厚度...")
                self.progress_detail.emit("采用传统插值方法...")
                thickness_grids = self.predictor.predict_grid(grid_x, grid_y)
                XI, YI = np.meshgrid(grid_x, grid_y)
                self.progress_percent.emit(50)
            else:
                self.progress.emit("正在使用GNN模型预测厚度...")
                self.progress_detail.emit("神经网络推理中...")
                model = self.predictor
                device = next(model.parameters()).device
                model.eval()
                data = self.data_result['data'].to(device)

                self.progress_percent.emit(25)

                if self._stop_requested:
                    return

                with torch.no_grad():
                    pred_thick, pred_exist = model(
                        data.x, data.edge_index,
                        data.edge_attr if hasattr(data, 'edge_attr') else None
                    )
                    pred_thick = pred_thick.cpu().numpy()
                    pred_exist = torch.sigmoid(pred_exist).cpu().numpy()

                self.progress_percent.emit(35)

                if self._stop_requested:
                    return

                from scipy.interpolate import griddata
                XI, YI = np.meshgrid(grid_x, grid_y)
                xi_flat, yi_flat = XI.flatten(), YI.flatten()

                thickness_grids = {}
                num_layers = len(self.data_result['layer_order'])
                for idx, layer_name in enumerate(self.data_result['layer_order']):
                    if self._stop_requested:
                        return

                    self.progress_detail.emit(f"处理地层: {layer_name}")

                    layer_thick = pred_thick[:, idx]
                    exist_mask = pred_exist[:, idx] > 0.5
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

                    # 每处理一层更新进度
                    progress = 35 + int((idx + 1) / num_layers * 15)
                    self.progress_percent.emit(progress)

            if self._stop_requested:
                return

            self.progress.emit("正在构建三维模型...")
            self.progress_percent.emit(55)
            self.progress_detail.emit("创建地质块体...")

            builder = GeologicalModelBuilder(
                layer_order=self.data_result['layer_order'],
                resolution=self.resolution,
                base_level=self.base_level,
                gap_value=self.gap_value
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(65)
            self.progress_detail.emit(f"生成 {len(self.data_result['layer_order'])} 个地层...")

            block_models, XI, YI = builder.build_model(
                thickness_grids=thickness_grids,
                x_range=x_range,
                y_range=y_range
            )

            if self._stop_requested:
                return

            self.progress_percent.emit(90)
            self.progress_detail.emit("模型构建完成!")

            self.progress.emit("✓ 三维模型构建完成!")
            self.progress_percent.emit(100)
            self.finished.emit(block_models, XI, YI)

        except Exception as e:
            self.error.emit(f"建模失败: {str(e)}\n{traceback.format_exc()}")


class ExportThread(QThread):
    """导出线程 - 用于F3Grid和FPN导出"""
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    progress_detail = pyqtSignal(str)
    finished = pyqtSignal(str)  # 返回输出文件路径
    error = pyqtSignal(str)

    def __init__(self, exporter, data, output_path, options, export_type="f3grid"):
        super().__init__()
        self.exporter = exporter
        self.data = data
        self.output_path = output_path
        self.options = options or {}
        self.export_type = export_type
        self._stop_requested = False

    def request_stop(self):
        """请求停止导出"""
        self._stop_requested = True

    def run(self):
        try:
            total_layers = len(self.data.get('layers', []))
            export_name = "FLAC3D" if self.export_type == "f3grid" else "FPN"

            self.progress.emit(f"正在准备{export_name}导出...")
            self.progress_percent.emit(5)
            self.progress_detail.emit(f"总地层数: {total_layers}")

            if self._stop_requested:
                return

            # 钩子导出器的内部方法来报告进度
            original_generate = self.exporter._generate_all_layers

            def generate_with_progress(*args, **kwargs):
                """带进度报告的层生成"""
                layers = args[0] if args else kwargs.get('layers', [])
                num_layers = len(layers)

                # 调用原始方法，但监听进度
                # 注意：这里简化处理，实际上原始方法会打印进度
                result = original_generate(*args, **kwargs)

                return result

            # 临时替换方法
            self.exporter._generate_all_layers = generate_with_progress

            self.progress_percent.emit(10)
            self.progress_detail.emit("开始生成网格...")

            if self._stop_requested:
                self.exporter._generate_all_layers = original_generate
                return

            # 执行导出
            import time
            start_time = time.time()

            result_path = self.exporter.export(self.data, self.output_path, self.options)

            elapsed = time.time() - start_time

            # 恢复原始方法
            self.exporter._generate_all_layers = original_generate

            if self._stop_requested:
                return

            self.progress_percent.emit(95)
            self.progress_detail.emit(f"耗时: {elapsed:.1f}秒")

            self.progress.emit(f"✓ {export_name}导出完成!")
            self.progress_percent.emit(100)
            self.finished.emit(result_path)

        except Exception as e:
            self.error.emit(f"导出失败: {str(e)}\n{traceback.format_exc()}")

