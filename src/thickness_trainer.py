"""
厚度预测训练模块 (重构版)

专门用于GNN厚度回归任务的训练和评估
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, List, Optional, Tuple, Callable
import os
import json
import time
from datetime import datetime

from .gnn_thickness_modeling import GNNThicknessPredictor, ThicknessLoss


class ThicknessTrainer:
    """
    厚度预测训练器

    专为厚度回归任务设计的训练流程
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'plateau',  # 'plateau' | 'cosine' | 'none'
        thick_weight: float = 1.0,
        exist_weight: float = 0.5,
        smooth_weight: float = 0.1
    ):
        """
        初始化

        Args:
            model: GNN厚度预测模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_type: 学习率调度类型
            thick_weight: 厚度损失权重
            exist_weight: 存在性损失权重
            smooth_weight: 平滑正则化权重
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度
        self.scheduler_type = scheduler_type
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        else:
            self.scheduler = None

        # 损失函数
        self.criterion = ThicknessLoss(
            thick_weight=thick_weight,
            exist_weight=exist_weight,
            smooth_weight=smooth_weight
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'lr': []
        }

        self.best_val_loss = float('inf')
        self.best_state = None
        self.epoch = 0

    def train_epoch(self, data: Data) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            data: PyG Data对象

        Returns:
            训练指标字典
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 前向传播
        pred_thick, pred_exist = self.model(
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )

        # 计算损失（仅在训练集上）
        train_mask = data.train_mask
        loss, loss_dict = self.criterion(
            pred_thick[train_mask],
            pred_exist[train_mask],
            data.y_thick[train_mask],
            data.y_exist[train_mask],
            data.y_mask[train_mask] if hasattr(data, 'y_mask') else None
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 计算评估指标
        with torch.no_grad():
            # 仅在存在的层上计算MAE和RMSE
            exist_mask = train_mask.unsqueeze(1) * data.y_exist.bool()
            if exist_mask.sum() > 0:
                mae = F.l1_loss(pred_thick[exist_mask], data.y_thick[exist_mask]).item()
                rmse = torch.sqrt(F.mse_loss(pred_thick[exist_mask], data.y_thick[exist_mask])).item()
            else:
                mae = 0.0
                rmse = 0.0

        return {
            'loss': loss.item(),
            'thick_loss': loss_dict['thick'],
            'exist_loss': loss_dict['exist'],
            'mae': mae,
            'rmse': rmse
        }

    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """
        评估模型

        Args:
            data: PyG Data对象
            mask: 评估掩码

        Returns:
            评估指标字典
        """
        self.model.eval()

        # 前向传播
        pred_thick, pred_exist = self.model(
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )

        # 计算损失
        loss, loss_dict = self.criterion(
            pred_thick[mask],
            pred_exist[mask],
            data.y_thick[mask],
            data.y_exist[mask],
            data.y_mask[mask] if hasattr(data, 'y_mask') else None
        )

        # 计算评估指标
        exist_mask = mask.unsqueeze(1) * data.y_exist.bool()
        if exist_mask.sum() > 0:
            mae = F.l1_loss(pred_thick[exist_mask], data.y_thick[exist_mask]).item()
            rmse = torch.sqrt(F.mse_loss(pred_thick[exist_mask], data.y_thick[exist_mask])).item()

            # 计算R²
            ss_res = ((pred_thick[exist_mask] - data.y_thick[exist_mask]) ** 2).sum()
            ss_tot = ((data.y_thick[exist_mask] - data.y_thick[exist_mask].mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2 = r2.item()
        else:
            mae = 0.0
            rmse = 0.0
            r2 = 0.0

        # 存在性分类准确率
        pred_exist_binary = (torch.sigmoid(pred_exist) > 0.5).float()
        exist_acc = (pred_exist_binary[mask] == data.y_exist[mask]).float().mean().item()

        return {
            'loss': loss.item(),
            'thick_loss': loss_dict['thick'],
            'exist_loss': loss_dict['exist'],
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'exist_acc': exist_acc
        }

    def train(
        self,
        data: Data,
        epochs: int = 200,
        patience: int = 30,
        verbose: bool = True,
        log_interval: int = 10,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        完整训练流程

        Args:
            data: PyG Data对象
            epochs: 训练轮数
            patience: 早停耐心值
            verbose: 是否打印日志
            log_interval: 日志打印间隔
            save_dir: 模型保存目录

        Returns:
            训练历史
        """
        # 移动数据到设备
        data = data.to(self.device)

        patience_counter = 0
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"开始训练 - 设备: {self.device}")
            print(f"训练集: {data.train_mask.sum().item()}, "
                  f"验证集: {data.val_mask.sum().item()}, "
                  f"测试集: {data.test_mask.sum().item()}")
            print(f"{'='*60}")

        for epoch in range(epochs):
            self.epoch = epoch + 1

            # 训练
            train_metrics = self.train_epoch(data)

            # 验证
            val_metrics = self.evaluate(data, data.val_mask)

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['lr'].append(current_lr)

            # 早停检查
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0

                # 保存最佳模型
                if save_dir:
                    self.save_checkpoint(save_dir, 'best_model.pt')
            else:
                patience_counter += 1

            # 打印日志
            if verbose and (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val MAE: {val_metrics['mae']:.3f}m | "
                      f"Val R²: {val_metrics['r2']:.3f} | "
                      f"LR: {current_lr:.2e}")

            # 早停
            if patience_counter >= patience:
                if verbose:
                    print(f"\n早停触发: {patience}轮验证损失未改善")
                break

        # 恢复最佳模型
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # 最终评估
        elapsed_time = time.time() - start_time
        test_metrics = self.evaluate(data, data.test_mask)

        if verbose:
            print(f"\n{'='*60}")
            print(f"训练完成 - 耗时: {elapsed_time:.1f}秒")
            print(f"最佳验证损失: {self.best_val_loss:.4f}")
            print(f"\n测试集评估:")
            print(f"  MAE: {test_metrics['mae']:.3f}m")
            print(f"  RMSE: {test_metrics['rmse']:.3f}m")
            print(f"  R²: {test_metrics['r2']:.3f}")
            print(f"  存在性准确率: {test_metrics['exist_acc']*100:.1f}%")
            print(f"{'='*60}\n")

        # 添加最终测试结果
        self.history['test_metrics'] = test_metrics

        return self.history

    def save_checkpoint(self, save_dir: str, filename: str = 'checkpoint.pt'):
        """保存检查点"""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']


class ThicknessEvaluator:
    """
    厚度预测评估器

    提供详细的模型评估和分析功能
    """

    def __init__(self, model: nn.Module, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测厚度

        Returns:
            pred_thick: 预测厚度 [N, L]
            pred_exist: 预测存在概率 [N, L]
        """
        data = data.to(self.device)
        pred_thick, pred_exist = self.model(
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )
        pred_thick = pred_thick.cpu().numpy()
        pred_exist = torch.sigmoid(pred_exist).cpu().numpy()
        return pred_thick, pred_exist

    @torch.no_grad()
    def evaluate_by_layer(
        self,
        data: Data,
        layer_order: List[str],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        按层评估

        Returns:
            每层的评估指标
        """
        data = data.to(self.device)
        if mask is None:
            mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=self.device)

        pred_thick, pred_exist = self.model(
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )

        results = {}
        for i, layer_name in enumerate(layer_order):
            # 只评估该层存在的样本
            layer_exist = data.y_exist[:, i].bool() & mask
            if layer_exist.sum() == 0:
                results[layer_name] = {
                    'mae': np.nan,
                    'rmse': np.nan,
                    'count': 0
                }
                continue

            pred = pred_thick[layer_exist, i]
            target = data.y_thick[layer_exist, i]

            mae = F.l1_loss(pred, target).item()
            rmse = torch.sqrt(F.mse_loss(pred, target)).item()

            # 相对误差
            mape = (torch.abs(pred - target) / (target + 1e-8)).mean().item() * 100

            results[layer_name] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'count': layer_exist.sum().item(),
                'mean_true': target.mean().item(),
                'mean_pred': pred.mean().item()
            }

        return results

    def generate_report(
        self,
        data: Data,
        layer_order: List[str],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告

        Returns:
            报告文本
        """
        # 按层评估
        layer_metrics = self.evaluate_by_layer(data, layer_order, data.test_mask)

        # 整体预测
        pred_thick, pred_exist = self.predict(data)

        # 生成报告
        report_lines = [
            "=" * 60,
            "GNN厚度预测模型评估报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "一、各层评估指标",
            "-" * 60,
            f"{'层名':<15} {'样本数':<8} {'MAE(m)':<10} {'RMSE(m)':<10} {'MAPE(%)':<10}",
            "-" * 60
        ]

        total_mae = []
        total_count = 0
        for layer_name, metrics in layer_metrics.items():
            if metrics['count'] > 0:
                report_lines.append(
                    f"{layer_name:<15} {metrics['count']:<8} "
                    f"{metrics['mae']:<10.3f} {metrics['rmse']:<10.3f} "
                    f"{metrics['mape']:<10.1f}"
                )
                total_mae.append(metrics['mae'])
                total_count += metrics['count']

        report_lines.extend([
            "-" * 60,
            f"{'平均':<15} {total_count:<8} {np.mean(total_mae):<10.3f}",
            "",
            "二、预测统计",
            "-" * 60,
        ])

        # 预测统计
        test_mask = data.test_mask.numpy()
        for i, layer_name in enumerate(layer_order):
            exist_mask = data.y_exist[:, i].numpy().astype(bool) & test_mask
            if exist_mask.sum() > 0:
                true_thick = data.y_thick[:, i].numpy()[exist_mask]
                pred = pred_thick[exist_mask, i]
                report_lines.append(
                    f"{layer_name}: "
                    f"真实厚度 {true_thick.mean():.2f}±{true_thick.std():.2f}m, "
                    f"预测厚度 {pred.mean():.2f}±{pred.std():.2f}m"
                )

        report_lines.extend([
            "",
            "=" * 60
        ])

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report


def create_trainer(
    num_features: int,
    num_layers: int,
    hidden_channels: int = 128,
    gnn_layers: int = 3,
    dropout: float = 0.2,
    conv_type: str = 'gatv2',
    device: str = 'auto',
    learning_rate: float = 0.001
) -> Tuple[GNNThicknessPredictor, ThicknessTrainer]:
    """
    创建模型和训练器的便捷函数

    Args:
        num_features: 输入特征维度
        num_layers: 输出层数（地层数量）
        hidden_channels: 隐藏层维度
        gnn_layers: GNN层数
        dropout: dropout比率
        conv_type: 卷积类型
        device: 计算设备
        learning_rate: 学习率

    Returns:
        (model, trainer) 元组
    """
    model = GNNThicknessPredictor(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        num_layers=gnn_layers,
        num_output_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type
    )

    trainer = ThicknessTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )

    return model, trainer


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("测试厚度预测训练模块...")

    # 模拟数据
    np.random.seed(42)
    num_nodes = 30
    num_features = 4
    num_layers = 5

    # 模拟特征和目标
    x = torch.randn(num_nodes, num_features)
    y_thick = torch.abs(torch.randn(num_nodes, num_layers)) * 3 + 1
    y_exist = (torch.rand(num_nodes, num_layers) > 0.3).float()
    y_thick = y_thick * y_exist

    # 构建简单的边
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, min(i+5, num_nodes)):
            edges.append([i, j])
            edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # 创建Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.shape[1], 1),
        y_thick=y_thick,
        y_exist=y_exist,
        y_mask=torch.ones(num_nodes, num_layers),
        train_mask=torch.zeros(num_nodes, dtype=torch.bool),
        val_mask=torch.zeros(num_nodes, dtype=torch.bool),
        test_mask=torch.zeros(num_nodes, dtype=torch.bool)
    )
    data.train_mask[:20] = True
    data.val_mask[20:25] = True
    data.test_mask[25:] = True

    # 创建训练器
    model, trainer = create_trainer(
        num_features=num_features,
        num_layers=num_layers,
        hidden_channels=64,
        gnn_layers=2
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    history = trainer.train(data, epochs=50, log_interval=10, verbose=True)

    # 评估
    evaluator = ThicknessEvaluator(model)
    layer_order = [f'Layer_{i}' for i in range(num_layers)]
    report = evaluator.generate_report(data, layer_order)
    print(report)
