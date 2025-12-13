"""
训练模块
包含训练循环、验证、评估和模型保存功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, Optional, Tuple, List, Callable
import os
import json
from datetime import datetime
from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    Focal Loss - 专门处理类别不平衡问题
    对于难分类的样本给予更高的权重
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失 - 防止过拟合，提高泛化能力
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # 计算损失
        if self.weight is not None:
            weight = self.weight[targets]
            loss = (-true_dist * log_probs).sum(dim=-1) * weight
        else:
            loss = (-true_dist * log_probs).sum(dim=-1)

        return loss.mean()


class GeoModelTrainer:
    """
    地质模型训练器
    负责模型训练、验证、评估和保存
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'plateau',
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = 'focal',  # 'ce', 'focal', 'label_smoothing'
        num_classes: int = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        """
        初始化训练器

        Args:
            model: GNN模型
            device: 计算设备 ('cuda', 'cpu', 'auto')
            learning_rate: 学习率
            weight_decay: 权重衰减 (L2正则化)
            optimizer_type: 优化器类型 ('adam', 'adamw')
            scheduler_type: 学习率调度器类型 ('plateau', 'cosine', 'onecycle', 'none')
            class_weights: 类别权重 (用于处理类别不平衡)
            loss_type: 损失函数类型 ('ce', 'focal', 'label_smoothing')
            num_classes: 类别数量 (label_smoothing需要)
            focal_gamma: Focal Loss的gamma参数
            label_smoothing: 标签平滑系数
        """
        # 设备设置
        if device == 'auto':
            if torch.cuda.is_available():
                try:
                    # 测试 CUDA 是否真正可用（兼容性检查）
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    self.device = torch.device('cuda')
                except Exception as e:
                    print(f"CUDA 不兼容当前 PyTorch 版本，回退到 CPU: {e}")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 模型
        self.model = model.to(self.device)

        # 优化器 - 使用AdamW作为默认，更好的正则化效果
        if optimizer_type == 'adam':
            self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"未知优化器: {optimizer_type}")

        # 学习率调度器
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        else:
            self.scheduler = None

        # 损失函数
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        self.loss_type = loss_type
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif loss_type == 'label_smoothing' and num_classes is not None:
            self.criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing, weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }

        # 最佳模型状态
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_model_state = None

    def train_epoch(self, data: Data) -> Tuple[float, float]:
        """
        训练一个epoch

        Args:
            data: PyG Data对象

        Returns:
            loss: 训练损失
            accuracy: 训练准确率
        """
        self.model.train()
        data = data.to(self.device)

        self.optimizer.zero_grad()

        # 前向传播
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            out = self.model(data.x, data.edge_index, data.edge_weight)
        else:
            out = self.model(data.x, data.edge_index)

        # 计算损失 (只在训练集上)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

        # 反向传播
        loss.backward()
        self.optimizer.step()

        # 计算准确率
        pred = out.argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
        accuracy = correct / data.train_mask.sum().item()

        return loss.item(), accuracy

    @torch.no_grad()
    def validate(self, data: Data) -> Tuple[float, float, float]:
        """
        在验证集上评估模型

        Args:
            data: PyG Data对象

        Returns:
            loss: 验证损失
            accuracy: 验证准确率
            f1: 验证F1分数 (macro)
        """
        self.model.eval()
        data = data.to(self.device)

        # 前向传播
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            out = self.model(data.x, data.edge_index, data.edge_weight)
        else:
            out = self.model(data.x, data.edge_index)

        # 计算损失
        loss = self.criterion(out[data.val_mask], data.y[data.val_mask]).item()

        # 计算准确率
        pred = out.argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
        accuracy = correct / data.val_mask.sum().item()

        # 计算F1分数
        y_true = data.y[data.val_mask].cpu().numpy()
        y_pred = pred[data.val_mask].cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='macro')

        return loss, accuracy, f1

    def train(
        self,
        data: Data,
        epochs: int = 200,
        patience: int = 30,
        min_delta: float = 1e-4,
        verbose: bool = True,
        save_path: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        完整训练流程

        Args:
            data: PyG Data对象
            epochs: 最大训练轮数
            patience: 早停耐心值
            min_delta: 最小改进阈值
            verbose: 是否打印训练过程
            save_path: 模型保存路径
            callback: 每个epoch后的回调函数 callback(epoch, train_loss, val_loss, val_acc)

        Returns:
            history: 训练历史字典
        """
        best_val_loss = float('inf')
        patience_counter = 0

        pbar = tqdm(range(epochs), desc="训练中") if verbose else range(epochs)

        for epoch in pbar:
            # 训练
            train_loss, train_acc = self.train_epoch(data)

            # 验证
            val_loss, val_acc, val_f1 = self.validate(data)

            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            # 更新学习率
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_acc)  # 使用准确率而非损失
            elif self.scheduler_type == 'cosine':
                self.scheduler.step()

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 早停检查
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n早停触发于epoch {epoch + 1}")
                    break

            # 更新进度条
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'val_f1': f'{val_f1:.4f}'
                })

            # 回调
            if callback:
                callback(epoch, train_loss, val_loss, val_acc)

        # 恢复最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        # 保存模型
        if save_path:
            self.save_model(save_path)

        print(f"\n训练完成! 最佳验证准确率: {self.best_val_acc:.4f}, F1: {self.best_val_f1:.4f}")

        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        data: Data,
        lithology_classes: Optional[List[str]] = None
    ) -> Dict:
        """
        在测试集上评估模型

        Args:
            data: PyG Data对象
            lithology_classes: 岩性类别名称列表

        Returns:
            results: 评估结果字典
        """
        self.model.eval()
        data = data.to(self.device)

        # 前向传播
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            out = self.model(data.x, data.edge_index, data.edge_weight)
        else:
            out = self.model(data.x, data.edge_index)

        # 获取预测和概率
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        # 测试集指标
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        y_probs = probs[data.test_mask].cpu().numpy()

        # 准确率
        accuracy = (y_pred == y_true).mean()

        # F1分数
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 分类报告
        target_names = lithology_classes if lithology_classes else [f"Class_{i}" for i in range(cm.shape[0])]
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_probs,
            'true_labels': y_true
        }

        # 打印报告
        print("\n========== 测试集评估结果 ==========")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=target_names))

        return results

    @torch.no_grad()
    def predict(
        self,
        data: Data,
        return_probs: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        对所有节点进行预测

        Args:
            data: PyG Data对象
            return_probs: 是否返回概率

        Returns:
            predictions: 预测的类别索引
            probabilities: 预测概率 (可选)
        """
        self.model.eval()
        data = data.to(self.device)

        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            out = self.model(data.x, data.edge_index, data.edge_weight)
        else:
            out = self.model(data.x, data.edge_index)

        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        predictions = pred.cpu().numpy()
        probabilities = probs.cpu().numpy() if return_probs else None

        return predictions, probabilities

    def save_model(self, path: str):
        """保存模型和训练状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1
        }

        torch.save(save_dict, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path: str):
        """加载模型和训练状态"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        print(f"模型已从 {path} 加载")


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    计算类别权重 (处理类别不平衡)

    Args:
        labels: 标签张量

    Returns:
        weights: 类别权重张量
    """
    class_counts = torch.bincount(labels)
    total = labels.size(0)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from models import get_model
    from data_loader import BoreholeDataProcessor

    print("测试训练模块...")

    # 准备数据
    processor = BoreholeDataProcessor(k_neighbors=6)
    df = processor.create_sample_data(num_boreholes=20, points_per_borehole=10)
    result = processor.process(df)

    data = result['data']
    num_features = result['num_features']
    num_classes = result['num_classes']

    # 创建模型
    model = get_model(
        'graphsage',
        in_channels=num_features,
        hidden_channels=32,
        out_channels=num_classes,
        num_layers=2
    )

    # 计算类别权重
    class_weights = compute_class_weights(data.y)

    # 创建训练器
    trainer = GeoModelTrainer(
        model=model,
        learning_rate=0.01,
        weight_decay=5e-4,
        class_weights=class_weights
    )

    # 训练
    history = trainer.train(data, epochs=50, patience=20, verbose=True)

    # 评估
    results = trainer.evaluate(data, result['lithology_classes'])

    # 预测
    predictions, probs = trainer.predict(data, return_probs=True)
    print(f"\n预测结果形状: {predictions.shape}")
    print(f"概率形状: {probs.shape}")
