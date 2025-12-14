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
import logging
import gc

# 尝试导入数据增强模块
try:
    from .augmentation import GraphAugmentation, FeatureMixup
except ImportError:
    try:
        from augmentation import GraphAugmentation, FeatureMixup
    except ImportError:
        GraphAugmentation = None
        FeatureMixup = None

class EMA:
    """
    指数移动平均 (Exponential Moving Average)
    用于平滑模型参数，减少训练波动，提高泛化能力
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        初始化EMA

        Args:
            model: 需要平滑的模型
            decay: 衰减率，越接近1平滑效果越强
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数到模型（用于验证/测试）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数（验证/测试后恢复训练）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


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
        label_smoothing: float = 0.1,
        use_augmentation: bool = False,
        augment_noise_std: float = 0.05,
        augment_edge_drop: float = 0.1,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        use_ema: bool = True,
        ema_decay: float = 0.995
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
            use_augmentation: 是否使用数据增强
            augment_noise_std: 节点特征噪声标准差
            augment_edge_drop: 边丢弃概率
            use_mixup: 是否使用Mixup
            mixup_alpha: Mixup的alpha参数
            use_ema: 是否使用EMA平滑训练
            ema_decay: EMA衰减率（越接近1平滑效果越强）
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

        # 学习率调度器 - 优化：使用稳定的调度策略
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate
        if scheduler_type == 'plateau':
            # 稳定策略：基于验证准确率调整，patience适中
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=40, min_lr=1e-6, verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        elif scheduler_type == 'step':
            # 简单的阶梯式衰减
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.5
            )
        elif scheduler_type == 'onecycle':
            # OneCycleLR - 需要知道总步数，暂时不在这里初始化
            self.scheduler = None
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

        # 设置日志
        self.logger = logging.getLogger('GeoModelTrainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            # 确保output目录存在
            os.makedirs('output', exist_ok=True)
            fh = logging.FileHandler('output/training.log', mode='w', encoding='utf-8')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # 数据增强
        self.use_augmentation = use_augmentation
        self.use_mixup = use_mixup
        if use_augmentation and GraphAugmentation is not None:
            self.augmentation = GraphAugmentation(
                node_noise_std=augment_noise_std,
                edge_drop_rate=augment_edge_drop
            )
            print(f"已启用数据增强: noise_std={augment_noise_std}, edge_drop={augment_edge_drop}")
        else:
            self.augmentation = None

        if use_mixup and FeatureMixup is not None:
            self.mixup = FeatureMixup(alpha=mixup_alpha)
            print(f"已启用Mixup: alpha={mixup_alpha}")
        else:
            self.mixup = None

        # EMA初始化
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"已启用EMA: decay={ema_decay}")
        else:
            self.ema = None

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

        # 应用数据增强
        if self.augmentation is not None:
            aug_data = self.augmentation.augment(data, training=True)
        else:
            aug_data = data

        self.optimizer.zero_grad()

        # 前向传播
        if hasattr(aug_data, 'edge_weight') and aug_data.edge_weight is not None:
            out = self.model(aug_data.x, aug_data.edge_index, aug_data.edge_weight)
        else:
            out = self.model(aug_data.x, aug_data.edge_index)

        # Mixup训练
        if self.mixup is not None:
            mixed_x, y_a, y_b, lam = self.mixup(aug_data.x, data.y, data.train_mask)
            # 用混合特征重新前向传播
            if hasattr(aug_data, 'edge_weight') and aug_data.edge_weight is not None:
                out_mixed = self.model(mixed_x, aug_data.edge_index, aug_data.edge_weight)
            else:
                out_mixed = self.model(mixed_x, aug_data.edge_index)
            # 混合损失
            loss = lam * self.criterion(out_mixed[data.train_mask], y_a) + \
                   (1 - lam) * self.criterion(out_mixed[data.train_mask], y_b)
        else:
            # 计算损失 (只在训练集上)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

        # 反向传播
        loss.backward()
        
        # 优化：梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 更新EMA
        if self.ema is not None:
            self.ema.update()

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

        # 使用EMA参数进行验证
        if self.ema is not None:
            self.ema.apply_shadow()

        try:
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
        finally:
            # 恢复原始参数
            if self.ema is not None:
                self.ema.restore()

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

            # 记录日志
            self.logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.6f}")

            # 更新学习率
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_acc)  # 使用准确率而非损失
            elif self.scheduler_type in ['cosine', 'step']:
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

        # 使用EMA参数进行评估
        if self.ema is not None:
            self.ema.apply_shadow()

        try:
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
        finally:
            # 恢复原始参数
            if self.ema is not None:
                self.ema.restore()

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

        # 使用EMA参数进行预测
        if self.ema is not None:
            self.ema.apply_shadow()

        try:
            if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                out = self.model(data.x, data.edge_index, data.edge_weight)
            else:
                out = self.model(data.x, data.edge_index)

            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)

            predictions = pred.cpu().numpy()
            probabilities = probs.cpu().numpy() if return_probs else None
        finally:
            # 恢复原始参数
            if self.ema is not None:
                self.ema.restore()

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

    def cleanup(self, data: Optional[Data] = None) -> Optional[Data]:
        """释放GPU资源，只保留结果。"""
        # 把模型和优化器状态搬回CPU，避免显存残留
        self.model = self.model.cpu()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

        # 可选地把数据也搬回CPU并返回，方便调用方继续用
        if data is not None:
            data = data.cpu()

        torch.cuda.empty_cache()
        gc.collect()
        return data


def compute_class_weights(labels: torch.Tensor, method: str = 'effective') -> torch.Tensor:
    """
    计算类别权重 (处理类别不平衡)

    Args:
        labels: 标签张量
        method: 权重计算方法
            - 'balanced': sklearn风格的平衡权重
            - 'effective': 有效样本数方法 (对极端不平衡更有效) - 默认推荐
            - 'sqrt': 开方平衡，温和处理不平衡
            - 'log': 对数平衡，更温和

    Returns:
        weights: 类别权重张量
    """
    # 确保标签是Long类型
    labels = labels.long()

    # 获取最大类别索引
    num_classes = labels.max().item() + 1

    # 计算每个类别的数量
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # 避免除以零
    class_counts = torch.clamp(class_counts, min=1.0)

    total = labels.size(0)

    if method == 'balanced':
        # sklearn风格: n_samples / (n_classes * n_samples_per_class)
        weights = total / (num_classes * class_counts)
    elif method == 'effective':
        # 有效样本数方法 (适合极端不平衡) - 推荐
        beta = 0.999  # 稍微降低beta，使权重更温和
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    elif method == 'sqrt':
        # 开方平衡 (温和处理)
        weights = torch.sqrt(total / class_counts)
    elif method == 'log':
        # 对数平衡 (更温和)
        weights = torch.log1p(total / class_counts)
    else:
        weights = total / (num_classes * class_counts)

    # 归一化权重，使其均值为1
    weights = weights / weights.mean()

    # 限制最大权重，防止某些类别权重过大导致不稳定
    max_weight = 5.0  # 降低最大权重，更稳定
    weights = torch.clamp(weights, max=max_weight)
    weights = weights / weights.mean()  # 再次归一化

    return weights


class EnsembleTrainer:
    """
    模型集成训练器
    通过训练多个模型并融合预测结果来提高准确率
    """

    def __init__(
        self,
        model_configs: List[dict],
        device: str = 'auto',
        voting: str = 'soft'  # 'soft' (概率平均) 或 'hard' (投票)
    ):
        """
        初始化集成训练器

        Args:
            model_configs: 模型配置列表，每个配置包含model和trainer参数
            device: 计算设备
            voting: 集成策略 ('soft' 或 'hard')
        """
        self.model_configs = model_configs
        self.voting = voting
        self.trainers = []

        # 设备设置
        if device == 'auto':
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    self.device = torch.device('cuda')
                except Exception:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

    def train_all(
        self,
        data: Data,
        epochs: int = 200,
        patience: int = 50,
        verbose: bool = True
    ) -> List[Dict]:
        """
        训练所有模型

        Args:
            data: PyG Data对象
            epochs: 训练轮数
            patience: 早停耐心值
            verbose: 是否打印进度

        Returns:
            histories: 所有模型的训练历史
        """
        histories = []

        for i, config in enumerate(self.model_configs):
            if verbose:
                print(f"\n{'='*50}")
                print(f"训练模型 {i+1}/{len(self.model_configs)}: {config.get('name', f'Model_{i}')}")
                print(f"{'='*50}")

            model = config['model']
            trainer_params = config.get('trainer_params', {})

            trainer = GeoModelTrainer(
                model=model,
                device=str(self.device),
                **trainer_params
            )

            history = trainer.train(
                data,
                epochs=epochs,
                patience=patience,
                verbose=verbose
            )

            self.trainers.append(trainer)
            histories.append(history)

        return histories

    @torch.no_grad()
    def predict(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用所有模型进行集成预测

        Args:
            data: PyG Data对象

        Returns:
            predictions: 集成预测结果
            probabilities: 集成概率
        """
        all_probs = []

        for trainer in self.trainers:
            _, probs = trainer.predict(data, return_probs=True)
            all_probs.append(probs)

        # 堆叠所有模型的预测概率
        stacked_probs = np.stack(all_probs, axis=0)  # [num_models, num_nodes, num_classes]

        if self.voting == 'soft':
            # 软投票：概率平均
            ensemble_probs = np.mean(stacked_probs, axis=0)
            predictions = np.argmax(ensemble_probs, axis=1)
        else:
            # 硬投票：多数投票
            all_preds = np.argmax(stacked_probs, axis=2)  # [num_models, num_nodes]
            predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 0, all_preds
            )
            ensemble_probs = np.mean(stacked_probs, axis=0)

        return predictions, ensemble_probs

    def evaluate(
        self,
        data: Data,
        lithology_classes: Optional[List[str]] = None
    ) -> Dict:
        """
        评估集成模型

        Args:
            data: PyG Data对象
            lithology_classes: 岩性类别名称列表

        Returns:
            results: 评估结果
        """
        predictions, probs = self.predict(data)

        # 获取测试集标签
        test_mask = data.test_mask.cpu().numpy()
        y_true = data.y.cpu().numpy()[test_mask]
        y_pred = predictions[test_mask]
        y_probs = probs[test_mask]

        # 计算指标
        accuracy = (y_pred == y_true).mean()
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)

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

        print(f"\n{'='*50}")
        print("集成模型评估结果")
        print(f"{'='*50}")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")

        # 对比单个模型
        print(f"\n单个模型性能对比:")
        for i, trainer in enumerate(self.trainers):
            print(f"  模型 {i+1}: Val Acc = {trainer.best_val_acc:.4f}, Val F1 = {trainer.best_val_f1:.4f}")

        return results


def create_ensemble_configs(
    num_features: int,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    num_models: int = 3
) -> List[dict]:
    """
    创建集成模型配置

    Args:
        num_features: 输入特征数
        num_classes: 输出类别数
        class_weights: 类别权重
        num_models: 模型数量

    Returns:
        configs: 模型配置列表
    """
    # 尝试导入模型
    try:
        from .models import get_model
    except ImportError:
        from models import get_model

    configs = []

    # 配置1: EnhancedGeoGNN (GATv2)
    if num_models >= 1:
        model1 = get_model(
            'enhanced',
            in_channels=num_features,
            hidden_channels=256,
            out_channels=num_classes,
            num_layers=3,
            dropout=0.2,
            heads=6
        )
        configs.append({
            'name': 'EnhancedGeoGNN_256',
            'model': model1,
            'trainer_params': {
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'class_weights': class_weights,
                'loss_type': 'focal',
                'num_classes': num_classes,
                'focal_gamma': 2.0,
                'use_augmentation': True,
                'augment_noise_std': 0.03,
                'augment_edge_drop': 0.05
            }
        })

    # 配置2: GraphSAGE
    if num_models >= 2:
        model2 = get_model(
            'graphsage',
            in_channels=num_features,
            hidden_channels=192,
            out_channels=num_classes,
            num_layers=4,
            dropout=0.15
        )
        configs.append({
            'name': 'GraphSAGE_192',
            'model': model2,
            'trainer_params': {
                'learning_rate': 0.0008,
                'weight_decay': 5e-5,
                'class_weights': class_weights,
                'loss_type': 'focal',
                'num_classes': num_classes,
                'focal_gamma': 1.5,
                'use_augmentation': True,
                'augment_noise_std': 0.02,
                'augment_edge_drop': 0.03
            }
        })

    # 配置3: GAT
    if num_models >= 3:
        model3 = get_model(
            'gat',
            in_channels=num_features,
            hidden_channels=128,
            out_channels=num_classes,
            num_layers=3,
            dropout=0.25,
            heads=8
        )
        configs.append({
            'name': 'GAT_128',
            'model': model3,
            'trainer_params': {
                'learning_rate': 0.0012,
                'weight_decay': 1e-4,
                'class_weights': class_weights,
                'loss_type': 'focal',
                'num_classes': num_classes,
                'focal_gamma': 2.5,
                'use_augmentation': True,
                'augment_noise_std': 0.04,
                'augment_edge_drop': 0.08
            }
        })

    return configs


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
