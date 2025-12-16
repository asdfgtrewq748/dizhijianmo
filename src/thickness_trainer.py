"""
厚度预测训练模块 (重构版)

专门用于GNN厚度回归任务的训练和评估
包含数据增强、K-fold交叉验证等优化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Optional, Tuple, Callable, Any
import os
import json
import time
from datetime import datetime
from sklearn.model_selection import KFold

from .gnn_thickness_modeling import GNNThicknessPredictor, ThicknessLoss


# =============================================================================
# 数据增强
# =============================================================================

class ThicknessDataAugmentation:
    """
    厚度预测数据增强

    针对小样本地质数据的增强策略
    """

    @staticmethod
    def add_noise(data: Data, noise_std: float = 0.05) -> Data:
        """
        添加高斯噪声到特征

        Args:
            data: 原始数据
            noise_std: 噪声标准差（相对于特征标准差）
        """
        data = data.clone()
        noise = torch.randn_like(data.x) * noise_std
        data.x = data.x + noise
        return data

    @staticmethod
    def add_thickness_noise(data: Data, noise_std: float = 0.1) -> Data:
        """
        添加厚度噪声（用于训练时增强鲁棒性）

        Args:
            data: 原始数据
            noise_std: 噪声标准差（相对于厚度）
        """
        data = data.clone()
        # 只对存在的层添加噪声
        noise = torch.randn_like(data.y_thick) * noise_std
        noise = noise * data.y_exist  # 只对存在的层添加
        data.y_thick = torch.clamp(data.y_thick + noise * data.y_thick, min=0.1)
        return data

    @staticmethod
    def random_dropout_features(data: Data, dropout_rate: float = 0.1) -> Data:
        """
        随机丢弃部分特征

        Args:
            data: 原始数据
            dropout_rate: 特征丢弃率
        """
        data = data.clone()
        mask = torch.rand_like(data.x) > dropout_rate
        data.x = data.x * mask
        return data

    @staticmethod
    def mixup(data: Data, alpha: float = 0.2) -> Data:
        """
        Mixup数据增强

        Args:
            data: 原始数据
            alpha: Beta分布参数
        """
        data = data.clone()
        n = data.x.shape[0]

        # 随机排列
        perm = torch.randperm(n)

        # Beta分布采样混合比例
        lam = np.random.beta(alpha, alpha)

        # 混合特征和标签
        data.x = lam * data.x + (1 - lam) * data.x[perm]
        data.y_thick = lam * data.y_thick + (1 - lam) * data.y_thick[perm]
        data.y_exist = torch.clamp(data.y_exist + data.y_exist[perm], max=1.0)

        return data

    @staticmethod
    def interpolate_virtual_boreholes(data: Data, n_virtual: int = 10) -> Data:
        """
        插值生成虚拟钻孔

        通过在现有钻孔之间插值创建新的训练样本

        Args:
            data: 原始数据
            n_virtual: 虚拟钻孔数量
        """
        data = data.clone()
        n_original = data.x.shape[0]

        if n_original < 3:
            return data

        # 生成虚拟钻孔
        virtual_x = []
        virtual_y_thick = []
        virtual_y_exist = []

        for _ in range(n_virtual):
            # 随机选择两个钻孔进行插值
            idx1, idx2 = np.random.choice(n_original, 2, replace=False)

            # 随机插值比例
            t = np.random.uniform(0.2, 0.8)

            # 插值特征
            x_new = t * data.x[idx1] + (1 - t) * data.x[idx2]

            # 插值厚度（仅对两个钻孔都存在的层）
            both_exist = data.y_exist[idx1] * data.y_exist[idx2]
            y_thick_new = t * data.y_thick[idx1] + (1 - t) * data.y_thick[idx2]
            y_thick_new = y_thick_new * both_exist

            virtual_x.append(x_new)
            virtual_y_thick.append(y_thick_new)
            virtual_y_exist.append(both_exist)

        # 合并
        virtual_x = torch.stack(virtual_x)
        virtual_y_thick = torch.stack(virtual_y_thick)
        virtual_y_exist = torch.stack(virtual_y_exist)

        data.x = torch.cat([data.x, virtual_x], dim=0)
        data.y_thick = torch.cat([data.y_thick, virtual_y_thick], dim=0)
        data.y_exist = torch.cat([data.y_exist, virtual_y_exist], dim=0)

        # 更新边（简单添加到现有边）
        # 为虚拟钻孔添加到最近钻孔的边
        from scipy.spatial import KDTree

        # 获取坐标（假设前两个特征是坐标）
        coords = data.x[:n_original, :2].numpy()
        virtual_coords = data.x[n_original:, :2].numpy()

        tree = KDTree(coords)
        new_edges = []

        for i, vc in enumerate(virtual_coords):
            # 找最近的3个钻孔
            _, indices = tree.query(vc, k=min(3, n_original))
            for j in indices:
                new_edges.append([n_original + i, j])
                new_edges.append([j, n_original + i])

        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)

            # 更新边属性
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                new_edge_attr = torch.ones(len(new_edges), data.edge_attr.shape[1])
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

        # 更新掩码
        n_total = data.x.shape[0]
        new_train_mask = torch.zeros(n_total, dtype=torch.bool)
        new_train_mask[:n_original] = data.train_mask
        new_train_mask[n_original:] = True  # 虚拟钻孔都用于训练
        data.train_mask = new_train_mask

        new_val_mask = torch.zeros(n_total, dtype=torch.bool)
        new_val_mask[:n_original] = data.val_mask
        data.val_mask = new_val_mask

        new_test_mask = torch.zeros(n_total, dtype=torch.bool)
        new_test_mask[:n_original] = data.test_mask
        data.test_mask = new_test_mask

        # 更新 y_mask
        if hasattr(data, 'y_mask'):
            new_y_mask = torch.ones(n_total, data.y_thick.shape[1])
            new_y_mask[:n_original] = data.y_mask
            data.y_mask = new_y_mask

        return data


class ThicknessTrainer:
    """
    厚度预测训练器

    专为厚度回归任务设计的训练流程

    针对小样本数据的优化策略：
    - 数据增强：噪声注入、mixup、虚拟钻孔插值
    - K-fold交叉验证：更稳定的评估
    - 梯度累积：模拟更大batch size
    - 标签平滑：防止过拟合
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'plateau',  # 'plateau' | 'cosine' | 'onecycle' | 'none'
        thick_weight: float = 1.0,
        exist_weight: float = 0.5,
        smooth_weight: float = 0.1,
        use_augmentation: bool = True,
        augmentation_prob: float = 0.5
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
            use_augmentation: 是否使用数据增强
            augmentation_prob: 数据增强应用概率
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # 数据增强设置
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        self.augmenter = ThicknessDataAugmentation()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度
        self.scheduler_type = scheduler_type
        self.scheduler = None  # 在train方法中根据epochs初始化

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

    def _init_scheduler(self, epochs: int):
        """初始化学习率调度器"""
        if self.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=15,
                min_lr=1e-6
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif self.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                total_steps=epochs,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            self.scheduler = None

    def _apply_augmentation(self, data: Data) -> Data:
        """应用数据增强"""
        if not self.use_augmentation:
            return data

        if np.random.random() < self.augmentation_prob:
            # 随机选择增强方法
            aug_methods = [
                lambda d: self.augmenter.add_noise(d, noise_std=0.03),
                lambda d: self.augmenter.random_dropout_features(d, dropout_rate=0.1),
                lambda d: self.augmenter.mixup(d, alpha=0.2),
            ]
            aug_fn = np.random.choice(aug_methods)
            try:
                data = aug_fn(data)
            except Exception:
                pass  # 增强失败时使用原数据

        return data

    def train_epoch(self, data: Data, use_aug: bool = True) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            data: PyG Data对象
            use_aug: 是否使用数据增强

        Returns:
            训练指标字典
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 应用数据增强
        if use_aug:
            train_data = self._apply_augmentation(data)
        else:
            train_data = data

        # 前向传播（使用增强后的数据）
        pred_thick, pred_exist = self.model(
            train_data.x,
            train_data.edge_index,
            train_data.edge_attr if hasattr(train_data, 'edge_attr') else None
        )

        # 计算损失（仅在训练集上）
        train_mask = train_data.train_mask
        loss, loss_dict = self.criterion(
            pred_thick[train_mask],
            pred_exist[train_mask],
            train_data.y_thick[train_mask],
            train_data.y_exist[train_mask],
            train_data.y_mask[train_mask] if hasattr(train_data, 'y_mask') else None
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 计算评估指标（使用原始数据，不受增强影响）
        with torch.no_grad():
            pred_thick_eval, pred_exist_eval = self.model(
                data.x,
                data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None
            )
            # 仅在存在的层上计算MAE和RMSE
            exist_mask = data.train_mask.unsqueeze(1) * data.y_exist.bool()
            if exist_mask.sum() > 0:
                mae = F.l1_loss(pred_thick_eval[exist_mask], data.y_thick[exist_mask]).item()
                rmse = torch.sqrt(F.mse_loss(pred_thick_eval[exist_mask], data.y_thick[exist_mask])).item()
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
        warmup_epochs: int = 20,
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
            warmup_epochs: 预热轮数（预热期间不触发早停）
            verbose: 是否打印日志
            log_interval: 日志打印间隔
            save_dir: 模型保存目录

        Returns:
            训练历史
        """
        # 初始化学习率调度器
        self._init_scheduler(epochs)

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
            print(f"数据增强: {'开启' if self.use_augmentation else '关闭'}")
            print(f"学习率调度: {self.scheduler_type}")
            print(f"预热轮数: {warmup_epochs}")
            print(f"{'='*60}")

        for epoch in range(epochs):
            self.epoch = epoch + 1
            is_warmup = epoch < warmup_epochs

            # 训练（预热期间降低增强概率）
            train_metrics = self.train_epoch(data, use_aug=self.use_augmentation and not is_warmup)

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

            # 早停检查（预热期间不触发早停）
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0

                # 保存最佳模型
                if save_dir:
                    self.save_checkpoint(save_dir, 'best_model.pt')
            elif not is_warmup:
                # 仅在预热期结束后累计patience计数
                patience_counter += 1

            # 打印日志
            if verbose and (epoch + 1) % log_interval == 0:
                warmup_status = " [预热]" if is_warmup else ""
                print(f"Epoch {epoch+1:3d}/{epochs}{warmup_status} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val MAE: {val_metrics['mae']:.3f}m | "
                      f"Val R²: {val_metrics['r2']:.3f} | "
                      f"LR: {current_lr:.2e}")

            # 早停（预热期结束后才检查）
            if not is_warmup and patience_counter >= patience:
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


def k_fold_cross_validation(
    model_class: type,
    model_kwargs: Dict,
    data: Data,
    n_splits: int = 5,
    epochs: int = 200,
    patience: int = 30,
    trainer_kwargs: Optional[Dict] = None,
    verbose: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    K-fold交叉验证

    专为小样本地质数据设计的K-fold交叉验证

    Args:
        model_class: 模型类
        model_kwargs: 模型初始化参数
        data: PyG Data对象（应包含所有数据，不需要预先划分）
        n_splits: fold数量
        epochs: 每个fold的训练轮数
        patience: 早停耐心值
        trainer_kwargs: 训练器额外参数
        verbose: 是否打印详细信息
        seed: 随机种子

    Returns:
        results: 交叉验证结果字典
    """
    from sklearn.model_selection import KFold
    import copy

    if trainer_kwargs is None:
        trainer_kwargs = {}

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 获取所有样本索引
    n_samples = data.x.shape[0]
    indices = np.arange(n_samples)

    # K-fold分割
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"开始 {n_splits}-Fold 交叉验证")
        print(f"总样本数: {n_samples}, 每个fold约 {n_samples//n_splits} 个测试样本")
        print(f"{'='*70}\n")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Fold {fold + 1}/{n_splits}")
            print(f"{'='*70}")

        # 进一步划分训练集和验证集（80% train, 20% val）
        n_train = int(len(train_val_idx) * 0.8)
        train_idx = train_val_idx[:n_train]
        val_idx = train_val_idx[n_train:]

        # 创建mask
        fold_data = copy.deepcopy(data)
        fold_data.train_mask = torch.zeros(n_samples, dtype=torch.bool)
        fold_data.val_mask = torch.zeros(n_samples, dtype=torch.bool)
        fold_data.test_mask = torch.zeros(n_samples, dtype=torch.bool)

        fold_data.train_mask[train_idx] = True
        fold_data.val_mask[val_idx] = True
        fold_data.test_mask[test_idx] = True

        if verbose:
            print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")

        # 创建新模型
        model = model_class(**model_kwargs)

        # 创建训练器
        trainer = ThicknessTrainer(model=model, **trainer_kwargs)

        # 训练
        history = trainer.train(
            data=fold_data,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        # 保存结果
        fold_result = {
            'fold': fold + 1,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'history': history,
            'test_metrics': history['test_metrics']
        }
        fold_results.append(fold_result)

        if verbose:
            metrics = fold_result['test_metrics']
            print(f"\nFold {fold + 1} 测试结果:")
            print(f"  MAE: {metrics['mae']:.3f}m")
            print(f"  RMSE: {metrics['rmse']:.3f}m")
            print(f"  R²: {metrics['r2']:.3f}")

    # 汇总结果
    test_maes = [r['test_metrics']['mae'] for r in fold_results]
    test_rmses = [r['test_metrics']['rmse'] for r in fold_results]
    test_r2s = [r['test_metrics']['r2'] for r in fold_results]

    summary = {
        'n_splits': n_splits,
        'fold_results': fold_results,
        'mae_mean': np.mean(test_maes),
        'mae_std': np.std(test_maes),
        'rmse_mean': np.mean(test_rmses),
        'rmse_std': np.std(test_rmses),
        'r2_mean': np.mean(test_r2s),
        'r2_std': np.std(test_r2s)
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"{n_splits}-Fold 交叉验证汇总结果")
        print(f"{'='*70}")
        print(f"MAE:  {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f} m")
        print(f"RMSE: {summary['rmse_mean']:.3f} ± {summary['rmse_std']:.3f} m")
        print(f"R²:   {summary['r2_mean']:.3f} ± {summary['r2_std']:.3f}")
        print(f"{'='*70}\n")

    return summary


def get_optimized_config_for_small_dataset(
    n_samples: int,
    n_layers: int,
    n_features: int
) -> Dict[str, Any]:
    """
    为小样本数据集生成优化的配置

    针对地质厚度预测的特殊优化：
    - 厚度范围大（0.5m~70m）：使用对数损失
    - 层数多（46层）：需要足够的模型容量
    - 样本少（28个）：需要强正则化

    Args:
        n_samples: 样本数量（钻孔数量）
        n_layers: 地层数量
        n_features: 特征数量

    Returns:
        config: 优化的配置字典
    """
    config = {
        'model': {},
        'trainer': {},
        'training': {}
    }

    # 计算任务复杂度：输出维度 / 样本数
    complexity_ratio = n_layers / max(n_samples, 1)
    print(f"[配置] 任务复杂度: {n_layers}层 / {n_samples}样本 = {complexity_ratio:.2f}")

    if complexity_ratio > 1.5:
        print("⚠️  警告：层数远多于样本数，预测会很困难！")

    # 模型配置 - 对于这种高维输出任务，需要足够容量
    # 关键：hidden_channels 需要与输出层数相匹配
    heads = 4
    hidden = max(128, n_layers * 3)  # 至少是层数的3倍
    hidden = min(hidden, 256)  # 但不超过256
    # 确保 hidden_channels 能被 heads 整除（GATv2Conv要求）
    hidden = (hidden // heads) * heads

    config['model']['hidden_channels'] = hidden
    config['model']['num_layers'] = 3
    config['model']['dropout'] = 0.2
    config['model']['heads'] = heads

    # 训练配置
    config['trainer']['learning_rate'] = 0.001
    config['trainer']['weight_decay'] = 1e-4
    config['trainer']['use_augmentation'] = False  # 小样本关闭增强
    config['trainer']['augmentation_prob'] = 0.0
    config['trainer']['scheduler_type'] = 'plateau'

    # 训练轮数 - 小样本需要更多轮来收敛
    config['training']['epochs'] = 500
    config['training']['patience'] = 60
    config['training']['warmup_epochs'] = 50

    # K-fold建议
    config['kfold'] = {
        'n_splits': 3,
        'use_kfold': False,
        'reason': f'样本量({n_samples})相对层数({n_layers})较少，不建议K-fold'
    }

    print(f"[配置] 模型: hidden={config['model']['hidden_channels']}, "
          f"layers={config['model']['num_layers']}, dropout={config['model']['dropout']}")
    print(f"[配置] 训练: epochs={config['training']['epochs']}, "
          f"patience={config['training']['patience']}, lr={config['trainer']['learning_rate']}")

    return config


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
    learning_rate: float = 0.001,
    use_augmentation: bool = False,
    scheduler_type: str = 'plateau',
    heads: int = 4,
    **trainer_kwargs
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
        use_augmentation: 是否使用数据增强
        scheduler_type: 学习率调度类型
        heads: 注意力头数
        **trainer_kwargs: 其他训练器参数

    Returns:
        (model, trainer) 元组
    """
    model = GNNThicknessPredictor(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        num_layers=gnn_layers,
        num_output_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        heads=heads
    )

    trainer = ThicknessTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        use_augmentation=use_augmentation,
        scheduler_type=scheduler_type,
        **trainer_kwargs
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
