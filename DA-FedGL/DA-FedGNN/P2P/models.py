# 合并后的 models.py - 包含所有模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np


# ============================================================================
# 基础 GIN 模型
# ============================================================================

class GIN(torch.nn.Module):
    """基础 GIN (Graph Isomorphism Network) 模型"""

    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data_or_x, edge_index=None, batch=None):
        """Support both forward(data) and forward(x, edge_index, batch)"""
        if edge_index is None and batch is None:
            # Called as forward(data)
            data = data_or_x
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
        else:
            # Called as forward(x, edge_index, batch)
            x = data_or_x

        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


# ============================================================================
# 自适应学习率调度器
# ============================================================================

class AdaptiveLearningRateScheduler:
    """
    自适应学习率调度器

    特性:
        1. 基于性能改进自动调整学习率
        2. 支持预热（warm-up）阶段
        3. 支持学习率衰减
        4. 防止学习率过小
    """

    def __init__(self, initial_lr=0.01, min_lr=1e-5, max_lr=0.1,
                 warmup_rounds=10, patience=20, decay_factor=0.8,
                 improvement_threshold=0.001):
        """
        Args:
            initial_lr: 初始学习率
            min_lr: 最小学习率
            max_lr: 最大学习率
            warmup_rounds: 预热轮数
            patience: 等待改进的轮数
            decay_factor: 衰减因子
            improvement_threshold: 改进阈值
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_rounds = warmup_rounds
        self.patience = patience
        self.decay_factor = decay_factor
        self.improvement_threshold = improvement_threshold

        # 状态跟踪
        self.round_count = 0
        self.best_performance = -float('inf')
        self.rounds_without_improvement = 0
        self.lr_changes = []  # 记录学习率变化

    def step(self, current_performance):
        """
        更新学习率

        Args:
            current_performance: 当前性能指标（越大越好）

        Returns:
            new_lr: 新的学习率
        """
        self.round_count += 1
        old_lr = self.current_lr

        # 阶段1: Warmup - 逐渐增加学习率
        if self.round_count <= self.warmup_rounds:
            warmup_progress = self.round_count / self.warmup_rounds
            self.current_lr = self.initial_lr * warmup_progress
            reason = "warmup"

        # 阶段2: 正常训练 - 基于性能调整
        else:
            # 检查是否有改进
            improvement = current_performance - self.best_performance

            if improvement > self.improvement_threshold:
                # 性能提升 - 保持或略微增加学习率
                self.best_performance = current_performance
                self.rounds_without_improvement = 0
                # 略微增加学习率（但不超过max_lr）
                self.current_lr = min(self.current_lr * 1.05, self.max_lr)
                reason = "improved"
            else:
                # 性能未提升
                self.rounds_without_improvement += 1

                if self.rounds_without_improvement >= self.patience:
                    # 长时间未改进 - 降低学习率
                    self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                    self.rounds_without_improvement = 0  # 重置计数
                    reason = "plateau"
                else:
                    reason = "stable"

        # 确保学习率在合理范围内
        self.current_lr = max(min(self.current_lr, self.max_lr), self.min_lr)

        # 记录变化
        if abs(old_lr - self.current_lr) > 1e-8:
            self.lr_changes.append({
                'round': self.round_count,
                'old_lr': old_lr,
                'new_lr': self.current_lr,
                'reason': reason,
                'performance': current_performance
            })

        return self.current_lr

    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr

    def reset(self):
        """重置调度器"""
        self.current_lr = self.initial_lr
        self.round_count = 0
        self.best_performance = -float('inf')
        self.rounds_without_improvement = 0
        self.lr_changes = []

    def get_summary(self):
        """获取学习率变化摘要"""
        if not self.lr_changes:
            return "No learning rate changes"

        summary = f"Learning Rate Changes (Total: {len(self.lr_changes)}):\n"
        for change in self.lr_changes[-5:]:  # 显示最近5次
            summary += f"  Round {change['round']}: {change['old_lr']:.6f} → {change['new_lr']:.6f} ({change['reason']})\n"
        return summary


# ============================================================================
# DaFedGNN 模型（APFL 算法实现）
# ============================================================================

class CorrectedDaFedGNNModel(nn.Module):
    """
    修正的DaFedGNN模型 - 严格按照Algorithm 1 (APFL)实现

    核心组件：
    - 本地模型 v_i：个性化参数，不参与联邦聚合
    - 全局模型 w_i：共享参数，参与联邦聚合
    - 混合系数 α_i：通过梯度下降优化，公式(10)

    最终预测：v̂_i = α_i * v_i + (1 - α_i) * w_i
    """

    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, device='cpu', client_id=None):
        super(CorrectedDaFedGNNModel, self).__init__()

        self.num_layers = nlayer
        self.dropout = dropout
        self.device = device
        self.nhid = nhid
        self.nclass = nclass
        self.client_id = client_id

        # 本地模型 v_i - 个性化参数，永不共享
        self.local_model = self._create_gin_model(nfeat, nhid, nclass, nlayer, prefix='local')

        # 全局模型 w_i - 共享参数，参与联邦聚合
        self.global_model = self._create_gin_model(nfeat, nhid, nclass, nlayer, prefix='global')

        # 可训练的alpha参数
        self.alpha = nn.Parameter(torch.tensor(0.5, device=device, dtype=torch.float32))

        # alpha更新相关超参数（与论文中的 warm-up / adaptation / fine-tune 对齐）
        self.eta_alpha = 0.01
        self.alpha_min = 0.0
        self.alpha_max = 1.0
        self.warmup_ratio = 0.2
        self.finetune_ratio = 0.8
        self.warmup_scale = 0.2
        self.finetune_scale = 0.2

        # 调试计数器
        self.alpha_update_count = 0

        # 固定alpha模式
        self.fixed_alpha = None  # None表示自适应模式，否则为固定值
        self.alpha_drift_count = 0  # 记录尝试更新但被阻止的次数

        # 三阶段自适应策略变量
        self.total_rounds = None  # 总训练轮数（需要在训练开始时设置）
        self.current_round = 0  # 当前轮次
        self.alpha_gradient_history = []  # 梯度历史（用于计算方差）
        self.alpha_momentum = 0.0  # 动量累积

        # 准确率历史（用于早期探索阶段）
        self.local_accuracy_history = []  # 本地模型准确率历史
        self.global_accuracy_history = []  # 全局模型准确率历史

    def _create_gin_model(self, nfeat, nhid, nclass, nlayer, prefix):
        """创建一个完整的GIN模型结构"""
        model = nn.ModuleDict()

        # Pre-processing layer
        model[f'{prefix}_pre'] = nn.Sequential(nn.Linear(nfeat, nhid))

        # Graph convolution layers
        model[f'{prefix}_graph_convs'] = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
        model[f'{prefix}_graph_convs'].append(GINConv(nn1))
        for l in range(nlayer - 1):
            nnk = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
            model[f'{prefix}_graph_convs'].append(GINConv(nnk))

        # Post-processing and readout
        model[f'{prefix}_post'] = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU())
        model[f'{prefix}_readout'] = nn.Sequential(nn.Linear(nhid, nclass))

        return model


    def _forward_model(self, model, data, prefix):
        """执行单个分支的前向传播，返回 softmax 之前的原始 logits。"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = model[f'{prefix}_pre'](x)

        for conv in model[f'{prefix}_graph_convs']:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        x = model[f'{prefix}_post'](x)
        x = F.dropout(x, self.dropout, training=self.training)
        logits = model[f'{prefix}_readout'](x)
        return logits

    def forward_local_only_logits(self, data):
        """仅使用本地分支，返回原始 logits。"""
        return self._forward_model(self.local_model, data, 'local')

    def forward_global_only_logits(self, data):
        """仅使用共享分支，返回原始 logits。"""
        return self._forward_model(self.global_model, data, 'global')

    def forward_local_only(self, data):
        """仅使用本地分支进行推理，返回 log-probabilities。"""
        return F.log_softmax(self.forward_local_only_logits(data), dim=1)

    def forward_global_only(self, data):
        """仅使用共享分支进行推理，返回 log-probabilities。"""
        return F.log_softmax(self.forward_global_only_logits(data), dim=1)

    def fuse_logits(self, local_logits, global_logits, alpha=None):
        """在 softmax 之前进行 logits 显式融合。"""
        alpha = self.alpha if alpha is None else alpha
        return alpha * local_logits + (1.0 - alpha) * global_logits

    def forward_logits(self, data):
        """双分支前向并返回融合后的原始 logits。"""
        local_logits = self.forward_local_only_logits(data)
        global_logits = self.forward_global_only_logits(data)
        return self.fuse_logits(local_logits, global_logits)

    def forward(self, data):
        """用于评估/推理，返回融合预测的 log-probabilities。"""
        return F.log_softmax(self.forward_logits(data), dim=1)

    def fused_loss(self, fused_logits, label):
        """训练阶段直接对融合 logits 计算交叉熵。"""
        return F.cross_entropy(fused_logits, label)

    def loss(self, pred, label):
        """兼容评估阶段的 NLL 损失。"""
        return F.nll_loss(pred, label)

    def update_alpha_apfl(self, data):
        """
        基于验证 batch 的 alpha 更新，严格遵循论文里的“train / val 解耦 + logits 融合”思路。

        1. 本地/共享分支参数已由训练 batch 更新；
        2. alpha 仅由验证 batch 的验证损失更新；
        3. 采用 warm-up / adaptation / fine-tune 三阶段调度；
        4. 更新后对 alpha 做区间投影。
        """
        if self.fixed_alpha is not None:
            self.alpha_drift_count += 1
            return

        self.alpha_update_count += 1
        progress = self.get_training_progress()
        old_alpha = float(self.alpha.detach().item())

        local_logits = self.forward_local_only_logits(data)
        global_logits = self.forward_global_only_logits(data)
        fused_logits = self.fuse_logits(local_logits, global_logits)
        target = data.y.to(self.device)

        val_loss = self.fused_loss(fused_logits, target)
        grad_alpha = torch.autograd.grad(
            outputs=val_loss,
            inputs=self.alpha,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].item()

        self.alpha_gradient_history.append(grad_alpha)
        if len(self.alpha_gradient_history) > 100:
            self.alpha_gradient_history.pop(0)

        if progress < self.warmup_ratio:
            phase = 'warmup'
            step_size = self.eta_alpha * self.warmup_scale
            effective_grad = grad_alpha
        elif progress < self.finetune_ratio:
            phase = 'adaptation'
            step_size = self.eta_alpha
            effective_grad = grad_alpha
        else:
            phase = 'fine_tune'
            step_size = self.eta_alpha * self.finetune_scale
            self.alpha_momentum = 0.9 * self.alpha_momentum + 0.1 * grad_alpha
            effective_grad = self.alpha_momentum

        new_alpha = old_alpha - step_size * effective_grad
        new_alpha = max(self.alpha_min, min(self.alpha_max, new_alpha))

        with torch.no_grad():
            self.alpha.fill_(new_alpha)

        if self.alpha_update_count <= 10 or self.alpha_update_count % 50 == 0:
            print(
                f"Client {self.client_id} [{phase}] "
                f"alpha {old_alpha:.4f} -> {new_alpha:.4f} "
                f"(grad={grad_alpha:.6f}, step={step_size:.6f}, progress={progress:.2%})"
            )

    def get_local_parameters(self):
        """获取本地模型参数 v_i（不参与联邦聚合）"""
        local_params = {}
        for name, param in self.local_model.items():
            for sub_name, sub_param in param.named_parameters():
                local_params['{}.{}'.format(name, sub_name)] = sub_param
        local_params['alpha'] = self.alpha
        return local_params

    def get_global_parameters(self):
        """获取全局模型参数 w_i（参与联邦聚合）

        注意：
        MultiDS 场景下不同客户端的类别数可能不同，
        因此分类头（readout）不参与共享，只共享 encoder / GNN backbone / post 层。
        """
        global_params = {}
        for name, param in self.global_model.items():
            # 跳过输出分类头，避免不同数据集类别数不同导致维度不一致
            if 'readout' in name:
                continue

            for sub_name, sub_param in param.named_parameters():
                global_params['{}.{}'.format(name, sub_name)] = sub_param
        return global_params

    def update_global_parameters(self, global_state_dict):
        """更新全局模型参数，保持本地参数不变"""
        updated_count = 0
        for name, module in self.global_model.items():
            module_state = module.state_dict()
            updated = False
            for param_name, param_tensor in module_state.items():
                full_param_name = '{}.{}'.format(name, param_name)
                if full_param_name in global_state_dict and param_tensor.shape == global_state_dict[full_param_name].shape:
                    module_state[param_name] = global_state_dict[full_param_name].clone()
                    updated = True
            if updated:
                module.load_state_dict(module_state)
                updated_count += 1
        return updated_count

    def get_alpha_value(self):
        return self.alpha.data.item()

    def set_fixed_alpha(self, alpha_value):
        self.fixed_alpha = alpha_value
        with torch.no_grad():
            self.alpha.fill_(alpha_value)
        self.alpha.requires_grad = False
        print(f"Client {self.client_id}: Alpha fixed at {alpha_value:.3f}")

    def set_adaptive_alpha(self):
        self.fixed_alpha = None
        self.alpha.requires_grad = True
        print(f"Client {self.client_id}: Alpha set to adaptive mode")

    def get_alpha_update_stats(self):
        progress = self.get_training_progress()
        if progress < self.warmup_ratio:
            current_phase = 'warmup'
        elif progress < self.finetune_ratio:
            current_phase = 'adaptation'
        else:
            current_phase = 'fine_tune'
        stats = {
            'current_alpha': self.alpha.data.item(),
            'current_momentum': self.alpha_momentum,
            'current_phase': current_phase,
            'training_progress': progress,
            'update_count': self.alpha_update_count,
            'fixed_alpha': self.fixed_alpha,
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'alpha_momentum': self.alpha_momentum,
            'gradient_history_length': len(self.alpha_gradient_history),
            'local_acc_history_length': len(self.local_accuracy_history),
            'global_acc_history_length': len(self.global_accuracy_history),
        }
        if self.alpha_gradient_history:
            stats['gradient_mean'] = np.mean(self.alpha_gradient_history)
            stats['gradient_std'] = np.std(self.alpha_gradient_history)
            stats['gradient_var'] = np.var(self.alpha_gradient_history)
        return stats

    def set_total_rounds(self, total_rounds):
        self.total_rounds = total_rounds
        print(f"Client {self.client_id}: Set total rounds to {total_rounds}")

    def increment_round(self):
        self.current_round += 1

    def get_training_progress(self):
        if self.total_rounds is None or self.total_rounds == 0:
            return 0.0
        return self.current_round / self.total_rounds

    def compute_model_accuracy(self, data, use_local=True):
        self.eval()
        with torch.no_grad():
            pred = self.forward_local_only(data) if use_local else self.forward_global_only(data)
            pred_class = pred.max(dim=1)[1]
            target = data.y.to(self.device)
            accuracy = (pred_class == target).sum().item() / len(target)
        self.train()
        return accuracy

# ============================================================================
# 向后兼容的别名
# ============================================================================

# DaFedGNN 模型的别名
AdaptiveLayeredPersonalizationDaFedGNN = CorrectedDaFedGNNModel
LayeredPersonalizationDaFedGNN = CorrectedDaFedGNNModel


# ============================================================================
# ServerGIN 模型（用于服务器端）
# ============================================================================

class serverGIN(torch.nn.Module):
    """服务器端 GIN 模型（用于联邦学习服务器）"""

    def __init__(self, nlayer, nhid):
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                           torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))


# ============================================================================
# FedEgo 模型（双层架构 - 新增）
# ============================================================================

class FedEgoModel(torch.nn.Module):
    """
    FedEgo双层模型架构

    论文: "FedEgo: Privacy-preserving Personalized Federated Graph Learning with Ego-graphs" (2022)

    架构:
    - Reduction Layers (MLP): 特征降维，全局共享（FedAvg聚合）
    - Personalization Layers (GIN): 图卷积，个性化（λ混合）

    简化版本:
    - 保留双层架构和λ自适应机制
    - 省略Mixup和Ego-graph采样
    - 直接使用完整图进行训练
    """

    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(FedEgoModel, self).__init__()

        # Reduction Layers: MLP进行特征降维
        self.reduction = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Personalization Layers: GIN进行图卷积（替代原论文的GraphSAGE）
        self.personalization = GIN(nhid, nhid, nclass, nlayer, dropout)

        self.dropout = dropout

    def forward(self, data):
        """完整的前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Step 1: Reduction (降维)
        x = self.reduction(x)

        # Step 2: Personalization (GIN)
        x = self.personalization.forward(x, edge_index, batch)

        return x

    def forward_reduction_only(self, data):
        """只通过reduction层（用于生成embedding）"""
        x = data.x
        x = self.reduction(x)
        return x

    def get_reduction_params(self):
        """返回reduction层参数（用于FedAvg聚合）"""
        return self.reduction.parameters()

    def get_personalization_params(self):
        """返回personalization层参数（用于λ混合）"""
        return self.personalization.parameters()

    def loss(self, pred, label):
        """计算损失"""
        return F.nll_loss(pred, label)


# ============================================================================
# PENS 模型（标准GIN - 新增）
# ============================================================================

class PENSModel(torch.nn.Module):
    """
    PENS模型：标准GIN

    论文: "Decentralized federated learning of deep neural networks on non-iid data" (2021)

    原论文使用CNN（用于图像分类），我们用GIN替代（用于图分类）

    核心机制:
    - Neighbor Discovery: 基于loss选择相似的neighbors
    - Selective Gossip: 只与选定的neighbors通信
    """

    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(PENSModel, self).__init__()
        self.gin = GIN(nfeat, nhid, nclass, nlayer, dropout)

    def forward(self, data):
        """前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return self.gin.forward(x, edge_index, batch)

    def loss(self, pred, label):
        """计算损失"""
        return F.nll_loss(pred, label)