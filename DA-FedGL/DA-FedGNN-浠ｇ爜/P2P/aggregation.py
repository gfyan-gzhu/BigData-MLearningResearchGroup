"""
统一的联邦聚合模块
集中处理所有联邦学习聚合相关功能
新增FedEgoAggregator（FedEgo专用聚合器）
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class FederatedAggregator(ABC):
    """联邦聚合器基类"""

    @staticmethod
    def weighted_average(client_weights_list, client_sizes):
        """
        加权平均聚合

        Args:
            client_weights_list: 客户端权重列表 [(weights_dict, size), ...]
            client_sizes: 客户端数据大小列表（可选，如果已包含在client_weights_list中）

        Returns:
            dict: 聚合后的权重字典
        """
        if not client_weights_list:
            return {}

        total_size = sum(size for _, size in client_weights_list)
        if total_size == 0:
            return {}

        # 获取所有参数名
        first_weights = client_weights_list[0][0]
        param_names = list(first_weights.keys())

        # 初始化聚合结果
        aggregated_weights = {}

        for param_name in param_names:
            weighted_sum = None

            for weights, size in client_weights_list:
                if param_name in weights:
                    if weighted_sum is None:
                        weighted_sum = torch.zeros_like(weights[param_name])

                    # 确保形状匹配
                    if weights[param_name].shape == weighted_sum.shape:
                        weighted_sum += weights[param_name] * size

            if weighted_sum is not None:
                aggregated_weights[param_name] = weighted_sum / total_size

        return aggregated_weights

    @staticmethod
    def collect_client_weights(clients):
        """
        收集客户端权重

        Args:
            clients: 客户端列表

        Returns:
            list: [(weights_dict, size), ...] 格式的列表
        """
        client_weights_list = []
        for client in clients:
            weights_copy = {k: v.data.clone() for k, v in client.W.items()}
            client_weights_list.append((weights_copy, client.train_size))
        return client_weights_list

    @staticmethod
    def broadcast_weights(clients, aggregated_weights):
        """
        将聚合后的权重广播给所有客户端

        Args:
            clients: 客户端列表
            aggregated_weights: 聚合后的权重字典
        """
        for client in clients:
            for k in aggregated_weights.keys():
                if k in client.W and aggregated_weights[k].shape == client.W[k].shape:
                    client.W[k].data = aggregated_weights[k].data.clone()


class FedAvgAggregator(FederatedAggregator):
    """FedAvg聚合器"""

    @staticmethod
    def aggregate(clients):
        """
        执行FedAvg聚合

        Args:
            clients: 客户端列表
        """
        # 收集客户端权重
        client_weights_list = FederatedAggregator.collect_client_weights(clients)

        # 加权平均
        aggregated_weights = FederatedAggregator.weighted_average(
            client_weights_list,
            [client.train_size for client in clients]
        )

        # 广播给所有客户端
        FederatedAggregator.broadcast_weights(clients, aggregated_weights)


class FedProxAggregator(FederatedAggregator):
    """FedProx聚合器"""

    @staticmethod
    def aggregate(clients):
        """
        执行FedProx聚合（与FedAvg相同的聚合方式，区别在于本地训练）

        Args:
            clients: 客户端列表
        """
        # FedProx的聚合方式与FedAvg相同
        FedAvgAggregator.aggregate(clients)

        # 更新缓存权重
        for client in clients:
            if hasattr(client, 'cache_weights'):
                client.cache_weights()


class DaFedGNNAggregator(FederatedAggregator):
    """DaFedGNN聚合器（用于全局模型部分）"""

    @staticmethod
    def aggregate_global_weights(clients):
        """
        聚合DaFedGNN客户端的全局权重

        Args:
            clients: DaFedGNN客户端列表
        """
        # 收集全局权重
        all_global_params = []
        total_size = 0

        for client in clients:
            if hasattr(client, 'get_global_weights_dafedgnn'):
                global_params = client.get_global_weights_dafedgnn()
                all_global_params.append((global_params, client.train_size))
                total_size += client.train_size

        if not all_global_params or total_size == 0:
            return

        # 聚合全局参数
        aggregated_global = {}
        first_params = all_global_params[0][0]

        for param_name in first_params.keys():
            weighted_sum = None

            for params, size in all_global_params:
                if param_name in params:
                    if weighted_sum is None:
                        weighted_sum = torch.zeros_like(params[param_name])

                    if params[param_name].shape == weighted_sum.shape:
                        weighted_sum += params[param_name] * size

            if weighted_sum is not None:
                aggregated_global[param_name] = weighted_sum / total_size

        # 广播聚合后的全局权重
        for client in clients:
            if hasattr(client, 'set_global_weights_dafedgnn'):
                client.set_global_weights_dafedgnn(aggregated_global)


class APFLAggregator(FederatedAggregator):
    """APFL聚合器（个性化联邦学习）"""

    @staticmethod
    def aggregate(clients, alpha_value=None):
        """
        执行APFL聚合

        Args:
            clients: 客户端列表
            alpha_value: 固定的α值（如果None，使用客户端自己的α）
        """
        # 如果指定了固定α值，设置所有客户端的α
        if alpha_value is not None:
            for client in clients:
                if hasattr(client, 'dafedgnn_model') and hasattr(client.dafedgnn_model, 'alpha'):
                    client.dafedgnn_model.alpha.data.fill_(alpha_value)
                    client.dafedgnn_model.alpha.requires_grad = False

        # 使用DaFedGNN聚合器处理全局部分
        DaFedGNNAggregator.aggregate_global_weights(clients)


class ClusteredFLAggregator(FederatedAggregator):
    """聚类联邦学习聚合器"""

    @staticmethod
    def aggregate_clusterwise(client_clusters):
        """
        按聚类进行聚合

        Args:
            client_clusters: 客户端聚类列表 [[client1, client2], [client3, client4], ...]
        """
        for cluster in client_clusters:
            if not cluster:
                continue

            # 收集聚类内的权重
            cluster_weights = []
            total_size = 0

            for client in cluster:
                weights_copy = {k: v.data.clone() for k, v in client.W.items()}
                cluster_weights.append((weights_copy, client.train_size))
                total_size += client.train_size

            if total_size == 0:
                continue

            # 聚合聚类内的权重
            aggregated = FederatedAggregator.weighted_average(
                cluster_weights,
                [client.train_size for client in cluster]
            )

            # 更新聚类内所有客户端的权重
            for client in cluster:
                for k in aggregated.keys():
                    if k in client.W and aggregated[k].shape == client.W[k].shape:
                        client.W[k].data = aggregated[k].data.clone()


# ============================================================================
# FedEgo 聚合器（新增）
# ============================================================================

class FedEgoAggregator(FederatedAggregator):
    """
    FedEgo专用聚合器
    
    论文: "FedEgo: Privacy-preserving Personalized Federated Graph Learning with Ego-graphs" (2022)
    
    功能：
    1. 聚合reduction层（FedAvg）
    2. 计算全局标签分布
    3. 基于EMD计算λ
    """
    
    @staticmethod
    def aggregate_reduction(client_params_list):
        """
        FedAvg聚合reduction层
        
        Args:
            client_params_list: 客户端参数列表
        
        Returns:
            聚合后的reduction层参数
        """
        if not client_params_list:
            return None
        
        avg_params = {}
        for key in client_params_list[0].keys():
            if 'reduction' in key:
                avg_params[key] = torch.stack([
                    params[key] for params in client_params_list
                ]).mean(dim=0)
        
        return avg_params
    
    @staticmethod
    def compute_lambda_from_emd(local_dist, global_dist, gamma=0.5):
        """
        基于EMD计算λ
        
        Args:
            local_dist: 本地标签分布
            global_dist: 全局标签分布
            gamma: 幂指数参数
        
        Returns:
            λ值（0-1之间）
        """
        all_classes = set(local_dist.keys()) | set(global_dist.keys())
        emd = sum(abs(local_dist.get(c, 0) - global_dist.get(c, 0)) 
                  for c in all_classes)
        lambda_val = (emd / 2.0) ** gamma
        return np.clip(lambda_val, 0.0, 1.0)


# 工具函数
def flatten(source):
    """将参数字典展平为一维张量"""
    return torch.cat([value.flatten() for value in source.values()])


def reduce_add_average(targets, sources, total_size):
    """
    将源参数加权平均后累加到目标参数

    Args:
        targets: 目标参数字典列表
        sources: 源参数和权重列表 [(params_dict, weight), ...]
        total_size: 总权重
    """
    for target in targets:
        for name in target:
            weighted_sum = torch.sum(
                torch.stack([source[0][name].data * source[1] for source in sources if name in source[0]]),
                dim=0
            )
            target[name].data += weighted_sum / total_size


# 向后兼容的函数
def aggregate_weights(server, selected_clients):
    """向后兼容的聚合函数（已弃用，请使用 FedAvgAggregator.aggregate）"""
    total_size = sum(client.train_size for client in selected_clients)

    for k in server.W.keys():
        weighted_sum = torch.sum(
            torch.stack([client.W[k].data * client.train_size for client in selected_clients]),
            dim=0
        )
        server.W[k].data = (weighted_sum / total_size).clone()


# 导出所有类和函数
__all__ = [
    'FederatedAggregator',
    'FedAvgAggregator',
    'FedProxAggregator',
    'DaFedGNNAggregator',
    'APFLAggregator',
    'ClusteredFLAggregator',
    'FedEgoAggregator',
    'flatten',
    'reduce_add_average',
    'aggregate_weights'
]