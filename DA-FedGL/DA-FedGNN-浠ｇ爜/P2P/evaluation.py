"""
统一的模型评估模块
集中处理所有评估相关功能，消除代码重复
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


class ModelEvaluator:
    """统一的模型评估器"""

    @staticmethod
    def evaluate_model(model, test_loader, device):
        """
        评估单个模型，返回所有指标

        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            device: 设备(cpu/cuda)

        Returns:
            dict: 包含所有评估指标的字典
        """
        model.eval()

        total_loss = 0.
        all_preds = []
        all_labels = []
        ngraphs = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)

                total_loss += loss.item() * batch.num_graphs
                pred_class = pred.max(dim=1)[1]

                all_preds.extend(pred_class.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                ngraphs += batch.num_graphs

        # 计算各项指标
        accuracy = sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)

        metrics = {
            'loss': total_loss / ngraphs,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'recall_macro': recall_macro,
            'precision_macro': precision_macro,
            'predictions': all_preds,
            'labels': all_labels
        }

        return metrics

    @staticmethod
    def evaluate_client(client):
        """
        评估单个客户端

        Args:
            client: 客户端对象

        Returns:
            dict: 客户端评估指标
        """
        # 使用客户端原有的evaluate函数获取基础指标
        if hasattr(client, 'evaluate_corrected_dafedgnn'):
            loss, acc = client.evaluate_corrected_dafedgnn()
        else:
            loss, acc = client.evaluate()

        # 获取详细指标
        test_loader = client.dataLoader['test']
        all_preds = []
        all_labels = []

        client.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(client.args.device)
                # 根据客户端类型选择模型
                if hasattr(client, 'dafedgnn_model'):
                    pred = client.dafedgnn_model(batch)
                else:
                    pred = client.model(batch)
                pred_class = pred.max(dim=1)[1]
                all_preds.extend(pred_class.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        # 计算指标
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)

        # 获取α值（如果有）
        alpha_value = None
        if hasattr(client, 'get_alpha_value'):
            alpha_value = client.get_alpha_value()

        return {
            'client_id': client.id,
            'client_name': client.name,
            'accuracy': acc,
            'loss': loss,
            'f1_macro': f1_macro,
            'recall_macro': recall_macro,
            'precision_macro': precision_macro,
            'alpha': alpha_value
        }

    @staticmethod
    def evaluate_all_clients(clients):
        """
        评估所有客户端

        Args:
            clients: 客户端列表

        Returns:
            list: 所有客户端的评估指标列表
        """
        return [ModelEvaluator.evaluate_client(client) for client in clients]

    @staticmethod
    def aggregate_metrics(client_metrics, round_num=None, algorithm_name=None):
        """
        聚合客户端指标

        Args:
            client_metrics: 客户端指标列表
            round_num: 轮次编号（可选）
            algorithm_name: 算法名称（可选）

        Returns:
            dict: 聚合后的指标
        """
        accuracies = [m['accuracy'] for m in client_metrics]
        f1_scores = [m['f1_macro'] for m in client_metrics]
        recalls = [m['recall_macro'] for m in client_metrics]
        precisions = [m['precision_macro'] for m in client_metrics]
        losses = [m['loss'] for m in client_metrics]

        aggregated = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses)
        }

        if round_num is not None:
            aggregated['round'] = round_num

        if algorithm_name is not None:
            aggregated['algorithm'] = algorithm_name

        # 添加α值统计（如果有）
        alpha_values = [m['alpha'] for m in client_metrics if m.get('alpha') is not None]
        if alpha_values:
            aggregated['mean_alpha'] = np.mean(alpha_values)
            aggregated['std_alpha'] = np.std(alpha_values)
            aggregated['min_alpha'] = np.min(alpha_values)
            aggregated['max_alpha'] = np.max(alpha_values)

        return aggregated

    @staticmethod
    def compute_confusion_matrix(client):
        """
        计算客户端的混淆矩阵

        Args:
            client: 客户端对象

        Returns:
            numpy.ndarray: 混淆矩阵
        """
        test_loader = client.dataLoader['test']
        all_preds = []
        all_labels = []

        client.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(client.args.device)
                if hasattr(client, 'dafedgnn_model'):
                    pred = client.dafedgnn_model(batch)
                else:
                    pred = client.model(batch)
                pred_class = pred.max(dim=1)[1]
                all_preds.extend(pred_class.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        return confusion_matrix(all_labels, all_preds)


# 保持向后兼容性的包装函数
def eval_gc_enhanced(model, test_loader, device):
    """向后兼容的评估函数（已弃用，请使用 ModelEvaluator.evaluate_model）"""
    return ModelEvaluator.evaluate_model(model, test_loader, device)


def evaluate_all_clients_enhanced(clients):
    """向后兼容的评估函数（已弃用，请使用 ModelEvaluator.evaluate_all_clients）"""
    return ModelEvaluator.evaluate_all_clients(clients)
