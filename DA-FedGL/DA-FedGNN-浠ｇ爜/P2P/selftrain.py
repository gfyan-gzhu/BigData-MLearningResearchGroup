"""Self-Training baseline."""

import pandas as pd
import numpy as np
from evaluation import ModelEvaluator

def run_selftrain_enhanced(clients, server, local_epoch):
    """增强版自训练，包含F1和Recall指标 (修复: 返回聚合格式)"""
    for client in clients:
        client.local_train(local_epoch)

    # 评估所有客户端
    client_metrics = ModelEvaluator.evaluate_all_clients(clients)

    # 聚合为与其他算法一致的格式
    accuracies = [m['accuracy'] for m in client_metrics]
    f1_scores = [m['f1_macro'] for m in client_metrics]
    recalls = [m['recall_macro'] for m in client_metrics]
    precisions = [m['precision_macro'] for m in client_metrics]
    losses = [m['loss'] for m in client_metrics]

    aggregated_result = {
        'round': [local_epoch],
        'mean_accuracy': [np.mean(accuracies)],
        'std_accuracy': [np.std(accuracies)],
        'mean_f1': [np.mean(f1_scores)],
        'std_f1': [np.std(f1_scores)],
        'mean_recall': [np.mean(recalls)],
        'std_recall': [np.std(recalls)],
        'mean_precision': [np.mean(precisions)],
        'std_precision': [np.std(precisions)],
        'mean_loss': [np.mean(losses)],
        'std_loss': [np.std(losses)]
    }

    return pd.DataFrame(aggregated_result)
