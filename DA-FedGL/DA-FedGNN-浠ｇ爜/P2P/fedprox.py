"""FedProx 训练流程。"""

import pandas as pd
import numpy as np
from evaluation import ModelEvaluator
from aggregation import FedProxAggregator


def run_fedprox_enhanced(clients, server, rounds, local_epoch, mu):
    """FedProx：返回评估历史和本地训练 loss 历史。"""
    for client in clients:
        client.cache_weights()

    performance_history = []
    local_train_history = []

    for c_round in range(1, rounds + 1):
        if c_round % 50 == 0:
            print(f"  > FedProx round {c_round}")

        round_client_epoch_losses = []
        for client in clients:
            epoch_losses = client.local_train_prox(local_epoch, mu)
            round_client_epoch_losses.append({
                'client_id': client.id,
                'client_name': client.name,
                'epoch_losses': epoch_losses,
            })

        for local_ep in range(local_epoch):
            losses_this_epoch = [rec['epoch_losses'][local_ep] for rec in round_client_epoch_losses]
            local_train_history.append({
                'round': c_round,
                'local_epoch': local_ep + 1,
                'global_local_epoch': (c_round - 1) * local_epoch + (local_ep + 1),
                'mean_train_loss': float(np.mean(losses_this_epoch)),
                'std_train_loss': float(np.std(losses_this_epoch)),
            })

        FedProxAggregator.aggregate(clients)

        if c_round % 10 == 0 or c_round == rounds:
            client_metrics = ModelEvaluator.evaluate_all_clients(clients)

            accuracies = [m['accuracy'] for m in client_metrics]
            f1_scores = [m['f1_macro'] for m in client_metrics]
            recalls = [m['recall_macro'] for m in client_metrics]
            precisions = [m['precision_macro'] for m in client_metrics]
            losses = [m['loss'] for m in client_metrics]

            performance_history.append({
                'round': c_round,
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'mean_recall': np.mean(recalls),
                'std_recall': np.std(recalls),
                'mean_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses),
            })

    return pd.DataFrame(performance_history), pd.DataFrame(local_train_history)