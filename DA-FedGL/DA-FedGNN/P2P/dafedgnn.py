"""DaFedGNN / 固定 α 训练流程（增加本地训练 loss 历史）。"""

import os
import pandas as pd
import numpy as np
import torch

from clients import CorrectedDaFedGNNClient_GC, SecureAggregationECDH, CRYPTO_AVAILABLE
from utils import evaluate_all_clients_enhanced, aggregate_client_metrics, save_experiment_results


def _ensure_dafed_clients(clients, communication_rounds):
    dafed_clients = []
    for client in clients:
        if hasattr(client, 'dafedgnn_model'):
            dafed_client = client
        else:
            dafed_client = CorrectedDaFedGNNClient_GC(
                client.model, client.id, client.name, client.train_size,
                client.dataLoader, client.optimizer, client.args
            )
        dafed_client.dafedgnn_model.set_total_rounds(communication_rounds)
        dafed_clients.append(dafed_client)
    return dafed_clients


def _setup_secure_aggregators(clients):
    if not CRYPTO_AVAILABLE:
        return None

    client_ids = [client.id for client in clients]
    secure_aggs = {client.id: SecureAggregationECDH(client.id, client_ids) for client in clients}
    public_keys = {cid: agg.get_public_key_bytes() for cid, agg in secure_aggs.items()}

    for cid, agg in secure_aggs.items():
        for peer_id, pubkey in public_keys.items():
            if peer_id != cid:
                agg.set_peer_public_key(peer_id, pubkey)
    return secure_aggs


def _average_update_dicts(update_dicts):
    if not update_dicts:
        return {}

    aggregated = {}
    all_param_names = sorted(set().union(*[upd.keys() for upd in update_dicts]))

    for name in all_param_names:
        tensors = [upd[name] for upd in update_dicts if name in upd]
        if not tensors:
            continue

        ref_shape = tensors[0].shape
        same_shape_tensors = [t for t in tensors if t.shape == ref_shape]

        # 只聚合所有客户端都形状一致的参数
        if len(same_shape_tensors) != len(tensors):
            print(f"[DaFedGNN] Skip incompatible parameter during aggregation: {name}, "
                  f"shapes={[tuple(t.shape) for t in tensors]}")
            continue

        aggregated[name] = torch.stack(same_shape_tensors, dim=0).mean(dim=0)

    return aggregated


def _aggregate_shared_updates(clients, round_number, secure_aggs=None):
    shared_updates = [client.get_shared_update_dafedgnn() for client in clients]

    if secure_aggs is not None:
        masked_updates = [
            secure_aggs[client.id].mask_model_parameters(update_dict, round_number)
            for client, update_dict in zip(clients, shared_updates)
        ]
        aggregated_update = _average_update_dicts(masked_updates)
    else:
        aggregated_update = _average_update_dicts(shared_updates)

    for client in clients:
        client.apply_aggregated_shared_update(aggregated_update)

    return aggregated_update


def _collect_alpha_stats(clients, round_number):
    alpha_stats = []
    for client in clients:
        stats = client.dafedgnn_model.get_alpha_update_stats()
        alpha_stats.append({
            'round': round_number,
            'client_id': client.id,
            'alpha': stats['current_alpha'],
            'momentum': stats['current_momentum'],
            'phase': stats['current_phase'],
            'progress': stats['training_progress'],
        })
    return alpha_stats


def _evaluate_round(clients, round_number, detailed_metrics):
    client_metrics = evaluate_all_clients_enhanced(clients)
    performance = aggregate_client_metrics(client_metrics, round_num=round_number)

    for metrics in client_metrics:
        detailed_metrics.append({'round': round_number, **metrics})

    return performance, client_metrics


def _append_local_loss_history(local_train_history, round_idx, local_epoch, round_client_epoch_losses):
    for local_ep in range(local_epoch):
        losses_this_epoch = [rec['epoch_losses'][local_ep] for rec in round_client_epoch_losses]
        local_train_history.append({
            'round': round_idx,
            'local_epoch': local_ep + 1,
            'global_local_epoch': (round_idx - 1) * local_epoch + (local_ep + 1),
            'mean_train_loss': float(np.mean(losses_this_epoch)),
            'std_train_loss': float(np.std(losses_this_epoch)),
        })


def run_alpha_fixed(clients, server, COMMUNICATION_ROUNDS, local_epoch, alpha_value, output_path=None):
    print(f"\n{'=' * 64}")
    print(f"运行 APFL-fixed α={alpha_value}（记录本地训练 loss）")
    print(f"{'=' * 64}")

    dafed_clients = _ensure_dafed_clients(clients, COMMUNICATION_ROUNDS)
    for client in dafed_clients:
        client.dafedgnn_model.set_fixed_alpha(alpha_value)

    secure_aggs = _setup_secure_aggregators(dafed_clients)

    performance_history = []
    detailed_metrics = []
    alpha_evolution = []
    local_train_history = []

    initial_perf, _ = _evaluate_round(dafed_clients, 0, detailed_metrics)
    performance_history.append(initial_perf)
    alpha_evolution.extend(_collect_alpha_stats(dafed_clients, 0))

    for round_idx in range(1, COMMUNICATION_ROUNDS + 1):
        round_client_epoch_losses = []

        for client in dafed_clients:
            client.dafedgnn_model.increment_round()
            epoch_losses = client.local_train_apfl(local_epoch, sync_gap=1)
            round_client_epoch_losses.append({
                'client_id': client.id,
                'client_name': client.name,
                'epoch_losses': epoch_losses,
            })

        _append_local_loss_history(local_train_history, round_idx, local_epoch, round_client_epoch_losses)
        _aggregate_shared_updates(dafed_clients, round_idx, secure_aggs=secure_aggs)

        if round_idx % 10 == 0 or round_idx == COMMUNICATION_ROUNDS:
            performance, _ = _evaluate_round(dafed_clients, round_idx, detailed_metrics)
            performance['algorithm'] = f'Alpha-{alpha_value}'
            performance_history.append(performance)
            alpha_evolution.extend(_collect_alpha_stats(dafed_clients, round_idx))

    results_df = pd.DataFrame(performance_history)
    results_df['algorithm'] = f'Alpha-{alpha_value}'
    results_df['alpha'] = alpha_value
    results_df['alpha_mode'] = 'fixed'

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        pd.DataFrame(alpha_evolution).to_csv(
            os.path.join(output_path, f'alpha_evolution_fixed_{alpha_value}.csv'),
            index=False
        )
        save_experiment_results(performance_history, detailed_metrics, f'Alpha_{alpha_value}', output_path)

    for client in dafed_clients:
        client.dafedgnn_model.set_adaptive_alpha()

    return results_df, performance_history, detailed_metrics, pd.DataFrame(local_train_history)


def run_dafedgnn_advanced(clients, server, COMMUNICATION_ROUNDS, local_epoch, output_path=None):
    print(f"\n{'=' * 72}")
    print("运行 DA-FedGNN（记录本地训练 loss）")
    print(f"{'=' * 72}")

    dafed_clients = _ensure_dafed_clients(clients, COMMUNICATION_ROUNDS)
    secure_aggs = _setup_secure_aggregators(dafed_clients)

    performance_history = []
    detailed_metrics = []
    alpha_evolution = []
    local_train_history = []

    initial_perf, _ = _evaluate_round(dafed_clients, 0, detailed_metrics)
    performance_history.append(initial_perf)
    alpha_evolution.extend(_collect_alpha_stats(dafed_clients, 0))

    for round_idx in range(1, COMMUNICATION_ROUNDS + 1):
        round_client_epoch_losses = []

        for client in dafed_clients:
            client.dafedgnn_model.increment_round()
            epoch_losses = client.local_train_apfl(local_epoch, sync_gap=1)
            round_client_epoch_losses.append({
                'client_id': client.id,
                'client_name': client.name,
                'epoch_losses': epoch_losses,
            })

        _append_local_loss_history(local_train_history, round_idx, local_epoch, round_client_epoch_losses)
        _aggregate_shared_updates(dafed_clients, round_idx, secure_aggs=secure_aggs)

        if round_idx % 10 == 0 or round_idx == 1 or round_idx == COMMUNICATION_ROUNDS:
            performance, _ = _evaluate_round(dafed_clients, round_idx, detailed_metrics)
            performance['algorithm'] = 'DaFedGNN'
            performance_history.append(performance)
            alpha_evolution.extend(_collect_alpha_stats(dafed_clients, round_idx))
            print(
                f"Round {round_idx}: "
                f"acc={performance['mean_accuracy']:.4f}, "
                f"f1={performance['mean_f1']:.4f}, "
                f"recall={performance['mean_recall']:.4f}, "
                f"alpha={performance.get('mean_alpha', np.nan):.4f}"
            )

    results_df = pd.DataFrame(performance_history)
    results_df['algorithm'] = 'DaFedGNN'
    results_df['alpha_mode'] = 'adaptive'
    results_df['secure_aggregation'] = secure_aggs is not None

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        pd.DataFrame(alpha_evolution).to_csv(
            os.path.join(output_path, 'alpha_evolution_dafedgnn.csv'),
            index=False
        )
        save_experiment_results(performance_history, detailed_metrics, 'DaFedGNN', output_path)

    return results_df, performance_history, detailed_metrics, alpha_evolution, [], pd.DataFrame(local_train_history)


def run_corrected_dafedgnn_enhanced(clients, server, COMMUNICATION_ROUNDS, local_epoch, output_path=None):
    results_df, performance_history, detailed_metrics, alpha_evolution, _, local_loss_df = run_dafedgnn_advanced(
        clients, server, COMMUNICATION_ROUNDS, local_epoch, output_path=output_path
    )
    return results_df, performance_history, detailed_metrics, local_loss_df


def run_apfl_fixed_alpha_generic(clients, server, rounds, local_epoch, alpha_value, output_path):
    return run_alpha_fixed(clients, server, rounds, local_epoch, alpha_value, output_path)