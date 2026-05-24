"""FedEgo 训练流程（含 MultiDS，增加本地训练 loss 历史）。"""

import os
import pandas as pd
import numpy as np
import torch
from models import FedEgoModel
from clients import FedEgoClient_GC
from server import FedEgoServer
from utils import evaluate_all_clients_enhanced, aggregate_client_metrics


def run_fedego(clients, server, COMMUNICATION_ROUNDS, local_epoch, output_path=None):
    print("\n" + "=" * 60)
    print("Running FedEgo (Dual-layer Architecture + Adaptive λ)")
    print("=" * 60)

    performance_history = []
    local_train_history = []

    client_metrics = evaluate_all_clients_enhanced(clients)
    initial_performance = aggregate_client_metrics(client_metrics, round_num=0)
    performance_history.append(initial_performance)

    for t in range(COMMUNICATION_ROUNDS):
        if (t + 1) % 10 == 0:
            print(f"  > FedEgo round {t + 1}/{COMMUNICATION_ROUNDS}")

        round_client_epoch_losses = []
        for client in clients:
            epoch_losses = client.local_train(local_epoch)
            round_client_epoch_losses.append({
                'client_id': client.id,
                'client_name': client.name,
                'epoch_losses': epoch_losses,
            })

        for local_ep in range(local_epoch):
            losses_this_epoch = [rec['epoch_losses'][local_ep] for rec in round_client_epoch_losses]
            local_train_history.append({
                'round': t + 1,
                'local_epoch': local_ep + 1,
                'global_local_epoch': t * local_epoch + (local_ep + 1),
                'mean_train_loss': float(np.mean(losses_this_epoch)),
                'std_train_loss': float(np.std(losses_this_epoch)),
            })

        client_dists = [c.compute_label_distribution() for c in clients]
        global_dist = server.compute_global_label_dist(client_dists)

        client_params = [c.model.state_dict() for c in clients]
        avg_reduction = server.aggregate_reduction_params(client_params)

        for client in clients:
            client_state = client.model.state_dict()
            client_state.update(avg_reduction)
            client.model.load_state_dict(client_state)

        global_person_params = server.get_global_personalization_params()
        lambdas = []
        for client in clients:
            lambda_val = client.compute_lambda(global_dist, gamma=0.5)
            client.mix_personalization_params(global_person_params, lambda_val)
            lambdas.append(lambda_val)

        if (t + 1) % 5 == 0 or (t + 1) == COMMUNICATION_ROUNDS:
            client_metrics = evaluate_all_clients_enhanced(clients)
            performance = aggregate_client_metrics(client_metrics, round_num=t + 1)
            performance['lambda_mean'] = np.mean(lambdas)
            performance['lambda_std'] = np.std(lambdas)
            performance_history.append(performance)

    results_df = pd.DataFrame(performance_history)
    local_loss_df = pd.DataFrame(local_train_history)

    if output_path:
        results_df.to_csv(os.path.join(output_path, 'fedego_loss_history.csv'), index=False)

    return results_df, performance_history, local_loss_df


def run_fedego_multids(clients, server, args, outpath, splitedData):
    print("\n[Experiment 8/9] FedEgo (Dual-layer + Adaptive lambda)")
    print("  Mode: MultiDS - Grouping by number of classes")

    client_groups = {}
    for client in clients:
        all_labels = []
        for batch in client.dataLoader['train']:
            all_labels.extend(batch.y.tolist())
        nclass = len(set(all_labels))
        client_groups.setdefault(nclass, []).append(client)

    all_group_results = []

    for group_idx, (nclass, group_clients) in enumerate(client_groups.items()):
        sample_data = next(iter(group_clients[0].dataLoader['train']))
        nfeat = sample_data.x.shape[1]

        fedego_clients = []
        for orig_client in group_clients:
            fedego_model = FedEgoModel(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                nlayer=args.nlayer,
                dropout=args.dropout
            ).to(args.device)

            fedego_optimizer = torch.optim.Adam(
                fedego_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

            fedego_client = FedEgoClient_GC(
                fedego_model, orig_client.id, orig_client.name,
                orig_client.train_size, orig_client.dataLoader,
                fedego_optimizer, args
            )
            fedego_clients.append(fedego_client)

        fedego_server_model = FedEgoModel(
            nfeat=nfeat,
            nhid=args.hidden,
            nclass=nclass,
            nlayer=args.nlayer,
            dropout=args.dropout
        ).to(args.device)
        fedego_server = FedEgoServer(fedego_server_model, args.device)

        try:
            group_results, _, _ = run_fedego(
                fedego_clients, fedego_server,
                args.num_rounds, args.local_epoch,
                output_path=outpath
            )
            if isinstance(group_results, pd.DataFrame) and not group_results.empty:
                group_results['group_idx'] = group_idx
                group_results['nclass'] = nclass
                all_group_results.append(group_results)
        except Exception as e:
            print(f"    ❌ Group {group_idx + 1} failed: {e}")
            continue

    if all_group_results:
        combined_results = pd.concat(all_group_results, ignore_index=True)

        final_accs = combined_results.groupby('group_idx')['mean_accuracy'].last()
        final_f1s = combined_results.groupby('group_idx')['mean_f1'].last()
        final_recalls = combined_results.groupby('group_idx')['mean_recall'].last()
        final_losses = combined_results.groupby('group_idx')['mean_loss'].last()

        return pd.DataFrame([{
            'round': args.num_rounds,
            'mean_accuracy': final_accs.mean(),
            'std_accuracy': final_accs.std() if len(final_accs) > 1 else 0,
            'mean_f1': final_f1s.mean(),
            'std_f1': final_f1s.std() if len(final_f1s) > 1 else 0,
            'mean_recall': final_recalls.mean(),
            'std_recall': final_recalls.std() if len(final_recalls) > 1 else 0,
            'mean_loss': final_losses.mean(),
            'std_loss': final_losses.std() if len(final_losses) > 1 else 0
        }])

    return pd.DataFrame()