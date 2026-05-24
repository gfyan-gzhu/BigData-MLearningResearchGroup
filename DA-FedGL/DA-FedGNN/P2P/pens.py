"""PENS 训练流程（含 MultiDS，增加本地训练 loss 历史）。"""

import os
import random
import pandas as pd
import numpy as np
import torch
from models import PENSModel
from clients import PENSClient_GC
from utils import evaluate_all_clients_enhanced, aggregate_client_metrics


def run_pens(clients, server, COMMUNICATION_ROUNDS, local_epoch, T_discovery=200, m=2, output_path=None):
    N_clients = len(clients)
    performance_history = []
    local_train_history = []

    if N_clients < 2:
        for c_round in range(1, COMMUNICATION_ROUNDS + 1):
            round_client_epoch_losses = []
            for client in clients:
                epoch_losses = client.local_train(local_epoch)
                round_client_epoch_losses.append({'epoch_losses': epoch_losses})

            for local_ep in range(local_epoch):
                losses_this_epoch = [rec['epoch_losses'][local_ep] for rec in round_client_epoch_losses]
                local_train_history.append({
                    'round': c_round,
                    'local_epoch': local_ep + 1,
                    'global_local_epoch': (c_round - 1) * local_epoch + (local_ep + 1),
                    'mean_train_loss': float(np.mean(losses_this_epoch)),
                    'std_train_loss': float(np.std(losses_this_epoch)),
                })

        client_metrics = evaluate_all_clients_enhanced(clients)
        if client_metrics:
            acc = [m['accuracy'] for m in client_metrics]
            f1s = [m['f1_macro'] for m in client_metrics]
            rec = [m['recall_macro'] for m in client_metrics]
            los = [m['loss'] for m in client_metrics]
            result = pd.DataFrame([{
                'round': COMMUNICATION_ROUNDS,
                'mean_accuracy': np.mean(acc),
                'std_accuracy': np.std(acc),
                'mean_f1': np.mean(f1s),
                'std_f1': np.std(f1s),
                'mean_recall': np.mean(rec),
                'std_recall': np.std(rec),
                'mean_loss': np.mean(los),
                'std_loss': np.std(los),
                'phase': 0,
                'avg_neighbors': 0,
            }])
            return result, [result.to_dict('records')[0]], pd.DataFrame(local_train_history)

        return pd.DataFrame(), [], pd.DataFrame(local_train_history)

    m = min(max(1, m), N_clients - 1)
    T_discovery = min(T_discovery, COMMUNICATION_ROUNDS)

    client_metrics = evaluate_all_clients_enhanced(clients)
    initial_performance = aggregate_client_metrics(client_metrics, round_num=0)
    initial_performance['phase'] = 0
    performance_history.append(initial_performance)

    # Phase 1: Neighbor Discovery
    for t in range(T_discovery):
        all_models = [c.model.state_dict() for c in clients]

        for i, client in enumerate(clients):
            other_ids = [j for j in range(N_clients) if j != i]
            n_sampled = min(10, len(other_ids))
            if n_sampled == 0:
                continue

            sampled_ids = random.sample(other_ids, n_sampled)
            received_models = {}
            for j in sampled_ids:
                loss = client.evaluate_model_loss(all_models[j])
                received_models[j] = (all_models[j], loss)

            top_m_ids = client.select_top_m(received_models, m)
            if top_m_ids:
                top_m_params = [all_models[j] for j in top_m_ids]
                avg_params = {}
                for key in top_m_params[0].keys():
                    avg_params[key] = torch.stack([p[key] for p in top_m_params]).mean(dim=0)
                client.model.load_state_dict(avg_params)

        round_client_epoch_losses = []
        for client in clients:
            epoch_losses = client.local_train(local_epoch)
            round_client_epoch_losses.append({'epoch_losses': epoch_losses})

        for local_ep in range(local_epoch):
            losses_this_epoch = [rec['epoch_losses'][local_ep] for rec in round_client_epoch_losses]
            local_train_history.append({
                'round': t + 1,
                'local_epoch': local_ep + 1,
                'global_local_epoch': t * local_epoch + (local_ep + 1),
                'mean_train_loss': float(np.mean(losses_this_epoch)),
                'std_train_loss': float(np.std(losses_this_epoch)),
            })

        if (t + 1) % 20 == 0 or (t + 1) == T_discovery:
            client_metrics = evaluate_all_clients_enhanced(clients)
            performance = aggregate_client_metrics(client_metrics, round_num=t + 1)
            performance['phase'] = 1
            performance_history.append(performance)

    neighbor_counts = []
    for client in clients:
        neighbors = client.finalize_neighbors(T_discovery, N_clients, m)
        neighbor_counts.append(len(neighbors))

    # Phase 2: Selective Gossip
    for t in range(T_discovery, COMMUNICATION_ROUNDS):
        all_models = {i: c.model.state_dict() for i, c in enumerate(clients)}

        for client in clients:
            avg_params = client.gossip_with_neighbors(all_models)
            client.model.load_state_dict(avg_params)

        round_client_epoch_losses = []
        for client in clients:
            epoch_losses = client.local_train(local_epoch)
            round_client_epoch_losses.append({'epoch_losses': epoch_losses})

        for local_ep in range(local_epoch):
            losses_this_epoch = [rec['epoch_losses'][local_ep] for rec in round_client_epoch_losses]
            local_train_history.append({
                'round': t + 1,
                'local_epoch': local_ep + 1,
                'global_local_epoch': t * local_epoch + (local_ep + 1),
                'mean_train_loss': float(np.mean(losses_this_epoch)),
                'std_train_loss': float(np.std(losses_this_epoch)),
            })

        if (t + 1) % 20 == 0 or (t + 1) == COMMUNICATION_ROUNDS:
            client_metrics = evaluate_all_clients_enhanced(clients)
            performance = aggregate_client_metrics(client_metrics, round_num=t + 1)
            performance['phase'] = 2
            performance['avg_neighbors'] = np.mean(neighbor_counts)
            performance_history.append(performance)

    results_df = pd.DataFrame(performance_history)
    local_loss_df = pd.DataFrame(local_train_history)

    if output_path:
        results_df.to_csv(os.path.join(output_path, 'pens_loss_history.csv'), index=False)

    return results_df, performance_history, local_loss_df


def run_pens_multids(clients, server, args, outpath, splitedData):
    print("\n[Experiment 9/9] PENS (Decentralized + Neighbor Selection)")
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

        pens_clients = []
        for orig_client in group_clients:
            pens_model = PENSModel(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                nlayer=args.nlayer,
                dropout=args.dropout
            ).to(args.device)

            pens_optimizer = torch.optim.Adam(
                pens_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

            pens_client = PENSClient_GC(
                pens_model, orig_client.id, orig_client.name,
                orig_client.train_size, orig_client.dataLoader,
                pens_optimizer, args
            )
            pens_clients.append(pens_client)

        try:
            group_results, _, _ = run_pens(
                pens_clients, None,
                args.num_rounds, args.local_epoch,
                T_discovery=min(200, args.num_rounds),
                m=min(2, len(pens_clients) - 1),
                output_path=outpath
            )
            if isinstance(group_results, pd.DataFrame) and not group_results.empty:
                group_results['group_idx'] = group_idx
                group_results['nclass'] = nclass
                group_results['group_type'] = 'PENS'
                all_group_results.append(group_results)
        except Exception as e:
            print(f"    ❌ Warning: Group {group_idx + 1} PENS failed: {e}")

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