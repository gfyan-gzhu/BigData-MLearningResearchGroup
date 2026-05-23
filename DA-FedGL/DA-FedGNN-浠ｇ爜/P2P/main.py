# -*- coding: utf-8 -*-
"""
重构后的 main.py
将各算法训练流程拆到 algorithms/ 目录，保留 clients.py / models.py / aggregation.py 的合并结构。
"""

import os
import argparse
import random
import copy
import warnings

import numpy as np
import torch

import setup
from selftrain import run_selftrain_enhanced
from fedavg import run_fedavg_enhanced
from fedprox import run_fedprox_enhanced
from dafedgnn import run_alpha_fixed, run_dafedgnn_advanced
from fedego import run_fedego, run_fedego_multids
from pens import run_pens, run_pens_multids
from dfedgnn import run_dfedgnn
from models import FedEgoModel, PENSModel
from clients import FedEgoClient_GC, PENSClient_GC
from server import FedEgoServer
from utils import save_results, generate_final_report, save_local_loss_history

warnings.filterwarnings("ignore", message="'data.DataLoader' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


def run_full_experiment_suite(clients, server, args, outpath, splitedData):
    """Run complete experiment suite with all comparison algorithms."""

    print("\n" + "=" * 80)
    print("FEDERATED LEARNING COMPREHENSIVE EXPERIMENTS")

    is_multi_ds = len(splitedData) > 1

    if is_multi_ds:
        print("Mode: MultiDS (multiple datasets)")
        print("Algorithms: All 10 algorithms (FedEgo and PENS will run per-dataset)")
    else:
        print("Mode: SingleDS (one dataset)")
        print("Algorithms: All 10 algorithms")

    print("=" * 80)

    all_results = {}

    print("\n[Experiment 1/9] Self-Training (Baseline)")
    st_frame = run_selftrain_enhanced(copy.deepcopy(clients), server, local_epoch=100)
    all_results['Self-Training'] = st_frame
    save_results(st_frame, outpath, 'selftrain_enhanced', args)

    print("\n[Experiment 2/9] FedAvg")
    fedavg_frame, fedavg_local_loss = run_fedavg_enhanced(
        copy.deepcopy(clients), server, args.num_rounds, args.local_epoch
    )
    all_results['FedAvg'] = fedavg_frame
    save_results(fedavg_frame, outpath, 'fedavg_enhanced', args)
    save_local_loss_history(fedavg_local_loss, outpath, 'fedavg_enhanced', args)

    print("\n[Experiment 3/9] FedProx (mu=0.01)")
    fedprox_frame, fedprox_local_loss = run_fedprox_enhanced(
        copy.deepcopy(clients), server, args.num_rounds, args.local_epoch, mu=0.01
    )
    all_results['FedProx'] = fedprox_frame
    save_results(fedprox_frame, outpath, 'fedprox_enhanced', args)
    save_local_loss_history(fedprox_local_loss, outpath, 'fedprox_enhanced', args)

    for idx, alpha in enumerate([0.2, 0.5, 0.8], start=4):
        print(f"\n[Experiment {idx}/9] Alpha-{alpha} (Fixed alpha)")
        alpha_results, _, _, alpha_local_loss = run_alpha_fixed(
            copy.deepcopy(clients), server, args.num_rounds, args.local_epoch,
            alpha_value=alpha, output_path=outpath
        )
        all_results[f'Alpha-{alpha}'] = alpha_results
        save_results(alpha_results, outpath, f'alpha_{alpha}_fixed', args)
        save_local_loss_history(alpha_local_loss, outpath, f'alpha_{alpha}_fixed', args)

    print("\n[Experiment 7/9] DaFedGNN-Advanced")
    corrected_clients, corrected_server, _ = setup.setup_corrected_dafedgnn_devices(splitedData, args)
    dafed_results, _, _, _, _, dafed_local_loss = run_dafedgnn_advanced(
        corrected_clients, corrected_server, args.num_rounds, args.local_epoch, output_path=outpath
    )
    all_results['DaFedGNN-Advanced'] = dafed_results
    save_results(dafed_results, outpath, 'dafedgnn_advanced', args)
    save_local_loss_history(dafed_local_loss, outpath, 'dafedgnn_advanced', args)

    print("\n[Experiment 8/9] D-FedGNN (Decentralized Peer-to-Peer)")
    dfed_clients, dfed_server, _ = setup.setup_dfedgnn_devices(splitedData, args)
    dfed_results, _, _, dfed_local_loss = run_dfedgnn(
        dfed_clients, dfed_server, args.num_rounds, args.local_epoch,
        topology='ring', secure_aggregation=False, output_path=outpath
    )
    all_results['D-FedGNN'] = dfed_results
    save_results(dfed_results, outpath, 'dfedgnn', args)
    save_local_loss_history(dfed_local_loss, outpath, 'dfedgnn', args)

    # if is_multi_ds:
    #     fedego_results = run_fedego_multids(clients, server, args, outpath, splitedData)
    #     all_results['FedEgo'] = fedego_results
    #     save_results(fedego_results, outpath, 'fedego', args)
    # else:
    #     print("\n[Experiment 8/9] FedEgo (Dual-layer + Adaptive lambda)")
    #     sample_data = next(iter(clients[0].dataLoader['train']))
    #     nfeat = sample_data.x.shape[1]
    #     all_labels = []
    #     for batch in clients[0].dataLoader['train']:
    #         all_labels.extend(batch.y.tolist())
    #     nclass = len(set(all_labels))
    #
    #     fedego_clients = []
    #     for orig_client in clients:
    #         fedego_model = FedEgoModel(
    #             nfeat=nfeat,
    #             nhid=args.hidden,
    #             nclass=nclass,
    #             nlayer=args.nlayer,
    #             dropout=args.dropout,
    #         ).to(args.device)
    #
    #         fedego_optimizer = torch.optim.Adam(
    #             fedego_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    #         )
    #
    #         fedego_client = FedEgoClient_GC(
    #             fedego_model,
    #             orig_client.id,
    #             orig_client.name,
    #             orig_client.train_size,
    #             orig_client.dataLoader,
    #             fedego_optimizer,
    #             args,
    #         )
    #         fedego_clients.append(fedego_client)
    #
    #     fedego_server_model = FedEgoModel(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         nlayer=args.nlayer,
    #         dropout=args.dropout,
    #     ).to(args.device)
    #     fedego_server = FedEgoServer(fedego_server_model, args.device)
    #
    #     fedego_results, _, fedego_local_loss = run_fedego(
    #         fedego_clients, fedego_server, args.num_rounds, args.local_epoch, output_path=outpath
    #     )
    #     all_results['FedEgo'] = fedego_results
    #     save_results(fedego_results, outpath, 'fedego', args)
    #     save_local_loss_history(fedego_local_loss, outpath, 'fedego', args)

    if is_multi_ds:
        pens_results = run_pens_multids(clients, server, args, outpath, splitedData)
        all_results['PENS'] = pens_results
        save_results(pens_results, outpath, 'pens', args)
    else:
        print("\n[Experiment 9/9] PENS (Decentralized + Neighbor Selection)")
        sample_data = next(iter(clients[0].dataLoader['train']))
        nfeat = sample_data.x.shape[1]
        all_labels = []
        for batch in clients[0].dataLoader['train']:
            all_labels.extend(batch.y.tolist())
        nclass = len(set(all_labels))

        pens_clients = []
        for orig_client in clients:
            pens_model = PENSModel(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                nlayer=args.nlayer,
                dropout=args.dropout,
            ).to(args.device)

            pens_optimizer = torch.optim.Adam(
                pens_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

            pens_client = PENSClient_GC(
                pens_model,
                orig_client.id,
                orig_client.name,
                orig_client.train_size,
                orig_client.dataLoader,
                pens_optimizer,
                args,
            )
            pens_clients.append(pens_client)

        pens_results, _, pens_local_loss = run_pens(
            pens_clients, None, args.num_rounds, args.local_epoch,
            T_discovery=200, m=2, output_path=outpath
        )
        all_results['PENS'] = pens_results
        save_results(pens_results, outpath, 'pens', args)
        save_local_loss_history(pens_local_loss, outpath, 'pens', args)

    generate_final_report(all_results, outpath)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='biochem')
    parser.add_argument('--model', type=str, default='gin')
    parser.add_argument('--num_rounds', type=int, default=400)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nlayer', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', type=int, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outpath = f"./outputs/multiDS/{args.dataset}_run{args.repeat if args.repeat else 1}_seed{args.seed}"
    os.makedirs(outpath, exist_ok=True)

    print("\n🚀 Starting Experiment Suite")
    print(f"  Dataset: {args.dataset}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Output: {outpath}")

    splitedData, df = setup.prepareData_multiDS(
        datapath='../data', group=args.dataset, batchSize=128, convert_x=False, seed=args.seed, target_dim=64
    )

    clients, server, idx_clients = setup.setup_devices(splitedData, args)
    first_dataset = list(splitedData.keys())[0]
    dataloaders, num_node_features, num_graph_labels, train_size = splitedData[first_dataset]
    args.num_features = num_node_features
    args.nclass = num_graph_labels

    print(f"  Loaded {len(splitedData)} datasets")
    print(f"  Features: {args.num_features}, Classes: {args.nclass}")

    all_results = run_full_experiment_suite(clients, server, args, outpath, splitedData)

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"All results saved in: {outpath}")
    print("=" * 80)
