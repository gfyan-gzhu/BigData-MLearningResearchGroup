"""D-FedGNN 训练流程（去中心化 peer-to-peer 聚合）。"""

import os
import copy
import warnings
import pandas as pd
import numpy as np
import torch

from utils import evaluate_all_clients_enhanced, aggregate_client_metrics
from clients import CRYPTO_AVAILABLE, SecureAggregationECDH


class DFedGNNTopology:
    """构建 D-FedGNN 的对称双随机混合矩阵 A。"""

    @staticmethod
    def build(num_clients, topology='ring'):
        if num_clients <= 0:
            raise ValueError('num_clients must be positive')

        if num_clients == 1:
            return np.eye(1, dtype=np.float32)

        topology = topology.lower()
        if topology == 'fully_connected':
            return np.ones((num_clients, num_clients), dtype=np.float32) / float(num_clients)

        if topology != 'ring':
            raise ValueError(f'Unsupported topology: {topology}')

        A = np.zeros((num_clients, num_clients), dtype=np.float32)
        if num_clients == 2:
            A[0, 0] = A[0, 1] = 0.5
            A[1, 0] = A[1, 1] = 0.5
            return A

        for i in range(num_clients):
            A[i, i] = 1.0 / 3.0
            A[i, (i - 1) % num_clients] = 1.0 / 3.0
            A[i, (i + 1) % num_clients] = 1.0 / 3.0
        return A


class DFedGNNAggregator:
    """按混合矩阵执行点对点加权聚合。"""

    @staticmethod
    def _clone_state_dict(model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def _weighted_sum(state_dicts, weights, self_state=None):
        """
        对一组客户端 state_dict 做加权求和。

        若某个参数在不同客户端之间 shape 不一致，
        则说明它不适合跨客户端聚合（典型情况：不同数据集类别数不同导致分类头维度不同）。
        这类参数直接保留当前客户端自己的参数 self_state[name]。
        """
        aggregated = {}
        first = state_dicts[0]

        for name in first.keys():
            tensors = [state[name].detach() for state in state_dicts]

            ref_shape = tensors[0].shape
            all_same_shape = all(t.shape == ref_shape for t in tensors)

            if not all_same_shape:
                if self_state is None:
                    # 保险兜底：没有 self_state 时，至少保留第一个
                    aggregated[name] = tensors[0].clone()
                else:
                    aggregated[name] = self_state[name].detach().clone()

                print(f"[D-FedGNN] Skip incompatible parameter: {name}, "
                      f"shapes={[tuple(t.shape) for t in tensors]}")
                continue

            weighted = None
            for tensor, w in zip(tensors, weights):
                weighted = tensor * w if weighted is None else weighted + tensor * w

            aggregated[name] = weighted

        return aggregated

    @staticmethod
    def _setup_secure_aggregators(client_ids):
        if not CRYPTO_AVAILABLE:
            return None
        secure_aggs = {cid: SecureAggregationECDH(cid, client_ids) for cid in client_ids}
        public_keys = {cid: agg.get_public_key_bytes() for cid, agg in secure_aggs.items()}
        for cid, agg in secure_aggs.items():
            for peer_id, pubkey in public_keys.items():
                if peer_id != cid:
                    agg.set_peer_public_key(peer_id, pubkey)
        return secure_aggs

    @staticmethod
    def _masked_full_average(states, client_ids, round_number):
        """
        安全聚合只在 fully_connected 等权平均时严格成立；
        ring/非均匀权重场景默认不启用，避免得到错误结果。
        """
        secure_aggs = DFedGNNAggregator._setup_secure_aggregators(client_ids)
        if secure_aggs is None:
            return None
        masked_states = [secure_aggs[cid].mask_model_parameters(state, round_number) for cid, state in zip(client_ids, states)]
        uniform_weights = [1.0 / len(masked_states)] * len(masked_states)
        return DFedGNNAggregator._weighted_sum(masked_states, uniform_weights)

    @staticmethod
    def aggregate(clients, mixing_matrix, round_number, secure_aggregation=False):
        num_clients = len(clients)
        states = [DFedGNNAggregator._clone_state_dict(client.model) for client in clients]
        new_states = []

        if secure_aggregation and np.allclose(mixing_matrix, np.ones_like(mixing_matrix) / num_clients):
            client_ids = [client.id for client in clients]
            averaged_state = DFedGNNAggregator._masked_full_average(states, client_ids, round_number)
            if averaged_state is not None:
                new_states = [copy.deepcopy(averaged_state) for _ in clients]
            else:
                warnings.warn('Secure aggregation requested but cryptography is unavailable; fallback to plaintext averaging.')

        if not new_states:
            for i in range(num_clients):
                row_weights = mixing_matrix[i].tolist()
                new_states.append(
                    DFedGNNAggregator._weighted_sum(
                        states,
                        row_weights,
                        self_state=states[i]
                    )
                )

        for client, state in zip(clients, new_states):
            client.model.load_state_dict(state)
            client.W = {key: value for key, value in client.model.named_parameters()}

        return new_states


def run_dfedgnn(clients, server, COMMUNICATION_ROUNDS, local_epoch, topology='ring',
                secure_aggregation=False, eval_every=10, output_path=None):
    print('\n' + '=' * 68)
    print(f'Running D-FedGNN (topology={topology}, secure={secure_aggregation})')
    print('=' * 68)

    num_clients = len(clients)
    mixing_matrix = DFedGNNTopology.build(num_clients, topology=topology)

    if secure_aggregation and topology != 'fully_connected':
        warnings.warn(
            '当前 secure aggregation 仅对 fully_connected 等权平均严格适配；已自动降级为明文去中心化聚合。'
        )
        secure_aggregation = False

    performance_history = []
    detailed_metrics = []
    local_train_history = []

    initial_metrics = evaluate_all_clients_enhanced(clients)
    initial_perf = aggregate_client_metrics(initial_metrics, round_num=0)
    initial_perf['algorithm'] = 'D-FedGNN'
    initial_perf['topology'] = topology
    initial_perf['secure_aggregation'] = secure_aggregation
    performance_history.append(initial_perf)
    for m in initial_metrics:
        detailed_metrics.append({'round': 0, **m})

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if c_round % 20 == 0 or c_round == 1 or c_round == COMMUNICATION_ROUNDS:
            print(f'  > D-FedGNN round {c_round}/{COMMUNICATION_ROUNDS}')

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
                'round': c_round,
                'local_epoch': local_ep + 1,
                'global_local_epoch': (c_round - 1) * local_epoch + (local_ep + 1),
                'mean_train_loss': float(np.mean(losses_this_epoch)),
                'std_train_loss': float(np.std(losses_this_epoch)),
            })

        DFedGNNAggregator.aggregate(
            clients,
            mixing_matrix=mixing_matrix,
            round_number=c_round,
            secure_aggregation=secure_aggregation,
        )

        if c_round % eval_every == 0 or c_round == COMMUNICATION_ROUNDS:
            client_metrics = evaluate_all_clients_enhanced(clients)
            performance = aggregate_client_metrics(client_metrics, round_num=c_round)
            performance['algorithm'] = 'D-FedGNN'
            performance['topology'] = topology
            performance['secure_aggregation'] = secure_aggregation
            performance_history.append(performance)
            for metrics in client_metrics:
                detailed_metrics.append({'round': c_round, **metrics})

    results_df = pd.DataFrame(performance_history)
    local_loss_df = pd.DataFrame(local_train_history)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        results_df.to_csv(os.path.join(output_path, 'dfedgnn.csv'), index=False)
        local_loss_df.to_csv(os.path.join(output_path, 'dfedgnn_local_loss.csv'), index=False)

    return results_df, performance_history, detailed_metrics, local_loss_df
