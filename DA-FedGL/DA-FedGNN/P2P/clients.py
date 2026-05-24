"""
联邦学习客户端模块
包含基础客户端、DA-FedGNN、FedEgo、PENS 的客户端实现。
本版本增加：返回每个 local epoch 的平均训练损失，用于绘制本地训练 loss 曲线。
"""

import copy
import hashlib
import torch

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: cryptography library not available. Secure aggregation will be disabled.")


class Client_GC:
    """基础图分类客户端"""

    def __init__(self, model, id, name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = id
        self.name = name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

    def local_train(self, local_epoch):
        """本地训练（FedAvg风格），返回每个 local epoch 的平均训练损失。"""
        self.model.train()
        epoch_losses = []

        for _ in range(local_epoch):
            epoch_loss_sum = 0.0
            epoch_graphs = 0

            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                self.optimizer.zero_grad()

                pred = self.model(data)
                label = data.y
                loss = self.model.loss(pred, label)

                loss.backward()
                self.optimizer.step()

                epoch_loss_sum += loss.item() * data.num_graphs
                epoch_graphs += data.num_graphs

            epoch_losses.append(epoch_loss_sum / max(epoch_graphs, 1))

        self.clone_model_paramenter(self.W_old, self.W)
        self.compute_weight_update(self.W_old, self.W, device=self.args.device)
        return epoch_losses

    def local_train_prox(self, local_epoch, mu):
        """本地训练（FedProx风格），返回每个 local epoch 的平均训练损失。"""
        self.model.train()
        epoch_losses = []

        for _ in range(local_epoch):
            epoch_loss_sum = 0.0
            epoch_graphs = 0

            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                self.optimizer.zero_grad()

                pred = self.model(data)
                label = data.y
                base_loss = self.model.loss(pred, label)

                prox_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in self.W_old:
                        prox_term += ((param - self.W_old[name]) ** 2).sum()

                total_loss = base_loss + (mu / 2.0) * prox_term
                total_loss.backward()
                self.optimizer.step()

                epoch_loss_sum += total_loss.item() * data.num_graphs
                epoch_graphs += data.num_graphs

            epoch_losses.append(epoch_loss_sum / max(epoch_graphs, 1))

        self.clone_model_paramenter(self.W_old, self.W)
        self.compute_weight_update(self.W_old, self.W, device=self.args.device)
        return epoch_losses

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        ngraphs = 0

        with torch.no_grad():
            for data in self.dataLoader['test']:
                data = data.to(self.args.device)
                pred = self.model(data)
                label = data.y
                test_loss += self.model.loss(pred, label).item() * data.num_graphs
                test_acc += pred.max(dim=1)[1].eq(label).sum().item()
                ngraphs += data.num_graphs

        return test_loss / ngraphs, test_acc / ngraphs

    def clone_model_paramenter(self, param_target, param_source):
        for name in param_source:
            param_target[name].data = param_source[name].data.clone()

    def compute_weight_update(self, W_old, W_new, device=None):
        for k in W_new.keys():
            self.dW[k] = W_new[k].data - W_old[k].data

    def reset(self):
        copy_model = copy.deepcopy(self.model)
        self.W = {key: value for key, value in copy_model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in copy_model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in copy_model.named_parameters()}

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()


class CorrectedDaFedGNNClient_GC(Client_GC):
    """DA-FedGNN 客户端。"""

    def __init__(self, model, id, name, train_size, dataLoader, optimizer, args):
        super().__init__(model, id, name, train_size, dataLoader, optimizer, args)

        from models import CorrectedDaFedGNNModel

        sample_batch = next(iter(dataLoader['train']))
        num_features = sample_batch.x.shape[1]

        train_labels = []
        for batch in dataLoader['train']:
            train_labels.extend(batch.y.tolist())
        num_classes = len(set(train_labels))

        self.dafedgnn_model = CorrectedDaFedGNNModel(
            nfeat=num_features,
            nhid=args.hidden,
            nclass=num_classes,
            nlayer=args.nlayer,
            dropout=args.dropout,
            device=args.device,
            client_id=id,
        ).to(args.device)

        self.dafedgnn_optimizer = torch.optim.Adam(
            self.dafedgnn_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.W = {key: value for key, value in self.dafedgnn_model.named_parameters()}
        self.cached_global_state = None
        self.latest_shared_update = None

    def local_train_apfl(self, local_epoch, sync_gap=1):
        """
        DA-FedGNN 本地训练：
        - train batch 更新本地/共享分支；
        - val batch 更新 alpha；
        - 返回每个 local epoch 的平均融合训练损失。
        """
        self.dafedgnn_model.train()
        self.cached_global_state = {
            k: v.data.clone() for k, v in self.dafedgnn_model.get_global_parameters().items()
        }
        epoch_losses = []

        val_loader = self.dataLoader.get('val')
        val_iter = iter(val_loader) if val_loader is not None else None

        for _ in range(local_epoch):
            epoch_loss_sum = 0.0
            epoch_graphs = 0

            for batch_idx, data in enumerate(self.dataLoader['train']):
                data = data.to(self.args.device)
                self.dafedgnn_optimizer.zero_grad()

                fused_logits = self.dafedgnn_model.forward_logits(data)
                loss = self.dafedgnn_model.fused_loss(fused_logits, data.y)

                loss.backward()
                self.dafedgnn_optimizer.step()

                epoch_loss_sum += loss.item() * data.num_graphs
                epoch_graphs += data.num_graphs

                if (batch_idx + 1) % sync_gap == 0 and val_iter is not None:
                    try:
                        val_data = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_data = next(val_iter)
                    val_data = val_data.to(self.args.device)
                    self.dafedgnn_model.update_alpha_apfl(val_data)

            epoch_losses.append(epoch_loss_sum / max(epoch_graphs, 1))

        current_global = self.dafedgnn_model.get_global_parameters()
        self.latest_shared_update = {
            k: current_global[k].data.clone() - self.cached_global_state[k]
            for k in current_global.keys()
        }
        return epoch_losses

    def evaluate_corrected_dafedgnn(self):
        self.dafedgnn_model.eval()
        test_loss = 0.0
        test_acc = 0.0
        ngraphs = 0

        with torch.no_grad():
            for data in self.dataLoader['test']:
                data = data.to(self.args.device)
                pred = self.dafedgnn_model(data)
                label = data.y
                test_loss += self.dafedgnn_model.loss(pred, label).item() * data.num_graphs
                test_acc += pred.max(dim=1)[1].eq(label).sum().item()
                ngraphs += data.num_graphs

        return test_loss / ngraphs, test_acc / ngraphs

    def get_global_weights_dafedgnn(self):
        return self.dafedgnn_model.get_global_parameters()

    def set_global_weights_dafedgnn(self, global_state_dict):
        self.dafedgnn_model.update_global_parameters(global_state_dict)
        self.W = {key: value for key, value in self.dafedgnn_model.named_parameters()}

    def get_shared_update_dafedgnn(self):
        if self.latest_shared_update is None:
            current_global = self.dafedgnn_model.get_global_parameters()
            if self.cached_global_state is None:
                self.cached_global_state = {k: v.data.clone() for k, v in current_global.items()}
            self.latest_shared_update = {
                k: current_global[k].data.clone() - self.cached_global_state[k]
                for k in current_global.keys()
            }
        return {k: v.clone() for k, v in self.latest_shared_update.items()}

    def apply_aggregated_shared_update(self, aggregated_update):
        if self.cached_global_state is None:
            self.cached_global_state = {
                k: v.data.clone() for k, v in self.dafedgnn_model.get_global_parameters().items()
            }

        new_global_state = {}
        for k, base_tensor in self.cached_global_state.items():
            delta = aggregated_update.get(k)
            if delta is None:
                continue
            new_global_state[k] = base_tensor + delta.to(base_tensor.device)

        self.set_global_weights_dafedgnn(new_global_state)

    def get_alpha_value(self):
        return self.dafedgnn_model.get_alpha_value()

    def evaluate(self):
        return self.evaluate_corrected_dafedgnn()


if CRYPTO_AVAILABLE:
    class SecureAggregationECDH:
        """基于 ECDH + PRG 的成对掩码安全聚合。"""

        def __init__(self, client_id, all_client_ids):
            self.client_id = client_id
            self.all_client_ids = list(all_client_ids)
            self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            self.public_key = self.private_key.public_key()
            self.shared_keys = {}

        def get_public_key_bytes(self):
            return self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        def set_peer_public_key(self, peer_id, peer_public_key_bytes):
            peer_public_key = serialization.load_pem_public_key(
                peer_public_key_bytes,
                backend=default_backend(),
            )
            self.shared_keys[peer_id] = self.private_key.exchange(ec.ECDH(), peer_public_key)

        def _mask_seed(self, peer_id, round_number, param_name):
            shared_key = self.shared_keys[peer_id]
            payload = shared_key + str(round_number).encode('utf-8') + param_name.encode('utf-8')
            digest = hashlib.sha256(payload).digest()
            return int.from_bytes(digest[:8], byteorder='big', signed=False)

        def _sample_mask_like(self, tensor, seed):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            mask = torch.randn(tensor.numel(), generator=generator, dtype=tensor.dtype).reshape_as(tensor)
            return mask.to(tensor.device)

        def mask_model_parameters(self, model_params, round_number):
            masked_params = {}
            for name, param in model_params.items():
                mask = torch.zeros_like(param)
                for peer_id in sorted(self.shared_keys.keys()):
                    seed = self._mask_seed(peer_id, round_number, name)
                    peer_mask = self._sample_mask_like(param, seed)
                    if self.client_id < peer_id:
                        mask = mask + peer_mask
                    else:
                        mask = mask - peer_mask
                masked_params[name] = param + mask
            return masked_params
else:
    class SecureAggregationECDH:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Secure aggregation requires cryptography library")


class DFedGNNClient_GC(Client_GC):
    """D-FedGNN 客户端：复用基础本地训练，仅增加状态读写接口。"""

    def get_model_state(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.W = {key: value for key, value in self.model.named_parameters()}



class FedEgoClient_GC(Client_GC):
    """FedEgo客户端 - 双层架构 + 自适应lambda"""

    def __init__(self, model, id, name, train_size, dataLoader, optimizer, args):
        super().__init__(model, id, name, train_size, dataLoader, optimizer, args)
        self.lambda_param = 0.5
        self.label_distribution = None
        self.local_embedding = None

    def compute_label_distribution(self):
        label_counts = {}
        total = 0
        for data in self.dataLoader['train']:
            labels = data.y.tolist()
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                total += 1
        self.label_distribution = {label: count / total for label, count in label_counts.items()}
        return self.label_distribution

    def get_local_embedding(self):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                if hasattr(self.model, 'get_embedding'):
                    emb = self.model.get_embedding(data)
                else:
                    emb = self.model(data, return_embedding=True) if 'return_embedding' in self.model.forward.__code__.co_varnames else self.model(data)
                embeddings.append(emb.mean(dim=0))
        self.local_embedding = torch.stack(embeddings).mean(dim=0)
        return self.local_embedding

    def compute_lambda_from_emd(self, other_distributions):
        if self.label_distribution is None:
            self.compute_label_distribution()
        if not other_distributions:
            self.lambda_param = 0.5
            return self.lambda_param

        all_labels = set(self.label_distribution.keys())
        for dist in other_distributions:
            all_labels.update(dist.keys())
        all_labels = sorted(list(all_labels))

        distances = []
        for other_dist in other_distributions:
            dist_sum = 0
            for label in all_labels:
                p = self.label_distribution.get(label, 1e-10)
                q = other_dist.get(label, 1e-10)
                dist_sum += abs(p - q)
            distances.append(dist_sum)

        avg_distance = sum(distances) / len(distances)
        self.lambda_param = 0.2 + 0.6 * min(1.0, avg_distance / 2.0)
        return self.lambda_param

    def local_train(self, local_epoch, global_model=None):
        """FedEgo本地训练，返回每个 local epoch 的平均训练损失。"""
        self.model.train()
        epoch_losses = []

        for _ in range(local_epoch):
            epoch_loss_sum = 0.0
            epoch_graphs = 0

            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                self.optimizer.zero_grad()

                pred = self.model(data)
                label = data.y
                loss = self.model.loss(pred, label)

                if global_model is not None:
                    reg_loss = 0.0
                    for (_, param1), (_, param2) in zip(
                        self.model.named_parameters(),
                        global_model.named_parameters()
                    ):
                        reg_loss += torch.norm(param1 - param2) ** 2
                    loss = loss + 0.01 * reg_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss_sum += loss.item() * data.num_graphs
                epoch_graphs += data.num_graphs

            epoch_losses.append(epoch_loss_sum / max(epoch_graphs, 1))

        return epoch_losses

    def compute_lambda(self, global_dist, gamma=0.5):
        if self.label_distribution is None:
            self.compute_label_distribution()
        all_labels = set(list(self.label_distribution.keys()) + list(global_dist.keys()))
        distance = sum(abs(self.label_distribution.get(l, 1e-10) - global_dist.get(l, 1e-10)) for l in all_labels)
        self.lambda_param = 0.2 + 0.6 * min(1.0, distance * gamma)
        return self.lambda_param

    def mix_personalization_params(self, global_person_params, lambda_val):
        self.lambda_param = lambda_val
        current_state = self.model.state_dict()
        for key in current_state.keys():
            if "gin" in key.lower() or "personalization" in key.lower():
                if key in global_person_params:
                    current_state[key] = lambda_val * current_state[key] + (1 - lambda_val) * global_person_params[key]
        self.model.load_state_dict(current_state)

    def get_personalized_model(self, global_model):
        personalized_state = {}
        local_state = self.model.state_dict()
        global_state = global_model.state_dict()

        for key in local_state.keys():
            personalized_state[key] = (
                self.lambda_param * local_state[key] +
                (1 - self.lambda_param) * global_state[key]
            )

        personalized_model = copy.deepcopy(self.model)
        personalized_model.load_state_dict(personalized_state)
        return personalized_model


class PENSClient_GC(Client_GC):
    """PENS客户端 - 去中心化 + 邻居选择"""

    def __init__(self, model, id, name, train_size, dataLoader, optimizer, args):
        super().__init__(model, id, name, train_size, dataLoader, optimizer, args)
        self.neighbors = []
        self.neighbor_models = {}
        self.neighbor_scores = {}

    def evaluate_neighbor(self, neighbor_model):
        neighbor_model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                pred = neighbor_model(data)
                label = data.y
                pred_label = pred.argmax(dim=1)
                total_correct += (pred_label == label).sum().item()
                total_samples += len(label)

        return total_correct / total_samples if total_samples > 0 else 0

    def evaluate_model_loss(self, model_state_dict):
        current_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                pred = self.model(data)
                label = data.y
                loss = self.model.loss(pred, label)
                total_loss += loss.item() * data.num_graphs
                total_samples += data.num_graphs

        self.model.load_state_dict(current_state)
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def select_top_m(self, received_models, m):
        sorted_models = sorted(received_models.items(), key=lambda x: x[1][1])
        return [client_id for client_id, _ in sorted_models[:m]]

    def select_neighbors(self, all_clients, m=2):
        scores = {}
        for client in all_clients:
            if client.id != self.id:
                score = self.evaluate_neighbor(client.model)
                scores[client.id] = score
                self.neighbor_models[client.id] = copy.deepcopy(client.model)

        sorted_neighbors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.neighbors = [nid for nid, _ in sorted_neighbors[:m]]
        self.neighbor_scores = dict(sorted_neighbors[:m])
        return self.neighbors

    def aggregate_with_neighbors(self):
        if not self.neighbors:
            return

        aggregated_state = {
            key: param.data.clone()
            for key, param in self.model.named_parameters()
        }

        total_weight = 1.0
        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbor_models:
                neighbor_model = self.neighbor_models[neighbor_id]
                weight = self.neighbor_scores.get(neighbor_id, 0.5)
                total_weight += weight

                for (name, param), (_, n_param) in zip(
                    self.model.named_parameters(),
                    neighbor_model.named_parameters()
                ):
                    aggregated_state[name] += weight * n_param.data

        for name in aggregated_state:
            aggregated_state[name] /= total_weight

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(aggregated_state[name])

    def finalize_neighbors(self, T_discovery, N_clients, m):
        if self.neighbors:
            return self.neighbors
        if self.neighbor_scores:
            sorted_neighbors = sorted(self.neighbor_scores.items(), key=lambda x: x[1], reverse=True)
            self.neighbors = [nid for nid, _ in sorted_neighbors[:m]]
        return self.neighbors

    def gossip_with_neighbors(self, all_models):
        if not self.neighbors:
            return self.model.state_dict()

        selected_states = [copy.deepcopy(self.model.state_dict())]
        for neighbor_id in self.neighbors:
            if neighbor_id in all_models:
                selected_states.append(all_models[neighbor_id])

        avg_params = {}
        for key in selected_states[0].keys():
            avg_params[key] = torch.stack([state[key] for state in selected_states]).mean(dim=0)
        return avg_params

    def local_train(self, local_epoch):
        """PENS本地训练，返回每个 local epoch 的平均训练损失。"""
        self.model.train()
        epoch_losses = []

        for _ in range(local_epoch):
            epoch_loss_sum = 0.0
            epoch_graphs = 0

            for data in self.dataLoader['train']:
                data = data.to(self.args.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                label = data.y
                loss = self.model.loss(pred, label)
                loss.backward()
                self.optimizer.step()

                epoch_loss_sum += loss.item() * data.num_graphs
                epoch_graphs += data.num_graphs

            epoch_losses.append(epoch_loss_sum / max(epoch_graphs, 1))

        return epoch_losses


__all__ = [
    'Client_GC',
    'DFedGNNClient_GC',
    'CorrectedDaFedGNNClient_GC',
    'FedEgoClient_GC',
    'PENSClient_GC',
    'SecureAggregationECDH',
    'CRYPTO_AVAILABLE',
]