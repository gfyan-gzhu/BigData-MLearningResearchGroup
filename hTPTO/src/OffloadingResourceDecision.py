import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from simEnvParameter import *
import copy
import math

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """SumTree数据结构，用于优先经验回放"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]



class PrioritizedACReplayBuffer:
    """AC专用的优先经验回放缓冲区"""
    def __init__(self, capacity, device, max_subtasks=DEVICE_NUM * 7, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.max_subtasks = max_subtasks
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done):
        # 动作预处理（与原 ACReplayBuffer 一致）
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, (list, np.ndarray)):
            if len(action) > 0 and isinstance(action[0], (list, np.ndarray)):
                flat_action = []
                for sublist in action:
                    flat_action.extend(sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
                action = flat_action
            action = [int(x) for x in action]
        else:
            action = [int(action)]

        if len(action) != self.max_subtasks:
            if len(action) < self.max_subtasks:
                action = action + [0] * (self.max_subtasks - len(action))
            else:
                action = action[:self.max_subtasks]

        experience = (state, action, reward, next_state, done)
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        self.tree.add(max_p, experience)

    def sample(self, batch_size):
        if self.tree.n_entries < batch_size:
            empty_state = np.zeros(self.max_subtasks, dtype=np.float32)
            empty_action = [0] * self.max_subtasks
            states = torch.FloatTensor([empty_state] * batch_size).to(self.device)
            actions = torch.LongTensor([empty_action] * batch_size).to(self.device)
            rewards = torch.FloatTensor([0.0] * batch_size).to(self.device)
            next_states = torch.FloatTensor([empty_state] * batch_size).to(self.device)
            dones = torch.FloatTensor([1.0] * batch_size).to(self.device)
            indices = np.zeros(batch_size, dtype=int)
            weights = torch.FloatTensor([1.0] * batch_size).to(self.device)
            return states, actions, rewards, next_states, dones, indices, weights

        indices = np.empty(batch_size, dtype=int)
        weights = np.empty(batch_size, dtype=np.float32)
        experiences = []

        p_total = self.tree.total()
        segment = p_total / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            indices[i] = idx
            experiences.append(data)
            prob = p / p_total
            weights[i] = (self.tree.n_entries * prob) ** (-self.beta)

        weights = weights / weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        states = [e[0] for e in experiences]
        actions = [e[1] for e in experiences]
        rewards = [e[2] for e in experiences]
        next_states = [e[3] for e in experiences]
        dones = [e[4] for e in experiences]

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            priority = (np.abs(td_err) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class SimpleGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_features, adjacency_matrix=None):
        batch_size, num_nodes, _ = node_features.shape

        if adjacency_matrix is None:
            adjacency_matrix = torch.ones(batch_size, num_nodes, num_nodes, device=node_features.device)
            adjacency_matrix = adjacency_matrix - torch.eye(num_nodes, device=node_features.device).unsqueeze(0)

        degree = adjacency_matrix.sum(dim=-1, keepdim=True)
        attention_weights = adjacency_matrix / (degree + 1e-8)

        aggregated = torch.bmm(attention_weights, node_features)
        output = self.linear(aggregated)
        output = self.activation(output)
        output = self.dropout(output)

        return output


class TaskDAGEncoder(nn.Module):
    def __init__(self, node_feat_dim=5, hidden_dim=64, num_gcn_layers=2):
        super(TaskDAGEncoder, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.gcn_layers.append(SimpleGraphConv(hidden_dim, hidden_dim))

        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.dependency_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

    def forward(self, node_features, adjacency_matrix=None):
        batch_size, num_nodes, _ = node_features.shape

        if adjacency_matrix is None:
            adjacency_matrix = self._create_default_dag_matrix(num_nodes).to(node_features.device)
            adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        node_embeddings = self.node_encoder(
            node_features.reshape(-1, self.node_feat_dim)
        ).reshape(batch_size, num_nodes, -1)

        for gcn_layer in self.gcn_layers:
            node_embeddings = gcn_layer(node_embeddings, adjacency_matrix)

        mean_pool = node_embeddings.mean(dim=1)
        max_pool = node_embeddings.max(dim=1)[0]

        graph_features = torch.cat([mean_pool, max_pool], dim=-1)
        graph_embedding = self.graph_pool(graph_features)

        return node_embeddings, graph_embedding

    def _create_default_dag_matrix(self, num_nodes=7):
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        edges = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5), (3, 6), (4, 6), (5, 6)]

        for src, dst in edges:
            adj_matrix[src, dst] = 1

        for i in range(num_nodes):
            adj_matrix[i, i] = 1

        return adj_matrix


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=128, num_layers=2, dropout=0.1):
        super(TemporalFeatureExtractor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

    def forward(self, temporal_features, seq_lengths=None):
        batch_size, seq_len, _ = temporal_features.shape

        lstm_out, (hidden, cell) = self.lstm(temporal_features)

        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]

        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)

        mean_lstm_out = lstm_out.mean(dim=1)
        combined = torch.cat([final_hidden, mean_lstm_out], dim=-1)

        temporal_embedding = self.temporal_encoder(combined)

        return temporal_embedding


class UAVFeatureEncoder(nn.Module):
    def __init__(self, uav_feat_dim=8, hidden_dim=64):
        super(UAVFeatureEncoder, self).__init__()

        self.uav_feat_dim = uav_feat_dim
        self.hidden_dim = hidden_dim

        self.uav_encoder = nn.Sequential(
            nn.Linear(uav_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.relation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.load_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )

    def forward(self, uav_features):
        batch_size, num_uavs, _ = uav_features.shape

        encoded_uavs = self.uav_encoder(
            uav_features.reshape(-1, self.uav_feat_dim)
        ).reshape(batch_size, num_uavs, -1)

        uav_embeddings = []
        for b in range(batch_size):
            batch_uavs = encoded_uavs[b]
            batch_embeddings = []

            for i in range(num_uavs):
                uav_i = batch_uavs[i]
                relations = []

                for j in range(num_uavs):
                    if i != j:
                        uav_j = batch_uavs[j]
                        pos_i = uav_features[b, i, :3]
                        pos_j = uav_features[b, j, :3]
                        distance = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)

                        relation_input = torch.cat([uav_i, uav_j, distance], dim=-1)
                        relation_feat = self.relation_encoder(relation_input)
                        relations.append(relation_feat)

                if relations:
                    mean_relation = torch.stack(relations).mean(dim=0)
                else:
                    mean_relation = torch.zeros_like(uav_i[:self.hidden_dim // 2])

                load_factor = uav_features[b, i, -1].unsqueeze(0)
                load_input = torch.cat([uav_i, load_factor], dim=-1)
                load_feat = self.load_encoder(load_input)

                combined = torch.cat([uav_i, mean_relation, load_feat], dim=-1)
                linear_proj = nn.Linear(combined.shape[-1], self.hidden_dim // 4).to(combined.device)
                combined_proj = linear_proj(combined)
                batch_embeddings.append(combined_proj)

            uav_embeddings.append(torch.stack(batch_embeddings))

        return torch.stack(uav_embeddings, dim=0)


class GNNLSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_subtasks=DEVICE_NUM * 7,
                 hidden_dim=256, num_uavs=UAV_MAX_NUM, dropout=0.1):
        super(GNNLSTMActor, self).__init__()

        self.max_subtasks = max_subtasks
        self.action_dim = action_dim
        self.num_uavs = num_uavs
        self.hidden_dim = hidden_dim

        self.device_feat_dim = DEVICE_NUM * 4
        self.task_feat_dim = DEVICE_NUM * (3 + 14 + 7 + 35)
        self.uav_feat_dim = UAV_MAX_NUM * 8
        self.decision_feat_dim = UAV_MAX_NUM * 4

        expected_total_dim = self.device_feat_dim + self.task_feat_dim + self.uav_feat_dim + self.decision_feat_dim
        print(f"AC Actor期望维度: {expected_total_dim}")
        if state_dim != expected_total_dim:
            print(f"警告：状态维度不匹配，期望{expected_total_dim}，实际{state_dim}")
            self.state_dim_adjusted = expected_total_dim
        else:
            self.state_dim_adjusted = state_dim

        self.device_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.device_dim_adjust = nn.Linear(16, 16)

        self.dag_encoder = TaskDAGEncoder(
            node_feat_dim=5,
            hidden_dim=64,
            num_gcn_layers=2
        )

        self.temporal_extractor = TemporalFeatureExtractor(
            input_dim=16 + 3,
            hidden_dim=128,
            num_layers=2,
            dropout=dropout
        )

        self.uav_encoder = UAVFeatureEncoder(
            uav_feat_dim=8,
            hidden_dim=64
        )

        self.decision_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        fusion_input_dim = 16 + 32 + 64 + 64 + 32

        print(f"AC Actor融合输入维度: {fusion_input_dim}")

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.shared_decision_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 64, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Softmax(dim=-1)
        )

        self.load_balancer = nn.Sequential(
            nn.Linear(num_uavs, 32),
            nn.ReLU(),
            nn.Linear(32, num_uavs),
            nn.Sigmoid()
        )

    def forward(self, state, uav_features=None, training=True):
        batch_size = state.shape[0]

        if state.shape[1] != self.state_dim_adjusted:
            print(f"警告：状态维度不匹配，期望{self.state_dim_adjusted}，实际{state.shape[1]}")
            if state.shape[1] < self.state_dim_adjusted:
                padding = torch.zeros(batch_size, self.state_dim_adjusted - state.shape[1],
                                      device=state.device)
                state = torch.cat([state, padding], dim=1)
            else:
                state = state[:, :self.state_dim_adjusted]

        device_features = state[:, :self.device_feat_dim].reshape(batch_size, DEVICE_NUM, 4)
        task_features = state[:, self.device_feat_dim:self.device_feat_dim + self.task_feat_dim]

        subtask_features = task_features.reshape(batch_size, DEVICE_NUM, 59)
        subtask_node_features = subtask_features[:, :, 3 + 14 + 7:]
        subtask_node_features = subtask_node_features.reshape(batch_size, DEVICE_NUM, 7, 5)

        if uav_features is None:
            uav_start_idx = self.device_feat_dim + self.task_feat_dim
            uav_end_idx = uav_start_idx + self.uav_feat_dim
            uav_features = state[:, uav_start_idx:uav_end_idx]
            uav_features = uav_features.reshape(batch_size, UAV_MAX_NUM, 8)

        encoded_devices = self.device_encoder(
            device_features.reshape(-1, 4)
        ).reshape(batch_size, DEVICE_NUM, -1)

        device_embedding_temp = encoded_devices.mean(dim=1)
        device_embedding = self.device_dim_adjust(device_embedding_temp)

        node_embeddings, graph_embedding = self.dag_encoder(
            subtask_node_features.reshape(batch_size * DEVICE_NUM, 7, 5)
        )

        node_embeddings = node_embeddings.reshape(batch_size, DEVICE_NUM * 7, -1)
        graph_embedding = graph_embedding.reshape(batch_size, DEVICE_NUM, -1).mean(dim=1)

        temporal_input = torch.cat([
            encoded_devices,
            subtask_features[:, :, :3]
        ], dim=-1)

        temporal_embedding = self.temporal_extractor(temporal_input)

        uav_embedding = self.uav_encoder(uav_features)
        uav_embedding_flat = uav_embedding.reshape(batch_size, -1)

        decision_start_idx = self.device_feat_dim + self.task_feat_dim + self.uav_feat_dim
        decision_features = state[:, decision_start_idx:decision_start_idx + self.decision_feat_dim]
        decision_features = decision_features.reshape(batch_size, UAV_MAX_NUM, 4)
        encoded_decisions = self.decision_encoder(
            decision_features.reshape(-1, 4)
        ).reshape(batch_size, -1)

        fusion_input = torch.cat([
            device_embedding,
            graph_embedding,
            temporal_embedding,
            uav_embedding_flat,
            encoded_decisions
        ], dim=-1)

        fused_features = self.feature_fusion(fusion_input)

        fused_expanded = fused_features.unsqueeze(1).repeat(1, node_embeddings.shape[1], 1)

        uav_loads = uav_features[:, :, -1]
        load_factors = self.load_balancer(uav_loads)

        action_probs = []
        for i in range(min(self.max_subtasks, fused_expanded.shape[1])):
            if fused_expanded.shape[1] > i:
                node_feat = fused_expanded[:, i, :]
                node_specific = node_embeddings[:, i, :] if node_embeddings.shape[1] > i else torch.zeros_like(
                    node_feat)
                combined = torch.cat([node_feat, node_specific], dim=-1)
            else:
                combined = fused_features

            probs = self.shared_decision_head(combined)

            action_probs.append(probs)

        while len(action_probs) < self.max_subtasks:
            action_probs.append(action_probs[-1] if action_probs else
                                torch.ones(batch_size, self.action_dim, device=state.device) / self.action_dim)

        return torch.stack(action_probs, dim=1), load_factors


class ACCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super(ACCritic, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state):
        encoded = self.state_encoder(state)
        return self.value_head(encoded)


class ActorCritic:
    def __init__(self, state_dim, action_dim, max_subtasks=DEVICE_NUM * 7,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
                 buffer_capacity=100000, batch_size=128, device=None):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_subtasks = max_subtasks
        self.gamma = gamma
        self.batch_size = batch_size

        self.actor = GNNLSTMActor(state_dim, action_dim, max_subtasks).to(self.device)
        self.critic = ACCritic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = PrioritizedACReplayBuffer(buffer_capacity, self.device, max_subtasks)
        self.train_step = 0
        self.epsilon = 0.3

    def select_action(self, state, epsilon=None, num_using_uavs=None, uav_features=None):
        if epsilon is None:
            epsilon = self.epsilon

        if num_using_uavs is None:
            num_using_uavs = self.action_dim - 1

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if uav_features is not None:
            uav_features_tensor = torch.FloatTensor(uav_features).unsqueeze(0).to(self.device)
        else:
            device_feat_dim = DEVICE_NUM * 4
            task_feat_dim = DEVICE_NUM * 59
            uav_feat_start = device_feat_dim + task_feat_dim
            uav_features_tensor = state_tensor[:, uav_feat_start:uav_feat_start + UAV_MAX_NUM * 8]
            uav_features_tensor = uav_features_tensor.reshape(1, UAV_MAX_NUM, 8)

        with torch.no_grad():
            action_probs, predicted_loads = self.actor(state_tensor, uav_features_tensor)

        actions = []
        valid_uav_indices = list(range(1, min(num_using_uavs + 1, self.action_dim)))
        uav_assignment_counts = {i: 0 for i in valid_uav_indices}

        for subtask_idx in range(min(self.max_subtasks, action_probs.shape[1])):
            probs = action_probs[0, subtask_idx].cpu().numpy()

            if np.any(np.isnan(probs)) or np.any(probs < 0) or np.sum(probs) == 0:
                probs = np.ones(self.action_dim) / self.action_dim
            else:
                probs = probs / np.sum(probs)

            valid_probs = np.zeros_like(probs)
            valid_probs[0] = probs[0]

            # 只允许选择有效的无人机索引
            for i in valid_uav_indices:
                if i < len(probs):
                    valid_probs[i] = probs[i]

            if random.random() < epsilon:
                valid_actions = [0] + valid_uav_indices

                if len(valid_actions) > 1:
                    weights = [1.0]
                    for i in range(1, len(valid_actions)):
                        weight = 1.0 / (1 + uav_assignment_counts.get(valid_actions[i], 0))
                        weights.append(weight)

                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    action = np.random.choice(valid_actions, p=weights)
                else:
                    action = 0
            else:
                action = np.argmax(valid_probs)
                # 确保动作在有效范围内
                if action >= len(valid_probs) or (action > 0 and action not in valid_uav_indices):
                    action = 0

            actions.append(action)

            if action > 0 and action in uav_assignment_counts:
                uav_assignment_counts[action] += 1

        if len(actions) < self.max_subtasks:
            actions = actions + [0] * (self.max_subtasks - len(actions))
        elif len(actions) > self.max_subtasks:
            actions = actions[:self.max_subtasks]

        return actions, action_probs

    def push_experience(self, state, action, reward, next_state, done):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if isinstance(action, (list, np.ndarray)):
            if len(action) > 0 and isinstance(action[0], (list, np.ndarray)):
                flat_action = []
                for sublist in action:
                    if isinstance(sublist, (list, np.ndarray)):
                        flat_action.extend(sublist)
                    else:
                        flat_action.append(sublist)
                action = flat_action

            action = [int(x) for x in action] if len(action) > 0 else [0] * self.max_subtasks
        else:
            action = [int(action)]

        if len(action) != self.max_subtasks:
            if len(action) < self.max_subtasks:
                action = action + [0] * (self.max_subtasks - len(action))
            else:
                action = action[:self.max_subtasks]

        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        try:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

            # 确保 actions 形状正确（与原逻辑一致）
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            if actions.shape[1] != self.max_subtasks:
                if actions.shape[1] == 1 and self.max_subtasks > 1:
                    actions = actions.repeat(1, self.max_subtasks)
                elif actions.shape[1] > self.max_subtasks:
                    actions = actions[:, :self.max_subtasks]
                elif actions.shape[1] < self.max_subtasks:
                    padding = torch.zeros(actions.shape[0], self.max_subtasks - actions.shape[1],
                                          dtype=actions.dtype, device=self.device)
                    actions = torch.cat([actions, padding], dim=1)

            with torch.no_grad():
                next_values = self.critic(next_states).squeeze()
                if next_values.dim() == 0:
                    next_values = next_values.unsqueeze(0)
                target_values = rewards + self.gamma * next_values * (1 - dones)

            current_values = self.critic(states).squeeze()
            if current_values.dim() == 0:
                current_values = current_values.unsqueeze(0)
            if target_values.shape != current_values.shape:
                target_values = target_values.view_as(current_values)

            # 带权重的 critic 损失
            critic_loss = (weights * F.mse_loss(current_values, target_values, reduction='none')).mean()

            # 计算 TD 误差用于更新优先级
            td_errors = (current_values - target_values).detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, td_errors)

            # 计算 actor 损失（与原逻辑相同，但使用当前值）
            action_probs, predicted_loads = self.actor(states)
            advantages = target_values - current_values

            log_probs = []
            for i in range(self.max_subtasks):
                if i < action_probs.shape[1]:
                    valid_actions = torch.clamp(actions[:, i], 0, self.action_dim - 1)
                    probs_i = action_probs[:, i, :].clone()
                    gathered_probs = probs_i.gather(1, valid_actions.unsqueeze(1)).squeeze()
                    log_prob = torch.log(gathered_probs.clamp(min=1e-10))
                    log_probs.append(log_prob)
                else:
                    if action_probs.shape[1] > 0:
                        last_idx = action_probs.shape[1] - 1
                        valid_actions = torch.clamp(actions[:, i], 0, self.action_dim - 1)
                        probs_last = action_probs[:, last_idx, :].clone()
                        gathered_probs = probs_last.gather(1, valid_actions.unsqueeze(1)).squeeze()
                        log_prob = torch.log(gathered_probs.clamp(min=1e-10))
                    else:
                        log_prob = torch.log(torch.ones_like(actions[:, i]).float() / self.action_dim)
                    log_probs.append(log_prob)

            avg_log_probs = torch.stack(log_probs, dim=1).mean(dim=1)
            if advantages.shape != avg_log_probs.shape:
                advantages = advantages.view_as(avg_log_probs)
            actor_loss = -(avg_log_probs * advantages.detach()).mean()

            # 负载均衡损失（与原逻辑一致）
            load_balance_loss = 0.0
            variance_list = []
            for batch_idx in range(actions.size(0)):
                uav_counts = {}
                for i in range(min(self.max_subtasks, actions.size(1))):
                    action_val = actions[batch_idx, i].item()
                    if action_val > 0:
                        uav_counts[action_val] = uav_counts.get(action_val, 0) + 1
                if uav_counts:
                    counts = list(uav_counts.values())
                    mean_count = np.mean(counts)
                    variance = np.mean([(c - mean_count) ** 2 for c in counts])
                    variance_list.append(variance)
                else:
                    variance_list.append(0.0)

            if variance_list:
                load_balance_loss_tensor = torch.tensor(variance_list, device=self.device, requires_grad=False).mean()
            else:
                load_balance_loss_tensor = torch.tensor(0.0, device=self.device, requires_grad=False)

            total_actor_loss = actor_loss + 0.1 * load_balance_loss_tensor

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            self.train_step += 1
            return actor_loss.item(), critic_loss.item()

        except Exception as e:
            print(f"[ERROR] AC更新过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    def save_models(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)

    def load_models(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def preprocess_ac_state(devices, tasks, using_uavs, uavs_coordinate_decisions, current_time_slot):
    state_vector = []

    device_features = []
    for device in devices:
        if hasattr(device, 'coordinate') and len(device.coordinate) >= 2:
            device_features.extend(device.coordinate[:2])
        else:
            device_features.extend([0.0, 0.0])

        if hasattr(device, 'fun_calculate_queue_length'):
            queue_length = device.fun_calculate_queue_length()
            device_features.append(float(queue_length))
        else:
            device_features.append(0.0)

        if hasattr(device, 'compute_resource'):
            device_features.append(device.compute_resource)
        else:
            device_features.append(DEVICE_RESOURCE)

    while len(device_features) < DEVICE_NUM * 4:
        device_features.extend([0.0, 0.0, 0.0, DEVICE_RESOURCE])

    state_vector.extend(device_features[:DEVICE_NUM * 4])

    task_features = []
    for device in devices:
        task = None
        if hasattr(device, 'task') and current_time_slot in device.task:
            task = device.task[current_time_slot]

        if task is not None:
            task_features.append(float(task.data_length))
            task_features.append(float(task.max_finish_time))
            task_features.append(float(task.generate_time))

            if hasattr(task, '_build_dag'):
                dag = task._build_dag()
                subtasks = task.subtasks

                in_degrees = [0] * 7
                out_degrees = [0] * 7
                for u, successors in dag.items():
                    out_degrees[u] = len(successors)
                    for v in successors:
                        in_degrees[v] += 1

                task_features.extend(in_degrees)
                task_features.extend(out_degrees)

                critical_path_info = compute_critical_path_info(subtasks, dag)
                task_features.extend(critical_path_info)

                for st in subtasks:
                    task_features.append(float(st.data_length))
                    task_features.append(float(st.max_finish_time))
                    task_features.append(float(len(st.predecessors)))

                    if hasattr(st, 'compute_delay'):
                        task_features.append(float(st.compute_delay))
                    else:
                        task_features.append(0.0)

                    if hasattr(st, 'finished_time'):
                        task_features.append(float(st.finished_time))
                    else:
                        task_features.append(0.0)
            else:
                task_features.extend([0.0, 0.0, 0.0])
                task_features.extend([0] * 14)
                task_features.extend([0.0] * 7)
                task_features.extend([0.0] * 35)
        else:
            task_features.extend([0.0, 0.0, 0.0])
            task_features.extend([0] * 14)
            task_features.extend([0.0] * 7)
            task_features.extend([0.0] * 35)

    expected_task_features = DEVICE_NUM * (3 + 14 + 7 + 35)
    while len(task_features) < expected_task_features:
        task_features.extend([0.0, 0.0, 0.0])
        task_features.extend([0] * 14)
        task_features.extend([0.0] * 7)
        task_features.extend([0.0] * 35)

    state_vector.extend(task_features[:expected_task_features])

    uav_features = []
    for uav in using_uavs:
        if hasattr(uav, 'coordinate') and len(uav.coordinate) >= 3:
            uav_features.extend(uav.coordinate)
        else:
            uav_features.extend([0.0, 0.0, UAV_MIN_HIGH])

        if hasattr(uav, 'fun_calculate_queue_length'):
            queue_length = uav.fun_calculate_queue_length()
            uav_features.append(float(queue_length))
        else:
            uav_features.append(0.0)

        if hasattr(uav, 'compute_resource'):
            uav_features.append(float(uav.compute_resource))
        else:
            uav_features.append(float(UAV_RESOURCE))

        if hasattr(uav, 'bandwidth'):
            uav_features.append(float(uav.bandwidth))
        else:
            uav_features.append(float(UAV_BANDWIDTH))

        if hasattr(uav, 'fun_calculate_queue_length'):
            queue_length = uav.fun_calculate_queue_length()
            max_queue_capacity = UAV_RESOURCE * UNITE_SLOT_LENGTH
            load_level = min(1.0, queue_length / max_queue_capacity) if max_queue_capacity > 0 else 0.0
            uav_features.append(float(load_level))
        else:
            uav_features.append(0.0)

        if hasattr(uav, 'queue_record') and len(uav.queue_record) > 1:
            recent_loads = list(uav.queue_record.values())[-3:]
            avg_recent_load = sum(recent_loads) / len(recent_loads) if recent_loads else 0.0
            uav_features.append(float(avg_recent_load))
        else:
            uav_features.append(0.0)

    while len(uav_features) < UAV_MAX_NUM * 8:
        uav_features.extend([0.0, 0.0, UAV_MIN_HIGH, 0.0, UAV_RESOURCE, UAV_BANDWIDTH, 0.0, 0.0])

    state_vector.extend(uav_features[:UAV_MAX_NUM * 8])

    decision_features = []
    if uavs_coordinate_decisions:
        for decision in uavs_coordinate_decisions:
            if len(decision) >= 4:
                decision_features.extend([float(decision[0]), float(decision[1]),
                                          float(decision[2]), float(decision[3])])
            else:
                decision_features.extend([0.0, 0.0, 0.0, UAV_MIN_HIGH])
    else:
        decision_features = [0.0, 0.0, 0.0, UAV_MIN_HIGH] * UAV_MAX_NUM

    while len(decision_features) < UAV_MAX_NUM * 4:
        decision_features.extend([0.0, 0.0, 0.0, UAV_MIN_HIGH])

    state_vector.extend(decision_features[:UAV_MAX_NUM * 4])

    state_vector = np.array(state_vector, dtype=np.float32)

    device_dim = DEVICE_NUM * 4
    task_dim = DEVICE_NUM * (3 + 14 + 7 + 35)
    uav_dim = UAV_MAX_NUM * 8
    decision_dim = UAV_MAX_NUM * 4
    expected_total_dim = device_dim + task_dim + uav_dim + decision_dim

    if len(state_vector) != expected_total_dim:
        print(f"AC状态向量维度: 期望={expected_total_dim}, 实际={len(state_vector)}")
        if len(state_vector) < expected_total_dim:
            padding = np.zeros(expected_total_dim - len(state_vector), dtype=np.float32)
            state_vector = np.concatenate([state_vector, padding])
        else:
            state_vector = state_vector[:expected_total_dim]

    if len(state_vector) > 0:
        mean = np.mean(state_vector)
        std = np.std(state_vector)
        if std > 1e-8:
            state_vector = (state_vector - mean) / std
        else:
            state_vector = state_vector - mean

    return state_vector


def compute_critical_path_info(subtasks, dag):
    if not subtasks:
        return [0] * 7

    earliest_start = {i: 0.0 for i in range(7)}
    for u in topological_sort(dag):
        for v in dag.get(u, []):
            earliest_start[v] = max(earliest_start[v],
                                    earliest_start[u] + subtasks[u].max_finish_time)

    latest_start = {i: float('inf') for i in range(7)}
    latest_start[6] = earliest_start[6]

    reverse_dag = build_reverse_dag(dag)
    for u in reversed(topological_sort(dag)):
        if u in reverse_dag:
            for v in reverse_dag[u]:
                latest_start[v] = min(latest_start[v],
                                      latest_start[u] - subtasks[v].max_finish_time)

    slack = [latest_start[i] - earliest_start[i] for i in range(7)]
    critical_flags = [1.0 if s <= 1e-6 else 0.0 for s in slack]

    return critical_flags


def topological_sort(dag):
    from collections import deque
    in_degree = {i: 0 for i in range(7)}
    for u in dag:
        for v in dag[u]:
            in_degree[v] += 1

    queue = deque([i for i in range(7) if in_degree[i] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in dag.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return result


def build_reverse_dag(dag):
    reverse = {}
    for u in dag:
        for v in dag[u]:
            if v not in reverse:
                reverse[v] = []
            reverse[v].append(u)
    return reverse


def get_uav_load_info(selected_uavs):
    load_info = []
    for uav in selected_uavs:
        if hasattr(uav, 'fun_calculate_queue_length'):
            queue_length = uav.fun_calculate_queue_length()
            max_queue_capacity = UAV_RESOURCE * UNITE_SLOT_LENGTH
            load_level = min(1.0, queue_length / max_queue_capacity) if max_queue_capacity > 0 else 0.0
            load_info.append(load_level)
        else:
            load_info.append(0.0)

    while len(load_info) < UAV_MAX_NUM:
        load_info.append(0.0)

    return load_info


def postprocess_ac_action(raw_actions, selected_uavs, selected_uav_indices, num_tasks):
    offload_decisions = []
    num_selected_uavs = len(selected_uavs)

    if num_selected_uavs == 0:
        print("警告：没有可用无人机，所有任务将本地计算")
        for _ in range(num_tasks):
            task_decision = [(1, 0, -1) for _ in range(7)]
            offload_decisions.append(task_decision)
        return offload_decisions

    num_subtasks_per_task = 7
    expected_length = DEVICE_NUM * num_subtasks_per_task

    if len(raw_actions) != expected_length:
        print(f"警告：AC动作长度不正确: {len(raw_actions)}，期望{expected_length}")
        if len(raw_actions) < expected_length:
            raw_actions = raw_actions + [0] * (expected_length - len(raw_actions))
        else:
            raw_actions = raw_actions[:expected_length]

    # 构建全局无人机索引到选中无人机索引的映射
    global_to_selected_map = {}
    for selected_idx, global_idx in enumerate(selected_uav_indices):
        global_to_selected_map[global_idx] = selected_idx + 1  # +1 因为0是本地计算

    for task_idx in range(num_tasks):
        task_decision = []
        start_idx = task_idx * num_subtasks_per_task

        for subtask_idx in range(num_subtasks_per_task):
            action_idx = start_idx + subtask_idx
            if action_idx < len(raw_actions):
                action = raw_actions[action_idx]
            else:
                action = 0

            if action == 0:
                lc, uc, uav_idx = 1, 0, -1
            else:
                # 检查动作是否在有效的选中无人机中
                if action in global_to_selected_map.values():
                    # 找到对应的全局索引
                    for global_idx, selected_action in global_to_selected_map.items():
                        if selected_action == action:
                            lc, uc = 0, 1
                            uav_idx = global_idx
                            break
                    else:
                        # 不应该发生，但为了安全
                        lc, uc, uav_idx = 1, 0, -1
                else:
                    # 如果动作不在选中无人机中，转为本地计算
                    print(f"警告：动作{action}不在选中无人机范围{list(global_to_selected_map.values())}中，转为本地计算")
                    lc, uc, uav_idx = 1, 0, -1

            task_decision.append((lc, uc, uav_idx))

        offload_decisions.append(task_decision)

    return offload_decisions

class ACTrainer:
    def __init__(self, state_dim, num_uavs, max_subtasks=DEVICE_NUM * 7, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action_dim = 1 + num_uavs
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            max_subtasks=max_subtasks,
            actor_lr=AC_LR_ACTOR,
            critic_lr=AC_LR_CRITIC,
            gamma=GAMMA,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            device=self.device
        )
        self.epsilon = 0.4
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.1
        self.num_uavs = num_uavs
        self.max_subtasks = max_subtasks

    def get_actions(self, state, training=True, num_using_uavs=None, uav_load_info=None):
        if training:
            actions, action_probs = self.actor_critic.select_action(state, self.epsilon, num_using_uavs, uav_load_info)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        else:
            actions, action_probs = self.actor_critic.select_action(state, 0.0, num_using_uavs, uav_load_info)
        return actions, action_probs

    def store_experience(self, state, action, reward, next_state, done):
        self.actor_critic.push_experience(state, action, reward, next_state, done)

    def update(self):
        return self.actor_critic.update()

    def save_model(self, filepath):
        self.actor_critic.save_models(filepath)

    def load_model(self, filepath):
        self.actor_critic.load_models(filepath)


class MultiTaskAC:
    def __init__(self, state_dim, num_uavs, max_subtasks=DEVICE_NUM * 7, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac_trainer = ACTrainer(state_dim, num_uavs, max_subtasks, self.device)
        self.num_uavs = num_uavs

    def get_offload_decisions(self, state, training=True, selected_uavs=None,
                              selected_uav_indices=None, num_tasks=1):
        if selected_uavs is not None:
            num_selected_uavs = len(selected_uavs)
        else:
            num_selected_uavs = self.num_uavs
            selected_uavs = [None] * self.num_uavs
            selected_uav_indices = list(range(self.num_uavs))

        uav_load_info = get_uav_load_info(selected_uavs) if selected_uavs[0] is not None else None

        actions, action_probs = self.ac_trainer.get_actions(
            state, training, num_selected_uavs, uav_load_info
        )

        offload_decisions = postprocess_ac_action(actions, selected_uavs, selected_uav_indices, num_tasks)
        return offload_decisions, action_probs

    def store_experience(self, state, action, reward, next_state, done):
        self.ac_trainer.store_experience(state, action, reward, next_state, done)

    def update(self):
        return self.ac_trainer.update()

    def save_model(self, filepath):
        self.ac_trainer.save_model(filepath)

    def load_model(self, filepath):
        self.ac_trainer.load_model(filepath)