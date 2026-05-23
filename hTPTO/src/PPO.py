# PPO.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import math
from simEnvParameter import *

Experience = namedtuple('Experience', ['state', 'action_cont', 'action_disc', 'reward', 'next_state', 'done', 'log_prob'])

class Actor(nn.Module):
    """混合动作Actor：连续轨迹 + 离散卸载决策"""
    def __init__(self, state_dim, cont_dim=UAV_MAX_NUM*4, disc_dims=(DEVICE_NUM*7, UAV_MAX_NUM+1), hidden_dim=512):
        super(Actor, self).__init__()
        self.cont_dim = cont_dim
        self.disc_dims = disc_dims  # (num_subtasks, num_actions)
        self.num_subtasks = disc_dims[0]
        self.num_actions = disc_dims[1]

        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 连续动作头：输出均值和log标准差
        self.cont_mean = nn.Linear(hidden_dim, cont_dim)
        self.cont_log_std = nn.Parameter(torch.zeros(cont_dim))

        # 离散动作头：输出每个子任务的logits
        self.disc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_subtasks * self.num_actions)
        )

    def forward(self, state):
        features = self.feature(state)
        cont_mean = self.cont_mean(features)
        cont_log_std = self.cont_log_std.expand_as(cont_mean)  # 广播

        disc_logits = self.disc_head(features)
        disc_logits = disc_logits.view(-1, self.num_subtasks, self.num_actions)
        return cont_mean, cont_log_std, disc_logits

    def sample(self, state):
        """
        采样动作，返回：
        - cont_action: 连续动作 (batch, cont_dim)
        - disc_action: 离散动作 (batch, num_subtasks)
        - log_prob: 动作的联合对数概率
        - entropy: 策略熵
        """
        cont_mean, cont_log_std, disc_logits = self.forward(state)

        # 连续部分采样
        cont_std = cont_log_std.exp()
        normal = torch.distributions.Normal(cont_mean, cont_std)
        cont_action = normal.rsample()  # 可重参数化采样
        cont_log_prob = normal.log_prob(cont_action).sum(dim=-1)

        # 离散部分采样
        disc_probs = F.softmax(disc_logits, dim=-1)
        dist = torch.distributions.Categorical(disc_probs)
        disc_action = dist.sample()  # (batch, num_subtasks)
        disc_log_prob = dist.log_prob(disc_action).sum(dim=-1)

        # 联合对数概率
        log_prob = cont_log_prob + disc_log_prob
        entropy = normal.entropy().sum(dim=-1).mean() + dist.entropy().sum(dim=-1).mean()

        return cont_action, disc_action, log_prob, entropy

    def evaluate(self, state, cont_action, disc_action):
        """
        评估给定状态和动作的对数概率、熵
        """
        cont_mean, cont_log_std, disc_logits = self.forward(state)

        # 连续部分
        cont_std = cont_log_std.exp()
        normal = torch.distributions.Normal(cont_mean, cont_std)
        cont_log_prob = normal.log_prob(cont_action).sum(dim=-1)

        # 离散部分
        disc_probs = F.softmax(disc_logits, dim=-1)
        dist = torch.distributions.Categorical(disc_probs)
        disc_log_prob = dist.log_prob(disc_action).sum(dim=-1)

        log_prob = cont_log_prob + disc_log_prob
        entropy = normal.entropy().sum(dim=-1).mean() + dist.entropy().sum(dim=-1).mean()

        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class PPOReplayBuffer:
    def __init__(self, capacity, state_dim, cont_dim, num_subtasks):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.cont_dim = cont_dim
        self.num_subtasks = num_subtasks

    def push(self, state, cont_action, disc_action, reward, next_state, done, log_prob):
        # 转换为numpy并展平
        if isinstance(state, np.ndarray):
            state = state.flatten()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.flatten()
        self.buffer.append(Experience(state, cont_action, disc_action, reward, next_state, done, log_prob))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.FloatTensor([e.state for e in batch])
        cont_actions = torch.FloatTensor([e.action_cont for e in batch])
        disc_actions = torch.LongTensor([e.action_disc for e in batch])  # (batch, num_subtasks)
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        old_log_probs = torch.FloatTensor([e.log_prob for e in batch])
        return states, cont_actions, disc_actions, rewards, next_states, dones, old_log_probs

    def __len__(self):
        return len(self.buffer)


class PPO:
    def __init__(self, state_dim, cont_dim=UAV_MAX_NUM*4, num_subtasks=DEVICE_NUM*7, num_actions=UAV_MAX_NUM+1,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, lam=0.95, clip_epsilon=0.2,
                 buffer_capacity=10000, batch_size=64, update_epochs=10, device='cpu'):
        self.device = device
        self.target_offload_ratio = TARGET_OFFLOAD_RATIO
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        self.actor = Actor(state_dim, cont_dim, (num_subtasks, num_actions)).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = PPOReplayBuffer(buffer_capacity, state_dim, cont_dim, num_subtasks)

        self.train_step = 0

    def select_action(self, state, training=True):
        """
        输入单个状态（numpy），返回动作和对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cont_action, disc_action, log_prob, _ = self.actor.sample(state_tensor)
        cont_action = cont_action.cpu().numpy().flatten()
        disc_action = disc_action.cpu().numpy().flatten()
        log_prob = log_prob.item()
        return cont_action, disc_action, log_prob

    def store_experience(self, state, cont_action, disc_action, reward, next_state, done, log_prob):
        self.buffer.push(state, cont_action, disc_action, reward, next_state, done, log_prob)

    def compute_gae(self, rewards, values, next_value, dones):
        """
        计算GAE优势
        rewards: list of length T
        values: list of length T
        next_value: float, value of next state after last step
        dones: list of bool
        """
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t+1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0, 0

        # 从buffer中收集所有数据用于计算GAE
        states, cont_actions, disc_actions, rewards, next_states, dones, old_log_probs = self.buffer.sample(len(self.buffer))

        # 将数据移至设备
        states = states.to(self.device)
        cont_actions = cont_actions.to(self.device)
        disc_actions = disc_actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # 计算价值和优势
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_value = self.critic(next_states[-1].unsqueeze(0)).squeeze().item()

        # 转换为列表
        rewards_list = rewards.cpu().tolist()
        values_list = values.cpu().tolist()
        dones_list = dones.cpu().tolist()

        advantages, returns = self.compute_gae(rewards_list, values_list, next_value, dones_list)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮更新
        actor_losses = []
        critic_losses = []
        for _ in range(self.update_epochs):
            # 随机打乱
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                batch_states = states[idx]
                batch_cont_actions = cont_actions[idx]
                batch_disc_actions = disc_actions[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                batch_old_log_probs = old_log_probs[idx]

                # 评估当前策略
                log_probs, entropy = self.actor.evaluate(batch_states, batch_cont_actions, batch_disc_actions)
                values_pred = self.critic(batch_states).squeeze()

                # 计算比率
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # 计算actor损失（clip）
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 计算critic损失
                critic_loss = F.mse_loss(values_pred, batch_returns)

                # 总损失（可选加上熵正则）
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                # 更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # 清空buffer
        self.buffer.buffer.clear()
        self.train_step += 1
        return np.mean(actor_losses), np.mean(critic_losses)

    def save_model(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


def postprocess_ppo_action(cont_action, disc_action, selected_uav_indices, num_tasks, all_uavs):
    """
    将PPO输出的连续和离散动作转换为环境需要的格式
    cont_action: (16,) 连续值，每个无人机4个 (theta1, theta2, d, h)
    disc_action: (140,) 每个子任务的设备选择 (0-4)
    selected_uav_indices: 当前时隙选中的无人机全局索引列表
    num_tasks: 当前任务数量
    all_uavs: 所有无人机对象列表（用于计算目标坐标）
    返回:
    - uav_target_coords: 列表，每个元素为 (x, y, z) 目标坐标
    - offload_decisions: 列表，每个任务包含7个子任务决策 (lc, uc, uav_idx)
    """
    import math
    # 1. 无人机目标坐标
    uav_target_coords = []
    for i in range(UAV_MAX_NUM):
        uav = all_uavs[i]
        start = i * 4
        theta1_raw = cont_action[start]
        theta2_raw = cont_action[start+1]
        d_raw = cont_action[start+2]
        h_raw = cont_action[start+3]

        # 映射到有效范围
        theta1 = round((theta1_raw + 1) * 90 / 2)
        theta1 = max(0, min(180, theta1))
        theta1 = round(theta1 / 90) * 90 % 360

        theta2 = (theta2_raw + 1) * 45
        theta2 = max(0.1, min(89.9, theta2))

        d = (d_raw + 1) * (UAV_MAX_SPEED * UNITE_SLOT_LENGTH / 2)
        d = max(UAV_MAX_SPEED * 0.1, min(d, UAV_MAX_SPEED * UNITE_SLOT_LENGTH))

        h = (h_raw + 1) / 2 * (UAV_MAX_HIGH - UAV_MIN_HIGH) + UAV_MIN_HIGH
        h = max(UAV_MIN_HIGH + 5, min(h, UAV_MAX_HIGH - 5))

        # 计算目标坐标
        dx = d * math.cos(math.radians(theta1))
        dy = d * math.sin(math.radians(theta1))
        x_new = uav.coordinate[0] + dx
        y_new = uav.coordinate[1] + dy
        z_new = h

        # 边界约束
        x_new = max(1.0, min(x_new, GROUND_LENGTH - 1.0))
        y_new = max(1.0, min(y_new, GROUND_WIDTH - 1.0))
        z_new = max(UAV_MIN_HIGH, min(z_new, UAV_MAX_HIGH))

        # 速度约束
        dx_actual = x_new - uav.coordinate[0]
        dy_actual = y_new - uav.coordinate[1]
        dz_actual = z_new - uav.coordinate[2]
        dist_actual = math.sqrt(dx_actual**2 + dy_actual**2 + dz_actual**2)
        max_allowed = UAV_MAX_SPEED * UNITE_SLOT_LENGTH
        if dist_actual > max_allowed + 1e-6:
            scale = max_allowed / dist_actual
            dx_actual *= scale
            dy_actual *= scale
            dz_actual *= scale
            x_new = uav.coordinate[0] + dx_actual
            y_new = uav.coordinate[1] + dy_actual
            z_new = uav.coordinate[2] + dz_actual
            # 再次边界裁剪
            x_new = max(1.0, min(x_new, GROUND_LENGTH - 1.0))
            y_new = max(1.0, min(y_new, GROUND_WIDTH - 1.0))
            z_new = max(UAV_MIN_HIGH, min(z_new, UAV_MAX_HIGH))

        uav_target_coords.append((x_new, y_new, z_new))

    # 2. 卸载决策
    num_selected = len(selected_uav_indices)
    offload_decisions = []
    subtasks_per_task = 7
    for task_idx in range(num_tasks):
        task_decision = []
        for subtask_idx in range(subtasks_per_task):
            global_idx = task_idx * subtasks_per_task + subtask_idx
            if global_idx >= len(disc_action):
                action = 0
            else:
                action = disc_action[global_idx]
            if action == 0:
                lc, uc, uav_idx = 1, 0, -1
            else:
                if 1 <= action <= num_selected:
                    uav_global_idx = selected_uav_indices[action - 1]
                    lc, uc, uav_idx = 0, 1, uav_global_idx
                else:
                    lc, uc, uav_idx = 1, 0, -1  # 非法则本地
            task_decision.append((lc, uc, uav_idx))
        offload_decisions.append(task_decision)

    return uav_target_coords, offload_decisions