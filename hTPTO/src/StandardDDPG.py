import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import copy
import math
from simEnvParameter import *

from utils import *
from UAVNumberOptimize import optimize_uav_number

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# ==================== 新增：经验回放缓冲区 ====================
class ReplayBuffer:
    """经验回放缓冲区，存储并采样经验"""
    def __init__(self, capacity, device, state_dim):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一个批次的经验，并转换为张量"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量并移至设备
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ==================== 原有类定义保持不变 ====================

def combine_rewards(ddpg_reward: float, ac_reward: float) -> float:
    """
    将无人机路径奖励（DDPG reward）和任务卸载奖励（AC reward）相加，
    得到与分层强化学习方案一致的总奖励。
    在标准 DDPG 训练中，应使用此总奖励作为每一步的奖励值进行存储和优化。
    """
    return ddpg_reward + ac_reward


class Actor(nn.Module):
    """DDPG Actor网络：输入状态，输出连续动作（联合动作）"""
    def __init__(self, state_dim, action_dim, hidden_dim=512, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """DDPG Critic网络：输入状态+动作，输出Q值"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class StandardDDPG:
    """标准DDPG算法，处理联合动作"""
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 buffer_capacity=100000,
                 batch_size=128,
                 device=None):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # 网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 经验回放（使用新增的ReplayBuffer）
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.device, state_dim)

        self.train_step = 0

    def select_action(self, state, noise_scale=0.1):
        """根据当前策略选择动作（添加探索噪声）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        # 添加噪声
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)
        return action

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        try:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # 计算目标Q值
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

            # 更新Critic
            current_q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # 更新Actor
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 软更新目标网络
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            self.train_step += 1
            return critic_loss.item(), actor_loss.item()

        except Exception as e:
            print(f"[ERROR] StandardDDPG更新出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0

    def save_models(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)

    def load_models(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class StandardDDPGTrainer:
    """标准DDPG训练器，封装与环境交互的逻辑"""
    def __init__(self, state_dim=1316, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = UAV_MAX_NUM * 4 + DEVICE_NUM * 7  # 轨迹 + 卸载
        self.target_offload_ratio = TARGET_OFFLOAD_RATIO

        self.ddpg = StandardDDPG(
            state_dim=state_dim,
            action_dim=self.action_dim,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.005,
            buffer_capacity=100000,
            batch_size=128,
            device=self.device
        )

        self.noise_scale = 0.3
        self.noise_decay = 0.9995
        self.min_noise_scale = 0.05

    def get_actions(self, state, training=True):
        """返回原始连续动作"""
        if training:
            actions = self.ddpg.select_action(state, self.noise_scale)
            self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
        else:
            actions = self.ddpg.select_action(state, 0.0)
        return actions

    def postprocess_actions(self, raw_actions, selected_uav_indices, num_tasks, all_uavs):
        """
        将原始连续动作转换为轨迹决策（坐标）和卸载决策
        raw_actions: (total_action_dim,) 的numpy数组
        selected_uav_indices: 当前时隙选中的无人机全局索引列表
        num_tasks: 当前任务数量
        all_uavs: 所有无人机对象列表（用于计算目标坐标）
        返回: (uav_target_coords, ac_offload_decisions)
        """
        import math
        total_dim = len(raw_actions)
        traj_dim = UAV_MAX_NUM * 4
        offload_dim = DEVICE_NUM * 7

        # 提取轨迹部分
        traj_raw = raw_actions[:traj_dim]
        uav_target_coords = []
        num_uavs = UAV_MAX_NUM
        for i in range(num_uavs):
            uav = all_uavs[i]
            start = i * 4
            if start + 3 < len(traj_raw):
                action = traj_raw[start:start + 4]
            else:
                action = np.random.uniform(-0.8, 0.8, size=4)

            # 将动作转换为角度、距离、高度
            theta1_raw = (action[0] + 1) * 90
            theta1_raw = max(0, min(theta1_raw, 180))
            theta1 = round(theta1_raw / 90) * 90 % 360
            theta2_raw = 1 / (1 + np.exp(-action[1]))
            theta2 = theta2_raw * 90
            theta2 = max(0.1, min(theta2, 89.9))
            d_raw = action[2]
            d_sigmoid = 1 / (1 + np.exp(-d_raw))
            d = d_sigmoid * UAV_MAX_SPEED * 0.8
            d = max(UAV_MAX_SPEED * 0.1, min(d, UAV_MAX_SPEED * 0.8))
            h_raw = action[3]
            h_sigmoid = 1 / (1 + np.exp(-h_raw))
            h = UAV_MIN_HIGH + h_sigmoid * (UAV_MAX_HIGH - UAV_MIN_HIGH)
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

            # 速度约束（确保移动距离不超过最大允许）
            dx_actual = x_new - uav.coordinate[0]
            dy_actual = y_new - uav.coordinate[1]
            dz_actual = z_new - uav.coordinate[2]
            dist_actual = math.sqrt(dx_actual ** 2 + dy_actual ** 2 + dz_actual ** 2)
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

        # 处理卸载部分（保持不变）
        offload_raw = raw_actions[traj_dim:traj_dim + offload_dim]
        num_selected = len(selected_uav_indices)
        num_options = num_selected + 1
        ac_offload_decisions = []

        subtasks_per_task = 7
        for t in range(num_tasks):
            task_decision = []
            for s in range(subtasks_per_task):
                idx = t * subtasks_per_task + s
                if idx >= len(offload_raw):
                    val = 0.0
                else:
                    val = offload_raw[idx]
                bin_width = 2.0 / num_options
                bin_idx = int((val + 1.0) // bin_width)
                bin_idx = max(0, min(bin_idx, num_options - 1))
                if bin_idx == 0:
                    lc, uc, uav_idx = 1, 0, -1
                else:
                    uav_global_idx = selected_uav_indices[bin_idx - 1]
                    lc, uc, uav_idx = 0, 1, uav_global_idx
                task_decision.append((lc, uc, uav_idx))
            ac_offload_decisions.append(task_decision)

        return uav_target_coords, ac_offload_decisions

    def store_experience(self, state, action, reward, next_state, done):
        """
        存储经验到回放缓冲区。
        注意：为了与分层强化学习方案的奖励构成保持一致，reward 应使用总奖励，
        即 combine_rewards(ddpg_reward, ac_reward) 的结果。
        """
        self.ddpg.push_experience(state, action, reward, next_state, done)

    def update(self):
        return self.ddpg.update()

    def save_model(self, filepath):
        self.ddpg.save_models(filepath)

    def load_model(self, filepath):
        self.ddpg.load_models(filepath)


def preprocess_standard_state(devices, tasks, using_uavs, offload_decisions):
    """
    标准DDPG的状态预处理，复用原有的preprocess_ddpg_state
    """
    from uavTrajectoryDecision import preprocess_ddpg_state
    return preprocess_ddpg_state(devices, tasks, using_uavs, offload_decisions)