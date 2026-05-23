# ==================== DDPG_LDPG.py ====================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy
from collections import deque, namedtuple

from UAVNumberOptimize import optimize_uav_number
from env_main import MultiUavEnv
from simEnvParameter import *
from utils import *
from uavTrajectoryDecision import postprocess_ddpg_action, preprocess_ddpg_state

# -------------------- 经验回放（支持序列） --------------------
SequenceExperience = namedtuple('SequenceExperience',
                                ['state_seq', 'action', 'reward', 'next_state', 'done'])

class SequenceReplayBuffer:
    def __init__(self, capacity, device, state_dim=1316, seq_len=4):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.state_dim = state_dim
        self.seq_len = seq_len

    def push(self, state_seq, action, reward, next_state, done):
        if len(state_seq) != self.seq_len:
            return
        self.buffer.append(SequenceExperience(state_seq, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"经验不足: {len(self.buffer)} < {batch_size}")
        experiences = random.sample(self.buffer, batch_size)
        states_seq = torch.FloatTensor(np.array([e.state_seq for e in experiences])).to(self.device)
        actions = torch.FloatTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(self.device)
        return states_seq, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# -------------------- DDPG 模块：无人机轨迹决策 --------------------
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class DDPGTrajectory:
    def __init__(self, state_dim, action_dim,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_capacity=100000, batch_size=64, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = DDPGActor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = deque(maxlen=buffer_capacity)

    def select_action(self, state, noise_scale=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)
        return action

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item()

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic_target.load_state_dict(ckpt['critic_target'])


# -------------------- LDPG 模块：任务卸载决策（使用LSTM） --------------------
class LSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, lstm_layers=2, seq_len=4):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(state_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state_seq):
        lstm_out, (h_n, c_n) = self.lstm(state_seq)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)

class LSTMCritic(nn.Module):
    def __init__(self, state_dim, action_embed_dim, hidden_dim=256, lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_seq, action_embed):
        lstm_out, (h_n, c_n) = self.lstm(state_seq)
        last_hidden = h_n[-1]
        x = torch.cat([last_hidden, action_embed], dim=-1)
        return self.fc(x)

class LDPGOffload:
    def __init__(self, state_dim, num_subtasks, num_uavs, seq_len=4,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_capacity=50000, batch_size=32, device='cpu'):
        self.device = device
        self.num_subtasks = num_subtasks
        self.num_uavs = num_uavs
        self.action_dim = num_subtasks * (num_uavs + 1)
        self.seq_len = seq_len
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = LSTMActor(state_dim, self.action_dim, seq_len=seq_len).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = LSTMCritic(state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = SequenceReplayBuffer(buffer_capacity, device, state_dim, seq_len)

    def _action_to_onehot(self, actions):
        batch_size = actions.shape[0]
        onehot = torch.zeros(batch_size, self.num_subtasks, self.num_uavs + 1, device=self.device)
        onehot.scatter_(2, actions.unsqueeze(-1).long(), 1)
        return onehot.view(batch_size, -1)

    def select_action(self, state_seq, training=True, available_uavs=None, epsilon=0.2):
        seq_tensor = torch.FloatTensor(np.array(state_seq)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(seq_tensor).cpu().numpy()[0]
        logits = logits.reshape(self.num_subtasks, self.num_uavs + 1)

        if available_uavs is not None:
            mask = np.zeros((self.num_subtasks, self.num_uavs + 1), dtype=bool)
            mask[:, 0] = True
            for uav_idx in available_uavs:
                if 0 <= uav_idx < self.num_uavs:
                    mask[:, uav_idx + 1] = True
            logits[~mask] = -1e9

        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        if training:
            actions = np.array([np.random.choice(self.num_uavs + 1, p=probs[i]) for i in range(self.num_subtasks)])
        else:
            actions = np.argmax(probs, axis=-1)
        return actions

    def push_experience(self, state_seq, action, reward, next_state, done):
        self.replay_buffer.push(state_seq, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        states_seq, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        actions = actions.long()
        action_embed = self._action_to_onehot(actions).detach()

        with torch.no_grad():
            next_logits = self.actor_target(states_seq)
            next_probs = F.softmax(next_logits.view(-1, self.num_subtasks, self.num_uavs + 1), dim=-1)
            next_actions = next_probs.argmax(dim=-1)
            next_action_embed = self._action_to_onehot(next_actions)
            target_q = self.critic_target(states_seq, next_action_embed)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

        current_q = self.critic(states_seq, action_embed)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        logits = self.actor(states_seq)
        probs = F.softmax(logits.view(-1, self.num_subtasks, self.num_uavs + 1), dim=-1)
        max_actions = probs.argmax(dim=-1).detach()
        max_action_embed = self._action_to_onehot(max_actions)
        actor_loss = -self.critic(states_seq, max_action_embed).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item()

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic_target.load_state_dict(ckpt['critic_target'])


# -------------------- DDPG-LDPG 联合算法 --------------------
class DDPG_LDPG_Algorithm:
    def __init__(self, total_episodes=None, device=None, target_offload_ratio=TARGET_OFFLOAD_RATIO):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = MultiUavEnv(time_slot=1)
        self.total_episodes = total_episodes if total_episodes is not None else TOTAL_EPISODES
        self.target_offload_ratio = target_offload_ratio

        self.state_dim = 1316
        self.num_uavs = UAV_MAX_NUM
        self.num_subtasks = DEVICE_NUM * 7

        self.traj_action_dim = self.num_uavs * 4
        self.offload_action_dim = self.num_subtasks * (self.num_uavs + 1)

        # 初始化两个模块
        self.ddpg_traj = DDPGTrajectory(
            state_dim=self.state_dim,
            action_dim=self.traj_action_dim,
            device=self.device
        )
        self.ldpg_offload = LDPGOffload(
            state_dim=self.state_dim,
            num_subtasks=self.num_subtasks,
            num_uavs=self.num_uavs,
            seq_len=4,
            device=self.device
        )

        self.fix_ddpg = False
        self.alternate_interval = ALTERNATE_INTERCAL
        self.ddpg_update_count = 0
        self.ldpg_update_count = 0

        self.state_history = deque(maxlen=self.ldpg_offload.seq_len)
        self.rewards_per_episode = []  # 记录每个 episode 的平均总奖励

    def preprocess_state(self, devices, tasks, uavs, offload_decisions):
        return preprocess_ddpg_state(devices, tasks, uavs, offload_decisions)

    def offload_action_to_decisions(self, actions, tasks, selected_uav_indices):
        decisions = []
        subtask_idx = 0
        for task in tasks:
            task_dec = []
            for _ in task.subtasks:
                target = actions[subtask_idx]
                if target == 0:
                    task_dec.append((1, 0, -1))
                else:
                    uav_global_idx = target - 1
                    if uav_global_idx in selected_uav_indices:
                        task_dec.append((0, 1, uav_global_idx))
                    else:
                        task_dec.append((1, 0, -1))
                subtask_idx += 1
            decisions.append(task_dec)
        return decisions

    def train(self):
        print("开始 DDPG-LDPG 交替训练...")
        self.state_history.clear()

        for episode in range(1, self.total_episodes + 1):
            print(f"\n{'='*60}")
            print(f"Episode {episode}")
            self.env.reset(time_slot=1)
            self.state_history.clear()

            if episode % self.alternate_interval == 0:
                self.fix_ddpg = not self.fix_ddpg
                mode = "固定DDPG训练LDPG" if self.fix_ddpg else "固定LDPG训练DDPG"
                print(f"切换模式: {mode}")

            episode_total_reward = 0  # 累积总奖励 (ddpg_reward + ac_reward)
            episode_steps = 0
            prev_offload_decisions = []

            for time_slot in range(1, TOTAL_TIME_SLOT + 1):
                print(f"\n--- 时隙 {time_slot} ---")

                current_devices = self.env.devices
                current_hasTaskDevices = self.env.hasTaskDevices
                current_tasks = self.env.tasks
                current_all_uavs = self.env.all_uavs

                optimized_uav_num, selected_uav_indices = optimize_uav_number(
                    current_hasTaskDevices, current_all_uavs, time_slot
                )
                selected_uavs = [current_all_uavs[i] for i in selected_uav_indices]
                self.env.uav_num = optimized_uav_num
                self.env.selected_uavs = selected_uavs
                self.env.selected_uav_indices = selected_uav_indices

                if time_slot == 1 or len(prev_offload_decisions) == 0:
                    random_offload = random_offload_decisions(current_tasks, optimized_uav_num,
                                                               current_devices, selected_uavs)
                    state = self.preprocess_state(current_hasTaskDevices, current_tasks,
                                                  current_all_uavs, [random_offload])
                else:
                    state = self.preprocess_state(current_hasTaskDevices, current_tasks,
                                                  current_all_uavs, prev_offload_decisions)

                self.state_history.append(state)
                while len(self.state_history) < self.ldpg_offload.seq_len:
                    self.state_history.append(state)
                state_seq = list(self.state_history)

                # 根据当前模式生成动作
                if self.fix_ddpg:
                    # 固定 DDPG，训练 LDPG
                    with torch.no_grad():
                        traj_actions = self.ddpg_traj.select_action(state, noise_scale=0.0)
                    offload_actions = self.ldpg_offload.select_action(state_seq, training=True,
                                                                       available_uavs=selected_uav_indices,
                                                                       epsilon=0.2)
                else:
                    # 固定 LDPG，训练 DDPG
                    with torch.no_grad():
                        offload_actions = self.ldpg_offload.select_action(state_seq, training=False,
                                                                           available_uavs=selected_uav_indices)
                    traj_actions = self.ddpg_traj.select_action(state, noise_scale=0.2)

                # 后处理得到决策
                offload_decisions = self.offload_action_to_decisions(offload_actions, current_tasks,
                                                                      selected_uav_indices)
                traj_decisions = postprocess_ddpg_action(traj_actions, current_all_uavs, time_slot)

                # 环境 step，返回上下层奖励
                ddpg_reward, ac_reward, done = self.env.step(traj_decisions, offload_decisions, time_slot)

                total_reward = ddpg_reward + ac_reward  # 用于记录和保存

                # 计算下一状态
                next_state = self.preprocess_state(self.env.hasTaskDevices, self.env.tasks,
                                                   self.env.all_uavs, offload_decisions)

                # 根据当前模式存储经验，使用对应的奖励
                if self.fix_ddpg:
                    # 训练 LDPG，使用 ac_reward
                    self.ldpg_offload.push_experience(state_seq, offload_actions, ac_reward, next_state, done)
                    self.ldpg_update_count += 1
                else:
                    # 训练 DDPG，使用 ddpg_reward
                    self.ddpg_traj.push_experience(state, traj_actions, ddpg_reward, next_state, done)
                    self.ddpg_update_count += 1

                # 定期更新网络
                if not self.fix_ddpg and time_slot % 3 == 0:
                    cl, al = self.ddpg_traj.update()
                    if cl:
                        print(f"DDPG轨迹更新: Critic {cl:.6f}, Actor {al:.6f}")
                elif self.fix_ddpg and time_slot % 3 == 0:
                    cl, al = self.ldpg_offload.update()
                    if cl:
                        print(f"LDPG卸载更新: Critic {cl:.6f}, Actor {al:.6f}")

                episode_total_reward += total_reward
                episode_steps += 1
                prev_offload_decisions = offload_decisions

                print(f"时隙总奖励: {total_reward:.6f} (DDPG: {ddpg_reward:.6f}, AC: {ac_reward:.6f})")

            avg_total_reward = episode_total_reward / max(episode_steps, 1)
            self.rewards_per_episode.append(avg_total_reward)
            print(f"Episode {episode} 平均总奖励: {avg_total_reward:.6f}")

        # 训练结束，保存最终模型
        self._save_final_models()
        return self.rewards_per_episode

    def _save_final_models(self):
        # DDPG 学习率
        ddpg_actor_lr = self.ddpg_traj.actor_optimizer.param_groups[0]['lr']
        ddpg_critic_lr = self.ddpg_traj.critic_optimizer.param_groups[0]['lr']
        # LDPG 学习率
        ldpg_actor_lr = self.ldpg_offload.actor_optimizer.param_groups[0]['lr']
        ldpg_critic_lr = self.ldpg_offload.critic_optimizer.param_groups[0]['lr']

        def fmt_lr(lr):
            s = f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
            return s

        ddpg_actor_str = fmt_lr(ddpg_actor_lr)
        ddpg_critic_str = fmt_lr(ddpg_critic_lr)
        ldpg_actor_str = fmt_lr(ldpg_actor_lr)
        ldpg_critic_str = fmt_lr(ldpg_critic_lr)
        target_str = f"{self.target_offload_ratio:.2f}".replace('.', '_')

        # DDPG 模型文件名
        ddpg_filename = f"DDPG_LDPG_ddpg_actor{ddpg_actor_str}_critic{ddpg_critic_str}_target{target_str}.pth"
        self.ddpg_traj.save_model(f'models/{ddpg_filename}')

        # LDPG 模型文件名
        ldpg_filename = f"DDPG_LDPG_ldpg_actor{ldpg_actor_str}_critic{ldpg_critic_str}_target{target_str}.pth"
        self.ldpg_offload.save_model(f'models/{ldpg_filename}')

        print(f"最终模型已保存: {ddpg_filename}, {ldpg_filename}")

    def save_model(self, path_prefix):
        # 旧方法，不再使用
        pass

    def load_model(self, path_prefix):
        # 旧方法，不再使用
        pass