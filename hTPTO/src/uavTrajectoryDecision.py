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

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区（替代原ReplayBuffer）"""
    def __init__(self, capacity, device, state_dim=1316, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.alpha = alpha          # 优先级指数
        self.beta = beta             # 重要性采样指数
        self.beta_increment = beta_increment
        self.epsilon = 1e-6          # 防止优先级为0

    def push(self, state, action, reward, next_state, done):
        try:
            st = np.asarray(state, dtype=np.float32).flatten()
            nst = np.asarray(next_state, dtype=np.float32).flatten()
            if (st.shape == (self.state_dim,) and nst.shape == (self.state_dim,) and
                not np.any(np.isnan(st)) and not np.any(np.isnan(nst))):
                experience = (st, action, reward, nst, done)
                max_p = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
                self.tree.add(max_p, experience)
            else:
                print(f"[WARN] 丢弃无效经验: state_shape={st.shape}, next_state_shape={nst.shape}")
        except Exception as e:
            print(f"[ERROR] 处理经验时出错: {e}")

    def sample(self, batch_size):
        if self.tree.n_entries < batch_size:
            empty_state = np.zeros(self.state_dim, dtype=np.float32)
            empty_action = np.zeros(UAV_MAX_NUM * 3, dtype=np.float32)
            states = torch.FloatTensor([empty_state] * batch_size).to(self.device)
            actions = torch.FloatTensor([empty_action] * batch_size).to(self.device)
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

        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in experiences]).to(self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            priority = (np.abs(td_err) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries

def safe_projection(current_pos, target_norm, device):
    """
    输入：
        current_pos : Tensor shape (3,) 当前无人机坐标 [x,y,z]
        target_norm : Tensor shape (3,) 网络输出的归一化目标坐标 [x_norm,y_norm,z_norm]
        device      : torch.device
    返回：
        angle : 安全移动的水平角度 (弧度)
        d     : 安全移动的水平距离
        h     : 安全移动后的高度
        safe_pos : Tensor shape (3,) 安全的目标坐标
    """
    # 1. 反归一化得到实际目标坐标
    x_target = target_norm[0] * GROUND_LENGTH
    y_target = target_norm[1] * GROUND_WIDTH
    z_target = UAV_MIN_HIGH + target_norm[2] * (UAV_MAX_HIGH - UAV_MIN_HIGH)

    # 2. 计算位移向量
    dx = x_target - current_pos[0]
    dy = y_target - current_pos[1]
    dz = z_target - current_pos[2]
    dist = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

    # 3. 速度约束：若位移超过最大允许距离，则缩放
    max_speed = UAV_MAX_SPEED * UNITE_SLOT_LENGTH
    scale = torch.where(dist > max_speed, max_speed / dist, torch.tensor(1.0, device=device))
    dx_safe = dx * scale
    dy_safe = dy * scale
    dz_safe = dz * scale

    # 4. 计算安全新坐标（已满足速度约束）
    x_safe = current_pos[0] + dx_safe
    y_safe = current_pos[1] + dy_safe
    z_safe = current_pos[2] + dz_safe

    # 5. 边界约束：将坐标裁剪到允许范围内
    x_safe = torch.clamp(x_safe, 0.0, GROUND_LENGTH)
    y_safe = torch.clamp(y_safe, 0.0, GROUND_WIDTH)
    z_safe = torch.clamp(z_safe, UAV_MIN_HIGH, UAV_MAX_HIGH)

    # 6. 重新计算最终位移（可能因边界裁剪而再次改变）
    dx_final = x_safe - current_pos[0]
    dy_final = y_safe - current_pos[1]
    dz_final = z_safe - current_pos[2]
    d_final = torch.sqrt(dx_final**2 + dy_final**2 + 1e-8)
    angle_final = torch.atan2(dy_final, dx_final)
    h_final = z_safe

    # 将角度转换到 [0, 2π) 范围
    angle_final = angle_final % (2 * math.pi)

    return angle_final, d_final, h_final, torch.stack([x_safe, y_safe, z_safe])

class Actor(nn.Module):
    def __init__(self, state_dim=1316, hidden_dim=512):
        super(Actor, self).__init__()
        # 网络结构不变，但输出维度仍为 UAV_MAX_NUM * 3
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, UAV_MAX_NUM * 3)
        )

    def forward(self, state):
        raw = self.network(state)  # shape: (batch, UAV_MAX_NUM*3)
        # 使用 sigmoid 将输出压缩到 [0,1]，表示归一化目标坐标
        return torch.sigmoid(raw)


class Critic(nn.Module):
    def __init__(self, state_dim=1316, action_dim=UAV_MAX_NUM * 3, hidden_dim=512):
        super(Critic, self).__init__()
        total_input_dim = state_dim + action_dim

        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


class MADDPG:
    def __init__(self, state_dim=1316, action_dim_per_uav=3, hidden_dim=512,
                 lr_actor=DDPG_LR_ACTOR, lr_critic=DDPG_LR_CRITIC, gamma=GAMMA, tau=DDPG_TAU,
                 buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, device=None):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim_per_uav = action_dim_per_uav
        self.total_action_dim = UAV_MAX_NUM * action_dim_per_uav
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, self.device, state_dim)

        self.actor = Actor(state_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, self.total_action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.train_step = 0

        self.noise_scale = DDPG_MIN_NOISE_SCALE
        self.noise_decay = DDPG_NOISE_DECAY
        self.min_noise_scale = DDPG_MIN_NOISE_SCALE

    def select_actions(self, state, current_uavs_coords=None, noise_scale=None):
        """
        state: numpy 数组，状态向量
        current_uavs_coords: 每个无人机的当前坐标，list of [x,y,z]
        noise_scale: 探索噪声标准差（在归一化坐标空间添加）
        """
        if noise_scale is None:
            noise_scale = self.noise_scale

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 网络输出归一化目标坐标 (batch, UAV_MAX_NUM*3)
            target_norm = self.actor(state_tensor)   # shape (1, UAV_MAX_NUM*3)

        # 重塑为 (UAV_MAX_NUM, 3)
        target_norm = target_norm.view(UAV_MAX_NUM, 3)

        safe_actions = []   # 存储最终的 (angle, d, h) 列表
        for i in range(UAV_MAX_NUM):
            current_pos = current_uavs_coords[i] if current_uavs_coords is not None else [0,0,UAV_MIN_HIGH]
            current_tensor = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
            target_norm_i = target_norm[i]

            # 可选：在目标坐标上添加探索噪声（保证安全）
            if noise_scale > 0:
                noise = torch.randn(3, device=self.device) * noise_scale
                target_norm_i = torch.clamp(target_norm_i + noise, 0.0, 1.0)

            # 可微投影得到安全动作
            angle, d, h, safe_pos = safe_projection(current_tensor, target_norm_i, self.device)

            safe_actions.extend([angle.item(), d.item(), h.item()])

        return np.array(safe_actions, dtype=np.float32)

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        try:
            # 使用优先采样
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))

            current_q = self.critic(states, actions)

            # 计算带权重的 critic 损失（importance sampling weights）
            critic_loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

            # 计算 TD 误差用于更新优先级
            td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, td_errors)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # actor 损失不变（仍使用当前策略网络）
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
            print(f"[ERROR] DDPG更新过程中出错: {e}")
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


def preprocess_ddpg_state(devices, tasks, using_uavs, offload_decisions):
    # 保持原有实现不变，此处省略以节省篇幅，实际应与原文件相同
    state_vector = []
    # 1. 设备信息 (150维)
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
    expected_device_dim = DEVICE_NUM * 3
    while len(device_features) < expected_device_dim:
        device_features.append(0.0)
    state_vector.extend(device_features[:expected_device_dim])

    # 2. 任务信息 (100维)
    task_features = []
    for device in devices:
        device_has_task = False
        for task in tasks:
            if hasattr(task, 'device_id') and task.device_id == device.id:
                task_features.append(float(task.data_length))
                task_features.append(float(task.max_finish_time))
                device_has_task = True
                break
        if not device_has_task:
            task_features.extend([0.0, 0.0])
    expected_task_dim = DEVICE_NUM * 2
    while len(task_features) < expected_task_dim:
        task_features.extend([0.0, 0.0])
    state_vector.extend(task_features[:expected_task_dim])

    # 3. 无人机信息 (16维)
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
    expected_uav_dim = UAV_MAX_NUM * 4
    while len(uav_features) < expected_uav_dim:
        uav_features.extend([0.0, 0.0, UAV_MIN_HIGH, 0.0])
    state_vector.extend(uav_features[:expected_uav_dim])

    # 4. 卸载决策信息 (1050维)
    decision_features = []
    if offload_decisions and len(offload_decisions) > 0:
        for decision_list in offload_decisions:
            if decision_list:
                for decision in decision_list:
                    if len(decision) >= 3:
                        decision_features.extend([float(decision[0]), float(decision[1]), float(decision[2])])
                    else:
                        decision_features.extend([0.0, 0.0, -1.0])
            else:
                decision_features.extend([0.0, 0.0, -1.0] * 7)
    else:
        decision_features = [0.0, 0.0, -1.0] * DEVICE_NUM * 7
    expected_decision_dim = DEVICE_NUM * 7 * 3
    while len(decision_features) < expected_decision_dim:
        decision_features.extend([0.0, 0.0, -1.0])
    state_vector.extend(decision_features[:expected_decision_dim])

    state_vector = np.array(state_vector, dtype=np.float32)
    mean = np.mean(state_vector)
    std = np.std(state_vector)
    if std > 1e-8:
        state_vector = (state_vector - mean) / std
    else:
        state_vector = state_vector - mean
    return state_vector


def postprocess_ddpg_action(raw_actions, all_uavs, current_time_slot):
    """
    将DDPG输出的原始动作（angle, d, h）转换为安全的目标坐标列表。
    返回列表，每个元素为 (x, y, z)，即无人机下一时刻的安全坐标。
    """
    uav_target_coords = []
    num_all_uavs = len(all_uavs)

    raw_actions = np.asarray(raw_actions).flatten()
    expected_actions_length = UAV_MAX_NUM * 3

    if len(raw_actions) != expected_actions_length:
        print(f"警告：DDPG动作维度不正确: {len(raw_actions)}，期望{expected_actions_length}")
        # 生成随机安全动作
        raw_actions = []
        for i in range(UAV_MAX_NUM):
            angle = np.random.uniform(0, 2 * math.pi)
            d = np.random.uniform(0, UAV_MAX_SPEED * UNITE_SLOT_LENGTH)
            h = np.random.uniform(UAV_MIN_HIGH, UAV_MAX_HIGH)
            raw_actions.extend([angle, d, h])
        raw_actions = np.array(raw_actions)

    for i in range(num_all_uavs):
        uav = all_uavs[i]
        base = i * 3
        angle = raw_actions[base]
        d = raw_actions[base + 1]
        h = raw_actions[base + 2]

        # 计算位移
        dx = d * math.cos(angle)
        dy = d * math.sin(angle)
        dz = h - uav.coordinate[2]

        # 计算新坐标（未约束）
        x_new = uav.coordinate[0] + dx
        y_new = uav.coordinate[1] + dy
        z_new = h

        # 边界约束
        x_new = max(1.0, min(x_new, GROUND_LENGTH - 1.0))
        y_new = max(1.0, min(y_new, GROUND_WIDTH - 1.0))
        z_new = max(UAV_MIN_HIGH, min(z_new, UAV_MAX_HIGH))

        # 重新计算实际位移
        dx_actual = x_new - uav.coordinate[0]
        dy_actual = y_new - uav.coordinate[1]
        dz_actual = z_new - uav.coordinate[2]
        dist_actual = math.sqrt(dx_actual**2 + dy_actual**2 + dz_actual**2)

        # 速度约束
        max_allowed = UAV_MAX_SPEED * UNITE_SLOT_LENGTH
        if dist_actual > max_allowed + 1e-6:  # 增加容差
            scale = max_allowed / dist_actual
            dx_actual *= scale
            dy_actual *= scale
            dz_actual *= scale
            x_new = uav.coordinate[0] + dx_actual
            y_new = uav.coordinate[1] + dy_actual
            z_new = uav.coordinate[2] + dz_actual
            # 再次边界裁剪（防止缩放后越界）
            x_new = max(1.0, min(x_new, GROUND_LENGTH - 1.0))
            y_new = max(1.0, min(y_new, GROUND_WIDTH - 1.0))
            z_new = max(UAV_MIN_HIGH, min(z_new, UAV_MAX_HIGH))

        uav_target_coords.append((x_new, y_new, z_new))

    return uav_target_coords


class DDPGTrainer:
    def __init__(self, state_dim=1316, action_dim_per_uav=3, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.maddpg = MADDPG(
            state_dim=state_dim,
            action_dim_per_uav=action_dim_per_uav,
            hidden_dim=512,
            lr_actor=DDPG_LR_ACTOR,
            lr_critic=DDPG_LR_CRITIC,
            gamma=GAMMA,
            tau=DDPG_TAU,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            device=self.device
        )

        self.noise_scale = self.maddpg.noise_scale
        self.noise_decay = self.maddpg.noise_decay
        self.min_noise_scale = self.maddpg.min_noise_scale

    def get_actions(self, state, uavs_coords=None, training=True):
        if training:
            actions = self.maddpg.select_actions(state, uavs_coords, self.noise_scale)
            self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
        else:
            actions = self.maddpg.select_actions(state, uavs_coords, 0.0)
        return actions

    def store_experience(self, state, action, reward, next_state, done):
        state_np = np.asarray(state).flatten()
        next_state_np = np.asarray(next_state).flatten()

        if state_np.shape[0] != 1316 or next_state_np.shape[0] != 1316:
            print(f"[WARNING] 状态维度不一致: state={state_np.shape[0]} next_state={next_state_np.shape[0]}")
            return

        self.maddpg.push_experience(state_np, action, reward, next_state_np, done)

    def update(self):
        return self.maddpg.update()

    def save_model(self, filepath):
        self.maddpg.save_models(filepath)

    def load_model(self, filepath):
        self.maddpg.load_models(filepath)