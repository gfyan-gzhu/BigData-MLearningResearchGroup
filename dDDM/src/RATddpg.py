# RATddpg.py - 修复版本
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
import math
from collections import deque
from simEnvParameter import *
from utils import *
from coordinateManage import get_coordinate, change_coordinate
from tqdm import tqdm


# ====================== 网络结构（严格对齐论文算法2） ======================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        # 论文中网络结构：1024, 800, 600神经元
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 论文中网络结构：1024, 800, 600神经元
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)


# ====================== 优先经验回放缓冲区（对齐论文） ======================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return [], [], [], [], [], [], []

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = self.size
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*samples)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # 避免零优先级

    def __len__(self):
        return self.size


# ====================== DDPG Agent（修复版本） ======================
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, device='cpu'):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 使用优先经验回放（论文中PER）
        self.replay_buffer = PrioritizedReplayBuffer(capacity=200000)
        self.max_action = max_action
        self.device = device

        self.tau = 0.005
        self.gamma = 0.99
        self.batch_size = 128

        # 改进的探索策略 - 增加探索
        self.noise_scale = 1.2  # 增加初始噪声
        self.noise_decay = 0.998
        self.min_noise = 0.3  # 增加最小噪声

        # 训练统计
        self.train_step = 0
        self.best_completion_rate = 0
        self.episode_rewards = []
        self.episode_completion_rates = []

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            # 增加方向多样性噪声
            angle_noise = np.random.uniform(-np.pi, np.pi)
            magnitude = np.random.normal(0, self.noise_scale)

            # 将极坐标噪声转换为直角坐标
            noise_x = magnitude * np.cos(angle_noise)
            noise_y = magnitude * np.sin(angle_noise)

            action[0] += noise_x
            action[1] += noise_y

            # 确保动作在合理范围内
            action_norm = np.linalg.norm(action[:2])
            if action_norm > 1.0:
                action[:2] = action[:2] / action_norm

            action = np.clip(action, -self.max_action, self.max_action)
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.train_step += 1

        # 动态调整学习率
        if self.train_step % 1000 == 0:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = max(1e-5, param_group['lr'] * 0.995)
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = max(1e-4, param_group['lr'] * 0.995)

        # 使用优先经验回放采样
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = reward + (1 - done) * self.gamma * self.critic_target(next_state, next_action)

        current_Q = self.critic(state, action)
        td_errors = (current_Q - target_Q).abs().cpu().data.numpy().flatten()

        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors)

        # 使用重要性采样权重的加权损失
        critic_loss = (weights * F.mse_loss(current_Q, target_Q, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 策略延迟更新
        if self.train_step % 2 == 0:
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale,
            'best_completion_rate': self.best_completion_rate,
            'episode_rewards': self.episode_rewards,
            'episode_completion_rates': self.episode_completion_rates,
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)

        # 确保网络结构一致
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        self.noise_scale = checkpoint.get('noise_scale', 0.1)
        self.best_completion_rate = checkpoint.get('best_completion_rate', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_completion_rates = checkpoint.get('episode_completion_rates', [])


# ====================== 改进的卸载匹配算法（提高卸载比例和资源分配） ======================
def matching_algorithm(tasks, devices, uav):
    """
    改进的卸载匹配算法，提高卸载比例和资源分配
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return {}, UAV_MAX_RESOURCE * 0.1, 0, 0, 0

    # 初始化关联列表和资源分配
    A = {}  # 用户关联: A[i] = 0(本地) 或 1(无人机)

    # 第一阶段：计算所有任务的本地和卸载执行指标
    task_info = []

    # 提高初始资源分配比例
    initial_resource = UAV_MAX_RESOURCE * 0.7

    for i, task in enumerate(tasks):
        device = devices[task.device_id]

        # 计算本地执行时延和是否能满足截止时间
        local_compute_delay = device_compute_delay(task, 0)
        local_queue_delay = device_queue_delay(device)
        D_local = local_compute_delay + local_queue_delay
        can_meet_local = D_local <= task.max_finish_time

        # 计算卸载到无人机的时延
        tx_delay = data_transmission_delay(task, 1, device, uav)
        compute_delay = uav_compute_delay(task, 1, initial_resource)
        queue_delay = uav_queue_delay(uav, initial_resource)
        D_offload = max(tx_delay, queue_delay) + compute_delay
        can_meet_offload = D_offload <= task.max_finish_time

        # 计算效益（时延改善）
        delay_improvement = D_local - D_offload if can_meet_offload else -float('inf')

        # 计算能耗成本
        E_local = 0  # 假设本地计算能耗较小
        transmission_energy = uav_receive_energy(task, 1, device, uav)
        compute_energy = uav_compute_energy(task, 1, initial_resource)
        E_offload = transmission_energy + compute_energy
        energy_cost = E_offload - E_local

        # 计算效益-成本比（时延改善/能耗开销）
        if energy_cost > 0 and delay_improvement > 0:
            benefit_cost_ratio = delay_improvement / energy_cost
        else:
            benefit_cost_ratio = -float('inf') if delay_improvement <= 0 else float('inf')

        task_info.append({
            'task_id': i,
            'device_id': task.device_id,
            'task': task,
            'device': device,
            'D_local': D_local,
            'D_offload': D_offload,
            'can_meet_deadline_local': can_meet_local,
            'can_meet_deadline_offload': can_meet_offload,
            'delay_improvement': delay_improvement,
            'energy_cost': energy_cost,
            'benefit_cost_ratio': benefit_cost_ratio,
            'priority': 0
        })

    # 第二阶段：任务分类和优先级分配 - 鼓励更多卸载
    critical_tasks = []  # 无法本地完成但可卸载完成的任务
    beneficial_tasks = []  # 可本地完成但卸载更优的任务
    normal_tasks = []  # 其他任务

    for info in task_info:
        if not info['can_meet_deadline_local'] and info['can_meet_deadline_offload']:
            info['priority'] = 2  # 最高优先级
            critical_tasks.append(info)
        elif info['can_meet_deadline_local'] and info['delay_improvement'] > 0 and info[
            'benefit_cost_ratio'] > 0.05:  # 降低阈值
            info['priority'] = 1  # 中等优先级
            beneficial_tasks.append(info)
        else:
            info['priority'] = 0  # 低优先级，但仍然考虑卸载
            normal_tasks.append(info)

    # 第三阶段：按优先级排序
    critical_tasks_sorted = sorted(critical_tasks, key=lambda x: x['benefit_cost_ratio'], reverse=True)
    beneficial_tasks_sorted = sorted(beneficial_tasks, key=lambda x: x['benefit_cost_ratio'], reverse=True)
    normal_tasks_sorted = sorted(normal_tasks, key=lambda x: x['benefit_cost_ratio'], reverse=True)

    all_tasks_sorted = critical_tasks_sorted + beneficial_tasks_sorted + normal_tasks_sorted

    # 第四阶段：资源约束下的任务选择 - 提高卸载比例
    current_resource = initial_resource
    selected_tasks = []

    for task_info_item in all_tasks_sorted:
        # 检查时延约束
        if not task_info_item['can_meet_deadline_offload']:
            continue

        # 提高卸载比例：选择更多任务卸载
        if len(selected_tasks) < n_tasks * 0.75:
            A[task_info_item['task_id']] = 1
            selected_tasks.append(task_info_item)
        else:
            A[task_info_item['task_id']] = 0

    # 第五阶段：动态调整资源分配 - 提高资源分配
    if selected_tasks:
        # 基于选中的任务计算所需资源
        required_resources = []
        for task_info_item in selected_tasks:
            tx_delay = data_transmission_delay(task_info_item['task'], 1, task_info_item['device'], uav)
            available_time = task_info_item['task'].max_finish_time - tx_delay
            if available_time > 0:
                min_resource = (task_info_item['task'].data_length * UAV_1BIT_CYCLE) / available_time
                required_resources.append(min_resource)

        if required_resources:
            # 使用更高的分位数作为目标资源
            target_resource = np.percentile(required_resources, 85)  # 提高到85%分位数
            F_j = min(max(target_resource, UAV_MAX_RESOURCE * 0.6), UAV_MAX_RESOURCE * 0.75)  # 提高资源分配范围
        else:
            F_j = UAV_MAX_RESOURCE * 0.7  # 提高默认资源分配
    else:
        F_j = UAV_MAX_RESOURCE * 0.1

    # 计算性能指标
    total_completed = 0
    total_delay = 0

    for i, task in enumerate(tasks):
        device = devices[task.device_id]
        if i in A and A[i] == 1:  # 卸载任务
            tx_delay = data_transmission_delay(task, 1, device, uav)
            compute_delay = uav_compute_delay(task, 1, F_j)
            queue_delay = uav_queue_delay(uav, F_j)
            task_delay = max(tx_delay, queue_delay) + compute_delay
        else:  # 本地任务
            compute_delay = device_compute_delay(task, 0)
            queue_delay = device_queue_delay(device)
            task_delay = compute_delay + queue_delay

        total_delay += task_delay
        if task_delay <= task.max_finish_time:
            total_completed += 1

    avg_delay = total_delay / n_tasks if n_tasks > 0 else 0
    completion_rate = total_completed / n_tasks if n_tasks > 0 else 0

    return A, F_j, completion_rate, total_completed, avg_delay


# ====================== 环境封装（修复版本） ======================
class RATEnvironment:
    def __init__(self, tasks, devices, uav, time_slot):
        self.tasks = tasks
        self.devices = devices
        self.uav = uav
        self.time_slot = time_slot
        self.device_map = {d.id: d for d in devices}

    def get_state(self):
        """
        获取状态信息，专注于无人机部署决策
        状态维度: 12维（对齐论文思路）
        """
        state = []

        # 1. 无人机位置 (2维) - 归一化
        state.append(self.uav.coordinate[0] / GROUND_LENGTH)
        state.append(self.uav.coordinate[1] / GROUND_WIDTH)

        # 2. 任务设备位置统计 (4维)
        task_coords = []
        task_data_sizes = []
        task_deadlines = []

        for task in self.tasks:
            device = self.device_map[task.device_id]
            task_coords.append(device.coordinate)
            task_data_sizes.append(task.data_length)
            task_deadlines.append(task.max_finish_time)

        if task_coords:
            # 任务设备平均位置 (2维)
            avg_x = np.mean([c[0] for c in task_coords]) / GROUND_LENGTH
            avg_y = np.mean([c[1] for c in task_coords]) / GROUND_WIDTH
            state.extend([avg_x, avg_y])

            # 任务设备位置标准差 (2维) - 反映设备分散程度
            std_x = np.std([c[0] for c in task_coords]) / GROUND_LENGTH
            std_y = np.std([c[1] for c in task_coords]) / GROUND_WIDTH
            state.extend([std_x, std_y])

            # 任务数据量统计 (2维)
            state.append(np.mean(task_data_sizes) / 2e6)  # 平均数据量
            state.append(np.sum(task_data_sizes) / 1e7)  # 总数据量

            # 任务紧急程度统计 (2维)
            avg_deadline = np.mean(task_deadlines) / 2.0
            min_deadline = np.min(task_deadlines) / 2.0
            state.extend([avg_deadline, min_deadline])
        else:
            # 无任务时填充0
            state.extend([0, 0, 0, 0, 0, 0, 0, 0])

        # 3. 设备到无人机的距离统计 (2维)
        distances = []
        for task in self.tasks:
            device = self.device_map[task.device_id]
            dist = math.sqrt((self.uav.coordinate[0] - device.coordinate[0]) ** 2 +
                             (self.uav.coordinate[1] - device.coordinate[1]) ** 2)
            distances.append(dist)

        if distances:
            max_dist = math.sqrt(GROUND_LENGTH ** 2 + GROUND_WIDTH ** 2)
            state.append(np.mean(distances) / max_dist)  # 平均距离
            state.append(np.min(distances) / max_dist)  # 最小距离
        else:
            state.extend([0, 0])

        return np.array(state, dtype=np.float32)

    def apply_speed_constraint(self, current_coord, target_coord):
        """改进的速度约束，避免边界停滞"""
        dx = target_coord[0] - current_coord[0]
        dy = target_coord[1] - current_coord[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        max_distance = UAV_MAX_SPEED * UNITE_SLOT_LENGTH

        if distance > max_distance:
            scale = max_distance / distance
            dx *= scale
            dy *= scale
            target_coord = [current_coord[0] + dx, current_coord[1] + dy]

        # 边界反弹机制
        new_x, new_y = target_coord
        bounce_factor = 0.8  # 反弹系数

        if new_x <= 0:
            new_x = 10  # 反弹到离边界一定距离
        elif new_x >= GROUND_LENGTH:
            new_x = GROUND_LENGTH - 10

        if new_y <= 0:
            new_y = 10
        elif new_y >= GROUND_WIDTH:
            new_y = GROUND_WIDTH - 10

        return [new_x, new_y]

    def parse_action(self, action):
        """改进的动作解析 - 鼓励移动"""
        # 动作前2个元素是无人机移动向量
        move_x = action[0]
        move_y = action[1]

        # 计算移动向量的模长
        move_norm = math.sqrt(move_x ** 2 + move_y ** 2)

        # 如果模长大于1，则归一化
        if move_norm > 1.0:
            move_x = move_x / move_norm
            move_y = move_y / move_norm

        # 计算最大允许移动距离
        max_move_distance = UAV_MAX_SPEED * UNITE_SLOT_LENGTH

        # 确保移动距离合理 - 鼓励更多移动
        actual_move_distance = max_move_distance * (0.5 + 0.5 * (move_norm if move_norm <= 1 else 1.0))

        # 将归一化向量缩放到实际移动距离
        move_x = move_x * actual_move_distance
        move_y = move_y * actual_move_distance

        # 计算新位置
        current_x, current_y = self.uav.coordinate
        new_x = current_x + move_x
        new_y = current_y + move_y

        # 确保新位置在地图范围内
        new_x = np.clip(new_x, 0, GROUND_LENGTH)
        new_y = np.clip(new_y, 0, GROUND_WIDTH)

        return [new_x, new_y]

    def compute_reward(self, uav_position):
        """
        改进的奖励函数：基于当前状态（决策前的队列状态）计算奖励
        重点：提高完成率权重，鼓励移动，提高卸载比例
        """
        # 应用速度约束
        constrained_coord = self.apply_speed_constraint(self.uav.coordinate, uav_position)

        # 计算飞行能耗
        flight_energy = flight_energy_consumption(self.uav.coordinate, constrained_coord)

        # 使用改进的匹配算法进行卸载决策和资源分配
        A, F_j, completion_rate, completed_tasks, avg_delay = matching_algorithm(self.tasks, self.devices, self.uav)

        # 构建卸载比例向量
        offload_ratios = []
        uav_resource = F_j
        total_completed = 0
        total_delay = 0
        total_energy = flight_energy

        # 计算任务完成情况 - 使用当前队列状态（决策前）
        for i, task in enumerate(self.tasks):
            device = self.device_map[task.device_id]

            if i in A and A[i] == 1:
                offload_ratio = 1.0  # 卸载到无人机
                offload_ratios.append(offload_ratio)

                # 计算时延 - 使用当前队列状态
                device_compute_delay_val = device_compute_delay(task, offload_ratio)
                device_queue_delay_val = device_queue_delay(device)  # 当前队列状态
                transmission_delay_val = data_transmission_delay(task, offload_ratio, device, self.uav)
                uav_queue_delay_val = uav_queue_delay(self.uav, uav_resource)  # 当前队列状态
                uav_compute_delay_val = uav_compute_delay(task, offload_ratio, uav_resource)

                task_delay = (device_compute_delay_val + device_queue_delay_val +
                              max(transmission_delay_val, uav_queue_delay_val) +
                              uav_compute_delay_val)

                # 计算能耗
                transmission_energy = uav_receive_energy(task, offload_ratio, device, self.uav)
                compute_energy = uav_compute_energy(task, offload_ratio, uav_resource)
                total_energy += transmission_energy + compute_energy
            else:
                offload_ratio = 0.0  # 本地计算
                offload_ratios.append(offload_ratio)

                # 计算时延 - 使用当前队列状态
                device_compute_delay_val = device_compute_delay(task, offload_ratio)
                device_queue_delay_val = device_queue_delay(device)  # 当前队列状态
                task_delay = device_compute_delay_val + device_queue_delay_val

            total_delay += task_delay
            if task_delay <= task.max_finish_time:
                total_completed += 1

        # 计算实际完成率
        actual_completion_rate = total_completed / len(self.tasks) if self.tasks else 0
        avg_delay = total_delay / len(self.tasks) if self.tasks else 0

        # 改进的奖励设计 - 重点提高完成率和鼓励移动
        reward = 0.0

        # 1. 主要奖励：任务完成率（大幅提高权重）
        completion_reward = actual_completion_rate * 500.0  # 提高权重
        reward += completion_reward

        # 2. 时延惩罚（适中惩罚，避免过度影响）
        delay_penalty = avg_delay * 20.0  # 降低惩罚权重
        reward -= delay_penalty

        # 3. 惩罚：无人机总能耗（较小权重）
        energy_penalty = -total_energy * 0.001  # 降低能耗惩罚
        reward += energy_penalty

        # 4. 移动奖励：强烈鼓励向任务中心移动
        if self.tasks:
            coords = [self.device_map[t.device_id].coordinate for t in self.tasks]
            center_x = np.mean([c[0] for c in coords])
            center_y = np.mean([c[1] for c in coords])

            old_dist = math.sqrt((self.uav.coordinate[0] - center_x) ** 2 +
                                 (self.uav.coordinate[1] - center_y) ** 2)
            new_dist = math.sqrt((constrained_coord[0] - center_x) ** 2 +
                                 (constrained_coord[1] - center_y) ** 2)

            if new_dist < old_dist:
                improvement = (old_dist - new_dist) / (old_dist + 1e-8)
                reward += improvement * 300.0  # 提高移动奖励
            else:
                # 即使远离也给予小惩罚，但不要太大
                deterioration = (new_dist - old_dist) / (new_dist + 1e-8)
                reward -= deterioration * 20.0

        # 5. 额外移动奖励：鼓励任何移动（避免停滞）
        move_distance = math.sqrt((constrained_coord[0] - self.uav.coordinate[0]) ** 2 +
                                  (constrained_coord[1] - self.uav.coordinate[1]) ** 2)
        if move_distance > 0.1:  # 如果有明显移动
            reward += move_distance * 5.0  # 根据移动距离给予奖励

        # 6. 双向移动奖励 - 确保在x和y方向都有移动
        dx = abs(uav_position[0] - self.uav.coordinate[0])
        dy = abs(uav_position[1] - self.uav.coordinate[1])
        threshold = 0.05 * UAV_MAX_SPEED * UNITE_SLOT_LENGTH  # 降低阈值

        if dx > threshold and dy > threshold:
            reward += 800.0  # 提高双向移动奖励
        elif dx > threshold or dy > threshold:
            reward += 15.0  # 单向移动奖励

        # 7. 资源利用率奖励 - 鼓励高资源利用率
        resource_utilization = uav_resource / UAV_MAX_RESOURCE
        if 0.7 <= resource_utilization <= 0.95:  # 提高目标范围
            resource_reward = 30.0
        else:
            resource_reward = -abs(resource_utilization - 0.85) * 50.0  # 调整目标值
        reward += resource_reward

        # 8. 卸载比例奖励（鼓励高卸载比例）
        offload_ratio = sum(offload_ratios) / len(offload_ratios) if offload_ratios else 0
        if 0.6 <= offload_ratio <= 0.95:  # 提高目标范围
            offload_reward = 25.0
        else:
            offload_reward = -abs(offload_ratio - 0.8) * 40.0  # 调整目标值
        reward += offload_reward

        # 边界惩罚和中心奖励
        center_x, center_y = GROUND_LENGTH / 2, GROUND_WIDTH / 2
        dist_to_center = math.sqrt((uav_position[0] - center_x) ** 2 +
                                   (uav_position[1] - center_y) ** 2)
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

        # 中心位置奖励
        center_reward = (1 - dist_to_center / max_dist) * 50.0

        # 边界惩罚
        boundary_penalty = 0
        boundary_threshold = 50  # 距离边界的阈值
        if (uav_position[0] < boundary_threshold or
                uav_position[0] > GROUND_LENGTH - boundary_threshold or
                uav_position[1] < boundary_threshold or
                uav_position[1] > GROUND_WIDTH - boundary_threshold):
            boundary_penalty = -100.0

        reward += center_reward + boundary_penalty

        return reward, constrained_coord, offload_ratios, uav_resource, flight_energy, total_energy, avg_delay, total_completed, actual_completion_rate

    def step(self, action):
        """
        按照正确顺序执行：计算奖励 → 更新队列 → 获取新状态
        """
        # 1. 计算奖励 - 基于当前状态（决策前的队列状态）
        uav_position = self.parse_action(action)
        reward, constrained_coord, offload_ratios, uav_resource, flight_energy, total_energy, avg_delay, completed_count, completion_rate = self.compute_reward(
            uav_position)

        # 2. 更新无人机位置
        self.uav.coordinate = constrained_coord

        # 3. 构建决策
        decision = (np.array(offload_ratios), np.float64(uav_resource))

        # 4. 队列入队出队操作 - 更新队列状态
        enqueue_dequeue(self.tasks, decision, self.devices, self.uav, self.time_slot)

        # 5. 获取下一个状态 - 使用更新后的队列状态
        next_state = self.get_state()
        done = False

        info = {
            'offload_ratios': offload_ratios,
            'uav_resource': uav_resource,
            'uav_position': constrained_coord,
            'flight_energy': flight_energy,
            'total_energy': total_energy,
            'avg_delay': avg_delay,
            'completed_count': completed_count,
            'completion_rate': completion_rate,
            'total_tasks': len(self.tasks)
        }

        return next_state, reward, done, info


# ====================== 决策函数（修复版本） ======================
def rat_trajectory_decision(tasks, devices, uav, time_slot, ddpg_agent, training=True):
    print(f"RAT - 当前无人机位置: {uav.coordinate}")
    print(f"++++++++++++++++++++++++++++++++++++RAT开始 - 时隙{time_slot}")

    env = RATEnvironment(tasks, devices, uav, time_slot)
    state = env.get_state()

    if training:
        action = ddpg_agent.select_action(state)
    else:
        # 确保状态数据在正确的设备上
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ddpg_agent.device)
        with torch.no_grad():
            action = ddpg_agent.actor(state_tensor).cpu().data.numpy().flatten()

    # step函数内部已经完成了：奖励计算 → 队列更新 → 获取新状态
    next_state, reward, done, info = env.step(action)

    if training:
        ddpg_agent.store_transition(state, action, reward, next_state, done)
        # 增加训练频率
        for _ in range(2):
            ddpg_agent.train()

    # 更新无人机状态
    uav.coordinate = info['uav_position']
    uav.trajectory[time_slot] = uav.coordinate
    uav.flight_energy[time_slot] = info['flight_energy']

    # 计算移动距离和方向
    if time_slot > 0 and (time_slot - 1) in uav.trajectory:
        prev_pos = uav.trajectory[time_slot - 1]
        move_dist = math.sqrt((uav.coordinate[0] - prev_pos[0]) ** 2 + (uav.coordinate[1] - prev_pos[1]) ** 2)
    else:
        move_dist = 0

    print(f"RAT - 新无人机位置: [{uav.coordinate[0]:.2f}, {uav.coordinate[1]:.2f}]")
    if time_slot > 0 and (time_slot - 1) in uav.trajectory:
        prev_pos = uav.trajectory[time_slot - 1]
        print(f"RAT - X方向移动: {uav.coordinate[0] - prev_pos[0]:.2f}m")
        print(f"RAT - Y方向移动: {uav.coordinate[1] - prev_pos[1]:.2f}m")
    print(f"RAT - 移动距离: {move_dist:.2f}m")

    # 使用改进的匹配算法进行卸载决策和资源分配
    A, F_j, completion_rate, completed_tasks, avg_delay = matching_algorithm(tasks, devices, uav)

    # 构建决策结果
    offload_ratios = []
    for i in range(len(tasks)):
        if i in A and A[i] == 1:
            offload_ratios.append(1.0)
        else:
            offload_ratios.append(0.0)

    uav_resource = F_j
    offload_count = sum(offload_ratios)
    offload_ratio = offload_count / len(offload_ratios) if offload_ratios else 0

    # 计算详细的任务完成情况
    total_completed = 0
    total_delay = 0

    for i, task in enumerate(tasks):
        device = devices[task.device_id]
        offload_ratio_val = offload_ratios[i]

        # 与task_time_energy函数相同的时延计算逻辑
        if offload_ratio_val > 0:  # 卸载任务
            device_compute_delay_val = device_compute_delay(task, offload_ratio_val)
            device_queue_delay_val = device_queue_delay(device)
            tx_delay = data_transmission_delay(task, offload_ratio_val, device, uav)
            uav_queue_delay_val = uav_queue_delay(uav, uav_resource)
            uav_compute_delay_val = uav_compute_delay(task, offload_ratio_val, uav_resource)

            task_delay = (device_compute_delay_val + device_queue_delay_val +
                          max(tx_delay, uav_queue_delay_val) + uav_compute_delay_val)
        else:  # 本地任务
            device_compute_delay_val = device_compute_delay(task, offload_ratio_val)
            device_queue_delay_val = device_queue_delay(device)
            task_delay = device_compute_delay_val + device_queue_delay_val

        total_delay += task_delay
        if task_delay <= task.max_finish_time:
            total_completed += 1

    avg_delay = total_delay / len(tasks) if tasks else 0
    completion_rate = total_completed / len(tasks) if tasks else 0

    print(f"RAT - 卸载决策: {['本地' if r == 0 else '无人机' for r in offload_ratios]}")
    print(f"RAT - 卸载任务数: {offload_count}/{len(offload_ratios)}")
    print(f"RAT - 卸载比例: {offload_ratio:.2%}")
    print(f"RAT - 无人机资源分配: {uav_resource:.2f}/{UAV_MAX_RESOURCE}")
    print(f"RAT - 资源利用率: {uav_resource / UAV_MAX_RESOURCE:.2%}")
    print(f"RAT - 飞行能耗: {info['flight_energy']:.4f}J")
    print(f"RAT - 无人机总能耗: {info['total_energy']:.4f}J")
    print(f"RAT - 平均任务时延: {avg_delay:.4f}s")
    print(f"RAT - 任务完成数: {total_completed}/{len(tasks)}")
    print(f"RAT - 任务完成率: {completion_rate:.2%}")
    print(f"RAT - 奖励: {reward:.4f}")

    decision = (np.array(offload_ratios), np.float64(uav_resource))
    return info['uav_position'], decision, reward, info


# ====================== 初始化与训练（修复版本） ======================
def initialize_rat_agent(device='cpu', device_num=None):
    """
    初始化RAT DDPG智能体
    状态维度: 12, 动作维度: 2
    """
    state_dim = 12
    action_dim = 2
    max_action = 1.0
    ddpg_agent = DDPG(state_dim, action_dim, max_action, device='cuda')
    ddpg_agent.device_num = device_num if device_num is not None else DEVICE_NUM
    print(f"RAT Agent initialized with state_dim={state_dim}, action_dim={action_dim}, device_num={ddpg_agent.device_num}")
    return ddpg_agent


def run_rat_training(devices, uav, num_episodes=1000, total_time_slots=TOTAL_TIME_SLOT):
    print(f"开始RATddpg训练，共{num_episodes}个episode...")
    device_num = len(devices)
    rat_agent = initialize_rat_agent(device_num=device_num)
    episode_rewards = []
    episode_completion_rates = []
    best_reward = -float('inf')

    # 使用tqdm进度条
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        print(f"\n=== 开始Episode {episode + 1}/{num_episodes} ===")
        episode_devices = copy.deepcopy(devices)
        episode_uav = copy.deepcopy(uav)
        get_coordinate(episode_devices)

        episode_reward = 0
        episode_completed = 0
        episode_tasks = 0

        for time_slot in range(1, total_time_slots + 1):
            change_coordinate(episode_devices, time_slot)
            tasks = []
            for j, device in enumerate(episode_devices):
                if device.fun_generate_task(time_slot):
                    tasks.append(device.task[time_slot])

            if len(tasks) > 0:
                _, decision, reward, info = rat_trajectory_decision(tasks, episode_devices, episode_uav,
                                                                    time_slot, rat_agent, training=True)
                episode_reward += reward
                episode_completed += info['completed_count']
                episode_tasks += len(tasks)
            episode_uav.trajectory[time_slot] = episode_uav.coordinate

        uav_energy_record(episode_uav, episode_devices)
        episode_rewards.append(episode_reward)

        # 计算episode完成率
        episode_completion_rate = episode_completed / episode_tasks if episode_tasks > 0 else 0
        episode_completion_rates.append(episode_completion_rate)

        # 记录到agent中用于绘图
        rat_agent.episode_rewards = episode_rewards
        rat_agent.episode_completion_rates = episode_completion_rates

        print(f"Episode {episode + 1} 完成，总奖励: {episode_reward:.4f}")
        print(f"  完成率: {episode_completion_rate:.2%}")

        # 更新最佳完成率
        if episode_completion_rate > rat_agent.best_completion_rate:
            rat_agent.best_completion_rate = episode_completion_rate

        # 保存最佳模型（降低条件以鼓励学习）
        model_filename = f'rat_best_model_{device_num}.pth'
        if episode_reward > best_reward and episode_completion_rate >= 0.5:
            best_reward = episode_reward
            rat_agent.save(model_filename)
            print(f"  新的最佳模型已保存到 {model_filename}，奖励: {best_reward:.4f}, 完成率: {episode_completion_rate:.2%}")

        # 每50个episode输出统计信息
        if episode % 50 == 0 and episode > 0:
            recent_rewards = episode_rewards[-50:]
            recent_completions = episode_completion_rates[-50:]

            avg_reward = np.mean(recent_rewards)
            avg_completion = np.mean(recent_completions)

            print(f"\n=== 最近50个episode统计 ===")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均完成率: {avg_completion:.2%}")
            print(f"  当前探索噪声: {rat_agent.noise_scale:.4f}")

    print("RATddpg训练完成!")

    # 输出最终统计
    final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_avg_completion = np.mean(episode_completion_rates[-100:]) if len(
        episode_completion_rates) >= 100 else np.mean(episode_completion_rates)

    print(f"\n=== 最终统计 ===")
    print(f"  最后100个episode平均奖励: {final_avg_reward:.4f}")
    print(f"  最后100个episode平均完成率: {final_avg_completion:.2%}")
    print(f"  最佳完成率: {rat_agent.best_completion_rate:.2%}")

    return rat_agent, episode_rewards, episode_completion_rates


def run_rat_testing(devices, uav, rat_agent, total_time_slots=TOTAL_TIME_SLOT):
    """运行RAT算法测试，确保数据一致性"""
    print("开始RAT算法测试...")
    device_num = len(devices)
    test_devices = copy.deepcopy(devices)
    test_uav = copy.deepcopy(uav)
    get_coordinate(test_devices)

    test_rewards = []
    total_tasks = 0
    total_completed = 0
    total_energy_consumption = 0
    total_offload_ratio = 0
    total_resource_utilization = 0
    total_delay_sum = 0  # 修改：改为时延总和
    time_slots_with_tasks = 0

    # 测试数据记录
    test_data = {
        'time_slots': [],
        'tasks_count': [],
        'completed_tasks': [],
        'rewards': [],
        'energy_consumption': [],
        'offload_ratios': [],
        'resource_utilization': [],
        'uav_positions': [],
        'task_delays': [],
        'completion_rates': [],
    }

    for time_slot in tqdm(range(1, total_time_slots + 1), desc="Testing Time Slots"):
        change_coordinate(test_devices, time_slot)
        tasks = []
        for j, device in enumerate(test_devices):
            if device.fun_generate_task(time_slot):
                tasks.append(device.task[time_slot])

        if len(tasks) > 0:
            try:
                _, decision, reward, info = rat_trajectory_decision(
                    tasks, test_devices, test_uav, time_slot, rat_agent, training=False
                )

                test_rewards.append(reward)
                total_tasks += len(tasks)
                total_completed += info['completed_count']
                total_energy_consumption += info['total_energy']
                total_offload_ratio += sum(info['offload_ratios']) / len(info['offload_ratios']) if info[
                    'offload_ratios'] else 0
                total_resource_utilization += info['uav_resource'] / UAV_MAX_RESOURCE
                total_delay_sum += info['avg_delay'] * len(tasks)  # 修改：累加时延总和
                time_slots_with_tasks += 1

                # 记录测试数据
                test_data['time_slots'].append(time_slot)
                test_data['tasks_count'].append(len(tasks))
                test_data['completed_tasks'].append(info['completed_count'])
                test_data['rewards'].append(reward)
                test_data['energy_consumption'].append(info['total_energy'])
                test_data['offload_ratios'].append(
                    sum(info['offload_ratios']) / len(info['offload_ratios']) if info['offload_ratios'] else 0)
                test_data['resource_utilization'].append(info['uav_resource'] / UAV_MAX_RESOURCE)
                test_data['uav_positions'].append(info['uav_position'])
                test_data['task_delays'].append(info['avg_delay'])
                test_data['completion_rates'].append(info['completion_rate'])

            except Exception as e:
                print(f"测试时隙{time_slot}发生错误: {e}")
                continue

        test_uav.trajectory[time_slot] = test_uav.coordinate

    uav_energy_record(test_uav, test_devices)

    # 计算平均值
    avg_offload_ratio = total_offload_ratio / time_slots_with_tasks if time_slots_with_tasks > 0 else 0
    avg_resource_utilization = total_resource_utilization / time_slots_with_tasks if time_slots_with_tasks > 0 else 0
    avg_delay = total_delay_sum / total_tasks if total_tasks > 0 else 0  # 修改：按任务总数计算平均时延
    completion_rate = total_completed / total_tasks if total_tasks > 0 else 0
    avg_energy_consumption = total_energy_consumption / time_slots_with_tasks if time_slots_with_tasks > 0 else 0

    # 输出测试结果
    print("\n" + "=" * 50)
    print("RAT 算法测试完成!")
    print("=" * 50)
    print(f"总任务数: {total_tasks}")
    print(f"完成任务数: {total_completed}")
    print(f"任务完成率: {completion_rate:.2%}")
    print(f"平均奖励: {np.mean(test_rewards):.4f}")
    print(f"平均无人机能耗: {avg_energy_consumption:.4f}J")
    print(f"平均卸载比例: {avg_offload_ratio * 100:.2f}%")
    print(f"平均资源利用率: {avg_resource_utilization * 100:.1f}%")
    print(f"平均任务时延: {avg_delay:.4f}s")
    print(f"无人机轨迹：{test_uav.trajectory}\n")

    return test_devices, test_uav, test_rewards, test_data, avg_delay

def run_rat_experiment(devices, uav, rat_agent, total_time_slots=TOTAL_TIME_SLOT):
    """
    运行RAT算法实验，确保输出信息与保存数据一致
    """
    print("开始RAT算法实验...")
    device_num = len(devices)
    experiment_devices = copy.deepcopy(devices)
    experiment_uav = copy.deepcopy(uav)
    get_coordinate(experiment_devices)

    experiment_rewards = []
    total_tasks = 0
    total_completed = 0
    total_energy_consumption = 0
    total_delay_sum = 0  # 改为累加所有任务的时延总和
    total_offload_ratio = 0
    total_resource_utilization = 0
    time_slots_with_tasks = 0

    # 实验数据记录
    experiment_data = {
        'time_slots': [],
        'tasks_count': [],
        'completed_tasks': [],
        'rewards': [],
        'energy_consumption': [],
        'delays': [],
        'uav_positions': [],
        'offload_ratios': [],
        'resource_utilization': [],
        'move_distances': [],
        'completion_rates': []
    }

    for time_slot in range(1, total_time_slots + 1):
        change_coordinate(experiment_devices, time_slot)
        tasks = []
        for j, device in enumerate(experiment_devices):
            if device.fun_generate_task(time_slot):
                tasks.append(device.task[time_slot])

        if len(tasks) > 0:
            _, decision, reward, info = rat_trajectory_decision(
                tasks, experiment_devices, experiment_uav, time_slot, rat_agent, training=False
            )
            # 注意：不再需要外部的enqueue_dequeue调用，已在step函数内部完成

            experiment_rewards.append(reward)
            total_tasks += len(tasks)
            total_completed += info['completed_count']
            total_energy_consumption += info['total_energy']

            # 修改这里：累加所有任务的时延总和，而不是平均时延
            # info['avg_delay'] 是当前时隙内任务的平均时延，乘以任务数得到时延总和
            total_delay_sum += info['avg_delay'] * len(tasks)

            total_offload_ratio += sum(info['offload_ratios']) / len(info['offload_ratios']) if info[
                'offload_ratios'] else 0
            total_resource_utilization += info['uav_resource'] / UAV_MAX_RESOURCE
            time_slots_with_tasks += 1

            # 记录实验数据
            experiment_data['time_slots'].append(time_slot)
            experiment_data['tasks_count'].append(len(tasks))
            experiment_data['completed_tasks'].append(info['completed_count'])
            experiment_data['rewards'].append(reward)
            experiment_data['energy_consumption'].append(info['total_energy'])
            experiment_data['delays'].append(info['avg_delay'])
            experiment_data['uav_positions'].append(info['uav_position'])
            experiment_data['offload_ratios'].append(
                sum(info['offload_ratios']) / len(info['offload_ratios']) if info['offload_ratios'] else 0)
            experiment_data['resource_utilization'].append(info['uav_resource'] / UAV_MAX_RESOURCE)
            experiment_data['completion_rates'].append(info['completion_rate'])

        experiment_uav.trajectory[time_slot] = experiment_uav.coordinate

    uav_energy_record(experiment_uav, experiment_devices)

    # 计算平均值 - 修改平均时延的计算方式
    completion_rate = total_completed / total_tasks if total_tasks > 0 else 0
    avg_offload_ratio = total_offload_ratio / time_slots_with_tasks if time_slots_with_tasks > 0 else 0
    avg_resource_utilization = total_resource_utilization / time_slots_with_tasks if time_slots_with_tasks > 0 else 0

    # 修改这里：平均时延 = 所有任务时延总和 / 总任务数
    avg_delay = total_delay_sum / total_tasks if total_tasks > 0 else 0

    avg_energy = total_energy_consumption / time_slots_with_tasks if time_slots_with_tasks > 0 else 0

    # 输出实验结果 - 确保与保存数据一致
    print("\n" + "=" * 50)
    print("RAT 算法实验完成!")
    print("=" * 50)
    print(f"总任务数: {total_tasks}")
    print(f"完成任务数: {total_completed}")
    print(f"任务完成率: {completion_rate:.2%}")
    print(f"平均时延: {avg_delay:.4f}s")
    print(f"平均奖励: {np.mean(experiment_rewards):.4f}")
    print(f"平均无人机能耗: {avg_energy:.4f}J")
    print(f"平均卸载比例: {avg_offload_ratio:.3f}")
    print(f"平均资源利用率: {avg_resource_utilization * 100:.1f}%")
    print(f"无人机轨迹：{experiment_uav.trajectory}\n")


    return experiment_devices, experiment_uav, experiment_rewards, experiment_data, avg_delay

