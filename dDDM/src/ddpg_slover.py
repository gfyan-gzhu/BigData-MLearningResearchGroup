# ddpg_slover.py - 优化版本
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
from tqdm import tqdm  # 添加进度条


# ====================== 网络结构 ======================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)


# ====================== 优化后的DDPG Agent ======================
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, device='cpu'):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-5)  # 降低学习率

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)  # 降低学习率

        self.replay_buffer = deque(maxlen=100000)  # 减小经验池
        self.max_action = max_action
        self.device = device

        self.tau = 0.002
        self.gamma = 0.99  # 提高折扣因子
        self.batch_size = 128  # 减小批次大小

        self.noise_scale = 0.8  # 增加初始噪声
        self.noise_decay = 0.998  # 减缓噪声衰减
        self.min_noise = 0.2  # 提高最小噪声

        # 训练统计
        self.train_step = 0
        self.best_completion_rate = 0
        self.episode_rewards = []
        self.episode_completion_rates = []

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            noise = np.clip(noise, -self.noise_scale * 2, self.noise_scale * 2)
            action += noise
            action = np.clip(action, -self.max_action, self.max_action)
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.train_step += 1

        # 动态调整学习率 - 更平缓的衰减
        if self.train_step % 500 == 0:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = max(1e-6, param_group['lr'] * 0.998)
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = max(1e-5, param_group['lr'] * 0.998)

        batch = random.sample(self.replay_buffer, self.batch_size)
        state = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        action = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
        reward = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        done = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = reward + (1 - done) * self.gamma * self.critic_target(next_state, next_action)

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.8)
        self.critic_optimizer.step()

        # 策略延迟更新
        if self.train_step % 2 == 0:
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8)
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

    def load(self, filename, device_num=None):
        checkpoint = torch.load(filename)

        # 动态调整动作维度
        if device_num is not None:
            action_dim = 2 + device_num + 1
            state_dim = 23
            max_action = 1.0

            self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=8e-5)

            self.critic = Critic(state_dim, action_dim).to(self.device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=8e-4)

        # 加载匹配的参数
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()

        actor_loaded_state_dict = {}
        for key in actor_state_dict.keys():
            if key in checkpoint['actor'] and actor_state_dict[key].shape == checkpoint['actor'][key].shape:
                actor_loaded_state_dict[key] = checkpoint['actor'][key]
            else:
                print(f"Warning: Actor parameter {key} shape mismatch, using initialization")
                actor_loaded_state_dict[key] = actor_state_dict[key]

        critic_loaded_state_dict = {}
        for key in critic_state_dict.keys():
            if key in checkpoint['critic'] and critic_state_dict[key].shape == checkpoint['critic'][key].shape:
                critic_loaded_state_dict[key] = checkpoint['critic'][key]
            else:
                print(f"Warning: Critic parameter {key} shape mismatch, using initialization")
                critic_loaded_state_dict[key] = critic_state_dict[key]

        self.actor.load_state_dict(actor_loaded_state_dict)
        self.critic.load_state_dict(critic_loaded_state_dict)
        self.actor_target.load_state_dict(actor_loaded_state_dict)
        self.critic_target.load_state_dict(critic_loaded_state_dict)

        try:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        except:
            print("Warning: Optimizer state shape mismatch, using default optimizer state")

        self.noise_scale = checkpoint.get('noise_scale', 0.1)
        self.best_completion_rate = checkpoint.get('best_completion_rate', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_completion_rates = checkpoint.get('episode_completion_rates', [])


# ====================== 优化后的环境封装 ======================
class DDPGEnvironment:
    def __init__(self, tasks, devices, uav, time_slot):
        self.tasks = tasks
        self.devices = devices
        self.uav = uav
        self.time_slot = time_slot
        self.device_map = {d.id: d for d in devices}

    def get_state(self):
        state = []

        # 无人机位置信息
        state.append(self.uav.coordinate[0] / GROUND_LENGTH)
        state.append(self.uav.coordinate[1] / GROUND_WIDTH)

        # 任务统计信息
        state.append(len(self.tasks) / len(self.devices))

        if self.tasks:
            # 任务数据量统计
            data_lengths = [t.data_length for t in self.tasks]
            state.append(np.mean(data_lengths) / 2e6)
            state.append(np.std(data_lengths) / 2e6)
            state.append(np.max(data_lengths) / 2e6)

            # 任务完成时间统计
            finish_times = [t.max_finish_time for t in self.tasks]
            state.append(np.mean(finish_times) / 2.0)
            state.append(np.std(finish_times) / 2.0)
            state.append(np.max(finish_times) / 2.0)
        else:
            state.extend([0, 0, 0, 0, 0, 0])

        # 无人机队列信息
        state.append(min(self.uav.fun_calculate_queue_length() / 1e7, 1.0))

        # 设备位置和距离统计
        device_coords = []
        device_distances = []
        has_task_devices = 0

        for device in self.devices:
            has_task = 1.0 if device.id in [t.device_id for t in self.tasks] else 0.0
            if has_task:
                has_task_devices += 1
                device_coords.append(device.coordinate)
                dist = math.sqrt((self.uav.coordinate[0] - device.coordinate[0]) ** 2 +
                                 (self.uav.coordinate[1] - device.coordinate[1]) ** 2)
                device_distances.append(dist)

        state.append(has_task_devices / len(self.devices))

        if device_coords:
            coords_array = np.array(device_coords)
            state.append(np.mean(coords_array[:, 0]) / GROUND_LENGTH)
            state.append(np.mean(coords_array[:, 1]) / GROUND_WIDTH)
            state.append(np.std(coords_array[:, 0]) / GROUND_LENGTH)
            state.append(np.std(coords_array[:, 1]) / GROUND_WIDTH)

            state.append(np.mean(device_distances) / math.sqrt(GROUND_LENGTH ** 2 + GROUND_WIDTH ** 2))
            state.append(np.std(device_distances) / math.sqrt(GROUND_LENGTH ** 2 + GROUND_WIDTH ** 2))
            state.append(np.max(device_distances) / math.sqrt(GROUND_LENGTH ** 2 + GROUND_WIDTH ** 2))
        else:
            state.extend([0, 0, 0, 0, 0, 0, 0])

        # 任务中心信息
        if self.tasks:
            coords = [self.device_map[t.device_id].coordinate for t in self.tasks]
            center_x = np.mean([c[0] for c in coords]) / GROUND_LENGTH
            center_y = np.mean([c[1] for c in coords]) / GROUND_WIDTH
            state.extend([center_x, center_y])

            dist_to_center = math.sqrt((self.uav.coordinate[0] - center_x * GROUND_LENGTH) ** 2 +
                                       (self.uav.coordinate[1] - center_y * GROUND_WIDTH) ** 2)
            state.append(min(dist_to_center / math.sqrt(GROUND_LENGTH ** 2 + GROUND_WIDTH ** 2), 1.0))

            if len(coords) > 1:
                std_x = np.std([c[0] for c in coords]) / GROUND_LENGTH
                std_y = np.std([c[1] for c in coords]) / GROUND_WIDTH
                state.append((std_x + std_y) / 2)
            else:
                state.append(0)
        else:
            state.extend([0, 0, 0, 0])

        # 资源需求估计
        if self.tasks:
            total_compute_demand = sum([t.data_length * UAV_1BIT_CYCLE for t in self.tasks])
            state.append(min(total_compute_demand / (UAV_MAX_RESOURCE * 10), 1.0))
        else:
            state.append(0)

        return np.array(state, dtype=np.float32)

    def parse_action_optimized(self, action):
        """
        优化动作解析：提高卸载比例和资源利用率
        """
        # 1. 无人机移动向量
        move_x = action[0]
        move_y = action[1]

        # 确保在两个方向都有移动
        if abs(move_x) < 0.1 and abs(move_y) < 0.1:
            # 如果移动向量太小，添加偏向任务中心的移动
            if self.tasks:
                coords = [self.device_map[t.device_id].coordinate for t in self.tasks]
                center_x = np.mean([c[0] for c in coords])
                center_y = np.mean([c[1] for c in coords])

                dx = center_x - self.uav.coordinate[0]
                dy = center_y - self.uav.coordinate[1]
                norm = math.sqrt(dx ** 2 + dy ** 2)

                if norm > 0:
                    move_x = dx / norm * 0.8  # 增加移动权重
                    move_y = dy / norm * 0.8
            else:
                move_x = np.random.uniform(-0.5, 0.5)
                move_y = np.random.uniform(-0.5, 0.5)

        # 归一化移动向量
        move_norm = math.sqrt(move_x ** 2 + move_y ** 2)
        if move_norm > 1.0:
            move_x = move_x / move_norm
            move_y = move_y / move_norm

        # 计算最大允许移动距离
        max_move_distance = UAV_MAX_SPEED * UNITE_SLOT_LENGTH

        # 确保移动距离合理
        random_scale = np.random.uniform(0.5, 0.95)  # 增加移动范围
        actual_move_distance = max_move_distance * random_scale

        move_x = move_x * actual_move_distance
        move_y = move_y * actual_move_distance

        # 计算新位置
        current_x, current_y = self.uav.coordinate
        new_x = current_x + move_x
        new_y = current_y + move_y

        # 确保新位置在地图范围内
        new_x = np.clip(new_x, 0, GROUND_LENGTH)
        new_y = np.clip(new_y, 0, GROUND_WIDTH)

        uav_position = [new_x, new_y]

        # 2. 卸载比例 - 大幅提高默认卸载比例
        offload_ratios = []
        device_count = len(self.devices)

        for i in range(device_count):
            offload_idx = 2 + i
            if offload_idx < len(action):
                raw_value = action[offload_idx]
                # 使用更积极的sigmoid变换，提高卸载比例
                offload_ratio = 1.0 / (1.0 + np.exp(-raw_value * 1.5))  # 增加系数
                offload_ratio = np.clip(offload_ratio, 0.0, 1.0)
            else:
                offload_ratio = 0.3  # 提高默认卸载比例

            offload_ratios.append(offload_ratio)

        # 3. 无人机资源分配 - 大幅提高资源利用率
        resource_idx = 2 + device_count
        if resource_idx < len(action):
            raw_value = action[resource_idx]
            # 大幅提高资源分配，目标85%利用率
            uav_resource = 0.7 + 0.25 * (1.0 / (1.0 + np.exp(-raw_value * 2.0)))  # 70%-95%的资源
            uav_resource = uav_resource * UAV_MAX_RESOURCE
            uav_resource = np.clip(uav_resource, UAV_MAX_RESOURCE * 0.6, UAV_MAX_RESOURCE * 0.95)
        else:
            uav_resource = UAV_MAX_RESOURCE * 0.85  # 默认85%

        return uav_position, offload_ratios, uav_resource

    def compute_reward_optimized(self, action):
        """
        计算奖励
        """
        uav_position, all_offload_ratios, uav_resource = self.parse_action_optimized(action)

        # 只取当前存在任务的卸载比例
        current_offload_ratios = []
        for task in self.tasks:
            if task.device_id < len(all_offload_ratios):
                current_offload_ratios.append(all_offload_ratios[task.device_id])
            else:
                current_offload_ratios.append(0.0)

        # 计算飞行能耗
        flight_energy = flight_energy_consumption(self.uav.coordinate, uav_position)

        # 计算移动距离
        move_distance = math.sqrt(
            (uav_position[0] - self.uav.coordinate[0]) ** 2 +
            (uav_position[1] - self.uav.coordinate[1]) ** 2
        )

        total_delay = 0
        total_energy = flight_energy
        completed = 0

        # 计算任务完成情况 - 使用当前队列状态（决策前）
        for i, task in enumerate(self.tasks):
            device = self.device_map[task.device_id]
            offload_ratio = current_offload_ratios[i]

            # 计算任务时延 - 使用当前队列状态
            if offload_ratio > 0:
                device_compute_delay_val = device_compute_delay(task, offload_ratio)
                device_queue_delay_val = device_queue_delay(device)  # 当前队列状态
                transmission_delay_val = data_transmission_delay(task, offload_ratio, device, self.uav)
                uav_queue_delay_val = uav_queue_delay(self.uav, uav_resource)  # 当前队列状态
                uav_compute_delay_val = uav_compute_delay(task, offload_ratio, uav_resource)

                compute_delay = (device_compute_delay_val + device_queue_delay_val +
                                 max(transmission_delay_val, uav_queue_delay_val) +
                                 uav_compute_delay_val)
            else:
                device_compute_delay_val = device_compute_delay(task, offload_ratio)
                device_queue_delay_val = device_queue_delay(device)  # 当前队列状态
                compute_delay = device_compute_delay_val + device_queue_delay_val

            total_delay += compute_delay

            # 计算能耗
            if offload_ratio > 0:
                total_energy += (
                        uav_compute_energy(task, offload_ratio, uav_resource) +
                        uav_receive_energy(task, offload_ratio, device, self.uav)
                )

            # 检查是否满足截止时间
            if compute_delay <= task.max_finish_time:
                completed += 1

        completion_rate = completed / len(self.tasks) if self.tasks else 0
        avg_offload_ratio = np.mean(current_offload_ratios) if current_offload_ratios else 0
        avg_task_delay = total_delay / len(self.tasks) if self.tasks else 0

        # ========== 改进的奖励函数 - 重点提高完成率到85% ==========
        reward = 0.0

        # 1. 完成率奖励（最高优先级）- 大幅提高权重和奖励结构，目标85%
        if completion_rate >= 0.85:
            completion_reward = 1200.0  # 达到85%完成率给予极大奖励
        elif completion_rate >= 0.75:
            completion_reward = 800.0
        elif completion_rate >= 0.65:
            completion_reward = 500.0
        elif completion_rate >= 0.55:
            completion_reward = 300.0
        elif completion_rate >= 0.45:
            completion_reward = 150.0
        else:
            completion_reward = completion_rate * 200.0  # 基础奖励

        reward += completion_reward

        # 2. 完成率进步奖励 - 鼓励提高完成率
        if hasattr(self, 'last_completion_rate'):
            improvement = completion_rate - self.last_completion_rate
            if improvement > 0:
                reward += improvement * 600.0  # 提高进步奖励权重
        self.last_completion_rate = completion_rate

        # 3. 时延惩罚 - 适度惩罚，不要过度影响完成率
        delay_penalty = avg_task_delay * 15.0  # 进一步降低时延惩罚权重
        reward -= delay_penalty

        # 4. 移动奖励 - 鼓励有效移动
        if move_distance > 0 and self.tasks:
            movement_reward = min(move_distance * 3.0, 30.0)
            reward += movement_reward

            # 向任务中心移动的额外奖励
            coords = [self.device_map[t.device_id].coordinate for t in self.tasks]
            center_x = np.mean([c[0] for c in coords])
            center_y = np.mean([c[1] for c in coords])

            old_dist = math.sqrt((self.uav.coordinate[0] - center_x) ** 2 +
                                 (self.uav.coordinate[1] - center_y) ** 2)
            new_dist = math.sqrt((uav_position[0] - center_x) ** 2 +
                                 (uav_position[1] - center_y) ** 2)

            if new_dist < old_dist:
                improvement = (old_dist - new_dist) / (old_dist + 1e-8)
                center_reward = improvement * 100.0  # 提高中心移动奖励
                reward += center_reward

        # 5. 双向移动奖励 - 确保在x和y方向都有移动
        dx = abs(uav_position[0] - self.uav.coordinate[0])
        dy = abs(uav_position[1] - self.uav.coordinate[1])
        threshold = 0.05 * UAV_MAX_SPEED * UNITE_SLOT_LENGTH  # 降低阈值

        if dx > threshold and dy > threshold:
            reward += 40.0  # 提高双向移动奖励
        elif dx > threshold or dy > threshold:
            reward += 15.0  # 单向移动奖励

        # 6. 避免原地不动的惩罚
        if move_distance < 0.5:  # 降低惩罚阈值
            reward -= 10.0  # 降低惩罚力度

        # 7. 卸载比例奖励 - 鼓励合理卸载
        if avg_offload_ratio > 0.7:
            offload_reward = 80.0
        elif avg_offload_ratio > 0.6:
            offload_reward = 50.0
        elif avg_offload_ratio > 0.5:
            offload_reward = 30.0
        elif avg_offload_ratio > 0.4:
            offload_reward = 15.0
        else:
            offload_reward = avg_offload_ratio * 10.0
        reward += offload_reward

        # 8. 能耗惩罚（相对较轻）
        energy_penalty = total_energy * 0.00003  # 进一步降低能耗惩罚
        reward -= energy_penalty

        # 9. 资源利用率奖励 - 鼓励高资源利用率
        resource_utilization = uav_resource / UAV_MAX_RESOURCE
        if resource_utilization > 0.85:
            resource_reward = 60.0
        elif resource_utilization > 0.75:
            resource_reward = 40.0
        elif resource_utilization > 0.65:
            resource_reward = 25.0
        else:
            resource_reward = resource_utilization * 10.0
        reward += resource_reward

        # 10. 任务数量奖励 - 鼓励处理更多任务
        if len(self.tasks) > 0:
            task_count_reward = min(len(self.tasks) * 3.0, 30.0)
            reward += task_count_reward

        return reward, total_delay, total_energy, (len(self.tasks) - completed), \
               flight_energy, current_offload_ratios, uav_resource, uav_position, move_distance

    def step(self, action):
        """
        按照正确顺序执行：计算奖励 → 更新队列 → 获取新状态
        """
        # 1. 计算奖励 - 基于当前状态（决策前的队列状态）
        reward, \
        total_delay, \
        total_energy, \
        constraint_violation, \
        flight_energy, \
        current_offload_ratios, \
        uav_resource, \
        uav_position, \
        move_distance = self.compute_reward_optimized(
            action)

        # 2. 更新无人机位置
        self.uav.coordinate = uav_position

        # 3. 构建决策
        decision = (np.array(current_offload_ratios), np.float64(uav_resource))

        # 4. 队列入队出队操作 - 更新队列状态
        enqueue_dequeue(self.tasks, decision, self.devices, self.uav, self.time_slot)

        # 5. 获取下一个状态 - 使用更新后的队列状态
        next_state = self.get_state()
        done = True

        # 计算实际完成率
        actual_completed = len(self.tasks) - constraint_violation
        completion_rate = actual_completed / len(self.tasks) if self.tasks else 0

        # 计算平均时延
        avg_delay = total_delay / len(self.tasks) if self.tasks else 0

        info = {
            'offload_ratios': current_offload_ratios,
            'uav_resource': uav_resource,
            'uav_position': uav_position,
            'total_delay': total_delay,
            'avg_delay': avg_delay,  # 新增平均时延
            'total_energy': total_energy,
            'flight_energy': flight_energy,
            'constraint_violation': constraint_violation,
            'completed_count': actual_completed,
            'completion_rate': completion_rate,
            'move_distance': move_distance,
            'total_tasks': len(self.tasks)
        }
        return next_state, reward, done, info


# ====================== 决策函数 ======================
def ddpg_trajectory_and_offloading_decision(tasks, devices, uav, time_slot, ddpg_agent, training=True):
    print(f"DDPG - 当前无人机位置: {uav.coordinate}")
    print(f"++++++++++++++++++++++++++++++++++++DDPG开始 - 时隙{time_slot}")

    env = DDPGEnvironment(tasks, devices, uav, time_slot)
    state = env.get_state()

    if training:
        action = ddpg_agent.select_action(state)
    else:
        # 修复：在测试时也要将状态数据移动到正确的设备
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ddpg_agent.device)
        with torch.no_grad():
            action = ddpg_agent.actor(state_tensor).cpu().data.numpy().flatten()

    # step函数内部已经完成了：奖励计算 → 队列更新 → 获取新状态
    next_state, reward, done, info = env.step(action)

    if training:
        ddpg_agent.store_transition(state, action, reward, next_state, done)
        for _ in range(2):
            ddpg_agent.train()

    uav.coordinate = info['uav_position']
    uav.trajectory[time_slot] = uav.coordinate
    uav.flight_energy[time_slot] = info['flight_energy']

    # 计算实际移动距离
    if time_slot > 0 and (time_slot - 1) in uav.trajectory:
        prev_pos = uav.trajectory[time_slot - 1]
        move_dist = math.sqrt((uav.coordinate[0] - prev_pos[0]) ** 2 + (uav.coordinate[1] - prev_pos[1]) ** 2)
    else:
        move_dist = info.get('move_distance', 0)

    print(f"DDPG - 新无人机位置: [{uav.coordinate[0]:.2f}, {uav.coordinate[1]:.2f}]")
    print(f"DDPG - 移动距离: {move_dist:.2f}m")
    print(f"DDPG - X方向移动: {uav.coordinate[0] - uav.trajectory.get(time_slot - 1, uav.coordinate)[0]:.2f}m")
    print(f"DDPG - Y方向移动: {uav.coordinate[1] - uav.trajectory.get(time_slot - 1, uav.coordinate)[1]:.2f}m")

    offload_ratios_str = [round(r, 3) for r in info['offload_ratios']]
    print(f"DDPG - 卸载比例列表: {offload_ratios_str}")
    print(f"DDPG - 平均卸载比例: {np.mean(info['offload_ratios']):.3f}")
    print(f"DDPG - 无人机资源分配: {info['uav_resource']:.2f} (利用率: {info['uav_resource'] / UAV_MAX_RESOURCE * 100:.1f}%)")
    print(f"DDPG - 平均时延: {info['avg_delay']:.4f}s, 总能耗: {info['total_energy']:.4f}J")  # 改为平均时延
    print(f"DDPG - 飞行能耗: {info['flight_energy']:.4f}J")
    print(f"DDPG - 任务完成: {info['completed_count']}/{len(tasks)}")
    print(f"DDPG - 完成率: {info['completion_rate']:.2%}")
    print(f"DDPG - 奖励: {reward:.4f}")
    print(f"++++++++++++++++++++++++++++++++++++DDPG结束 - 时隙{time_slot}\n")

    decision = (np.array(info['offload_ratios']), np.float64(info['uav_resource']))

    return info['uav_position'], decision, reward, info


# ====================== 初始化与训练 ======================
def initialize_ddpg_agent(device='cpu', device_num=None):
    state_dim = 23
    if device_num is None:
        device_num = DEVICE_NUM
    action_dim = 2 + device_num + 1
    max_action = 1.0
    ddpg_agent = DDPG(state_dim, action_dim, max_action, device='cuda')
    ddpg_agent.device_num = device_num  # 保存设备数量信息
    print(f"DDPG Agent initialized with state_dim={state_dim}, action_dim={action_dim}, device_num={device_num}")
    return ddpg_agent


def run_ddpg_training(devices, uav, num_episodes=1000, total_time_slots=TOTAL_TIME_SLOT):
    print(f"开始DDPG训练，共{num_episodes}个episode...")
    device_num = len(devices)
    ddpg_agent = initialize_ddpg_agent(device_num=device_num)
    episode_rewards = []
    episode_completion_rates = []
    episode_offload_ratios = []
    best_reward = -float('inf')

    # 使用tqdm添加进度条
    for episode in tqdm(range(num_episodes), desc="DDPG Training Progress"):
        if episode % 50 == 0:  # 每50个episode输出详细信息
            print(f"\n=== 开始Episode {episode + 1}/{num_episodes} ===")
        else:
            print(f"\n开始Episode {episode + 1}/{num_episodes}")

        episode_devices = copy.deepcopy(devices)
        episode_uav = copy.deepcopy(uav)
        get_coordinate(episode_devices)

        episode_reward = 0
        episode_completed = 0
        episode_tasks = 0
        episode_offload_ratio = 0

        for time_slot in range(1, total_time_slots + 1):
            change_coordinate(episode_devices, time_slot)
            tasks = []
            for j, device in enumerate(episode_devices):
                if device.fun_generate_task(time_slot):
                    tasks.append(device.task[time_slot])

            if len(tasks) > 0:
                _, decision, reward, info = ddpg_trajectory_and_offloading_decision(tasks, episode_devices, episode_uav,
                                                                                    time_slot, ddpg_agent,
                                                                                    training=True)
                # 注意：不再需要外部的enqueue_dequeue调用，已在step函数内部完成
                episode_reward += reward
                episode_completed += info['completed_count']
                episode_tasks += len(tasks)
                episode_offload_ratio += np.mean(info['offload_ratios'])
            episode_uav.trajectory[time_slot] = episode_uav.coordinate

        uav_energy_record(episode_uav, episode_devices)

        # 计算episode统计
        episode_completion_rate = episode_completed / episode_tasks if episode_tasks > 0 else 0
        avg_offload_ratio = episode_offload_ratio / total_time_slots if total_time_slots > 0 else 0

        episode_rewards.append(episode_reward)
        episode_completion_rates.append(episode_completion_rate)
        episode_offload_ratios.append(avg_offload_ratio)

        # 记录到agent中用于绘图
        ddpg_agent.episode_rewards = episode_rewards
        ddpg_agent.episode_completion_rates = episode_completion_rates

        print(f"Episode {episode + 1} 完成，总奖励: {episode_reward:.4f}")
        print(f"  完成率: {episode_completion_rate:.2%}, 平均卸载比例: {avg_offload_ratio:.3f}")

        # 更新最佳完成率
        if episode_completion_rate > ddpg_agent.best_completion_rate:
            ddpg_agent.best_completion_rate = episode_completion_rate

        # 保存最佳模型
        save_model = False
        if episode_completion_rate >= 0.75 and episode_reward > best_reward:  # 降低保存阈值到75%
            best_reward = episode_reward
            save_model = True
        elif episode % 100 == 0 and episode_completion_rate > 0.65:  # 降低保存阈值
            save_model = True

        model_filename = f'ddpg_best_model_{device_num}.pth'
        if save_model:
            ddpg_agent.save(model_filename)
            print(f"  模型已保存到 {model_filename}，完成率: {episode_completion_rate:.2%}, 奖励: {episode_reward:.4f}")

        # 每50个episode输出统计信息
        if episode % 50 == 0 and episode > 0:
            recent_rewards = episode_rewards[-50:]
            recent_completions = episode_completion_rates[-50:]
            recent_offloads = episode_offload_ratios[-50:]

            avg_reward = np.mean(recent_rewards)
            avg_completion = np.mean(recent_completions)
            avg_offload = np.mean(recent_offloads)

            print(f"\n=== 最近50个episode统计 ===")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均完成率: {avg_completion:.2%}")
            print(f"  平均卸载比例: {avg_offload:.3f}")
            print(f"  当前探索噪声: {ddpg_agent.noise_scale:.4f}")

    print("DDPG训练完成!")

    # 输出最终统计
    final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_avg_completion = np.mean(episode_completion_rates[-100:]) if len(
        episode_completion_rates) >= 100 else np.mean(episode_completion_rates)
    final_avg_offload = np.mean(episode_offload_ratios[-100:]) if len(episode_offload_ratios) >= 100 else np.mean(
        episode_offload_ratios)

    print(f"\n=== 最终统计 ===")
    print(f"  最后100个episode平均奖励: {final_avg_reward:.4f}")
    print(f"  最后100个episode平均完成率: {final_avg_completion:.2%}")
    print(f"  最后100个episode平均卸载比例: {final_avg_offload:.3f}")
    print(f"  最佳完成率: {ddpg_agent.best_completion_rate:.2%}")



    return ddpg_agent, episode_rewards, episode_completion_rates, episode_offload_ratios


def run_ddpg_testing(devices, uav, ddpg_agent, total_time_slots=TOTAL_TIME_SLOT):
    print("开始DDPG测试...")
    device_num = len(devices)
    test_devices = copy.deepcopy(devices)
    test_uav = copy.deepcopy(uav)
    get_coordinate(test_devices)

    test_rewards = []
    total_tasks = 0
    total_completed = 0
    total_offload_ratio = 0
    total_delay_sum = 0  # 修改：改为时延总和
    time_slots_with_tasks = 0

    for time_slot in range(1, total_time_slots + 1):
        change_coordinate(test_devices, time_slot)
        tasks = []
        for j, device in enumerate(test_devices):
            if device.fun_generate_task(time_slot):
                tasks.append(device.task[time_slot])

        if len(tasks) > 0:
            _, decision, reward, info = ddpg_trajectory_and_offloading_decision(tasks, test_devices, test_uav,
                                                                                time_slot, ddpg_agent, training=False)
            # 注意：不再需要外部的enqueue_dequeue调用，已在step函数内部完成
            test_rewards.append(reward)
            total_tasks += len(tasks)
            total_completed += info['completed_count']
            total_offload_ratio += np.mean(info['offload_ratios'])
            total_delay_sum += info['avg_delay'] * len(tasks)  # 修改：累加时延总和
            time_slots_with_tasks += 1
        test_uav.trajectory[time_slot] = test_uav.coordinate

    uav_energy_record(test_uav, test_devices)

    completion_rate = total_completed / total_tasks if total_tasks > 0 else 0
    avg_offload_ratio = total_offload_ratio / time_slots_with_tasks if time_slots_with_tasks > 0 else 0
    avg_delay = total_delay_sum / total_tasks if total_tasks > 0 else 0  # 修改：按任务总数计算平均时延

    print(f"DDPG测试完成!")
    print(f"  总任务数: {total_tasks}")
    print(f"  完成任务数: {total_completed}")
    print(f"  任务完成率: {completion_rate:.2%}")
    print(f"  平均卸载比例: {avg_offload_ratio:.3f}")
    print(f"  平均时延: {avg_delay:.4f}s")
    print(f"  平均奖励: {np.mean(test_rewards):.4f}")
    print(f"  无人机轨迹: {test_uav.trajectory}")
    print(f"  无人机轨迹点数: {len(test_uav.trajectory)}")

    return test_devices, test_uav, test_rewards, avg_delay

def run_ddpg_experiment(devices, uav, ddpg_agent, total_time_slots=TOTAL_TIME_SLOT):
    """
    运行DDPG算法实验，确保输出信息与保存数据一致
    """
    print("开始DDPG算法实验...")
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
        'move_distances': []
    }

    for time_slot in range(1, total_time_slots + 1):
        change_coordinate(experiment_devices, time_slot)
        tasks = []
        for j, device in enumerate(experiment_devices):
            if device.fun_generate_task(time_slot):
                tasks.append(device.task[time_slot])

        if len(tasks) > 0:
            _, decision, reward, info = ddpg_trajectory_and_offloading_decision(
                tasks, experiment_devices, experiment_uav, time_slot, ddpg_agent, training=False
            )
            # 注意：不再需要外部的enqueue_dequeue调用，已在step函数内部完成

            experiment_rewards.append(reward)
            total_tasks += len(tasks)
            total_completed += info['completed_count']
            total_energy_consumption += info['total_energy']

            # 修改这里：累加所有任务的时延总和，而不是平均时延
            # info['avg_delay'] 是当前时隙内任务的平均时延，乘以任务数得到时延总和
            total_delay_sum += info['avg_delay'] * len(tasks)

            total_offload_ratio += np.mean(info['offload_ratios'])
            total_resource_utilization += info['uav_resource'] / UAV_MAX_RESOURCE
            time_slots_with_tasks += 1

            # 记录实验数据
            experiment_data['time_slots'].append(time_slot)
            experiment_data['tasks_count'].append(len(tasks))
            experiment_data['completed_tasks'].append(info['completed_count'])
            experiment_data['rewards'].append(reward)
            experiment_data['energy_consumption'].append(info['total_energy'])
            experiment_data['delays'].append(info['avg_delay'])  # 使用平均时延
            experiment_data['uav_positions'].append(info['uav_position'])
            experiment_data['offload_ratios'].append(np.mean(info['offload_ratios']))
            experiment_data['resource_utilization'].append(info['uav_resource'] / UAV_MAX_RESOURCE)
            experiment_data['move_distances'].append(info.get('move_distance', 0))

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
    print("DDPG 算法实验完成!")
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

