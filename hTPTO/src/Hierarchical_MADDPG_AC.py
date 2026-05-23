import numpy as np
import torch
from env_main import MultiUavEnv
from UAVNumberOptimize import optimize_uav_number
from uavTrajectoryDecision import *
from OffloadingResourceDecision import MultiTaskAC, preprocess_ac_state, postprocess_ac_action
from utils import *
import os
from collections import deque
from uavTrajectoryDecision import SumTree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

KEEP_LAST_EXPERIENCE = 1000

class HierarchicalRLAlgorithm:
    def __init__(self,
                 ddpg_state_dim=1316,
                 ac_state_dim=None,
                 ddpg_action_dim_per_uav=3,
                 ac_action_dim=5,
                 max_subtasks=DEVICE_NUM * 7,
                 total_episodes=None,
                 target_offload_ratio=TARGET_OFFLOAD_RATIO,
                 keep_last_experience=KEEP_LAST_EXPERIENCE,
                 device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if isinstance(device, torch.device):
                self.device = device
            else:
                # 尝试将传入的值（如字符串）转换为 torch.device
                self.device = torch.device(device)

        self.env = MultiUavEnv(time_slot=1)
        self.device = device
        self.target_offload_ratio = target_offload_ratio
        self.keep_last_experience = keep_last_experience

        self.alternate_update_counter = 0  # 用于跟踪主网络更新次数
        self.alternate_update_interval = ALTERNATE_UPDATE_INTERVAL  # 每5次主网络更新，执行一次另一网络更新
        self._original_ddpg_actor_lr = None  # 用于保存原始学习率
        self._original_ddpg_critic_lr = None
        self._original_ac_actor_lr = None
        self._original_ac_critic_lr = None

        if ac_state_dim is None:
            ac_state_dim = self._calculate_ac_state_dim()

        print(f"AC状态维度: {ac_state_dim}")
        print(f"DDPG状态维度: {ddpg_state_dim}")
        print(f"最大子任务数量: {max_subtasks}")

        self.ddpg_trainer = DDPGTrainer(
            state_dim=ddpg_state_dim,
            action_dim_per_uav=ddpg_action_dim_per_uav,
            device=self.device
        )

        self.ac_trainer = MultiTaskAC(
            state_dim=ac_state_dim,
            num_uavs=UAV_MAX_NUM,
            max_subtasks=max_subtasks,
            device=self.device
        )

        self.total_episodes = total_episodes if total_episodes is not None else TOTAL_EPISODES
        self.current_episode = 0

        self.ddpg_success_count = 0
        self.ac_success_count = 0
        self.total_steps = 0

        self.fix_ddpg_train_ac = True
        self.alternate_interval = ALTERNATE_INTERCAL
        self.ac_update_count = 0
        self.ddpg_update_count = 0

        self.uav_task_stats = {i: 0.0 for i in range(UAV_MAX_NUM)}
        self.uav_subtask_counts = {i: 0 for i in range(UAV_MAX_NUM)}

        os.makedirs('models', exist_ok=True)

        self._initialize_networks()

        # 记录每个episode的平均总奖励和平均AC奖励
        self.ac_rewards_per_episode = []
        self.total_rewards_per_episode = []   # 新增：记录总平均奖励

        self.best_ddpg_model_path = 'models/best_ddpg_temp.pth'
        self.best_ac_model_path = 'models/best_ac_temp.pth'

    def _set_finetune_lr(self, mode):
        """
        根据当前模式设置学习率：
        mode='fix_ddpg' 表示固定DDPG训练AC，则DDPG微调（低学习率），AC恢复原学习率
        mode='fix_ac'   表示固定AC训练DDPG，则AC微调（低学习率），DDPG恢复原学习率
        """
        finetune_scale = 0.1  # 微调学习率为原学习率的10%

        # 保存原始学习率（仅在首次调用时保存）
        if self._original_ddpg_actor_lr is None:
            self._original_ddpg_actor_lr = self.ddpg_trainer.maddpg.actor_optimizer.param_groups[0]['lr']
            self._original_ddpg_critic_lr = self.ddpg_trainer.maddpg.critic_optimizer.param_groups[0]['lr']
            self._original_ac_actor_lr = self.ac_trainer.ac_trainer.actor_critic.actor_optimizer.param_groups[0]['lr']
            self._original_ac_critic_lr = self.ac_trainer.ac_trainer.actor_critic.critic_optimizer.param_groups[0]['lr']

        if mode == 'fix_ddpg':
            # DDPG 微调
            for param_group in self.ddpg_trainer.maddpg.actor_optimizer.param_groups:
                param_group['lr'] = self._original_ddpg_actor_lr * finetune_scale
            for param_group in self.ddpg_trainer.maddpg.critic_optimizer.param_groups:
                param_group['lr'] = self._original_ddpg_critic_lr * finetune_scale
            # AC 恢复原学习率
            for param_group in self.ac_trainer.ac_trainer.actor_critic.actor_optimizer.param_groups:
                param_group['lr'] = self._original_ac_actor_lr
            for param_group in self.ac_trainer.ac_trainer.actor_critic.critic_optimizer.param_groups:
                param_group['lr'] = self._original_ac_critic_lr
        elif mode == 'fix_ac':
            # AC 微调
            for param_group in self.ac_trainer.ac_trainer.actor_critic.actor_optimizer.param_groups:
                param_group['lr'] = self._original_ac_actor_lr * finetune_scale
            for param_group in self.ac_trainer.ac_trainer.actor_critic.critic_optimizer.param_groups:
                param_group['lr'] = self._original_ac_critic_lr * finetune_scale
            # DDPG 恢复原学习率
            for param_group in self.ddpg_trainer.maddpg.actor_optimizer.param_groups:
                param_group['lr'] = self._original_ddpg_actor_lr
            for param_group in self.ddpg_trainer.maddpg.critic_optimizer.param_groups:
                param_group['lr'] = self._original_ddpg_critic_lr

    def _calculate_ac_state_dim(self):
        device_dim = DEVICE_NUM * 4
        task_dim = DEVICE_NUM * (3 + 14 + 7 + 35)
        uav_dim = UAV_MAX_NUM * 8
        decision_dim = UAV_MAX_NUM * 4

        total_dim = device_dim + task_dim + uav_dim + decision_dim
        print(f"AC状态维度计算：设备{device_dim} + 任务{task_dim} + 无人机{uav_dim} + 决策{decision_dim} = {total_dim}")
        return total_dim

    def _initialize_networks(self):
        print("初始化网络...")

        dummy_state = np.random.randn(1316)
        # 获取当前所有无人机坐标
        current_uavs_coords = [uav.coordinate for uav in self.env.all_uavs]
        ddpg_actions = self.ddpg_trainer.get_actions(dummy_state, current_uavs_coords,training=False)
        print(f"DDPG初始动作维度: {len(ddpg_actions)} (期望: {UAV_MAX_NUM * 3})")

        dummy_ac_state = np.random.randn(self._calculate_ac_state_dim())
        ac_actions, _ = self.ac_trainer.ac_trainer.get_actions(dummy_ac_state, training=False)
        print(f"AC初始动作维度: {len(ac_actions)} (期望: {DEVICE_NUM * 7})")

    def _reset_uav_task_stats(self):
        self.uav_task_stats = {i: 0.0 for i in range(UAV_MAX_NUM)}
        self.uav_subtask_counts = {i: 0 for i in range(UAV_MAX_NUM)}

    def _update_uav_task_stats(self, ac_offload_decisions, current_tasks):
        for task_idx, task_decision in enumerate(ac_offload_decisions):
            if task_idx >= len(current_tasks):
                continue
            task = current_tasks[task_idx]
            for subtask_idx, (lc, uc, uav_idx) in enumerate(task_decision):
                if subtask_idx >= len(task.subtasks):
                    continue
                subtask = task.subtasks[subtask_idx]
                if uc == 1 and uav_idx >= 0 and uav_idx < UAV_MAX_NUM:
                    data_mb = subtask.data_length / (1024 * 1024)
                    self.uav_task_stats[uav_idx] += data_mb
                    self.uav_subtask_counts[uav_idx] += 1

    def _print_uav_task_stats(self, time_slot):
        print(f"\n========== 时隙 {time_slot} 无人机任务统计 ==========")
        print("无人机ID | 任务数据量(MB) | 子任务数量 | 负载比例")
        print("-" * 50)
        total_data = sum(self.uav_task_stats.values())
        for uav_id in range(UAV_MAX_NUM):
            data = self.uav_task_stats[uav_id]
            count = self.uav_subtask_counts[uav_id]
            load_ratio = data / total_data if total_data > 0 else 0
            print(f"   {uav_id}    |     {data:.3f}     |     {count}     |   {load_ratio:.2%}")
        print("=" * 50)

    def _partial_clear_replay_buffer(self, buffer, keep_last):
        if hasattr(buffer, 'buffer') and isinstance(buffer.buffer, deque):
            current_len = len(buffer.buffer)
            if current_len > keep_last:
                buffer.buffer = deque(list(buffer.buffer)[-keep_last:], maxlen=buffer.buffer.maxlen)
                print(f"经验池已部分清空，保留最近 {keep_last} 条，原长度 {current_len}")
            else:
                print(f"经验池长度 {current_len} 小于保留阈值 {keep_last}，不执行清空")
        else:
            print("警告：无法识别经验池类型，不清空")

    def _save_best_models(self):
        if self.fix_ddpg_train_ac:
            self.ac_trainer.save_model(self.best_ac_model_path)
            print(f"已保存 AC 模型到 {self.best_ac_model_path}")
        else:
            self.ddpg_trainer.save_model(self.best_ddpg_model_path)
            print(f"已保存 DDPG 模型到 {self.best_ddpg_model_path}")

    def _load_best_models(self):
        if self.fix_ddpg_train_ac:
            if os.path.exists(self.best_ac_model_path):
                self.ac_trainer.load_model(self.best_ac_model_path)
                print(f"已加载 AC 模型从 {self.best_ac_model_path}")
            else:
                print("未找到之前保存的 AC 模型，使用随机初始化")
        else:
            if os.path.exists(self.best_ddpg_model_path):
                self.ddpg_trainer.load_model(self.best_ddpg_model_path)
                print(f"已加载 DDPG 模型从 {self.best_ddpg_model_path}")
            else:
                print("未找到之前保存的 DDPG 模型，使用随机初始化")

    def train(self):
        print("开始分层强化学习训练（交替优化模式）..........\n")
        print("优化策略: 固定DDPG训练AC ↔ 固定AC训练DDPG")
        print("采用优先经验回放，交替阶段微调被固定网络，每5步交替更新另一网络")
        print("并移除了部分清空经验池的操作")

        # 重置经验池（优先经验回放内部已清空，此处可保留或显式重置）
        self.ddpg_trainer.maddpg.replay_buffer.tree = SumTree(BUFFER_CAPACITY)  # 需要 import SumTree
        self.ac_trainer.ac_trainer.actor_critic.replay_buffer.tree = SumTree(BUFFER_CAPACITY)
        print("已重置 DDPG 和 AC 经验池")

        prev_ac_offload_decisions = []
        self.ac_rewards_per_episode = []
        self.total_rewards_per_episode = []

        # 交替更新计数器
        self.alternate_update_counter = 0
        self.alternate_update_interval = 5

        for episode in range(1, self.total_episodes + 1):
            print(f"\n{'=' * 60}")
            print(f"========================= episode {episode} =========================")
            print(f"{'=' * 60}")

            # 交替模式切换
            if episode % self.alternate_interval == 0:
                self._save_best_models()
                self.fix_ddpg_train_ac = not self.fix_ddpg_train_ac
                mode = "固定DDPG训练AC" if self.fix_ddpg_train_ac else "固定AC训练DDPG"
                print(f"\n切换优化模式: {mode}")

                # 设置微调学习率
                if self.fix_ddpg_train_ac:
                    self._set_finetune_lr('fix_ddpg')
                else:
                    self._set_finetune_lr('fix_ac')

                self._load_best_models()
                # 不再部分清空经验池

            self.current_episode = episode
            episode_reward = 0
            episode_ac_reward = 0
            episode_ddpg_success = 0
            episode_ac_success = 0
            episode_steps = 0

            self._reset_uav_task_stats()

            episode_total_delay = 0.0
            episode_total_energy = 0.0
            episode_total_subtasks = 0

            self.env.reset(time_slot=1)

            for time_slot in range(1, TOTAL_TIME_SLOT + 1):
                print(f"\n{'*' * 50}")
                print(f"time_slot = {time_slot}")
                print(f"{'*' * 50}")

                current_devices = self.env.devices
                current_hasTaskDevices = self.env.hasTaskDevices
                current_tasks = self.env.tasks
                current_using_uavs = self.env.using_uavs

                print(f"当前任务数: {len(current_tasks)}, 有任务设备数: {len(current_hasTaskDevices)}")

                optimized_uav_num, selected_uav_indices = optimize_uav_number(
                    current_hasTaskDevices, self.env.all_uavs, time_slot
                )
                selected_uavs = [self.env.all_uavs[i] for i in selected_uav_indices]

                self.env.uav_num = optimized_uav_num
                self.env.selected_uavs = selected_uavs
                self.env.selected_uav_indices = selected_uav_indices

                print(f"优化后的无人机数量: {optimized_uav_num}")
                print(f"被选中的无人机索引: {selected_uav_indices}")

                if time_slot == 1 or len(prev_ac_offload_decisions) == 0:
                    randomOffloadDecisions = random_offload_decisions(current_tasks,
                                                                      optimized_uav_num,
                                                                      current_devices,
                                                                      selected_uavs)
                    ddpg_state = preprocess_ddpg_state(
                        current_hasTaskDevices,
                        current_tasks,
                        self.env.all_uavs,
                        [randomOffloadDecisions]
                    )
                else:
                    ddpg_state = preprocess_ddpg_state(
                        current_hasTaskDevices,
                        current_tasks,
                        self.env.all_uavs,
                        prev_ac_offload_decisions
                    )

                if ddpg_state.shape[0] != 1316:
                    print(f"错误：DDPG状态维度不正确: {ddpg_state.shape}，跳过该时隙")
                    continue

                ddpg_actions = self.ddpg_trainer.get_actions(ddpg_state, training=(not self.fix_ddpg_train_ac))
                ddpg_trajectory_decisions = postprocess_ddpg_action(
                    ddpg_actions, self.env.all_uavs, time_slot
                )

                ddpg_constraints_valid = True
                for uav_idx in selected_uav_indices:
                    uav = self.env.all_uavs[uav_idx]
                    target_coord = ddpg_trajectory_decisions[uav_idx]
                    if not ddpg_constrain_coord(uav, target_coord):
                        print(f"时隙 {time_slot}: UAV {uav_idx} 目标坐标 {target_coord} 违反约束")
                        ddpg_constraints_valid = False

                if ddpg_constraints_valid:
                    episode_ddpg_success += 1

                if len(ddpg_trajectory_decisions) != len(self.env.all_uavs):
                    print(f"错误：无人机坐标决策数量不正确: {len(ddpg_trajectory_decisions)}，使用随机决策")
                    ddpg_trajectory_decisions = []
                    for i in range(len(self.env.all_uavs)):
                        theta1 = np.random.choice([0, 90, 180, 270])
                        theta2 = np.random.uniform(0, 90)
                        d = np.random.uniform(UAV_MAX_SPEED * 0.2, UAV_MAX_SPEED * 0.8)
                        h = np.random.uniform(UAV_MIN_HIGH + 1, UAV_MAX_HIGH - 1)
                        ddpg_trajectory_decisions.append((theta1, theta2, d, h))

                ac_state = preprocess_ac_state(
                    current_hasTaskDevices,
                    current_tasks,
                    self.env.all_uavs,
                    ddpg_trajectory_decisions,
                    time_slot
                )

                if self.fix_ddpg_train_ac:
                    ac_actions, action_probs = self.ac_trainer.ac_trainer.get_actions(
                        ac_state, training=True, num_using_uavs=optimized_uav_num
                    )
                else:
                    ac_actions, action_probs = self.ac_trainer.ac_trainer.get_actions(
                        ac_state, training=False, num_using_uavs=optimized_uav_num
                    )

                ac_offload_decisions = postprocess_ac_action(
                    ac_actions, selected_uavs, selected_uav_indices, len(current_tasks)
                )

                if len(ac_offload_decisions) != len(current_tasks):
                    print(f"错误：卸载决策数量不正确: {len(ac_offload_decisions)}，期望{len(current_tasks)}")
                    if len(ac_offload_decisions) < len(current_tasks):
                        for i in range(len(ac_offload_decisions), len(current_tasks)):
                            ac_offload_decisions.append([(1, 0, -1)] * 7)
                    else:
                        ac_offload_decisions = ac_offload_decisions[:len(current_tasks)]

                self._update_uav_task_stats(ac_offload_decisions, current_tasks)

                if time_slot % 5 == 0:
                    self._print_uav_task_stats(time_slot)

                prev_ac_offload_decisions = ac_offload_decisions

                if current_tasks and ac_offload_decisions:
                    task_time_energy(current_tasks, current_devices, self.env.all_uavs, ac_offload_decisions)
                    for task, decision in zip(current_tasks, ac_offload_decisions):
                        subtasks = task.subtasks
                        dev = current_devices[task.device_id]
                        for i, (lc, uc, uav_idx) in enumerate(decision):
                            if i >= len(subtasks): continue
                            st = subtasks[i]
                            episode_total_subtasks += 1
                            if lc:
                                comp_d = device_compute_delay(st)
                                queue_d = device_queue_delay(dev)
                                trans_d = 0.0
                                energy = device_compute_energy(st)
                            else:
                                if uav_idx < 0 or uav_idx >= len(self.env.all_uavs): continue
                                comp_d = uav_compute_delay(st)
                                queue_d = uav_queue_delay(self.env.all_uavs[uav_idx])
                                trans_d = data_transmission_delay(st, dev, self.env.all_uavs[uav_idx])
                                energy = uav_compute_energy(st)
                            if not st.predecessors:
                                earliest_start = st.generate_time
                            else:
                                max_pred_finished = 0.0
                                for pred in st.predecessors:
                                    if hasattr(pred, 'finished_time'):
                                        max_pred_finished = max(max_pred_finished, pred.finished_time)
                                earliest_start = max_pred_finished
                            dependency_wait_d = max(0, st.finished_time - comp_d - earliest_start)
                            wait_d = max(queue_d, dependency_wait_d, trans_d)
                            episode_total_delay += comp_d + wait_d
                            episode_total_energy += energy

                ddpg_reward, ac_reward, done = self.env.step(
                    ddpg_trajectory_decisions, ac_offload_decisions, time_slot
                )

                if ac_reward > 0:
                    episode_ac_success += 1

                next_hasTaskDevices = self.env.hasTaskDevices
                next_tasks = self.env.tasks
                next_optimized_uav_num, next_selected_uav_indices = optimize_uav_number(
                    next_hasTaskDevices, self.env.all_uavs, time_slot + 1
                )
                next_selected_uavs = [self.env.all_uavs[i] for i in next_selected_uav_indices]

                next_ddpg_state = preprocess_ddpg_state(
                    next_hasTaskDevices, next_tasks, self.env.all_uavs, ac_offload_decisions
                )
                next_ac_state = preprocess_ac_state(
                    next_hasTaskDevices, next_tasks, self.env.all_uavs, ddpg_trajectory_decisions, time_slot + 1
                )

                # 存储经验（使用优先经验回放）
                if not self.fix_ddpg_train_ac:
                    self.ddpg_trainer.store_experience(
                        ddpg_state, ddpg_actions, ddpg_reward, next_ddpg_state, done
                    )
                    self.ddpg_update_count += 1
                else:
                    self.ac_trainer.store_experience(
                        ac_state, ac_actions, ac_reward, next_ac_state, done
                    )
                    self.ac_update_count += 1

                # 主网络更新
                updated = False
                if not self.fix_ddpg_train_ac:
                    ddpg_critic_loss, ddpg_actor_loss = self.ddpg_trainer.update()
                    if ddpg_critic_loss != 0.0:
                        print(f"DDPG更新 - Critic损失: {ddpg_critic_loss:.6f}, Actor损失: {ddpg_actor_loss:.6f}")
                        self.alternate_update_counter += 1
                        updated = True
                else:
                    ac_actor_loss, ac_critic_loss = self.ac_trainer.update()
                    if ac_actor_loss != 0.0:
                        print(f"AC更新 - Actor损失: {ac_actor_loss:.6f}, Critic损失: {ac_critic_loss:.6f}")
                        self.alternate_update_counter += 1
                        updated = True

                # 交替更新另一网络（每 alternate_update_interval 次主网络更新后）
                if updated and self.alternate_update_counter % self.alternate_update_interval == 0:
                    if self.fix_ddpg_train_ac:
                        ddpg_critic_loss, ddpg_actor_loss = self.ddpg_trainer.update()
                        if ddpg_critic_loss != 0.0:
                            print(f"交替更新 DDPG - Critic损失: {ddpg_critic_loss:.6f}, Actor损失: {ddpg_actor_loss:.6f}")
                    else:
                        ac_actor_loss, ac_critic_loss = self.ac_trainer.update()
                        if ac_actor_loss != 0.0:
                            print(f"交替更新 AC - Actor损失: {ac_actor_loss:.6f}, Critic损失: {ac_critic_loss:.6f}")

                print(f"ac_reward = {ac_reward:.6f}\t, ddpg_reward = {ddpg_reward:.6f}")
                episode_reward += ac_reward + ddpg_reward
                episode_ac_reward += ac_reward
                episode_steps += 1

                if time_slot % 10 == 0:
                    print(f"当前优化模式: {'固定DDPG训练AC' if self.fix_ddpg_train_ac else '固定AC训练DDPG'}")

            # 一个episode结束
            avg_episode_reward = episode_reward / max(episode_steps, 1)
            avg_ac_episode_reward = episode_ac_reward / max(episode_steps, 1)
            ddpg_success_rate = episode_ddpg_success / max(episode_steps, 1)
            ac_success_rate = episode_ac_success / max(episode_steps, 1)
            avg_episode_delay = episode_total_delay / max(episode_total_subtasks, 1)
            avg_episode_energy = episode_total_energy / max(episode_total_subtasks, 1)

            print(f"\n轮次 {episode}/{self.total_episodes}, 平均总奖励: {avg_episode_reward:.6f}")
            print(f"平均AC奖励: {avg_ac_episode_reward:.6f}")
            print(f"平均时延: {avg_episode_delay:.4f}s, 平均能耗: {avg_episode_energy:.4f}J")
            print(f"DDPG约束满足率: {ddpg_success_rate:.2%}, AC决策合法率: {ac_success_rate:.2%}")

            print(f"\nEpisode {episode} 无人机任务统计摘要:")
            for uav_id in range(UAV_MAX_NUM):
                data = self.uav_task_stats[uav_id]
                count = self.uav_subtask_counts[uav_id]
                if count > 0:
                    print(f"  无人机{uav_id}: {data:.3f} MB, {count}个子任务")

            self.ddpg_success_count += episode_ddpg_success
            self.ac_success_count += episode_ac_success
            self.total_steps += episode_steps

            if self.total_steps > 0:
                print(f"全局统计: DDPG约束满足率: {self.ddpg_success_count / self.total_steps:.2%}, "
                      f"AC决策合法率: {self.ac_success_count / self.total_steps:.2%}")
                print(f"更新统计: DDPG更新次数(经验存储): {self.ddpg_update_count}, AC更新次数(经验存储): {self.ac_update_count}")

            self.ac_rewards_per_episode.append(avg_ac_episode_reward)
            self.total_rewards_per_episode.append(avg_episode_reward)

        self._save_final_models()
        return self.total_rewards_per_episode

    def _save_final_models(self):
        ddpg_actor_lr = self.ddpg_trainer.maddpg.actor_optimizer.param_groups[0]['lr']
        ddpg_critic_lr = self.ddpg_trainer.maddpg.critic_optimizer.param_groups[0]['lr']
        ac_actor_lr = self.ac_trainer.ac_trainer.actor_critic.actor_optimizer.param_groups[0]['lr']
        ac_critic_lr = self.ac_trainer.ac_trainer.actor_critic.critic_optimizer.param_groups[0]['lr']

        def fmt_lr(lr):
            s = f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
            return s

        ddpg_actor_str = fmt_lr(ddpg_actor_lr)
        ddpg_critic_str = fmt_lr(ddpg_critic_lr)
        ac_actor_str = fmt_lr(ac_actor_lr)
        ac_critic_str = fmt_lr(ac_critic_lr)
        target_str = f"{self.target_offload_ratio:.2f}".replace('.', '_')

        ddpg_filename = f"Hierarchical_ddpg_actor{ddpg_actor_str}_critic{ddpg_critic_str}_target{target_str}.pth"
        self.ddpg_trainer.save_model(f'models/{ddpg_filename}')

        ac_filename = f"Hierarchical_ac_actor{ac_actor_str}_critic{ac_critic_str}_target{target_str}.pth"
        self.ac_trainer.save_model(f'models/{ac_filename}')

        print(f"最终模型已保存: {ddpg_filename}, {ac_filename}")

    def evaluate(self, num_episodes=10):
        print("开始模型评估...")
        total_evaluation_reward = 0
        evaluation_stats = {i: {'data': 0.0, 'count': 0} for i in range(UAV_MAX_NUM)}
        for episode in range(num_episodes):
            episode_reward = 0
            self.env.reset(time_slot=1)
            episode_stats = {i: {'data': 0.0, 'count': 0} for i in range(UAV_MAX_NUM)}
            for time_slot in range(1, TOTAL_TIME_SLOT + 1):
                current_devices = self.env.devices
                current_tasks = self.env.tasks
                optimized_uav_num, selected_uav_indices = optimize_uav_number(
                    current_devices, self.env.all_uavs, time_slot
                )
                selected_uavs = [self.env.all_uavs[i] for i in selected_uav_indices]
                self.env.uav_num = optimized_uav_num
                self.env.selected_uavs = selected_uavs
                self.env.selected_uav_indices = selected_uav_indices

                ddpg_state = preprocess_ddpg_state(
                    current_devices, current_tasks, self.env.all_uavs, []
                )
                ddpg_actions = self.ddpg_trainer.get_actions(ddpg_state, training=False)
                ddpg_trajectory_decisions = postprocess_ddpg_action(
                    ddpg_actions, self.env.all_uavs, time_slot
                )

                ac_state = preprocess_ac_state(
                    current_devices, current_tasks, self.env.all_uavs, ddpg_trajectory_decisions, time_slot
                )
                ac_actions, _ = self.ac_trainer.ac_trainer.get_actions(
                    ac_state, training=False, num_using_uavs=optimized_uav_num
                )
                ac_offload_decisions = postprocess_ac_action(
                    ac_actions, selected_uavs, selected_uav_indices, len(current_tasks)
                )

                for task_idx, task_decision in enumerate(ac_offload_decisions):
                    if task_idx >= len(current_tasks): continue
                    task = current_tasks[task_idx]
                    for subtask_idx, (lc, uc, uav_idx) in enumerate(task_decision):
                        if subtask_idx >= len(task.subtasks): continue
                        subtask = task.subtasks[subtask_idx]
                        if uc == 1 and uav_idx >= 0 and uav_idx < UAV_MAX_NUM:
                            data_mb = subtask.data_length / (1024 * 1024)
                            episode_stats[uav_idx]['data'] += data_mb
                            episode_stats[uav_idx]['count'] += 1

                ddpg_reward, ac_reward, done = self.env.step(
                    ddpg_trajectory_decisions, ac_offload_decisions, time_slot
                )
                episode_reward += ac_reward

            for uav_id in range(UAV_MAX_NUM):
                evaluation_stats[uav_id]['data'] += episode_stats[uav_id]['data']
                evaluation_stats[uav_id]['count'] += episode_stats[uav_id]['count']

            avg_episode_reward = episode_reward / min(time_slot, TOTAL_TIME_SLOT)
            total_evaluation_reward += avg_episode_reward
            print(f"评估轮次 {episode + 1}: 平均奖励 {avg_episode_reward:.6f}")

        print(f"\n========== 评估结果：无人机任务统计 ==========")
        total_data = sum(stats['data'] for stats in evaluation_stats.values())
        total_count = sum(stats['count'] for stats in evaluation_stats.values())
        for uav_id in range(UAV_MAX_NUM):
            data = evaluation_stats[uav_id]['data'] / num_episodes
            count = evaluation_stats[uav_id]['count'] / num_episodes
            load_ratio = data / total_data * num_episodes if total_data > 0 else 0
            print(f"无人机{uav_id}: {data:.3f} MB/轮, {count:.1f} 子任务/轮, 负载比例: {load_ratio:.2%}")
        final_avg_reward = total_evaluation_reward / num_episodes
        print(f"\n最终评估结果: 平均奖励 {final_avg_reward:.6f}")
        return final_avg_reward

    def save_models(self, checkpoint=False):
        pass

    def load_models(self, ddpg_path, ac_path):
        try:
            self.ddpg_trainer.load_model(ddpg_path)
            self.ac_trainer.load_model(ac_path)
            print("模型加载成功")
        except FileNotFoundError:
            print("未找到模型文件，使用随机初始化的模型")