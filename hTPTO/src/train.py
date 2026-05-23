# train.py (修改后)
import pandas as pd
from Hierarchical_MADDPG_AC import HierarchicalRLAlgorithm, random_offload_decisions, preprocess_ddpg_state
from StandardDDPG import StandardDDPGTrainer, preprocess_standard_state
from PPO import PPO, postprocess_ppo_action
from env_main import MultiUavEnv
from UAVNumberOptimize import optimize_uav_number
from DDPG_LDPG import *
from simEnvParameter import *
import os
import torch
import numpy as np

def fmt_lr(lr):
    """格式化学习率，如 1e-04 → 1e-4"""
    s = f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
    return s

def train_hierarchical(episodes=None, excel_filename=None):
    """分层强化学习训练（算法内部已保存最终模型，此处保存总平均奖励记录）"""
    algorithm = HierarchicalRLAlgorithm(total_episodes=episodes, target_offload_ratio=TARGET_OFFLOAD_RATIO)
    # train() 现在返回总平均奖励列表
    total_rewards = algorithm.train()
    if excel_filename is None:
        lr_ac_actor = AC_LR_ACTOR
        lr_ac_critic = AC_LR_CRITIC
        # 文件名可自定义，此处仍使用 ac 字样或改为 total，为清晰起见改为 total
        excel_filename = f"hierarchical_results_ac_lr_{lr_ac_actor:.1e}_{lr_ac_critic:.1e}.xlsx"
    if not excel_filename.endswith('.xlsx'):
        excel_filename += '.xlsx'
    df = pd.DataFrame({
        'episode': list(range(1, len(total_rewards) + 1)),
        'total_avg_reward': total_rewards   # 保存总平均奖励
    })
    df.to_excel(excel_filename, index=False)
    print(f"\n训练完成，总平均奖励数据已保存至: {os.path.abspath(excel_filename)}")
    return algorithm

def train_standard_ddpg(episodes=None, excel_filename=None):
    """标准DDPG训练（使用总奖励，即路径奖励+卸载奖励，与分层强化学习保持一致）"""
    print("开始标准DDPG训练...")
    from StandardDDPG import combine_rewards  # 导入辅助函数

    env = MultiUavEnv(time_slot=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = StandardDDPGTrainer(state_dim=1316, device=device)

    total_episodes = episodes if episodes is not None else TOTAL_EPISODES
    rewards_per_episode = []  # 记录每个 episode 的总平均奖励

    os.makedirs('models', exist_ok=True)

    for episode in range(1, total_episodes + 1):
        print(f"\n{'='*60}")
        print(f"========================= Episode {episode} =========================")
        env.reset(time_slot=1)
        episode_total_reward = 0   # 累积总奖励
        episode_steps = 0
        prev_offload_decisions = []

        for time_slot in range(1, TOTAL_TIME_SLOT + 1):
            print(f"\n{'*'*50}")
            print(f"time_slot = {time_slot}")

            current_devices = env.devices
            current_hasTaskDevices = env.hasTaskDevices
            current_tasks = env.tasks
            current_all_uavs = env.all_uavs

            optimized_uav_num, selected_uav_indices = optimize_uav_number(
                current_hasTaskDevices, current_all_uavs, time_slot
            )
            selected_uavs = [current_all_uavs[i] for i in selected_uav_indices]
            env.uav_num = optimized_uav_num
            env.selected_uavs = selected_uavs
            env.selected_uav_indices = selected_uav_indices

            if time_slot == 1 or len(prev_offload_decisions) == 0:
                random_offload = random_offload_decisions(current_tasks, optimized_uav_num,
                                                           current_devices, selected_uavs)
                state = preprocess_standard_state(current_hasTaskDevices, current_tasks,
                                                  current_all_uavs, [random_offload])
            else:
                state = preprocess_standard_state(current_hasTaskDevices, current_tasks,
                                                  current_all_uavs, prev_offload_decisions)

            raw_actions = trainer.get_actions(state, training=True)
            traj_decisions, offload_decisions = trainer.postprocess_actions(
                raw_actions, selected_uav_indices, len(current_tasks), current_all_uavs
            )

            ddpg_reward, ac_reward, done = env.step(traj_decisions, offload_decisions, time_slot)

            # 计算总奖励（路径奖励 + 卸载奖励）
            total_reward = combine_rewards(ddpg_reward, ac_reward)   # 或直接 total_reward = ddpg_reward + ac_reward

            next_hasTaskDevices = env.hasTaskDevices
            next_tasks = env.tasks
            next_state = preprocess_standard_state(next_hasTaskDevices, next_tasks,
                                                    env.all_uavs, offload_decisions)

            # 存储总奖励
            trainer.store_experience(state, raw_actions, total_reward, next_state, done)

            if time_slot % 3 == 0:
                critic_loss, actor_loss = trainer.update()
                if critic_loss and actor_loss:
                    print(f"更新: Critic损失={critic_loss:.6f}, Actor损失={actor_loss:.6f}")

            episode_total_reward += total_reward
            episode_steps += 1
            prev_offload_decisions = offload_decisions

            print(f"时隙 {time_slot} DDPG奖励: {ddpg_reward:.6f}, AC奖励: {ac_reward:.6f}, 总奖励: {total_reward:.6f}")

        avg_total_reward = episode_total_reward / max(episode_steps, 1)
        rewards_per_episode.append(avg_total_reward)
        print(f"\nEpisode {episode} 平均总奖励: {avg_total_reward:.6f}")

    # 保存最终模型
    actor_lr = trainer.ddpg.actor_optimizer.param_groups[0]['lr']
    critic_lr = trainer.ddpg.critic_optimizer.param_groups[0]['lr']
    target_str = f"{TARGET_OFFLOAD_RATIO:.2f}".replace('.', '_')
    filename = f"StandardDDPG_actor{fmt_lr(actor_lr)}_critic{fmt_lr(critic_lr)}_target{target_str}.pth"
    trainer.save_model(os.path.join('models', filename))
    print(f"最终模型已保存: {filename}")

    # 保存训练结果到Excel
    if excel_filename is None:
        excel_filename = f"standard_ddpg_results_ep{total_episodes}.xlsx"
    if not excel_filename.endswith('.xlsx'):
        excel_filename += '.xlsx'
    df = pd.DataFrame({
        'episode': list(range(1, len(rewards_per_episode) + 1)),
        'avg_total_reward': rewards_per_episode   # 保存总平均奖励
    })
    df.to_excel(excel_filename, index=False)
    print(f"\n标准DDPG训练完成，总平均奖励数据已保存至: {os.path.abspath(excel_filename)}")
    return trainer

def train_ppo(episodes=None, excel_filename=None):
    """
    PPO训练（修改为使用总奖励 = ddpg_reward + ac_reward，并保存平均总奖励）
    """
    print("开始PPO训练（使用总奖励）...")
    env = MultiUavEnv(time_slot=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 1316
    cont_dim = UAV_MAX_NUM * 4
    num_subtasks = DEVICE_NUM * 7
    num_actions = UAV_MAX_NUM + 1

    ppo = PPO(
        state_dim=state_dim,
        cont_dim=cont_dim,
        num_subtasks=num_subtasks,
        num_actions=num_actions,
        lr_actor=PPO_LR_ACTOR,
        lr_critic=PPO_LR_CRITIC,
        gamma=GAMMA,
        lam=PPO_LAMBDA,
        clip_epsilon=PPO_CLIP_EPSILON,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=PPO_BATCH_SIZE,
        update_epochs=PPO_UPDATE_EPOCHS,
        device=device,
    )

    total_episodes = episodes if episodes is not None else TOTAL_EPISODES
    total_rewards_per_episode = []   # 记录每个episode的平均总奖励
    os.makedirs('models', exist_ok=True)

    for episode in range(1, total_episodes + 1):
        print(f"\n{'='*60}")
        print(f"========================= PPO Episode {episode} =========================")
        env.reset(time_slot=1)
        episode_total_reward = 0      # 累积总奖励（时隙和）
        episode_steps = 0
        prev_offload_decisions = []

        for time_slot in range(1, TOTAL_TIME_SLOT + 1):
            print(f"\n{'*'*50}")
            print(f"time_slot = {time_slot}")

            current_devices = env.devices
            current_hasTaskDevices = env.hasTaskDevices
            current_tasks = env.tasks
            current_all_uavs = env.all_uavs

            optimized_uav_num, selected_uav_indices = optimize_uav_number(
                current_hasTaskDevices, current_all_uavs, time_slot
            )
            selected_uavs = [current_all_uavs[i] for i in selected_uav_indices]
            env.uav_num = optimized_uav_num
            env.selected_uavs = selected_uavs
            env.selected_uav_indices = selected_uav_indices

            if time_slot == 1 or len(prev_offload_decisions) == 0:
                random_offload = random_offload_decisions(current_tasks, optimized_uav_num,
                                                           current_devices, selected_uavs)
                state = preprocess_ddpg_state(current_hasTaskDevices, current_tasks,
                                               current_all_uavs, [random_offload])
            else:
                state = preprocess_ddpg_state(current_hasTaskDevices, current_tasks,
                                               current_all_uavs, prev_offload_decisions)

            cont_action, disc_action, log_prob = ppo.select_action(state, training=True)

            traj_decisions, offload_decisions = postprocess_ppo_action(
                cont_action, disc_action, selected_uav_indices, len(current_tasks), current_all_uavs
            )

            ddpg_reward, ac_reward, done = env.step(traj_decisions, offload_decisions, time_slot)

            # ---------- 修改点：使用总奖励 ----------
            total_reward = ddpg_reward + ac_reward   # 上下两层奖励之和

            next_hasTaskDevices = env.hasTaskDevices
            next_tasks = env.tasks
            next_state = preprocess_ddpg_state(next_hasTaskDevices, next_tasks,
                                                 env.all_uavs, offload_decisions)

            # 存储经验时使用总奖励
            ppo.store_experience(state, cont_action, disc_action, total_reward, next_state, done, log_prob)

            episode_total_reward += total_reward
            episode_steps += 1
            prev_offload_decisions = offload_decisions

            print(f"时隙 {time_slot} 总奖励: {total_reward:.6f} (DDPG: {ddpg_reward:.6f}, AC: {ac_reward:.6f})")

        # 一个episode结束，执行PPO更新
        if len(ppo.buffer) > 0:
            actor_loss, critic_loss = ppo.update()
            print(f"PPO更新: Actor损失={actor_loss:.6f}, Critic损失={critic_loss:.6f}")

        # 计算本episode的平均总奖励
        avg_total_reward = episode_total_reward / max(episode_steps, 1)
        total_rewards_per_episode.append(avg_total_reward)
        print(f"\nEpisode {episode} 平均总奖励: {avg_total_reward:.6f}")

    # 保存最终模型
    actor_lr = ppo.actor_optimizer.param_groups[0]['lr']
    critic_lr = ppo.critic_optimizer.param_groups[0]['lr']
    target_str = f"{TARGET_OFFLOAD_RATIO:.2f}".replace('.', '_')
    filename = f"PPO_actor{fmt_lr(actor_lr)}_critic{fmt_lr(critic_lr)}_target{target_str}.pth"
    ppo.save_model(os.path.join('models', filename))
    print(f"最终模型已保存: {filename}")

    # 保存训练结果到Excel（记录平均总奖励）
    if excel_filename is None:
        excel_filename = f"ppo_results_ep{total_episodes}.xlsx"
    if not excel_filename.endswith('.xlsx'):
        excel_filename += '.xlsx'
    df = pd.DataFrame({
        'episode': list(range(1, len(total_rewards_per_episode) + 1)),
        'avg_total_reward': total_rewards_per_episode   # 保存平均总奖励
    })
    df.to_excel(excel_filename, index=False)
    print(f"\nPPO训练完成，总奖励数据已保存至: {os.path.abspath(excel_filename)}")
    return ppo

def train_ddpg_ldpg(episodes=None, excel_filename=None):
    """DDPG-LDPG 算法训练（算法内部已保存最终模型）"""
    algorithm = DDPG_LDPG_Algorithm(total_episodes=episodes, target_offload_ratio = TARGET_OFFLOAD_RATIO)
    rewards = algorithm.train()
    if excel_filename is None:
        excel_filename = f"ddpg_ldpg_results_ep{episodes}.xlsx"
    if not excel_filename.endswith('.xlsx'):
        excel_filename += '.xlsx'
    df = pd.DataFrame({
        'episode': list(range(1, len(rewards) + 1)),
        'avg_reward': rewards
    })
    df.to_excel(excel_filename, index=False)
    print(f"\nDDPG-LDPG训练完成，奖励数据已保存至: {os.path.abspath(excel_filename)}")
    return algorithm



def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # 选择训练算法（根据需要取消注释）

    # 1. 分层强化学习
    algorithm = train_hierarchical(episodes=TOTAL_EPISODES)

    # 2. 标准DDPG
    # trainer = train_standard_ddpg(episodes=TOTAL_EPISODES)

    # 3. PPO训练
    # ppo_trainer = train_ppo(episodes=TOTAL_EPISODES)

    # 4. DDPG_LDPG训练
    # algorithm = train_ddpg_ldpg(episodes=TOTAL_EPISODES)


    pass

if __name__ == "__main__":
    main()