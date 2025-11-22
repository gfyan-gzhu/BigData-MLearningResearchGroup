import random
import torch
import numpy as np
import rl_utils
from openpyxl import Workbook

from DQN import DQN
from PPO import PPO
from cached_env.cached_env_main import VecCachedEnv


# 初始化参数和经验池
def initialization_DQN(vehicle_num, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                       epsilon, target_update, device, buffer_size):
    agents_DQN = []
    replay_buffers_DQN = []
    for i in range(0, vehicle_num):
        agents_DQN.append(DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                              epsilon, target_update, device))
        replay_buffers_DQN.append(rl_utils.ReplayBuffer(buffer_size))

    return agents_DQN, replay_buffers_DQN


def initialization_PPO(vehicle_num, state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr,
                       lmbda, epochs, eps, gamma, device, buffer_size):
    agents_PPO = []
    replay_buffers_PPO = []
    for i in range(0, vehicle_num):
        agents_PPO.append(PPO(state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr,
                              lmbda, epochs, eps, gamma, device))
        replay_buffers_PPO.append(rl_utils.ReplayBuffer(buffer_size))
    return agents_PPO, replay_buffers_PPO


# 训练过程
def train_off_policy_agent(env, agents_DQN, replay_buffers_DQN, num_episodes,
                           agents_PPO, replay_buffers_PPO, minimal_size, batch_size):
    return_list = []
    for episode in range(num_episodes):
        print('<<<<<<<<<Episode: %s' % (episode + 1))
        state_off, state_cac, state_cac_conv = env.reset()
        episode_return = 0
        sum_time = 0
        done = False
        hitNum = 0
        successNum = 0
        localComputeNum = 0
        for t in range(1, num_T + 1):
            for i in range(0, env.vehicle_num):  # TV数20

                action_off = agents_DQN[i].take_action(state_off)  # 卸载决策
                if action_off == 0 and env.cached[env.task_messages[1]][env.vehicle_count] == 0:
                    # 任务本地计算且任务内容没有在本地缓存
                    action_cac = agents_PPO[i].take_action(state_cac, state_cac_conv)
                else:
                    action_cac = 0
                action = [action_off, action_cac]
                next_state_off, next_state_cac, next_state_cac_conv, reward, done, cost_time, isHit,\
                isLocalCompute, isSuccess \
                    = env.step(action)  # vec执行动作
                # 存储经验
                replay_buffers_DQN[i].add_decide(state_off, action_off, reward, next_state_off, done)
                if action_off == 0 and env.cached[env.task_messages[1]][env.vehicle_count] == 0:
                    # 任务本地计算且任务内容没有在本地缓存
                    replay_buffers_PPO[i].add_cache(state_cac, state_cac_conv, action_cac, reward,
                                                    next_state_cac, next_state_cac_conv, done)
                # 更新状态
                state_off = next_state_off
                state_cac = next_state_cac
                state_cac_conv = next_state_cac_conv
                sum_time += cost_time
                episode_return += reward
                hitNum += isHit
                localComputeNum += isLocalCompute
                successNum += isSuccess

                if replay_buffers_DQN[i].size() >= minimal_size:
                    states_off, actions_off, rewards_off, next_states_off, dones_off \
                        = replay_buffers_DQN[i].sample_decide(batch_size)  # 采样
                    transition_dict_dqn = {'states': states_off,
                                           'actions': actions_off,
                                           'rewards': rewards_off,
                                           'next_states': next_states_off,
                                           'dones': dones_off}
                    agents_DQN[i].update(transition_dict_dqn)  # 学习
                if replay_buffers_PPO[i].size() >= minimal_size:
                    states_cac, states_cac_conv, actions_cac, rewards_cac, next_states_cac, next_states_cac_conv, \
                    dones_cac = replay_buffers_PPO[i].sample_cache(batch_size)
                    transition_dict_ppo = {'states_cac': states_cac,
                                           'states_cac_conv': states_cac_conv,
                                           'actions': actions_cac,
                                           'next_states': next_states_cac,
                                           'next_states_conv': next_states_cac_conv,
                                           'rewards': rewards_cac,
                                           'dones': dones_cac}
                    agents_PPO[i].update(transition_dict_ppo)

        ave_episode_return = episode_return / num_T / env.vehicle_num
        ave_time = sum_time / (num_T * env.vehicle_num)
        hitRatio = hitNum / localComputeNum
        successRatio = successNum / (num_T * env.vehicle_num)
        print("第{}轮训练平均奖励 ave={}".format(episode + 1, ave_episode_return))
        return_list.append(ave_episode_return)
        # 写入excel表
        sheet_toc['A{}'.format(episode + 2)] = episode + 1
        sheet_toc['B{}'.format(episode + 2)] = ave_episode_return
        sheet_toc['C{}'.format(episode + 2)] = ave_time
        sheet_toc['D{}'.format(episode + 2)] = hitRatio
        sheet_toc['E{}'.format(episode + 2)] = successRatio
        wb.save(savepath)
    return return_list


if __name__ == '__main__':
    # 初始化环境
    env_name = 'Vec1.0'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    T = 1  # 一个时隙大小
    num_T = 35  # 时隙个数
    env = VecCachedEnv(num_T, T)
    state_off, state_cac, state_cac_conv = env.reset()
    # 公共参数
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    num_episodes = 300  # 训练轮数
    buffer_size = 10000  # 5000
    minimal_size = 1000  # 开始学习的界限 500
    batch_size = 128  # 初始 128
    # DQN 参数
    """
    学习率：dqn-lr=0.01,alr=0.00001,clr=0.0001
    """
    learning_rate = 0.01  # 初始 0.0001  ---{0.0001,0.001,0.01}
    hidden_dim_dqn = 128  # 隐藏层的层数 128
    gamma_dqn = 0.99  # 折扣因子
    state_dim_off = len(state_off)
    action_dim_off = len(env.action_space_offloading)
    epsilon = 0.01  # epsilon-贪婪策略
    target_update = 10  # 目标(target)网络更新频率
    # PPO 参数
    actor_lr_ppo = 0.00001  # 初始 0.0001 {0.0001, 0.001, 0.01} 0.00001
    critic_lr_ppo = 0.0001  # 初始 0.001     0.0001
    hidden_dim_ppo = 128  # # 隐藏层的层数 128
    gamma_ppo = 0.99
    lmbda = 0.95  # 计算优势函数的参数 初始0.95
    epochs = 10  # 一条序列的数据用来训练轮数 初始10
    eps = 0.2  # 0.005 PPO中截断范围的参数初始=0.2 (0.01 900多次出现nan)
    state_dim_cac = len(state_cac)
    state_dim_conv = 1
    action_dim_cac = len(env.action_space_cached)
    # 初始化
    agents_DQN, replay_buffers_DQN = \
        initialization_DQN(env.vehicle_num, state_dim_off, hidden_dim_dqn, action_dim_off, learning_rate, gamma_dqn,
                           epsilon, target_update, device, buffer_size)
    agents_PPO, replay_buffers_PPO = \
        initialization_PPO(env.vehicle_num, state_dim_cac, state_dim_conv, hidden_dim_ppo, action_dim_cac, actor_lr_ppo,
                           critic_lr_ppo,
                           lmbda, epochs, eps, gamma_ppo, device, buffer_size)

    wb = Workbook()
    # 选择默认工作表
    sheet_toc = wb.active
    sheet_toc.title = "toc奖励值"
    # 或者创建一个新的工作表
    # 写入表头
    sheet_toc.append(['episode', 'ave_reward', 'ave_time', 'hitRatio', 'successRatio'])
    savepath = 'data' \
               '\\toc-dqnlr-{}-alr-{}-clr-{}-forecast_LSTM-episode={}-input_size=[4,5].xlsx'.format(
        learning_rate,
        actor_lr_ppo,
        critic_lr_ppo,
        num_episodes,

    )

    return_list = train_off_policy_agent(env, agents_DQN, replay_buffers_DQN, num_episodes,
                                         agents_PPO, replay_buffers_PPO, minimal_size, batch_size)
