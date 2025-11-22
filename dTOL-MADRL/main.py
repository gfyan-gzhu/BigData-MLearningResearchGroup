import random
import torch
import numpy as np
import rl_utils
import xlwt

from SAC import SAC
from DDPG import DDPG
from env.env_main import VecEnv


# 初始化参数和经验池
def initialization_SAC(TV_num, state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
                       target_entropy, tau, gamma, device, buffer_size):
    agents_SAC = []
    replay_buffers_SAC = []
    for i in range(0, TV_num):
        agents_SAC.append(SAC(state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
                              target_entropy, tau, gamma, device))
        replay_buffers_SAC.append(rl_utils.ReplayBuffer(buffer_size))

    return agents_SAC, replay_buffers_SAC


def initialization_DDPG(TV_num, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr,
                        critic_lr, tau, gamma, device, batch_size, buffer_size):
    agents_DDPG = []
    replay_buffers_DDPG = []
    for i in range(0, TV_num):
        agents_DDPG.append(DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr,
                                critic_lr, tau, gamma, device, batch_size))
        replay_buffers_DDPG.append(rl_utils.ReplayBuffer(buffer_size))
    return agents_DDPG, replay_buffers_DDPG


# 训练过程
def train_off_policy_agent(env, agents_SAC, replay_buffers_SAC, num_episodes,
                           agents_DDPG, replay_buffers_DDPG, minimal_size, batch_size):
    return_list = []
    for episode in range(num_episodes):
        print('<<<<<<<<<Episode: %s' % (episode + 1))
        state_fc, state_conv = env.reset()
        episode_return = 0
        sum_time = 0
        state_rate_pre = []
        rates = []
        reward = 0
        done = False
        count = 0
        for t in range(1, num_T + 1):
            for i in range(0, env.TV_num):  # TV数20
                decides = agents_SAC[i].take_action(state_fc, state_conv)  # 选择卸载位置
                decides = sum(decides, [])
                state_rate = env.get_state3(decides)
                if t == 1:
                    state_rate_pre = state_rate  # 临时存储DDPG前一动作
                if t > 1:
                    # 存储经验
                    replay_buffers_DDPG[i].add_rate(state_rate_pre, rates, reward, state_rate, done)
                    state_rate_pre = state_rate

                rates_ = agents_DDPG[i].take_action(state_rate)  # 分配卸载比例
                rates = sum(rates_, [])
                action = [[decides[0], rates[0]], [decides[1], rates[1]], [decides[2], rates[2]]]  # 构建动作
                next_state_fc, next_state_conv, reward, done, cost_time = env.step(action)  # vec执行动作
                if cost_time == 0:
                    count += 1
                replay_buffers_SAC[i].add_decide(state_fc, state_conv, decides, reward, next_state_fc
                                                 , next_state_conv, done)  # 存储经验
                # 更新状态
                state_fc = next_state_fc
                state_conv = next_state_conv
                sum_time += cost_time
                episode_return += reward

                if replay_buffers_SAC[i].size() >= minimal_size:
                    states_fc, states_conv, actions, rewards, next_states_fc, next_states_conv, dones \
                        = replay_buffers_SAC[i].sample_decide(batch_size)  # 采样
                    transition_dict_sac = {'states_fc': states_fc, 'states_conv': states_conv,
                                           'actions': actions, 'rewards': rewards,
                                           'next_states_fc': next_states_fc,
                                           'next_states_conv': next_states_conv,
                                           'dones': dones}
                    agents_SAC[i].update(transition_dict_sac)  # 学习
                if replay_buffers_DDPG[i].size() >= minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffers_DDPG[i].sample_rate(batch_size)
                    transition_dict_ddpg = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                            'dones': b_d}
                    agents_DDPG[i].update(transition_dict_ddpg)

        ave_episode_return = episode_return / num_T / env.TV_num
        ave_time = sum_time / (num_T * env.TV_num - count)
        print("第{}轮训练平均奖励 ave={}".format(episode + 1, ave_episode_return))
        print(ave_time)
        return_list.append(ave_episode_return)
        # 写入excel表
        sheet_sac.write(episode + 1, 0, episode + 1)
        sheet_sac.write(episode + 1, 1, ave_episode_return)
        sheet_sac.write(episode + 1, 2, ave_time)
        book.save(savepath)

    return return_list


if __name__ == '__main__':
    # 初始化环境
    env_name = 'Vec1.0'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    T = 1  # 一个时隙大小
    num_T = 10  # 时隙个数
    env = VecEnv(num_T, T)
    state_fc, state_conv = env.reset()
    # 公共参数
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    num_episodes = 1500  # 训练轮数
    buffer_size = 5000
    minimal_size = 500  # 开始学习的界限
    batch_size = 128  # 初始 128
    # SAC 参数
    # sac: a-lr=0.0001 c-lr=0.001 ddpg: a-lr=0.0001 c-lr=0.001
    actor_lr_sac = 0.0001   # 初始 0.0001
    critic_lr_sac = 0.001  # 初始 0.001
    alpha_lr_sac = 0.05  # 温度系数学习率、探索
    hidden_dim_sac = 128  # 隐藏层的层数 128
    gamma_sac = 0.9
    tau_sac = 0.005  # 软更新参数,更新目标网络时控制更新幅度
    target_entropy_sac = -1
    state_dim_fc = len(state_fc)
    state_dim_conv = 1
    action_dim_sac = len(env.action_space)
    # DDPG 参数
    actor_lr_ddpg = 0.0001
    critic_lr_ddpg = 0.001
    hidden_dim_ddpg = 128
    gamma_ddpg = 0.9
    tau_ddpg = 0.005  # 软更新参数
    sigma_ddpg = 0.01  # 高斯噪声标准差
    state_dim_ddpg = 11
    action_dim_ddpg = 3
    action_bound = 1  # 动作最大值
    # 初始化
    agents_SAC, replay_buffers_SAC = \
        initialization_SAC(env.TV_num, state_dim_fc, state_dim_conv, hidden_dim_sac,
                           action_dim_sac, actor_lr_sac, critic_lr_sac, alpha_lr_sac,
                           target_entropy_sac, tau_sac, gamma_sac, device, buffer_size)
    agents_DDPG, replay_buffers_DDPG = \
        initialization_DDPG(env.TV_num, state_dim_ddpg, hidden_dim_ddpg, action_dim_ddpg,
                            action_bound, sigma_ddpg, actor_lr_ddpg, critic_lr_ddpg, tau_ddpg,
                            gamma_ddpg, device, batch_size, buffer_size)

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_sac = book.add_sheet('sac奖励值', cell_overwrite_ok=True)
    col = ('episode', 'ave_reward', 'ave_time',)
    for i in range(0, 3):
        sheet_sac.write(0, i, col[i])
    savepath = 'E:\python\pytorch\dTOL-MADRL\data' \
               '\DTOMDRL-20-a-{}-c-{}-da-{}-dc-{}.xls'.format(actor_lr_sac, critic_lr_sac, actor_lr_ddpg, critic_lr_ddpg)
    return_list = train_off_policy_agent(env, agents_SAC, replay_buffers_SAC, num_episodes,
                                         agents_DDPG, replay_buffers_DDPG, minimal_size,
                                         batch_size)

