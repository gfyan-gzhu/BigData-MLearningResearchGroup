import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import env.env_main as env_main


TV_num = env_main.vehicle_num_TV * env_main.num_lanes
input_size = 128 + (TV_num - 10) * 16

class PolicyNet(nn.Module):
    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, device):
        super(PolicyNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # 卷积层处理的状态
        self.conv1 = nn.Conv2d(in_channels=state_dim_conv, out_channels=16, kernel_size=(3, 3))
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层处理的状态
        self.fc2 = nn.Linear(state_dim_fc, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # 共享的隐藏层,全连接层
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc4 = nn.Linear(hidden_dim, 3 * action_dim)
        # 初始化网络参数
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 对卷积层使用初始化函数
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LSTM):
                # 对LSTM层使用初始化函数
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.constant_(param.data, 0)
            elif isinstance(layer, nn.Linear):
                # 对线性层使用初始化函数
                init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

    def forward(self, x_fc, x_conv, hidden=None):
        # 卷积层处理
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = self.pool1(x_conv)  # 添加池化层
        x_conv = x_conv.view(x_conv.size(0), -1)  # 展平卷积输出

        # LSTM 层处理
        x_conv = x_conv.unsqueeze(1)  # 增加时间步维度
        lstm_out, hidden = self.lstm(x_conv, hidden)
        x_conv = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        x_conv = F.relu(self.fc1(x_conv))
        # 全连接层处理
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = F.relu(self.fc3(x_fc))
        # 合并卷积和全连接层的输出
        x = x_conv + x_fc  # 或者可以使用其他方式来整合
        # 共享隐藏层
        x = F.relu(self.shared_fc(x))
        # 输出动作的概率分布
        action_probs = F.softmax(self.fc4(x), dim=1)
        return action_probs.view(-1, 3, self.action_dim)


class QValueNet(nn.Module):
    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, device):
        super(QValueNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # 卷积层处理的状态 torch.Size([1, 1, 20, 6])
        self.conv1 = nn.Conv2d(in_channels=state_dim_conv, out_channels=16, kernel_size=(3, 3))
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层处理的状态
        self.fc2 = nn.Linear(state_dim_fc, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # 共享的隐藏层,全连接层
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc4 = nn.Linear(hidden_dim, 3 * action_dim)
        # 初始化网络参数
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 对卷积层使用初始化函数
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LSTM):
                # 对LSTM层使用初始化函数
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.constant_(param.data, 0)
            elif isinstance(layer, nn.Linear):
                # 对线性层使用初始化函数
                init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

    def forward(self, x_fc, x_conv, hidden=None):
        # 卷积层处理
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = self.pool1(x_conv)  # 添加池化层
        x_conv = x_conv.view(x_conv.size(0), -1)  # 展平卷积输出

        # LSTM 层处理
        x_conv = x_conv.unsqueeze(1)  # 增加时间步维度
        lstm_out, hidden = self.lstm(x_conv, hidden)
        x_conv = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        x_conv = F.relu(self.fc1(x_conv))
        # 全连接层处理
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = F.relu(self.fc3(x_fc))
        # 合并卷积和全连接层的输出
        x = x_conv + x_fc  # 或者可以使用其他方式来整合
        # 共享隐藏层
        x = F.relu(self.shared_fc(x))
        return self.fc4(x)


class SAC:
    """处理离散动作的SAC算法"""

    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim, device).to(device)

        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim, device).to(device)

        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim, device).to(device)

        # 第一个目标Q网络
        self.target_critic_1 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim,
                                         action_dim, device).to(device)

        # 第二个目标Q网络
        self.target_critic_2 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim,
                                         action_dim, device).to(device)

        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 使用Adam优化器 (输入为评估网络(策略网络)的参数和学习率)，更新模型参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新参数，更新目标网络
        self.device = device

    def min_max_scaling_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        scaled_tensor = (tensor - min_val) / (max_val - min_val)
        return scaled_tensor

    def take_action(self, state_fc, state_conv):
        # 将状态转为tensor类型
        state_fc = torch.tensor([state_fc], dtype=torch.float).to(self.device)  # 全连接层
        state_conv = torch.tensor(state_conv, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)  # 卷积层

        # 对state_fc进行归一化
        state_fc_normalized = self.min_max_scaling_tensor(state_fc)
        # 对state_conv进行归一化
        state_conv_normalized = self.min_max_scaling_tensor(state_conv)

        # 获取动作概率分布
        action_probs = self.actor(state_fc_normalized, state_conv_normalized)
        # 使用概率分布获取动作
        action_dist = torch.distributions.Categorical(action_probs)
        actions = action_dist.sample()
        return actions.tolist()

    def calc_target(self, rewards, next_states_fc, next_states_conv, dones):
        # 得到下一状态的概率分布
        next_probs = self.actor(next_states_fc, next_states_conv)
        # 计算下一状态的目标Q值
        q1_value = self.target_critic_1(next_states_fc, next_states_conv)
        q2_value = self.target_critic_2(next_states_fc, next_states_conv)
        next_probs = next_probs.view(q1_value.size(0), q1_value.size(1))
        # 计算下一状态的熵
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        # 用min操作选择两个Q值中较小的一个，并与动作概率相乘
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        # 计算目标Q值
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    # 更新目标Q网络的参数
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states_fc = torch.tensor(transition_dict['states_fc'],
                                 dtype=torch.float).to(self.device)
        states_conv = torch.tensor(transition_dict['states_conv'], dtype=torch.float) \
            .unsqueeze(0).mean(dim=1, keepdim=True).to(self.device)  # 卷积层


        actions = torch.tensor(transition_dict['actions']).unsqueeze(0).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states_fc = torch.tensor(transition_dict['next_states_fc'],
                                      dtype=torch.float).to(self.device)
        next_states_conv = torch.tensor(transition_dict['next_states_conv'], dtype=torch.float) \
            .unsqueeze(0).mean(dim=1, keepdim=True).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states_fc, next_states_conv, dones)

        # 计算每个动作的 Q 值
        critic_1_q_values = self.critic_1(states_fc, states_conv).gather(1, actions)
        # 对 Q 值进行整合，例如取平均值
        average_1_q_value = torch.mean(critic_1_q_values, dim=1)
        # 调整average_1_q_value维度
        average_1_q_value = average_1_q_value.expand_as(td_target)
        critic_1_loss = torch.mean(
            F.mse_loss(average_1_q_value, td_target.detach()))

        critic_2_q_values = self.critic_2(states_fc, states_conv).gather(1, actions)
        # 对 Q 值进行整合，例如取平均值
        average_2_q_value = torch.mean(critic_2_q_values, dim=1)
        # 调整average_2_q_value维度
        average_2_q_value = average_2_q_value.expand_as(td_target)
        critic_2_loss = torch.mean(
            F.mse_loss(average_2_q_value, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states_fc, states_conv)
        # 计算q值
        q1_value = self.critic_1(states_fc, states_conv)
        q2_value = self.critic_2(states_fc, states_conv)
        # 调整张量的形状
        probs = probs.view(q1_value.size(0), q2_value.size(1))

        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵，求期望(和连续动作空间不同点,连续动作空间直接从策略网络输出，取负值)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha（温度参数，和熵有关）值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 更新目标Q网络的参数(使用两个V critic, 提高算法稳定性)
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
