import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # 50x50 -> 50x50
            nn.Conv2d(in_channels=state_dim_conv, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 50x50 -> 25x25
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 25x25 -> 25x25
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 25x25 -> 12x12
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),  # 12x12 -> 12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12x12 -> 6x6
        )

        # 卷积层处理的状态 torch.Size([1, 1, 50, 50])
        self.conv1 = nn.Conv2d(in_channels=state_dim_conv, out_channels=hidden_dim, kernel_size=(3, 3))
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=5)

        # 全局平均池化层，将特征图尺寸减小到 [1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层处理的状态
        self.fc2 = nn.Linear(state_dim_fc, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # 共享的隐藏层,全连接层
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        # 初始化网络参数
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 对卷积层使用初始化函数
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # 对线性层使用初始化函数
                init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

    def forward(self, x_fc, x_conv):
        # 卷积层处理
        x_conv = self.conv_layers(x_conv)
        # 全局平均池化，将特征图尺寸减小到 [1, 1]
        x_conv = self.global_avg_pool(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)  # 展平卷积输出
        x_conv = F.relu(self.fc1(x_conv))
        # 全连接层处理
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = F.relu(self.fc3(x_fc))
        # 合并卷积和全连接层的输出
        x = x_conv + x_fc
        # 共享隐藏层
        x = F.relu(self.shared_fc(x))
        # 输出动作的概率分布
        action_probs = F.softmax(self.fc4(x), dim=1)
        return action_probs


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim):
        super(ValueNet, self).__init__()
        self.state_dim_conv = state_dim_conv
        self.conv_layers = nn.Sequential(
            # 50x50 -> 50x50
            nn.Conv2d(in_channels=state_dim_conv, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 50x50 -> 25x25
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 25x25 -> 25x25
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 25x25 -> 12x12
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),  # 12x12 -> 12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12x12 -> 6x6
        )

        # 卷积层处理的状态
        self.conv1 = nn.Conv2d(in_channels=state_dim_conv, out_channels=hidden_dim,
                               kernel_size=(3, 3), )
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=5)

        # 全局平均池化层，将特征图尺寸减小到 [1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层处理的状态
        self.fc2 = nn.Linear(state_dim_fc, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # 共享的隐藏层,全连接层
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc4 = nn.Linear(hidden_dim, 1)
        # 初始化网络参数
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 对卷积层使用初始化函数
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # 对线性层使用初始化函数
                init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

    def forward(self, x_fc, x_conv):
        # 卷积层处理
        x_conv = self.conv_layers(x_conv)
        # 全局平均池化，将特征图尺寸减小到 [1, 1]
        x_conv = self.global_avg_pool(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)  # 展平卷积输出
        x_conv = F.relu(self.fc1(x_conv))
        # 全连接层处理
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = F.relu(self.fc3(x_fc))
        # 合并卷积和全连接层的输出
        x = x_conv + x_fc
        # 共享隐藏层
        x = F.relu(self.shared_fc(x))
        # x = F.tanh(x)
        return self.fc4(x)


class PPO:
    """PPO算法,采用截断方式 """

    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        """
        :param state_dim_conv: conv状态维度
        :param state_dim_fc: fc状态维度
        :param hidden_dim: 隐藏层维度
        :param action_dim: 动作维度
        :param actor_lr: actor网络学习率
        :param critic_lr: critic网络学习率
        :param lmbda:计算优势函数的参数
        :param epochs:一条序列的数据用来训练轮数
        :param eps:PPO中截断范围的参数
        :param gamma:折扣因子
        :param device:GPU或CPU运行
        """
        self.actor = PolicyNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim_fc, state_dim_conv, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
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
        probs = self.actor(state_fc_normalized, state_conv_normalized)

        # 你可以根据需要更改用于替换的值
        probs = torch.nan_to_num(probs, nan=0.0)
        # 确保概率分布仍然归一化，因为替换NaN可能会破坏归一化
        probs = probs / probs.sum()
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states_fc = torch.tensor(transition_dict['states_cac'],
                                 dtype=torch.float).to(self.device)
        states_conv = torch.tensor(transition_dict['states_cac_conv'], dtype=torch.float) \
            .unsqueeze(0).mean(dim=1, keepdim=True).to(self.device)  # 卷积层

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states_fc = torch.tensor(transition_dict['next_states'],
                                      dtype=torch.float).to(self.device)
        next_states_conv = torch.tensor(transition_dict['next_states_conv'], dtype=torch.float) \
            .unsqueeze(0).mean(dim=1, keepdim=True).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 检测是否有NaN 或 Inf 值
        assert torch.isfinite(states_fc).all(), "states_fc contain NaN or Inf"
        assert torch.isfinite(states_conv).all(), "states_conv contain NaN or Inf"
        assert torch.isfinite(actions).all(), "actions contain NaN or Inf"
        assert torch.isfinite(rewards).all(), "rewards contain NaN or Inf"
        assert torch.isfinite(next_states_fc).all(), "next_states_fc contain NaN or Inf"
        assert torch.isfinite(next_states_conv).all(), "next_states_conv contain NaN or Inf"
        assert torch.isfinite(dones).all(), "dones contain NaN or Inf"

        # 奖励标准化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        td_target = rewards + self.gamma * self.critic(next_states_fc, next_states_conv) * (1 -
                                                                                            dones)
        td_delta = td_target - self.critic(states_fc, states_conv)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states_fc, states_conv)
                                  .gather(1, actions) + 1e-8).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states_fc, states_conv).gather(1, actions) + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states_fc, states_conv), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
