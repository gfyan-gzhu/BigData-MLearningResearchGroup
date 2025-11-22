import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_decide(self, state_fc, state_conv, action, reward, next_state_fc, next_state_conv, done):
        self.buffer.append((state_fc, state_conv, action, reward, next_state_fc, next_state_conv, done))

    def sample_decide(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state_fc, state_conv, action, reward, next_state_fc, next_state_conv, done = zip(*transitions)
        return np.array(state_fc), state_conv, sum(action, []), reward, np.array(next_state_fc), \
               next_state_conv, done

    def add_rate(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_rate(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), sum(action, []), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class ReplayBufferRate:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), sum(action, []), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def save_sac_model(agent):
    torch.save(agent.actor.state_dict(), 'model\sac\\actor.pth')
    torch.save(agent.critic_1.state_dict(), 'model\sac\critic_1.pth')
    torch.save(agent.critic_2.state_dict(), 'model\sac\critic_2.pth')
    torch.save(agent.target_critic_1.state_dict(), 'model\sac\\target_critic_1.pth')
    torch.save(agent.target_critic_2.state_dict(), 'model\sac\\target_critic_2.pth')


def save_ddpg_model(agent):
    torch.save(agent.actor.state_dict(), 'model\ddpg\\actor.pth')
    torch.save(agent.critic.state_dict(), 'model\ddpg\critic.pth')
    torch.save(agent.target_actor.state_dict(), 'model\ddpg\\target_actor.pth')
    torch.save(agent.target_critic.state_dict(), 'model\ddpg\\target_critic.pth')


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
