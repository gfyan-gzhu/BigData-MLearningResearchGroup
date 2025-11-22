import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=1, fc1_dim=256, fc2_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.prob = nn.Linear(fc2_dim, action_dim)


    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.prob(x).view(-1)
        prob = torch.softmax(x, dim=-1)

        return prob
