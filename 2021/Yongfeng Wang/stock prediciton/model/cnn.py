import torch
import torch.nn as nn
import torch.nn.functional as F
# class Config():
#
#
def test():
    print('success')

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(config.input_size, config.cnn_filter, 1)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(0.01)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(in_features=16, out_features=config.output_size)

    def forward(self, x):
        out = x.permute(0, 2, 1) # 32, 10, 8 => 32, 8, 10
        out = self.conv1d(out) # 32,16,10
        out = torch.tanh(out)
        out = self.maxpool(out) # 32, 16, 10
        out = torch.relu(out)
        out = self.maxpool1(out).squeeze() # 32, 16
        out = torch.relu(out)
        out = self.linear(out) # 32, 2
        out = torch.tanh(out)
        return out