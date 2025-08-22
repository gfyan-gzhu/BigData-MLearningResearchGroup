import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from config import Config

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, planes, rotio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shareMlp = nn.Sequential(
            nn.Conv1d(planes, planes // rotio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(planes // rotio, planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(self.shareMlp(x))
        max_out = self.max_pool(self.shareMlp(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class cbam(nn.Module):

    def __init__(self, planes):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes)  # planes是feature map的通道个数
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  # 广播机制  64, 16, 1 => 64, 16, 10
        x = self.sa(x) * x  # 广播机制 64, 1, 1 => 64, 16, 10
        return x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.lstm_layers,
                            batch_first=True,
                            dropout=config.dropout_rate)
        self.cbam = cbam(config.hidden_size)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, hidden = self.lstm(x) # 64, 10, 16
        out = torch.tanh(out)
        out = out.transpose(-1, -2) # 64, 16, 10
        out = self.cbam(out)
        out = self.max_pool(out).squeeze()
        out = self.linear(out) # 64, 2
        out = torch.sigmoid(out)
        return out