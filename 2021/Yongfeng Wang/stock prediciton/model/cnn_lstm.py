import torch
import torch.nn as nn
import torch.nn.functional as F
# class Config():
#
#
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(config.input_size, config.cnn_filter, 1)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(0.01)
        self.lstm = nn.LSTM(input_size=config.cnn_filter,
                            hidden_size=config.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=config.dropout_rate)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x):
        out = x.permute(0, 2, 1) # 64, 10, 9 => 64, 9, 10
        out = self.conv1d(out) # 64, 32, 10
        out = torch.tanh(out)
        out = self.maxpool(out) # 64, 32, 10
        out = torch.relu(out)
        # out = self.drop(out)

        out = out.permute(0, 2, 1) # 64, 32, 10 => 64, 10, 32
        out, _ = self.lstm(out) # 64, 10, 64
        out = torch.tanh(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool1(out).squeeze() # 64, 64
        out = self.linear(out) # 64, 2
        out = torch.tanh(out)
        return out