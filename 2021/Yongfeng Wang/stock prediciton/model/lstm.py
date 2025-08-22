import torch
import torch.nn as nn
import torch.nn.functional as F
# class Config():
#
#
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=config.dropout_rate)
        self.linear = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)
        self.maxpool = nn.MaxPool1d(config.time_step)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, hidden = self.lstm(x) # 64, 5, 1024
        print(out.size())
        out = torch.tanh(out)
        out = out.permute(0, 2, 1) # 64, 1024, 5 / 64, 73, 5
        out = self.maxpool(out).squeeze() # 64, 1024 / 64, 73
        out = self.linear(out) # 64, 2
        # out = out.softmax(dim=1)
        out = torch.tanh(out)
        # print(out.size())
        return out