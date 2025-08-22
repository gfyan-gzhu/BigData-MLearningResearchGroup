import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if True and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU

torch.manual_seed(41)
lstm_layers = 1
# Bi-LSTM模型
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(config.input_size,
                            config.hidden_size,
                            num_layers=lstm_layers,
                            bidirectional=True)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(config.hidden_size, config.output_size)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x):
        # lstm的输入维度为 [seq_len, batch, input_size]
        # x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)  # [sentence_length, batch_size, embedding_size]
        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)

        # 设置lstm最初的前项输出
        # h_0 = torch.randn(1,0.1,-0.1).to(device)
        # c_0 = torch.randn(lstm_layers * self.config.num_directions, batch_size, self.config.hidden_size).to(device)

        # out[seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x)

        # 将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        out = forward_out + backward_out  # [seq_len, batch, hidden_size] # 10, 32, 8

        out = out.permute(1, 0, 2)  # [batch, seq_len, hidden_size] # 32, 10, 8
        # 为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  # [batch, num_layers * num_directions,  hidden_size] # 32, 10, 8
        h_n = torch.sum(h_n, dim=1)  # [batch, 1,  hidden_size] # 32, 10, 8
        h_n = h_n.squeeze(dim=1)  # [batch, hidden_size] # 32, 10, 8

        attention_w = self.attention_weights_layer(h_n)  # [batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1)  # [batch, 1, hidden_size]
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  # [batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  # [batch, 1, seq_len],权重归一化

        x = torch.bmm(softmax_w, out)  # [batch, 1, hidden_size]
        x = x.squeeze(dim=1)  # [batch, hidden_size]
        x = self.liner(x)
        x = self.act_func(x)
        return x


