import torch
import torch.nn as nn
import torch.nn.functional as F
# class Config():
#
#
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        hidden_size = config.hidden_size
        # num_attention_heads = config.num_attention_heads
        # if hidden_size % num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (hidden_size, num_attention_heads))
        # self.num_attention_heads = num_attention_heads
        # self.attention_head_size = int(hidden_size / num_attention_heads)
        # self.all_head_size = hidden_size

        self.encoder = nn.LSTM(input_size=config.input_size, # 9 ，多一组情绪值
                            hidden_size=config.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=config.dropout_rate)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size, 1))
        self.decoder = nn.Linear(config.hidden_size, 2)

        nn.init.uniform_(self.w_omega, -0.1, 0.1) # 初始化参数
        nn.init.uniform_(self.u_omega, -0.1, 0.1) # 初始化参数

    def forward(self, x):
        # print(x.size()) # 64, 10, 8
        out, hidden = self.encoder(x) # 64, 10, 16
        print(out.size())
        # Attention过程
        u = torch.tanh(torch.matmul(out, self.w_omega))
        print(u.size())
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega) # 64, 10, 1
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1) # 64, 10, 1
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = out * att_score # 64, 10, 8
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和 64, 8
        print(feat.size())
        # feat形状是(batch_size, 2 * num_hiddens)
        out = self.decoder(feat)
        # out形状是(batch_size, 2)
        out = torch.tanh(out)
        return out