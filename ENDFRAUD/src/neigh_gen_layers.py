import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, config=None, bias=True, device=None):
        super(GraphConvolution, self).__init__()
        self.in_feats = in_features
        self.out_feats = out_features
        # Determine device
        if device is not None:
            self.device = device
        elif config is not None and hasattr(config, 'cuda') and config.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.weight = Parameter(torch.rand(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.rand(out_features))
        else:
            self.register_parameter('bias', None)
        # Move parameters to device
        self.to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_feats) + ' -> ' \
            + str(self.out_feats) + ')'
