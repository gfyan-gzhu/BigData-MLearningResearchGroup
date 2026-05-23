import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, negative_sampling, to_dense_adj
from torch_scatter import scatter_add


# =========================
# 时间先验
# =========================
class TimeAwarePrior(nn.Module):
    def __init__(self, latent_dim, hidden_dim, device):
        super(TimeAwarePrior, self).__init__()
        self.prior_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        ).to(device)

    def forward(self, z_prev):
        out = self.prior_net(z_prev)
        mu_prior = out[:, :out.size(1) // 2]
        logvar_prior = out[:, out.size(1) // 2:]
        std_prior = torch.exp(0.5 * logvar_prior)
        return mu_prior, std_prior


# =========================
# 单个 Beta Wavelet Filter
# W_{p,q} = ((L/2)^p (I-L/2)^q) / (2 B(p+1, q+1))
# =========================
class BetaWaveletFilter(nn.Module):
    def __init__(self, device, p=1, q=1):
        super(BetaWaveletFilter, self).__init__()
        self.device = device
        self.p = p
        self.q = q

    def normalized_propagation(self, edge_index, x, num_nodes):
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index

        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_j = x[col] * norm.view(-1, 1)
        out = scatter_add(x_j, row, dim=0, dim_size=num_nodes)
        return out

    def laplacian_op(self, edge_index, x, num_nodes):
        # Lx = x - D^{-1/2} A D^{-1/2} x
        Ax = self.normalized_propagation(edge_index, x, num_nodes)
        return x - Ax

    def apply_half_L(self, edge_index, x, num_nodes):
        return 0.5 * self.laplacian_op(edge_index, x, num_nodes)

    def apply_I_minus_half_L(self, edge_index, x, num_nodes):
        return x - 0.5 * self.laplacian_op(edge_index, x, num_nodes)

    def repeated_op(self, op_type, edge_index, x, num_nodes, times):
        out = x
        for _ in range(times):
            if op_type == 'half_L':
                out = self.apply_half_L(edge_index, out, num_nodes)
            elif op_type == 'I_minus_half_L':
                out = self.apply_I_minus_half_L(edge_index, out, num_nodes)
        return out

    def beta_func(self, a, b):
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # (L/2)^p
        out = self.repeated_op('half_L', edge_index, x, num_nodes, self.p)

        # (I - L/2)^q
        out = self.repeated_op('I_minus_half_L', edge_index, out, num_nodes, self.q)

        norm_const = 2.0 * self.beta_func(self.p + 1, self.q + 1)
        out = out / norm_const
        return out


# =========================
# 多组 Beta Wavelet Filter Bank
# 使用 (0,C), (1,C-1), ..., (C,0)
# =========================
class BetaWaveletBank(nn.Module):
    def __init__(self, hidden_dim, device, C=1):
        super(BetaWaveletBank, self).__init__()
        self.device = device
        self.C = C

        self.filters = nn.ModuleList([
            BetaWaveletFilter(device=device, p=i, q=C - i)
            for i in range(C + 1)
        ])

        self.fuse = nn.Linear((C + 1) * hidden_dim, hidden_dim).to(device)

    def forward(self, x, edge_index):
        multi_scale = []
        for filt in self.filters:
            z = filt(x, edge_index)   # [N, hidden_dim]
            multi_scale.append(z)

        z_cat = torch.cat(multi_scale, dim=1)   # [N, (C+1)*hidden_dim]
        z_fused = self.fuse(z_cat)              # [N, hidden_dim]
        return z_fused


# =========================
# Beta Wavelet Conv
# 对应论文里的 B(A_t, Y)
# =========================
class BetaWaveletConv(nn.Module):
    def __init__(self, hidden_dim, device, C=1):
        super(BetaWaveletConv, self).__init__()
        self.bank = BetaWaveletBank(hidden_dim=hidden_dim, device=device, C=C).to(device)

    def forward(self, x, edge_index):
        return self.bank(x, edge_index)


# =========================
# BW-GRU Cell
# Z_t = sigmoid(B(A, W_z x + U_z h_prev + b_z))
# R_t = sigmoid(B(A, W_r x + U_r h_prev + b_r))
# H~_t = tanh(B(A, W_h x + U_h(r_t * h_prev) + b_h))
# =========================
class BetaGraphGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, C=1):
        super(BetaGraphGRUCell, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # update gate
        self.W_z = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.bw_z = BetaWaveletConv(hidden_dim, device, C=C).to(device)

        # reset gate
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.bw_r = BetaWaveletConv(hidden_dim, device, C=C).to(device)

        # candidate hidden state
        self.W_h = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.bw_h = BetaWaveletConv(hidden_dim, device, C=C).to(device)

    def forward(self, x, edge_index, h_prev):
        # 更新门
        z_linear = self.W_z(x) + self.U_z(h_prev)
        z_t = torch.sigmoid(self.bw_z(z_linear, edge_index))

        # 重置门
        r_linear = self.W_r(x) + self.U_r(h_prev)
        r_t = torch.sigmoid(self.bw_r(r_linear, edge_index))

        # 候选隐藏状态
        h_linear = self.W_h(x) + self.U_h(r_t * h_prev)
        h_tilde = torch.tanh(self.bw_h(h_linear, edge_index))

        # GRU 更新
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_t


# =========================
# BW-GRU
# =========================
class BetaGraphGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, device, C=1):
        super(BetaGraphGRU, self).__init__()
        self.device = device
        self.layer_num = layer_num
        self.cells = nn.ModuleList([
            BetaGraphGRUCell(
                input_dim=input_size if i == 0 else hidden_size,
                hidden_dim=hidden_size,
                device=device,
                C=C
            )
            for i in range(layer_num)
        ])

    def forward(self, x, edge_index, h):
        h_list = []
        current_input = x
        for i in range(self.layer_num):
            h_i = self.cells[i](current_input, edge_index, h[i])
            h_list.append(h_i)
            current_input = h_i
        return torch.stack(h_list, dim=0)   # [layer_num, N, hidden_size]


# =========================
# Generative
# =========================
class Generative(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, layer_num, device, C=1):
        super(Generative, self).__init__()
        self.device = device

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU()
        ).to(device)

        self.rnn = BetaGraphGRU(h_dim, h_dim, layer_num, device, C=C)

        self.enc_mean = nn.Linear(h_dim, z_dim).to(device)
        self.enc_std = nn.Linear(h_dim, z_dim).to(device)

        self.time_aware_prior = TimeAwarePrior(z_dim, h_dim, device)

    def forward(self, x, h, z_prev, edge_index):
        phiX = self.phi_x(x)   # [N, h_dim]

        h_out = self.rnn(phiX, edge_index, h)   # [layer_num, N, h_dim]
        H_t = h_out[-1]                         # [N, h_dim]

        enc_x_mean = self.enc_mean(H_t)                          # [N, z_dim]
        enc_x_std = F.softplus(self.enc_std(H_t)) + 1e-3        # [N, z_dim]

        if z_prev is None:
            prior_x_mean = torch.zeros_like(enc_x_mean)
            prior_x_std = torch.ones_like(enc_x_std)
        else:
            prior_x_mean, prior_x_std = self.time_aware_prior(z_prev)

        z = self.random_sample(enc_x_mean, enc_x_std)

        return (prior_x_mean, prior_x_std), (enc_x_mean, enc_x_std), z, h_out

    def random_sample(self, mean, std):
        eps = torch.randn_like(std)
        return eps * std + mean


# =========================
# Dual Decoder
# =========================
class DualDecoder(nn.Module):
    """
    属性解码 + 结构解码
    """
    def __init__(self, latent_dim, feature_dim, hidden_dim, layer_num, device, C=1):
        super(DualDecoder, self).__init__()
        self.device = device
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim

        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        ).to(device)

        self.attr_decoder_gru = BetaGraphGRU(hidden_dim, hidden_dim, layer_num, device, C=C)

        self.attr_linear = nn.Linear(hidden_dim, feature_dim).to(device)

        self.struct_weight = Parameter(torch.FloatTensor(latent_dim, latent_dim).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.struct_weight.size(1))
        self.struct_weight.data.uniform_(-stdv, stdv)

    def forward(self, z_t, edge_index, h_prev_dec, num_nodes):
        phiZ = self.phi_z(z_t)   # [N, hidden_dim]

        h_out_dec = self.attr_decoder_gru(phiZ, edge_index, h_prev_dec)   # [L, N, hidden_dim]
        h_t_dec = h_out_dec[-1]                                            # [N, hidden_dim]

        x_recon_t = self.attr_linear(h_t_dec)                              # [N, feature_dim]

        adj_recon_t = torch.sigmoid(torch.mm(torch.mm(z_t, self.struct_weight), z_t.t()))

        return x_recon_t, adj_recon_t, h_out_dec


# =========================
# 整体模型
# =========================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # C=1 表示使用 (0,1) 和 (1,0) 两个滤波器
        # C=2，对应 (0,2)(1,1)(2,0)
        self.encoder = Generative(
            args.x_dim, args.h_dim, args.z_dim,
            args.layer_num, args.device, C=getattr(args, 'wavelet_C', 1)
        )
        self.decoder = DualDecoder(
            args.z_dim, args.x_dim, args.h_dim,
            args.layer_num, args.device, C=getattr(args, 'wavelet_C', 1)
        )

        self.mse = nn.MSELoss(reduction='none')
        self.eps = args.eps
        self.device = args.device
        self.layer_num = args.layer_num
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.EPS = 1e-15

    def forward(self, dataloader, y_rect=None, h_t=None):
        kld_loss_total, recon_loss_total = 0, 0
        attr_loss_total, struct_loss_total = 0, 0
        all_z, all_node_idx = [], []
        score_list, next_y_list = [], []

        z_prev = None

        # 初始化解码器隐藏状态
        h_t_dec = None

        for t, data in enumerate(dataloader):
            data = data.to(self.device)
            x, edge_index, node_index = data.x, data.edge_index, data.node_index
            num_nodes = x.size(0)

            if h_t is None:
                h_t = torch.zeros(self.layer_num, num_nodes, self.h_dim, device=self.device)
            if h_t_dec is None:
                h_t_dec = torch.zeros(self.layer_num, num_nodes, self.h_dim, device=self.device)

            # 编码
            (prior_mean_t, prior_std_t), (enc_mean_t, enc_std_t), z_t, h_t = self.encoder(
                x, h_t, z_prev, edge_index
            )
            z_prev = z_t.detach()

            # 解码
            x_recon_t, adj_recon_t, h_t_dec = self.decoder(z_t, edge_index, h_t_dec, num_nodes)

            # KL
            kl_div = self._kld_gauss_elementwise(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            kld_loss_total += kl_div.mean()

            # 属性重构误差
            attr_err = torch.mean(self.mse(x, x_recon_t), dim=1)

            # 结构重构误差（训练用：正边 + 负采样）
            W = self.decoder.struct_weight  # [z_dim, z_dim]

            # ===== 正边 =====
            if edge_index.size(1) == 0:
                attr_loss_mean = attr_err.mean()
                struct_loss_mean = torch.tensor(0.0, device=self.device)
                attr_loss_total += attr_loss_mean
                struct_loss_total += struct_loss_mean
                recon_loss_total += attr_loss_mean + struct_loss_mean
                struct_err_node = torch.zeros(num_nodes, device=self.device)
            else:
                u, v = edge_index
                pos_logits = torch.sum((z_t[u] @ W) * z_t[v], dim=1)
                pos_prob = torch.sigmoid(pos_logits)
                pos_loss = -torch.log(pos_prob + 1e-8)

                num_neg = max(1, int(u.size(0)))
                neg_edge_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=num_neg
                )
                u_neg, v_neg = neg_edge_index
                neg_logits = torch.sum((z_t[u_neg] @ W) * z_t[v_neg], dim=1)
                neg_prob = torch.sigmoid(neg_logits)
                neg_loss = -torch.log(1.0 - neg_prob + 1e-8)

                struct_loss = torch.cat([pos_loss, neg_loss], dim=0)
                attr_loss_mean = attr_err.mean()
                struct_loss_mean = struct_loss.mean()
                attr_loss_total += attr_loss_mean
                struct_loss_total += struct_loss_mean
                recon_loss_total += attr_loss_mean + struct_loss_mean

                struct_err_node = torch.zeros(num_nodes, device=self.device)
                edge_count = torch.zeros(num_nodes, device=self.device)
                struct_err_node.index_add_(0, u, pos_loss)
                struct_err_node.index_add_(0, v, pos_loss)

                edge_count.index_add_(0, u, torch.ones_like(pos_loss))
                edge_count.index_add_(0, v, torch.ones_like(pos_loss))

                struct_err_node = struct_err_node / (edge_count + 1e-8)

            # 7. 节点异常打分（仅验证/测试）
            if not self.training:
                def robust_norm(tensor):
                    q75 = torch.quantile(tensor, 0.75)
                    q25 = torch.quantile(tensor, 0.25)
                    iqr = q75 - q25
                    if iqr > 0:
                        return (tensor - q25) / (iqr + 1e-8)
                    else:
                        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


                attr_score = robust_norm(torch.log1p(attr_err))
                struct_score = robust_norm(torch.log1p(struct_err_node))

                total_score = attr_score + struct_score
                node_anomaly_score = total_score
                score_list.append(node_anomaly_score.unsqueeze(1))

                print(
                    f"[Time {t}] "
                    f"attr_score_mean={attr_score.mean().item():.4f}, "
                    f"struct_score_mean={struct_score.mean().item():.4f}, "
                    f"struct_err_mean={struct_err_node.mean().item():.4f}"
                )
            else:
                score_list.append(torch.zeros(num_nodes, 1, device=self.device))



            all_z.append(z_t)
            all_node_idx.append(node_index)

        recon_loss = recon_loss_total / len(dataloader)
        kld_loss = kld_loss_total / len(dataloader)
        attr_loss = attr_loss_total / len(dataloader)
        struct_loss = struct_loss_total / len(dataloader)

        return (
            struct_loss,
            attr_loss,
            recon_loss,
            kld_loss,
            torch.tensor(0.0, device=self.device),
            next_y_list,
            h_t,
            score_list
        )

    def _kld_gauss_elementwise(self, mean_1, std_1, mean_2, std_2):
        kld_element = (
            2 * torch.log(std_2 + self.eps)
            - 2 * torch.log(std_1 + self.eps)
            + (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2))
            / torch.pow(std_2 + self.eps, 2)
            - 1
        )
        return 0.5 * torch.sum(kld_element, dim=1)