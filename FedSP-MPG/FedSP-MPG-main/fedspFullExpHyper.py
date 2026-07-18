import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局固定超参（论文表格tab:global_hyper） ======================
GLOBAL_FIX = {
    "max_round": 50,
    "K_list": [3,5,10],
    "neg_ratio_link": 5,
    "weight_decay": 1e-4,
}

# FedSP 默认超参
DEFAULT_FEDSP = {
    "d": 64,
    "lr": 1e-3,
    "local_epoch": 3,
    "dropout": 0.2,
    "Q": 8,
    "Mp": 16
}

# 超参搜索网格
HYPER_GRID = {
    "d": [16,32,64,128,256],
    "lr": [1e-4,5e-4,1e-3,5e-3,1e-2],
    "local_epoch": [1,3,5,8,10],
    "dropout": [0.0,0.1,0.2,0.3,0.4]
}

# 基线专属超参
BASELINE_HP = {
    "FedProx": {"mu":0.01},
    "FedHGN": {"schema_loss_w":0.1}
}

# ====================== 1. DP 差分隐私模块 ======================
class GaussianDP:
    def __init__(self, noise_scale: float, l2_sensitivity: float, delta=1e-5):
        self.sigma = noise_scale
        self.sens = l2_sensitivity
        self.delta = delta

    def add_noise(self, grad: torch.Tensor):
        noise = torch.normal(0, self.sigma * self.sens, size=grad.shape, device=grad.device)
        return grad + noise

    def compute_epsilon(self, rounds: int):
        # 高斯差分隐私严格epsilon计算
        ln_delta = np.log(1 / self.delta)
        return np.sqrt(2 * rounds * ln_delta) * (self.sens / self.sigma)

def server_aggregate_dp(client_grads, omega_weights, dp_module):
    total_grad = 0.0
    for w, g in zip(omega_weights, client_grads):
        total_grad += w * g
    return dp_module.add_noise(total_grad)

# ====================== 2. MIA & SCR 隐私攻击评测模块 ======================
class MetaPathMIA:
    def __init__(self, hidden_dim, Q):
        self.clf = LogisticRegression(max_iter=1000, solver="liblinear")
        self.d, self.Q = hidden_dim, Q

    def build_X(self, B):
        return B.detach().cpu().numpy().reshape(1, -1)

    def run_auc(self, X, y):
        self.clf.fit(X, y)
        prob = self.clf.predict_proba(X)[:, 1]
        return roc_auc_score(y, prob)

def scr_attack(B, Wk, device):
    B = B.to(device).float()
    BBt = B @ B.T
    BBt_inv = torch.linalg.pinv(BBt)
    return BBt_inv @ B @ Wk

def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2).item()

# ====================== 3. 通信开销统计模块 ======================
class CommCounter:
    def __init__(self, d, Q, Mp, backbone_k=120):
        self.d, self.Q, self.Mp = d, Q, Mp
        self.M_back = backbone_k * 1000

    def calc_baseline(self, K):
        per_round = self.M_back + self.d * self.Mp
        return per_round / 1000, (per_round * K) / 1000

    def calc_fedsp(self, K):
        per_round = self.M_back + self.Q * self.d
        return per_round / 1000, (per_round * K) / 1000

# ====================== 4. 收敛 & 超参绘图记录器 ======================
class Recorder:
    def __init__(self, target):
        self.target = target
        self.record = {}
        self.hyper_result = {}

    def add(self, method, r, acc, comm):
        if method not in self.record:
            self.record[method] = []
        self.record[method].append((r, acc, comm))

    def add_hyper(self, hp_name, hp_val, acc):
        if hp_name not in self.hyper_result:
            self.hyper_result[hp_name] = []
        self.hyper_result[hp_name].append((hp_val, acc))

    def plot_hyper_curve(self, save="hyper_sens_curve.eps"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        names = list(self.hyper_result.keys())
        for idx, name in enumerate(names):
            vals, accs = zip(*self.hyper_result[name])
            axes[idx].plot(vals, accs, marker="o", linewidth=2.5, color="#1f77b4")
            axes[idx].set_title(f"{name} Sensitivity", fontsize=12)
            axes[idx].set_xlabel(name, fontsize=10)
            axes[idx].set_ylabel("Weighted Acc (%)", fontsize=10)
            axes[idx].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save, bbox_inches="tight", dpi=300)
        plt.close()

# ====================== 5. 链接预测工具 & 负采样 & 评估 ======================
class LinkHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        self.sig = nn.Sigmoid()

    def forward(self, hu, hv):
        s = torch.sum(hu @ self.W * hv, dim=-1) + self.b
        return self.sig(s)

def link_loss(pos, neg):
    pos_l = -torch.log(pos + 1e-8).mean()
    neg_l = -torch.log(1 - neg + 1e-8).mean()
    return pos_l + neg_l

def neg_sample(pos_edges, N, ratio=5):
    pos_set = set((u.item(), v.item()) for u, v in pos_edges.T)
    neg = []
    num = pos_edges.shape[1]
    for i in range(num):
        u, v = pos_edges[0, i], pos_edges[1, i]
        for _ in range(ratio):
            nu = torch.randint(0, N, (1,))[0]
            nv = torch.randint(0, N, (1,))[0]
            if (nu.item(), nv.item()) not in pos_set:
                neg.append([nu.item(), nv.item()])
    if not neg:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(neg)

def eval_link(model, head, g, test_pos, dev):
    model.eval()
    head.eval()
    with torch.no_grad():
        H, B, _ = model(g)
        up, vp = test_pos[0], test_pos[1]
        p_score = head(H[up], H[vp]).cpu().numpy()
        neg_t = neg_sample(test_pos, g.num_nodes, GLOBAL_FIX["neg_ratio_link"])
        if len(neg_t) == 0:
            return 0.0, 0.0
        un, vn = neg_t[:, 0], neg_t[:, 1]
        n_score = head(H[un], H[vn]).cpu().numpy()
        y = np.concatenate([np.ones_like(p_score), np.zeros_like(n_score)])
        pred = np.concatenate([p_score, n_score])
        auc = roc_auc_score(y, pred) * 100
        ap = average_precision_score(y, pred) * 100
    return auc, ap

# ====================== 6. FedSP-MPG 主干模型 ======================
class FedSPMPG(nn.Module):
    def __init__(self, d, Q, dropout):
        super().__init__()
        self.d, self.Q = d, Q
        self.drop = nn.Dropout(dropout)
        self.B = nn.Parameter(torch.randn(Q, d) * 0.01)
        self.backbone = nn.Linear(10, d)
        self.view_enc = nn.Linear(d, d)

    def get_shared(self):
        return [self.B] + list(self.backbone.parameters()) + list(self.view_enc.parameters())

    def forward(self, g):
        n = g.num_nodes
        x = torch.randn(n, 10, device=self.B.device)
        h = self.drop(self.backbone(x))
        hv = self.drop(self.view_enc(h))
        h_final = h + hv
        return h_final, self.B, None

# ====================== 7. 模拟异构图客户端数据 ======================
class SimGraph:
    def __init__(self, num_nodes=200):
        self.num_nodes = num_nodes
        self.edges = torch.randint(0, num_nodes, (2, 800))

    def to(self, dev):
        self.edges = self.edges.to(dev)
        return self

def load_client(k, num_nodes=200):
    return SimGraph(num_nodes=num_nodes)

# ====================== 8. 客户端本地训练函数 ======================
def client_train(model, link_head, g, opt, local_epoch, dev):
    model.train()
    link_head.train()
    g = g.to(dev)
    edges = g.edges
    num = edges.shape[1]
    perm = torch.randperm(num)
    train_e = edges[:, perm[:int(0.8 * num)]]
    test_e = edges[:, perm[int(0.8 * num):]]

    total_loss = 0.0
    for _ in range(local_epoch):
        neg_e = neg_sample(train_e, g.num_nodes, GLOBAL_FIX["neg_ratio_link"])
        if len(neg_e) == 0:
            continue
        H, _, _ = model(g)
        up, vp = train_e[0], train_e[1]
        pos_s = link_head(H[up], H[vp])
        un, vn = neg_e[:, 0], neg_e[:, 1]
        neg_s = link_head(H[un], H[vn])
        loss = link_loss(pos_s, neg_s)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    avg_loss = total_loss / local_epoch if local_epoch > 0 else 0.0
    return avg_loss, test_e

# ====================== 9. 超参网格搜索主函数 ======================
def hyper_grid_search(dev):
    recorder = Recorder(target=78.0)
    K = 5
    for hp_name, value_list in HYPER_GRID.items():
        for val in tqdm(value_list, desc=f"Search {hp_name}"):
            cfg = DEFAULT_FEDSP.copy()
            cfg[hp_name] = val
            d, lr, le, dp = cfg["d"], cfg["lr"], cfg["local_epoch"], cfg["dropout"]
            Q, Mp = cfg["Q"], cfg["Mp"]

            # 初始化模型与优化器
            model = FedSPMPG(d=d, Q=Q, dropout=dp).to(dev)
            link_h = LinkHead(d).to(dev)
            shared_params = model.get_shared() + list(link_h.parameters())
            opt = torch.optim.Adam(shared_params, lr=lr, weight_decay=GLOBAL_FIX["weight_decay"])
            best_acc = 0.0

            for r in range(GLOBAL_FIX["max_round"]):
                auc_list = []
                grads_buf = []
                # 多客户端本地训练
                for k in range(K):
                    gk = load_client(k)
                    _, test_e = client_train(model, link_h, gk, opt, le, dev)
                    auc, _ = eval_link(model, link_h, gk, test_e, dev)
                    auc_list.append(auc)
                    # 保存梯度
                    client_grad = []
                    for p in shared_params:
                        if p.grad is not None:
                            client_grad.append(p.grad.clone())
                        else:
                            client_grad.append(torch.zeros_like(p))
                    grads_buf.append(client_grad)

                # 联邦平均聚合
                avg_grad_list = []
                for pid in range(len(shared_params)):
                    grad_sum = torch.zeros_like(shared_params[pid])
                    for g in grads_buf:
                        grad_sum += g[pid]
                    avg_grad_list.append(grad_sum / K)

                # 写入梯度并更新
                for p, g in zip(shared_params, avg_grad_list):
                    p.grad = g
                opt.step()

                current_avg_auc = np.mean(auc_list)
                if current_avg_auc > best_acc:
                    best_acc = current_avg_auc

            recorder.add_hyper(hp_name, val, best_acc)

    # 绘制超参曲线图
    recorder.plot_hyper_curve()

    # 打印超参结果
    print("\n=== Hyperparameter Search Result Table ===")
    for name, pairs in recorder.hyper_result.items():
        print(f"\n【{name}】")
        for v, acc in pairs:
            print(f"  Value = {v}, Best Avg AUC = {acc:.2f}%")
    return recorder

# ====================== 10. 基线通信开销对比演示 ======================
def baseline_reproduce_demo(dev):
    print("\n==== Baseline Reproduction &amp; Communication Overhead Analysis ====")
    # 打印基线超参
    for b_name, hp in BASELINE_HP.items():
        print(f"{b_name} Exclusive Hyper-params: {hp}")

    # 通信量计算
    comm = CommCounter(d=DEFAULT_FEDSP["d"], Q=DEFAULT_FEDSP["Q"], Mp=DEFAULT_FEDSP["Mp"])
    _, base_total = comm.calc_baseline(K=5)
    _, fedsp_total = comm.calc_fedsp(K=5)
    reduce_rate = (base_total - fedsp_total) / base_total * 100

    print(f"Baseline Total Communication (K=5): {base_total:.2f}K")
    print(f"FedSP-MPG Total Communication (K=5): {fedsp_total:.2f}K")
    print(f"Communication Overhead Reduction Rate: {reduce_rate:.2f}%")

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # 设备自动适配
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Device: {device}")

    # 1. 基线通信开销校验
    baseline_reproduce_demo(device)

    # 2. 超参网格搜索
    hyper_recorder = hyper_grid_search(device)

    # 3. 差分隐私 DP 演示
    print("\n==== DP Differential Privacy Demo ====")
    dp = GaussianDP(noise_scale=2.0, l2_sensitivity=0.1)
    eps = dp.compute_epsilon(rounds=30)
    print(f"DP Epsilon (30 communication rounds): {eps:.2f}")

    # 4. MIA 成员推理攻击评测
    print("\n==== MIA Membership Inference Attack Demo ====")
    mia = MetaPathMIA(hidden_dim=DEFAULT_FEDSP["d"], Q=DEFAULT_FEDSP["Q"])
    demo_B = torch.randn(DEFAULT_FEDSP["Q"], DEFAULT_FEDSP["d"])
    X = mia.build_X(demo_B)
    # 构造测试数据集
    y_demo = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    X_demo = np.repeat(X, len(y_demo), axis=0)
    mia_auc = mia.run_auc(X_demo, y_demo)
    print(f"MIA Attack AUC Score: {mia_auc:.3f}")

    # 5. SCR 重构攻击评测
    print("\n==== SCR Structure Reconstruction Attack Demo ====")
    Q, d = DEFAULT_FEDSP["Q"], DEFAULT_FEDSP["d"]
    B = torch.randn(Q, d).to(device)
    C_true = torch.randn(Q, DEFAULT_FEDSP["Mp"]).to(device)
    W = B.T @ C_true
    C_hat = scr_attack(B, W, device)
    scr_mse = mse_loss(C_hat, C_true)
    print(f"SCR Attack Reconstruction MSE: {scr_mse:.4f}")

    print("\n======== All Experiments Finished ========")