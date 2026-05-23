# FedSPMPG.py
# Add-on framework (non-intrusive): DOES NOT modify FedHGN.py/HGNModel.py.
# Use framework name in main.py: "FedSP-MPG"

import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch as th
import torch.nn.functional as F
import tqdm

import dgl  # noqa: F401

from Decoders import NodeClassifier
from HGNModel import HGNModel
from utils import load_data, align_schemas, EarlyStopping

from mp_modules_addon import (
    enum_metapaths,
    build_metapath_views,
    MPViewEncoder,
    DecoupledNodeWiseMPGating,
    entropy_from_alpha_mean,
)


LEGACY_ABLATIONS = {"B", "C", "B+C"}
SPMPG_MP_ABLATIONS = {"no_mp", "uniform", "static", "no_residual"}


def _validate_ablation(ablation: Optional[str]) -> None:
    allowed = LEGACY_ABLATIONS | SPMPG_MP_ABLATIONS | {None}
    if ablation not in allowed:
        raise ValueError(
            f"Unknown ablation: {ablation}. "
            f"Allowed: {sorted(x for x in allowed if x is not None)}"
        )


def _uses_private_base_coeffs(ablation: Optional[str]) -> bool:
    """
    这些情形下，FedHGN 底座逻辑保持不动：
    - full model (None)
    - FedHGN legacy ablation "B"
    - 你新增的 4 个 MP 分支消融
    """
    return ablation is None or ablation == "B" or ablation in SPMPG_MP_ABLATIONS


def _macro_micro_f1(y_true: th.Tensor, y_pred: th.Tensor, num_classes: int) -> tuple[float, float]:
    eps = 1e-12
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    tp_total = fp_total = fn_total = 0.0
    f1s = []

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        tp_total += tp
        fp_total += fp
        fn_total += fn

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        f1s.append(f1)

    macro = float(sum(f1s) / max(len(f1s), 1))
    micro = float(2 * tp_total / (2 * tp_total + fp_total + fn_total + eps))
    return macro, micro


class SPMPGEncoder(th.nn.Module):
    """Wrapper: base HGNN + MP-view encoder + shared gating bases."""
    def __init__(self, args: Namespace, out_dim: int, ntypes, etypes, canonical_etypes, num_nodes_dict):
        super().__init__()
        self.base = HGNModel(args, out_dim, ntypes, etypes, canonical_etypes, num_nodes_dict)
        self.mp_view_encoder = MPViewEncoder(
            args.hidden_dim,
            num_layers=args.mp_view_num_layers,
            dropout=args.mp_view_dropout,
        )
        self.mp_gating = DecoupledNodeWiseMPGating(
            args.hidden_dim,
            num_bases=args.mp_num_gating_bases,
        )


class Client:
    def __init__(self, args: Namespace, data: tuple, client_id: int) -> None:
        self.args = args
        self.id = client_id
        self.lr = args.lr
        self.optim = args.optim
        self.weight_decay = args.weight_decay
        self.num_local_epochs = args.num_local_epochs
        self.align_reg = args.align_reg
        self.ablation = args.ablation
        self.task = args.task
        self.device = args.device

        _validate_ablation(self.ablation)

        if _uses_private_base_coeffs(self.ablation):
            self.others_basis_coeffs_encoder = None
            self.others_basis_coeffs_decoder = None

        if self.task != "node_classification":
            raise ValueError(f"Unsupported task for FedSP-MPG: {self.task}")

        g, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = data
        self.g = g.to(self.device)
        self.g_cpu = g if args.device.type == "cpu" else g.to("cpu")

        self.ntypes = g.ntypes
        self.etypes = list(dict.fromkeys(g.etypes))
        self.canonical_etypes = g.canonical_etypes
        self.out_dim = out_dim

        self.train_nid_dict = {k: v.to(self.device) for k, v in train_nid_dict.items()}
        self.val_nid_dict = {k: v.to(self.device) for k, v in val_nid_dict.items()}
        self.test_nid_dict = {k: v.to(self.device) for k, v in test_nid_dict.items()}
        self.num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}

        assert len(self.train_nid_dict.keys()) == 1
        assert len(self.val_nid_dict.keys()) == 1
        assert len(self.test_nid_dict.keys()) == 1
        self.target_ntype = list(self.train_nid_dict.keys())[0]

        # private metapaths
        self.metapaths = enum_metapaths(
            canonical_etypes=list(self.canonical_etypes),
            target_ntype=self.target_ntype,
            max_len=args.mp_max_len,
            max_paths=args.mp_max_paths,
            seed=args.random_seed + 97 * client_id,
        )
        self.mp_views = [vg.to(self.device) for vg in build_metapath_views(self.g_cpu, self.metapaths)]

        print(f"[Client {self.id}] #metapaths={len(self.metapaths)}  #views={len(self.mp_views)}")

        self.encoder = SPMPGEncoder(
            args,
            args.hidden_dim,
            self.ntypes,
            self.etypes,
            self.canonical_etypes,
            self.num_nodes_dict,
        ).to(self.device)
        self.decoder = NodeClassifier(args.hidden_dim, self.out_dim).to(self.device)

        # private per-metapath coefficients
        self.mp_coeffs = self.encoder.mp_gating.init_local_coeffs(
            len(self.mp_views),
            device=self.device,
        )

        self.global_entropy = 0.0
        self.last_entropy = 0.0

    def set_global_entropy(self, value: float) -> None:
        self.global_entropy = float(value)

    def set_others_basis_coeffs(self, others_basis_coeffs_encoder, others_basis_coeffs_decoder):
        if _uses_private_base_coeffs(self.ablation):
            self.others_basis_coeffs_encoder = others_basis_coeffs_encoder
            self.others_basis_coeffs_decoder = others_basis_coeffs_decoder
        else:
            raise AssertionError("This ablation mode does not use private base basis coeffs.")

    def compute_alignment_regularization(self) -> th.Tensor:
        reg = 0
        local_basis_coeffs_encoder = th.stack(
            [th.stack([param for param in param_dict.values()]) for param_dict in self.encoder.base.basis_coeffs_encoder]
        )
        diff = local_basis_coeffs_encoder.unsqueeze(2) - self.others_basis_coeffs_encoder.unsqueeze(1)
        min_diff, _ = th.min(th.sum(th.square(diff), dim=-1), dim=-1)
        reg += min_diff.sum()

        if self.others_basis_coeffs_decoder is not None:
            diff = (
                th.stack([v for v in self.decoder.basis_coeffs_decoder.values()], dim=0).unsqueeze(1)
                - self.others_basis_coeffs_decoder
            )
            min_diff, _ = th.min(th.sum(th.square(diff), dim=-1), dim=-1)
            reg += min_diff.sum()

        return reg

    def _forward_mp_branch(self, h_base: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        z_list = [self.encoder.mp_view_encoder(vg, h_base) for vg in self.mp_views]
        z_stack = th.stack(z_list, dim=0)  # [P, N, D]

        num_paths = z_stack.size(0)

        if self.ablation == "uniform":
            alpha_mean = th.full(
                (num_paths,),
                1.0 / max(num_paths, 1),
                device=self.device,
                dtype=z_stack.dtype,
            )
            fused = z_stack.mean(dim=0)

        elif self.ablation == "static":
            # 客户端级静态权重：先对每条路径做全局评分，再 softmax
            W = self.mp_coeffs @ self.encoder.mp_gating.bases          # [P, D]
            scores = (z_stack * W[:, None, :]).sum(-1).mean(dim=1)    # [P]
            alpha_mean = F.softmax(scores, dim=0)                     # [P]
            fused = (alpha_mean[:, None, None] * z_stack).sum(dim=0)  # [N, D]

        else:
            # full / no_residual: 节点级动态语义选择
            fused, alpha = self.encoder.mp_gating(z_stack, self.mp_coeffs)
            alpha_mean = alpha.mean(dim=0)  # [P]

        if self.ablation != "no_residual":
            fused = fused + h_base

        return fused, alpha_mean

    def _forward_full_fused(self) -> tuple[th.Tensor, th.Tensor, th.Tensor, float]:
        h_dict = self.encoder.base(self.g, {})
        h_base = h_dict[self.target_ntype]

        # no_mp 或者本地根本枚举不到 meta-path，都直接退回 base
        if self.ablation == "no_mp" or len(self.mp_views) == 0:
            alpha_mean = th.ones(1, device=self.device, dtype=h_base.dtype)
            ent_t = entropy_from_alpha_mean(alpha_mean)   # = 0
            ent = float(ent_t.detach().item())
            logits = self.decoder(h_base)
            return logits, alpha_mean, ent_t, ent

        fused, alpha_mean = self._forward_mp_branch(h_base)

        ent_t = entropy_from_alpha_mean(alpha_mean)
        ent = float(ent_t.detach().item())

        logits = self.decoder(fused)
        return logits, alpha_mean, ent_t, ent

    def local_update(self) -> float:
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.ablation != "no_mp":
            params += [self.mp_coeffs]

        if self.optim == "Adam":
            optimizer = th.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            optimizer = th.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")

        self.encoder.train()
        self.decoder.train()

        train_nids = self.train_nid_dict[self.target_ntype]
        y = self.g.ndata["y"][self.target_ntype][train_nids].to(self.device)
        avg_epoch_loss = 0.0

        with tqdm.tqdm(range(self.num_local_epochs), desc=f"Client {self.id} (FedSP-MPG)") as tq:
            for _ in tq:
                logits, _alpha_mean, ent_t, ent = self._forward_full_fused()
                self.last_entropy = ent

                logp = F.log_softmax(logits[train_nids], dim=-1)
                task_loss = F.nll_loss(logp, y)

                align_reg_term = (
                    self.compute_alignment_regularization() * self.align_reg
                    if _uses_private_base_coeffs(self.ablation) else 0.0
                )

                global_ent_t = th.tensor(self.global_entropy, device=self.device, dtype=ent_t.dtype)
                if float(self.args.lambda_stb) > 0:
                    stb_term = (ent_t - global_ent_t).pow(2) * float(self.args.lambda_stb)
                else:
                    stb_term = th.tensor(0.0, device=self.device)

                loss = task_loss + align_reg_term + stb_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_epoch_loss += float(task_loss.item())

                tq.set_postfix(
                    {"epoch-loss": f"{float(loss.item()):.4f}", "H": f"{self.last_entropy:.3f}"},
                    refresh=False,
                )

        return avg_epoch_loss / max(self.num_local_epochs, 1)

    @th.no_grad()
    def local_evaluate(self, is_test: bool = False) -> dict[str, float]:
        self.encoder.eval()
        self.decoder.eval()

        nid_dict = self.test_nid_dict if is_test else self.val_nid_dict
        nids = nid_dict[self.target_ntype]
        y = self.g.ndata["y"][self.target_ntype][nids].to(self.device)

        logits, _alpha_mean, _ent_t, ent = self._forward_full_fused()
        pred = th.argmax(logits[nids], dim=-1)

        acc = float((pred == y).float().mean().item())
        macro, micro = _macro_micro_f1(y.cpu(), pred.cpu(), num_classes=self.out_dim)

        return {
            "accuracy": acc,
            "macro-f1": macro,
            "micro-f1": micro,
            "entropy": float(ent),
        }


class Server:
    def __init__(
        self,
        args: Namespace,
        ntypes: list[str],
        etypes: list[str],
        canonical_etypes: list[tuple[str, str, str]],
        out_dim: Optional[int] = None,
    ) -> None:
        self.num_clients = args.num_clients
        self.ablation = args.ablation

        _validate_ablation(self.ablation)

        # 保持 FedHGN 底座私有 basis coeffs 的逻辑不动
        if _uses_private_base_coeffs(self.ablation):
            dummy_base = HGNModel(
                args,
                args.hidden_dim,
                ["ntype"],
                ["etype"],
                [("ntype", "etype", "ntype")],
                {"ntype": 1},
            )
        else:
            dummy_base = HGNModel(
                args,
                args.hidden_dim,
                ntypes,
                etypes,
                canonical_etypes,
                {ntype: 1 for ntype in ntypes},
            )

        dummy = th.nn.Module()
        dummy.base = dummy_base
        dummy.mp_view_encoder = MPViewEncoder(
            args.hidden_dim,
            num_layers=args.mp_view_num_layers,
            dropout=args.mp_view_dropout,
        )
        dummy.mp_gating = DecoupledNodeWiseMPGating(
            args.hidden_dim,
            num_bases=args.mp_num_gating_bases,
        )

        dummy.to(args.device)
        state_dict_encoder = dummy.state_dict()

        keys_to_remove = [k for k in state_dict_encoder.keys() if k.startswith("base.embed_layer")]

        if _uses_private_base_coeffs(self.ablation):
            keys_to_remove += [
                k for k in list(state_dict_encoder.keys())
                if k.startswith("base.basis_coeffs_encoder")
            ]
        elif self.ablation == "C":
            keys_to_remove += [k for k in list(state_dict_encoder.keys()) if "bases" in k]

        for k in set(keys_to_remove):
            if k in state_dict_encoder:
                del state_dict_encoder[k]
        self.state_dict_encoder = state_dict_encoder

        assert isinstance(out_dim, int)
        dummy_decoder = NodeClassifier(args.hidden_dim, out_dim).to(args.device)
        self.state_dict_decoder = dummy_decoder.state_dict()

        if _uses_private_base_coeffs(self.ablation):
            self.all_clients_basis_coeffs_encoder = [
                th.zeros((args.num_layers, 1, args.num_bases), device=args.device)
                for _ in range(self.num_clients)
            ]
            self.all_clients_basis_coeffs_decoder = None

        self.global_entropy = 0.0

    def send_model(self, client: Client) -> None:
        client.encoder.load_state_dict(self.state_dict_encoder, strict=False)
        client.decoder.load_state_dict(self.state_dict_decoder, strict=False)
        client.set_global_entropy(self.global_entropy)

        if _uses_private_base_coeffs(self.ablation):
            others_basis_coeffs_encoder = th.cat(
                [self.all_clients_basis_coeffs_encoder[i] for i in range(self.num_clients) if i != client.id],
                dim=1,
            )
            client.set_others_basis_coeffs(others_basis_coeffs_encoder, None)

    def aggregate_model(self, clients: list[Client], client_weights: Optional[list[float]] = None) -> None:
        enc_list = [c.encoder.state_dict() for c in clients]
        dec_list = [c.decoder.state_dict() for c in clients]
        client_weights = (
            [1.0 / len(clients) for _ in range(len(clients))]
            if client_weights is None else client_weights
        )

        for key in list(self.state_dict_encoder.keys()):
            total_weight = 0.0
            agg = 0.0
            for sd, w in zip(enc_list, client_weights):
                if key in sd:
                    agg = agg + sd[key] * w
                    total_weight += w
            if total_weight > 0:
                self.state_dict_encoder[key] = agg / total_weight

        for key in list(self.state_dict_decoder.keys()):
            total_weight = 0.0
            agg = 0.0
            for sd, w in zip(dec_list, client_weights):
                if key in sd:
                    agg = agg + sd[key] * w
                    total_weight += w
            if total_weight > 0:
                self.state_dict_decoder[key] = agg / total_weight

        self.global_entropy = float(sum(w * c.last_entropy for c, w in zip(clients, client_weights)))

        if _uses_private_base_coeffs(self.ablation):
            for c in clients:
                self.all_clients_basis_coeffs_encoder[c.id] = th.stack(
                    [th.stack([param.detach() for param in pd.values()]) for pd in c.encoder.base.basis_coeffs_encoder]
                )


class FedSPMPG:
    def __init__(self, args: Namespace) -> None:
        self.max_rounds = args.max_rounds
        self.num_clients = args.num_clients
        self.fraction = args.fraction
        self.task = args.task
        self.val_interval = args.val_interval
        self.patience = args.patience
        self.save_path = args.save_path
        self.ablation = args.ablation

        _validate_ablation(self.ablation)

        if self.task != "node_classification":
            raise ValueError(f"FedSP-MPG supports node_classification only, got {self.task}")

        g_list, out_dim, train_list, val_list, test_list = load_data(args)
        ntypes, etypes, canonical_etypes = align_schemas(g_list)

        self.clients = [
            Client(args, (g_list[i], out_dim, train_list[i], val_list[i], test_list[i]), i)
            for i in range(self.num_clients)
        ]
        self.server = Server(args, ntypes, etypes, canonical_etypes, out_dim)

        self.train_client_weights = [sum(len(v) for v in d.values()) for d in train_list]
        tot = sum(self.train_client_weights)
        self.train_client_weights = [w / tot for w in self.train_client_weights]

        self.val_client_weights = [sum(len(v) for v in d.values()) for d in val_list]
        tot = sum(self.val_client_weights)
        self.val_client_weights = [w / tot for w in self.val_client_weights]

        self.test_client_weights = [sum(len(v) for v in d.values()) for d in test_list]
        tot = sum(self.test_client_weights)
        self.test_client_weights = [w / tot for w in self.test_client_weights]

    def train(self) -> None:
        sample_size = max(round(self.fraction * self.num_clients), 1)
        early_stopping = EarlyStopping(
            patience=self.patience,
            mode="score",
            save_path=self.save_path,
            verbose=True,
        )

        with tqdm.tqdm(range(self.max_rounds), desc="FedSP-MPG") as tq:
            for r in tq:
                selected = random.sample(self.clients, sample_size)

                for c in selected:
                    self.server.send_model(c)

                round_loss = 0.0
                for c in selected:
                    round_loss += c.local_update()
                round_loss /= sample_size

                weights = [self.train_client_weights[c.id] for c in selected]
                s = sum(weights)
                weights = [w / s for w in weights]
                self.server.aggregate_model(selected, weights)

                tq.set_postfix(
                    {"round-loss": f"{round_loss:.4f}", "H*": f"{self.server.global_entropy:.3f}"},
                    refresh=False,
                )

                if (r + 1) % self.val_interval == 0:
                    val_res = self.evaluate(is_test=False)
                    tq.set_postfix(
                        {k: f"{v:.4f}" for k, v in val_res.items() if k != "entropy"}
                        | {"entropy": f"{val_res.get('entropy', 0.0):.3f}"},
                        refresh=False,
                    )
                    early_stopping(val_res["accuracy"], callback=self.save_checkpoint)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

    def evaluate(self, is_test: bool = False) -> dict[str, float]:
        for c in self.clients:
            self.server.send_model(c)

        weights = self.test_client_weights if is_test else self.val_client_weights
        avg = defaultdict(float)

        for c, w in zip(self.clients, weights):
            res = c.local_evaluate(is_test=is_test)
            for k, v in res.items():
                avg[k] += v * w

        return dict(avg)

    def save_checkpoint(self, save_path: str) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # save server shared states
        th.save(self.server.state_dict_encoder, save_path / "server_encoder.pt")
        th.save(self.server.state_dict_decoder, save_path / "server_decoder.pt")
        th.save({"entropy": self.server.global_entropy}, save_path / "server_misc.pt")

        if _uses_private_base_coeffs(self.ablation):
            th.save(
                self.server.all_clients_basis_coeffs_encoder,
                save_path / "all_clients_basis_coeffs_encoder.pt",
            )

        # save each client's full state + private mp coeffs
        for i, client in enumerate(self.clients):
            th.save(client.encoder.state_dict(), save_path / f"client_{i}_encoder.pt")
            th.save(client.decoder.state_dict(), save_path / f"client_{i}_decoder.pt")
            th.save(
                {
                    "mp_coeffs": client.mp_coeffs.detach().cpu(),
                    "global_entropy": float(client.global_entropy),
                    "last_entropy": float(client.last_entropy),
                },
                save_path / f"client_{i}_private.pt",
            )

    def load_checkpoint(self, save_path: str) -> None:
        save_path = Path(save_path)

        # load server shared states
        self.server.state_dict_encoder = th.load(
            save_path / "server_encoder.pt", map_location="cpu"
        )
        self.server.state_dict_decoder = th.load(
            save_path / "server_decoder.pt", map_location="cpu"
        )

        misc_path = save_path / "server_misc.pt"
        if misc_path.exists():
            misc = th.load(misc_path, map_location="cpu")
            self.server.global_entropy = float(misc.get("entropy", 0.0))
        else:
            self.server.global_entropy = 0.0

        if _uses_private_base_coeffs(self.ablation):
            coeff_path = save_path / "all_clients_basis_coeffs_encoder.pt"
            if coeff_path.exists():
                loaded = th.load(coeff_path, map_location="cpu")
                self.server.all_clients_basis_coeffs_encoder = [
                    t.to(self.clients[0].device) for t in loaded
                ]

        # load each client's full state + private mp coeffs
        for i, client in enumerate(self.clients):
            client.encoder.load_state_dict(
                th.load(save_path / f"client_{i}_encoder.pt", map_location=client.device),
                strict=False,
            )
            client.decoder.load_state_dict(
                th.load(save_path / f"client_{i}_decoder.pt", map_location=client.device),
                strict=False,
            )

            private_state = th.load(
                save_path / f"client_{i}_private.pt", map_location="cpu"
            )
            client.mp_coeffs = th.nn.Parameter(
                private_state["mp_coeffs"].to(client.device)
            )
            client.global_entropy = float(
                private_state.get("global_entropy", self.server.global_entropy)
            )
            client.last_entropy = float(private_state.get("last_entropy", 0.0))