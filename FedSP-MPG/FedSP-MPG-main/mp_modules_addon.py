# mp_modules_addon.py
from __future__ import annotations
from typing import Dict, List, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn

CanonicalEType = Tuple[str, str, str]  # (src_ntype, etype, dst_ntype)


def enum_metapaths(
    canonical_etypes: List[CanonicalEType],
    target_ntype: str,
    max_len: int = 3,
    max_paths: int = 16,
    seed: int = 0,
) -> List[List[CanonicalEType]]:
    """
    Enumerate metapaths as LISTS OF CANONICAL ETYPES, not etype strings.
    This avoids DGL ambiguity when the same etype name appears under multiple (src,dst) pairs.

    Note:
        We intentionally make the returned metapaths deterministic by sorting,
        instead of random-shuffling then truncating. This reduces result variance
        across datasets and seeds.
    """
    _ = seed  # keep signature stable

    # Build adjacency over node types, storing canonical etype tuples
    outgoing: Dict[str, List[Tuple[CanonicalEType, str]]] = {}
    for s, e, t in canonical_etypes:
        ce: CanonicalEType = (s, e, t)
        outgoing.setdefault(s, []).append((ce, t))

    paths: List[List[CanonicalEType]] = []

    def dfs(cur_ntype: str, cur_path: List[CanonicalEType], depth: int):
        if depth > max_len:
            return
        if depth > 0 and cur_ntype == target_ntype:
            paths.append(cur_path.copy())
        if depth == max_len:
            return
        for (ce, nxt) in outgoing.get(cur_ntype, []):
            cur_path.append(ce)
            dfs(nxt, cur_path, depth + 1)
            cur_path.pop()

    dfs(target_ntype, [], 0)

    # de-dup + deterministic select
    uniq: List[List[CanonicalEType]] = []
    seen = set()
    for p in paths:
        key = tuple(p)
        if key not in seen and len(p) > 0:
            seen.add(key)
            uniq.append(p)

    # stable ordering: shorter first, then lexicographic over canonical etypes
    uniq = sorted(
        uniq,
        key=lambda p: (
            len(p),
            tuple(f"{s}|{e}|{t}" for (s, e, t) in p),
        ),
    )
    return uniq[:max_paths]


def build_metapath_views(g: dgl.DGLHeteroGraph, metapaths):
    """
    metapaths: List[List[CanonicalEType]] where CanonicalEType=(src, etype, dst)
    """
    views = []
    canon_set = set(g.canonical_etypes)

    for mp in metapaths:
        # only keep valid canonical metapaths under this local schema
        if not all(ce in canon_set for ce in mp):
            continue

        vg = dgl.metapath_reachable_graph(g, mp)

        # if no edge is reachable, add self-loop as a safe fallback
        if vg.num_edges() == 0:
            vg = dgl.add_self_loop(vg)

        views.append(vg)

    return views


class MPViewEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        num_layers = max(int(num_layers), 1)
        self.layers = nn.ModuleList([
            dglnn.GraphConv(
                dim,
                dim,
                norm="right",
                weight=True,
                bias=True,
                allow_zero_in_degree=True,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, x: th.Tensor) -> th.Tensor:
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class DecoupledNodeWiseMPGating(nn.Module):
    def __init__(self, dim: int, num_bases: int = 8):
        super().__init__()
        self.dim = dim
        self.num_bases = num_bases
        self.bases = nn.Parameter(th.empty(num_bases, dim))
        nn.init.xavier_uniform_(self.bases)

    def init_local_coeffs(self, num_paths: int, device=None) -> nn.Parameter:
        coeffs = nn.Parameter(th.empty(num_paths, self.num_bases, device=device))
        nn.init.xavier_uniform_(coeffs)
        return coeffs

    def forward(self, z_stack: th.Tensor, coeffs: th.Tensor):
        """
        z_stack: [P, N, D]
        coeffs:  [P, B]
        """
        W = coeffs @ self.bases                      # [P, D]
        scores = (z_stack * W[:, None, :]).sum(-1)  # [P, N]
        scores = scores.transpose(0, 1)             # [N, P]
        alpha = F.softmax(scores, dim=1)            # [N, P]
        fused = (alpha.unsqueeze(-1) * z_stack.transpose(0, 1)).sum(dim=1)  # [N, D]
        return fused, alpha


def entropy_from_alpha_mean(alpha_mean: th.Tensor) -> th.Tensor:
    eps = 1e-12
    a = alpha_mean.clamp_min(eps)
    return -(a * a.log()).sum()