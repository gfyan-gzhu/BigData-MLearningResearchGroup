import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

try:
    import networkx as nx
    from gensim.models import Word2Vec
    _HAS_N2V_DEPS = True
except Exception:
    _HAS_N2V_DEPS = False


def _safe_np_load(file_path: str) -> Dict[str, np.ndarray]:
    obj = np.load(file_path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def load_star_npz(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _safe_np_load(file_path)
    if 'attmats' not in data or 'adjs' not in data:
        raise KeyError(f"Expected keys 'attmats' and 'adjs' in {file_path}, found {list(data.keys())}")

    attmats = np.asarray(data['attmats'])
    adjs = np.asarray(data['adjs'])
    labels = np.asarray(data['labels']) if 'labels' in data else np.zeros(attmats.shape[0], dtype=np.int64)

    if attmats.ndim != 3:
        raise ValueError(f'attmats should be [N, T, F], got shape={attmats.shape}')
    if adjs.ndim != 3:
        raise ValueError(f'adjs should be [T, N, N], got shape={adjs.shape}')

    n_node, n_time, _ = attmats.shape
    if adjs.shape[0] != n_time or adjs.shape[1] != n_node or adjs.shape[2] != n_node:
        raise ValueError(f'Shape mismatch: attmats={attmats.shape}, adjs={adjs.shape}')

    return attmats.astype(np.float32), adjs, labels


def _normalize_train_only(attmats: np.ndarray, train_end_t: int) -> np.ndarray:
    n_node, _, feat_dim = attmats.shape
    flat = attmats.reshape(-1, feat_dim)
    train_flat = attmats[:, :train_end_t, :].reshape(-1, feat_dim)

    scaler = StandardScaler()
    scaler.fit(train_flat)
    flat = scaler.transform(flat)
    flat = np.nan_to_num(flat, nan=0.0, posinf=10.0, neginf=-10.0)
    return flat.reshape(n_node, -1, feat_dim).astype(np.float32)


def _matrix_to_edge_index(adj: np.ndarray, undirected: bool = True) -> torch.Tensor:
    adj = np.asarray(adj)
    src, dst = np.nonzero(adj)
    if src.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    if undirected:
        rev = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index, _ = coalesce(edge_index, None, adj.shape[0], adj.shape[1])
    return edge_index


def _active_nodes_from_adj(adj: np.ndarray) -> np.ndarray:
    deg = np.asarray(adj).sum(axis=0) + np.asarray(adj).sum(axis=1)
    active = np.nonzero(deg > 0)[0]
    return active.astype(np.int64)


def _sample_attr_anomaly_nodes(candidates: np.ndarray, n_attr: int, rng: np.random.Generator) -> np.ndarray:
    if len(candidates) == 0 or n_attr <= 0:
        return np.array([], dtype=np.int64)
    n_attr = min(n_attr, len(candidates))
    return rng.choice(candidates, size=n_attr, replace=False).astype(np.int64)


def _sample_struct_anomaly_nodes(candidates: np.ndarray, n_struct: int, rng: np.random.Generator) -> np.ndarray:
    if len(candidates) == 0 or n_struct <= 0:
        return np.array([], dtype=np.int64)
    n_struct = min(n_struct, len(candidates))
    return rng.choice(candidates, size=n_struct, replace=False).astype(np.int64)


def _inject_attribute_anomalies_global(
    x: np.ndarray,
    attr_nodes_global: np.ndarray,
    active_nodes_global: np.ndarray,
    candidate_pool_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x_new = x.copy()
    if attr_nodes_global.size == 0:
        return x_new

    active_nodes_global = np.asarray(active_nodes_global, dtype=np.int64)
    for nid in attr_nodes_global:
        candidates = active_nodes_global[active_nodes_global != nid]
        if candidates.size == 0:
            continue
        if candidates.size > candidate_pool_size:
            sampled = rng.choice(candidates, size=candidate_pool_size, replace=False)
        else:
            sampled = candidates

        src_feat = x[nid]
        cand_feat = x[sampled]
        dists = ((cand_feat - src_feat[None, :]) ** 2).sum(axis=1)
        farthest = sampled[int(np.argmax(dists))]
        x_new[nid] = x[farthest]

    return x_new


def _inject_structural_anomalies_global(
    edge_index: torch.Tensor,
    num_nodes: int,
    struct_nodes_global: np.ndarray,
    struct_clique_size: int,
    undirected: bool,
) -> torch.Tensor:
    if struct_nodes_global.size == 0 or struct_clique_size < 2:
        return edge_index

    edge_set = set((int(u), int(v)) for u, v in edge_index.t().tolist())
    nodes = np.asarray(struct_nodes_global, dtype=np.int64).tolist()

    for i in range(0, len(nodes), struct_clique_size):
        group = nodes[i:i + struct_clique_size]
        if len(group) < 2:
            continue
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                u, v = int(group[a]), int(group[b])
                edge_set.add((u, v))
                if undirected:
                    edge_set.add((v, u))

    if len(edge_set) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    edges, _ = coalesce(edges, None, num_nodes, num_nodes)
    return edges


def _make_data_full(
    x: np.ndarray,
    edge_index: torch.Tensor,
    y: np.ndarray,
    active_mask: np.ndarray,
) -> Data:
    num_nodes = x.shape[0]
    data = Data(
        x=torch.from_numpy(x).float(),
        edge_index=edge_index.long(),
        y=torch.from_numpy(y).float(),
        node_id=torch.arange(num_nodes, dtype=torch.long),
    )
    data.active_mask = torch.from_numpy(active_mask.astype(np.bool_))
    data.eval_mask = torch.from_numpy(active_mask.astype(np.bool_))
    return data


def _random_walk(G, start: int, walk_length: int, rng: np.random.Generator) -> List[str]:
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        walk.append(int(rng.choice(nbrs)))
    return [str(v) for v in walk]


def _snapshot_node2vec_features(
    adjs: np.ndarray,
    dim: int,
    walk_length: int,
    walks_per_node: int,
    window: int,
    seed: int,
    undirected: bool,
) -> np.ndarray:
    if not _HAS_N2V_DEPS:
        raise ImportError('node2vec mode requires networkx and gensim')

    t_steps, n_node, _ = adjs.shape
    rng = np.random.default_rng(seed)
    out = np.zeros((n_node, t_steps, dim), dtype=np.float32)

    for t in range(t_steps):
        adj_t = adjs[t]
        src, dst = np.nonzero(adj_t)
        G = nx.Graph() if undirected else nx.DiGraph()
        G.add_nodes_from(range(n_node))
        G.add_edges_from(zip(src.tolist(), dst.tolist()))

        walks = []
        active_nodes = [n for n in G.nodes() if G.degree(n) > 0]
        if not active_nodes:
            continue
        for _ in range(walks_per_node):
            rng.shuffle(active_nodes)
            for n in active_nodes:
                walks.append(_random_walk(G, int(n), walk_length, rng))

        model = Word2Vec(
            sentences=walks,
            vector_size=dim,
            window=window,
            min_count=0,
            sg=1,
            workers=1,
            seed=seed,
        )
        for key in model.wv.index_to_key:
            out[int(key), t] = model.wv[key]
    return out


def build_stripe_like_star_dataset(
    file_path: str,
    device: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    feature_mode: str = 'raw',
    undirected: bool = True,
    anomaly_ratio: float = 0.05,
    struct_clique_size: int = 5,
    attr_candidate_pool_size: int = 20,
    random_seed: int = 72,
    n2v_dim: int = 64,
    n2v_walk_length: int = 20,
    n2v_walks_per_node: int = 10,
    n2v_window: int = 10,
) -> Tuple[List[Data], int, int, Dict[str, int]]:
    attmats, adjs, labels = load_star_npz(file_path)

    n_node, n_time, _ = attmats.shape
    train_size = max(1, int(n_time * train_ratio))
    val_size = max(1, int(n_time * val_ratio))
    if train_size + val_size >= n_time:
        val_size = max(1, n_time - train_size - 1)
    test_start = train_size + val_size
    if test_start >= n_time:
        raise ValueError('train_ratio + val_ratio leaves no test snapshots')

    if feature_mode == 'raw':
        features = _normalize_train_only(attmats, train_size)
        final_feat_dim = features.shape[2]
    elif feature_mode == 'node2vec':
        features = _snapshot_node2vec_features(
            adjs=adjs,
            dim=n2v_dim,
            walk_length=n2v_walk_length,
            walks_per_node=n2v_walks_per_node,
            window=n2v_window,
            seed=random_seed,
            undirected=undirected,
        )
        features = _normalize_train_only(features, train_size)
        final_feat_dim = features.shape[2]
    else:
        raise ValueError("feature_mode must be 'raw' or 'node2vec'")

    rng = np.random.default_rng(random_seed)
    data_list: List[Data] = []

    for t in range(n_time):
        feat_t = features[:, t, :].astype(np.float32)
        adj_t = np.asarray(adjs[t])
        active_global = _active_nodes_from_adj(adj_t)
        active_mask = np.zeros(n_node, dtype=np.bool_)
        active_mask[active_global] = True

        edge_index_t = _matrix_to_edge_index(adj_t, undirected=undirected)
        x_t = feat_t.copy()
        y_t = np.zeros(n_node, dtype=np.float32)

        if t >= test_start and active_global.size > 0:
            target_total = max(1, int(np.ceil(active_global.shape[0] * anomaly_ratio)))
            n_attr = target_total // 2
            n_struct = target_total - n_attr

            attr_nodes = _sample_attr_anomaly_nodes(active_global, n_attr, rng)
            remain = np.setdiff1d(active_global, attr_nodes)
            struct_nodes = _sample_struct_anomaly_nodes(remain, n_struct, rng)

            x_t = _inject_attribute_anomalies_global(
                x=x_t,
                attr_nodes_global=attr_nodes,
                active_nodes_global=active_global,
                candidate_pool_size=attr_candidate_pool_size,
                rng=rng,
            )
            edge_index_t = _inject_structural_anomalies_global(
                edge_index=edge_index_t,
                num_nodes=n_node,
                struct_nodes_global=struct_nodes,
                struct_clique_size=struct_clique_size,
                undirected=undirected,
            )

            anom_nodes = np.union1d(attr_nodes, struct_nodes)
            y_t[anom_nodes] = 1.0

        data = _make_data_full(x=x_t, edge_index=edge_index_t, y=y_t, active_mask=active_mask)
        data.t = torch.tensor([t], dtype=torch.long)
        data.split = 'train' if t < train_size else ('val' if t < test_start else 'test')
        data_list.append(data.to(device))

    meta = {
        'num_nodes_global': int(n_node),
        'num_snapshots': int(n_time),
        'train_size': int(train_size),
        'val_size': int(val_size),
        'test_size': int(n_time - test_start),
        'num_classes_original': int(labels.shape[1]) if labels.ndim == 2 else int(len(np.unique(labels))),
    }
    return data_list, train_size, final_feat_dim, meta


def save_dataset(data_list: List[Data], out_path: str, train_size: int, feat_dim: int, meta: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save({
        'data_list': [d.cpu() for d in data_list],
        'train_size': train_size,
        'feat_dim': feat_dim,
        'meta': meta,
    }, out_path)


def main():
    parser = argparse.ArgumentParser(description='Prepare dynamic anomaly dataset from DBLP npz')
    parser.add_argument('--file_path', type=str, default='data/DBLP3.npz')
    parser.add_argument('--out_path', type=str, default='prepared_star_stripe_like.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--feature_mode', type=str, default='raw', choices=['raw', 'node2vec'])
    parser.add_argument('--undirected', action='store_true')
    parser.add_argument('--anomaly_ratio', type=float, default=0.05)
    parser.add_argument('--struct_clique_size', type=int, default=5)
    parser.add_argument('--attr_candidate_pool_size', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=72)
    parser.add_argument('--n2v_dim', type=int, default=64)
    parser.add_argument('--n2v_walk_length', type=int, default=20)
    parser.add_argument('--n2v_walks_per_node', type=int, default=10)
    parser.add_argument('--n2v_window', type=int, default=10)
    args = parser.parse_args()

    data_list, train_size, feat_dim, meta = build_stripe_like_star_dataset(
        file_path=args.file_path,
        device=args.device,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_mode=args.feature_mode,
        undirected=args.undirected,
        anomaly_ratio=args.anomaly_ratio,
        struct_clique_size=args.struct_clique_size,
        attr_candidate_pool_size=args.attr_candidate_pool_size,
        random_seed=args.random_seed,
        n2v_dim=args.n2v_dim,
        n2v_walk_length=args.n2v_walk_length,
        n2v_walks_per_node=args.n2v_walks_per_node,
        n2v_window=args.n2v_window,
    )

    save_dataset(data_list, args.out_path, train_size, feat_dim, meta)
    print(f'saved to {args.out_path}')
    print(meta)


if __name__ == '__main__':
    main()
