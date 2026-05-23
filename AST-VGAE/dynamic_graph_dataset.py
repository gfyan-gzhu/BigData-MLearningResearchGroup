import os
import copy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


# =========================
# Basic IO utilities
# =========================
def _read_jodie_style_csv(file_path):
    df_header = pd.read_csv(file_path, nrows=0)
    total_cols = df_header.shape[1]
    if total_cols < 4:
        df = pd.read_csv(file_path, sep=None, engine='python', skiprows=1, header=None)
    else:
        df = pd.read_csv(file_path, skiprows=1, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"文件列数异常，至少应有4列，当前只有 {df.shape[1]} 列: {file_path}")
    base_cols = ['user_id', 'item_id', 'timestamp', 'state_label']
    feat_dim = df.shape[1] - 4
    feat_cols = [f'feat_{i}' for i in range(feat_dim)]
    df.columns = base_cols + feat_cols
    return df, feat_cols


def _build_node_mapping(df):
    nodes = pd.unique(df[['user_id', 'item_id']].values.ravel())
    node_map = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    df['src'] = df['user_id'].map(node_map)
    df['dst'] = df['item_id'].map(node_map)
    return df, node_map, num_nodes


def _normalize_features_train_only(df, feat_cols, train_end_idx):
    if len(feat_cols) == 0:
        return df
    feats = df[feat_cols].values.astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    train_end_idx = int(max(train_end_idx, 1))
    train_end_idx = min(train_end_idx, len(df))
    scaler = StandardScaler()
    scaler.fit(feats[:train_end_idx])
    feats = scaler.transform(feats)
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    df[feat_cols] = feats
    return df


def _make_undirected_edge_index(edge_index):
    if edge_index.numel() == 0:
        return edge_index
    rev_edge_index = edge_index[[1, 0], :]
    edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


# =========================
# Feature aggregation
# =========================
def _aggregate_node_features_mean(active_src, active_dst, active_feats, num_nodes, feat_dim):
    snap_x = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    snap_cnt = np.zeros((num_nodes, 1), dtype=np.float32)
    for idx in range(len(active_src)):
        u = active_src[idx]
        v = active_dst[idx]
        f = active_feats[idx]
        snap_x[u] += f
        snap_x[v] += f
        snap_cnt[u] += 1.0
        snap_cnt[v] += 1.0
    snap_cnt[snap_cnt == 0] = 1.0
    return snap_x / snap_cnt


def _aggregate_node_features_mean_max(active_src, active_dst, active_feats, num_nodes, feat_dim):
    x_sum = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    x_cnt = np.zeros((num_nodes, 1), dtype=np.float32)
    x_max = np.full((num_nodes, feat_dim), -np.inf, dtype=np.float32)
    for idx in range(len(active_src)):
        u = active_src[idx]
        v = active_dst[idx]
        f = active_feats[idx]
        x_sum[u] += f
        x_sum[v] += f
        x_cnt[u] += 1.0
        x_cnt[v] += 1.0
        x_max[u] = np.maximum(x_max[u], f)
        x_max[v] = np.maximum(x_max[v], f)
    x_cnt[x_cnt == 0] = 1.0
    x_mean = x_sum / x_cnt
    x_max[np.isneginf(x_max)] = 0.0
    return np.concatenate([x_mean, x_max], axis=1)


def _build_onehot_features(num_nodes, onehot_dim):
    snap_x = np.zeros((num_nodes, onehot_dim), dtype=np.float32)
    for node_id in range(num_nodes):
        snap_x[node_id, node_id % onehot_dim] = 1.0
    return snap_x


# =========================
# Dataset loading
# =========================
def _load_jodie_csv_dataset(
    file_path,
    snap_size,
    device,
    train_ratio,
    use_onehot=False,
    onehot_dim=64,
    feature_agg='mean',
    undirected=False,
    sort_by_time=True,
):
    print(f"Reading CSV data from {file_path} ...")
    df, feat_cols = _read_jodie_style_csv(file_path)
    print(f"Successfully loaded {len(feat_cols)} edge features.")

    if sort_by_time:
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

    df, _, num_nodes = _build_node_mapping(df)

    num_interactions = len(df)
    num_snaps = int(np.ceil(num_interactions / snap_size))
    train_size = max(1, int(num_snaps * train_ratio))
    train_end_idx = min(train_size * snap_size, num_interactions)

    if not use_onehot and len(feat_cols) > 0:
        df = _normalize_features_train_only(df, feat_cols, train_end_idx)

    if use_onehot:
        actual_feat_dim = onehot_dim
    else:
        if feature_agg == 'mean':
            actual_feat_dim = len(feat_cols)
        elif feature_agg == 'mean_max':
            actual_feat_dim = 2 * len(feat_cols)
        else:
            raise ValueError("feature_agg must be 'mean' or 'mean_max'")

    print(f"num_nodes = {num_nodes}, num_interactions = {num_interactions}, num_snaps = {num_snaps}")
    print(f"feature mode = {'onehot' if use_onehot else feature_agg}, actual_feat_dim = {actual_feat_dim}")

    data_list = []
    for i in range(num_snaps):
        start_idx = i * snap_size
        end_idx = min((i + 1) * snap_size, num_interactions)
        snap_df = df.iloc[start_idx:end_idx]
        active_src = snap_df['src'].values.astype(np.int64)
        active_dst = snap_df['dst'].values.astype(np.int64)

        if len(active_src) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.from_numpy(np.stack([active_src, active_dst], axis=0)).long()
        if undirected:
            edge_index = _make_undirected_edge_index(edge_index)

        # unsupervised setting: training does not use anomaly labels.
        y = np.zeros(num_nodes, dtype=np.float32)

        if use_onehot:
            x = _build_onehot_features(num_nodes, onehot_dim)
        else:
            if len(feat_cols) == 0:
                raise ValueError("当前数据没有边特征，不能使用原始特征模式。")
            active_feats = snap_df[feat_cols].values.astype(np.float32)
            if feature_agg == 'mean':
                x = _aggregate_node_features_mean(active_src, active_dst, active_feats, num_nodes, len(feat_cols))
            else:
                x = _aggregate_node_features_mean_max(active_src, active_dst, active_feats, num_nodes, len(feat_cols))

        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        data = Data(
            x=torch.from_numpy(x).float(),
            edge_index=edge_index,
            y=torch.from_numpy(y).float(),
            node_index=torch.arange(num_nodes).long(),
        )
        data_list.append(data.to(device))

    return data_list, train_size, actual_feat_dim


def _load_star_npz_dataset(file_path, device, train_ratio, undirected=False):
    print(f"Reading STAR npz data from {file_path} ...")
    file = np.load(file_path, allow_pickle=True)
    if 'attmats' not in file or 'adjs' not in file:
        raise ValueError("文件缺少 attmats/adjs 键")

    attmats = np.asarray(file['attmats'])
    adjs = np.asarray(file['adjs'])
    labels = file['labels'] if 'labels' in file else None

    if attmats.ndim != 3:
        raise ValueError(f"attmats 维度应为 [N, T, F]，当前为 {attmats.shape}")
    if adjs.ndim != 3:
        raise ValueError(f"adjs 维度应为 [T, N, N]，当前为 {adjs.shape}")

    num_nodes, num_snaps, feat_dim = attmats.shape
    if adjs.shape[0] != num_snaps or adjs.shape[1] != num_nodes or adjs.shape[2] != num_nodes:
        raise ValueError(f"attmats 与 adjs 形状不匹配: attmats={attmats.shape}, adjs={adjs.shape}")

    train_size = max(1, int(num_snaps * train_ratio))
    data_list = []

    print(f"num_nodes = {num_nodes}, num_snaps = {num_snaps}, feat_dim = {feat_dim}")

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels.reshape(-1)
        if labels.shape[0] != num_nodes:
            labels = None

    for t in range(num_snaps):
        x_t = attmats[:, t, :].astype(np.float32)
        x_t = np.nan_to_num(x_t, nan=0.0, posinf=10.0, neginf=-10.0)

        adj_t = adjs[t]
        if hasattr(adj_t, 'toarray'):
            adj_t = adj_t.toarray()
        adj_t = np.asarray(adj_t)
        src, dst = np.nonzero(adj_t)
        if len(src) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        if undirected:
            edge_index = _make_undirected_edge_index(edge_index)

        y_t = np.zeros(num_nodes, dtype=np.float32)
        data = Data(
            x=torch.from_numpy(x_t).float(),
            edge_index=edge_index,
            y=torch.from_numpy(y_t).float(),
            node_index=torch.arange(num_nodes).long(),
        )
        if labels is not None:
            data.orig_y = torch.from_numpy(labels.copy())
        data_list.append(data.to(device))

    return data_list, train_size, feat_dim


# =========================
# anomaly injection on TEST only
# =========================
def _nodes_in_snapshot(data: Data) -> np.ndarray:
    num_nodes = data.x.size(0)
    if data.edge_index.numel() == 0:
        return np.arange(num_nodes, dtype=np.int64)
    nodes = torch.unique(data.edge_index).detach().cpu().numpy().astype(np.int64)
    if nodes.size == 0:
        return np.arange(num_nodes, dtype=np.int64)
    return nodes


def _replace_feature_with_farthest(x_np, node_idx, candidate_nodes, k, rng):
    if candidate_nodes.size <= 1:
        return
    candidates = candidate_nodes[candidate_nodes != node_idx]
    if candidates.size == 0:
        return
    if candidates.size > k:
        sampled = rng.choice(candidates, size=k, replace=False)
    else:
        sampled = candidates
    base = x_np[node_idx]
    cand_feats = x_np[sampled]
    dists = np.sum((cand_feats - base[None, :]) ** 2, axis=1)
    farthest = sampled[int(np.argmax(dists))]
    x_np[node_idx] = x_np[farthest]


def _sample_disjoint_nodes(candidates, n_select, used, rng):
    pool = np.setdiff1d(candidates, np.asarray(list(used), dtype=np.int64), assume_unique=False)
    if pool.size == 0 or n_select <= 0:
        return np.array([], dtype=np.int64)
    n_select = min(n_select, pool.size)
    selected = rng.choice(pool, size=n_select, replace=False)
    return np.asarray(selected, dtype=np.int64)


def inject_test_anomalies(
    data_list,
    test_start,
    anomaly_ratio=0.05,
    struct_clique_size=5,
    attr_candidate_k=50,
    random_seed=72,
):
    """
    - inject anomalies only on test snapshots
    - equal numbers of structural and attribute anomalies for each snapshot
    - labels y=1 only for injected nodes in test snapshots
    """
    rng = np.random.default_rng(random_seed)
    out = []

    for idx, data in enumerate(data_list):
        d = copy.copy(data)
        d.x = data.x.clone()
        d.edge_index = data.edge_index.clone()
        d.y = torch.zeros_like(data.y)
        if hasattr(data, 'orig_y'):
            d.orig_y = data.orig_y.clone()

        if idx < test_start:
            out.append(d)
            continue

        num_nodes = d.x.size(0)
        snapshot_nodes = _nodes_in_snapshot(d)
        if snapshot_nodes.size == 0:
            out.append(d)
            continue

        # same number of structural and attribute anomalies
        total_anom = max(2, int(round(snapshot_nodes.size * anomaly_ratio)))
        if total_anom % 2 == 1:
            total_anom += 1
        n_attr = total_anom // 2
        n_struct = total_anom // 2

        used_nodes = set()
        attr_nodes = _sample_disjoint_nodes(snapshot_nodes, n_attr, used_nodes, rng)
        used_nodes.update(attr_nodes.tolist())
        struct_nodes = _sample_disjoint_nodes(snapshot_nodes, n_struct, used_nodes, rng)
        used_nodes.update(struct_nodes.tolist())

        x_np = d.x.detach().cpu().numpy().copy()
        for nid in attr_nodes:
            _replace_feature_with_farthest(x_np, int(nid), snapshot_nodes, attr_candidate_k, rng)
        d.x = torch.from_numpy(x_np).float().to(d.x.device)

        # structural anomalies: connect selected nodes into q cliques of size Np
        if struct_nodes.size > 1:
            clique_size = max(2, int(struct_clique_size))
            groups = [struct_nodes[i:i + clique_size] for i in range(0, len(struct_nodes), clique_size)]
            existing = set((int(u), int(v)) for u, v in d.edge_index.t().detach().cpu().tolist())
            for group in groups:
                if len(group) < 2:
                    continue
                for i in range(len(group)):
                    for j in range(len(group)):
                        if i == j:
                            continue
                        existing.add((int(group[i]), int(group[j])))
            if existing:
                edge_arr = np.array(list(existing), dtype=np.int64)
                d.edge_index = torch.from_numpy(edge_arr.T).long().to(d.edge_index.device)

        anom_nodes = np.union1d(attr_nodes, struct_nodes)
        if anom_nodes.size > 0:
            y = d.y.detach().cpu().numpy()
            y[anom_nodes] = 1.0
            d.y = torch.from_numpy(y).float().to(data.y.device)

        out.append(d)

    return out


def generate_dataset(
    file_path,
    device,
    snap_size,
    train_ratio,
    use_onehot=False,
    onehot_dim=128,
    feature_agg='mean',
    undirected=False,
    sort_by_time=True,
):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npz':
        return _load_star_npz_dataset(
            file_path=file_path,
            device=device,
            train_ratio=train_ratio,
            undirected=undirected,
        )
    return _load_jodie_csv_dataset(
        file_path=file_path,
        snap_size=snap_size,
        device=device,
        train_ratio=train_ratio,
        use_onehot=use_onehot,
        onehot_dim=onehot_dim,
        feature_agg=feature_agg,
        undirected=undirected,
        sort_by_time=sort_by_time,
    )
