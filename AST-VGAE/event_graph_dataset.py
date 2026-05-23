import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from torch_geometric.nn import Node2Vec
from pygod.generator import gen_contextual_outlier, gen_structural_outlier


def _read_jodie_style_csv(file_path):
    """
    读取 /Wikipedia/Reddit/MOOC CSV
    前4列为:user_id, item_id, timestamp, state_label
    """
    df_header = pd.read_csv(file_path, nrows=0)
    total_cols = df_header.shape[1]

    if total_cols < 4:
        df = pd.read_csv(file_path, sep=None, engine='python', skiprows=1, header=None)
    else:
        df = pd.read_csv(file_path, skiprows=1, header=None)

    if df.shape[1] < 4:
        raise ValueError(f"文件列数异常，至少应有4列，当前只有 {df.shape[1]} 列: {file_path}")

    df = df.iloc[:, :4].copy()
    df.columns = ['user_id', 'item_id', 'timestamp', 'state_label']
    return df


def _build_node_mapping(df):
    """
    将 user_id / item_id 统一映射到一个节点空间
    """
    nodes = pd.unique(df[['user_id', 'item_id']].values.ravel())
    node_map = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    df['src'] = df['user_id'].map(node_map).astype(np.int64)
    df['dst'] = df['item_id'].map(node_map).astype(np.int64)

    return df, node_map, num_nodes


def _make_undirected_edge_index(edge_index):
    """
    将边转为无向：补反向边
    """
    rev_edge_index = edge_index[[1, 0], :]
    edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    return edge_index


def _dataset_name_from_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def _safe_float_str(x):
    return str(x).replace('.', 'p')


def _build_train_union_edge_index(df, snap_size, train_size, num_nodes, undirected=True):
    """
    只用训练阶段 snapshot 的边，合成一张训练并图
    """
    train_end_idx = min(train_size * snap_size, len(df))
    train_df = df.iloc[:train_end_idx]

    src = train_df['src'].values.astype(np.int64)
    dst = train_df['dst'].values.astype(np.int64)

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    if undirected:
        edge_index = _make_undirected_edge_index(edge_index)

    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def _train_node2vec(
    edge_index,
    num_nodes,
    device,
    embedding_dim=64,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    lr=0.01,
    epochs=50,
    batch_size=256
):
    """
    在训练期并图上训练一次 Node2Vec
    """
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        num_nodes=num_nodes,
        sparse=True
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[Node2Vec] Epoch {epoch:03d}, Loss = {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        emb = model().detach().cpu().numpy().astype(np.float32)

    return emb


def _get_n2v_cache_path(
    cache_root,
    dataset_name,
    snap_size,
    train_ratio,
    sort_by_time,
    seed,
    n2v_dim,
    n2v_walk_length,
    n2v_context_size,
    n2v_walks_per_node,
    n2v_negative_samples,
    n2v_p,
    n2v_q,
    n2v_lr,
    n2v_epochs,
    n2v_batch_size
):
    os.makedirs(os.path.join(cache_root, "node2vec_4"), exist_ok=True)
    fname = (
        f"{dataset_name}"
        f"_snap{snap_size}"
        f"_tr{_safe_float_str(train_ratio)}"
        f"_sort{int(sort_by_time)}"
        f"_seed{seed}"
        f"_dim{n2v_dim}"
        f"_wl{n2v_walk_length}"
        f"_ctx{n2v_context_size}"
        f"_wpn{n2v_walks_per_node}"
        f"_neg{n2v_negative_samples}"
        f"_p{_safe_float_str(n2v_p)}"
        f"_q{_safe_float_str(n2v_q)}"
        f"_lr{_safe_float_str(n2v_lr)}"
        f"_ep{n2v_epochs}"
        f"_bs{n2v_batch_size}.pt"
    )
    return os.path.join(cache_root, "node2vec_4", fname)


def _get_final_cache_path(
    cache_root,
    dataset_name,
    snap_size,
    train_ratio,
    sort_by_time,
    seed,
    n2v_dim,
    n2v_walk_length,
    n2v_context_size,
    n2v_walks_per_node,
    n2v_negative_samples,
    n2v_p,
    n2v_q,
    n2v_lr,
    n2v_epochs,
    n2v_batch_size,
    anomaly_ratio,
    context_k,
    struct_m,
    struct_p,
    inject_test_only=True
):
    os.makedirs(os.path.join(cache_root, "pygod_data_4"), exist_ok=True)
    fname = (
        f"{dataset_name}"
        f"_snap{snap_size}"
        f"_tr{_safe_float_str(train_ratio)}"
        f"_sort{int(sort_by_time)}"
        f"_seed{seed}"
        f"_dim{n2v_dim}"
        f"_wl{n2v_walk_length}"
        f"_ctx{n2v_context_size}"
        f"_wpn{n2v_walks_per_node}"
        f"_neg{n2v_negative_samples}"
        f"_p{_safe_float_str(n2v_p)}"
        f"_q{_safe_float_str(n2v_q)}"
        f"_lr{_safe_float_str(n2v_lr)}"
        f"_ep{n2v_epochs}"
        f"_bs{n2v_batch_size}"
        f"_ar{_safe_float_str(anomaly_ratio)}"
        f"_ck{context_k}"
        f"_sm{struct_m}"
        f"_sp{_safe_float_str(struct_p)}"
        f"_testonly{int(inject_test_only)}.pt"
    )
    return os.path.join(cache_root, "pygod_data_4", fname)


def _load_or_train_node2vec(
    train_union_edge_index,
    num_nodes,
    device,
    cache_root,
    dataset_name,
    snap_size,
    train_ratio,
    sort_by_time,
    seed,
    n2v_dim,
    n2v_walk_length,
    n2v_context_size,
    n2v_walks_per_node,
    n2v_negative_samples,
    n2v_p,
    n2v_q,
    n2v_lr,
    n2v_epochs,
    n2v_batch_size,
    use_cache=True
):
    cache_path = _get_n2v_cache_path(
        cache_root=cache_root,
        dataset_name=dataset_name,
        snap_size=snap_size,
        train_ratio=train_ratio,
        sort_by_time=sort_by_time,
        seed=seed,
        n2v_dim=n2v_dim,
        n2v_walk_length=n2v_walk_length,
        n2v_context_size=n2v_context_size,
        n2v_walks_per_node=n2v_walks_per_node,
        n2v_negative_samples=n2v_negative_samples,
        n2v_p=n2v_p,
        n2v_q=n2v_q,
        n2v_lr=n2v_lr,
        n2v_epochs=n2v_epochs,
        n2v_batch_size=n2v_batch_size
    )

    if use_cache and os.path.exists(cache_path):
        print(f"Load cached Node2Vec from: {cache_path}")
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        return cache["node2vec_feat"].astype(np.float32)

    print("No cached Node2Vec found. Start training Node2Vec ...")
    node2vec_feat = _train_node2vec(
        edge_index=train_union_edge_index,
        num_nodes=num_nodes,
        device=device,
        embedding_dim=n2v_dim,
        walk_length=n2v_walk_length,
        context_size=n2v_context_size,
        walks_per_node=n2v_walks_per_node,
        num_negative_samples=n2v_negative_samples,
        p=n2v_p,
        q=n2v_q,
        lr=n2v_lr,
        epochs=n2v_epochs,
        batch_size=n2v_batch_size
    )

    if use_cache:
        torch.save(
            {
                "node2vec_feat": node2vec_feat,
                "num_nodes": num_nodes
            },
            cache_path
        )
        print(f"Saved Node2Vec cache to: {cache_path}")

    return node2vec_feat


def _build_snapshot_data_with_pygod(
    edge_index,
    node2vec_feat,
    num_nodes,
    anomaly_ratio=0.05,
    context_k=50,
    struct_m=5,
    struct_p=0.0,
    seed=72
):
    """
    1. 节点特征直接用 Node2Vec
    2. 属性异常用 PyGOD 的 gen_contextual_outlier
    3. 结构异常用 PyGOD 的 gen_structural_outlier
    """
    base_data = Data(
        x=torch.from_numpy(node2vec_feat).float(),
        edge_index=edge_index.clone(),
        node_index=torch.arange(num_nodes).long()
    )

    context_n = max(1, int(num_nodes * anomaly_ratio / 2))
    struct_total_nodes = max(1, int(num_nodes * anomaly_ratio / 2))

    context_k = min(context_k, max(1, num_nodes - 1))

    struct_m = min(struct_m, num_nodes)
    if struct_total_nodes < struct_m:
        struct_m_eff = struct_total_nodes
        struct_n = 1
    else:
        struct_m_eff = struct_m
        struct_n = max(1, struct_total_nodes // struct_m_eff)

    data_ctx, y_ctx = gen_contextual_outlier(
        base_data,
        n=context_n,
        k=context_k,
        seed=seed
    )

    data_final, y_struct = gen_structural_outlier(
        data_ctx,
        m=struct_m_eff,
        n=struct_n,
        p=struct_p,
        directed=False,
        seed=seed + 1000
    )

    y_final = torch.logical_or(y_ctx.bool(), y_struct.bool()).float()
    data_final.y = y_final

    return data_final


def load_wikipedia_data(
    file_path,
    snap_size,
    device,
    train_ratio,
    undirected=True,
    sort_by_time=True,
    n2v_dim=64,
    n2v_walk_length=20,
    n2v_context_size=10,
    n2v_walks_per_node=10,
    n2v_negative_samples=1,
    n2v_p=1.0,
    n2v_q=1.0,
    n2v_lr=0.01,
    n2v_epochs=50,
    n2v_batch_size=256,
    anomaly_ratio=0.05,
    context_k=50,
    struct_m=5,
    struct_p=0.0,
    seed=72,
    cache_root="cache",
    use_cache=True,
    inject_test_only=True
):
    """
    动态图数据处理：
    1. 不再使用原始边特征
    2. 节点特征使用训练期并图上的 Node2Vec
    3. 属性异常 / 结构异常使用 PyGOD 注入
    """
    print(f"Reading data from {file_path} ...")

    df = _read_jodie_style_csv(file_path)

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

    if sort_by_time:
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

    df, node_map, num_nodes = _build_node_mapping(df)

    num_interactions = len(df)
    num_snaps = int(np.ceil(num_interactions / snap_size))
    train_size = max(1, int(num_snaps * train_ratio))

    print(f"num_nodes = {num_nodes}, num_interactions = {num_interactions}, num_snaps = {num_snaps}")
    print(f"train_size = {train_size}")

    dataset_name = _dataset_name_from_path(file_path)

    final_cache_path = _get_final_cache_path(
        cache_root=cache_root,
        dataset_name=dataset_name,
        snap_size=snap_size,
        train_ratio=train_ratio,
        sort_by_time=sort_by_time,
        seed=seed,
        n2v_dim=n2v_dim,
        n2v_walk_length=n2v_walk_length,
        n2v_context_size=n2v_context_size,
        n2v_walks_per_node=n2v_walks_per_node,
        n2v_negative_samples=n2v_negative_samples,
        n2v_p=n2v_p,
        n2v_q=n2v_q,
        n2v_lr=n2v_lr,
        n2v_epochs=n2v_epochs,
        n2v_batch_size=n2v_batch_size,
        anomaly_ratio=anomaly_ratio,
        context_k=context_k,
        struct_m=struct_m,
        struct_p=struct_p,
        inject_test_only=inject_test_only
    )

    if use_cache and os.path.exists(final_cache_path):
        print(f"Load cached final PyGOD data from: {final_cache_path}")
        cache = torch.load(final_cache_path, map_location='cpu', weights_only=False)
        data_list_cpu = cache["data_list"]
        feat_dim = cache["feat_dim"]
        data_list = [data.to(device) for data in data_list_cpu]
        return data_list, num_nodes, feat_dim, train_size

    train_union_edge_index = _build_train_union_edge_index(
        df=df,
        snap_size=snap_size,
        train_size=train_size,
        num_nodes=num_nodes,
        undirected=undirected
    )

    node2vec_feat = _load_or_train_node2vec(
        train_union_edge_index=train_union_edge_index,
        num_nodes=num_nodes,
        device=device,
        cache_root=cache_root,
        dataset_name=dataset_name,
        snap_size=snap_size,
        train_ratio=train_ratio,
        sort_by_time=sort_by_time,
        seed=seed,
        n2v_dim=n2v_dim,
        n2v_walk_length=n2v_walk_length,
        n2v_context_size=n2v_context_size,
        n2v_walks_per_node=n2v_walks_per_node,
        n2v_negative_samples=n2v_negative_samples,
        n2v_p=n2v_p,
        n2v_q=n2v_q,
        n2v_lr=n2v_lr,
        n2v_epochs=n2v_epochs,
        n2v_batch_size=n2v_batch_size,
        use_cache=use_cache
    )

    feat_dim = node2vec_feat.shape[1]
    print(f"feature mode = node2vec_5, actual_feat_dim = {feat_dim}")

    data_list_cpu = []

    for i in range(num_snaps):
        start_idx = i * snap_size
        end_idx = min((i + 1) * snap_size, num_interactions)
        snap_df = df.iloc[start_idx:end_idx]

        active_src = snap_df['src'].values.astype(np.int64)
        active_dst = snap_df['dst'].values.astype(np.int64)

        edge_index = torch.from_numpy(
            np.stack([active_src, active_dst], axis=0)
        ).long()

        if undirected:
            edge_index = _make_undirected_edge_index(edge_index)

        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

        if inject_test_only and i < train_size:
            data = Data(
                x=torch.from_numpy(node2vec_feat).float(),
                edge_index=edge_index.clone(),
                node_index=torch.arange(num_nodes).long(),
                y=torch.zeros(num_nodes).float()
            )
            tag = "train_no_injection"
        else:
            data = _build_snapshot_data_with_pygod(
                edge_index=edge_index,
                node2vec_feat=node2vec_feat,
                num_nodes=num_nodes,
                anomaly_ratio=anomaly_ratio,
                context_k=context_k,
                struct_m=struct_m,
                struct_p=struct_p,
                seed=seed + i
            )
            tag = "test_injected" if inject_test_only else "all_injected"

        data_list_cpu.append(data)

        print(
            f"snapshot {i}: "
            f"edges={edge_index.size(1)}, "
            f"anomaly_nodes={(data.y > 0).sum().item()}, "
            f"{tag}"
        )

    if use_cache:
        torch.save(
            {
                "data_list": data_list_cpu,
                "feat_dim": feat_dim
            },
            final_cache_path
        )
        print(f"Saved final PyGOD data cache to: {final_cache_path}")

    data_list = [data.to(device) for data in data_list_cpu]
    return data_list, num_nodes, feat_dim, train_size


def generate_dataset(
    file_path,
    device,
    snap_size,
    train_ratio,
    undirected=True,
    sort_by_time=True,
    n2v_dim=64,
    n2v_walk_length=20,
    n2v_context_size=10,
    n2v_walks_per_node=10,
    n2v_negative_samples=1,
    n2v_p=1.0,
    n2v_q=1.0,
    n2v_lr=0.01,
    n2v_epochs=50,
    n2v_batch_size=256,
    anomaly_ratio=0.05,
    context_k=50,
    struct_m=5,
    struct_p=0.0,
    seed=72,
    cache_root="cache",
    use_cache=True,
    inject_test_only=True
):
    data_list, num_nodes, feat_dim, train_size = load_wikipedia_data(
        file_path=file_path,
        snap_size=snap_size,
        device=device,
        train_ratio=train_ratio,
        undirected=undirected,
        sort_by_time=sort_by_time,
        n2v_dim=n2v_dim,
        n2v_walk_length=n2v_walk_length,
        n2v_context_size=n2v_context_size,
        n2v_walks_per_node=n2v_walks_per_node,
        n2v_negative_samples=n2v_negative_samples,
        n2v_p=n2v_p,
        n2v_q=n2v_q,
        n2v_lr=n2v_lr,
        n2v_epochs=n2v_epochs,
        n2v_batch_size=n2v_batch_size,
        anomaly_ratio=anomaly_ratio,
        context_k=context_k,
        struct_m=struct_m,
        struct_p=struct_p,
        seed=seed,
        cache_root=cache_root,
        use_cache=use_cache,
        inject_test_only=inject_test_only
    )
    return data_list, train_size, feat_dim