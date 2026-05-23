import random
from random import choices
import numpy as np
import pandas as pd

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.transforms import OneHotDegree

from models import GIN, serverGIN
from server import Server
from clients import Client_GC, DFedGNNClient_GC, CorrectedDaFedGNNClient_GC, CRYPTO_AVAILABLE
from server import get_maxDegree, get_stats, split_data, get_numGraphLabels

# ✨ 新增：用于特征维度统一
try:
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Warning: sklearn not available, feature alignment disabled")


def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i * minSize:(i + 1) * minSize])
        for g in graphs[num_client * minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False):
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = [x for x in tudataset]
    print("  **", data, len(graphs))

    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, df


def align_features(graphs, target_dim=64, dataset_name="Unknown"):
    """
    将图数据集的节点特征统一到目标维度

    Args:
        graphs: 图数据列表
        target_dim: 目标特征维度（默认64）
        dataset_name: 数据集名称（用于日志）

    Returns:
        aligned_graphs: 特征维度统一后的图数据列表
    """
    if len(graphs) == 0:
        return graphs

    current_dim = graphs[0].x.shape[1]

    if current_dim == target_dim:
        # 已经是目标维度，无需处理
        return graphs

    aligned_graphs = []

    if current_dim > target_dim:
        # 降维：使用PCA
        if not SKLEARN_AVAILABLE:
            print(f"  ⚠️  {dataset_name}: sklearn not available, skipping PCA (keeping {current_dim}D)")
            return graphs

        print(f"  📉 {dataset_name}: Reducing {current_dim}D → {target_dim}D (PCA)")

        # 收集所有节点特征
        all_features = []
        node_counts = []
        for graph in graphs:
            all_features.append(graph.x.cpu().numpy())
            node_counts.append(graph.x.shape[0])
        all_features = np.vstack(all_features)

        # 训练PCA
        pca = PCA(n_components=target_dim, random_state=42)
        pca.fit(all_features)

        # 转换每个图的特征
        start_idx = 0
        for i, graph in enumerate(graphs):
            num_nodes = node_counts[i]
            original_features = graph.x.cpu().numpy()
            reduced_features = pca.transform(original_features)

            # 创建新的图数据
            new_graph = Data(
                x=torch.FloatTensor(reduced_features),
                edge_index=graph.edge_index.clone(),
                y=graph.y.clone()
            )

            # 保留其他属性
            for key in graph.keys():
                if key not in ['x', 'edge_index', 'y']:
                    setattr(new_graph, key, getattr(graph, key))

            aligned_graphs.append(new_graph)

        explained_var = pca.explained_variance_ratio_.sum() * 100
        print(f"     ✓ Preserved {explained_var:.1f}% variance")

    else:
        # 升维：零填充
        print(f"  📈 {dataset_name}: Padding {current_dim}D → {target_dim}D (Zero-padding)")

        for graph in graphs:
            num_nodes = graph.x.shape[0]
            padding = torch.zeros(num_nodes, target_dim - current_dim)
            padded_features = torch.cat([graph.x, padding], dim=1)

            # 创建新的图数据
            new_graph = Data(
                x=padded_features,
                edge_index=graph.edge_index.clone(),
                y=graph.y.clone()
            )

            # 保留其他属性
            for key in graph.keys():
                if key not in ['x', 'edge_index', 'y']:
                    setattr(new_graph, key, getattr(graph, key))

            aligned_graphs.append(new_graph)

    return aligned_graphs


def prepareData_multiDS(datapath, group='small', batchSize=32, convert_x=False, seed=None, target_dim=None):
    """
    准备MultiDS模式的数据

    Args:
        datapath: 数据路径
        group: 数据集组名 ('biochem', 'molecules', 'small', 'mix', etc.)
        batchSize: 批次大小
        convert_x: 是否转换特征为one-hot度数
        seed: 随机种子
        target_dim: ✨ 新增！目标特征维度，如果指定则统一所有数据集的特征维度
                    - None: 保持原始特征维度（默认）
                    - int (如64): 使用PCA降维/零填充统一到该维度

    Returns:
        splitedData: 分割后的数据字典
        df: 数据统计DataFrame

    示例:
        # 保持原始维度（当前行为）
        splitedData, df = prepareData_multiDS('data', 'biochem')

        # 统一到64维（推荐用于MultiDS联邦学习）
        splitedData, df = prepareData_multiDS('data', 'biochem', target_dim=64)
    """
    assert group in ['molecules', 'molecules_tiny', 'small', 'mix', "mix_tiny", "biochem", "biochem_tiny"]

    # ✨ 如果指定了target_dim，打印提示
    if target_dim is not None:
        print(f"\n{'=' * 60}")
        print(f"🎯 特征维度统一模式：目标维度 = {target_dim}")
        print(f"{'=' * 60}\n")

    if group == 'molecules' or group == 'molecules_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if group == 'small':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics
    if group == 'mix' or group == 'mix_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",  # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]  # social networks
    if group == 'biochem' or group == 'biochem_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)
            if convert_x:
                maxdegree = get_maxDegree(tudataset)
                tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))

        # ✨ 新增：特征维度统一（如果指定了target_dim）
        if target_dim is not None:
            graphs = align_features(graphs, target_dim=target_dim, dataset_name=data)

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        if group.endswith('tiny'):
            graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
            graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        # ✨ 修改：特征维度使用对齐后的值
        num_node_features = graphs[0].x.shape[1] if target_dim is None else target_dim
        num_graph_labels = get_numGraphLabels(graphs_train)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    return splitedData, df


def setup_devices(splitedData, args):
    """创建原版客户端设备"""
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))

    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients


def setup_dfedgnn_devices(splitedData, args):
    """创建 D-FedGNN 客户端设备（无中心化聚合服务器参与训练）。"""
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, cmodel_gc.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        clients.append(DFedGNNClient_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))

    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients


def setup_corrected_dafedgnn_devices(splitedData, args):
    """创建修正的DaFedGNN客户端设备"""

    print("Setting up Corrected DaFedGNN clients...")
    print("  Features: Trainable alpha, Global/Personal parameter separation")

    idx_clients = {}
    clients = []

    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]

        # 创建基础模型（用于兼容性）
        cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, cmodel_gc.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

        # 创建修正的DaFedGNN客户端
        client = CorrectedDaFedGNNClient_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args)
        clients.append(client)

        print(f"  Client {idx} ({ds}) created with corrected DaFedGNN")

    # 创建服务器
    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)

    print(f"Created {len(clients)} Corrected DaFedGNN clients")
    return clients, server, idx_clients


# 为了向后兼容，保留原有函数别名
def setup_improved_dafedgnn_devices(splitedData, args):
    """重定向到修正版本以保持向后兼容性"""
    return setup_corrected_dafedgnn_devices(splitedData, args)


def setup_dafedgnn_devices(splitedData, args):
    """重定向到修正版本以保持向后兼容性"""
    return setup_corrected_dafedgnn_devices(splitedData, args)


# 以下为原有的其他设备设置函数，保持兼容性

class ImprovedDaFedGNNClient_GC(CorrectedDaFedGNNClient_GC):
    """
    基于论文方法的改进DaFedGNN客户端
    """

    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        super().__init__(model, client_id, client_name, train_size, dataLoader, optimizer, args)

        # DaFedGNN 安全聚合参数
        self._secure_aggregator = None
        self._secure_keys_setup = False
        self.secure_enabled = CRYPTO_AVAILABLE
        self.round_number = 0

        # 存储客户端ID列表
        self._all_client_ids = [client_id]  # 初始化时只有自己，后续需要更新

        # 存储全局模型权重用于α更新
        self.global_model_weights = None

    def set_all_client_ids(self, all_client_ids):
        """设置所有客户端ID"""
        self._all_client_ids = all_client_ids

    def set_global_model_weights(self, global_weights):
        """设置全局模型权重用于α更新"""
        self.global_model_weights = {k: v.clone() for k, v in global_weights.items()}

    @property
    def secure_aggregator(self):
        """延迟初始化安全聚合器"""
        if self._secure_aggregator is None and self.secure_enabled:
            try:
                from clients import SecureAggregationECDH
                self._secure_aggregator = SecureAggregationECDH(self.id, self._all_client_ids)
                print(f"Client {self.id}: Improved DaFedGNN secure aggregation initialized")
            except Exception as e:
                print(f"Client {self.id}: Failed to initialize secure aggregation: {e}")
                self.secure_enabled = False
        return self._secure_aggregator

    def setup_secure_keys(self, all_clients):
        """设置安全密钥"""
        if not self.secure_enabled or self._secure_keys_setup:
            return self._secure_keys_setup

        try:
            if self.secure_aggregator is None:
                return False

            success_count = 0
            for other_client in all_clients:
                if (other_client.id != self.id and
                        hasattr(other_client, 'secure_aggregator') and
                        other_client.secure_aggregator is not None):
                    try:
                        peer_public_key = other_client.secure_aggregator.get_public_key_bytes()
                        self.secure_aggregator.set_peer_public_key(other_client.id, peer_public_key)
                        success_count += 1
                    except Exception as e:
                        print(f"Failed to exchange keys with client {other_client.id}: {e}")

            self._secure_keys_setup = success_count > 0
            if self._secure_keys_setup:
                print(f"Client {self.id}: Established secure keys with {success_count} peers")
            return self._secure_keys_setup

        except Exception as e:
            print(f"Client {self.id}: Failed to setup secure keys: {e}")
            return False

    def get_aggregatable_parameters(self):
        """获取可聚合的参数，排除alpha相关参数"""
        aggregatable = {}
        for name, param in self.W.items():
            # 排除alpha相关参数，这些参数应该保持本地化
            if 'alpha' not in name.lower():
                aggregatable[name] = param.data.clone()
        return aggregatable

    def get_masked_parameters(self):
        """获取掩码后的模型参数（用于安全聚合），排除alpha参数"""
        if not self.secure_enabled or self.secure_aggregator is None:
            return self.get_aggregatable_parameters()

        try:
            # 只对可聚合的参数进行掩码处理
            model_params = self.get_aggregatable_parameters()
            masked_params = self.secure_aggregator.mask_model_parameters(model_params, self.round_number)
            return masked_params
        except Exception as e:
            print(f"Client {self.id}: Failed to mask parameters, using original: {e}")
            return self.get_aggregatable_parameters()

    def get_shared_parameters(self):
        """获取当前应该共享的参数（基于动态决策）"""
        if hasattr(self, 'dafedgnn_model') and hasattr(self.dafedgnn_model, 'get_global_parameters'):
            return self.dafedgnn_model.get_global_parameters()
        else:
            # 回退到排除alpha的参数
            return {k: v.data.clone() for k, v in self.W.items() if 'alpha' not in k.lower()}

    def get_personal_parameters(self):
        """获取应该保持个性化的参数（基于动态决策）"""
        if hasattr(self, 'dafedgnn_model') and hasattr(self.dafedgnn_model, 'get_personal_parameters'):
            return self.dafedgnn_model.get_personal_parameters()
        else:
            return {}

    def update_sharing_decisions(self, all_clients_params):
        """更新协作决策（如果支持）"""
        if hasattr(self, 'dafedgnn_model') and hasattr(self.dafedgnn_model, 'update_sharing_decisions'):
            self.dafedgnn_model.update_sharing_decisions(all_clients_params)

    def update_shared_parameters(self, shared_weights):
        """更新共享参数"""
        if hasattr(self, 'dafedgnn_model'):
            self.dafedgnn_model.update_shared_parameters(shared_weights)
            # 同步到W字典
            self.W = {key: value for key, value in self.dafedgnn_model.named_parameters()}
        else:
            # 原有的更新方法
            for name, new_param in shared_weights.items():
                if name in self.W and 'alpha' not in name.lower():
                    try:
                        if self.W[name].shape == new_param.shape:
                            self.W[name].data = new_param.data.clone()
                    except Exception as e:
                        print(f"Warning: Failed to update parameter {name}: {e}")

    def __getstate__(self):
        """自定义序列化，排除不可序列化的对象"""
        state = self.__dict__.copy()
        # 移除不可序列化的密码学对象
        if '_secure_aggregator' in state:
            state['_secure_aggregator'] = None
        return state

    def __setstate__(self, state):
        """自定义反序列化"""
        self.__dict__.update(state)
        # 重新初始化安全聚合器（如果需要）
        self._secure_aggregator = None
        self._secure_keys_setup = False