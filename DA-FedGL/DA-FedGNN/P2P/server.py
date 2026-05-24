#     server.py -
#   FedEgoServer

import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw


class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(
                torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]),
                          dim=0), total_size).clone()

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]


# ============================================================================
# FedEgo Server
# ============================================================================

class FedEgoServer(Server):
    """
    FedEgo

      : "FedEgo: Privacy-preserving Personalized Federated Graph Learning with Ego-graphs" (2022)


    1.   reduction  FedAvg
    2.
    3.   global personalization
    """

    def __init__(self, model, device):
        super().__init__(model, device)
        self.global_label_dist = {}  #

    def compute_global_label_dist(self, client_dists):
        """
                 P_g(y)

        Args:
            client_dists:

        Returns:

        """
        all_classes = set()
        for dist in client_dists:
            all_classes.update(dist.keys())

        #     client
        self.global_label_dist = {}
        for c in all_classes:
            self.global_label_dist[c] = np.mean([
                dist.get(c, 0.0) for dist in client_dists
            ])

        return self.global_label_dist

    def aggregate_reduction_params(self, client_params):
        """
        FedAvg  reduction

        Args:
            client_params:

        Returns:
                reduction
        """
        if not client_params:
            return None

        #
        avg_params = {}
        num_clients = len(client_params)

        for key in client_params[0].keys():
            if 'reduction' in key:  # reduction
                avg_params[key] = torch.stack([
                    params[key] for params in client_params
                ]).mean(dim=0)

        return avg_params

    def get_global_personalization_params(self):
        """
            personalization
            client

        Returns:
              personalization
        """
        global_person_params = []
        for name, param in self.model.named_parameters():
            if 'personalization' in name:
                global_person_params.append(param.data.clone())

        return global_person_params


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()


def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(
                torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0),
                total_size).clone()
            target[name].data += tmp


import torch
from torch_geometric.utils import to_networkx, degree
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs


def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree


def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs


def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle,
                                              random_state=seed)
    return graphs_tv, graphs_test


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes / numGraphs, numEdges / numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df