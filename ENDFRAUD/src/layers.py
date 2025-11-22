import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from operator import itemgetter
import math
import random
from .choose_neighs import choose_neighs_and_get_features
from .choose_neighs2 import choose_neighs_and_get_features2
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("使用的设备:", device)

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):

    def __init__(self, features, pe_features, feature_dim, embed_dim,
                 train_pos, adj_lists, intraggs, inter='GNN', cuda=False):
        """
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
        super(InterAgg, self).__init__()

        self.features = features
        self.pe_features = pe_features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.cuda = cuda
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda
        self.train_pos = train_pos
        self.dropout = nn.Dropout(p=0.5)
        # initial filtering thresholds
        self.thresholds = [0.5, 0.5, 0.5]

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim * len(intraggs) + self.feat_dim, self.embed_dim))
        # self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim * 6, self.embed_dim))
        init.xavier_uniform_(self.weight)

        # label predictor for similarity measure
        self.label_clf = nn.Linear(self.feat_dim, 2)

        # initialize the parameter logs
        self.weights_log = []
        self.thresholds_log = [self.thresholds]
        self.relation_score_log = []

    def forward(self, nodes, labels, train_flag=True):
        """
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
                                 set.union(*to_neighs[2], set(nodes)))

        # calculate label-aware scores
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
            batch_pe_features = self.pe_features(torch.cuda.LongTensor(list(unique_nodes)))
            pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))
            batch_pe_features = self.pe_features(torch.LongTensor(list(unique_nodes)))
            pos_features = self.features(torch.LongTensor(list(self.train_pos)))
        batch_scores = self.label_clf(batch_features)
        pos_scores = self.label_clf(pos_features)
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # get neighbor node id list for each batch node and relation
        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        # assign label-aware scores to neighbor nodes for each batch node and relation
        r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
        r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
        r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

        # count the number of neighbors kept for aggregation for each batch node and relation
        r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
        r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
        r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

        r1_features = [batch_features[itemgetter(*to_neigh)(id_mapping), :] for to_neigh in r1_list]
        r2_features = [batch_features[itemgetter(*to_neigh)(id_mapping), :] for to_neigh in r2_list]
        r3_features = [batch_features[itemgetter(*to_neigh)(id_mapping), :] for to_neigh in r3_list]

        r1_pe_features = [batch_pe_features[itemgetter(*to_neigh)(id_mapping), :] for to_neigh in r1_list]
        r2_pe_features = [batch_pe_features[itemgetter(*to_neigh)(id_mapping), :] for to_neigh in r2_list]
        r3_pe_features = [batch_pe_features[itemgetter(*to_neigh)(id_mapping), :] for to_neigh in r3_list]

        # intra-aggregation steps for each relation
        # Eq. (8) in the paper
        # r1_feats, r1_scores, r1_gen_feats, r1_raw_feats = self.intra_agg1.forward(nodes, labels, r1_list, center_scores,
        #                                                                           r1_scores, pos_scores,
        #                                                                           r1_sample_num_list, train_flag,
        #                                                                           r1_pe_features)
        # r2_feats, r2_scores, r2_gen_feats, r2_raw_feats = self.intra_agg2.forward(nodes, labels, r2_list, center_scores,
        #                                                                           r2_scores, pos_scores,
        #                                                                           r2_sample_num_list, train_flag,
        #                                                                           r2_pe_features)
        # r3_feats, r3_scores, r3_gen_feats, r3_raw_feats = self.intra_agg3.forward(nodes, labels, r3_list, center_scores,
        #                                                                           r3_scores, pos_scores,
        #                                                                           r3_sample_num_list, train_flag,
        #                                                                           r3_pe_features)
        r1_feats, r1_scores, r1_gen_feats, r1_gen2_feats, r1_raw_feats, r1_top = self.intra_agg1.forward(nodes, labels, r1_list, center_scores,
                                                                                  r1_scores, pos_scores,
                                                                                  r1_sample_num_list, train_flag)
        r2_feats, r2_scores, r2_gen_feats, r2_gen2_feats, r2_raw_feats, r2_top = self.intra_agg2.forward(nodes, labels, r2_list, center_scores,
                                                                                  r2_scores, pos_scores,
                                                                                  r2_sample_num_list, train_flag)
        r3_feats, r3_scores, r3_gen_feats, r3_gen2_feats, r3_raw_feats, r3_top = self.intra_agg3.forward(nodes, labels, r3_list, center_scores,
                                                                                  r3_scores, pos_scores,
                                                                                  r3_sample_num_list, train_flag)

        # gen_node_feats = torch.cat((r1_gen_feats, r2_gen_feats, r3_gen_feats), dim=1)
        # gen_node_feats = F.relu(gen_node_feats.mm(self.weight).t())
        gen_feats = []
        gen_feats.append(r1_gen_feats)
        gen_feats.append(r2_gen_feats)
        gen_feats.append(r3_gen_feats)

        gen2_feats = []
        gen2_feats.append(r1_gen2_feats)
        gen2_feats.append(r2_gen2_feats)
        gen2_feats.append(r3_gen2_feats)

        raw_feats = []
        raw_feats.append(r1_raw_feats)
        raw_feats.append(r2_raw_feats)
        raw_feats.append(r3_raw_feats)

        top_feats = torch.cat((r1_gen_feats, r2_gen_feats, r3_gen_feats),dim=1)
        raw_feats2 = torch.cat((r1_raw_feats, r2_raw_feats, r3_raw_feats), dim=1)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features(index)
        # self_feats = self.dropout(self_feats)
        # number of nodes in a batch
        n = len(nodes)

        # concat the intra-aggregated embeddings from each relation
        # Eq. (9) in the paper

        # cat_feats = torch.cat((r1_gen_feats, r2_gen_feats, r3_gen_feats, r1_feats, r2_feats, r3_feats), dim=1)
        cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
        combined = F.relu(cat_feats.mm(self.weight).t())

        return combined, center_scores, gen_feats, gen2_feats, raw_feats, raw_feats2


class IntraAgg(nn.Module):

    def __init__(self, features, pe_features, feat_dim, embed_dim, train_pos, rho, gen, cuda=True):
        """
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param rho: the ratio of the oversample neighbors for the minority class
		:param cuda: whether to use GPU
		"""
        super(IntraAgg, self).__init__()

        self.features = features
        self.pe_features = pe_features
        self.cuda = cuda
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.train_pos = train_pos
        self.rho = rho
        self.gen = gen
        self.dropout = nn.Dropout(p=0.5)
        self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
        self.weight1 = nn.Parameter(torch.FloatTensor(self.feat_dim*4, self.embed_dim))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
        self.weight3 = nn.Parameter(torch.FloatTensor(3 * self.feat_dim, self.embed_dim))
        self.weight4 = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
        self.weight5 = nn.Parameter(torch.FloatTensor(4 * self.feat_dim, self.embed_dim))
        self.weight6 = nn.Parameter(torch.FloatTensor(2 * self.feat_dim, self.embed_dim))
        self.weight7 = nn.Parameter(torch.FloatTensor(2 * self.feat_dim, self.feat_dim))
        self.label_clf = nn.Linear(self.feat_dim, 2)
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.weight1)
        init.xavier_uniform_(self.weight2)
        init.xavier_uniform_(self.weight3)
        init.xavier_uniform_(self.weight4)
        init.xavier_uniform_(self.weight5)
        init.xavier_uniform_(self.weight6)
        init.xavier_uniform_(self.weight7)
    def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list,
                train_flag):
        """
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
		:param train_flag: indicates whether in training or testing mode
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

        if self.cuda:
            self_feats = self.features(torch.LongTensor(nodes).cuda())
        else:
            self_feats = self.features(torch.LongTensor(nodes))
        agg_all_feats = get_agg_feats(self.features, to_neighs_list, self.cuda)
        sampled_list = random_edge_sampling(self_feats, to_neighs_list, p=0.5, seed=42)
        agg_all_feats2 = get_agg_feats(self.features, sampled_list, self.cuda)
        cat_feats = torch.cat((self_feats, agg_all_feats), dim=1)
        cat2_feats = torch.cat((self_feats, agg_all_feats2), dim=1)
        gen_feats, gen2_feats, raw_feats = self.gen(
            cat_feats, cat2_feats, self_feats
        )


        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        # center_feats = self.features(torch.LongTensor(nodes))
        center_pe_feats = self.pe_features(torch.LongTensor(nodes))
        # gen_feats, raw_feats = self.gen(center_feats)
        # center_scores = self.label_clf(center_feats)
        gene_feats = torch.cat((center_pe_feats, gen_feats), dim=1)
        samp_neighs, non_samp_neighs, samp_scores = choose_neighs_and_get_features(self.features, gen_feats, to_neighs_list, sample_list, self.pe_features, center_pe_feats)
       # 消融实验 随机邻居选择
       #  samp_neighs, non_samp_neighs, samp_scores = choose_neighs_randomly(to_neighs_list, sample_list)
        # filer neighbors under given relation in the train mode
        # if train_flag:
        # 	samp_neighs, non_samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
        # else:
        # 	samp_neighs, non_samp_neighs, samp_scores = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)

        # if train_flag:
        #     samp_neighs, non_samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list,
        #                                                                    sample_list, self.rho, pos_scores, self.train_pos)
        # else:
        #     samp_neighs, non_samp_neighs, samp_scores = choose_step_test(batch_scores, batch_labels, neigh_scores, to_neighs_list,
        #                                                                  sample_list)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        unique_nodes_list2 = list(set.union(*non_samp_neighs))
        unique_nodes2 = {n: i for i, n in enumerate(unique_nodes_list2)}

        # intra-relation aggregation only with sampled neighbors
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        mask2 = Variable(torch.zeros(len(non_samp_neighs), len(unique_nodes2)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        column_indices2 = [unique_nodes2[n] for non_samp_neigh in non_samp_neighs for n in non_samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        row_indices2 = [i for i in range(len(non_samp_neighs)) for _ in range(len(non_samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mask2[row_indices2, column_indices2] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        num_neigh2 = mask2.sum(1, keepdim=True)
        mask = mask.div(num_neigh)  # mean aggregator
        mask2 = mask2.div(num_neigh2)
        if self.cuda:
            self_feats = self.features(torch.LongTensor(nodes).cuda())
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            embed_matrix2 = self.features(torch.LongTensor(unique_nodes_list2).cuda())
        else:
            self_feats = self.features(torch.LongTensor(nodes))
            # self_gen_feats, self_raw_feats = self.gen(self_feats)
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            embed_matrix2 = self.features(torch.LongTensor(unique_nodes_list2))
        agg_feats = mask.mm(embed_matrix)  # single relation aggregator
        agg_feats2 = mask2.mm(embed_matrix2)
        cat_feats = torch.cat((self_feats, agg_feats), dim=1)
        h_f_feats = F.relu((self_feats - agg_feats2).mm(self.weight2))
        l_f_feats = F.relu(cat_feats.mm(self.weight))
        h_top = F.relu(torch.cat((l_f_feats, h_f_feats), dim=1).mm(self.weight5))
        feats = torch.cat((h_top, gen_feats), dim=1)
        to_feats = F.relu(feats.mm(self.weight1))
        return to_feats, samp_scores, gen_feats, gen2_feats, raw_feats, h_top

def get_agg_feats(features, samp_neighs, cuda):
    neighs = []
    for samp in samp_neighs:
        neighs.append(set(samp))
    unique_nodes_list = list(set.union(*neighs))
    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

    # intra-relation aggregation only with sampled neighbors
    mask = Variable(torch.zeros(len(neighs), len(unique_nodes)))
    column_indices = [unique_nodes[n] for samp_neigh in neighs for n in samp_neigh]
    row_indices = [i for i in range(len(neighs)) for _ in range(len(neighs[i]))]
    mask[row_indices, column_indices] = 1
    if cuda:
        mask = mask.cuda()
    num_neigh = mask.sum(1, keepdim=True)
    mask = mask.div(num_neigh)  # mean aggregator
    if cuda:
        # self_feats = self.features(torch.LongTensor(nodes).cuda())
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        # self_feats = self.features(torch.LongTensor(nodes))
        embed_matrix = features(torch.LongTensor(unique_nodes_list))
    agg_feats = mask.mm(embed_matrix)

    return agg_feats


def choose_step_neighs(center_scores,  center_labels, neigh_scores, neighs_list, sample_list, sample_rate, minor_scores,minor_list):
    """
	Choose step for neighborhood sampling
	:param center_scores: the label-aware scores of batch nodes
	:param center_labels: the label of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class in one relation
	:param minor_list: minority node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:para sample_rate: the ratio of the oversample neighbors for the minority class
	"""
    samp_neighs = []
    samp_score_diff = []
    non_samp_neighs = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        # top-p sampling according to distance ranking
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]

            non_selected_neighs = [neighs_indices[i] for i in selected_neigh_indices[num_sample:]]

            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            non_selected_neighs = [neighs_indices[0]]
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

        # if isinstance(non_selected_neighs, np.int32):
        # 	non_selected_neighs = [non_selected_neighs]  # 将单个整数转换为包含该整数的列表
        non_samp_neighs.append(set(non_selected_neighs))

    return samp_neighs, non_samp_neighs, samp_score_diff


def choose_step_test(center_scores,  center_labels, neigh_scores, neighs_list, sample_list):
    """
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

    samp_neighs = []
    samp_scores = []
    non_samp_neighs = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

        # top-p sampling according to distance ranking and thresholds
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            non_selected_neighs = [neighs_indices[n] for n in selected_indices[num_sample:]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            non_selected_neighs = [neighs_indices[0]]
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

        # if isinstance(non_selected_neighs, np.int32):
        # 	non_selected_neighs = [non_selected_neighs]  # 将单个整数转换为包含该整数的列表
        non_samp_neighs.append(set(non_selected_neighs))  # 现在可以添加到集合中
    return samp_neighs, non_samp_neighs, samp_scores



# from torch.autograd import Variable


class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionAggregator, self).__init__()
        self.attention_layer = nn.Linear(feature_dim * 2, 1)  # 计算注意力分数的线性层

    def forward(self, center_feats, neighbor_feats):
        # center_feats: 中心节点的特征
        # neighbor_feats: 邻居节点的特征
        # print(center_feats)
        # print(neighbor_feats)
        # 将中心节点特征与邻居节点特征连接
        combined_feats = torch.cat([center_feats.unsqueeze(0).expand(neighbor_feats.size(0), -1), neighbor_feats], dim=1)
        attention_scores = self.attention_layer(combined_feats)
        attention_weights = torch.softmax(attention_scores, dim=1)  # 计算注意力权重

        return attention_weights


def get_attention_agg_feats(features, samp_neighs, cuda):
    neighs = []
    for samp in samp_neighs:
        neighs.append(set(samp))
    unique_nodes_list = list(set.union(*neighs))
    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

    # 获取邻居特征
    if cuda:
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        embed_matrix = features(torch.LongTensor(unique_nodes_list))

    # 初始化注意力聚合器
    feature_dim = embed_matrix.size(1)  # 特征维度
    attention_aggregator = AttentionAggregator(feature_dim)

    agg_feats = []
    for i, samp_neigh in enumerate(neighs):
        # 中心节点特征
        center_node_feats = embed_matrix[i]  # 假设每个样本的中心节点在 embed_matrix 的对应行中
        neighbor_indices = [unique_nodes[n] for n in samp_neigh]
        neighbor_feats = embed_matrix[neighbor_indices]

        # 计算注意力权重并加权聚合
        attention_weights = attention_aggregator(center_node_feats, neighbor_feats)
        # weighted_sum = (attention_weights * neighbor_feats).sum(dim=1)  # 加权求和
        weighted_sum = (attention_weights * neighbor_feats)
        weighted_sum = weighted_sum.sum(dim=0, keepdim=True)
        agg_feats.append(weighted_sum)

    # agg_feats = torch.stack(agg_feats)  # 将结果堆叠成一个张量
    # print(agg_feats)
    agg_feats = torch.cat(agg_feats, dim=0)
    return agg_feats


def random_edge_sampling(center_feats, to_neighs_list, p=0.5, seed=None):
    """
    对图中的边进行随机采样

    参数:
    center_feats: 中心节点特征
    to_neighs_list: 邻居节点列表
    p: 采样概率，默认为1.0，需要是浮点数
    seed: 随机种子

    返回:
    sampled_neighs_list: 采样后的邻居节点列表
    """
    if seed is not None:
        torch.manual_seed(seed)

    num_nodes = len(center_feats)
    sampled_neighs_list = []

    # 确保p是浮点数且在有效范围内
    p = float(p)
    p = max(0.0, min(1.0, p))  # 确保p在[0,1]范围内

    for node_idx in range(num_nodes):
        neighs = to_neighs_list[node_idx]

        # 处理空邻居列表的情况
        if len(neighs) == 0:
            sampled_neighs_list.append([])
            continue

        # 处理单个邻居的情况
        if len(neighs) == 1:
            # 对单个邻居使用更高的采样概率，避免孤立节点
            sample_p = min(1.0, p + 0.6)  # 提高采样概率，但不超过1
            bernoulli = torch.distributions.Bernoulli(torch.tensor(sample_p, dtype=torch.float32))
            samples = bernoulli.sample(torch.Size([1]))
            sampled_neighs = [neighs[0]] if samples[0] == 1 else []
            sampled_neighs_list.append(sampled_neighs)
            continue

        # 处理多个邻居的情况
        try:
            # 创建伯努利分布
            bernoulli = torch.distributions.Bernoulli(torch.tensor(p, dtype=torch.float32))
            samples = bernoulli.sample(torch.Size([len(neighs)]))

            # 采样边
            sampled_neighs = [neigh for neigh, sample in zip(neighs, samples) if sample == 1]

            # 确保至少保留一个邻居（如果原本有邻居的话）
            if len(sampled_neighs) == 0 and len(neighs) > 0:
                # 随机选择一个邻居
                random_idx = torch.randint(0, len(neighs), (1,)).item()
                sampled_neighs = [neighs[random_idx]]

            sampled_neighs_list.append(sampled_neighs)

        except Exception as e:
            print(f"Error occurred while sampling edges for node {node_idx}: {str(e)}")
            # 发生错误时，保留原始邻居列表
            sampled_neighs_list.append(neighs)

    return sampled_neighs_list


def choose_neighs_randomly(neighs_list, sample_list):
    """
    Randomly selects neighboring nodes for each target node.
    :param neighs_list: list of neighbors
    :param sample_list: number of neighbors to sample for each target node
    """
    samp_neighs = []
    non_samp_neighs = []
    samp_score_diff = []

    # Loop over all nodes and their neighbors
    for idx, neighs in enumerate(neighs_list):
        # If there's only one neighbor, select it directly
        if len(neighs) == 1:
            samp_neighs.append(set(neighs))
            non_samp_neighs.append(set(neighs))
            continue

        # Number of neighbors to sample for this node
        num_sample = sample_list[idx]

        # Randomly shuffle neighbors
        shuffled_neighs = random.sample(neighs, len(neighs))

        # Select neighbors
        selected_neighs = shuffled_neighs[:num_sample]
        non_selected_neighs = shuffled_neighs[num_sample:]

        samp_neighs.append(set(selected_neighs))
        non_samp_neighs.append(set(non_selected_neighs))

    return samp_neighs, non_samp_neighs, samp_score_diff
