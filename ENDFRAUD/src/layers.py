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



class InterAgg(nn.Module):

    def __init__(self, features, pe_features, feature_dim, embed_dim,
                 train_pos, adj_lists, intraggs, inter='GNN', cuda=False):
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

        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
                                 set.union(*to_neighs[2], set(nodes)))

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

        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
        r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
        r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

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

        cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
        combined = F.relu(cat_feats.mm(self.weight).t())

        return combined, center_scores, gen_feats, gen2_feats, raw_feats, raw_feats2


class IntraAgg(nn.Module):

    def __init__(self, features, pe_features, feat_dim, embed_dim, train_pos, rho, gen, cuda=True):
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

        center_pe_feats = self.pe_features(torch.LongTensor(nodes))

        gene_feats = torch.cat((center_pe_feats, gen_feats), dim=1)
        samp_neighs, non_samp_neighs, samp_scores = choose_neighs_and_get_features(self.features, gen_feats, to_neighs_list, sample_list, self.pe_features, center_pe_feats)
       #  samp_neighs, non_samp_neighs, samp_scores = choose_neighs_randomly(to_neighs_list, sample_list)
        # filer neighbors under given relation in the train mode
        # 	samp_neighs, non_samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
        # 	samp_neighs, non_samp_neighs, samp_scores = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)

        #     samp_neighs, non_samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list,
        #                                                                    sample_list, self.rho, pos_scores, self.train_pos)
        # else:
        #     samp_neighs, non_samp_neighs, samp_scores = choose_step_test(batch_scores, batch_labels, neigh_scores, to_neighs_list,
        #                                                                  sample_list)

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        unique_nodes_list2 = list(set.union(*non_samp_neighs))
        unique_nodes2 = {n: i for i, n in enumerate(unique_nodes_list2)}

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
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        # self_feats = self.features(torch.LongTensor(nodes))
        embed_matrix = features(torch.LongTensor(unique_nodes_list))
    agg_feats = mask.mm(embed_matrix)

    return agg_feats


def choose_step_neighs(center_scores,  center_labels, neigh_scores, neighs_list, sample_list, sample_rate, minor_scores,minor_list):
    samp_neighs = []
    samp_score_diff = []
    non_samp_neighs = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

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
        # 	non_selected_neighs = [non_selected_neighs]
        non_samp_neighs.append(set(non_selected_neighs))

    return samp_neighs, non_samp_neighs, samp_score_diff


def choose_step_test(center_scores,  center_labels, neigh_scores, neighs_list, sample_list):


    samp_neighs = []
    samp_scores = []
    non_samp_neighs = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

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
        # 	non_selected_neighs = [non_selected_neighs]
        non_samp_neighs.append(set(non_selected_neighs))
    return samp_neighs, non_samp_neighs, samp_scores




class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionAggregator, self).__init__()
        self.attention_layer = nn.Linear(feature_dim * 2, 1)

    def forward(self, center_feats, neighbor_feats):

        combined_feats = torch.cat([center_feats.unsqueeze(0).expand(neighbor_feats.size(0), -1), neighbor_feats], dim=1)
        attention_scores = self.attention_layer(combined_feats)
        attention_weights = torch.softmax(attention_scores, dim=1)

        return attention_weights


def get_attention_agg_feats(features, samp_neighs, cuda):
    neighs = []
    for samp in samp_neighs:
        neighs.append(set(samp))
    unique_nodes_list = list(set.union(*neighs))
    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

    if cuda:
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        embed_matrix = features(torch.LongTensor(unique_nodes_list))

    feature_dim = embed_matrix.size(1)
    attention_aggregator = AttentionAggregator(feature_dim)

    agg_feats = []
    for i, samp_neigh in enumerate(neighs):

        center_node_feats = embed_matrix[i]
        neighbor_indices = [unique_nodes[n] for n in samp_neigh]
        neighbor_feats = embed_matrix[neighbor_indices]

        attention_weights = attention_aggregator(center_node_feats, neighbor_feats)
        # weighted_sum = (attention_weights * neighbor_feats).sum(dim=1)  # 加权求和
        weighted_sum = (attention_weights * neighbor_feats)
        weighted_sum = weighted_sum.sum(dim=0, keepdim=True)
        agg_feats.append(weighted_sum)

    # agg_feats = torch.stack(agg_feats)
    # print(agg_feats)
    agg_feats = torch.cat(agg_feats, dim=0)
    return agg_feats


def random_edge_sampling(center_feats, to_neighs_list, p=0.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    num_nodes = len(center_feats)
    sampled_neighs_list = []

    p = float(p)
    p = max(0.0, min(1.0, p))

    for node_idx in range(num_nodes):
        neighs = to_neighs_list[node_idx]

        if len(neighs) == 0:
            sampled_neighs_list.append([])
            continue

        if len(neighs) == 1:
            sample_p = min(1.0, p + 0.6)
            bernoulli = torch.distributions.Bernoulli(torch.tensor(sample_p, dtype=torch.float32))
            samples = bernoulli.sample(torch.Size([1]))
            sampled_neighs = [neighs[0]] if samples[0] == 1 else []
            sampled_neighs_list.append(sampled_neighs)
            continue

        try:
            bernoulli = torch.distributions.Bernoulli(torch.tensor(p, dtype=torch.float32))
            samples = bernoulli.sample(torch.Size([len(neighs)]))

            sampled_neighs = [neigh for neigh, sample in zip(neighs, samples) if sample == 1]

            if len(sampled_neighs) == 0 and len(neighs) > 0:
                random_idx = torch.randint(0, len(neighs), (1,)).item()
                sampled_neighs = [neighs[random_idx]]

            sampled_neighs_list.append(sampled_neighs)

        except Exception as e:
            print(f"Error occurred while sampling edges for node {node_idx}: {str(e)}")
            sampled_neighs_list.append(neighs)

    return sampled_neighs_list


def choose_neighs_randomly(neighs_list, sample_list):
    samp_neighs = []
    non_samp_neighs = []
    samp_score_diff = []

    for idx, neighs in enumerate(neighs_list):
        if len(neighs) == 1:
            samp_neighs.append(set(neighs))
            non_samp_neighs.append(set(neighs))
            continue

        num_sample = sample_list[idx]

        shuffled_neighs = random.sample(neighs, len(neighs))
        selected_neighs = shuffled_neighs[:num_sample]
        non_selected_neighs = shuffled_neighs[num_sample:]

        samp_neighs.append(set(selected_neighs))
        non_samp_neighs.append(set(non_selected_neighs))

    return samp_neighs, non_samp_neighs, samp_score_diff
