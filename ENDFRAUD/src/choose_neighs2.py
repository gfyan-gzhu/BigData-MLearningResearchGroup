import torch
from .policyNetwork import PolicyNetwork

def choose_neighs_and_get_features2(features, gen_feats, idx, neighs):

    samp_neighs = []
    non_samp_neighs = []
    samp_score_diff = []
    # the first node in neighs_list is the node itself, which needs to be removed
    gen_feat = gen_feats[idx]
    gen_feat = gen_feat.expand(len(neighs), gen_feat.shape[0])
    neighs_feature = features(torch.LongTensor(neighs[0:]))
    observation = torch.cat((neighs_feature, gen_feat), dim=1)
    pn = PolicyNetwork(gen_feat.shape[1]+neighs_feature.shape[1], 1, neighs_feature.shape[1]*8, neighs_feature.shape[1]*8)
    action = pn(observation).squeeze(0)
    return action
