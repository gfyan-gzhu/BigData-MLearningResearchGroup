import torch
from .policyNetwork import PolicyNetwork
from .choose_neighs2 import choose_neighs_and_get_features2
def choose_neighs_and_get_features(features, gen_feats, neighs_list, sample_list, pe_features, center_pe_feats):

    samp_neighs = []
    non_samp_neighs = []
    samp_score_diff = []
    actions = []
    for idx, neighs in enumerate(neighs_list):
        if len(neighs) == 1:
            samp_neighs.append(set(neighs))
            non_samp_neighs.append(set(neighs))
            continue
        else:
            gen_feat = gen_feats[idx]
            gen_feat = gen_feat.expand(len(neighs), gen_feat.shape[0])
            neighs_feature = features(torch.LongTensor(neighs[0:]))

            observation = torch.cat((neighs_feature, gen_feat), dim=1)

            pn = PolicyNetwork(gen_feat.shape[1]+neighs_feature.shape[1], 1, neighs_feature.shape[1]*8, neighs_feature.shape[1]*8)

            action1 = pn(observation).squeeze(0)
            action2 = choose_neighs_and_get_features2(pe_features, center_pe_feats, idx, neighs)
            action = action1 + action2
            actions.append(action)
        neighs_indices = neighs_list[idx]
        sorted_action, sorted_neigh_indices = torch.sort(action, dim=0, descending=True)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        num_sample = sample_list[idx]
        if len(neighs_list[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            non_selected_neighs = [neighs_indices[i] for i in selected_neigh_indices[num_sample:]]
        else:
            selected_neighs = neighs_indices
            non_selected_neighs = [neighs_indices[0]]
        samp_neighs.append(set(selected_neighs))
        non_samp_neighs.append(set(non_selected_neighs))
    return samp_neighs, non_samp_neighs, samp_score_diff
