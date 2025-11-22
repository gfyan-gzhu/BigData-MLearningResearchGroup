import torch
import torch.nn as nn
import torch.nn.functional as F
from src.neigh_gen_layers import GraphConvolution


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)


class Gen(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(Gen, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.temperature = 0.2
        self.discriminator = Discriminator(2 * feat_shape, 1)
        # The generator network layers
        self.fc1 = nn.Linear(32, 64).requires_grad_(True)
        self.fc2 = nn.Linear(64, 128).requires_grad_(True)
        self.fc3 = nn.Linear(64, 32).requires_grad_(True)
        self.fc4 = nn.Linear(128, 64).requires_grad_(True)
        self.fc_flat = nn.Linear(2048, latent_dim).requires_grad_(True)
        self.bn0 = nn.BatchNorm1d(32).requires_grad_(False)
        self.bn1 = nn.BatchNorm1d(64).requires_grad_(False)
        self.bn2 = nn.BatchNorm1d(2048).requires_grad_(False)
        self.bn3 = nn.BatchNorm1d(latent_dim).requires_grad_(False)
        self.dropout = dropout

    def forward(self, x, y, z):
        x = self.bn1(x)
        y = self.bn1(y)
        z = self.bn0(z)
        z = self.fc1(z)
        raw_feats = torch.relu(z)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc4(x))
        y = self.fc2(y)
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.relu(self.fc4(y))

        return x, y, raw_feats


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        self.f_k_env = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        self.temperature = 0.2
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round


    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def get_contrast_sample(self, gen_feats, gen2_feats, raw_feats, labels):
        classify_label_1 = []
        classify_label_0 = []
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1):
                classify_label_1.append(i)
            else:
                classify_label_0.append(i)


        positive_feats = torch.mean(gen_feats[classify_label_1], dim=0, keepdim=True)
        negative_feats = torch.mean(gen_feats[classify_label_0], dim=0, keepdim=True)

        positive2_feats = torch.mean(gen2_feats[classify_label_1], dim=0, keepdim=True)
        negative2_feats = torch.mean(gen2_feats[classify_label_0], dim=0, keepdim=True)

        positive_sample = []
        negative_sample = []
        positive2_sample = []
        negative2_sample = []
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1):
                positive_sample.append(positive_feats)
                negative_sample.append(negative_feats)
                positive2_sample.append(positive2_feats)
                negative2_sample.append(negative2_feats)
            else:
                positive_sample.append(negative_feats)
                negative_sample.append(positive_feats)
                positive2_sample.append(negative_feats)
                negative2_sample.append(positive_feats)


        positive_sample = torch.cat(positive_sample)
        negative_sample = torch.cat(negative_sample)
        positive2_sample = torch.cat(positive2_sample)
        negative2_sample = torch.cat(negative2_sample)


        return positive_sample, negative_sample, positive2_sample, negative2_sample, raw_positive_sample, raw_negative_sample



    def contrastive_loss(self, anchor, positive_1, negative_1, positive_2, negative_2, raw_positive, raw_negative,
                         labels, tau=0.1):


        sim_pos_1 = torch.mm(anchor, positive_1.T) / tau
        sim_pos_2 = torch.mm(anchor, positive_2.T) / tau
        sim_pos_3 = torch.mm(anchor, raw_positive.T) / tau
        sim_neg_1 = torch.mm(anchor, negative_1.T) / tau
        sim_neg_2 = torch.mm(anchor, negative_2.T) / tau
        sim_neg_3 = torch.mm(anchor, raw_negative.T) / tau

        sim_pos_1 = torch.clamp(sim_pos_1, min=-10, max=10)
        sim_pos_2 = torch.clamp(sim_pos_2, min=-10, max=10)
        sim_pos_3 = torch.clamp(sim_pos_3, min=-10, max=10)
        sim_neg_1 = torch.clamp(sim_neg_1, min=-10, max=10)
        sim_neg_2 = torch.clamp(sim_neg_2, min=-10, max=10)
        sim_neg_3 = torch.clamp(sim_neg_3, min=-10, max=10)
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        pos_scores = torch.exp(sim_pos_1) * label_mask + torch.exp(sim_pos_2) * label_mask + torch.exp(
            sim_pos_3) * label_mask

        neg_scores = torch.exp(sim_neg_1) * (1 - label_mask) + torch.exp(sim_neg_2) * (1 - label_mask) + torch.exp(
            sim_neg_3) * (1 - label_mask)

        pos_sum = torch.sum(pos_scores, dim=1) + 1e-8
        neg_sum = torch.sum(neg_scores, dim=1) + 1e-8

        if torch.isnan(pos_sum).any():
            print("NaN found in pos_sum")
        if torch.isnan(neg_sum).any():
            print("NaN found in neg_sum")

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))

        if torch.isnan(loss).any():
            print("NaN found in loss")

        return loss.mean()

    def get_contrast_loss(self, gen_feats, gen2_feats, raw_feats, labels):
        """
        :param gen_feats: [batchsize, 64]
        :param labels: labels
        :return: loss
        """
        positive_sample, negative_sample, positive2_sample, negative2_sample, raw_positive_sample, raw_negative_sample = self.get_contrast_sample(
            gen_feats, gen2_feats, raw_feats, labels)
        context_logits = self.contrastive_learning2(gen_feats, gen2_feats, raw_feats, labels)

        return context_logits

    def forward(self, gen_feats, gen2_feats, raw_feats, labels):
        """
        contrastive learning loss
        """

        return self.get_contrast_loss(gen_feats, gen2_feats, raw_feats, labels)

    def contrastive_learning(self, z, z_tilde, raw_feats, y, tau=0.1):

        z = F.normalize(z, dim=1)
        z_tilde = F.normalize(z_tilde, dim=1)


        sim_matrix = torch.mm(z, z_tilde.t()) / tau
        sim_matrix1 = torch.mm(z, z.t()) / tau


        y = y.view(-1, 1)
        label_matrix = (y == y.t()).float()


        positive_pairs = torch.exp(sim_matrix) * label_matrix + torch.exp(sim_matrix1) * label_matrix + torch.exp(sim_matrix2) * label_matrix
        positive_sum = positive_pairs.sum(dim=1)


        exp_sim_matrix = torch.exp(sim_matrix) + torch.exp(sim_matrix1) + torch.exp(sim_matrix2)
        denominator = exp_sim_matrix.sum(dim=1)


        eps = 1e-8
        loss = -torch.log(positive_sum / (denominator + eps) + eps)


        return loss.mean()

    def contrastive_learning2(self, z, z_tilde, raw_feats, y, tau=0.1):


        z_norm = torch.norm(z, p=2, dim=1, keepdim=True)
        z_tilde_norm = torch.norm(z_tilde, p=2, dim=1, keepdim=True)


        eps = 1e-8
        z_norm = torch.clamp(z_norm, min=eps)
        z_tilde_norm = torch.clamp(z_tilde_norm, min=eps)


        z = z / z_norm
        z_tilde = z_tilde / z_tilde_norm


        sim_matrix = torch.mm(z, z_tilde.t()) / tau
        sim_matrix1 = torch.mm(z, z.t()) / tau


        y = y.view(-1, 1)
        label_matrix = (y == y.t()).float()
        max_sim = torch.max(torch.stack([
            torch.max(sim_matrix, dim=1)[0],
            torch.max(sim_matrix1, dim=1)[0],

        ]), dim=0)[0]

        exp_sim = torch.exp(sim_matrix - max_sim.view(-1, 1))
        exp_sim1 = torch.exp(sim_matrix1 - max_sim.view(-1, 1))


        positive_pairs = (exp_sim * label_matrix +
                          exp_sim1 * label_matrix)
        positive_sum = positive_pairs.sum(dim=1)


        denominator = (exp_sim.sum(dim=1) +
                       exp_sim1.sum(dim=1) )

        eps = 1e-8

        positive_sum = torch.clamp(positive_sum, min=eps)
        denominator = torch.clamp(denominator, min=eps)

        loss = -torch.log(positive_sum / denominator)

        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        valid_samples = torch.isfinite(loss).float().sum()
        if valid_samples > 0:
            return loss.sum() / valid_samples
        else:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
