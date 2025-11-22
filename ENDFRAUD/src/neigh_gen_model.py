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
        # Discriminator is used to judge the similarity between the generated guidance node features and the positive and negative samples
        # self.discriminator = Discriminator(2 * feat_shape, 1)
        self.discriminator = Discriminator(2 * feat_shape, 1)

        # The generator network layers
        # self.fc1 = nn.Linear(latent_dim, 512).requires_grad_(True)
        self.fc1 = nn.Linear(32, 64).requires_grad_(True)
        self.fc2 = nn.Linear(64, 128).requires_grad_(True)
        self.fc3 = nn.Linear(64, 32).requires_grad_(True)
        self.fc4 = nn.Linear(128, 64).requires_grad_(True)
        self.fc_flat = nn.Linear(2048, latent_dim).requires_grad_(True)
        # self.bn0 = nn.BatchNorm1d(latent_dim).requires_grad_(False)
        self.bn0 = nn.BatchNorm1d(32).requires_grad_(False)
        # These normalization layers are optional, but self.bn0 is needed, otherwise it cannot be trained well
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
        # context-level
        self.f_k = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        # local-level
        self.f_k_env = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        self.temperature = 0.2
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

        # self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim * len(intraggs) + self.feat_dim, self.embed_dim))
        # init.xavier_uniform_(self.weight)

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

        raw_positive_feats = torch.mean(raw_feats[classify_label_1], dim=0, keepdim=True)
        raw_negative_feats = torch.mean(raw_feats[classify_label_0], dim=0, keepdim=True)

        """context - level"""
        positive_sample = []
        negative_sample = []
        positive2_sample = []
        negative2_sample = []
        raw_positive_sample = []
        raw_negative_sample = []
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1):
                positive_sample.append(positive_feats)
                negative_sample.append(negative_feats)
                positive2_sample.append(positive2_feats)
                negative2_sample.append(negative2_feats)
                raw_positive_sample.append(raw_positive_feats)
                raw_negative_sample.append(raw_negative_feats)
            else:
                positive_sample.append(negative_feats)
                negative_sample.append(positive_feats)
                positive2_sample.append(negative_feats)
                negative2_sample.append(positive_feats)
                raw_positive_sample.append(raw_negative_feats)
                raw_negative_sample.append(raw_positive_feats)

        positive_sample = torch.cat(positive_sample)
        negative_sample = torch.cat(negative_sample)
        positive2_sample = torch.cat(positive2_sample)
        negative2_sample = torch.cat(negative2_sample)
        raw_positive_sample = torch.cat(raw_positive_sample)
        raw_negative_sample = torch.cat(raw_negative_sample)

        """local-level"""
        # get the environment info

        return positive_sample, negative_sample, positive2_sample, negative2_sample, raw_positive_sample, raw_negative_sample



    def contrastive_loss(self, anchor, positive_1, negative_1, positive_2, negative_2, raw_positive, raw_negative,
                         labels, tau=0.1):
        """
        计算节点级别的对比损失
        anchor: 原始视图中的锚点嵌入 (B, D)
        positive_1: 邻居子图中的正样本嵌入 (B, D)
        positive_2: 邻居增强子图中的正样本嵌入 (B, D)
        negative_1: 邻居子图中的负样本嵌入 (B, D)
        negative_2: 邻居增强子图中的负样本嵌入 (B, D)
        labels: 标签，用于构建正样本和负样本掩码 (B,)
        tau: 温度系数，用于控制相似度的平滑度
        """

        # 计算相似度，并限制值的范围，防止数值溢出
        sim_pos_1 = torch.mm(anchor, positive_1.T) / tau
        sim_pos_2 = torch.mm(anchor, positive_2.T) / tau
        sim_pos_3 = torch.mm(anchor, raw_positive.T) / tau
        sim_neg_1 = torch.mm(anchor, negative_1.T) / tau
        sim_neg_2 = torch.mm(anchor, negative_2.T) / tau
        sim_neg_3 = torch.mm(anchor, raw_negative.T) / tau

        # 对相似度进行clamp，限制在一个合理的范围内
        sim_pos_1 = torch.clamp(sim_pos_1, min=-10, max=10)
        sim_pos_2 = torch.clamp(sim_pos_2, min=-10, max=10)
        sim_pos_3 = torch.clamp(sim_pos_3, min=-10, max=10)
        sim_neg_1 = torch.clamp(sim_neg_1, min=-10, max=10)
        sim_neg_2 = torch.clamp(sim_neg_2, min=-10, max=10)
        sim_neg_3 = torch.clamp(sim_neg_3, min=-10, max=10)
        # 创建正样本掩码 (label_mask)
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # 正样本分数
        pos_scores = torch.exp(sim_pos_1) * label_mask + torch.exp(sim_pos_2) * label_mask + torch.exp(
            sim_pos_3) * label_mask

        # 负样本分数 (1 - label_mask 表示负样本对)
        neg_scores = torch.exp(sim_neg_1) * (1 - label_mask) + torch.exp(sim_neg_2) * (1 - label_mask) + torch.exp(
            sim_neg_3) * (1 - label_mask)

        # 防止正负样本分数的总和为 0
        pos_sum = torch.sum(pos_scores, dim=1) + 1e-8  # 添加 1e-8 防止 NaN
        neg_sum = torch.sum(neg_scores, dim=1) + 1e-8  # 添加 1e-8 防止 NaN

        # 检查是否存在 NaN
        if torch.isnan(pos_sum).any():
            print("NaN found in pos_sum")
        if torch.isnan(neg_sum).any():
            print("NaN found in neg_sum")

        # 计算 InfoNCE 损失
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))

        # 检查是否存在 NaN
        if torch.isnan(loss).any():
            print("NaN found in loss")

        return loss.mean()

    def get_contrast_loss(self, gen_feats, gen2_feats, raw_feats, labels):
        """
        :param gen_feats: [batchsize, 64]
        :param labels: labels
        :return: loss
        """
        gen_loss = []
        positive_sample, negative_sample, positive2_sample, negative2_sample, raw_positive_sample, raw_negative_sample = self.get_contrast_sample(
            gen_feats, gen2_feats, raw_feats, labels)
        context_logits = self.contrastive_learning2(gen_feats, gen2_feats, raw_feats, labels)

        return context_logits

    def forward(self, gen_feats, gen2_feats, raw_feats, labels):
        """
        contrastive learning loss
        """
        # gen_loss = []
        #
        # for feats in gen_feats:
        #     gen_loss.append(self.get_contrast_loss(feats, labels))
        #
        # return gen_loss

        return self.get_contrast_loss(gen_feats, gen2_feats, raw_feats, labels)

    def contrastive_learning(self, z, z_tilde, raw_feats, y, tau=0.1):
        # 归一化特征向量
        z = F.normalize(z, dim=1)  # [1024, 32]
        z_tilde = F.normalize(z_tilde, dim=1)  # [1024, 32]

        # 计算所有节点对之间的相似度矩阵
        sim_matrix = torch.mm(z, z_tilde.t()) / tau  # [1024, 1024]
        sim_matrix1 = torch.mm(z, z.t()) / tau
        sim_matrix2 = torch.mm(z, raw_feats.t()) / tau
        # 创建标签矩阵 (1 if yi=yj, 0 otherwise)
        y = y.view(-1, 1)  # [1024, 1]
        label_matrix = (y == y.t()).float()  # [1024, 1024]

        # 计算分子 (positive pairs)
        positive_pairs = torch.exp(sim_matrix) * label_matrix + torch.exp(sim_matrix1) * label_matrix + torch.exp(sim_matrix2) * label_matrix
        positive_sum = positive_pairs.sum(dim=1)  # [1024]

        # 计算分母 (all pairs)
        exp_sim_matrix = torch.exp(sim_matrix) + torch.exp(sim_matrix1) + torch.exp(sim_matrix2) # [1024, 1024]
        denominator = exp_sim_matrix.sum(dim=1)  # [1024]

        # 避免数值不稳定性
        eps = 1e-8
        loss = -torch.log(positive_sum / (denominator + eps) + eps)

        # 返回平均损失
        return loss.mean()

    def contrastive_learning2(self, z, z_tilde, raw_feats, y, tau=0.1):
        """
        参数:
        - z: 原始图的节点表示 [1024, 32]
        - z_tilde: 增强图的节点表示 [1024, 32]
        - raw_feats: 原始特征 [1024, 32]
        - y: 节点标签 [1024]
        - tau: 温度参数

        返回:
        - loss: 标量损失值
        """
        # 1. 特征归一化，添加检查
        z_norm = torch.norm(z, p=2, dim=1, keepdim=True)
        z_tilde_norm = torch.norm(z_tilde, p=2, dim=1, keepdim=True)
        raw_feats_norm = torch.norm(raw_feats, p=2, dim=1, keepdim=True)

        # 检查是否有零范数的向量
        eps = 1e-8
        z_norm = torch.clamp(z_norm, min=eps)
        z_tilde_norm = torch.clamp(z_tilde_norm, min=eps)
        raw_feats_norm = torch.clamp(raw_feats_norm, min=eps)

        z = z / z_norm
        z_tilde = z_tilde / z_tilde_norm
        raw_feats = raw_feats / raw_feats_norm

        # 2. 计算相似度矩阵，使用更稳定的实现
        sim_matrix = torch.mm(z, z_tilde.t()) / tau  # [1024, 1024]
        sim_matrix1 = torch.mm(z, z.t()) / tau
        sim_matrix2 = torch.mm(z, raw_feats.t()) / tau

        # 3. 创建标签矩阵
        y = y.view(-1, 1)  # [1024, 1]
        label_matrix = (y == y.t()).float()  # [1024, 1024]

        # 4. 使用log-sum-exp技巧来提高数值稳定性
        max_sim = torch.max(torch.stack([
            torch.max(sim_matrix, dim=1)[0],
            torch.max(sim_matrix1, dim=1)[0],
            torch.max(sim_matrix2, dim=1)[0]
        ]), dim=0)[0]

        # 5. 计算分子（positive pairs）
        exp_sim = torch.exp(sim_matrix - max_sim.view(-1, 1))
        exp_sim1 = torch.exp(sim_matrix1 - max_sim.view(-1, 1))
        exp_sim2 = torch.exp(sim_matrix2 - max_sim.view(-1, 1))

        positive_pairs = (exp_sim * label_matrix +
                          exp_sim1 * label_matrix +
                          exp_sim2 * label_matrix)
        positive_sum = positive_pairs.sum(dim=1)  # [1024]

        # 6. 计算分母（all pairs）
        denominator = (exp_sim.sum(dim=1) +
                       exp_sim1.sum(dim=1) +
                       exp_sim2.sum(dim=1))  # [1024]

        # 7. 计算损失，添加数值稳定性保护
        eps = 1e-8

        # 检查是否有零值或无穷大
        positive_sum = torch.clamp(positive_sum, min=eps)
        denominator = torch.clamp(denominator, min=eps)

        # 计算最终损失
        loss = -torch.log(positive_sum / denominator)

        # 8. 处理可能的无效值
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        # 9. 返回平均损失
        valid_samples = torch.isfinite(loss).float().sum()
        if valid_samples > 0:
            return loss.sum() / valid_samples
        else:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
