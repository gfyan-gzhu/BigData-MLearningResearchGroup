import torch
import torch.nn as nn
from torch.nn import init


class PCALayer(nn.Module):

	def __init__(self, num_classes, inter1, lambda_1, nei_gen):
		"""
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(PCALayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()
		self.nei_gen = nei_gen
		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1
		self.device = 'cuda'

	def forward(self, nodes, labels, train_flag=True):
		embeds1, label_scores, gen_feats, gen2_feat, raw_feats, top_feats = self.inter1(nodes, labels, train_flag)
		scores = self.weight.mm(embeds1)
		return embeds1.t(), scores.t(), label_scores, gen_feats, gen2_feat, raw_feats, top_feats

	def to_prob(self, nodes, labels, train_flag=True):
		embeds1, gnn_logits, label_logits, gen_feats, gen2_feat, raw_feats, top_feats = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)
		return gnn_scores, label_scores

	def loss(self, nodes, labels, train_flag=True):
		# labels = labels.to(self.device)
		embeds1, gnn_scores, label_scores, gen_feats, gen2_feats, raw_feats, top_feats = self.forward(nodes, labels, train_flag)
		# label_scores = label_scores.to(self.device)
		label_loss = self.xent(label_scores, labels.squeeze())
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# the loss function
		# final_loss = gnn_loss + self.lambda_1 * label_loss
		final_loss = gnn_loss
		cons_loss = []
		for i in range(len(self.nei_gen)):
			c_loss = self.nei_gen[i].discriminator.forward(gen_feats[i], gen2_feats[i], raw_feats[i], labels)
			cons_loss.append(c_loss)
		context_loss = sum(cons_loss)
		final_loss = final_loss + self.lambda_1 * cons_loss
		return final_loss
