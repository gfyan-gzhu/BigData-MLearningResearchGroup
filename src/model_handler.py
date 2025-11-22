import time, datetime
import os
import random as rd
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from .utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step, undersample
from .model import PCALayer
from .layers import InterAgg, IntraAgg
from .graphsage import *
from src.neigh_gen_model import Gen

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler(object):

	def __init__(self, config):
		args = argparse.Namespace(**config)
		# load graph, feature, and label
		[homo, relation1, relation2, relation3], feat_data, labels, pe_feat = load_data(args.data_name,
																						prefix=args.data_dir)
		# 转换为 PyTorch tensor

		# train_test split
		np.random.seed(args.seed)
		random.seed(args.seed)
		if args.data_name == 'yelp':
			index = list(range(len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels,
																	train_size=args.train_ratio,
																	random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=args.test_ratio,
																	random_state=2, shuffle=True)

		elif args.data_name == 'amazon':  # amazon
			# 0-3304 are unlabeled nodes
			index = list(range(3305, len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
																	train_size=args.train_ratio, random_state=2,
																	shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=args.test_ratio, random_state=2,
																	shuffle=True)

		print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},' +
			  f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
		print(f"Classification threshold: {args.thres}")
		print(f"Feature dimension: {feat_data.shape[1]}")

		# split pos neg sets for under-sampling
		train_pos, train_neg = pos_neg_split(idx_train, y_train)

		# if args.data == 'amazon':
		feat_data = normalize(feat_data)
		# train_feats = feat_data[np.array(idx_train)]
		# scaler = StandardScaler()
		# scaler.fit(train_feats)
		# feat_data = scaler.transform(feat_data)
		args.cuda = not args.no_cuda and torch.cuda.is_available()
		os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
		# set input graph
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = homo
		else:
			adj_lists = [relation1, relation2, relation3]

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')

		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'pe_feat': pe_feat, 'adj_lists': adj_lists,
						'homo': homo,
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
						'train_pos': train_pos, 'train_neg': train_neg}

	def train(self):
		args = self.args
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
		pe_feat = self.dataset['pe_feat']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset[
			'idx_test'], self.dataset['y_test']
		train_pos, train_neg = self.dataset['train_pos'], self.dataset['train_neg']
		features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
		features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
		pe_features = nn.Embedding(pe_feat.shape[0], pe_feat.shape[1])
		pe_features.weight = nn.Parameter(torch.FloatTensor(pe_feat), requires_grad=False)
		dropout = nn.Dropout(p=0.5)  # 50% dropout probability
		if args.cuda:
			features.cuda()
			pe_features.cuda()
		neigh_gen1 = Gen(2 * feat_data.shape[1], 0.5, 5, feat_data.shape[1])
		neigh_gen2 = Gen(2 * feat_data.shape[1], 0.5, 5, feat_data.shape[1])
		neigh_gen3 = Gen(2 * feat_data.shape[1], 0.5, 5, feat_data.shape[1])
		neigh_gen = [neigh_gen1, neigh_gen2, neigh_gen3]

		# build one-layer models
		if args.model == 'PCGNN':
			intra1 = IntraAgg(features, pe_features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'],
							  args.rho, neigh_gen1, cuda=args.cuda)
			intra2 = IntraAgg(features, pe_features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'],
							  args.rho, neigh_gen2, cuda=args.cuda)
			intra3 = IntraAgg(features, pe_features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'],
							  args.rho, neigh_gen3, cuda=args.cuda)
			inter1 = InterAgg(features, pe_features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'],
							  adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False,
							   cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True,
								 cuda=args.cuda)

		if args.model == 'PCGNN':
			gnn_model = PCALayer(2, inter1, args.alpha, neigh_gen)
		elif args.model == 'SAGE':
			# the vanilla GraphSAGE model as baseline
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		if args.cuda:
			gnn_model.cuda()

		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
									 weight_decay=args.weight_decay)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.00001)
		for i in range(3):
			for name, i in gnn_model.nei_gen[i].named_parameters():
				i.requires_grad = True
		# optimizer_gen = []
		# for i in range(3):
		# 	optimizer_gen.append(
		# 		torch.optim.Adam(
		# 			filter(lambda p: p.requires_grad, neigh_gen[i].parameters()),
		# 			lr=args.gen_lr,
		# 			weight_decay=args.gen_weight_decay,
		# 		)
		# 	)
		#
		# # BCE
		# if args.cuda:
		# 	b_xent = nn.BCEWithLogitsLoss(
		# 		reduction="none", pos_weight=torch.tensor([args.negsamp_ratio])
		# 	).cuda()
		# else:
		# 	b_xent = nn.BCEWithLogitsLoss(
		# 		reduction="none", pos_weight=torch.tensor([args.negsamp_ratio])
		# 	)

		timestamp = time.time()
		timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
		dir_saver = args.save_dir + timestamp
		path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
		f1_mac_best, auc_best, ep_best = 0, 0, -1

		# train the model
		for epoch in range(args.num_epochs):
			# sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos'])*2)
			sampled_idx_train = undersample(train_pos, train_neg, scale=2)
			rd.shuffle(sampled_idx_train)
			random.shuffle(sampled_idx_train)

			num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

			loss = 0.0
			epoch_time = 0
			gen_loss_avg = [0, 0, 0]

			losses = []
			# mini-batch training
			for batch in range(num_batches):
				start_time = time.time()
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
				batch_nodes = sampled_idx_train[i_start:i_end]
				batch_label = self.dataset['labels'][np.array(batch_nodes)]


				optimizer.zero_grad()
				# Forward pass with dropout
				# h = features(torch.tensor(batch_nodes))  # Get node features
				# h = dropout(h)  # Apply dropout

				loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
				# optimizer.zero_grad()
				loss.backward()
				# losses.append(float(loss.item()))
				# losses.append(loss.item())
				optimizer.step()
				# scheduler.step()


				end_time = time.time()
				epoch_time += end_time - start_time
				loss += loss.item()

			print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
			if epoch <= 220:
				# print(f'Epoch: {epoch}, loss: {total_loss.item() / num_batches}, time: {epoch_time}s')
				# Valid the model for every $valid_epoch$ epoch
				if epoch % args.valid_epochs == 0:
					if args.model == 'SAGE' or args.model == 'GCN':
						print("Valid at epoch {}".format(epoch))
						f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model,
																					   args.batch_size, args.thres)
						if auc_val > auc_best:
							f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
							if not os.path.exists(dir_saver):
								os.makedirs(dir_saver)
							print('  Saving model ...')
							torch.save(gnn_model.state_dict(), path_saver)
					else:
						print("Valid at epoch {}".format(epoch))
						f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_pcgnn(idx_valid, y_valid, gnn_model,
																						args.batch_size, args.thres)
						if auc_val > auc_best:
							f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
							if not os.path.exists(dir_saver):
								os.makedirs(dir_saver)
							print('  Saving model ...')
							torch.save(gnn_model.state_dict(), path_saver)
			else:
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model,
																				   args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				else:
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_pcgnn(idx_valid, y_valid, gnn_model,
																					args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)


		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))
		if args.model == 'SAGE' or args.model == 'GCN':
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model,
																				args.batch_size, args.thres)
		else:
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, gnn_model,
																				 args.batch_size, args.thres)
		return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test

	def load_and_visualize_embeddings(self, model_path, save_path=None, batch_size=128):
		"""
		Load a trained model and visualize node embeddings
		Args:
			model_path: Path to the saved model
			save_path: Optional path to save the visualization plot
			batch_size: Batch size for processing nodes
		"""
		try:
			print(f"Loading model from {model_path}...")

			# Initialize the model structure (same as in train())
			features = nn.Embedding(self.dataset['feat_data'].shape[0], self.dataset['feat_data'].shape[1])
			features.weight = nn.Parameter(torch.FloatTensor(self.dataset['feat_data']), requires_grad=False)
			pe_features = nn.Embedding(self.dataset['pe_feat'].shape[0], self.dataset['pe_feat'].shape[1])
			pe_features.weight = nn.Parameter(torch.FloatTensor(self.dataset['pe_feat']), requires_grad=False)
			idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset[
				'idx_test'], self.dataset['y_test']
			if self.args.cuda:
				features.cuda()
			neigh_gen1 = Gen(2 * self.dataset['feat_data'].shape[1], 0.5, 5, self.dataset['feat_data'].shape[1])
			neigh_gen2 = Gen(2 * self.dataset['feat_data'].shape[1], 0.5, 5, self.dataset['feat_data'].shape[1])
			neigh_gen3 = Gen(2 * self.dataset['feat_data'].shape[1], 0.5, 5, self.dataset['feat_data'].shape[1])
			neigh_gen = [neigh_gen1, neigh_gen2, neigh_gen3]

			# Build model architecture based on model type
			if self.args.model == 'PCGNN':
				intra1 = IntraAgg(features, pe_features, self.dataset['feat_data'].shape[1], self.args.emb_size,
								  self.dataset['train_pos'],
								  self.args.rho, neigh_gen1, cuda=self.args.cuda)
				intra2 = IntraAgg(features, pe_features, self.dataset['feat_data'].shape[1], self.args.emb_size,
								  self.dataset['train_pos'],
								  self.args.rho, neigh_gen2, cuda=self.args.cuda)
				intra3 = IntraAgg(features, pe_features, self.dataset['feat_data'].shape[1], self.args.emb_size,
								  self.dataset['train_pos'],
								  self.args.rho, neigh_gen3, cuda=self.args.cuda)
				inter1 = InterAgg(features, pe_features, self.dataset['feat_data'].shape[1], self.args.emb_size,
								  self.dataset['train_pos'],
								  self.dataset['adj_lists'], [intra1, intra2, intra3], inter=self.args.multi_relation,
								  cuda=self.args.cuda)
				self.gnn_model = PCALayer(2, inter1, self.args.alpha, neigh_gen)

			elif self.args.model == 'SAGE':
				agg_sage = MeanAggregator(features, cuda=self.args.cuda)
				enc_sage = Encoder(features, self.dataset['feat_data'].shape[1], self.args.emb_size,
								   self.dataset['adj_lists'], agg_sage, gcn=False, cuda=self.args.cuda)
				enc_sage.num_samples = 5
				self.gnn_model = GraphSage(2, enc_sage)
			elif self.args.model == 'GCN':
				agg_gcn = GCNAggregator(features, cuda=self.args.cuda)
				enc_gcn = GCNEncoder(features, self.dataset['feat_data'].shape[1], self.args.emb_size,
									 self.dataset['adj_lists'], agg_gcn, gcn=True, cuda=self.args.cuda)
				self.gnn_model = GCN(2, enc_gcn)

			# Move model to GPU if available
			if self.args.cuda:
				self.gnn_model.cuda()

			# Load the saved model state
			self.gnn_model.load_state_dict(torch.load(model_path))

			# Set model to evaluation mode
			self.gnn_model.eval()
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, self.gnn_model,
																				 self.args.batch_size, self.args.thres)
			print(f1_mac_test)
			print(auc_test)
			print(gmean_test)
			# Call the visualization method
			self.visualize_embeddings(save_path=save_path, batch_size=batch_size)

		except Exception as e:
			print(f"Error in loading model and visualization: {e}")
			import traceback
			traceback.print_exc()

	def visualize_embeddings(self, save_path=None, batch_size=128):
		"""
		Visualize node embeddings after model training with memory optimization
		Args:
			save_path: Optional path to save the visualization plot
			batch_size: Batch size for processing nodes
		"""
		try:
			if self.gnn_model is None:
				raise ValueError("Model hasn't been trained yet. Please train the model first.")

			print("Generating embeddings for visualization...")

			# Clear any existing matplotlib figures
			plt.close('all')

			# Clear CUDA cache if using GPU
			if self.args.cuda:
				torch.cuda.empty_cache()

			# Set model to evaluation mode
			self.gnn_model.eval()
			features = self.dataset['feat_data']
			labels = self.dataset['labels']

			# Convert labels to numpy array if needed
			if isinstance(labels, list):
				labels = np.array(labels)

			# Get all node indices
			all_nodes = np.arange(len(features))
			num_nodes = len(all_nodes)

			# Process in smaller chunks for t-SNE
			# chunk_size = min(10000, num_nodes)  # Limit maximum nodes for t-SNE
			chunk_size = min(20000, num_nodes)
			if num_nodes > chunk_size:
				# Randomly sample nodes if too many
				np.random.seed(42)
				selected_indices = np.random.choice(num_nodes, chunk_size, replace=False)
				selected_indices = np.sort(selected_indices)  # Sort indices to maintain order
				all_nodes = all_nodes[selected_indices]
				labels = labels[selected_indices]
				num_nodes = chunk_size
				print(f"Sampling {chunk_size} nodes for visualization...")

			# Get embedding dimension from model
			test_batch = all_nodes[:min(batch_size, len(all_nodes))]
			try:
				if self.args.model == 'PCGNN':
					test_labels = labels[:len(test_batch)]  # Fix: Use correct indexing
					if self.args.cuda:
						test_labels = torch.LongTensor(test_labels).cuda()
					else:
						test_labels = torch.LongTensor(test_labels)
					with torch.no_grad():
						_, _, _, _, _, _, test_emb = self.gnn_model(test_batch, Variable(test_labels))
				else:
					with torch.no_grad():
						test_emb = self.gnn_model.forward(test_batch)
				emb_dim = test_emb.shape[1]
			except Exception as e:
				print(f"Error getting embedding dimension: {e}")
				print(f"test_batch shape: {len(test_batch)}")
				print(f"labels shape: {len(labels)}")
				raise e

			# Initialize embeddings array
			all_embeddings = np.zeros((num_nodes, emb_dim), dtype=np.float32)

			# Get embeddings batch by batch
			print("Computing embeddings...")
			with torch.no_grad():
				for i in range(0, num_nodes, batch_size):
					batch_nodes = all_nodes[i:min(i + batch_size, num_nodes)]
					try:
						if self.args.model == 'PCGNN':
							batch_labels = labels[i:i + len(batch_nodes)]  # Fix: Use correct indexing
							if self.args.cuda:
								batch_labels = torch.LongTensor(batch_labels).cuda()
							else:
								batch_labels = torch.LongTensor(batch_labels)
							_, _, _, _, _, _, emb = self.gnn_model(batch_nodes, Variable(batch_labels))
						else:
							emb = self.gnn_model.forward(batch_nodes)

						# Move to CPU and convert to numpy
						if self.args.cuda:
							emb = emb.cpu()
						all_embeddings[i:i + len(batch_nodes)] = emb.numpy().astype(np.float32)

						# Clear GPU memory
						if self.args.cuda:
							del emb
							torch.cuda.empty_cache()

					except Exception as e:
						print(f"Error processing batch {i // batch_size}: {e}")
						print(f"Batch size: {len(batch_nodes)}")
						print(f"Labels shape: {len(labels)}")
						print(f"Current index: {i}")
						raise e

			print("Applying t-SNE dimensionality reduction...")
			try:
				# Apply t-SNE with lower perplexity
				tsne = TSNE(n_components=2,
							random_state=42,
							perplexity=min(30, num_nodes - 1),
							n_iter=1000,
							verbose=1)
				embeddings_2d = tsne.fit_transform(all_embeddings)

				# Clear memory
				del all_embeddings
				if self.args.cuda:
					torch.cuda.empty_cache()

			except Exception as e:
				print(f"Error in t-SNE: {e}")
				raise e

			print("Creating visualization...")
			try:
				# Create new figure
				plt.figure(figsize=(12, 10))

				# Create scatter plot
				scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
									  c=labels, cmap='coolwarm', alpha=0.6,
									  s=50)

				# plt.colorbar(scatter, label='Class')
				# plt.title(f"Node Embeddings Visualization ({self.args.model})\nDataset: {self.args.data_name}")
				# plt.xlabel("t-SNE Dimension 1")
				# plt.ylabel("t-SNE Dimension 2")
				# Adjust tick font size
				plt.tick_params(axis='both', which='major', labelsize=25)  # 设置刻度字体大小为14
				plt.tick_params(axis='both', which='minor', labelsize=12)  # 可选：设置次级刻度字体大小为12
				# Add legend
				# unique_labels = np.unique(labels)
				# legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
				# 							  markerfacecolor=scatter.cmap(scatter.norm(label)),
				# 							  label=f'Class {label}', markersize=10)
				# 				   for label in unique_labels]
				# plt.legend(handles=legend_elements)

				# Save or show plot
				if save_path:
					plt.savefig(save_path, bbox_inches='tight', dpi=300)
					print(f"Visualization saved to {save_path}")
				else:
					plt.show()

			except Exception as e:
				print(f"Error in plotting: {e}")
				raise e

			finally:
				plt.close('all')
				if self.args.cuda:
					torch.cuda.empty_cache()

		except Exception as e:
			print(f"Error in visualization process: {e}")
			import traceback
			traceback.print_exc()

		print("Visualization process completed!")
