import os
import copy
import numpy as np
import random
import argparse
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from datetime import datetime

from ast_vgae_dblp import Model
from dynamic_graph_dataset import generate_dataset, inject_test_anomalies


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='specify device')
    parser.add_argument('--dataset_path', type=str, default='data/DBLP3.npz', help='path to dataset file, csv or STAR npz')
    parser.add_argument('--snap_size', type=int, default=10000, help='number of interactions per snapshot for csv datasets')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio of snapshots for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of snapshots for validation')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--eps', type=float, default=1e-6, help='eps for numerical stability')
    parser.add_argument('--layer_num', type=int, default=1, help='rnn layers')
    parser.add_argument('--h_dim', type=int, default=64, help='hidden channels')
    parser.add_argument('--z_dim', type=int, default=32, help='latent channels')
    parser.add_argument('--wavelet_C', type=int, default=2, help='beta wavelet bank order')
    parser.add_argument('--undirected', action='store_true', help='make STAR/csv snapshots undirected')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='test-set anomaly ratio per snapshot')
    parser.add_argument('--struct_clique_size', type=int, default=5, help='Np in structural anomaly injection')
    parser.add_argument('--attr_candidate_k', type=int, default=50, help='k for attribute anomaly injection')
    parser.add_argument('--random_seed', type=int, default=7, help='random seed')
    return parser.parse_args()

def compute_auc(score_list, data_split):
    scores, labels = [], []
    for t in range(len(score_list)):
        score = score_list[t].detach().cpu().numpy().squeeze()
        label = data_split[t].y.detach().cpu().numpy()
        if len(np.unique(label)) <= 2:
            scores.append(score)
            labels.append(label)
    if not scores:
        return None
    flat_scores = np.concatenate(scores)
    flat_labels = np.concatenate(labels)
    if len(np.unique(flat_labels)) <= 1:
        return None
    return roc_auc_score(flat_labels, flat_scores)


if __name__ == '__main__':
    args = args_parser()
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Loading dataset from {args.dataset_path} ...")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(args.dataset_path)

    data_list, train_size, feat_dim = generate_dataset(
        args.dataset_path,
        args.device,
        args.snap_size,
        args.train_ratio,
        undirected=args.undirected,
    )
    args.x_dim = feat_dim
    print(f"Dataset loaded. Snapshots: {len(data_list)}, Train: {train_size}, Feature Dim: {feat_dim}")

    total_size = len(data_list)
    train_end = train_size
    val_size = max(1, int(round(args.val_ratio * total_size)))
    val_start = train_end
    val_end = min(total_size, train_end + val_size)
    if val_end >= total_size:
        val_end = max(train_end + 1, total_size - 1)
    test_start = val_end

    #anomalies are injected only in the testing phase.
    data_list = inject_test_anomalies(
        data_list,
        test_start=val_start,
        anomaly_ratio=args.anomaly_ratio,
        struct_clique_size=args.struct_clique_size,
        attr_candidate_k=args.attr_candidate_k,
        random_seed=args.random_seed,
    )

    data_train = data_list[:train_end]
    data_val = data_list[val_start:val_end]
    data_test = data_list[test_start:]
    print(f"Data split - Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}")
    print(
        f"anomaly injection on TEST only: anomaly_ratio={args.anomaly_ratio}, "
        f"struct_clique_size={args.struct_clique_size}, attr_candidate_k={args.attr_candidate_k}"
    )

    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_model_state = None
    best_val_auc = -1.0
    best_val_loss = float('inf')

    print("Starting training ...")
    for epoch in tqdm(range(args.epochs)):
        model.train()
        struct_loss, attr_loss, gen_loss, kl_loss, _, _, h_t, _ = model(data_train)
        loss = gen_loss + kl_loss
        print(
            f"Epoch {epoch}: loss={loss.item():.4f}, gen={gen_loss.item():.4f}, "
            f"kl={kl_loss.item():.4f}, struct={struct_loss.item():.4f}, attr={attr_loss.item():.4f}"
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_struct_loss, val_attr_loss, val_gen_loss, val_kl_loss, _, _, _, val_score_list = model(data_val, h_t=h_t)
            current_val_auc = compute_auc(val_score_list, data_val)
            current_val_loss = (10 * val_gen_loss + val_kl_loss).item()

            improved = False
            if current_val_auc is not None:
                print(f"Epoch {epoch}: Val AUC = {current_val_auc:.4f}")
                if current_val_auc > best_val_auc:
                    best_val_auc = current_val_auc
                    improved = True
            else:
                print(f"Epoch {epoch}: Val AUC unavailable, fallback to val loss = {current_val_loss:.4f}")
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    improved = True

            if improved:
                best_model_state = copy.deepcopy(model.state_dict())
                os.makedirs('saved_models', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch,
                    'best_val_auc': best_val_auc,
                    'best_val_loss': best_val_loss,
                    'args': vars(args),
                }, f'saved_models/best_model_{timestamp}.pth')
                print(f"Epoch {epoch}: saved new best model")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    print("Testing with best model ...")
    with torch.no_grad():
        _, _, _, _, _, _, _, test_score_list = model(data_test, h_t=None)
    test_auc = compute_auc(test_score_list, data_test)
    if test_auc is None:
        print("Test AUC unavailable under current labels.")
    else:
        print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Training finished. Best Validation AUC: {best_val_auc:.4f}")
