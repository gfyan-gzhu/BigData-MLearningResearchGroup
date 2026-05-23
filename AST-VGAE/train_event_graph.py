#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import argparse
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from datetime import datetime

# 导入集成后的模型
from ast_vgae import Model
from event_graph_dataset import generate_dataset
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='specify device')

    # ===== dataset parameters =====
    parser.add_argument('--dataset_path', type=str, default='data/mooc.csv', help='path to JODIE-style csv dataset')
    parser.add_argument('--snap_size', type=int, default=10000, help='number of interactions per snapshot')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio of snapshots for training')
    
    # ===== training parameters =====
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    # ===== hyper-parameters =====
    parser.add_argument('--eps', type=float, default=1e-6, help='eps for numerical stability')
    parser.add_argument('--layer_num', type=int, default=1, help='rnn layers')
    parser.add_argument('--h_dim', type=int, default=64, help='hidden channels')
    parser.add_argument('--z_dim', type=int, default=64, help='latent channels')
    
    # 这些参数将根据数据集自动更新
    parser.add_argument('--x_dim', type=int, default=64, help='input channels (will be auto-updated)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    SEED = 100
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    args = args_parser()

    print(f"Loading dataset from {args.dataset_path} ...")
    if not os.path.exists(args.dataset_path):
        print(f"Error: {args.dataset_path} not found. Please ensure the file is in the correct path.")
        exit()

    data_list, train_size, feat_dim = generate_dataset(
        file_path=args.dataset_path,
        device=args.device,
        snap_size=args.snap_size,
        train_ratio=args.train_ratio,
        undirected=True,
        sort_by_time=True,
        n2v_dim=64,
        n2v_walk_length=20,
        n2v_context_size=10,
        n2v_walks_per_node=10,
        n2v_negative_samples=1,
        n2v_p=1.0,
        n2v_q=1.0,
        n2v_lr=0.01,
        n2v_epochs=50,
        n2v_batch_size=256,
        anomaly_ratio=0.01,
        context_k=50,
        struct_m=5,
        struct_p=0.0,
        seed=72,
        cache_root="cache",
        use_cache=True
    )
    args.x_dim = feat_dim
    
    args.x_dim = feat_dim # 自动更新特征维度
    print(f"Dataset loaded. Snapshots: {len(data_list)}, Train: {train_size}, Feature Dim: {feat_dim}")

    # 2. 初始化模型
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 3. 数据集划分: 70%训练, 15%验证, 15%测试
    total_size = len(data_list)
    train_end = train_size  # 70%
    val_start = train_end
    val_end = train_end + int(0.15 * total_size)  # 验证集15%
    test_start = val_end  # 测试集剩余15%
    
    data_train = data_list[:train_end]
    data_val = data_list[val_start:val_end]
    data_test = data_list[test_start:]
    
    print(f"Data split - Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}")
    
    max_auc = 0.0
    best_model_state = None
    best_val_auc = 0.0
    
    print("Starting training ...")
    for epoch in tqdm(range(args.epochs)):
        model.train()
        # 前向传播计算损失
        struct_loss, attr_loss, gen_loss, kl_loss, mem_loss, _, h_t, _ = model(data_train)

        loss = gen_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        
        # 4. 验证
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # 验证集评估
                _, _, _, _, _, _, _, val_score_list = model(data_val, h_t=h_t)
                
                val_scores = []
                val_labels = []
                
                for t in range(len(val_score_list)):
                    score = val_score_list[t].cpu().numpy().squeeze()
                    label = data_val[t].y.cpu().numpy()
                    mask = label > -1

                    if mask.any():
                        val_scores.append(score[mask])
                        val_labels.append(label[mask])

                
                if val_scores:
                    val_flat_scores = np.concatenate(val_scores)
                    val_flat_labels = np.concatenate(val_labels)
                    
                    # 检查标签是否包含两类
                    if len(np.unique(val_flat_labels)) > 1:
                        current_val_auc = roc_auc_score(val_flat_labels, val_flat_scores)
                        
                        # 保存最佳验证模型
                        if current_val_auc > best_val_auc:
                            best_val_auc = current_val_auc
                            best_model_state = model.state_dict().copy()
                            
                            # 保存最佳模型到文件
                            if not os.path.exists('saved_models'):
                                os.makedirs('saved_models')
                            
                            # 添加当前时间戳
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_path = f'saved_models/best_model_{timestamp}_epoch_{epoch}_val_auc_{current_val_auc:.4f}.pth'
                            torch.save({
                                'model_state_dict': best_model_state,
                                'epoch': epoch,
                                'val_auc': current_val_auc,
                                'args': args
                            }, model_path)
                            
                            print(f" Epoch {epoch}: New best validation AUC = {current_val_auc:.4f}, model saved to {model_path}")
                        
                        print(f" Epoch {epoch}: Val AUC = {current_val_auc:.4f} (Best: {best_val_auc:.4f})")
                    else:
                        print(f" Epoch {epoch}: Validation labels only contain one class, skipping AUC.")
    
    # 5. 使用最佳验证模型进行测试
    if best_model_state is not None:
        # 加载最佳验证模型
        model.load_state_dict(best_model_state)
        model.eval()
        
        print("Testing with the best validation model...")
        with torch.no_grad():
            _, _, _, _, _, _, _, test_score_list = model(data_test, h_t=h_t)
        
        test_scores = []
        test_labels = []
        
        for t in range(len(test_score_list)):
            score = test_score_list[t].cpu().numpy().squeeze()
            label = data_test[t].y.cpu().numpy()
            mask = label > -1

            if mask.any():
                test_scores.append(score[mask])
                test_labels.append(label[mask])

        if test_scores:
            test_flat_scores = np.concatenate(test_scores)
            test_flat_labels = np.concatenate(test_labels)
            
            # 检查标签是否包含两类
            if len(np.unique(test_flat_labels)) > 1:
                final_test_auc = roc_auc_score(test_flat_labels, test_flat_scores)
                print(f"Final Test AUC with best validation model: {final_test_auc:.4f}")
            else:
                print("Test labels only contain one class, skipping AUC.")
    else:
        print("No validation AUC was computed.")
    
    print(f"Training finished. Best Validation AUC: {best_val_auc:.4f}")

