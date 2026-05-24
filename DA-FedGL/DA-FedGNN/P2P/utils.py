"""
通用工具函数模块
包含结果保存、指标聚合、最终报告和可视化等公共功能。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation import ModelEvaluator

def evaluate_all_clients_enhanced(clients):
    """
    增强版客户端评估，包含所有指标
    ✅ 重构: 使用 ModelEvaluator.evaluate_all_clients
    """
    return ModelEvaluator.evaluate_all_clients(clients)


def aggregate_client_metrics(client_metrics, round_num, alpha_value=None):
    """聚合客户端指标"""
    accuracies = [m['accuracy'] for m in client_metrics]
    f1_scores = [m['f1_macro'] for m in client_metrics]
    recalls = [m['recall_macro'] for m in client_metrics]
    precisions = [m['precision_macro'] for m in client_metrics]
    losses = [m['loss'] for m in client_metrics]

    aggregated = {
        'round': round_num,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'max_accuracy': np.max(accuracies),
        'min_accuracy': np.min(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses)
    }

    if alpha_value is not None:
        aggregated['alpha'] = alpha_value
    else:
        # 对于自适应α，记录分布
        alphas = [m['alpha'] for m in client_metrics if m['alpha'] is not None]
        if alphas:
            aggregated['mean_alpha'] = np.mean(alphas)
            aggregated['std_alpha'] = np.std(alphas)
            aggregated['min_alpha'] = np.min(alphas)
            aggregated['max_alpha'] = np.max(alphas)

    return aggregated


def generate_comparison_report(all_results, all_histories, all_detailed, output_path):
    """生成综合对比报告和可视化"""

    # 创建输出目录
    os.makedirs(os.path.join(output_path, 'alpha_comparison'), exist_ok=True)

    # 1. 性能对比表
    comparison_df = pd.DataFrame()
    for algo_name, results in all_results.items():
        if isinstance(results, pd.DataFrame):
            avg_metrics = {
                'Algorithm': algo_name,
                'Accuracy': results['accuracy'].mean() if 'accuracy' in results else 0,
                'F1-Score': results['f1_macro'].mean() if 'f1_macro' in results else 0,
                'Recall': results['recall_macro'].mean() if 'recall_macro' in results else 0,
                'Precision': results['precision_macro'].mean() if 'precision_macro' in results else 0
            }
            comparison_df = comparison_df.append(avg_metrics, ignore_index=True)

    comparison_df.to_csv(os.path.join(output_path, 'alpha_comparison', 'performance_comparison.csv'))

    # 2. 创建可视化
    create_alpha_comparison_plots(all_histories, all_detailed, output_path)

    # 3. 统计分析
    perform_statistical_analysis(all_results, all_detailed, output_path)

    print("\n对比报告生成完成！")
    print(f"结果保存在: {os.path.join(output_path, 'alpha_comparison')}")


def create_alpha_comparison_plots(all_histories, all_detailed, output_path):
    """创建α对比相关的可视化"""

    plt.style.use('seaborn-v0_8')
    fig_path = os.path.join(output_path, 'alpha_comparison', 'figures')
    os.makedirs(fig_path, exist_ok=True)

    # 1. 性能对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    algorithms = list(all_histories.keys())
    metrics_to_plot = ['mean_accuracy', 'mean_f1', 'mean_recall', 'mean_precision']
    metric_names = ['Accuracy', 'F1-Score', 'Recall', 'Precision']

    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]

        final_values = []
        errors = []

        for algo in algorithms:
            if algo in all_histories and all_histories[algo]:
                final_round = all_histories[algo][-1]
                final_values.append(final_round.get(metric, 0))
                errors.append(final_round.get(metric.replace('mean', 'std'), 0))
            else:
                final_values.append(0)
                errors.append(0)

        bars = ax.bar(algorithms, final_values, yerr=errors, capsize=5)
        ax.set_title(f'{name} Comparison')
        ax.set_ylabel(name)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')

        # 标注数值
        for bar, val in zip(bars, final_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'performance_comparison.png'), dpi=300)
    plt.close()

    # 2. 损失曲线对比
    fig, ax = plt.subplots(figsize=(12, 6))

    for algo, history in all_histories.items():
        if history:
            rounds = [h['round'] for h in history]
            losses = [h['mean_loss'] for h in history]
            ax.plot(rounds, losses, marker='o', label=algo, linewidth=2)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'loss_curves.png'), dpi=300)
    plt.close()

    # 3. α值演化热力图（仅DaFedGNN）
    if 'DaFedGNN-Adaptive' in all_detailed:
        create_alpha_evolution_heatmap(all_detailed['DaFedGNN-Adaptive'], fig_path)

    print("可视化图表已生成")


def create_alpha_evolution_heatmap(detailed_metrics, fig_path):
    """创建α值演化热力图"""

    # 提取α值数据
    alpha_data = {}
    for record in detailed_metrics:
        if record.get('alpha') is not None:
            client_id = record['client_id']
            round_num = record['round']

            if client_id not in alpha_data:
                alpha_data[client_id] = {}
            alpha_data[client_id][round_num] = record['alpha']

    if not alpha_data:
        return

    # 创建矩阵
    clients = sorted(alpha_data.keys())
    rounds = sorted(set(r for c_data in alpha_data.values() for r in c_data.keys()))

    matrix = np.zeros((len(clients), len(rounds)))
    for i, client in enumerate(clients):
        for j, round_num in enumerate(rounds):
            matrix[i, j] = alpha_data.get(client, {}).get(round_num, 0.5)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1,
                xticklabels=rounds, yticklabels=[f'Client {c}' for c in clients],
                cbar_kws={'label': 'α value'})

    ax.set_title('α Value Evolution Across Clients (DaFedGNN-Adaptive)')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Client')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'alpha_evolution_heatmap.png'), dpi=300)
    plt.close()


def save_experiment_results(performance_history, detailed_metrics, algorithm_name, output_path):
    """保存实验结果"""

    results_path = os.path.join(output_path, 'alpha_comparison', 'raw_data')
    os.makedirs(results_path, exist_ok=True)

    # 保存性能历史
    if performance_history:
        pd.DataFrame(performance_history).to_csv(
            os.path.join(results_path, f'{algorithm_name}_performance.csv'),
            index=False
        )

    # 保存详细指标
    if detailed_metrics:
        pd.DataFrame(detailed_metrics).to_csv(
            os.path.join(results_path, f'{algorithm_name}_detailed.csv'),
            index=False
        )


def perform_statistical_analysis(all_results, all_detailed, output_path):
    """执行统计分析"""
    from scipy import stats

    analysis_results = []

    # 对比DaFedGNN与固定α方法
    if 'DaFedGNN-Adaptive' in all_results:
        dafedgnn_accs = all_results['DaFedGNN-Adaptive']['accuracy'].values if isinstance(
            all_results['DaFedGNN-Adaptive'], pd.DataFrame) else []

        for algo in ['APFL-0.2', 'APFL-0.5', 'APFL-0.8']:
            if algo in all_results:
                fixed_accs = all_results[algo]['accuracy'].values if isinstance(
                    all_results[algo], pd.DataFrame) else []

                if len(dafedgnn_accs) > 0 and len(fixed_accs) > 0:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(dafedgnn_accs, fixed_accs)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(dafedgnn_accs) ** 2 + np.std(fixed_accs) ** 2) / 2)
                    cohens_d = (np.mean(dafedgnn_accs) - np.mean(fixed_accs)) / pooled_std if pooled_std > 0 else 0

                    analysis_results.append({
                        'Comparison': f'DaFedGNN vs {algo}',
                        'DaFedGNN_Mean': np.mean(dafedgnn_accs),
                        f'{algo}_Mean': np.mean(fixed_accs),
                        'Difference': np.mean(dafedgnn_accs) - np.mean(fixed_accs),
                        'T-statistic': t_stat,
                        'P-value': p_value,
                        'Cohens_d': cohens_d,
                        'Significant': 'Yes' if p_value < 0.05 else 'No'
                    })

    # 保存统计分析结果
    if analysis_results:
        stats_df = pd.DataFrame(analysis_results)
        stats_df.to_csv(os.path.join(output_path, 'alpha_comparison', 'statistical_analysis.csv'), index=False)

        print("\n统计分析结果：")
        print(stats_df.to_string())


def save_results(frame, outpath, algorithm_name, args):
    """Save experiment results"""
    if args.repeat is None:
        outfile = os.path.join(outpath, f'{algorithm_name}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_{algorithm_name}.csv')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    frame.to_csv(outfile)
    print(f"  Results saved to: {outfile}")

def save_local_loss_history(frame, outpath, algorithm_name, args):
    """保存本地训练 loss 历史。"""
    if frame is None or frame.empty:
        return

    if args.repeat is None:
        outfile = os.path.join(outpath, f'{algorithm_name}_local_loss.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_{algorithm_name}_local_loss.csv')

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    frame.to_csv(outfile, index=False)
    print(f"  Local loss history saved to: {outfile}")


def generate_final_report(all_results, outpath):
    """Generate final comparison report (修复: 兼容两种列名格式)"""
    print("\n" + "=" * 80)
    print("GENERATING FINAL COMPARISON REPORT")
    print("=" * 80)

    # Collect final performance from all algorithms
    summary = []
    for algo_name, results in all_results.items():
        if isinstance(results, pd.DataFrame) and not results.empty:
            # Get last row (final performance)
            final_row = results.iloc[-1]

            # ✅ 修复: 兼容两种列名格式 (mean_accuracy 和 accuracy)
            avg_acc = final_row.get('mean_accuracy', final_row.get('accuracy', 0))
            std_acc = final_row.get('std_accuracy', 0)
            avg_f1 = final_row.get('mean_f1', final_row.get('f1_macro', 0))
            std_f1 = final_row.get('std_f1', 0)
            avg_recall = final_row.get('mean_recall', final_row.get('recall_macro', 0))
            std_recall = final_row.get('std_recall', 0)
            avg_loss = final_row.get('mean_loss', final_row.get('loss', 0))
            final_alpha = final_row.get('alpha', final_row.get('mean_alpha',
                                                               final_row.get('final_alpha', 'N/A')))

            summary.append({
                'Algorithm': algo_name,
                'Avg_Accuracy': avg_acc,
                'Std_Accuracy': std_acc,
                'Avg_F1': avg_f1,
                'Std_F1': std_f1,
                'Avg_Recall': avg_recall,
                'Std_Recall': std_recall,
                'Avg_Loss': avg_loss,
                'Final_Alpha': final_alpha
            })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Avg_Accuracy', ascending=False)

    # Print summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print("-" * 80)

    # Save summary
    summary_path = os.path.join(outpath, 'final_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")

    # Find best algorithm
    best_algo = summary_df.iloc[0]
    print(f"\n🏆 Best Algorithm: {best_algo['Algorithm']}")
    print(f"   - Accuracy: {best_algo['Avg_Accuracy']:.4f} ± {best_algo['Std_Accuracy']:.4f}")
    print(f"   - F1-Score: {best_algo['Avg_F1']:.4f} ± {best_algo['Std_F1']:.4f}")
    print(f"   - Recall: {best_algo['Avg_Recall']:.4f} ± {best_algo['Std_Recall']:.4f}")
