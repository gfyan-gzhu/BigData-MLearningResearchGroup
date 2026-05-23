import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import logging
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """分析配置类"""
    raw_data_dir: Optional[str] = None
    output_dir: str = 'analysis_results'
    baseline_algorithm: str = 'selftrain'
    metric_column: str = 'test_acc'
    confidence_level: float = 0.95
    max_clients_heatmap: int = 20
    figure_dpi: int = 300
    color_palette: str = 'Set2'


class FederatedLearningAnalyzer:
    """
    联邦学习算法性能分析器

    功能特性:
    - 多次实验数据聚合与统计分析
    - 算法性能对比与可视化
    - 统计显著性检验
    - 详细报告生成
    - 交互式图表
    """

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.aggregated_data: Dict[str, pd.DataFrame] = {}
        self.comparison_results: Dict[str, Dict] = {}
        self.statistics: Dict[str, Any] = {}

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette(self.config.color_palette)

    def discover_algorithms(self, data_dir: str) -> List[str]:
        """智能发现数据目录中的算法"""
        logger.info(f"扫描目录寻找算法: {data_dir}")

        patterns = [
            '*_accuracy_*_GC.csv',  # 标准格式
            '*_accuracy_*.csv',  # 简化格式
            'avg_accuracy_*.csv'  # 已聚合格式
        ]

        algorithms = set()

        for pattern in patterns:
            files = glob.glob(os.path.join(data_dir, pattern))
            for file in files:
                filename = Path(file).stem

                # 提取算法名称的多种模式
                if '_accuracy_' in filename:
                    if filename.startswith('avg_accuracy_'):
                        # avg_accuracy_dafedgnn.csv -> dafedgnn
                        algo = filename.replace('avg_accuracy_', '')
                    else:
                        # 1_accuracy_dafedgnn_GC.csv -> dafedgnn
                        parts = filename.split('_accuracy_')[1]
                        algo = parts.replace('_GC', '').replace('.csv', '')
                    algorithms.add(algo)

        algorithms = sorted(list(algorithms))
        logger.info(f"发现 {len(algorithms)} 个算法: {algorithms}")
        return algorithms

    def aggregate_experiment_data(self, data_dir: str, algorithm: str) -> Optional[pd.DataFrame]:
        """聚合单个算法的多次实验数据 - 修正版本"""
        logger.info(f"聚合算法 {algorithm} 的实验数据")

        # 寻找实验文件
        experiment_files = []

        # 模式1: 编号_accuracy_algorithm_GC.csv
        for exp_num in range(1, 21):  # 支持更多实验次数
            patterns = [
                f"{exp_num}_accuracy_{algorithm}_GC.csv",
                f"{exp_num}_accuracy_{algorithm}.csv",
                f"exp{exp_num}_accuracy_{algorithm}.csv"
            ]

            for pattern in patterns:
                file_path = os.path.join(data_dir, pattern)
                if os.path.exists(file_path):
                    experiment_files.append(file_path)
                    break

        if not experiment_files:
            logger.warning(f"未找到算法 {algorithm} 的实验文件")
            return None

        # 读取并聚合数据
        dfs = []
        for file_path in experiment_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.debug(f"读取: {Path(file_path).name}, 客户端数: {len(df)}")
            except Exception as e:
                logger.error(f"读取文件错误 {file_path}: {e}")

        if not dfs:
            return None

        # 统计聚合 - 确保按客户端维度聚合
        if len(dfs) == 1:
            result_df = dfs[0].copy()
            logger.info(f"算法 {algorithm} 仅有1次实验，直接使用原数据")
        else:
            # 确保正确的维度聚合
            logger.info(f"算法 {algorithm} 有 {len(dfs)} 次重复实验，开始客户端级别聚合")

            # 检查所有实验的客户端数是否一致
            client_counts = [len(df) for df in dfs]
            if len(set(client_counts)) > 1:
                logger.warning(f"不同实验的客户端数不一致: {client_counts}")
                # 取最小客户端数进行聚合
                min_clients = min(client_counts)
                dfs = [df.iloc[:min_clients] for df in dfs]
                logger.info(f"统一使用前 {min_clients} 个客户端进行聚合")

            # 初始化结果DataFrame
            result_df = dfs[0].copy()
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns

            # 对每个数值列进行客户端级别的聚合
            for col in numeric_cols:
                # 构造形状为 (实验次数, 客户端数) 的矩阵
                values_matrix = np.array([df[col].values for df in dfs])  # shape: (n_experiments, n_clients)

                # 按客户端维度计算统计量（axis=0表示对实验次数这个维度求统计）
                result_df[col] = np.mean(values_matrix, axis=0)  # 每个客户端的平均准确率
                result_df[f'{col}_std'] = np.std(values_matrix, axis=0, ddof=1)  # 每个客户端的标准差
                result_df[f'{col}_sem'] = np.std(values_matrix, axis=0, ddof=1) / np.sqrt(len(dfs))  # 标准误

                # 置信区间
                try:
                    from scipy import stats
                    t_critical = stats.t.ppf((1 + self.config.confidence_level) / 2, len(dfs) - 1)
                    margin_error = t_critical * result_df[f'{col}_sem']
                    result_df[f'{col}_ci_lower'] = result_df[col] - margin_error
                    result_df[f'{col}_ci_upper'] = result_df[col] + margin_error
                except ImportError:
                    logger.warning("scipy未安装，跳过置信区间计算")

            logger.info(f"算法 {algorithm} 聚合完成：{len(dfs)} 次实验 × {len(result_df)} 个客户端")

            # 输出聚合后的整体统计信息
            overall_mean = result_df[self.config.metric_column].mean()
            overall_std = result_df[self.config.metric_column].std()
            logger.info(f"算法 {algorithm} 整体平均准确率: {overall_mean:.4f} ± {overall_std:.4f}")

        return result_df

    def aggregate_all_results(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """聚合所有算法的实验结果"""
        algorithms = self.discover_algorithms(data_dir)
        results = {}

        for algorithm in algorithms:
            df = self.aggregate_experiment_data(data_dir, algorithm)
            if df is not None:
                results[algorithm] = df

        logger.info(f"总共聚合了 {len(results)} 个算法的数据")
        return results

    def save_aggregated_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """保存聚合结果 - 修正版本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存各算法详细结果
        for algorithm, df in results.items():
            output_file = os.path.join(self.config.output_dir, f'avg_accuracy_{algorithm}.csv')
            df.to_csv(output_file, index=False)
            logger.info(f"保存: {output_file}")

        # 创建算法性能汇总 - 修改计算逻辑
        summary_data = []
        for algorithm, df in results.items():
            if self.config.metric_column in df.columns and len(df) > 0:
                metric_col = self.config.metric_column
                std_col = f'{metric_col}_std'

                # 修改：计算所有客户端准确率的平均值，而不是最后一次的准确率
                final_accuracy = df[metric_col].mean()  # 所有客户端准确率的平均值

                # 计算标准差（如果有多次实验的话）
                if std_col in df.columns:
                    # 使用各客户端标准差的平均值作为整体标准差的估计
                    avg_std = df[std_col].mean()
                else:
                    # 如果没有标准差列，使用准确率的标准差
                    avg_std = df[metric_col].std()

                summary_entry = {
                    'Algorithm': algorithm,
                    'Final_Accuracy': final_accuracy,  # 现在是所有客户端的平均准确率
                    'Average_Accuracy': final_accuracy,  # 保持一致
                    'Std_Accuracy': avg_std,  # 标准差
                    'Min_Accuracy': df[metric_col].min(),  # 最低客户端准确率
                    'Max_Accuracy': df[metric_col].max(),  # 最高客户端准确率
                    'Clients_Count': len(df)  # 客户端数量
                }
                summary_data.append(summary_entry)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Final_Accuracy', ascending=False)

            summary_file = os.path.join(self.config.output_dir, f'algorithm_summary_{timestamp}.csv')
            summary_df.to_csv(summary_file, index=False)

            print("\n" + "=" * 80)
            print("算法性能排行 (按所有客户端平均准确率)")  # 更新标题说明
            print("=" * 80)
            for i, (_, row) in enumerate(summary_df.iterrows(), 1):
                print(f"{i:2d}. {row['Algorithm']:15s}: {row['Final_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
            print("=" * 80)

    def load_aggregated_data(self, data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """加载已聚合的数据"""
        if data_dir is None:
            data_dir = self.config.output_dir

        data = {}
        csv_files = glob.glob(os.path.join(data_dir, 'avg_accuracy_*.csv'))

        if not csv_files:
            logger.warning(f"在 {data_dir} 中未找到聚合数据文件")
            return data

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                algo_name = Path(file_path).stem.replace('avg_accuracy_', '')
                data[algo_name] = df
                logger.info(f"加载 {algo_name}: {df.shape}")
            except Exception as e:
                logger.error(f"加载错误 {file_path}: {e}")

        return data

    def statistical_comparison(self, data: Dict[str, pd.DataFrame],
                               baseline_key: str) -> Dict[str, Dict]:
        """执行统计显著性检验的算法对比"""
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy未安装，跳过统计检验，仅进行基础对比")
            return self._basic_comparison(data, baseline_key)

        if baseline_key not in data:
            logger.error(f"基准算法 {baseline_key} 未找到!")
            return {}

        baseline_data = data[baseline_key]
        results = {}
        metric_col = self.config.metric_column

        for algo_name, algo_data in data.items():
            if algo_name == baseline_key:
                continue

            if metric_col not in algo_data.columns or metric_col not in baseline_data.columns:
                logger.warning(f"{metric_col} 列在 {algo_name} 或基准数据中未找到")
                continue

            if len(algo_data) != len(baseline_data):
                logger.warning(f"{algo_name} 与基准数据样本数量不一致")
                continue

            # 计算改进指标
            improvement = algo_data[metric_col] - baseline_data[metric_col]

            # 统计检验
            try:
                t_stat, p_value = stats.ttest_rel(algo_data[metric_col], baseline_data[metric_col])
            except Exception as e:
                logger.warning(f"t检验失败 {algo_name}: {e}")
                t_stat, p_value = 0.0, 1.0

            # Wilcoxon符号秩检验（非参数）
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(improvement)
            except Exception as e:
                logger.warning(f"Wilcoxon检验失败 {algo_name}: {e}")
                wilcoxon_stat, wilcoxon_p = None, None

            # 效应量 (Cohen's d)
            try:
                pooled_std = np.sqrt((algo_data[metric_col].var() + baseline_data[metric_col].var()) / 2)
                cohens_d = improvement.mean() / pooled_std if pooled_std > 0 else 0.0
            except Exception as e:
                logger.warning(f"效应量计算失败 {algo_name}: {e}")
                cohens_d = 0.0

            # 置信区间
            try:
                sem = improvement.std() / np.sqrt(len(improvement)) if len(improvement) > 0 else 0
                ci_lower = improvement.mean() - 1.96 * sem
                ci_upper = improvement.mean() + 1.96 * sem
            except Exception as e:
                logger.warning(f"置信区间计算失败 {algo_name}: {e}")
                ci_lower = ci_upper = improvement.mean()

            # 统计结果
            results[algo_name] = {
                'improved_clients': int((improvement > 0).sum()),
                'degraded_clients': int((improvement < 0).sum()),
                'unchanged_clients': int((improvement == 0).sum()),
                'total_clients': int(len(algo_data)),
                'avg_improvement': float(improvement.mean()),
                'std_improvement': float(improvement.std()),
                'max_improvement': float(improvement.max()),
                'min_improvement': float(improvement.min()),
                'median_improvement': float(improvement.median()),
                'improvement_values': improvement.values.astype(float),

                # 统计检验结果
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),
                'wilcoxon_statistic': float(wilcoxon_stat) if wilcoxon_stat is not None else None,
                'wilcoxon_p_value': float(wilcoxon_p) if wilcoxon_p is not None else None,
                'cohens_d': float(cohens_d),
                'effect_size': self._interpret_effect_size(cohens_d),

                # 置信区间
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
            }

        return results

    def _basic_comparison(self, data: Dict[str, pd.DataFrame],
                          baseline_key: str) -> Dict[str, Dict]:
        """基础对比分析（当scipy不可用时）"""
        if baseline_key not in data:
            logger.error(f"基准算法 {baseline_key} 未找到!")
            return {}

        baseline_data = data[baseline_key]
        results = {}
        metric_col = self.config.metric_column

        for algo_name, algo_data in data.items():
            if algo_name == baseline_key:
                continue

            if metric_col not in algo_data.columns or metric_col not in baseline_data.columns:
                logger.warning(f"{metric_col} 列在 {algo_name} 或基准数据中未找到")
                continue

            if len(algo_data) != len(baseline_data):
                logger.warning(f"{algo_name} 与基准数据样本数量不一致")
                continue

            # 计算改进指标
            improvement = algo_data[metric_col] - baseline_data[metric_col]

            # 基础统计结果
            results[algo_name] = {
                'improved_clients': int((improvement > 0).sum()),
                'degraded_clients': int((improvement < 0).sum()),
                'unchanged_clients': int((improvement == 0).sum()),
                'total_clients': int(len(algo_data)),
                'avg_improvement': float(improvement.mean()),
                'std_improvement': float(improvement.std()),
                'max_improvement': float(improvement.max()),
                'min_improvement': float(improvement.min()),
                'median_improvement': float(improvement.median()),
                'improvement_values': improvement.values.astype(float),

                # 默认统计检验结果
                't_statistic': 0.0,
                'p_value': 1.0,
                'is_significant': False,
                'wilcoxon_statistic': None,
                'wilcoxon_p_value': None,
                'cohens_d': 0.0,
                'effect_size': 'unknown',

                # 简单置信区间
                'ci_lower': float(improvement.mean() - 1.96 * improvement.std() / np.sqrt(len(improvement))),
                'ci_upper': float(improvement.mean() + 1.96 * improvement.std() / np.sqrt(len(improvement))),
            }

        return results

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """解释效应量大小"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def create_comprehensive_visualizations(self, results: Dict[str, Dict],
                                            baseline_name: str = 'selftrain') -> plt.Figure:
        """创建全面的可视化分析"""
        if not results:
            logger.error("没有结果可以可视化!")
            return None

        # 创建大型图表布局
        fig = plt.figure(figsize=(24, 18))

        algorithms = list(results.keys())
        n_algos = len(algorithms)

        # 1. 性能改进对比 (左上)
        ax1 = plt.subplot(3, 4, 1)
        improved_counts = [results[algo]['improved_clients'] for algo in algorithms]
        degraded_counts = [results[algo]['degraded_clients'] for algo in algorithms]

        x = np.arange(n_algos)
        width = 0.35

        bars1 = ax1.bar(x - width / 2, improved_counts, width, label='Improved',
                        color='#2ecc71', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, degraded_counts, width, label='Degraded',
                        color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Algorithms')
        ax1.set_ylabel('Number of Clients')
        ax1.set_title(f'Client Performance Change vs {baseline_name.title()}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 2. 平均改进与置信区间 (右上)
        ax2 = plt.subplot(3, 4, 2)
        avg_improvements = [results[algo]['avg_improvement'] for algo in algorithms]
        ci_lower = [results[algo]['ci_lower'] for algo in algorithms]
        ci_upper = [results[algo]['ci_upper'] for algo in algorithms]

        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in avg_improvements]
        bars = ax2.bar(algorithms, avg_improvements, color=colors, alpha=0.7)

        # 添加置信区间
        for i, (algo, avg, lower, upper) in enumerate(zip(algorithms, avg_improvements, ci_lower, ci_upper)):
            ax2.plot([i, i], [lower, upper], 'k-', linewidth=2)
            ax2.plot([i - 0.1, i + 0.1], [lower, lower], 'k-', linewidth=2)
            ax2.plot([i - 0.1, i + 0.1], [upper, upper], 'k-', linewidth=2)

        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('Average Accuracy Improvement')
        ax2.set_title(f'Average Performance Improvement (95% CI)')
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)

        # 3. 改进分布箱线图 (左中)
        ax3 = plt.subplot(3, 4, 3)
        improvement_data = [results[algo]['improvement_values'] for algo in algorithms]

        bp = ax3.boxplot(improvement_data, labels=algorithms, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))

        ax3.set_xlabel('Algorithms')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.set_title(f'Distribution of Improvements')
        ax3.set_xticklabels(algorithms, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # 4. 统计显著性 (右中)
        ax4 = plt.subplot(3, 4, 4)
        p_values = [results[algo]['p_value'] for algo in algorithms]
        significance = ['Significant' if p < 0.05 else 'Not Significant' for p in p_values]

        colors = ['#2ecc71' if s == 'Significant' else '#95a5a6' for s in significance]
        bars = ax4.bar(algorithms, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)

        ax4.set_xlabel('Algorithms')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('Statistical Significance Test')
        ax4.set_xticklabels(algorithms, rotation=45, ha='right')
        ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--',
                    label='p=0.05 threshold', alpha=0.8)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 效应量 (左下)
        ax5 = plt.subplot(3, 4, 5)
        effect_sizes = [results[algo]['cohens_d'] for algo in algorithms]

        colors = ['#e74c3c' if abs(d) >= 0.8 else '#f39c12' if abs(d) >= 0.5 else
        '#f1c40f' if abs(d) >= 0.2 else '#95a5a6' for d in effect_sizes]

        bars = ax5.bar(algorithms, effect_sizes, color=colors, alpha=0.7)
        ax5.set_xlabel('Algorithms')
        ax5.set_ylabel("Cohen's d")
        ax5.set_title('Effect Size (Cohen\'s d)')
        ax5.set_xticklabels(algorithms, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 添加效应量解释
        ax5.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small')
        ax5.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium')
        ax5.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large')

        # 6. 改进百分比饼图 (右下)
        ax6 = plt.subplot(3, 4, 6)
        if algorithms:
            total_clients = results[algorithms[0]]['total_clients']
            improvement_percentages = [(results[algo]['improved_clients'] / total_clients) * 100
                                       for algo in algorithms]

            bars = ax6.bar(algorithms, improvement_percentages,
                           color=sns.color_palette("viridis", n_algos), alpha=0.8)
            ax6.set_xlabel('Algorithms')
            ax6.set_ylabel('Percentage of Improved Clients (%)')
            ax6.set_title('Success Rate Comparison')
            ax6.set_xticklabels(algorithms, rotation=45, ha='right')
            ax6.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, val in zip(bars, improvement_percentages):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        # 7-8. 详细统计表格
        ax7 = plt.subplot(3, 4, (7, 8))
        ax7.axis('tight')
        ax7.axis('off')

        # 创建详细统计表
        table_data = []
        headers = ['Algorithm', 'Improved\nClients', 'Avg Improve', 'p-value',
                   'Effect Size', 'Cohen\'s d']

        for algo in algorithms:
            stats = results[algo]
            table_data.append([
                algo,
                f"{stats['improved_clients']}/{stats['total_clients']}",
                f"{stats['avg_improvement']:.4f}",
                f"{stats['p_value']:.4f}",
                stats['effect_size'],
                f"{stats['cohens_d']:.3f}"
            ])

        table = ax7.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax7.set_title('Detailed Statistical Summary', pad=20, fontsize=14, fontweight='bold')

        # 9-12. 改进热力图
        ax9 = plt.subplot(3, 4, (9, 12))
        max_clients = min(self.config.max_clients_heatmap,
                          results[algorithms[0]]['total_clients'])

        heatmap_data = []
        for algo in algorithms:
            heatmap_data.append(results[algo]['improvement_values'][:max_clients])

        heatmap_matrix = np.array(heatmap_data)

        im = ax9.imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto',
                        vmin=-0.1, vmax=0.1)
        ax9.set_yticks(range(len(algorithms)))
        ax9.set_yticklabels(algorithms)
        ax9.set_xticks(range(0, max_clients, max(1, max_clients // 10)))
        ax9.set_xticklabels([f'C{i + 1}' for i in range(0, max_clients, max(1, max_clients // 10))])
        ax9.set_xlabel('Clients')
        ax9.set_ylabel('Algorithms')
        ax9.set_title(f'Client-wise Improvement Heatmap (First {max_clients} Clients)')

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax9, shrink=0.8)
        cbar.set_label('Accuracy Improvement')

        plt.tight_layout()
        return fig

    def generate_detailed_report(self, results: Dict[str, Dict],
                                 baseline_name: str = 'selftrain') -> str:
        """生成详细的分析报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
{'=' * 100}
联邦学习算法性能分析报告
{'=' * 100}
生成时间: {timestamp}
基准算法: {baseline_name.upper()}
评估指标: {self.config.metric_column}
置信水平: {self.config.confidence_level * 100}%
计算方法: 先计算每个客户端多次实验的平均准确率，再计算所有客户端的整体平均
{'=' * 100}

算法性能概览
{'-' * 50}
"""

        # 按平均改进值排序
        sorted_algos = sorted(results.items(),
                              key=lambda x: x[1]['avg_improvement'],
                              reverse=True)

        for i, (algo_name, stats) in enumerate(sorted_algos, 1):
            improvement_rate = stats['improved_clients'] / stats['total_clients'] * 100

            report += f"""
{i}. {algo_name.upper()}
   改进客户端: {stats['improved_clients']}/{stats['total_clients']} ({improvement_rate:.1f}%)
   平均改进: {stats['avg_improvement']:.6f} ± {stats['std_improvement']:.6f}
   改进范围: [{stats['min_improvement']:.6f}, {stats['max_improvement']:.6f}]
   统计显著性: {'显著' if stats['is_significant'] else '不显著'} (p={stats['p_value']:.4f})
   效应量: {stats['effect_size']} (Cohen's d={stats['cohens_d']:.3f})
"""

        report += f"""
{'=' * 100}
统计检验详情
{'-' * 50}
"""

        significant_algos = [name for name, stats in results.items() if stats['is_significant']]

        report += f"""
统计显著优于基准的算法 ({len(significant_algos)}/{len(results)})：
{', '.join(significant_algos) if significant_algos else '无'}

详细统计结果：
算法名称          改进客户端    平均改进      p值        效应量
{'-' * 80}
"""

        for algo_name, stats in sorted_algos:
            report += f"{algo_name:<15} {stats['improved_clients']:>3}/{stats['total_clients']:<3} "
            report += f"{stats['avg_improvement']:>10.6f}  {stats['p_value']:>8.4f}  "
            report += f"{stats['effect_size']:>8}\n"

        report += f"""
{'=' * 100}
关键发现与建议
{'-' * 50}
"""

        # 自动生成关键发现
        best_algo = sorted_algos[0]
        worst_algo = sorted_algos[-1]

        report += f"""
最佳算法: {best_algo[0]} 
   - 在 {best_algo[1]['improved_clients']} 个客户端上表现优于基准
   - 平均改进 {best_algo[1]['avg_improvement']:.6f}
   - {'统计显著' if best_algo[1]['is_significant'] else '统计不显著'}

需要改进的算法: {worst_algo[0]}
   - 仅在 {worst_algo[1]['improved_clients']} 个客户端上表现优于基准
   - 平均改进 {worst_algo[1]['avg_improvement']:.6f}

总体评价:
   - 共评估 {len(results)} 个算法
   - {len(significant_algos)} 个算法表现显著优于基准
   - 总客户端数: {results[best_algo[0]]['total_clients']}
   - 计算方法: 客户端级别的多次实验平均 → 整体平均
"""

        return report

    def save_analysis_results(self, results: Dict[str, Dict],
                              baseline_name: str, fig: plt.Figure) -> None:
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存可视化图表
        if fig:
            fig_path = os.path.join(self.config.output_dir,
                                    f'comprehensive_analysis_{timestamp}.png')
            fig.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            logger.info(f"图表已保存: {fig_path}")

        # 保存详细报告
        report = self.generate_detailed_report(results, baseline_name)
        report_path = os.path.join(self.config.output_dir,
                                   f'analysis_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"报告已保存: {report_path}")

        # 保存JSON格式的结果
        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        try:
            json_results = convert_to_serializable(results)

            json_path = os.path.join(self.config.output_dir,
                                     f'analysis_results_{timestamp}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON结果已保存: {json_path}")

        except Exception as e:
            logger.warning(f"JSON保存失败: {e}，跳过JSON输出")

        # 打印报告到控制台
        print(report)

    def run_complete_analysis(self, raw_data_dir: Optional[str] = None,
                              skip_aggregation: bool = False) -> bool:
        """运行完整的分析流水线"""
        logger.info("=" * 80)
        logger.info("开始联邦学习算法性能分析")
        logger.info("=" * 80)

        try:
            # 步骤1: 数据聚合
            if not skip_aggregation and raw_data_dir:
                logger.info("步骤1: 聚合原始实验数据...")
                if not os.path.exists(raw_data_dir):
                    logger.error(f"原始数据目录不存在: {raw_data_dir}")
                    return False

                self.aggregated_data = self.aggregate_all_results(raw_data_dir)

                if not self.aggregated_data:
                    logger.error("没有找到可处理的数据!")
                    return False

                self.save_aggregated_results(self.aggregated_data)
                data = self.aggregated_data
            else:
                # 步骤2: 加载已聚合的数据
                logger.info("步骤2: 加载已聚合的数据...")
                data = self.load_aggregated_data()

                if not data:
                    logger.error("没有找到聚合数据文件!")
                    return False

            # 步骤3: 统计分析
            logger.info("步骤3: 执行统计分析...")
            self.comparison_results = self.statistical_comparison(
                data, self.config.baseline_algorithm)

            if not self.comparison_results:
                logger.error("统计分析失败!")
                return False

            # 步骤4: 创建可视化
            logger.info("步骤4: 生成可视化图表...")
            fig = self.create_comprehensive_visualizations(
                self.comparison_results, self.config.baseline_algorithm)

            # 步骤5: 保存结果
            logger.info("步骤5: 保存分析结果...")
            self.save_analysis_results(self.comparison_results,
                                       self.config.baseline_algorithm, fig)

            logger.info("=" * 80)
            logger.info("分析流水线完成!")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    简化的主函数 - 直接输入数据目录进行完整分析
    """
    print("联邦学习算法性能分析工具")
    print("=" * 60)

    # 直接使用默认配置
    config = AnalysisConfig()
    print(f"使用默认配置: 基准={config.baseline_algorithm}, 指标={config.metric_column}")

    # 直接询问数据目录
    data_dir = input("\n请输入原始数据目录路径: ").strip()

    if not data_dir:
        # 提供默认路径
        data_dir = './outputs/P2P_seqLen10/oneDS-nonOverlap/NCI1-30clients/eps_0.04_0.08/repeats'
        print(f"使用默认路径: {data_dir}")

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请检查路径是否正确")
        return

    # 创建分析器并直接运行完整分析
    analyzer = FederatedLearningAnalyzer(config)

    print(f"\n开始分析数据目录: {data_dir}")
    print("分析模式: 完整分析 (从原始数据开始)")
    print("-" * 60)

    # 执行完整分析流程
    success = analyzer.run_complete_analysis(
        raw_data_dir=data_dir,
        skip_aggregation=False
    )

    if success:
        print("\n分析完成!")
        print(f"结果保存在: {config.output_dir}")

        # 询问是否打开结果目录
        if input("\n是否打开结果目录? (y/N): ").lower() == 'y':
            import subprocess
            import platform

            try:
                if platform.system() == 'Windows':
                    subprocess.run(['explorer', config.output_dir])
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', config.output_dir])
                else:  # Linux
                    subprocess.run(['xdg-open', config.output_dir])
            except:
                print(f"请手动打开目录: {config.output_dir}")
    else:
        print("\n分析过程中出现错误!")
        print("请检查:")
        print("1. 数据目录路径是否正确")
        print("2. 数据文件格式是否符合要求 (如: 1_accuracy_algorithm_GC.csv)")
        print("3. 是否有足够的权限访问目录")


if __name__ == "__main__":
    # 检查和安装依赖
    missing_deps = []

    try:
        import scipy.stats
    except ImportError:
        missing_deps.append('scipy')

    try:
        import seaborn
    except ImportError:
        missing_deps.append('seaborn')

    if missing_deps:
        print("检测到缺少以下依赖包:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print(f"\n安装命令: pip install {' '.join(missing_deps)}")

        if 'scipy' in missing_deps:
            print("注意: 没有scipy将跳过统计显著性检验，仅进行基础分析")

        choice = input("\n是否继续运行? (y/N): ").lower()
        if choice != 'y':
            print("已退出，请安装依赖后重试")
            exit(1)

    # 运行简化的主函数
    main()