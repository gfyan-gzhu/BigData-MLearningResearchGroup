import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from simEnvParameter import *
import seaborn as sns
import pandas as pd


def save_task_delay_data(devices, filename="filename"):
    task_delay_data = []

    # 时隙遍历结束后，遍历所有设备的任务列表
    for device in devices:
        for time_slot, task in device.task.items():  # 遍历设备的任务字典
            # 检查任务是否为 None
            if task is None:
                continue

            # 收集任务相关信息
            task_info = {
                'task data size（bit）': task.data_length,
                'transmission delay（s）': task.transmission_delay,
                'total delay（s）': task.finished_time - task.generate_time
            }
            task_delay_data.append(task_info)

    # 将收集到的数据转换为DataFrame
    df = pd.DataFrame(task_delay_data)

    # 确定文件保存的目录为当前目录下的../data/
    output_folder = os.path.join(os.getcwd(), '..', 'data')
    output_folder = os.path.abspath(output_folder)  # 获取绝对路径

    # 检查是否存在data文件夹，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建保存文件的完整路径
    file_path = os.path.join(output_folder, filename)

    # 保存数据到Excel文件
    df.to_excel(file_path, index=False)
    print(f"文件已保存至: {file_path}")




# 记录任务的时延和无人机方面的总能耗（接收数据能耗+计算能耗）

def save_UAV_energy_and_task_delay_data(devices, filename="filename"):
    task_energy_delay_data = []

    # 遍历所有设备的任务列表
    for device in devices:
        for time_slot, task in device.task.items():
            # 检查任务是否为 None
            if task is None:
                continue

            # 保存任务信息
            task_info = {
                'task data size (bit)': task.data_length,
                'offload ratio': task.offload_ratio,
                'energy consumption (J)': task.transmission_energy + task.compute_energy,
                'total delay (s)': task.finished_time - task.generate_time
            }
            task_energy_delay_data.append(task_info)

    # 将收集到的数据转换为DataFrame
    df = pd.DataFrame(task_energy_delay_data)

    # 确定文件保存的目录为当前目录下的../data/
    output_folder = os.path.join(os.getcwd(), '..', 'data')
    output_folder = os.path.abspath(output_folder)  # 获取绝对路径

    # 检查是否存在data文件夹，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建保存文件的完整路径
    file_path = os.path.join(output_folder, filename)

    # 保存数据到Excel文件
    df.to_excel(file_path, index=False)
    print(f"文件已保存至: {file_path}")


# 记录任务的时延信息
def save_task_delay_data(devices, filename="filename"):
    task_delay_data = []

    # 遍历所有设备的任务列表
    for device in devices:
        for time_slot, task in device.task.items():
            # 检查任务是否为 None
            if task is None:
                continue

            # 保存任务信息
            task_info = {
                'task data size (bit)': task.data_length,
                'offload ratio': task.offload_ratio,
                'total delay (s)': task.finished_time - task.generate_time
            }
            task_delay_data.append(task_info)

    # 将收集到的数据转换为DataFrame
    df = pd.DataFrame(task_delay_data)

    # 确定文件保存的目录为当前目录下的../data/
    output_folder = os.path.join(os.getcwd(), '..', 'data')
    output_folder = os.path.abspath(output_folder)  # 获取绝对路径

    # 检查是否存在data文件夹，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建保存文件的完整路径
    file_path = os.path.join(output_folder, filename)

    # 保存数据到Excel文件
    df.to_excel(file_path, index=False)
    print(f"文件已保存至: {file_path}")


# 记入无人机能耗信息
def save_UAV_energy_data(devices, filename="filename"):
    energy_data = []

    # 遍历所有设备的任务列表
    for device in devices:
        for time_slot, task in device.task.items():
            # 检查任务是否为 None
            if task is None:
                continue

            # 保存任务信息
            task_info = {
                'task data size (bit)': task.data_length,
                'offload ratio': task.offload_ratio,
                'energy consumption (J)': task.transmission_energy + task.compute_energy
            }
            energy_data.append(task_info)

    # 将收集到的数据转换为DataFrame
    df = pd.DataFrame(energy_data)

    # 确定文件保存的目录为当前目录下的../data/
    output_folder = os.path.join(os.getcwd(), '..', 'data')
    output_folder = os.path.abspath(output_folder)  # 获取绝对路径

    # 检查是否存在data文件夹，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建保存文件的完整路径
    file_path = os.path.join(output_folder, filename)

    # 保存数据到Excel文件
    df.to_excel(file_path, index=False)
    print(f"文件已保存至: {file_path}")





# 不同方案下的奖励对比
def plot_reward_comparison_smooth(methods, base_path='', smooth_window=15):
    """
    绘制不同方法的平均奖励随 episode 变化的比较图（支持滑动平均平滑）。

    参数:
        methods (list): 方案名称列表，如 ['hierarchical', 'standard_ddpg', 'ddpg_ldpg', 'ppo']
        base_path (str): Excel 文件所在目录
        smooth_window (int): 滑动平均窗口大小，若 <=1 则不进行平滑
    """
    # 颜色、标记、标签映射（与您提供的完全一致）
    method_color_map = {
        'ppo': '#2A9D8E',
        'ddpg_ldpg': '#0072BD',
        'standard_ddpg': '#B7B7ED',
        'hierarchical': '#D95319',
    }

    method_marker_map = {
        'hierarchical': 'o',
        'standard_ddpg': '^',
        'ddpg_ldpg': 's',
        'ppo': 'v',
    }

    method_label_map = {
        'hierarchical': 'hTPTO',
        'standard_ddpg': 'DDPG',
        'ddpg_ldpg': 'DDPG-LDPG',
        'ppo': 'PPO',
    }

    plt.figure(figsize=(12, 6))

    for method in methods:
        # 构造文件名：假设格式为 {method}_results_ep1000.xlsx
        file_name = f"{method}_results_ep1000.xlsx"
        full_path = os.path.join(base_path, file_name)

        if not os.path.exists(full_path):
            print(f"警告：无法找到文件 {full_path}，跳过该方法")
            continue

        try:
            df = pd.read_excel(full_path)
        except Exception as e:
            print(f"读取文件 {full_path} 失败: {e}")
            continue

        if df.shape[1] < 2:
            print(f"文件 {full_path} 列数不足2，跳过")
            continue

        episodes = df.iloc[:, 0]   # 第一列为 episode
        rewards = df.iloc[:, 1]    # 第二列为平均奖励

        # 平滑处理
        if smooth_window > 1:
            # 使用 rolling 滑动平均，center=True 使曲线对齐
            rewards_smooth = rewards.rolling(window=smooth_window, center=True, min_periods=1).mean()
            label_smooth = method_label_map.get(method, method)
        else:
            rewards_smooth = rewards
            label_smooth = method_label_map.get(method, method)

        color = method_color_map.get(method, 'gray')
        # 主曲线：平滑后的数据（粗线）
        plt.plot(episodes, rewards_smooth, label=label_smooth, color=color, linewidth=2)

        # 可选：用半透明细线绘制原始数据，以展示波动细节
        if smooth_window > 1:
            plt.plot(episodes, rewards, color=color, alpha=0.2, linewidth=0.8)

    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.title('Comparison of Average Reward under Different Algorithms (Smoothed)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()



# 不同交替间隔下的奖励对比
def plot_alternate_interval_comparison(intervals, base_path='', smooth_window=20):
    """
    绘制不同交替间隔下平均奖励随 episode 变化的对比图（垂直堆叠子图）。
    每个间隔对应一个子图，线条颜色按照给定映射，参数值以文本框形式标注在子图内。

    参数:
        intervals (list): 交替间隔列表，如 [5, 10, 15, 20, 25]
        base_path (str): Excel 文件所在目录
        smooth_window (int): 滑动平均窗口大小，若 <=1 则不进行平滑
    """
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    import numpy as np

    # 间隔颜色映射（可扩展）
    interval_color_map = {
        5: '#1f77b4',
        10: '#D95319',
        15: '#2ca02c',
        20: '#9467bd',
        25: '#A9A9A9'
    }

    # 读取所有间隔的数据
    data = {}
    for rho in intervals:
        file_name = f"hierarchical_results_ep1000_alternate_{rho}.xlsx"
        full_path = os.path.join(base_path, file_name)
        if not os.path.exists(full_path):
            print(f"警告：无法找到文件 {full_path}，跳过间隔 {rho}")
            continue
        try:
            df = pd.read_excel(full_path)
        except Exception as e:
            print(f"读取文件 {full_path} 失败: {e}")
            continue
        if df.shape[1] < 2:
            print(f"文件 {full_path} 列数不足2，跳过")
            continue
        episodes = df.iloc[:, 0]
        rewards = df.iloc[:, 1]
        data[rho] = (episodes, rewards)

    if not data:
        print("没有有效数据，无法绘图")
        return

    # 收集所有奖励数据以确定统一的 y 轴范围
    all_rewards = []
    for (ep, rew) in data.values():
        all_rewards.extend(rew)

    if all_rewards:
        y_min = min(all_rewards)
        y_max = max(all_rewards)
        # 扩展一点边距
        y_pad = 0.05 * (y_max - y_min)
        y_min = max(0, y_min - y_pad)  # 确保不低于0
        y_max = y_max + y_pad
        # 调整为 0.1 的倍数
        y_min = np.floor(y_min * 10) / 10
        y_max = np.ceil(y_max * 10) / 10
    else:
        y_min, y_max = 0, 1

    # 创建垂直子图，宽度调小为8，高度适当
    n_plots = len(data)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, sharex=True,
                              figsize=(8, 1.2 * n_plots), squeeze=False)
    axes = axes.flatten()

    # 按间隔排序，保持顺序
    sorted_rhos = sorted(data.keys())

    for idx, rho in enumerate(sorted_rhos):
        ax = axes[idx]
        episodes, rewards = data[rho]

        # 平滑处理
        if smooth_window > 1:
            rewards_smooth = rewards.rolling(window=smooth_window, center=True, min_periods=1).mean()
        else:
            rewards_smooth = rewards

        # 获取颜色
        color = interval_color_map.get(rho, 'steelblue')

        # 绘制原始数据（半透明细线）
        ax.plot(episodes, rewards, color=color, alpha=0.3, linewidth=0.8)
        # 绘制平滑曲线（粗线）
        ax.plot(episodes, rewards_smooth, color=color, linewidth=2)

        # 在子图左上角添加标注间隔值（无背景框）
        ax.text(0.02, 0.98, rf'$\varrho = {rho}$', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')  # 无 bbox

        ax.grid(True, linestyle='--', alpha=0.6)

        # 设置统一的纵轴范围
        ax.set_ylim(y_min, y_max)

        # 纵坐标刻度：保留两位小数，主刻度间隔 0.1
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        # 可选：设置次要刻度
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # 添加统一的纵轴标签，居中在整个图形左侧
    fig.supylabel('Average reward', fontsize=12, x=0.03)

    # 共享x轴标签
    axes[-1].set_xlabel('Episode', fontsize=12)

    plt.tight_layout()
    plt.show(block=False)



def plot_target_offload_reward_comparison(target_offload, base_path='', smooth_window=20):
    """
    绘制不同目标卸载比例下平均奖励随 episode 变化的对比图（垂直堆叠子图）。
    每个比例对应一个子图，线条颜色按映射，参数值以文本框形式标注在子图内。

    参数:
        target_offload (list): 目标卸载比例列表，如 [0.2, 0.4, 0.6, 0.8, 1.0]
        base_path (str): Excel 文件所在目录
        smooth_window (int): 滑动平均窗口大小，若 <=1 则不进行平滑
    """
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    import numpy as np

    # 目标卸载比例颜色映射（可扩展）
    target_offload_color_map = {
        0.2: '#1f77b4',
        0.4: '#2ca02c',
        0.6: '#D95319',
        0.8: '#9467bd',
        1.0: '#A9A9A9'
    }

    # 读取所有比例的数据
    data = {}
    for rho in target_offload:
        file_name = f"hierarchical_results_ep1000_target_offload_{rho}.xlsx"
        full_path = os.path.join(base_path, file_name)
        if not os.path.exists(full_path):
            print(f"警告：无法找到文件 {full_path}，跳过比例 {rho}")
            continue
        try:
            df = pd.read_excel(full_path)
        except Exception as e:
            print(f"读取文件 {full_path} 失败: {e}")
            continue
        if df.shape[1] < 2:
            print(f"文件 {full_path} 列数不足2，跳过")
            continue
        episodes = df.iloc[:, 0]
        rewards = df.iloc[:, 1]
        data[rho] = (episodes, rewards)

    if not data:
        print("没有有效数据，无法绘图")
        return

    # 收集所有奖励数据以确定统一的 y 轴范围
    all_rewards = []
    for (ep, rew) in data.values():
        all_rewards.extend(rew)

    if all_rewards:
        y_min = min(all_rewards)
        y_max = max(all_rewards)
        # 扩展一点边距
        y_pad = 0.05 * (y_max - y_min)
        y_min = max(0, y_min - y_pad)  # 确保不低于0
        y_max = y_max + y_pad
        # 调整为 0.1 的倍数
        y_min = np.floor(y_min * 10) / 10
        y_max = np.ceil(y_max * 10) / 10
    else:
        y_min, y_max = 0, 1

    # 创建垂直子图
    n_plots = len(data)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, sharex=True,
                              figsize=(8, 1.2 * n_plots), squeeze=False)
    axes = axes.flatten()

    # 按比例排序，保持顺序
    sorted_rhos = sorted(data.keys())

    for idx, rho in enumerate(sorted_rhos):
        ax = axes[idx]
        episodes, rewards = data[rho]

        # 平滑处理
        if smooth_window > 1:
            rewards_smooth = rewards.rolling(window=smooth_window, center=True, min_periods=1).mean()
        else:
            rewards_smooth = rewards

        # 获取颜色
        color = target_offload_color_map.get(rho, 'steelblue')

        # 绘制原始数据（半透明细线）
        ax.plot(episodes, rewards, color=color, alpha=0.3, linewidth=0.8)
        # 绘制平滑曲线（粗线）
        ax.plot(episodes, rewards_smooth, color=color, linewidth=2)

        # 在子图左上角添加标注比例值（无背景框）
        ax.text(0.02, 0.98, rf'$o^{{\mathrm{{set}}}} = {rho}$', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')

        ax.grid(True, linestyle='--', alpha=0.6)

        # 设置统一的纵轴范围
        ax.set_ylim(y_min, y_max)

        # 纵坐标刻度：保留两位小数，主刻度间隔 0.1
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # 添加统一的纵轴标签，居中在整个图形左侧
    fig.supylabel('Average reward', fontsize=12, x=0.03)

    # 共享x轴标签
    axes[-1].set_xlabel('Episode', fontsize=12)

    plt.tight_layout()
    plt.show(block=False)

# ==================== 目标卸载比例比较（折线图） ====================
def plot_target_offload_ratio_comparison(target_offload_ratios, methods, base_path=''):
    """
    绘制不同目标卸载比例下各算法的平均卸载比例比较图（折线图）。

    参数:
        target_offload_ratios (list): 目标卸载比例列表，如 [0.2, 0.4, 0.6, 0.8, 1.0]
        methods (list): 算法名称列表，如 ['ppo', 'standard_ddpg', 'ddpg_ldpg', 'hierarchical']
        base_path (str): 存放 Excel 文件的目录
    """
    # 方法名 -> 颜色（固定绑定）
    method_color_map = {
        'ppo': '#B55384',
        'standard_ddpg': '#B7B7EB',
        'ddpg_ldpg': '#0072BD',
        'hierarchical': '#D95319',
    }
    # 方法名 -> 标记形状
    method_marker_map = {
        'ppo': 'v',          # 倒三角形
        'standard_ddpg': '<', # 左三角形
        'ddpg_ldpg': '>',    # 右三角形
        'hierarchical': 'o', # 圆形
    }
    # 方法名 -> 图例标签
    method_label_map = {
        'hierarchical': 'hTPTO',
        'ddpg_ldpg': 'DDPG_LDPG',
        'standard_ddpg': 'DDPG',
        'ppo': 'PPO',
    }

    # 存储不同方法、不同目标卸载比例下的平均实际卸载比例
    data = {method: [] for method in methods}

    # 读取每个文件并计算平均卸载比例
    for val in target_offload_ratios:
        for method in methods:
            # 构造文件名，格式：{method}_target_offload_ratio_{val}.xlsx
            file_name = f'{method}_target_offload_ratio_{val}.xlsx'
            full_path = os.path.join(base_path, file_name)
            if not os.path.exists(full_path):
                print(f"警告：无法找到文件 {full_path}，跳过 {method} 在 target_offload_ratio={val} 的数据")
                data[method].append(0.0)
                continue
            try:
                df = pd.read_excel(full_path)
                # 计算卸载比例 = 卸载的子任务数 / 总子任务数
                offload_ratio = df['offloaded'].mean()
                data[method].append(offload_ratio)
            except Exception as e:
                print(f"读取文件 {full_path} 失败: {e}")
                data[method].append(0.0)

    # 创建折线图
    plt.figure(figsize=(10, 6))
    for method in methods:
        color = method_color_map.get(method, None)
        marker = method_marker_map.get(method, 'o')
        label = method_label_map.get(method, method)
        plt.plot(target_offload_ratios, data[method],
                 label=label, marker=marker, color=color,
                 linewidth=2, markersize=8)

    plt.xlabel('$o^{{\mathrm{{set}}}}}$ values', fontsize=12)
    plt.ylabel('Actual average offload ratio', fontsize=12)
    plt.title('Comparison of Average Offload Ratio under Different Target Offload Ratios', fontsize=14)
    plt.xticks(target_offload_ratios)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show(block=False)


# ==================== 任务数据量范围 vs 平均子任务时延（折线图） ====================
def plot_task_data_size_delay_comparison(task_intervals, methods, base_path=''):
    """
    绘制不同任务数据量范围下各算法的平均子任务时延比较图（折线图）。

    参数:
        task_intervals (list of tuple): 任务数据量范围列表，如 [(1,2), (2,3), (3,4), (4,5), (5,6)] (单位 Mbits)
        methods (list): 算法名称列表
        base_path (str): 存放 Excel 文件的目录
    """
    # 折线图样式映射（同 plot_target_offload_ratio_comparison）
    method_color_map = {
        'ppo': '#B55384',
        'standard_ddpg': '#B7B7EB',
        'ddpg_ldpg': '#0072BD',
        'hierarchical': '#D95319',
    }
    method_marker_map = {
        'ppo': 'v',
        'standard_ddpg': '<',
        'ddpg_ldpg': '>',
        'hierarchical': 'o',
    }
    method_label_map = {
        'hierarchical': 'hTPTO',
        'ddpg_ldpg': 'DDPG_LDPG',
        'standard_ddpg': 'DDPG',
        'ppo': 'PPO',
    }

    # 计算每个区间的中点，作为横坐标
    x_vals = [(low + high) / 2.0 for low, high in task_intervals]
    # 刻度标签：显示区间范围
    x_labels = [f'[{low},{high}]' for low, high in task_intervals]

    data = {method: [] for method in methods}

    for (low, high) in task_intervals:
        for method in methods:
            # 文件名格式：{method}_taskdata_{low}_{high}.xlsx
            file_name = f'{method}_taskdata_{low}_{high}.xlsx'
            full_path = os.path.join(base_path, file_name)
            if not os.path.exists(full_path):
                print(f"警告：无法找到文件 {full_path}，跳过 {method} 在区间 [{low},{high}] 的数据")
                data[method].append(0.0)
                continue
            try:
                df = pd.read_excel(full_path)
                # 计算平均子任务时延
                avg_delay = df['delay'].mean()
                data[method].append(avg_delay)
            except Exception as e:
                print(f"读取文件 {full_path} 失败: {e}")
                data[method].append(0.0)

    plt.figure(figsize=(10, 6))
    for method in methods:
        color = method_color_map.get(method, None)
        marker = method_marker_map.get(method, 'o')
        label = method_label_map.get(method, method)
        plt.plot(x_vals, data[method],
                 label=label, marker=marker, color=color,
                 linewidth=2, markersize=8)

    plt.xlabel('Task data size range (MB)', fontsize=12)
    plt.ylabel('Average delay (s)', fontsize=12)
    plt.title('Comparison of Average Subtask Delay under Different Task Data Sizes', fontsize=14)
    plt.xticks(x_vals, x_labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show(block=False)


# ==================== 任务数据量范围 vs 平均子任务能耗（柱状图） ====================
def plot_task_data_size_energy_comparison(task_intervals, methods, base_path=''):
    """
    绘制不同任务数据量范围下各算法的平均子任务能耗比较图（柱状图）。

    参数:
        task_intervals (list of tuple): 任务数据量范围列表，如 [(1,2), (2,3), (3,4), (4,5), (5,6)] (单位 Mbits)
        methods (list): 算法名称列表
        base_path (str): 存放 Excel 文件的目录
    """
    # 柱状图样式映射
    method_color_map = {
        'ppo': '#B6B6B4',
        'standard_ddpg': '#E8F5E9',
        'ddpg_ldpg': '#FFDFB8',
        'hierarchical': '#FF9494',
    }
    hatch_map = {
        'ppo': '----',      # 水平线
        'standard_ddpg': '++++',  # 十字线
        'ddpg_ldpg': 'xxxx',      # 交叉线
        'hierarchical': 'oooo'    # 圆圈
    }
    method_label_map = {
        'hierarchical': 'hTPTO',
        'ddpg_ldpg': 'DDPG_LDPG',
        'standard_ddpg': 'DDPG',
        'ppo': 'PPO',
    }

    # 横坐标：区间中点
    x_vals = [(low + high) / 2.0 for low, high in task_intervals]
    x_labels = [f'[{low},{high}]' for low, high in task_intervals]

    data = {method: [] for method in methods}

    for (low, high) in task_intervals:
        for method in methods:
            file_name = f'{method}_taskdata_{low}_{high}.xlsx'
            full_path = os.path.join(base_path, file_name)
            if not os.path.exists(full_path):
                print(f"警告：无法找到文件 {full_path}，跳过 {method} 在区间 [{low},{high}] 的数据")
                data[method].append(0.0)
                continue
            try:
                df = pd.read_excel(full_path)
                # 计算平均子任务能耗
                avg_energy = df['compute_energy'].mean()
                data[method].append(avg_energy)
            except Exception as e:
                print(f"读取文件 {full_path} 失败: {e}")
                data[method].append(0.0)

    # 柱状图布局
    n_methods = len(methods)
    width = 0.25
    group_gap = 0.2
    group_width = n_methods * width
    x_positions = []
    current_pos = 0
    for _ in x_vals:
        group_pos = [current_pos + j * width for j in range(n_methods)]
        x_positions.append(group_pos)
        current_pos += group_width + group_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(methods):
        xs = [x_positions[j][i] for j in range(len(x_vals))]
        ax.bar(xs, data[method], width=width,
               label=method_label_map.get(method, method),
               color=method_color_map.get(method, None),
               hatch=hatch_map.get(method, ''),
               edgecolor='black')

    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Task data size range (MB)', fontsize=12)
    ax.set_ylabel('Average compute energy (J)', fontsize=12)
    ax.set_title('Comparison of Average Subtask Energy Consumption under Different Task Data Sizes', fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)


# ==================== 无人机计算资源 vs 平均子任务时延（折线图） ====================
def plot_uav_compute_resource_delay_comparison(compute_resources, methods, base_path=''):
    """
    绘制不同无人机计算资源下各算法的平均子任务时延比较图（折线图）。
    参数:
        compute_resources (list): 无人机计算资源列表，如 [1e9, 2e9, 3e9, 4e9, 5e9] (cycle/s)
        methods (list): 算法名称列表
        base_path (str): 存放 Excel 文件的目录
    """
    # 折线图样式映射
    method_color_map = {
        'ppo': '#B55384',
        'standard_ddpg': '#B7B7EB',
        'ddpg_ldpg': '#0072BD',
        'hierarchical': '#D95319',
    }
    method_marker_map = {
        'ppo': 'v',
        'standard_ddpg': '<',
        'ddpg_ldpg': '>',
        'hierarchical': 'o',
    }
    method_label_map = {
        'hierarchical': 'hTPTO',
        'ddpg_ldpg': 'DDPG_LDPG',
        'standard_ddpg': 'DDPG',
        'ppo': 'PPO',
    }

    # 将计算资源转换为 GHz 作为横坐标
    x_vals = [res / 1e9 for res in compute_resources]
    # 修改点：将原先的 f'{res:.1f}' 改为整数显示
    x_labels = [f'{int(res)}' for res in x_vals]   # 或者 f'{res:.0f}'

    data = {method: [] for method in methods}

    for res in compute_resources:
        for method in methods:
            # 文件名：{method}_uav_compute_resource_{int(res)}.xlsx
            file_name = f'{method}_uav_compute_resource_{int(res)}.0.xlsx'
            full_path = os.path.join(base_path, file_name)
            if not os.path.exists(full_path):
                print(f"警告：无法找到文件 {full_path}，跳过 {method} 在 compute_resource={res} 的数据")
                data[method].append(0.0)
                continue
            try:
                df = pd.read_excel(full_path)
                avg_delay = df['delay'].mean()
                data[method].append(avg_delay)
            except Exception as e:
                print(f"读取文件 {full_path} 失败: {e}")
                data[method].append(0.0)

    plt.figure(figsize=(10, 6))
    for method in methods:
        color = method_color_map.get(method, None)
        marker = method_marker_map.get(method, 'o')
        label = method_label_map.get(method, method)
        plt.plot(x_vals, data[method],
                 label=label, marker=marker, color=color,
                 linewidth=2, markersize=8)

    plt.xlabel('UAV compute resource (GC/s)', fontsize=12)
    plt.ylabel('Average delay (s)', fontsize=12)
    plt.title('Comparison of Average Subtask Delay under Different UAV Compute Resources', fontsize=14)
    plt.xticks(x_vals, x_labels)  # 应用修改后的整数标签
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show(block=False)


# ==================== 无人机计算资源 vs 平均子任务能耗（柱状图） ====================
def plot_uav_compute_resource_energy_comparison(compute_resources, methods, base_path=''):
    """
    绘制不同无人机计算资源下各算法的平均子任务能耗比较图（柱状图）。
    参数:
        compute_resources (list): 无人机计算资源列表，如 [1e9, 2e9, 3e9, 4e9, 5e9] (cycle/s)
        methods (list): 算法名称列表
        base_path (str): 存放 Excel 文件的目录
    """
    # 柱状图样式映射
    method_color_map = {
        'ppo': '#B6B6B4',
        'standard_ddpg': '#E8F5E9',
        'ddpg_ldpg': '#FFDFB8',
        'hierarchical': '#FF9494',
    }
    hatch_map = {
        'ppo': '----',
        'standard_ddpg': '++++',
        'ddpg_ldpg': 'xxxx',
        'hierarchical': 'oooo'
    }
    method_label_map = {
        'hierarchical': 'hTPTO',
        'ddpg_ldpg': 'DDPG_LDPG',
        'standard_ddpg': 'DDPG',
        'ppo': 'PPO',
    }

    x_vals = [res / 1e9 for res in compute_resources]
    # 修改点：横坐标标签改为整数
    x_labels = [f'{int(res)}' for res in x_vals]

    data = {method: [] for method in methods}

    for res in compute_resources:
        for method in methods:
            file_name = f'{method}_uav_compute_resource_{int(res)}.0.xlsx'
            full_path = os.path.join(base_path, file_name)
            if not os.path.exists(full_path):
                print(f"警告：无法找到文件 {full_path}，跳过 {method} 在 compute_resource={res} 的数据")
                data[method].append(0.0)
                continue
            try:
                df = pd.read_excel(full_path)
                avg_energy = df['compute_energy'].mean()
                data[method].append(avg_energy)
            except Exception as e:
                print(f"读取文件 {full_path} 失败: {e}")
                data[method].append(0.0)

    # 柱状图布局
    n_methods = len(methods)
    width = 0.25
    group_gap = 0.2
    group_width = n_methods * width
    x_positions = []
    current_pos = 0
    for _ in x_vals:
        group_pos = [current_pos + j * width for j in range(n_methods)]
        x_positions.append(group_pos)
        current_pos += group_width + group_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(methods):
        xs = [x_positions[j][i] for j in range(len(x_vals))]
        ax.bar(xs, data[method], width=width,
               label=method_label_map.get(method, method),
               color=method_color_map.get(method, None),
               hatch=hatch_map.get(method, ''),
               edgecolor='black')

    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(x_labels)  # 应用整数标签
    ax.set_xlabel('UAV compute resource (GC/s)', fontsize=12)
    ax.set_ylabel('Average energy (J)', fontsize=12)
    ax.set_title('Comparison of Average Subtask Energy Consumption under Different UAV Compute Resources', fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)

# 绘制无人机3D轨迹
def plot_uav_3d_trajectory(algorithm_names, base_path='data/trajectory', save_fig=False):
    """
    绘制指定算法的无人机3D轨迹（不包含任务点）。
    每个算法一个图，显示所有无人机轨迹，同一算法内不同无人机用不同线型区分，算法颜色固定。

    参数:
        algorithm_names (list): 算法名称列表，如 ['hierarchical', 'standard_ddpg', 'ddpg_ldpg', 'ppo']
        base_path (str): 存放轨迹Excel文件的文件夹路径
        save_fig (bool): 是否保存图片到同一文件夹
    """
    # 算法颜色映射（参考draw文件中的曲线图颜色）
    method_color_map = {
        'hierarchical': '#D95319',   # 橙红
        'standard_ddpg': '#B7B7EB',  # 浅紫
        'ddpg_ldpg': '#0072BD',      # 深蓝
        'ppo': '#2A9D8E',            # 绿松石
    }
    # 线型列表，用于同一算法内区分不同无人机
    linestyles = ['-', '--', '-.', ':']

    for algo_name in algorithm_names:
        # 构造轨迹文件路径
        traj_filepath = os.path.join(base_path, f'{algo_name}_trajectories.xlsx')

        if not os.path.exists(traj_filepath):
            print(f"轨迹文件不存在: {traj_filepath}，跳过")
            continue

        # 读取轨迹数据
        traj_df = pd.read_excel(traj_filepath)

        if traj_df.empty:
            print(f"轨迹文件为空: {traj_filepath}，跳过")
            continue

        # 获取算法颜色
        color = method_color_map.get(algo_name, 'gray')

        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 获取所有无人机ID
        if 'uav_id' not in traj_df.columns:
            print(f"轨迹文件中缺少 'uav_id' 列，跳过 {algo_name}")
            continue

        uav_ids = sorted(traj_df['uav_id'].unique())
        # 用于存储图例句柄和标签
        handles = []
        labels = []

        # 为每个无人机绘制轨迹
        for i, uav_id in enumerate(uav_ids):
            # 显示编号从1开始
            uav_label = f'UAV {i+1}'
            # 循环选择线型
            linestyle = linestyles[i % len(linestyles)]

            df_uav = traj_df[traj_df['uav_id'] == uav_id].sort_values('time_slot')

            # 绘制轨迹线（同时作为图例句柄）
            line, = ax.plot(df_uav['x'], df_uav['y'], df_uav['z'],
                            color=color, linestyle=linestyle, linewidth=2,
                            label=uav_label)
            handles.append(line)
            labels.append(uav_label)

            # 标记起点（三角形）
            ax.scatter(df_uav.iloc[0]['x'], df_uav.iloc[0]['y'], df_uav.iloc[0]['z'],
                       color=color, marker='^', s=50, edgecolors='black')
            # 标记终点（方形）
            ax.scatter(df_uav.iloc[-1]['x'], df_uav.iloc[-1]['y'], df_uav.iloc[-1]['z'],
                       color=color, marker='s', s=50, edgecolors='black')

        # 创建起始和终点的代理艺术家，用于图例
        from matplotlib.lines import Line2D
        start_proxy = Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
                             markeredgecolor='black', markersize=8, linestyle='None',
                             label='Start Position')
        end_proxy = Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                           markeredgecolor='black', markersize=8, linestyle='None',
                           label='End Position')
        handles.extend([start_proxy, end_proxy])
        labels.extend(['Start Position', 'End Position'])

        # 设置坐标轴标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'UAV Trajectories - {algo_name}')

        # 添加图例（包含无人机线型、起始、终点）
        ax.legend(handles, labels, loc='best', fontsize=10)

        # 显示网格
        ax.grid(True)

        if save_fig:
            plt.savefig(os.path.join(base_path, f'{algo_name}_trajectory.png'), dpi=150)
        plt.show(block=False)


def main():
    # 动态获取 data 文件夹的绝对路径
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 获取src所在目录的父目录

    # 不同方案的奖励对比
    # 定义算法列表（与实验中使用的名称一致）
    methods = ['hierarchical', 'standard_ddpg', 'ddpg_ldpg', 'ppo']

    # 不同方案的奖励对比
    data_folder_reward_episode = os.path.join(parent_directory, 'data/reward_episode')
    plot_reward_comparison_smooth(methods, base_path=data_folder_reward_episode)

    # 不同交替间隔下奖励对比
    alternate = [5, 10, 15, 20, 25]
    data_folder_alternate = os.path.join(parent_directory, 'data/alternate')
    plot_alternate_interval_comparison(alternate, data_folder_alternate)

    # 不同目标卸载参数下奖励对比（只需两个参数）
    target_offload = [0.2, 0.4, 0.6, 0.8, 1.0]
    data_folder_target_offload = os.path.join(parent_directory, 'data/target_offload')
    plot_target_offload_reward_comparison(target_offload, data_folder_target_offload)

    # 不同目标卸载参数下各方案卸载比例对比（需要 methods 参数）
    data_folder_target_offload_ratio = os.path.join(parent_directory, 'data/target_offload_ratio')
    plot_target_offload_ratio_comparison(target_offload, methods, base_path=data_folder_target_offload_ratio)

    # 不同任务大小下各方案时延和能耗对比
    task_data_size = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    data_folder_task_data_size = os.path.join(parent_directory, 'data/task_data_size')
    plot_task_data_size_delay_comparison(task_data_size, methods, base_path=data_folder_task_data_size)
    plot_task_data_size_energy_comparison(task_data_size, methods, base_path=data_folder_task_data_size)

    # 不同无人机处理能力下各方案时延和能耗对比
    uav_compute_resource = [1e9, 2e9, 3e9, 4e9, 5e9]
    data_folder_uav_compute_resource = os.path.join(parent_directory, 'data/uav_compute_resource')
    plot_uav_compute_resource_delay_comparison(uav_compute_resource, methods,
                                               base_path=data_folder_uav_compute_resource)
    plot_uav_compute_resource_energy_comparison(uav_compute_resource, methods,
                                                base_path=data_folder_uav_compute_resource)


    # 绘制无人机轨迹和任务分布
    algorithms = ['hierarchical', 'standard_ddpg', 'ddpg_ldpg', 'ppo']
    data_folder_trajectory = os.path.join(parent_directory, 'data/uav_trajectory_and_task_info')

    plot_uav_3d_trajectory(algorithms, base_path=data_folder_trajectory, save_fig=True)

    plt.show(block=True)



if __name__ == '__main__':
   main()