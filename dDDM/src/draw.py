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


# 初始位置绘图
def plot_initial_positions(devices, uav):
    # 创建一个新的图形
    plt.figure(figsize=(10, 8))

    # 绘制设备位置
    device_coords = [device.coordinate for device in devices]
    plt.scatter([coord[0] for coord in device_coords], [coord[1] for coord in device_coords],
                marker='^', color='#EAB883', label='MDs', s=90, edgecolor='black')
    # 添加一个不可见的无人机散点图以保留在图例中
    plt.scatter([], [], marker='*', color='#4485C7', edgecolor='#4485C7', label='UAV', s=90)
    # 坐标轴标签
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # 设置坐标轴的范围
    plt.xlim(0, GROUND_LENGTH)  # 根据实际需求调整范围
    plt.ylim(0, GROUND_WIDTH)  # 根据实际需求调整范围

    # 设置坐标轴的刻度
    x_ticks = list(range(0, GROUND_LENGTH + 1, 50))  # 从0到GROUND_LENGTH，每隔50一个刻度
    y_ticks = list(range(0, GROUND_WIDTH + 1, 50))  # 从0到GROUND_WIDTH，每隔50一个刻度
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # 显示网格
    plt.grid(True)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show(block=False)


# 画各迭代次数的平均收敛度值图像
def plot_average_convergence_single(iterative_recording, max_iter=100):
    """
    绘制每个迭代次数的平均最好成绩的曲线

    参数:
        iterative_recording: 字典，key 为时隙，value 为每个时隙的迭代记录，包含迭代次数、最优得分、最优位置
        max_iter: 最大迭代次数
    """
    # 准备存储每个迭代次数的平均最佳得分
    average_best_scores_per_iteration = []

    # 遍历每个迭代次数，从 1 到 max_iter
    for iteration in range(1, max_iter + 1):
        # 存储该迭代次数下的所有时隙的得分
        scores_at_iteration = []

        # 遍历每个时隙
        for time_slot, iterations in iterative_recording.items():
            # 找到该时隙中对应迭代次数的得分
            if iteration <= len(iterations):  # 确保该时隙的迭代次数足够
                score = iterations[iteration - 1][1]  # 迭代次数从 1 开始，索引从 0 开始
                scores_at_iteration.append(score)

        # 计算该迭代次数下的平均得分
        if scores_at_iteration:
            avg_best_score = sum(scores_at_iteration) / len(scores_at_iteration)
        else:
            avg_best_score = 0  # 如果没有得分，则认为得分为 0

        # 将计算的平均得分添加到列表中
        average_best_scores_per_iteration.append(avg_best_score)

    # 绘制平均收敛情况图，去掉圆点，只显示曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), average_best_scores_per_iteration, linestyle='-', color='b')

    # 设置图表标题和标签
    plt.title('Average Best Score Across All Time Slots for Each Iteration')
    plt.xlabel('Iteration number')
    plt.ylabel('Average best score')

    # 显示图形
    plt.grid(True)
    plt.show(block=False)


# 绘制算法迭代图
def plot_average_convergence(iterative_recording1,
                             iterative_recording2,
                             label1='Experiment 1',
                             label2='Experiment 2',
                             max_iter=100):
    # 创建图形
    plt.figure(figsize=(10, 6))

    # 计算第一个iterative_recording1的平均最佳得分
    average_best_scores_per_iteration1 = []
    for iteration in range(1, max_iter + 1):
        scores_at_iteration = []
        # 遍历所有时隙
        for time_slot, iterations in iterative_recording1.items():
            if iteration <= len(iterations):
                score = iterations[iteration - 1][1]  # 获取该时隙的当前迭代得分
                scores_at_iteration.append(score)
        # 计算当前迭代的所有时隙的平均得分
        avg_best_score1 = sum(scores_at_iteration) / len(iterative_recording1) if scores_at_iteration else 0
        average_best_scores_per_iteration1.append(avg_best_score1)

    # 计算第二个iterative_recording2的平均最佳得分
    average_best_scores_per_iteration2 = []
    for iteration in range(1, max_iter + 1):
        scores_at_iteration = []
        # 遍历所有时隙
        for time_slot, iterations in iterative_recording2.items():
            if iteration <= len(iterations):
                score = iterations[iteration - 1][1]  # 获取该时隙的当前迭代得分
                scores_at_iteration.append(score)
        # 计算当前迭代的所有时隙的平均得分
        avg_best_score2 = sum(scores_at_iteration) / len(iterative_recording2) if scores_at_iteration else 0
        average_best_scores_per_iteration2.append(avg_best_score2)

    # 绘制两个记录的收敛曲线
    plt.plot(range(1, max_iter + 1), average_best_scores_per_iteration1, label=label1, color='b', linestyle='-')
    plt.plot(range(1, max_iter + 1), average_best_scores_per_iteration2, label=label2, color='r', linestyle='--')

    # 设置图表标题和标签
    plt.title('Average Best Score Across All Time Slots for Each Iteration')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness value')

    # 设置横纵坐标的起点
    plt.xlim(0, max_iter + 1)  # 横坐标从0开始

    # 计算纵轴范围
    min_score1 = min(average_best_scores_per_iteration1)
    min_score2 = min(average_best_scores_per_iteration2)
    overall_min = min(min_score1, min_score2)

    max_score1 = max(average_best_scores_per_iteration1)
    max_score2 = max(average_best_scores_per_iteration2)
    overall_max = max(max_score1, max_score2)

    # 计算边距，确保当数据范围为零时仍有合适边距
    if overall_max != overall_min:
        margin = 0.1 * (overall_max - overall_min)
    else:
        margin = 0.1  # 当所有数据相同时使用固定边距

    plt.ylim(overall_min - margin, overall_max + margin)

    # 定义用于格式化y轴刻度标签的函数
    def format_y_tick(value, tick_number):
        formatted_value = f"{value:.2f}"  # 格式化为保留两位小数
        # 去除末尾不必要的零和小数点
        return formatted_value.rstrip('0').rstrip('.') if '.' in formatted_value else formatted_value

    # 应用格式化函数到y轴
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_tick))

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show(block=False)


# 保存特定时隙任务分布以及无人机部署位置
def save_selected_time_slots_data(devices, uav_objects, uav_names, time_slots, filename_prefix="selected"):
    """
    保存指定时隙的任务分布和无人机轨迹数据

    参数:
        devices: 设备对象列表（原始设备）
        uav_objects: 无人机对象列表，顺序和uav_names对应
        uav_names: 无人机名称列表，如 ["random", "PSO", "HOLD", "DDPG", "RAT", "dDDM"]
        time_slots: 要保存的时隙列表，如 [5, 10, 20, 30]
        filename_prefix: 文件名前缀
    """
    # 创建保存数据的目录
    output_folder = os.path.join(os.getcwd(), '..', 'data', 'task distribution and uav deployment')
    os.makedirs(output_folder, exist_ok=True)

    # 遍历每个要保存的时隙
    for time_slot in time_slots:
        # 构建文件名
        filename = f"{filename_prefix}_time_slot_{time_slot}.xlsx"
        file_path = os.path.join(output_folder, filename)

        # 创建Excel写入对象
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # ===== 工作表1: 有任务的设备信息 =====
            task_data = []

            for device in devices:
                # 只保存当前时隙有任务的设备
                if time_slot in device.task and device.task[time_slot] is not None:
                    task = device.task[time_slot]

                    # 检查是否有轨迹记录
                    if time_slot not in device.trajectory:
                        continue

                    task_info = {
                        'time_slot': time_slot,
                        'device_id': device.id,
                        'device_x': device.trajectory[time_slot][0],
                        'device_y': device.trajectory[time_slot][1],
                        'task_data_length_bits': task.data_length,
                        'task_data_length_Mbits': task.data_length / 1e6
                    }
                    task_data.append(task_info)

            # 保存任务数据
            if task_data:
                tasks_df = pd.DataFrame(task_data)
                tasks_df.to_excel(writer, sheet_name='Tasks', index=False)
            else:
                # 创建空的工作表
                empty_task_df = pd.DataFrame(columns=['time_slot', 'device_id', 'device_x', 'device_y',
                                                      'task_data_length_bits', 'task_data_length_Mbits'])
                empty_task_df.to_excel(writer, sheet_name='Tasks', index=False)

            # ===== 工作表2: 无人机轨迹信息 =====
            uav_data = []

            for i, (uav_name, uav_obj) in enumerate(zip(uav_names, uav_objects)):
                # 获取当前时隙位置
                current_x, current_y = None, None
                if time_slot in uav_obj.trajectory:
                    current_x, current_y = uav_obj.trajectory[time_slot]

                # 获取前一个时隙位置
                prev_slot = time_slot - 1
                prev_x, prev_y = None, None
                if prev_slot in uav_obj.trajectory:
                    prev_x, prev_y = uav_obj.trajectory[prev_slot]

                uav_info = {
                    'algorithm': uav_name,
                    'time_slot': time_slot,
                    'current_x': current_x,
                    'current_y': current_y,
                    'prev_slot': prev_slot,
                    'prev_x': prev_x,
                    'prev_y': prev_y,
                    'has_prev': 1 if prev_slot in uav_obj.trajectory else 0
                }
                uav_data.append(uav_info)

            # 保存无人机数据
            uav_df = pd.DataFrame(uav_data)
            uav_df.to_excel(writer, sheet_name='UAV_Trajectory', index=False)

        print(f"时隙 {time_slot} 的数据已保存至: {file_path}")

    print(f"\n所有指定时隙的数据已保存到: {output_folder}")
    return output_folder


# 绘制时隙中的轨迹：无人机和设备分布
# 热力图
def plot_trajectory_with_kde_from_saved_data(time_slot,
                                             label1="UAV1",
                                             label2="UAV2",
                                             label3="UAV3",
                                             label4="UAV4",
                                             label5="UAV5",
                                             label6="UAV6",
                                             filename_prefix="selected"):
    """
    从保存的数据文件中读取并绘制指定时隙的任务热力图和无人机轨迹
    参数:
        time_slot: 要绘制的时隙
        label1-label6: 无人机标签
        filename_prefix: 文件名前缀
    """
    # 构建文件路径
    filename = f"{filename_prefix}_time_slot_{time_slot}.xlsx"
    file_path = os.path.join(os.getcwd(), '..', 'data', 'task distribution and uav deployment', filename)

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 读取Excel文件
    try:
        tasks_df = pd.read_excel(file_path, sheet_name='Tasks')
        uav_df = pd.read_excel(file_path, sheet_name='UAV_Trajectory')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 创建一个新的图形
    plt.figure(figsize=(10, 8))

    # 收集所有设备的坐标和任务大小
    x_coords = []
    y_coords = []
    task_sizes = []
    all_x = []
    all_y = []

    # 从保存的数据中获取设备信息
    if not tasks_df.empty:
        for _, row in tasks_df.iterrows():
            x = row['device_x']
            y = row['device_y']
            # 获取任务大小（以MB为单位）
            task_size = row['task_data_length_bits'] / (10 ** 6)

            x_coords.append(x)
            y_coords.append(y)
            task_sizes.append(task_size)
            all_x.append(x)
            all_y.append(y)

    # 添加无人机位置到总范围计算
    # 按照传入的标签顺序收集无人机
    labels = [label1, label2, label3, label4, label5, label6]

    for label in labels:
        # 查找对应标签的无人机数据
        uav_data = uav_df[uav_df['algorithm'] == label]
        if not uav_data.empty:
            row = uav_data.iloc[0]

            # 前一个时隙位置
            if row['has_prev'] == 1 and pd.notnull(row['prev_x']) and pd.notnull(row['prev_y']):
                all_x.append(row['prev_x'])
                all_y.append(row['prev_y'])

            # 当前时隙位置
            if pd.notnull(row['current_x']) and pd.notnull(row['current_y']):
                all_x.append(row['current_x'])
                all_y.append(row['current_y'])

    # 计算坐标范围（添加10%边界）
    if all_x and all_y:
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        # 添加10%的边界
        x_min = x_min - 0.1 * x_range
        x_max = x_max + 0.1 * x_range
        y_min = y_min - 0.1 * y_range
        y_max = y_max + 0.1 * y_range
    else:
        # 如果没有数据，使用默认范围
        x_min, x_max = 0, 100
        y_min, y_max = 0, 100

    # 确保范围有效
    if x_min == x_max:
        x_min -= 5
        x_max += 5
    if y_min == y_max:
        y_min -= 5
        y_max += 5

    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 绘制任务热力图（如果有任务的话）
    if x_coords and y_coords and task_sizes:
        # 创建包含数据的DataFrame
        df = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'TaskSize': task_sizes
        })

        # 绘制核密度估计图（带任务权重）
        kde = sns.kdeplot(
            data=df,
            x='X',
            y='Y',
            weights='TaskSize',  # 使用任务大小作为权重
            fill=True,
            cmap='YlGnBu',
            alpha=0.6,
            levels=10,
            thresh=0.05,
            clip=[[x_min, x_max], [y_min, y_max]]
        )

        # 添加颜色条并设置标签
        cbar = plt.colorbar(kde.collections[0])
        cbar.set_label('Task density (MB/m²)')
        cbar.ax.tick_params(labelsize=10)

        # 定义用于格式化颜色条刻度标签的函数
        def format_func(value, tick_number):
            return f"{value * 1e6:.1f}"

        # 应用格式化函数到颜色条
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        # 在颜色条的右上角添加放缩信息
        cbar.ax.text(
            1.08, 1.02, '×10⁻⁶', transform=cbar.ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left'
        )

    # 绘制无人机的轨迹
    def plot_uav_trajectory_from_data(label, color, edgecolor):
        # 查找对应标签的无人机数据
        uav_data = uav_df[uav_df['algorithm'] == label]
        if uav_data.empty:
            return

        row = uav_data.iloc[0]

        # 检查是否有前一个时隙和当前时隙的位置
        if (row['has_prev'] == 1 and
                pd.notnull(row['prev_x']) and pd.notnull(row['prev_y']) and
                pd.notnull(row['current_x']) and pd.notnull(row['current_y'])):

            start_x, start_y = row['prev_x'], row['prev_y']
            end_x, end_y = row['current_x'], row['current_y']

            # 绘制轨迹线
            plt.plot([start_x, end_x], [start_y, end_y],
                     color=color, linestyle='-', linewidth=2, marker=None, label=label)

            # 起始位置使用三角形标记
            plt.scatter(start_x, start_y, color=color, marker='^', s=80,
                        edgecolors=edgecolor, linewidths=1.5, facecolors='none')
            # 结束位置使用圆形标记
            plt.scatter(end_x, end_y, color=color, marker='o', s=80,
                        edgecolors=edgecolor, linewidths=1.5, facecolors='none')
        # 如果没有前一个时隙，只绘制当前时隙位置
        elif pd.notnull(row['current_x']) and pd.notnull(row['current_y']):
            end_x, end_y = row['current_x'], row['current_y']
            plt.scatter(end_x, end_y, color=color, marker='o', s=80,
                        edgecolors=edgecolor, linewidths=1.5, facecolors='none',
                        label=label)

    # 定义颜色（对应：random,PSO,HOLD,DDPG,RAT,dDDM）
    colors = ['#098036', '#F29D51', '#000000', '#631879', '#B58900', '#CE3831']

    # 绘制6个无人机的轨迹
    for i, label in enumerate(labels[:6]):
        color = colors[i]
        edgecolor = colors[i]
        plot_uav_trajectory_from_data(label, color, edgecolor)

    # 添加起始位置和结束位置的图例项
    start_marker = plt.scatter([], [], c='white', marker='^', s=100,
                               edgecolors='black', linewidths=1.5, facecolors='none',
                               label='Start Position')
    end_marker = plt.scatter([], [], c='white', marker='o', s=100,
                             edgecolors='black', linewidths=1.5, facecolors='none',
                             label='End Position')

    handles, labels_legend = plt.gca().get_legend_handles_labels()
    # 将新的图例句柄加入到现有句柄列表中
    handles.extend([start_marker, end_marker])

    # 创建图例并确保唯一性
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels_legend):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # 添加起始和结束位置标记（确保只出现一次）
    if 'Start Position' not in unique_labels:
        unique_labels.append('Start Position')
        unique_handles.append(start_marker)
    if 'End Position' not in unique_labels:
        unique_labels.append('End Position')
        unique_handles.append(end_marker)

    plt.legend(handles=unique_handles, loc='best', fontsize=10)

    # 添加坐标轴标签
    plt.xlabel('X(m)', fontsize=12)
    plt.ylabel('Y(m)', fontsize=12)

    # 添加标题
    plt.title(f'Task Density and UAV Trajectories at Time Slot {time_slot}', fontsize=14)

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)



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


# 记录飞行能耗
def save_uav_flight_energy_data(uav, filename="filename"):
    """
    保存单架无人机在各时隙的飞行能耗数据。

    :param uav: 单架无人机对象
    :param filename: 输出文件名
    """
    flight_energy_data = []

    # 遍历单架无人机的飞行能耗记录
    for time_slot, energy in uav.flight_energy.items():
        flight_info = {
            'Time Slot': time_slot,
            'Flight Energy Consumption（J）': energy
        }
        flight_energy_data.append(flight_info)

    # 将收集到的数据转换为DataFrame
    df = pd.DataFrame(flight_energy_data)

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


# 绘图：不同带宽的传输时延比较
def plot_average_transmission_delay_bandwidths(bandwidths, methods, base_path=''):
    """
    根据给定的不同带宽和方法，计算并绘制不同方案下的平均传输时延。
    方法名与颜色固定绑定，不再依赖索引顺序。

    参数:
        bandwidths (list): 带宽列表，如 ['20MHz', '25MHz', '30MHz', '35MHz', '40MHz']
        methods (list): 方案名称列表，如 ['random', 'PSO','HOLD', 'KMPSO']
        base_path (str): 文件所在的基准路径，默认为空字符串表示当前目录

    返回:
        None: 绘制图表
    """

    # 方法名 -> 颜色（固定绑定）
    method_color_map = {
        'random': '#EDB120',
        'DDPG': '#B55384',
        'RAT': '#B7B7EB',
        'PSO': '#2A9D8E',
        'HOLD': '#0072BD',
        'dDDM': '#D95319',
        'KMPSO': '#D95319',
    }

    # 方法名 -> 标记形状（dDDM使用圆形，其他使用不同形状）
    method_marker_map = {
        'random': '^',  # 三角形
        'DDPG': 's',  # 正方形
        'RAT': 'D',  # 菱形
        'PSO': 'v',  # 倒三角形
        'HOLD': '<',  # 左三角形
        'KMPSO': 'o',  # 圆形（保持不变）
    }

    # 方法名 -> 图例标签
    method_label_map = {
        'random': 'random',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'PSO': 'PSO',
        'HOLD': 'HOLD',
        'KMPSO': 'dDDM',
    }

    # 存储不同方法、不同带宽下的平均传输时延
    data = {method: [] for method in methods}

    # 从多个sheet文件中读取数据，并计算平均传输时延
    def get_avg_transmission_delay(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_delays = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'transmission delay（s）' in df.columns:
                    avg_delay = df['transmission delay（s）'].mean() * 1000  # 转换为毫秒
                    all_delays.append(avg_delay)
                elif 'transmission delay (s)' in df.columns:
                    avg_delay = df['transmission delay (s)'].mean() * 1000  # 转换为毫秒
                    all_delays.append(avg_delay)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到传输时延列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_delays:
            # 返回所有sheet的平均值
            return sum(all_delays) / len(all_delays)
        else:
            return None

    # 计算每个方法在不同带宽下的平均传输时延
    for bandwidth in bandwidths:
        for method in methods:
            file_name = f'devices_{method} BANDWIDTH={bandwidth} task info.xlsx'
            full_path = os.path.join(base_path, file_name)
            avg_delay = get_avg_transmission_delay(full_path)
            if avg_delay is not None:
                data[method].append(avg_delay)
            else:
                print(f"跳过 {method} 在 {bandwidth} 下的数据")
                data[method].append(0)  # 如果没有数据，填充0

    plt.figure(figsize=(10, 6))

    # 绘图：方法名 -> 固定颜色 & 固定标签 & 固定标记形状
    for method in methods:
        color = method_color_map.get(method, None)  # 找不到时返回 None
        label = method_label_map.get(method, method)  # 找不到时用原始方法名
        marker = method_marker_map.get(method, 'o')  # 默认圆形
        plt.plot(
            [bw.replace('MHz', '') for bw in bandwidths],
            data[method],
            label=label,
            marker=marker,
            color=color
        )

    plt.xlabel('Bandwidths (MHz)')
    plt.ylabel('Average transmission delay (ms)')
    plt.title('Comparison of average transmission delay under different schemes')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


# 绘图：不同无人机飞行速度下的无人机飞行能耗比较
def plot_flight_energy_consumption_speeds(speeds, methods, base_path=''):
    """
    根据给定的不同速度和方法，计算并绘制不同方案下的无人机飞行平均能耗。

    参数：
    - speeds: 飞行速度列表，如 [5, 10, 15]
    - methods: 方法名称列表，如 ['random', 'HOLD', 'PSO', 'KMPSO']
    - base_path: 文件所在根路径
    """
    # 存储不同方法、不同速度下的平均飞行能耗
    data = {method: [] for method in methods}

    # 从多个sheet文件中读取数据，并计算平均飞行能耗
    def get_avg_flight_energy(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_energies = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'Flight Energy Consumption（J）' in df.columns:
                    avg_energy = df['Flight Energy Consumption（J）'].mean()  # 计算平均能耗
                    all_energies.append(avg_energy)
                elif 'Flight Energy Consumption (J)' in df.columns:
                    avg_energy = df['Flight Energy Consumption (J)'].mean()  # 计算平均能耗
                    all_energies.append(avg_energy)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到飞行能耗列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_energies:
            # 返回所有sheet的平均值
            return sum(all_energies) / len(all_energies)
        else:
            return None

    # 计算每个方法在不同速度下的平均飞行能耗
    for speed in speeds:
        for method in methods:
            file_name = f'uav_{method} speed={speed}m flight energy record.xlsx'
            full_path = os.path.join(base_path, file_name)
            avg_energy = get_avg_flight_energy(full_path)
            if avg_energy is not None:
                data[method].append(avg_energy)
            else:
                print(f"跳过 {method} 在 {speed}m/s 下的数据")
                data[method].append(0)  # 如果没有数据，填充0

    # 绘图设置：增加组间间隔
    n_methods = len(methods)
    width = 0.25  # 柱子宽度
    group_gap = 0.2  # 组之间的空白间距
    group_width = n_methods * width  # 每组总宽度

    # 计算每组的起始位置（留出组间空隙）
    x_positions = []
    current_pos = 0
    for i in range(len(speeds)):
        group_pos = [current_pos + j * width for j in range(n_methods)]
        x_positions.append(group_pos)
        current_pos += group_width + group_gap

    # 颜色和标签一一对应
    color_map = {
        'random': '#DCE4E8',
        'HOLD': '#EAECE6',
        'DDPG': '#B6B6B4',
        'RAT': '#E8F5E9',
        'PSO': '#FFDFB8',
        'KMPSO': '#FF9494'
    }
    label_map = {
        'random': 'random',
        'HOLD': 'HOLD',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'PSO': 'PSO',
        'KMPSO': 'dDDM'
    }

    # 填充样式映射（dDDM不填充，其他使用不同线型填充）
    hatch_map = {
        'random': '////',  # 斜线
        'HOLD': '\\\\\\\\',  # 反斜线
        'DDPG': '||||',  # 垂直线
        'RAT': '----',  # 水平线
        'PSO': '++++',  # 十字线
        'KMPSO': 'oooo'  # dDDM不填充
    }

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        method_data = data[method]
        xs = [x_positions[j][i] for j in range(len(speeds))]
        ax.bar(xs, method_data, width=width, label=label_map[method],
               color=color_map[method], hatch=hatch_map[method], edgecolor='black')

    # 设置横坐标
    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f'{speed}' for speed in speeds])
    ax.set_xlabel('Speeds (m/s)')
    ax.set_ylabel('Average flight energy consumption (J)')
    ax.set_title('Comparison of average flight energy consumption under different schemes')

    # 设置纵坐标宽度
    max_energy = max([max(data[method]) for method in methods if data[method]])
    y_max = ((max_energy // 500) + 2) * 500
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 500))

    ax.legend()
    plt.grid(True, axis='y')  # y方向网格线
    plt.tight_layout()
    plt.show(block=False)


# 绘图：不同eta参数下的无人机飞行能耗比较
def plot_flight_energy_consumption_etas(etas, methods, base_path=''):
    """
    根据给定的不同eta（飞行能耗惩罚系数）和方法，计算并绘制不同方案下的无人机飞行平均能耗。
    参数:
        etas (list): eta参数列表，如 [0.0001, 0.0003, 0.0005(默认值), 0.0007, 0.0009]
        methods (list): 方法名称列表，如 ['random', 'HOLD', 'PSO', 'KMPSO']
        base_path (str): 文件所在根路径
    """
    # 存储不同方法、不同eta下的平均飞行能耗
    data = {method: [] for method in methods}

    # 从多个sheet文件中读取数据，并计算平均飞行能耗
    def get_avg_flight_energy(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_energies = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'Flight Energy Consumption（J）' in df.columns:
                    avg_energy = df['Flight Energy Consumption（J）'].mean()  # 计算平均能耗
                    all_energies.append(avg_energy)
                elif 'Flight Energy Consumption (J)' in df.columns:
                    avg_energy = df['Flight Energy Consumption (J)'].mean()  # 计算平均能耗
                    all_energies.append(avg_energy)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到飞行能耗列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_energies:
            # 返回所有sheet的平均值
            return sum(all_energies) / len(all_energies)
        else:
            return None

    # 计算每个方法在不同eta下的平均飞行能耗
    for eta in etas:
        for method in methods:
            # 构建文件名，使用精确的eta值
            if 'e' in str(eta):
                # 将科学计数法转换为十进制小数
                eta_str = f"{eta:.10f}".rstrip('0').rstrip('.')
            else:
                eta_str = str(eta)
            file_name = f'uav_{method} eta={eta_str} flight energy record.xlsx'
            full_path = os.path.join(base_path, file_name)
            avg_energy = get_avg_flight_energy(full_path)
            if avg_energy is not None:
                data[method].append(avg_energy)
            else:
                print(f"跳过 {method} 在 eta={eta} 下的数据")
                data[method].append(0)  # 如果没有数据，填充0

    # 绘图设置：增加组间间隔
    n_methods = len(methods)
    width = 0.25  # 柱子宽度
    group_gap = 0.2  # 组之间的空白间距
    group_width = n_methods * width  # 每组总宽度

    # 计算每组的起始位置（留出组间空隙）
    x_positions = []
    current_pos = 0
    for i in range(len(etas)):
        group_pos = [current_pos + j * width for j in range(n_methods)]
        x_positions.append(group_pos)
        current_pos += group_width + group_gap

    color_map = {
        'random': '#DCE4E8',
        'HOLD': '#EAECE6',
        'DDPG': '#B6B6B4',
        'RAT': '#E8F5E9',
        'PSO': '#FFDFB8',
        'KMPSO': '#FF9494'
    }
    label_map = {
        'random': 'random',
        'HOLD': 'HOLD',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'PSO': 'PSO',
        'KMPSO': 'dDDM'
    }

    hatch_map = {
        'random': '////',  # 斜线
        'HOLD': '\\\\\\\\',  # 反斜线
        'DDPG': '||||',  # 垂直线
        'RAT': '----',  # 水平线
        'PSO': '++++',  # 十字线
        'KMPSO': 'oooo'  # dDDM填充
    }

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        method_data = data[method]
        xs = [x_positions[j][i] for j in range(len(etas))]
        ax.bar(xs, method_data, width=width, label=label_map[method],
               color=color_map[method], hatch=hatch_map[method], edgecolor='black')

    # 设置横坐标
    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)

    # 简化横坐标标签：将eta乘以10^4后显示1,3,5,7,9
    eta_labels = []
    for eta in etas:
        # 将eta值乘以10^4，因为0.0001 * 10000 = 1
        scaled_value = eta * 1e4
        # 确保是整数
        if abs(scaled_value - round(scaled_value)) < 1e-10:
            label = f"{int(round(scaled_value))}"
        else:
            label = f"{scaled_value:.1f}"
        eta_labels.append(label)

    ax.set_xticklabels(eta_labels)

    # 在横坐标轴标题中添加10⁻⁴（因为乘以的是10^4，所以要显示10⁻⁴）
    ax.set_xlabel('η values(×10⁻⁴)', fontsize=12)
    ax.set_ylabel('Average flight energy consumption (J)', fontsize=12)
    ax.set_title('Comparison of average flight Energy consumption under different η values', fontsize=14)

    # 设置纵坐标宽度
    all_energies = []
    for method in methods:
        for energy in data[method]:
            if energy > 0:
                all_energies.append(energy)

    if all_energies:
        max_energy = max(all_energies)
        y_max = ((max_energy // 500) + 2) * 500
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, 500))
    else:
        # 如果没有有效数据，设置默认范围
        ax.set_ylim(0, 2000)
        ax.set_yticks(np.arange(0, 2001, 500))

    ax.legend()
    plt.grid(True, axis='y')  # y方向网格线
    plt.tight_layout()
    plt.show(block=False)


# 绘图：不同设备数量下的平均卸载比例
def plot_offload_ratio(devices_numbers, methods, base_path=''):
    """
    绘制不同设备数量下的平均卸载比例。
    方法名与颜色/标签固定绑定。
    """
    # 固定映射
    method_color_map = {
        'MATS': '#909291',
        'HOLD': '#2A9D8E',
        'random': '#EDB120',
        'DDPG': '#B55384',
        'RAT': '#B7B7EB',
        'BKA': '#0072BD',
        'IBKA': '#D95319',
    }

    # 标记形状映射
    method_marker_map = {
        'MATS': '^',  # 三角形
        'HOLD': 's',  # 正方形
        'random': 'D',  # 菱形
        'DDPG': 'v',  # 倒三角形
        'RAT': '<',  # 左三角形
        'BKA': '>',  # 右三角形
        'IBKA': 'o',  # 圆形（dDDM保持不变）
    }

    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }

    data = {m: [] for m in methods}

    def get_avg_offload_ratio(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_ratios = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'offload ratio' in df.columns:
                    avg_ratio = df['offload ratio'].mean()
                    all_ratios.append(avg_ratio)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到卸载比例列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_ratios:
            # 返回所有sheet的平均值
            return sum(all_ratios) / len(all_ratios)
        else:
            return None

    for dn in devices_numbers:
        for m in methods:
            fn = f'devices_{m} DEVICES={dn} task info.xlsx'
            fp = os.path.join(base_path, fn)
            ratio = get_avg_offload_ratio(fp)
            if ratio is not None:
                data[m].append(ratio)
            else:
                print(f"跳过 {m} 在 {dn} 设备下的数据")
                data[m].append(0)  # 如果没有数据，填充0

    plt.figure(figsize=(10, 6))
    for m in methods:
        plt.plot(devices_numbers,
                 data[m],
                 label=method_label_map.get(m, m),
                 marker=method_marker_map.get(m, 'o'),
                 color=method_color_map.get(m, None))
    plt.xticks(devices_numbers)
    plt.xlabel('Number of devices')
    plt.ylabel('Average offload ratio')
    plt.title('Comparison of average offload ratio under different schemes')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


# 绘图：不同设备数量下的平均总时延
def plot_total_delay(devices_numbers, methods, base_path=''):
    """
    绘制不同设备数量下的平均总时延。
    方法名与颜色/标签固定绑定。
    """
    method_color_map = {
        'MATS': '#909291',
        'HOLD': '#2A9D8E',
        'random': '#EDB120',
        'DDPG': '#B55384',
        'RAT': '#B7B7EB',
        'BKA': '#0072BD',
        'IBKA': '#D95319',
    }

    # 标记形状映射
    method_marker_map = {
        'MATS': '^',  # 三角形
        'HOLD': 's',  # 正方形
        'random': 'D',  # 菱形
        'DDPG': 'v',  # 倒三角形
        'RAT': '<',  # 左三角形
        'BKA': '>',  # 右三角形
        'IBKA': 'o',  # 圆形（dDDM保持不变）
    }

    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }

    data = {m: [] for m in methods}

    def get_avg_total_delay(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_delays = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'total delay (s)' in df.columns:
                    avg_delay = df['total delay (s)'].mean()
                    all_delays.append(avg_delay)
                elif 'total delay（s）' in df.columns:
                    avg_delay = df['total delay（s）'].mean()
                    all_delays.append(avg_delay)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到总时延列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_delays:
            # 返回所有sheet的平均值
            return sum(all_delays) / len(all_delays)
        else:
            return None

    for dn in devices_numbers:
        for m in methods:
            fn = f'devices_{m} DEVICES={dn} task info.xlsx'
            fp = os.path.join(base_path, fn)
            delay = get_avg_total_delay(fp)
            if delay is not None:
                data[m].append(delay)
            else:
                print(f"跳过 {m} 在 {dn} 设备下的数据")
                data[m].append(0)  # 如果没有数据，填充0

    plt.figure(figsize=(10, 6))
    for m in methods:
        plt.plot(devices_numbers,
                 data[m],
                 label=method_label_map.get(m, m),
                 marker=method_marker_map.get(m, 'o'),
                 color=method_color_map.get(m, None))
    plt.xticks(devices_numbers)
    plt.xlabel('Number of devices')
    plt.ylabel('Average total delay (s)')
    plt.title('Comparison of average total delay under different schemes')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


# 绘图：不同设备数量下的平均总能耗
def plot_total_energy_consumption(devices_numbers, methods, base_path=''):
    """
    绘制不同设备数量下的平均总能耗（柱状图）。
    方法名与颜色/标签固定绑定。
    """
    method_color_map = {
        'MATS': '#d6ccc2',
        'HOLD': '#EAECE6',
        'random': '#DCE4E8',
        'DDPG': '#B6B6B4',
        'RAT': '#E8F5E9',
        'BKA': '#FFDFB8',
        'IBKA': '#FF9494',
    }

    # 填充样式映射
    hatch_map = {
        'MATS': '////',  # 斜线
        'HOLD': '\\\\\\\\',  # 反斜线
        'random': '||||',  # 垂直线
        'DDPG': '----',  # 水平线
        'RAT': '++++',  # 十字线
        'BKA': 'xxxx',  # 交叉线
        'IBKA': 'oooo'  # dDDM不填充
    }

    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }

    data = {m: [] for m in methods}

    def get_avg_total_energy(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_energies = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'energy consumption (J)' in df.columns:
                    avg_energy = df['energy consumption (J)'].mean()
                    all_energies.append(avg_energy)
                elif 'energy consumption (J）' in df.columns:
                    avg_energy = df['energy consumption (J）'].mean()
                    all_energies.append(avg_energy)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到能耗列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_energies:
            # 返回所有sheet的平均值
            return sum(all_energies) / len(all_energies)
        else:
            return None

    for dn in devices_numbers:
        for m in methods:
            fn = f'devices_{m} DEVICES={dn} task info.xlsx'
            fp = os.path.join(base_path, fn)
            energy = get_avg_total_energy(fp)
            if energy is not None:
                data[m].append(energy)
            else:
                print(f"跳过 {m} 在 {dn} 设备下的数据")
                data[m].append(0)  # 如果没有数据，填充0

    # 柱状图布局参数
    n_methods = len(methods)
    width = 0.25
    group_gap = 0.2
    group_width = n_methods * width
    x_positions = []
    current_pos = 0
    for _ in devices_numbers:
        x_positions.append([current_pos + j * width for j in range(n_methods)])
        current_pos += group_width + group_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(methods):
        xs = [x_positions[j][i] for j in range(len(devices_numbers))]
        ax.bar(xs,
               data[m],
               width=width,
               label=method_label_map.get(m, m),
               color=method_color_map.get(m, None),
               hatch=hatch_map.get(m, ''),
               edgecolor='black')

    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(d) for d in devices_numbers])
    ax.set_xlabel('Number of devices')
    ax.set_ylabel('Average total energy consumption (J)')
    ax.set_title('Comparison of Average Total Energy Consumption Under Different Schemes')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


# 绘图：不同ALPHA参数下的总时延比较
def plot_total_delay_alphas(alphas, methods, base_path=''):
    """
    根据给定的不同ALPHA参数和方法，计算并绘制不同方案下的平均总时延。
    方法名与颜色固定绑定，不再依赖索引顺序。

    参数:
        alphas (list): ALPHA参数列表，如 [0.2, 0.4, 0.6, 0.8, 1.0]
        methods (list): 方案名称列表，如 ['random', 'BKA', 'IBKA', 'HOLD', 'MATS', 'DDPG', 'RAT']
        base_path (str): 文件所在的基准路径，默认为空字符串表示当前目录

    返回:
        None: 绘制图表
    """

    # 方法名 -> 颜色（固定绑定）
    method_color_map = {
        'MATS': '#909291',
        'HOLD': '#2A9D8E',
        'random': '#EDB120',
        'DDPG': '#B55384',
        'RAT': '#B7B7EB',
        'BKA': '#0072BD',
        'IBKA': '#D95319',
    }

    # 方法名 -> 标记形状
    method_marker_map = {
        'MATS': '^',  # 三角形
        'HOLD': 's',  # 正方形
        'random': 'D',  # 菱形
        'DDPG': 'v',  # 倒三角形
        'RAT': '<',  # 左三角形
        'BKA': '>',  # 右三角形
        'IBKA': 'o',  # 圆形
    }

    # 方法名 -> 图例标签
    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }

    # 存储不同方法、不同ALPHA参数下的平均总时延
    data = {method: [] for method in methods}

    # 从多个sheet文件中读取数据，并计算平均总时延
    def get_avg_total_delay(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_delays = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 确保列名正确
                if 'total delay (s)' in df.columns:
                    avg_delay = df['total delay (s)'].mean()
                    all_delays.append(avg_delay)
                elif 'total delay（s）' in df.columns:  # 检查中文括号格式
                    avg_delay = df['total delay（s）'].mean()
                    all_delays.append(avg_delay)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到总时延列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_delays:
            # 返回所有sheet的平均值
            return sum(all_delays) / len(all_delays)
        else:
            return None

    # 计算每个方法在不同ALPHA参数下的平均总时延
    for alpha in alphas:
        for method in methods:
            # 构建文件名
            file_name = f'devices_{method} ALPHA={alpha} BETA=0.8 task info.xlsx'
            full_path = os.path.join(base_path, file_name)
            avg_delay = get_avg_total_delay(full_path)
            if avg_delay is not None:
                data[method].append(avg_delay)
            else:
                print(f"跳过 {method} 在 ALPHA={alpha} 下的数据")
                data[method].append(0)  # 如果没有数据，填充0

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘图：方法名 -> 固定颜色 & 固定标签 & 固定标记形状
    for method in methods:
        color = method_color_map.get(method, None)
        label = method_label_map.get(method, method)
        marker = method_marker_map.get(method, 'o')

        plt.plot(
            alphas,
            data[method],
            label=label,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8
        )

    # 设置图表标题和标签
    plt.xlabel('α values', fontsize=12)
    plt.ylabel('Average total delay (s)', fontsize=12)
    plt.title('Comparison of Average Total Delay Under Different α Values', fontsize=14)

    # 设置横坐标刻度
    plt.xticks(alphas, [str(alpha) for alpha in alphas])

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图例
    plt.legend(loc='best', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show(block=False)


# 绘图：不同BETA参数下的平均总能耗比较
def plot_total_energy_consumption_betas(betas, methods, base_path=''):
    """
    绘制不同 BETA 值下的平均总能耗（柱状图）。
    方法名与颜色/标签/填充样式固定绑定。
    假设数据文件名为：
        devices_<method> ALPHA=0.2 BETA=<beta> task info.xlsx
    其中 <beta> 取 0.2、0.4、0.6、0.8、1.0
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # ---------------------------- 样式表 ----------------------------
    method_color_map = {
        'MATS': '#d6ccc2',
        'HOLD': '#EAECE6',
        'random': '#DCE4E8',
        'DDPG': '#B6B6B4',
        'RAT': '#E8F5E9',
        'BKA': '#FFDFB8',
        'IBKA': '#FF9494',
    }
    hatch_map = {
        'MATS': '////',
        'HOLD': '\\\\\\\\',
        'random': '||||',
        'DDPG': '----',
        'RAT': '++++',
        'BKA': 'xxxx',
        'IBKA': 'oooo'
    }
    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }
    # -----------------------------------------------------------------

    # 按 BETA 值组织数据
    data = {m: [] for m in methods}

    def get_avg_total_energy(file_path):
        if not os.path.exists(file_path):
            print(f'警告：无法找到文件 {file_path}')
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f'读取文件失败 {file_path}: {e}')
            return None

        all_energies = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 检查列名
                if 'energy consumption (J)' in df.columns:
                    avg_energy = df['energy consumption (J)'].mean()
                    all_energies.append(avg_energy)
                elif 'energy consumption (J）' in df.columns:
                    avg_energy = df['energy consumption (J）'].mean()
                    all_energies.append(avg_energy)
                else:
                    print(f'警告：sheet {sheet_name} 中未找到能耗列')
            except Exception as e:
                print(f'读取sheet {sheet_name} 失败: {e}')

        if all_energies:
            # 返回所有sheet的平均值
            return sum(all_energies) / len(all_energies)
        else:
            return None

    # 拼文件名：固定 ALPHA=0.2，只变 BETA
    for beta in betas:
        for m in methods:
            fn = f'devices_{m} ALPHA=0.2 BETA={beta} task info.xlsx'
            fp = os.path.join(base_path, fn)
            energy = get_avg_total_energy(fp)
            data[m].append(energy if energy is not None else 0.)

    # ---------------------------- 画图 ----------------------------
    n_methods = len(methods)
    width = 0.25
    group_gap = 0.2
    group_width = n_methods * width
    x_positions = []
    current_pos = 0
    for _ in betas:
        x_positions.append([current_pos + j * width for j in range(n_methods)])
        current_pos += group_width + group_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(methods):
        xs = [x_positions[j][i] for j in range(len(betas))]
        ax.bar(xs,
               data[m],
               width=width,
               label=method_label_map.get(m, m),
               color=method_color_map.get(m, None),
               hatch=hatch_map.get(m, ''),
               edgecolor='black')

    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_xlabel('β values')
    ax.set_ylabel('Average total energy consumption (J)')
    ax.set_title('Comparison of Average Total Energy Consumption Under Different β Values')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show(block=False)


# 绘图：不同LAMBDA_i参数下的平均总时延比较
def plot_total_delay_lambda(lambdas, methods, base_path=''):
    """
    绘制不同LAMBDA_i参数下的平均总时延比较图。
    横坐标将tasks/s转换为tasks/min。

    参数:
        lambdas (list): LAMBDA_i参数列表，如 [0.2, 0.4, 0.6, 0.8, 1.0] (tasks/s)
        methods (list): 方案名称列表，如 ['random', 'BKA', 'IBKA', 'HOLD', 'MATS', 'DDPG', 'RAT']
        base_path (str): 文件所在的基准路径，默认为空字符串表示当前目录

    返回:
        None: 绘制图表
    """

    # 方法名 -> 颜色（固定绑定）
    method_color_map = {
        'MATS': '#909291',
        'HOLD': '#2A9D8E',
        'random': '#EDB120',
        'DDPG': '#B55384',
        'RAT': '#B7B7EB',
        'BKA': '#0072BD',
        'IBKA': '#D95319',
    }

    # 方法名 -> 标记形状
    method_marker_map = {
        'MATS': '^',  # 三角形
        'HOLD': 's',  # 正方形
        'random': 'D',  # 菱形
        'DDPG': 'v',  # 倒三角形
        'RAT': '<',  # 左三角形
        'BKA': '>',  # 右三角形
        'IBKA': 'o',  # 圆形
    }

    # 方法名 -> 图例标签
    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }

    # 存储不同方法、不同LAMBDA_i参数下的平均总时延
    data = {method: [] for method in methods}

    # 从多个sheet文件中读取数据，并计算平均总时延
    def get_avg_total_delay(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_delays = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 确保列名正确
                if 'total delay (s)' in df.columns:
                    avg_delay = df['total delay (s)'].mean()
                    all_delays.append(avg_delay)
                elif 'total delay（s）' in df.columns:  # 检查中文括号格式
                    avg_delay = df['total delay（s）'].mean()
                    all_delays.append(avg_delay)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到总时延列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_delays:
            # 返回所有sheet的平均值
            return sum(all_delays) / len(all_delays)
        else:
            return None

    # 计算每个方法在不同LAMBDA_i参数下的平均总时延
    for lambda_val in lambdas:
        for method in methods:
            # 构建文件名，lambda_i值格式化
            lambda_str = f"{lambda_val:.1f}" if isinstance(lambda_val, float) else str(lambda_val)
            file_name = f'devices_{method} lambda_i={lambda_str} task info.xlsx'
            full_path = os.path.join(base_path, file_name)
            avg_delay = get_avg_total_delay(full_path)
            if avg_delay is not None:
                data[method].append(avg_delay)
            else:
                print(f"跳过 {method} 在 lambda_i={lambda_val} 下的数据")
                data[method].append(0)  # 如果没有数据，填充0

    # 将lambda_i从tasks/s转换为tasks/min
    lambdas_per_min = [lam * 60 for lam in lambdas]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘图：方法名 -> 固定颜色 & 固定标签 & 固定标记形状
    for method in methods:
        color = method_color_map.get(method, None)
        label = method_label_map.get(method, method)
        marker = method_marker_map.get(method, 'o')

        plt.plot(
            lambdas_per_min,
            data[method],
            label=label,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8
        )

    # 设置图表标题和标签

    plt.xlabel(r'$\lambda_{i,\tau_{j}}$ (tasks/min)', fontsize=12)
    plt.ylabel('Average total delay (s)', fontsize=12)
    plt.title('Comparison of Average Total Delay Under Different Task Arrival Rates', fontsize=14)

    # 设置横坐标刻度
    plt.xticks(lambdas_per_min, [str(int(lam)) for lam in lambdas_per_min])

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图例
    plt.legend(loc='best', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show(block=False)


# 绘图：不同LAMBDA_i参数下的平均总能耗比较
def plot_total_energy_consumption_lambda(lambdas, methods, base_path=''):
    """
    绘制不同LAMBDA_i参数下的平均总能耗比较图（柱状图）。
    横坐标将tasks/s转换为tasks/min。

    参数:
        lambdas (list): LAMBDA_i参数列表，如 [0.2, 0.4, 0.6, 0.8, 1.0] (tasks/s)
        methods (list): 方案名称列表，如 ['random', 'BKA', 'IBKA', 'HOLD', 'MATS', 'DDPG', 'RAT']
        base_path (str): 文件所在的基准路径，默认为空字符串表示当前目录

    返回:
        None: 绘制图表
    """

    # 方法名 -> 颜色（固定绑定）
    method_color_map = {
        'MATS': '#d6ccc2',
        'HOLD': '#EAECE6',
        'random': '#DCE4E8',
        'DDPG': '#B6B6B4',
        'RAT': '#E8F5E9',
        'BKA': '#FFDFB8',
        'IBKA': '#FF9494',
    }

    # 填充样式映射
    hatch_map = {
        'MATS': '////',  # 斜线
        'HOLD': '\\\\\\\\',  # 反斜线
        'random': '||||',  # 垂直线
        'DDPG': '----',  # 水平线
        'RAT': '++++',  # 十字线
        'BKA': 'xxxx',  # 交叉线
        'IBKA': 'oooo'  # dDDM不填充
    }

    # 方法名 -> 图例标签
    method_label_map = {
        'MATS': 'MATS',
        'HOLD': 'HOLD',
        'random': 'RTORA',
        'DDPG': 'DDPG',
        'RAT': 'RAT',
        'BKA': 'BKA',
        'IBKA': 'dDDM',
    }

    # 存储不同方法、不同LAMBDA_i参数下的平均总能耗
    data = {method: [] for method in methods}

    # 从多个sheet文件中读取数据，并计算平均总能耗
    def get_avg_total_energy(file_path):
        if not os.path.exists(file_path):
            print(f"警告：无法找到文件 {file_path}")
            return None

        # 读取所有sheet
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

        all_energies = []

        # 遍历每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 确保列名正确
                if 'energy consumption (J)' in df.columns:
                    avg_energy = df['energy consumption (J)'].mean()
                    all_energies.append(avg_energy)
                elif 'energy consumption (J）' in df.columns:
                    avg_energy = df['energy consumption (J）'].mean()
                    all_energies.append(avg_energy)
                else:
                    print(f"警告：sheet {sheet_name} 中未找到能耗列")
            except Exception as e:
                print(f"读取sheet {sheet_name} 失败: {e}")

        if all_energies:
            # 返回所有sheet的平均值
            return sum(all_energies) / len(all_energies)
        else:
            return None

    # 计算每个方法在不同LAMBDA_i参数下的平均总能耗
    for lambda_val in lambdas:
        for method in methods:
            # 构建文件名，lambda_i值格式化
            lambda_str = f"{lambda_val:.1f}" if isinstance(lambda_val, float) else str(lambda_val)
            file_name = f'devices_{method} lambda_i={lambda_str} task info.xlsx'
            full_path = os.path.join(base_path, file_name)
            avg_energy = get_avg_total_energy(full_path)
            if avg_energy is not None:
                data[method].append(avg_energy)
            else:
                print(f"跳过 {method} 在 lambda_i={lambda_val} 下的数据")
                data[method].append(0)  # 如果没有数据，填充0

    # 将lambda_i从tasks/s转换为tasks/min
    lambdas_per_min = [lam * 60 for lam in lambdas]

    # 柱状图布局参数
    n_methods = len(methods)
    width = 0.25  # 柱子宽度
    group_gap = 0.2  # 组之间的空白间距
    group_width = n_methods * width  # 每组总宽度

    # 计算每组的起始位置（留出组间空隙）
    x_positions = []
    current_pos = 0
    for _ in lambdas_per_min:
        group_pos = [current_pos + j * width for j in range(n_methods)]
        x_positions.append(group_pos)
        current_pos += group_width + group_gap

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        xs = [x_positions[j][i] for j in range(len(lambdas_per_min))]
        ax.bar(xs,
               data[method],
               width=width,
               label=method_label_map.get(method, method),
               color=method_color_map.get(method, None),
               hatch=hatch_map.get(method, ''),
               edgecolor='black')

    # 设置横坐标
    group_centers = [sum(pos) / len(pos) for pos in x_positions]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(int(lam)) for lam in lambdas_per_min])

    ax.set_xlabel(r'$\lambda_{i,\tau_{j}}$ (tasks/min)', fontsize=12)
    ax.set_ylabel('Average total energy consumption (J)', fontsize=12)
    ax.set_title('Comparison of Average Total Energy Consumption Under Different Task Arrival Rates', fontsize=14)

    # 格式化纵坐标刻度标签
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))

    # 显示图例
    ax.legend(loc='best', fontsize=10)

    # 添加网格
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show(block=False)


if __name__ == '__main__':
    # 动态获取 data 文件夹的绝对路径
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 获取src所在目录的父目录

    """轨迹方案比较"""
    # 不同带宽下的传输比较
    bandwidths = ['20MHz', '25MHz', '30MHz', '35MHz', '40MHz']
    methods = ['random', 'HOLD', 'DDPG', 'RAT', 'PSO', 'KMPSO']

    data_folder_bandwidths = os.path.join(parent_directory, 'data/bandwidths')
    plot_average_transmission_delay_bandwidths(bandwidths, methods, base_path=data_folder_bandwidths)

    # 不同飞行速度下的飞行能耗比较
    speeds = [20, 25, 30, 35, 40]
    data_folder_speeds = os.path.join(parent_directory, 'data/uav flight speeds')
    plot_flight_energy_consumption_speeds(speeds, methods, base_path=data_folder_speeds)

    """卸载和资源分配方案比较"""
    # 不同设备数量下的时延、卸载比例、能耗比较
    methods = ['MATS', 'HOLD', 'random', 'DDPG', 'RAT', 'BKA', 'IBKA']
    devices_numbers = [20, 30, 40, 50, 60]
    data_folder_device_numbers = os.path.join(parent_directory, 'data/device number')

    plot_offload_ratio(devices_numbers, methods, base_path=data_folder_device_numbers)
    plot_total_delay(devices_numbers, methods, base_path=data_folder_device_numbers)
    plot_total_energy_consumption(devices_numbers, methods, base_path=data_folder_device_numbers)

    """任务分布和部署比较"""
    # 从保存的数据中绘制指定时隙的图
    selected_time_slots = [5, 10, 20, 30]

    # 绘制每个时隙的图
    for time_slot in selected_time_slots:
        try:
            # 调用绘图函数，保持和原函数相同的参数顺序
            plot_trajectory_with_kde_from_saved_data(
                time_slot=time_slot,
                label1="random",
                label2="PSO",
                label3="HOLD",
                label4="DDPG",
                label5="RAT",
                label6="dDDM",
                filename_prefix="task distribution and uav deployment"
            )
        except Exception as e:
            print(f"绘制时隙 {time_slot} 失败: {e}")
            import traceback

            traceback.print_exc()

    """ALPHA参数对比"""
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    methods = ['MATS', 'HOLD', 'random', 'DDPG', 'RAT', 'BKA', 'IBKA']
    data_folder_alpha = os.path.join(parent_directory, 'data/ALPHA')
    plot_total_delay_alphas(alphas, methods, base_path=data_folder_alpha)

    """BETA参数对比"""
    betas = [0.2, 0.4, 0.6, 0.8, 1.0]
    methods = ['MATS', 'HOLD', 'random', 'DDPG', 'RAT', 'BKA', 'IBKA']
    data_folder_beta = os.path.join(parent_directory, 'data/BETA')
    plot_total_energy_consumption_betas(betas, methods, base_path=data_folder_beta)

    """ETA参数对比"""
    etas = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
    methods = ['random', 'HOLD', 'DDPG', 'RAT', 'PSO', 'KMPSO']
    data_folder_etas = os.path.join(parent_directory, 'data/ETA')
    plot_flight_energy_consumption_etas(etas, methods, base_path=data_folder_etas)

    """LAMBDA_i参数对比"""
    methods = ['MATS', 'HOLD', 'random', 'DDPG', 'RAT', 'BKA', 'IBKA']
    lambda_i = [0.2, 0.4, 0.6, 0.8, 1.0]
    data_folder_lambda = os.path.join(parent_directory, 'data/LAMBDA_i')

    plot_total_delay_lambda(lambda_i, methods, base_path=data_folder_lambda)
    plot_total_energy_consumption_lambda(lambda_i, methods, base_path=data_folder_lambda)

    plt.show(block=True)