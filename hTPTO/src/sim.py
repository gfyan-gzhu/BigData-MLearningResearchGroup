import os
import sys
import copy
import numpy as np
import torch
import random
import pandas as pd

# 确保项目路径可导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import simEnvParameter
from coordinateManage import get_coordinate, change_coordinate
from utils import *
from uavTrajectoryDecision import DDPGTrainer, preprocess_ddpg_state, postprocess_ddpg_action
from OffloadingResourceDecision import MultiTaskAC, preprocess_ac_state, postprocess_ac_action
from UAVNumberOptimize import optimize_uav_number
from draw import *   # 包含已有的保存函数，我们将补充两个新的保存函数

# 导入各算法类
from Hierarchical_MADDPG_AC import HierarchicalRLAlgorithm
from StandardDDPG import StandardDDPGTrainer, preprocess_standard_state
from DDPG_LDPG import DDPG_LDPG_Algorithm
from PPO import PPO, postprocess_ppo_action

# 设置随机种子以保证可重复性（但环境随机部分由基准控制，此种子影响基准生成）
SEED = 77
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# 设备选择
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 新增保存函数 ====================
def save_subtask_offload_info(devices, folder, filename):
    """保存每个子任务的数据大小和是否卸载计算标志"""
    data = []
    for dev in devices:
        for time_slot, task in dev.task.items():
            if task is None:
                continue
            for subtask in task.subtasks:
                offloaded = 1 if hasattr(subtask, 'uav_compute') and subtask.uav_compute else 0
                data.append({
                    'device_id': dev.id,
                    'time_slot': time_slot,
                    'task_id': id(task),
                    'subtask_index': subtask.subtask_id,
                    'data_length_bits': subtask.data_length,
                    'offloaded': offloaded,
                    'uav_index': subtask.uav_offload_location if offloaded else -1
                })
    df = pd.DataFrame(data)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    df.to_excel(filepath, index=False)
    print(f"子任务卸载信息已保存至: {filepath}")

def save_subtask_time_energy_data(devices, folder, filename):
    """保存每个子任务的完成时延和所耗计算资源（能耗）"""
    data = []
    for dev in devices:
        for time_slot, task in dev.task.items():
            if task is None:
                continue
            for subtask in task.subtasks:
                if hasattr(subtask, 'local_compute') and subtask.local_compute:
                    energy = subtask.device_compute_energy if hasattr(subtask, 'device_compute_energy') else 0.0
                else:
                    energy = subtask.uav_compute_energy if hasattr(subtask, 'uav_compute_energy') else 0.0
                generate_time = getattr(subtask, 'generate_time', 0.0)
                finished_time = getattr(subtask, 'finished_time', 0.0)
                delay = finished_time - generate_time if finished_time > 0 else 0.0
                data.append({
                    'device_id': dev.id,
                    'time_slot': time_slot,
                    'task_id': id(task),
                    'subtask_index': subtask.subtask_id,
                    'delay': delay,
                    'compute_energy': energy,
                    'offloaded': 1 if (hasattr(subtask, 'uav_compute') and subtask.uav_compute) else 0,
                    'data_length_bits': subtask.data_length
                })
    df = pd.DataFrame(data)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    df.to_excel(filepath, index=False)
    print(f"子任务时延能耗数据已保存至: {filepath}")

def save_uav_trajectories(all_uavs, folder, filename):
    """保存所有无人机的轨迹（每个时隙的坐标）"""
    data = []
    for uav in all_uavs:
        for time_slot, coord in uav.trajectory.items():
            x, y, z = coord[0], coord[1], coord[2] if len(coord) > 2 else 0.0
            data.append({
                'uav_id': uav.uav_id,
                'time_slot': time_slot,
                'x': x,
                'y': y,
                'z': z
            })
    df = pd.DataFrame(data)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    df.to_excel(filepath, index=False)
    print(f"无人机轨迹已保存至: {filepath}")

def save_task_info(devices, folder, filename):
    """保存每个任务的信息：时隙、设备ID、任务数据大小、设备位置（x,y）"""
    data = []
    for dev in devices:
        for time_slot, task in dev.task.items():
            if task is None:
                continue
            if time_slot in dev.trajectory:
                pos = dev.trajectory[time_slot]
                x, y = pos[0], pos[1]
            else:
                x, y = dev.coordinate[0], dev.coordinate[1]
            data.append({
                'device_id': dev.id,
                'time_slot': time_slot,
                'task_data_length_bits': task.data_length,
                'device_x': x,
                'device_y': y
            })
    df = pd.DataFrame(data)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    df.to_excel(filepath, index=False)
    print(f"任务信息已保存至: {filepath}")

# ==================== 实验运行函数（公平对比版本） ====================
def run_fair_experiment_for_param(algo_name, param_val, base_devices, base_all_uavs,
                                   total_time_slots, agent_getter, data_savers):
    """
    通用公平对比实验运行器
    algo_name: 算法名称（用于标识）
    param_val: 当前参数值（用于文件名）
    base_devices: 基准设备列表（每时隙会更新坐标和任务，但不被算法修改）
    base_all_uavs: 基准无人机列表（仅用于优化无人机数量，不参与决策）
    total_time_slots: 总时隙数
    agent_getter: 函数，输入设备数量等，返回算法实例（已加载模型）
    data_savers: 函数列表，每个函数接受算法副本的设备、无人机、参数值等并保存数据
    """
    # 为当前算法创建设备和无人机的深拷贝
    devices_copy = copy.deepcopy(base_devices)
    uavs_copy = copy.deepcopy(base_all_uavs)

    # 获取算法实例
    agent = agent_getter()

    # 用于存储历史卸载决策（如果算法需要）
    prev_offload_decisions = []

    # 时隙循环
    for t in range(1, total_time_slots + 1):
        # 1. 基准设备移动（已在外部完成，此处直接使用base_devices）
        # 2. 基准任务生成（已在外部完成，base_hasTaskDevices 和 base_tasks 可用）
        # 但我们需要从base_devices提取当前时隙的任务和设备信息
        base_hasTaskDevices = [dev for dev in base_devices if t in dev.task and dev.task[t] is not None]
        base_tasks = [dev.task[t] for dev in base_hasTaskDevices]

        # 3. 将基准坐标同步到算法副本
        for idx, dev in enumerate(base_devices):
            devices_copy[idx].coordinate = copy.deepcopy(dev.coordinate)
            devices_copy[idx].trajectory[t] = dev.coordinate.copy()  # 记录轨迹

        # 4. 将基准任务同步到算法副本（深拷贝任务对象）
        for dev in base_hasTaskDevices:
            task_copy = copy.deepcopy(dev.task[t])
            devices_copy[dev.id].task[t] = task_copy
        # 对于没有生成任务的设备，确保 task[t] 为 None
        for dev in base_devices:
            if t not in dev.task or dev.task[t] is None:
                if t in devices_copy[dev.id].task:
                    del devices_copy[dev.id].task[t]
                devices_copy[dev.id].task[t] = None

        # 提取算法副本的当前任务列表
        tasks_copy = [devices_copy[dev.id].task[t] for dev in base_hasTaskDevices]

        # 5. 优化无人机数量（基于基准设备，因为设备坐标相同）
        optimized_uav_num, selected_uav_indices = optimize_uav_number(
            base_hasTaskDevices, base_all_uavs, t
        )
        selected_uavs = [base_all_uavs[i] for i in selected_uav_indices]

        # 6. 根据算法类型调用相应的决策函数
        # 由于不同算法决策函数差异大，这里用分支处理
        if algo_name == 'hierarchical':
            # 构建 DDPG 状态
            if t == 1 or len(prev_offload_decisions) == 0:
                random_offload = random_offload_decisions(tasks_copy, optimized_uav_num,
                                                          devices_copy, selected_uavs)
                ddpg_state = preprocess_ddpg_state(base_hasTaskDevices, tasks_copy,
                                                    uavs_copy, [random_offload])
            else:
                ddpg_state = preprocess_ddpg_state(base_hasTaskDevices, tasks_copy,
                                                    uavs_copy, prev_offload_decisions)
            # DDPG 轨迹决策
            current_uavs_coords = [uav.coordinate for uav in uavs_copy]
            ddpg_actions = agent.ddpg_trainer.get_actions(ddpg_state, current_uavs_coords, training=False)
            traj_decisions = postprocess_ddpg_action(ddpg_actions, uavs_copy, t)
            # AC 卸载决策
            ac_state = preprocess_ac_state(base_hasTaskDevices, tasks_copy, uavs_copy,
                                           traj_decisions, t)
            ac_actions, _ = agent.ac_trainer.ac_trainer.get_actions(
                ac_state, training=False, num_using_uavs=optimized_uav_num
            )
            offload_decisions = postprocess_ac_action(ac_actions, selected_uavs,
                                                      selected_uav_indices, len(tasks_copy))
            # 更新无人机位置
            for idx, uav_idx in enumerate(selected_uav_indices):
                if idx < len(traj_decisions):
                    uavs_copy[uav_idx].coordinate = list(traj_decisions[uav_idx])
                    uavs_copy[uav_idx].trajectory[t] = list(traj_decisions[uav_idx])
            prev_offload_decisions = offload_decisions

        elif algo_name == 'standard_ddpg':
            if t == 1 or len(prev_offload_decisions) == 0:
                random_offload = random_offload_decisions(tasks_copy, optimized_uav_num,
                                                          devices_copy, selected_uavs)
                state = preprocess_standard_state(base_hasTaskDevices, tasks_copy,
                                                  uavs_copy, [random_offload])
            else:
                state = preprocess_standard_state(base_hasTaskDevices, tasks_copy,
                                                  uavs_copy, prev_offload_decisions)
            raw_actions = agent.get_actions(state, training=False)
            traj_decisions, offload_decisions = agent.postprocess_actions(
                raw_actions, selected_uav_indices, len(tasks_copy), uavs_copy
            )
            for idx, uav_idx in enumerate(selected_uav_indices):
                if idx < len(traj_decisions):
                    uavs_copy[uav_idx].coordinate = list(traj_decisions[uav_idx])
                    uavs_copy[uav_idx].trajectory[t] = list(traj_decisions[uav_idx])
            prev_offload_decisions = offload_decisions

        elif algo_name == 'ddpg_ldpg':
            if t == 1 or len(prev_offload_decisions) == 0:
                random_offload = random_offload_decisions(tasks_copy, optimized_uav_num,
                                                          devices_copy, selected_uavs)
                state = agent.preprocess_state(base_hasTaskDevices, tasks_copy,
                                               uavs_copy, [random_offload])
            else:
                state = agent.preprocess_state(base_hasTaskDevices, tasks_copy,
                                               uavs_copy, prev_offload_decisions)
            agent.state_history.append(state)
            while len(agent.state_history) < agent.ldpg_offload.seq_len:
                agent.state_history.append(state)
            state_seq = list(agent.state_history)
            traj_actions = agent.ddpg_traj.select_action(state, noise_scale=0.0)
            offload_actions = agent.ldpg_offload.select_action(
                state_seq, training=False, available_uavs=selected_uav_indices
            )
            offload_decisions = agent.offload_action_to_decisions(offload_actions, tasks_copy,
                                                                   selected_uav_indices)
            traj_decisions = postprocess_ddpg_action(traj_actions, uavs_copy, t)
            for idx, uav_idx in enumerate(selected_uav_indices):
                if idx < len(traj_decisions):
                    uavs_copy[uav_idx].coordinate = list(traj_decisions[uav_idx])
                    uavs_copy[uav_idx].trajectory[t] = list(traj_decisions[uav_idx])
            prev_offload_decisions = offload_decisions

        elif algo_name == 'ppo':
            state = preprocess_ddpg_state(base_hasTaskDevices, tasks_copy, uavs_copy, [])
            cont_action, disc_action, _ = agent.select_action(state, training=False)
            traj_decisions, offload_decisions = postprocess_ppo_action(
                cont_action, disc_action, selected_uav_indices, len(tasks_copy), uavs_copy
            )
            for idx, uav_idx in enumerate(selected_uav_indices):
                if idx < len(traj_decisions):
                    uavs_copy[uav_idx].coordinate = list(traj_decisions[uav_idx])
                    uavs_copy[uav_idx].trajectory[t] = list(traj_decisions[uav_idx])
            prev_offload_decisions = offload_decisions

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        # 7. 更新任务时延能耗和队列
        task_time_energy(tasks_copy, devices_copy, uavs_copy, offload_decisions)
        enqueue_dequeue(tasks_copy, devices_copy, uavs_copy, offload_decisions, t)

    # 时隙结束，调用数据保存函数
    for saver in data_savers:
        saver(devices_copy, uavs_copy, param_val)

    return devices_copy, uavs_copy

# ==================== 主实验流程 ====================
def main():
    # 保存原始参数，以便恢复
    orig_target_offload_ratio = simEnvParameter.TARGET_OFFLOAD_RATIO
    orig_task_lower = simEnvParameter.DEVICE_TASK_LOWER_LIMIT
    orig_task_upper = simEnvParameter.DEVICE_TASK_UPPER_LIMIT
    orig_uav_resource = simEnvParameter.UAV_RESOURCE

    # 定义参数列表
    target_offload_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    task_intervals = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]  # Mbits
    uav_compute_resources = [1e9, 2e9, 3e9, 4e9, 5e9]  # cycle/s

    total_time_slots = simEnvParameter.TOTAL_TIME_SLOT

    # 算法列表（名称、获取agent的函数、数据保存函数列表）
    algorithms = [
        ('hierarchical',
         lambda: HierarchicalRLAlgorithm(device=torch_device),
         [lambda d,u,val: save_subtask_offload_info(d, 'target_offload_ratio', f'hierarchical_{val}.xlsx'),
          lambda d,u,val: save_subtask_time_energy_data(d, 'task_data_size', f'hierarchical_{val}.xlsx')]),
        ('standard_ddpg',
         lambda: StandardDDPGTrainer(device=torch_device),
         [lambda d,u,val: save_subtask_offload_info(d, 'target_offload_ratio', f'standard_ddpg_{val}.xlsx'),
          lambda d,u,val: save_subtask_time_energy_data(d, 'task_data_size', f'standard_ddpg_{val}.xlsx')]),
        ('ddpg_ldpg',
         lambda: DDPG_LDPG_Algorithm(device=torch_device),
         [lambda d,u,val: save_subtask_offload_info(d, 'target_offload_ratio', f'ddpg_ldpg_{val}.xlsx'),
          lambda d,u,val: save_subtask_time_energy_data(d, 'task_data_size', f'ddpg_ldpg_{val}.xlsx')]),
        ('ppo',
         lambda: PPO(state_dim=1316,
                     cont_dim=simEnvParameter.UAV_MAX_NUM*4,
                     num_subtasks=simEnvParameter.DEVICE_NUM*7,
                     num_actions=simEnvParameter.UAV_MAX_NUM+1,
                     device=torch_device),
         [lambda d,u,val: save_subtask_offload_info(d, 'target_offload_ratio', f'ppo_{val}.xlsx'),
          lambda d,u,val: save_subtask_time_energy_data(d, 'task_data_size', f'ppo_{val}.xlsx')])
    ]

    # ========== 实验1：目标卸载比例 ==========
    param_name = 'target_offload_ratio'
    folder = param_name
    print(f"\n========== 开始实验: {param_name} ==========")

    for val in target_offload_ratios:
        print(f"\n--- 参数值: {val} ---")
        simEnvParameter.TARGET_OFFLOAD_RATIO = val

        # 创建基准设备和无人机（每次参数重置时重新生成，保证不同参数间场景不同）
        base_devices = [device(i) for i in range(simEnvParameter.DEVICE_NUM)]
        get_coordinate(base_devices)
        base_all_uavs = []
        for i in range(simEnvParameter.UAV_MAX_NUM):
            u = uav(i)
            u.fun_random_coordinate(i)
            base_all_uavs.append(u)

        # 预先生成整个时间轴的基准移动和任务
        for t in range(1, total_time_slots + 1):
            change_coordinate(base_devices, t)
            for dev in base_devices:
                dev.fun_generate_task(t)

        # 对每种算法运行公平对比实验
        for algo_name, agent_getter, data_savers in algorithms:
            print(f"  运行算法: {algo_name}")
            # 加载预训练模型（需根据实际情况实现agent_getter中的加载逻辑）
            # 这里简单在agent_getter中加载默认模型文件，实际可能需要根据设备数量等调整
            # 我们可以在agent_getter中编写加载代码，例如：
            # if algo_name == 'hierarchical':
            #     agent = HierarchicalRLAlgorithm(device=torch_device)
            #     agent.ddpg_trainer.load_model('models/Hierarchical_ddpg_best.pth')
            #     agent.ac_trainer.load_model('models/Hierarchical_ac_best.pth')
            # ...
            # 为了简化，我们假设agent_getter返回的agent已经加载好模型
            run_fair_experiment_for_param(
                algo_name, val,
                base_devices, base_all_uavs,
                total_time_slots,
                agent_getter,
                data_savers
            )

    # 恢复原始参数
    simEnvParameter.TARGET_OFFLOAD_RATIO = orig_target_offload_ratio
    #
    # # ========== 实验2：任务数据大小 ==========
    param_name = 'task_data_size'
    folder = param_name
    print(f"\n========== 开始实验: {param_name} ==========")

    for low, high in task_intervals:
        print(f"\n--- 任务数据量范围: [{low}, {high}] Mbits ---")
        simEnvParameter.DEVICE_TASK_LOWER_LIMIT = low
        simEnvParameter.DEVICE_TASK_UPPER_LIMIT = high

        base_devices = [device(i) for i in range(simEnvParameter.DEVICE_NUM)]
        get_coordinate(base_devices)
        base_all_uavs = []
        for i in range(simEnvParameter.UAV_MAX_NUM):
            u = uav(i)
            u.fun_random_coordinate(i)
            base_all_uavs.append(u)

        for t in range(1, total_time_slots + 1):
            change_coordinate(base_devices, t)
            for dev in base_devices:
                dev.fun_generate_task(t)

        for algo_name, agent_getter, data_savers in algorithms:
            print(f"  运行算法: {algo_name}")
            # 注意：这里我们使用不同的数据保存函数，实验2保存的是时延能耗数据
            # 我们之前定义的data_savers包含了两个函数，但第一个是卸载信息，第二个是时延能耗
            # 对于实验2，我们只想保存时延能耗，可以只使用第二个
            # 为了灵活，我们可以单独定义每个实验的保存函数列表
            # 这里简单复用，但注意输出文件名需要区分
            # 更好的做法是为每个实验单独定义数据保存函数
            run_fair_experiment_for_param(
                algo_name, f"{low}_{high}",
                base_devices, base_all_uavs,
                total_time_slots,
                agent_getter,
                [lambda d,u,val: save_subtask_time_energy_data(d, folder, f'{algo_name}_taskdata_{val}.xlsx')]
            )

    # 恢复原始参数
    simEnvParameter.DEVICE_TASK_LOWER_LIMIT = orig_task_lower
    simEnvParameter.DEVICE_TASK_UPPER_LIMIT = orig_task_upper
    #
    # ========== 实验3：无人机处理能力 ==========
    param_name = 'uav_compute_resource'
    folder = param_name
    print(f"\n========== 开始实验: {param_name} ==========")

    for val in uav_compute_resources:
        print(f"\n--- 参数值: {val} cycle/s ---")
        simEnvParameter.UAV_RESOURCE = val

        base_devices = [device(i) for i in range(simEnvParameter.DEVICE_NUM)]
        get_coordinate(base_devices)
        base_all_uavs = []
        for i in range(simEnvParameter.UAV_MAX_NUM):
            u = uav(i)
            u.fun_random_coordinate(i)
            base_all_uavs.append(u)

        for t in range(1, total_time_slots + 1):
            change_coordinate(base_devices, t)
            for dev in base_devices:
                dev.fun_generate_task(t)

        for algo_name, agent_getter, data_savers in algorithms:
            print(f"  运行算法: {algo_name}")
            run_fair_experiment_for_param(
                algo_name, val,
                base_devices, base_all_uavs,
                total_time_slots,
                agent_getter,
                [lambda d,u,val: save_subtask_time_energy_data(d, folder, f'{algo_name}_{param_name}_{val}.xlsx')]
            )

    # 恢复原始参数
    simEnvParameter.UAV_RESOURCE = orig_uav_resource

    # ========== 实验4：保存无人机轨迹和任务信息 ==========
    param_name = 'uav_trajectory_and_task_info'
    folder = param_name
    print(f"\n========== 开始实验: {param_name} ==========")

    # 使用默认参数重新生成基准
    base_devices = [device(i) for i in range(simEnvParameter.DEVICE_NUM)]
    get_coordinate(base_devices)
    base_all_uavs = []
    for i in range(simEnvParameter.UAV_MAX_NUM):
        u = uav(i)
        u.fun_random_coordinate(i)
        base_all_uavs.append(u)

    for t in range(1, total_time_slots + 1):
        change_coordinate(base_devices, t)
        for dev in base_devices:
            dev.fun_generate_task(t)

    # 为每种算法定义不同的数据保存函数
    algo_savers = {
        'hierarchical': [
            lambda d,u,val: save_uav_trajectories(u, folder, 'hierarchical_trajectories.xlsx'),
            lambda d,u,val: save_task_info(d, folder, 'hierarchical_tasks.xlsx')
        ],
        'standard_ddpg': [
            lambda d,u,val: save_uav_trajectories(u, folder, 'standard_ddpg_trajectories.xlsx'),
            lambda d,u,val: save_task_info(d, folder, 'standard_ddpg_tasks.xlsx')
        ],
        'ddpg_ldpg': [
            lambda d,u,val: save_uav_trajectories(u, folder, 'ddpg_ldpg_trajectories.xlsx'),
            lambda d,u,val: save_task_info(d, folder, 'ddpg_ldpg_tasks.xlsx')
        ],
        'ppo': [
            lambda d,u,val: save_uav_trajectories(u, folder, 'ppo_trajectories.xlsx'),
            lambda d,u,val: save_task_info(d, folder, 'ppo_tasks.xlsx')
        ]
    }

    for algo_name, agent_getter, _ in algorithms:
        print(f"  运行算法: {algo_name}")
        run_fair_experiment_for_param(
            algo_name, None,  # val 用 None
            base_devices, base_all_uavs,
            total_time_slots,
            agent_getter,
            algo_savers[algo_name]
        )

    print("\n所有实验完成！数据已保存至对应文件夹。")

if __name__ == "__main__":
    main()