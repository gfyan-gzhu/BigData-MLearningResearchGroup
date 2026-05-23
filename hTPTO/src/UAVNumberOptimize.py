# [file name]: UAVNumberOptimize.py
from utils import *
import math
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from simEnvParameter import *


def optimize_uav_number(devices: List[device],
                        all_uavs: List[uav],
                        current_time_slot: int,
                        w_positive=0.7,
                        w_negative=0.3,
                        uav_max_num=UAV_MAX_NUM) -> Tuple[int, List[int]]:
    """
    改进的无人机数量优化算法 - 使用新的贡献度计算公式

    新选择策略:
    1. 第一时隙: 使用理论最小无人机数量，选择索引最小的无人机
    2. 后续时隙:
       - 先尝试前一时隙选择（保持连续性）
       - 逐步探索：减少无人机数量或增加无人机数量
       - 始终选择满足时延约束且使用最少无人机的方案
       - 如果全部4台无人机仍不满足时延，使用全部4台

    参数:
    - devices: 设备列表
    - all_uavs: 所有无人机列表（4台）
    - current_time_slot: 当前时隙

    返回:
    - optimal_num: 优化后的无人机数量
    - selected_uav_indices: 被选中的无人机索引列表
    """

    total_uavs = len(all_uavs)

    # 使用模块级变量记录历史选择
    if not hasattr(optimize_uav_number, 'selection_history'):
        optimize_uav_number.selection_history = {}
    if not hasattr(optimize_uav_number, 'contribution_history'):
        optimize_uav_number.contribution_history = defaultdict(list)

    # 获取前一时隙的选择
    previous_selection = None
    if current_time_slot > 0:
        previous_selection = optimize_uav_number.selection_history.get(current_time_slot - 1)

    def get_current_task_amount(use_historical_ratio=True) -> Tuple[float, float, float, float]:
        """
        获取当前时隙的总任务量，并区分卸载任务量和本地任务量

        参数:
        - use_historical_ratio: 是否使用历史卸载比例，时隙0为False，后续时隙为True
        """
        total_computation = 0.0
        offload_computation = 0.0
        local_computation = 0.0
        total_data = 0.0  # 总任务数据量（比特）

        # 获取卸载比例
        if use_historical_ratio:
            historical_ratio = get_historical_offload_ratio()
        else:
            # 时隙0使用全部卸载
            historical_ratio = 1.0

        for dev in devices:
            if current_time_slot in dev.task and dev.task[current_time_slot] is not None:
                task = dev.task[current_time_slot]
                # 计算任务总计算量（数据长度 * 单位比特周期数）
                task_computation = sum(st.data_length * UAV_1BIT_CYCLE for st in task.subtasks)
                total_computation += task_computation
                total_data += task.data_length

                # 根据卸载比例估计当前任务的卸载量
                offload_computation += task_computation * historical_ratio
                local_computation += task_computation * (1 - historical_ratio)

        return total_computation, offload_computation, local_computation, total_data

    def get_historical_offload_ratio() -> float:
        """获取历史卸载比例 - 基于所有设备的历史任务offload_ratio属性"""
        total_data = 0.0
        total_offloaded_data = 0.0
        count = 0

        # 遍历所有设备和所有时隙（从0到当前时隙-1）
        for dev in devices:
            for slot in range(current_time_slot):
                if slot in dev.task and dev.task[slot] is not None:
                    task = dev.task[slot]
                    total_data += task.data_length
                    if hasattr(task, 'offload_ratio'):
                        total_offloaded_data += task.data_length * task.offload_ratio
                    else:
                        # 如果没有卸载比例属性，使用保守估计0.3
                        total_offloaded_data += task.data_length * 0.3
                    count += 1

        # 如果没有任何历史任务，使用默认值
        if count == 0:
            return 0.3  # 默认30%卸载比例

        return total_offloaded_data / total_data if total_data > 0 else 0.3

    def calculate_theoretical_min_uavs() -> int:
        """计算理论最小UAV数量 - 只在时隙0调用，不使用历史卸载比例"""
        # 时隙0：使用全部卸载的比例（1.0）
        total_computation, offload_computation, _, _ = get_current_task_amount(use_historical_ratio=False)

        # 如果没有任务，返回0
        if total_computation == 0:
            return 0

        uav_computation_capacity = UAV_RESOURCE * UNITE_SLOT_LENGTH
        if uav_computation_capacity <= 0:
            return 1  # 避免除以零错误

        # 时隙0：假设全部卸载，计算最小无人机数量
        min_uavs = math.ceil(offload_computation / uav_computation_capacity)
        return max(1, min(min_uavs, uav_max_num))

    def get_current_tasks_avg_max_finish_time() -> float:
        """计算当前时隙所有任务的最大完成时间平均值"""
        max_finish_times = []
        for dev in devices:
            if current_time_slot in dev.task and dev.task[current_time_slot] is not None:
                task = dev.task[current_time_slot]
                max_finish_times.append(task.max_finish_time)

        if not max_finish_times:
            return 2.0  # 默认值，如果没有任务

        return np.mean(max_finish_times)

    def calculate_uav_contribution_new(uav_idx: int, num_selected_uavs: int) -> float:
        """
        新的贡献度计算公式:
        Contri_s = w1 * (L_off/d_s) / sum_{s'=1}^{S_t}(L_off/d_s')
                 - w2 * (D_{s,t}^{queue} * c_s) / (F_s * |tau|)

        其中:
        - d_s: 所有有任务设备到无人机s的平均距离
        - 其他符号含义不变

        参数:
        - uav_idx: 无人机索引
        - num_selected_uavs: 当前选中的无人机数量（用于归一化分母）
        """
        uav_obj = all_uavs[uav_idx]

        # 获取任务信息（后续时隙使用历史卸载比例）
        if current_time_slot == 0:
            total_computation, offload_computation, local_computation, total_data = get_current_task_amount(
                use_historical_ratio=False)
        else:
            total_computation, offload_computation, local_computation, total_data = get_current_task_amount(
                use_historical_ratio=True)

        # 如果没有任务，返回0
        if total_computation == 0 or num_selected_uavs == 0:
            return 0.0

        # 计算平均卸载比例（后续时隙使用历史卸载比例，时隙0使用1.0）
        if current_time_slot == 0:
            historical_ratio = 1.0
        else:
            historical_ratio = get_historical_offload_ratio()

        # 1. 计算正贡献部分：任务接近度贡献
        positive_contribution = 0.0

        # 获取所有有任务设备
        active_devices = [dev for dev in devices
                          if current_time_slot in dev.task and dev.task[current_time_slot] is not None]

        if active_devices:
            # 计算所有有任务设备到无人机s的平均距离
            total_distance = 0.0
            device_count = 0

            for dev in active_devices:
                # 计算3D距离
                dist = distance_3d(uav_obj.coordinate, dev.coordinate[:2])
                total_distance += dist
                device_count += 1

            # 计算平均距离
            if device_count > 0:
                avg_distance = total_distance / device_count
            else:
                avg_distance = 1e6  # 如果没有有任务设备，设置一个很大的距离
        else:
            avg_distance = 1e6  # 如果没有有任务设备，设置一个很大的距离

        # 避免除零
        if avg_distance < 1e-6:
            avg_distance = 1e-6

        # L_total: 总任务数据量（比特）
        L_total = total_data

        # L_off = (ρ * L_total) / S_t
        L_off = (historical_ratio * L_total) / num_selected_uavs

        # 计算该无人机的任务接近度：L_off / d_s
        uav_task_proximity = L_off / avg_distance

        # 计算所有无人机的任务接近度总和（用于归一化）
        total_proximity_sum = 0.0

        for s_prime in range(total_uavs):
            uav_prime = all_uavs[s_prime]

            # 计算无人机s_prime的平均距离
            if active_devices:
                total_distance_prime = 0.0
                for dev in active_devices:
                    dist_prime = distance_3d(uav_prime.coordinate, dev.coordinate[:2])
                    total_distance_prime += dist_prime
                avg_distance_prime = total_distance_prime / len(active_devices)
            else:
                avg_distance_prime = 1e6  # 如果没有有任务设备，设置一个很大的距离

            if avg_distance_prime < 1e-6:
                avg_distance_prime = 1e-6

            # 计算无人机s_prime的任务接近度
            proximity_prime = L_off / avg_distance_prime
            total_proximity_sum += proximity_prime

        # 避免除零
        if total_proximity_sum < 1e-6:
            total_proximity_sum = 1e-6

        # 计算正贡献：归一化的任务接近度
        positive_contribution = uav_task_proximity / total_proximity_sum

        # 2. 计算负贡献部分：队列负载贡献
        negative_contribution = 0.0

        # D_{s,t}^{queue}: 无人机队列长度（比特）
        queue_length = uav_obj.fun_calculate_queue_length()

        # c_s: 单位比特计算所需周期数（UAV_1BIT_CYCLE）
        # F_s: 无人机计算资源（Hz，即UAV_RESOURCE）
        # |tau|: 时隙长度（UNITE_SLOT_LENGTH）

        if UAV_RESOURCE > 0 and UNITE_SLOT_LENGTH > 0:
            negative_contribution = (queue_length * UAV_1BIT_CYCLE) / (UAV_RESOURCE * UNITE_SLOT_LENGTH)

        # 3. 综合贡献度
        contribution = w_positive * positive_contribution - w_negative * negative_contribution

        # 记录贡献度历史
        optimize_uav_number.contribution_history[uav_idx].append(contribution)

        return contribution

    def get_sorted_uavs_by_contribution(num_selected_uavs: int = 1) -> List[Tuple[int, float]]:
        """获取按贡献度排序的无人机列表"""
        contributions = []
        for uav_idx in range(total_uavs):
            contribution = calculate_uav_contribution_new(uav_idx, num_selected_uavs)
            contributions.append((uav_idx, contribution))

        # 按贡献度从高到低排序
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions

    def estimate_average_delay(selected_uav_indices: List[int]) -> float:
        """评估使用指定无人机选择方案时的平均任务时延"""
        # 获取当前时隙的任务信息
        active_tasks = []
        task_computations = []  # 存储每个任务的计算量（cycles）
        task_data_lengths = []  # 存储每个任务的数据量（bits）
        task_devices = []  # 存储每个任务对应的设备

        for dev in devices:
            if current_time_slot in dev.task and dev.task[current_time_slot] is not None:
                task = dev.task[current_time_slot]
                active_tasks.append(task)

                # 计算任务总计算量（数据长度 * 单位比特周期数）
                total_computation = sum(st.data_length * UAV_1BIT_CYCLE for st in task.subtasks)
                task_computations.append(total_computation)

                # 任务总数据量（bits）
                task_data_lengths.append(task.data_length)
                task_devices.append(dev)

        # 如果没有任务，返回0
        if not active_tasks:
            return 0.0

        # 获取选中无人机的计算资源总和
        selected_uavs = [all_uavs[idx] for idx in selected_uav_indices]
        num_selected_uavs = len(selected_uavs)

        if num_selected_uavs == 0:
            # 没有选中无人机，所有任务本地计算
            total_local_delay = 0.0
            total_local_tasks = 0

            for i, task in enumerate(active_tasks):
                dev = task_devices[i]

                # 计算每个子任务的时延
                for st in task.subtasks:
                    # 本地计算时延 = 计算量 / 设备计算资源
                    local_compute_delay = (st.data_length * DEVICE_1BIT_CYCLE) / (DEVICE_RESOURCE * 1e6)
                    total_local_delay += local_compute_delay
                    total_local_tasks += 1

            avg_delay = total_local_delay / total_local_tasks if total_local_tasks > 0 else 0.0
            return avg_delay

        # 获取历史卸载比例，用于估计卸载任务量
        historical_ratio = get_historical_offload_ratio()
        # 确保卸载比例不会为0或太小
        if historical_ratio < 0.01:
            historical_ratio = 0.1  # 设置最小卸载比例

        # 计算总卸载计算量和总本地计算量（单位：cycles）
        total_computation = sum(task_computations)  # cycles
        total_offload_computation = total_computation * historical_ratio  # cycles
        total_local_computation = total_computation * (1 - historical_ratio)  # cycles

        # 计算选中无人机的总计算能力（Hz）
        total_uav_compute = sum(uav.compute_resource for uav in selected_uavs) * 1e6  # MHz转换为Hz

        # 1. 计算平均计算时延（卸载部分） - 单位：秒
        avg_offload_compute_delay = 0.0
        if total_uav_compute > 0 and total_offload_computation > 0:
            avg_offload_compute_delay = total_offload_computation / total_uav_compute
        else:
            avg_offload_compute_delay = 0.05  # 最小50ms

        # 2. 计算平均传输时延
        total_transmission_rate = 0.0
        transmission_count = 0

        for task_idx, task in enumerate(active_tasks):
            dev = task_devices[task_idx]

            for uav in selected_uavs:
                # 计算3D距离
                dist = distance_3d(uav.coordinate, dev.coordinate[:2])

                # 计算LoS概率
                prob_los = los_probability(uav.coordinate, dev.coordinate[:2])

                # 计算路径损耗
                pl_db = path_loss(dist, prob_los)

                # 计算数据速率(bps)
                rate = data_rate(pl_db)

                # 确保速率不为0
                if rate > 0:
                    total_transmission_rate += rate
                    transmission_count += 1

        avg_transmission_rate = total_transmission_rate / transmission_count if transmission_count > 0 else 10e6  # 默认10Mbps

        # 计算平均传输时延
        total_offload_data = sum(task_data_lengths) * historical_ratio  # bits
        if total_offload_data > 0 and avg_transmission_rate > 0:
            avg_transmission_delay = total_offload_data / (avg_transmission_rate * len(active_tasks))
        else:
            avg_transmission_delay = 0.05  # 最小50ms

        # 3. 计算平均队列时延
        # 计算无人机队列时延
        total_uav_queue_length = sum(uav.fun_calculate_queue_length() for uav in selected_uavs)  # bits
        avg_uav_queue_delay = 0.0
        if total_uav_compute > 0 and total_offload_computation > 0:
            avg_uav_queue_delay = (total_uav_queue_length * UAV_1BIT_CYCLE) / total_uav_compute

        # 计算本地设备队列时延
        total_device_queue_length = sum(dev.fun_calculate_queue_length() for dev in task_devices)  # bits
        total_device_compute = len(active_tasks) * DEVICE_RESOURCE * 1e6  # Hz
        avg_device_queue_delay = 0.0
        if total_device_compute > 0:
            avg_device_queue_delay = (total_device_queue_length * DEVICE_1BIT_CYCLE) / total_device_compute

        # 4. 计算本地计算时延
        avg_local_compute_delay = 0.0
        if DEVICE_RESOURCE > 0 and total_local_computation > 0:
            avg_local_compute_delay = (total_local_computation / len(active_tasks)) / (DEVICE_RESOURCE * 1e6)
        else:
            avg_local_compute_delay = 0.05  # 最小50ms

        # 5. 考虑任务依赖关系
        dependency_factor = 1.2  # 降低依赖因子，避免过度放大

        # 6. 计算总平均时延
        # 本地部分：本地计算时延 + 设备队列时延
        local_avg_delay = (avg_local_compute_delay + avg_device_queue_delay) * (1 - historical_ratio)

        # 卸载部分：计算时延 + 传输时延 + 无人机队列时延
        offload_avg_delay = (
                                        avg_offload_compute_delay + avg_transmission_delay + avg_uav_queue_delay) * historical_ratio

        total_avg_delay = (local_avg_delay + offload_avg_delay) * dependency_factor

        # 确保时延不为0且合理
        if total_avg_delay <= 0:
            total_avg_delay = 0.1  # 最小100ms

        # 确保时延不会过大
        total_avg_delay = min(total_avg_delay, 10.0)  # 最大10秒

        return total_avg_delay

    # ====================== 主算法开始 ======================

    # 检查当前时隙是否有任务
    has_tasks = any(
        current_time_slot in dev.task and dev.task[current_time_slot] is not None
        for dev in devices
    )

    # 情况1：当前时隙没有任务
    if not has_tasks:
        optimize_uav_number.selection_history[current_time_slot] = []
        return 0, []

    # 获取当前时隙任务平均最大完成时间（时延约束）
    avg_max_finish_time = get_current_tasks_avg_max_finish_time()

    # 情况2：第一时隙（时隙0） - 修改为按索引选择
    if current_time_slot == 0:
        # 使用理论最小无人机数量（假设全部卸载）
        optimal_num = calculate_theoretical_min_uavs()

        # 直接选择索引最小的 optimal_num 台无人机（不再使用贡献度排序）
        selected_uav_indices = list(range(optimal_num))

        # 验证时延约束
        estimated_avg_delay = estimate_average_delay(selected_uav_indices)
        if estimated_avg_delay > avg_max_finish_time:
            # 如果理论最小数量不满足时延约束，尝试增加无人机，仍按索引选择
            for num_uavs in range(optimal_num + 1, total_uavs + 1):
                candidate_indices = list(range(num_uavs))
                candidate_delay = estimate_average_delay(candidate_indices)
                if candidate_delay <= avg_max_finish_time:
                    optimal_num = num_uavs
                    selected_uav_indices = candidate_indices
                    break

        optimize_uav_number.selection_history[current_time_slot] = selected_uav_indices
        print(
            f"时隙 {current_time_slot}: 选择{optimal_num}台无人机，索引: {selected_uav_indices}，预计平均时延: {estimated_avg_delay:.2f}s，约束: {avg_max_finish_time:.2f}s")
        return optimal_num, selected_uav_indices

    # 情况3：后续时隙（保持不变，仍基于候选方案选择）
    candidate_solutions = []  # 存储(时延, 无人机数量, 无人机索引列表)

    # 阶段1: 尝试使用前一时隙的选择（如果存在）
    if previous_selection and len(previous_selection) > 0:
        # 验证前一时隙选择的无人机索引是否有效
        valid_previous = [idx for idx in previous_selection if 0 <= idx < total_uavs]
        if valid_previous:
            previous_delay = estimate_average_delay(valid_previous)
            if previous_delay <= avg_max_finish_time:
                candidate_solutions.append((previous_delay, len(valid_previous), valid_previous))

    # 阶段2: 探索减少无人机数量（如果前一时隙有选择且数量大于1）
    if previous_selection and len(previous_selection) > 1:
        # 为减少操作计算贡献度（使用当前数量-1）
        current_num = len(previous_selection)
        sorted_uavs = get_sorted_uavs_by_contribution(current_num - 1)

        # 从当前选择中移除贡献度最低的无人机
        # 先获取当前选择中无人机的贡献度
        uav_contributions = []
        for idx in previous_selection:
            contribution = calculate_uav_contribution_new(idx, current_num - 1)
            uav_contributions.append((idx, contribution))

        # 按贡献度升序排序
        uav_contributions.sort(key=lambda x: x[1])

        # 尝试减少1台无人机（移除贡献度最低的）
        candidate_indices = previous_selection.copy()
        lowest_contrib_uav = uav_contributions[0][0]
        candidate_indices.remove(lowest_contrib_uav)

        candidate_delay = estimate_average_delay(candidate_indices)
        if candidate_delay <= avg_max_finish_time:
            candidate_solutions.append((candidate_delay, len(candidate_indices), candidate_indices))

    # 阶段3: 探索增加无人机数量（如果前一时隙有选择且数量小于总无人机数）
    if previous_selection and len(previous_selection) < total_uavs:
        current_num = len(previous_selection)
        # 为增加操作计算贡献度（使用当前数量+1）
        sorted_uavs = get_sorted_uavs_by_contribution(current_num + 1)

        # 找到未选中无人机中贡献度最高的
        current_set = set(previous_selection)
        for uav_idx, _ in sorted_uavs:
            if uav_idx not in current_set:
                candidate_indices = previous_selection.copy()
                candidate_indices.append(uav_idx)
                candidate_delay = estimate_average_delay(candidate_indices)
                if candidate_delay <= avg_max_finish_time:
                    candidate_solutions.append((candidate_delay, current_num + 1, candidate_indices))
                break  # 只尝试增加一台

    # 阶段4: 如果没有找到任何满足时延约束的方案，使用全部无人机
    if not candidate_solutions:
        all_uav_indices = list(range(total_uavs))
        all_delay = estimate_average_delay(all_uav_indices)
        candidate_solutions.append((all_delay, total_uavs, all_uav_indices))

    # 选择最优方案：先按时延升序，再按无人机数量升序
    candidate_solutions.sort(key=lambda x: (x[0], x[1]))

    # 选择第一个（时延最小，如果时延相同则无人机数量最少）
    best_delay, optimal_num, selected_uav_indices = candidate_solutions[0]

    # 按无人机索引排序，确保一致性
    selected_uav_indices.sort()

    # 记录当前时隙的选择
    optimize_uav_number.selection_history[current_time_slot] = selected_uav_indices

    print(
        f"时隙 {current_time_slot}: 选择{optimal_num}台无人机，索引: {selected_uav_indices}，预计平均时延: {best_delay:.2f}s，约束: {avg_max_finish_time:.2f}s")

    return optimal_num, selected_uav_indices