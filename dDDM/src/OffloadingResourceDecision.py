from utils import *
import numpy as np
import math
from tqdm import *

# 反弹操作函数
def rebound(value, lower_bound, upper_bound):
    """
    对变量进行反弹操作，使其在 [lower_bound, upper_bound] 范围内。
    超出上下界时，进行上限或下限求余后返回到范围内。
    """
    if value < lower_bound:
        return lower_bound + (lower_bound - value) % (upper_bound - lower_bound)
    elif value > upper_bound:
        return upper_bound - (value - upper_bound) % (upper_bound - lower_bound)
    return value

# 黑翅鸢算法求解卸载和资源分配决策
def BKA_offloading_resource_decision(tasks, devices, uav, time_slot, iterative_recording_BKA,
                                     max_iter=100, pop_size=30, alpha=ALPHA, beta=BETA):
    """
    黑翅鸢算法求解卸载和资源分配决策。

    参数:
        tasks: 任务列表。
        devices: 设备列表。
        uav: 无人机对象。
        max_iter: 最大迭代次数。
        pop_size: 种群大小。
        alpha: 目标函数中时延权重。
        beta: 目标函数中能耗权重。

    返回:
        最优决策参数及其对应的目标函数值。
    """
    print(f"=========================================================BKA开始！")

    def initialize_population():
        """初始化种群，返回满足约束条件的个体集合"""
        population = []

        # 生成第一个满足约束的个体（死循环）
        while True:
            tasks_offload_ratio = np.random.uniform(0, 1, len(tasks))
            uav_compute_resource = np.random.uniform(0.1 * UAV_MAX_RESOURCE, UAV_MAX_RESOURCE)

            if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                population.append((tasks_offload_ratio.tolist(), uav_compute_resource))
                break

        # 在第一个个体附近扰动生成剩余个体
        while len(population) < pop_size:
            # 获取第一个个体作为基础
            base_ratio, base_resource = population[0]
            base_ratio = np.array(base_ratio)

            # 生成扰动：在基础值附近随机扰动
            perturb_ratio = base_ratio * np.random.uniform(0.9, 1.1, len(tasks))
            perturb_ratio = np.clip(perturb_ratio, 0, 1)

            perturb_resource = base_resource * np.random.uniform(0.9, 1.1)
            perturb_resource = np.clip(perturb_resource, 0.1 * UAV_MAX_RESOURCE, UAV_MAX_RESOURCE)

            # 检查约束
            if constrain(tasks, devices, uav, perturb_ratio, perturb_resource):
                population.append((perturb_ratio.tolist(), perturb_resource))

        return population

    population = initialize_population()

    fitness = []
    for ind in population:
        tasks_offload_ratio, uav_compute_resource = ind
        fitness_val = objective_function_Off_Res_Decision(tasks, devices, uav,
                                                          tasks_offload_ratio,
                                                          uav_compute_resource,
                                                          alpha, beta)
        fitness.append(fitness_val)

    leader = population[np.argmin(fitness)]
    leader_fitness = min(fitness)

    for iteration in tqdm(range(max_iter)):
        for i in range(pop_size):
            tasks_offload_ratio, uav_compute_resource = population[i]
            tasks_offload_ratio = np.array(tasks_offload_ratio)

            n = 0.05 * np.exp(-2 * (iteration / max_iter) ** 2)
            r = np.random.rand()
            if 0.9 < r:
                tasks_offload_ratio += n * (1 + np.sin(r)) * tasks_offload_ratio
                uav_compute_resource += n * (1 + np.sin(r)) * uav_compute_resource
            else:
                tasks_offload_ratio += n * (2 * r - 1) * tasks_offload_ratio
                uav_compute_resource += n * (2 * r - 1) * uav_compute_resource

            random_idx = np.random.randint(0, pop_size)
            random_ind = population[random_idx]
            random_tasks_offload_ratio, random_uav_compute_resource = random_ind
            random_fitness_val = objective_function_Off_Res_Decision(tasks, devices, uav,
                                                                     random_tasks_offload_ratio,
                                                                     random_uav_compute_resource,
                                                                     alpha, beta)
            m = 2 * np.sin(r + math.pi / 2)

            if fitness[i] < random_fitness_val:
                tasks_offload_ratio += np.random.standard_cauchy(len(tasks)) * (tasks_offload_ratio - leader[0])
                uav_compute_resource += np.random.standard_cauchy(1) * (uav_compute_resource - leader[1])
            else:
                tasks_offload_ratio += np.random.standard_cauchy(len(tasks)) * (leader[0] - m * tasks_offload_ratio)
                uav_compute_resource += np.random.standard_cauchy(1) * (leader[1] - m * uav_compute_resource)

            tasks_offload_ratio = np.array([rebound(x, 0, 1) for x in tasks_offload_ratio])
            uav_compute_resource = rebound(uav_compute_resource, 0, UAV_MAX_RESOURCE)

            if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                fitness_val = objective_function_Off_Res_Decision(tasks, devices, uav,
                                                                  tasks_offload_ratio,
                                                                  uav_compute_resource,
                                                                  alpha, beta)
                if fitness_val < fitness[i]:
                    population[i] = (tasks_offload_ratio.tolist(), uav_compute_resource)
                    fitness[i] = fitness_val

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < leader_fitness:
            leader = population[min_idx]
            leader_fitness = fitness[min_idx]

        if time_slot not in iterative_recording_BKA:
            iterative_recording_BKA[time_slot] = []
        iterative_recording_BKA[time_slot].append([iteration + 1, leader_fitness])

    # 检查最终leader是否满足约束
    if not constrain(tasks, devices, uav, leader[0], leader[1]):
        return BKA_offloading_resource_decision(tasks, devices, uav, time_slot, iterative_recording_BKA,
                                                max_iter, pop_size, alpha, beta)

    print(f"\nBKA leader_fitness = {leader_fitness}\n")
    print(f"{len(leader[0])}-->offloading_ratio = {leader[0]}\n"
          f"uav_resource = {leader[1]}\n")
    print(f"=========================================================BKA结束！")
    return leader



def add_to_experience_pool(task, decision,experience_pool):
    """
    添加记录到经验池
    :param task: 任务对象
    :param decision: 决策信息 (卸载比例, 设备资源, UAV资源)
    """
    record = {
        "data_ratio": task.data_length / task.max_finish_time,  # 特征：任务数据量与完成时间的比值
        "decision": decision  # 决策信息
    }
    experience_pool.append(record)


def find_closest_experience(task, experience_pool):
    """
    从经验池中找到与新任务最接近的记录
    :param task: 任务对象
    :return: 最接近的决策 (若无经验池则返回 None)
    """
    if not experience_pool:
        return None

    task_ratio = task.data_length / task.max_finish_time
    closest_record = min(experience_pool, key=lambda x: abs(x["data_ratio"] - task_ratio))

    return closest_record["decision"] if "decision" in closest_record else None

def IBKA_offloading_resource_decision(tasks, devices, uav, time_slot, experience_pool,
                                     iterative_recording_BKA, max_iter=100, pop_size=30,
                                     alpha=ALPHA, beta=BETA):
    """
    Improved Black Kite Algorithm (IBKA) for task offloading and resource allocation decision.
    """

    def initialize_population():
        """初始化种群，返回个体集合"""
        population = []

        # 基于经验池初始化（使用find_closest_experience为每个任务寻找经验）
        if experience_pool and len(experience_pool) > 0:
            # 为每个任务找到最接近的经验
            exp_ratios = []
            exp_resources = []

            for task in tasks:
                closest_decision = find_closest_experience(task, experience_pool)
                if closest_decision and isinstance(closest_decision, tuple) and len(closest_decision) == 2:
                    exp_ratio, exp_resource = closest_decision
                    exp_ratios.append(float(exp_ratio))
                    exp_resources.append(float(exp_resource))
                else:
                    # 如果没有找到经验，使用随机值
                    exp_ratios.append(np.random.random(0,1))
                    exp_resources.append(UAV_MAX_RESOURCE * np.random.random(0,1))

            # 创建个体
            tasks_offload_ratio = np.array(exp_ratios)
            # 使用经验资源的中位数或平均值
            uav_compute_resource = np.median(exp_resources) if exp_resources else UAV_MAX_RESOURCE * 0.5
            uav_compute_resource = np.clip(uav_compute_resource, 0.1 * UAV_MAX_RESOURCE, UAV_MAX_RESOURCE)

            # 检查约束
            if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                population.append((tasks_offload_ratio.tolist(), uav_compute_resource))

        # 补充剩余个体：随机初始化（死循环）
        remaining_count = pop_size - len(population)
        if remaining_count > 0:
            for _ in range(remaining_count):
                while True:
                    tasks_offload_ratio = np.random.uniform(0, 1, len(tasks))
                    uav_compute_resource = np.random.uniform(0.1 * UAV_MAX_RESOURCE, UAV_MAX_RESOURCE)

                    if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                        population.append((tasks_offload_ratio.tolist(), uav_compute_resource))
                        break

        return population
    population = initialize_population()

    fitness = [objective_function_Off_Res_Decision(tasks, devices, uav, ind[0], ind[1], alpha, beta) for ind in population]
    leader = population[np.argmin(fitness)]
    leader_fitness = min(fitness)

    # Iterative optimization
    for iteration in tqdm(range(max_iter)):
        for i in range(pop_size):
            tasks_offload_ratio, uav_compute_resource = map(np.array, population[i])
            n = 0.05 * np.exp(-2 * (iteration / max_iter) ** 2)
            r = np.random.rand()

            if r > 0.9:
                tasks_offload_ratio += n * (1 + np.sin(r)) * tasks_offload_ratio
                uav_compute_resource += np.float64(n * (1 + np.sin(r)) * uav_compute_resource)
            else:
                tasks_offload_ratio += n * (2 * r - 1) * tasks_offload_ratio
                uav_compute_resource += np.float64(n * (2 * r - 1) * uav_compute_resource)

            random_ind = population[np.random.randint(pop_size)]
            random_ratio, random_resource = random_ind
            random_fitness = objective_function_Off_Res_Decision(tasks, devices, uav, random_ratio, random_resource, alpha, beta)

            m = 2 * np.sin(r + np.pi / 2)
            if fitness[i] < random_fitness:
                tasks_offload_ratio += np.random.standard_cauchy(len(tasks)) * (tasks_offload_ratio - leader[0])
                uav_compute_resource += np.random.standard_cauchy() * (uav_compute_resource - leader[1])
            else:
                tasks_offload_ratio += np.random.standard_cauchy(len(tasks)) * (leader[0] - m * tasks_offload_ratio) * (random_ratio - m * tasks_offload_ratio)
                uav_compute_resource += np.random.standard_cauchy() * (leader[1] - m * uav_compute_resource) * (random_resource - m * uav_compute_resource)

            tasks_offload_ratio = np.clip([rebound(x, 0, 1) for x in tasks_offload_ratio], 0, 1)
            uav_compute_resource = rebound(uav_compute_resource, 0, UAV_MAX_RESOURCE)

            if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                new_fitness = objective_function_Off_Res_Decision(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource, alpha, beta)
                if new_fitness < fitness[i]:
                    population[i] = (tasks_offload_ratio.tolist(), uav_compute_resource)
                    fitness[i] = new_fitness

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < leader_fitness:
            leader = population[min_idx]
            leader_fitness = fitness[min_idx]

        iterative_recording_BKA.setdefault(time_slot, []).append([iteration + 1, leader_fitness])

    # 检查约束
    if  not constrain(tasks, devices, uav, leader[0], leader[1]):
        return BKA_offloading_resource_decision(tasks, devices, uav, time_slot, iterative_recording_BKA,
                                                max_iter, pop_size, alpha, beta)

    print(f"\nIBKA leader_fitness = {leader_fitness}\n")
    print(f"{len(leader[0])}-->offloading_ratio = {leader[0]}\n"
              f"uav_resource = {leader[1]}\n")
    for task_idx, task in enumerate(tasks):
        add_to_experience_pool(task, (leader[0][task_idx], leader[1]), experience_pool)

    uav_compute_resource = np.array([leader[1]], dtype=np.float64)
    return (leader[0], uav_compute_resource)


# 随机卸载和无人机资源分配
def random_offloading_resource(tasks, devices, uav, max_attempts=2000):
    """
    改进的随机卸载和资源分配算法，在保持约束条件的同时提高找到可行解的效率

    参数:
        tasks: 任务列表
        devices: 设备列表
        uav: 无人机对象
        max_attempts: 最大尝试次数

    返回:
        满足约束条件的卸载比例和无人机计算资源
    """
    from tqdm import tqdm

    # 定义策略及其权重（基于经验）
    strategies = [
        # (策略名称, 卸载比例范围, 资源范围, 权重)
        ("高卸载", (0.7, 1.0), (0.6, 1.0), 0.4),
        ("混合卸载", (0.3, 0.9), (0.4, 0.8), 0.3),
        ("保守卸载", (0.1, 0.5), (0.2, 0.6), 0.2),
        ("全卸载", (1.0, 1.0), (0.7, 1.0), 0.1)
    ]

    # 根据权重分配尝试次数
    total_weight = sum(weight for _, _, _, weight in strategies)
    strategy_attempts = []

    for name, ratio_range, resource_range, weight in strategies:
        attempts = int(max_attempts * (weight / total_weight))
        strategy_attempts.append((name, ratio_range, resource_range, attempts))

    # 添加进度条
    with tqdm(total=max_attempts, desc="寻找可行解") as pbar:
        # 尝试各种策略
        for strategy_name, ratio_range, resource_range, attempts in strategy_attempts:
            for attempt in range(attempts):
                # 生成卸载比例
                low_ratio, high_ratio = ratio_range
                tasks_offload_ratio = np.random.uniform(low_ratio, high_ratio, len(tasks))

                # 生成无人机资源 - 确保返回numpy数组格式
                low_resource, high_resource = resource_range
                uav_compute_resource = np.array([
                    np.random.uniform(
                        low_resource * UAV_MAX_RESOURCE,
                        high_resource * UAV_MAX_RESOURCE
                    )
                ])

                # 检查约束条件
                if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                    pbar.update(max_attempts - pbar.n)  # 快速完成进度条
                    return (tasks_offload_ratio, uav_compute_resource)

                pbar.update(1)

        # 如果所有策略都失败，尝试基于任务特征的智能随机
        remaining_attempts = max_attempts - pbar.n
        if remaining_attempts > 0:
            for attempt in range(remaining_attempts):
                tasks_offload_ratio = []

                # 根据任务特性调整卸载概率
                for task in tasks:
                    # 紧急任务（完成时间短）倾向于卸载
                    urgency_factor = 1.0 / (task.max_finish_time + 0.1)
                    # 大数据量任务倾向于卸载
                    size_factor = min(task.data_length / 2e6, 1.0)

                    # 综合因素决定卸载概率
                    base_prob = 0.3 + 0.4 * (urgency_factor * 0.6 + size_factor * 0.4)
                    offload_ratio = np.random.uniform(max(0, base_prob - 0.2), min(1, base_prob + 0.2))
                    tasks_offload_ratio.append(offload_ratio)

                tasks_offload_ratio = np.array(tasks_offload_ratio)

                # 智能分配无人机资源 - 确保返回numpy数组格式
                total_offload_workload = sum(
                    task.data_length * ratio * UAV_1BIT_CYCLE
                    for task, ratio in zip(tasks, tasks_offload_ratio)
                )
                # 资源分配考虑工作负载和一定的缓冲
                uav_compute_resource = np.array([
                    np.random.uniform(
                        total_offload_workload * 0.8,
                        min(total_offload_workload * 1.5, UAV_MAX_RESOURCE)
                    )
                ])

                if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
                    pbar.update(remaining_attempts - attempt)  # 快速完成进度条
                    return (tasks_offload_ratio, uav_compute_resource)

                pbar.update(1)

    # 最终保底策略：全部本地计算 - 确保返回numpy数组格式
    tasks_offload_ratio = np.zeros(len(tasks))
    uav_compute_resource = np.array([0.0])

    if constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
        return (tasks_offload_ratio, uav_compute_resource)
    else:
        raise ValueError("无法找到满足约束的卸载和资源分配方案")

# 论文：Mobility-Aware Joint Task Scheduling and Resource Allocation for Cooperative Mobile Edge Computing的算法复现

# MATS实现：去掉对卸载位置的排序，任务分配按照MATS的思路
def MATS_offloading_resource_decision(tasks,
                                      devices,
                                      uav,
                                      UAV_MAX_RESOURCE=UAV_MAX_RESOURCE,
                                      UAV_1BIT_CYCLE=UAV_1BIT_CYCLE):
    """
    实现 Saleem 等人提出的 MATS 算法，调整为适用于只有一个卸载目标（无人机）的环境

    参数:
        tasks: 任务列表。
        devices: 设备列表。
        uav: 无人机对象。
        UAV_MAX_RESOURCE: 无人机的最大计算资源。
        UAV_1BIT_CYCLE: 处理 1 bit 数据所需的 CPU 周期数。

    返回:
        np.array([a, f])，其中 a 是任务卸载比例数组，f 是无人机计算资源。
    """
    # 初始化任务卸载比例数组
    tasks_offload_ratio = [0] * len(tasks)
    # 初始化无人机计算资源
    uav_compute_resource = 0

    # 按计算复杂度降序排序任务
    sorted_tasks = sorted(enumerate(tasks), key=lambda x: UAV_1BIT_CYCLE * x[1].data_length, reverse=True)

    # 遍历排序后的任务，尝试分配到无人机
    for task_index, task in sorted_tasks:
        # 如果无人机资源已满，后续任务全部本地执行
        if uav_compute_resource >= UAV_MAX_RESOURCE:
            for remaining_task_index in range(task_index, len(tasks)):
                tasks_offload_ratio[remaining_task_index] = 0  # 后续任务全部本地执行
            break

        # 尝试将任务全部卸载到无人机
        offload_ratio = 1  # 全部卸载
        tasks_offload_ratio[task_index] = offload_ratio
        uav_compute_resource += UAV_1BIT_CYCLE * task.data_length

        # 检查是否满足约束条件
        compute_delay = device_compute_delay(
                                task,
                                tasks_offload_ratio[task_index]
                            ) +\
                             device_queue_delay(
                                 devices[task.device_id]
                             ) +\
                             max(
                                 data_transmission_delay(
                                     task,
                                     tasks_offload_ratio[task_index],
                                     devices[task.device_id],
                                     uav
                                 ),
                                 uav_queue_delay(
                                     uav,
                                     uav_compute_resource
                                 )+
                             uav_compute_delay(
                                 task,
                                 tasks_offload_ratio[task_index],
                                 uav_compute_resource
                             ))
        if compute_delay > task.max_finish_time:
            # 如果不满足约束条件，撤销分配，任务本地执行
            tasks_offload_ratio[task_index] = 0
            uav_compute_resource -= UAV_1BIT_CYCLE * task.data_length

    # 将任务卸载比例数组转换为 NumPy 数组
    tasks_offload_ratio = np.array(tasks_offload_ratio, dtype=np.float64)
    uav_compute_resource = np.float64(uav_compute_resource)

    print(f"\n{len(tasks_offload_ratio)}-->{tasks_offload_ratio}\n\n"
          f"uav_compute_resource = {uav_compute_resource}\n")
    # 返回封装后的结果
    return np.array([tasks_offload_ratio, uav_compute_resource])








