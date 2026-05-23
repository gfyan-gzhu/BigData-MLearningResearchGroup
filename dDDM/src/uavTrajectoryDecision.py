from utils import *
from KMeans import *
import math
import numpy as np
from simEnvParameter import *
from collections import defaultdict





def random_trajectory(uav,
                      time_slot,
                      max_distance=UAV_MAX_SPEED * UNITE_SLOT_LENGTH,
                      max_x=GROUND_WIDTH,
                      max_y=GROUND_LENGTH):
    """
    随机生成下一个坐标，确保与当前点的距离不超过最大速度限制，且坐标在 [0, max_x] 和 [0, max_y] 范围内。

    参数:
        uav: 当前的无人机对象
        time_slot: 当前的时隙
        max_distance: 最大移动距离（速度限制）
        max_x: 场景的最大 x 坐标，默认为 100
        max_y: 场景的最大 y 坐标，默认为 100

    返回:
        新的坐标 [x_next, y_next]
    """
    # 提取当前点的坐标
    x_current, y_current = uav.coordinate

    # 随机生成新坐标，直到满足最大移动距离限制
    while True:
        # 随机生成新的坐标
        x_next = np.random.uniform(0, max_x)
        y_next = np.random.uniform(0, max_y)

        # 计算当前坐标与新坐标的距离
        distance = math.sqrt((x_next - x_current) ** 2 + (y_next - y_current) ** 2)

        # 如果新坐标与当前坐标的距离不超过最大距离限制，跳出循环
        if distance <= max_distance:
            break  # 满足条件，退出循环

    # 更新无人机坐标和轨迹
    flight_energy = 0.5 * UNITE_SLOT_LENGTH * ((distance / UNITE_SLOT_LENGTH) ** 2) * UAV_QUALITY
    uav.flight_energy[time_slot] = flight_energy
    print(f"random - distance: {distance:.4f}, flight energy = {flight_energy:.3f}")

    return [x_next, y_next]


def find_nearest_cluster(current_cluster_idx, cluster, centroids):
    """
    查找与当前集群最邻近的集群

    参数:
        current_cluster_idx: 当前集群的索引
        cluster: 包含所有集群的列表，每个集群是一个设备索引列表
        centroids: 质心列表，每个元素是集群的质心坐标

    返回:
        nearest_cluster_idx: 最近的集群索引
    """
    min_distance = float('inf')
    nearest_cluster_idx = -1
    current_centroid = centroids[current_cluster_idx]

    for i, centroid in enumerate(centroids):
        if i != current_cluster_idx:  # 忽略自己
            distance = math.sqrt((centroid[0] - current_centroid[0]) ** 2 + (centroid[1] - current_centroid[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster_idx = i

    return nearest_cluster_idx


def merge_clusters(cluster1, cluster2, centroids, best_positions, best_scores):
    """
    合并两个集群并更新合并后的集群信息

    参数:
        cluster1: 第一个集群，包含设备索引列表
        cluster2: 第二个集群，包含设备索引列表
        centroids: 所有集群的质心列表
        best_positions: 所有粒子的历史最优位置列表
        best_scores: 所有粒子的历史最优适应度列表

    返回:
        new_centroid: 合并后的新质心
        new_best_positions: 合并后的最优位置列表
        new_best_scores: 合并后的最优得分列表
    """
    # 合并设备集
    merged_cluster = cluster1 + cluster2

    # 更新质心：合并后的质心为两个集群质心的平均值
    centroid_x = sum(centroids[i][0] for i in merged_cluster) / len(merged_cluster)
    centroid_y = sum(centroids[i][1] for i in merged_cluster) / len(merged_cluster)
    new_centroid = (centroid_x, centroid_y)

    # 更新最优位置和最优得分
    new_best_positions = []
    new_best_scores = []
    for i in merged_cluster:
        new_best_positions.append(best_positions[i])
        new_best_scores.append(best_scores[i])

    # 更新合并后的最优位置和最优得分
    best_idx = new_best_scores.index(min(new_best_scores))
    new_best_position = new_best_positions[best_idx]
    new_best_score = new_best_scores[best_idx]

    # 更新质心列表、历史最优位置和得分
    centroids.append(new_centroid)
    best_positions.append(new_best_position)
    best_scores.append(new_best_score)

    return new_centroid, new_best_positions, new_best_scores




def pso_trajectory_optimization(tasks,
                                devices,
                                uav,
                                time_slot,
                                iterative_recording,
                                particle_number=60,
                                max_iter=100,
                                w=0.8,
                                c1=1.5,
                                c2=1.5,
                                max_distance=UAV_MAX_SPEED * UNITE_SLOT_LENGTH,
                                ):
    """
    通过粒子群优化 (PSO) 求解最小化目标函数的无人机坐标

    参数:
        tasks: 当前任务列表
        devices: 当前设备列表
        uav: 无人机对象
        time_slot: 时隙
        iterative_recording: 记录每次迭代的结果
        max_iter: 最大迭代次数
        w: 惯性权重
        c1: 个人经验加速因子
        c2: 全局经验加速因子
        max_distance: 最大距离限制，用于判断粒子是否已经收敛
        GROUND_LENGTH: 地面长度
        GROUND_WIDTH: 地面宽度
        UNITE_SLOT_LENGTH: 单位时隙长度
        UAV_QUALITY: 无人机质量

    返回:
        best_position: 最优无人机坐标 (x, y)
    """
    print(f"PSO- uav coordinate: {uav.coordinate}")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<PSO开始")

    # 初始化粒子群的位置和速度
    n_particles = particle_number # 粒子数量
    particles = []  # 粒子群
    velocities = []  # 粒子速度
    best_positions = []  # 每个粒子历史最优位置
    best_scores = []  # 每个粒子历史最优适应度

    # 初始化全局最优位置和适应度
    global_best_position = None
    global_best_score = float('inf')  # 初始设置为无穷大，确保任何计算出的适应度都会比它小

    # 初始化粒子位置和速度
    for _ in range(n_particles):
        while True:
            x = np.random.uniform(0, GROUND_LENGTH)
            y = np.random.uniform(0, GROUND_WIDTH)
            distance = math.sqrt((x - uav.coordinate[0]) ** 2 + (y - uav.coordinate[1]) ** 2)
            if distance <= max_distance:
                break
        particles.append([x, y])
        velocities.append([np.random.uniform(0, max_distance), np.random.uniform(0, max_distance)])
        best_positions.append([x, y])
        score = object_function_trajectory(tasks, devices, uav, [x, y])
        best_scores.append(score)

        # 更新全局最优位置和适应度
        if score < global_best_score:
            global_best_position = [x, y].copy()
            global_best_score = score

    # 初始化iterative_recording，确保该时隙有记录
    iterative_recording[time_slot] = []

    # 迭代更新粒子群
    for iteration in range(max_iter):
        for i in range(n_particles):
            # 更新速度
            r1, r2 = np.random.random(), np.random.random()

            # 速度更新公式
            velocities[i][0] = w * velocities[i][0] + c1 * r1 * (best_positions[i][0] - particles[i][0]) + c2 * r2 * (
                    global_best_position[0] - particles[i][0])
            velocities[i][1] = w * velocities[i][1] + c1 * r1 * (best_positions[i][1] - particles[i][1]) + c2 * r2 * (
                    global_best_position[1] - particles[i][1])

            # 更新位置
            particles[i][0] += velocities[i][0]
            particles[i][1] += velocities[i][1]

            # 确保粒子位置在模拟区域内
            particles[i][0] = max(0, min(particles[i][0], GROUND_LENGTH))
            particles[i][1] = max(0, min(particles[i][1], GROUND_WIDTH))

            # 检查粒子与无人机之间的距离
            while True:
                distance = math.sqrt(
                    (particles[i][0] - uav.coordinate[0]) ** 2 + (particles[i][1] - uav.coordinate[1]) ** 2
                )
                if distance <= max_distance:
                    break  # 距离符合限制，退出
                else:
                    # 如果超出最大距离限制，反弹修正
                    particles[i][0] = uav.coordinate[0] + (particles[i][0] - uav.coordinate[0]) / 2
                    particles[i][1] = uav.coordinate[1] + (particles[i][1] - uav.coordinate[1]) / 2

            # 计算当前粒子的适应度
            current_score = object_function_trajectory(tasks, devices, uav, particles[i])

            # 更新粒子历史最优位置和适应度
            if current_score < best_scores[i]:
                best_positions[i] = particles[i].copy()  # 使用 copy 避免引用传递问题
                best_scores[i] = current_score

                # 如果当前粒子找到了更好的位置，则立即更新全局最优
                if current_score < global_best_score:
                    global_best_position = particles[i].copy()  # 同样使用 copy
                    global_best_score = current_score

        # 将当前迭代的全局最优适应度和最优位置保存到iterative_recording中
        iterative_recording[time_slot].append([iteration + 1, global_best_score])

    # 在所有迭代完成后，计算并输出最优位置和无人机当前位置的距离
    if global_best_position is not None:
        distance = math.sqrt(
            ((global_best_position[0] - uav.coordinate[0]) ** 2) + ((global_best_position[1] - uav.coordinate[1]) ** 2))

        # 如果满足距离限制，则返回全局最优位置，否则重新生成粒子并迭代
        if distance <= max_distance:
            # 记录无人机飞行能耗
            flight_energy = 0.5 * UNITE_SLOT_LENGTH * ((distance / UNITE_SLOT_LENGTH) ** 2) * UAV_QUALITY
            uav.flight_energy[time_slot] = flight_energy

            print(f"PSO - global_best_position: {global_best_position}\n"
                  f"PSO - global_best_score：{global_best_score}\n"
                  f"PSO - distance: {distance:.4f}, flight energy :{flight_energy:.3f}")

            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<PSO结束\n")
            return global_best_position
        else:
            # 如果不满足距离限制，重新生成粒子并继续迭代
            print("PSO - Distance limit not met, regenerating particles...")
            return pso_trajectory_optimization(tasks, devices, uav, time_slot, iterative_recording, max_iter, w, c1, c2,
                                               max_distance)



# 结合K-means
def kmpso_trajectory_optimization(tasks,
                                  devices,
                                  uav,
                                  time_slot,
                                  iterative_recording,
                                  k=4,
                                  cluster_particle_number=15,
                                  max_iter=100,
                                  c1_min=0.5,
                                  c1_max=2.0,
                                  c2_min=0.5,
                                  c2_max=2.0,
                                  w_sta=0.9,
                                  w_end=0.4,
                                  max_distance=UAV_MAX_SPEED * UNITE_SLOT_LENGTH):
    """
    改进的KMPSO算法，结合IPSO的动态因子和基于概率的扰动策略。
    保留集群合并操作，并基于任务密集区生成粒子。

    参数:
        tasks: 任务列表。
        devices: 设备列表。
        uav: 无人机对象。
        time_slot: 当前时隙。
        iterative_recording: 迭代记录字典。
        k: 初始聚类数量。
        cluster_particle_number: 每个集群的粒子数量。
        max_iter: 最大迭代次数。
        c1_min, c1_max: c1的动态调整范围。
        c2_min, c2_max: c2的动态调整范围。
        w_sta, w_end: 惯性权重的动态调整范围。
        max_distance: 最大距离限制。

    返回:
        best_position: 最优无人机坐标 (x, y)。
    """
    print(f"KMPSO- uav coordinate: {uav.coordinate}")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>KMPSO开始")

    # K-means算法获取质心和集群
    coordinate_list = [devices[task.device_id].coordinate for task in tasks]
    centroids, cluster = kmeans(coordinate_list, k)

    # 筛选有效质心并调整超出范围的质心
    uav_x, uav_y = uav.coordinate[0], uav.coordinate[1]
    valid_centroids = []
    for centroid in centroids:
        distance = math.sqrt((centroid[0] - uav_x) ** 2 + (centroid[1] - uav_y) ** 2)
        if distance <= max_distance:
            valid_centroids.append(centroid)
        else:
            direction = np.array(centroid) - np.array(uav.coordinate)
            unit_direction = direction / (np.linalg.norm(direction) + 1e-10)
            adjusted_point = np.array(uav.coordinate) + unit_direction * max_distance
            valid_centroids.append(adjusted_point.tolist())

    # 初始化粒子群
    particles = []  # 粒子位置
    velocities = []  # 粒子速度
    best_positions = []  # 粒子的历史最优位置
    best_scores = []  # 粒子的历史最优适应度
    global_best_position = None  # 全局最优位置
    global_best_score = float('inf')  # 全局最优得分

    # 基于有效质心生成粒子
    def generate_particle_near_centroid(centroid, uav_coord, max_dist, std=5):
        """在质心附近生成粒子，并确保距离限制"""
        while True:
            dx = np.random.normal(0, std)
            dy = np.random.normal(0, std)
            new_x = centroid[0] + dx
            new_y = centroid[1] + dy
            distance = math.sqrt((new_x - uav_coord[0]) ** 2 + (new_y - uav_coord[1]) ** 2)
            if distance <= max_dist:
                return [new_x, new_y]

    # 初始化每个集群的粒子
    for c in range(k):
        for _ in range(cluster_particle_number):
            chosen_centroid = valid_centroids[c]
            particle = generate_particle_near_centroid(chosen_centroid, uav.coordinate, max_distance, std=10)
            particle[0] = max(0, min(particle[0], GROUND_LENGTH))
            particle[1] = max(0, min(particle[1], GROUND_WIDTH))
            particles.append(particle)
            velocities.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            best_positions.append(particle.copy())
            score = object_function_trajectory(tasks, devices, uav, particle)
            best_scores.append(score)
            if score < global_best_score:
                global_best_position = particle.copy()
                global_best_score = score

    # 确保 iterative_recording 有时间槽的键
    iterative_recording.setdefault(time_slot, [])

    # 粒子群迭代优化
    for iteration in range(max_iter):
        # 动态调整 c1、c2 和 w
        for i in range(len(particles)):
            ratio = global_best_score / (best_scores[i] + 1e-10)  # 避免除以零
            c1 = (c1_min - c1_max) * ratio + c1_max
            c2 = (c2_max - c2_min) * ratio + c2_min
            w = w_end + (w_sta - w_end) * math.exp(-3 * (ratio ** 3))

            # 计算随机决策参数 K 和扰动因子 d
            f_worst = max(best_scores)
            f_i = best_scores[i]
            if f_worst != f_i:
                K = abs((global_best_score - f_i) / (f_worst - f_i))
            else:
                K = 0  # 如果分母为零，设置 K 为 0
            if f_i != 0:
                d = 0.2 * math.cos(math.pi * global_best_score / f_i)
            else:
                d = 0.2  # 如果分母为零，设置 d 为默认值

            # 更新速度（加入扰动因子 d）
            r1, r2 = np.random.random(), np.random.random()
            if np.random.random() <= K:  # 基于概率的扰动优化
                velocities[i][0] = w * velocities[i][0] + c1 * r1 * (d * best_positions[i][0] - particles[i][0]) \
                                   + c2 * r2 * (d * global_best_position[0] - particles[i][0])
                velocities[i][1] = w * velocities[i][1] + c1 * r1 * (d * best_positions[i][1] - particles[i][1]) \
                                   + c2 * r2 * (d * global_best_position[1] - particles[i][1])
            else:
                velocities[i][0] = w * velocities[i][0] + c1 * r1 * (best_positions[i][0] - particles[i][0]) \
                                   + c2 * r2 * (global_best_position[0] - particles[i][0])
                velocities[i][1] = w * velocities[i][1] + c1 * r1 * (best_positions[i][1] - particles[i][1]) \
                                   + c2 * r2 * (global_best_position[1] - particles[i][1])

            # 更新位置
            particles[i][0] += velocities[i][0]
            particles[i][1] += velocities[i][1]

            # 检查粒子是否超出边界
            particles[i][0] = max(0, min(particles[i][0], GROUND_LENGTH))
            particles[i][1] = max(0, min(particles[i][1], GROUND_WIDTH))

            # 检查粒子与无人机之间的距离
            while True:
                distance = math.sqrt(
                    (particles[i][0] - uav.coordinate[0]) ** 2 + (particles[i][1] - uav.coordinate[1]) ** 2
                )
                if distance <= max_distance:
                    break  # 距离符合限制，退出
                else:
                    # 如果超出最大距离限制，反弹修正
                    particles[i][0] = uav.coordinate[0] + (particles[i][0] - uav.coordinate[0]) / 2
                    particles[i][1] = uav.coordinate[1] + (particles[i][1] - uav.coordinate[1]) / 2

            # 更新粒子历史最优
            current_score = object_function_trajectory(tasks, devices, uav, particles[i])
            if current_score < best_scores[i]:
                best_positions[i] = particles[i].copy()
                best_scores[i] = current_score

                # 更新全局最优
                if current_score < global_best_score:
                    global_best_position = particles[i].copy()
                    global_best_score = current_score

        # 集群合并操作
        if iteration % 10 == 0:  # 每10次迭代检查一次集群合并
            unchanged_count = 0
            for c in range(k):
                if best_scores[c * cluster_particle_number] == best_scores[(c + 1) * cluster_particle_number - 1]:
                    unchanged_count += 1
            if unchanged_count > k // 2:  # 如果超过一半的集群未变化，触发合并
                for c in range(k):
                    nearest_cluster_idx = find_nearest_cluster(c, cluster, centroids)
                    merged_cluster = cluster[c] + cluster[nearest_cluster_idx]
                    merged_best_positions = best_positions[c * cluster_particle_number:(c + 1) * cluster_particle_number] + \
                                           best_positions[nearest_cluster_idx * cluster_particle_number:(nearest_cluster_idx + 1) * cluster_particle_number]
                    merged_best_scores = best_scores[c * cluster_particle_number:(c + 1) * cluster_particle_number] + \
                                        best_scores[nearest_cluster_idx * cluster_particle_number:(nearest_cluster_idx + 1) * cluster_particle_number]
                    best_idx = merged_best_scores.index(min(merged_best_scores))
                    global_best_position = merged_best_positions[best_idx]
                    global_best_score = merged_best_scores[best_idx]
                    k -= 1  # 更新集群数量

        # 保存迭代记录
        iterative_recording[time_slot].append([iteration + 1, global_best_score])

    # 计算飞行能耗
    distance = math.sqrt((global_best_position[0] - uav.coordinate[0]) ** 2 + (global_best_position[1] - uav.coordinate[1]) ** 2)
    flight_energy = 0.5 * UNITE_SLOT_LENGTH * ((distance / UNITE_SLOT_LENGTH) ** 2) * UAV_QUALITY
    uav.flight_energy[time_slot] = flight_energy

    print(f"KMPSO - global_best_position: {global_best_position}\n"
          f"KMPSO - global_best_score：{global_best_score}\n"
          f"KMPSO - distance: {distance:.4f}, flight energy :{flight_energy:.3f}")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>KMPSO结束\n")
    return global_best_position


# 论文Online UAV-Mounted Edge Server Dispatching for Mobile-to-Mobile Edge Computing的算法HOLD复现

# 根据通信半径r生成多个网格中心
def generate_grid_centers(r, area_width=GROUND_WIDTH, area_height=GROUND_LENGTH):
    """
    根据当前半径 r 生成覆盖整个区域的网格中心点。
    :param r: 当前通信半径（也作为网格步长）
    """
    x_coords = np.arange(r, area_width, r)
    y_coords = np.arange(r, area_height, r)
    grid_centers = [(x, y) for x in x_coords for y in y_coords]
    return grid_centers

def distance(p1, p2):
    """
    计算两点之间的欧几里得距离
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



# 最坏情况下的服务容量计算，对应文章中的C(r)
def max_service_capacity(r):
    maxServiceCapacity = (UAV_MAX_RESOURCE *
                       (1 -
                        (2e6/
                         (
                            (BANDWIDTH / DEVICE_NUM) *
                              np.log2
                                  (
                                  1 + ((CHANNEL_POWER_GAIN/(r**2) * DEVICE_TRANSMIT_POWER) / (NOISE_POWER**2))
                                  )
                            )
                         )
                        )
                       ) / (2e6 * UAV_1BIT_CYCLE)
    return maxServiceCapacity

# 通信范围下的服务容量计算，对应文章中的ε_j
def service_capacity(tasks_in_range):
    """
    统计当前网格下的任务数量
    :param tasks_in_range: 在通信范围内所有任务
    :return: 当前网格下的任务数量
    """
    return len(tasks_in_range)


    return int(total_service_capacity)

def UAV_HOLD_trajectory_optimization(tasks,
                                    devices,
                                    uav,
                                    time_slot,
                                    max_distance = UAV_MAX_SPEED * UNITE_SLOT_LENGTH,
                                    derta_r = 10,
                                    r_min = 5,
                                    r_max = 350):
    """
    实现论文中的 UAV HOLD 算法，确定无人机最佳悬停点。
    只部署一台无人机。
    与HOLD文章中判断不同，HOLD文章是在任务最密集的地方部署无人机，但本文默认无人机能为全局提供服务（集覆盖情况），
    所以采用判断网格（以r_max为半径）任务数量ε_j不能超过最大服务容量C(r)，通过目标函数筛选出最佳的部署位置
    :param tasks: 当前时隙的所有任务列表
    :param devices: 所有设备对象列表
    :param uav: 无人机对象
    :param time_slot: 当前时隙
    :param max_distance: 无人机的最大飞行距离
    :param derta_r: 半径增量步长
    :param r_min: 最小通信半径
    :param r_max: 最大通信半径
    :return: 返回最佳部署点 (x, y, h) 和卸载决策
    """
    print(f"HOLD- uav coordinate: {uav.coordinate}")
    print(f"====================================HOLD开始")
    # 用于保留覆盖和未覆盖任务的变量
    covered_tasks = []

    best_position = None
    best_score = float('inf')  # 假设目标函数值越小越好

    # 构建设备ID到设备对象的映射
    device_map = {device.id: device for device in devices}

    # 提取任务的设备坐标
    task_positions = []
    for task in tasks:
        device = device_map.get(task.device_id)
        if device:
            task_positions.append((device.coordinate[0], device.coordinate[1], task))

    # 逐步增加通信半径
    r = r_min
    while r <= r_max:
        # C(r)：当前通信半径下的最大服务能力
        service_cap = max_service_capacity(r)

        # 生成网格中心
        grid_centers = generate_grid_centers(r)

        # 筛选出与无人机当前坐标距离不超过 max_distance 的网格中心
        valid_centers = [center for center in grid_centers if distance(uav.coordinate, center) <= max_distance]

        # 如果没有有效的网格中心，跳过当前半径
        if not valid_centers:
            r += derta_r
            continue

        # 统计每个有效网格覆盖的任务
        grid_tasks = defaultdict(list)
        for center in valid_centers:
            covered = [task for (tx, ty, task) in task_positions if distance(center, (tx, ty)) <= r_max]
            grid_tasks[center] = covered

        # 遍历所有有效网格，判断是否满足 ε_j <= C(r)
        for center in valid_centers:
            covered = grid_tasks[center]

            if not covered:
                continue

            # 计算该网格中所有任务的总服务需求 ε_j
            epsilon_j = service_capacity(covered)

            # 判断是否可以部署
            if epsilon_j <= service_cap:
                # 使用目标函数评估该部署点
                score = object_function_trajectory(tasks, devices, uav, (center[0], center[1]))

                # 更新最优部署点
                if score < best_score:
                    best_score = score
                    best_position = (center[0], center[1])
                    covered_tasks = covered  # 保留覆盖的任务

        r += derta_r  # 增加通信半径

    # 如果没有找到满足条件的部署点，则在 valid_centers 中选择任务强度最大的网格进行部署
    if best_position is None:
        grid_centers = generate_grid_centers(r_min)
        valid_centers = [center for center in grid_centers if distance(uav.coordinate, center) <= max_distance]
        if not valid_centers:
            raise ValueError(f"No valid centers found within max_distance = {max_distance} for r_max = {r_max}.")

        grid_tasks = defaultdict(list)
        for center in valid_centers:
            covered = [task for (tx, ty, task) in task_positions if distance(center, (tx, ty)) <= r_max]
            grid_tasks[center] = covered

        # 选择任务强度最大的网格进行部署
        best_center = max(valid_centers, key=lambda c: len(grid_tasks[c]))
        best_position = (best_center[0], best_center[1])
        covered_tasks = grid_tasks[best_center]  # 保留覆盖的任务

    # 构建卸载比例列表，与 tasks 顺序保持一致
    offloading_ratio = []

    # 使用任务对象的内存地址作为唯一标识符
    covered_task_ids = {id(task) for task in covered_tasks}

    # 遍历原始 tasks，根据是否被覆盖设置卸载比例
    for task in tasks:
        if id(task) in covered_task_ids:
            offloading_ratio.append(1.0)  # 覆盖的任务卸载比例为 1
        else:
            offloading_ratio.append(0.0)  # 未覆盖的任务卸载比例为 0

    # 构造 decision 元组
    decision = (np.array(offloading_ratio), np.float64(UAV_MAX_RESOURCE))  # 设置卸载比例和无人机分配的计算资源(文章采用全部卸载，并且边缘服务器以恒定计算能力处理任务)
    fly_distance = distance(best_position, uav.coordinate)
    flight_energy = 0.5 * UNITE_SLOT_LENGTH * ((fly_distance / UNITE_SLOT_LENGTH) ** 2) * UAV_QUALITY
    uav.flight_energy[time_slot] = flight_energy
    print(f"HOLD - best_position = {best_position}\n"
          f"HOLD - decision = {decision}\n"
          f"HOLD - distance = {fly_distance},  flight_energy = {flight_energy}")
    print(f"====================================HOLD结束")

    return best_position, decision








