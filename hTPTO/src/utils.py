from simEnv import *
from simEnvParameter import *

import math

# ---------- 计算 UAV 与任务点之间的 3D 欧氏距离 ----------
def distance_3d(uav_pos, task_pos):
    dx = uav_pos[0] - task_pos[0]
    dy = uav_pos[1] - task_pos[1]
    dz = uav_pos[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

# ---------- 计算 UAV 与任务点之间为 LoS 链路的概率 ----------
def los_probability(uav_pos, task_pos, alpha=9.61, beta=0.16):
    dx = uav_pos[0] - task_pos[0]
    dy = uav_pos[1] - task_pos[1]
    d_2d = math.sqrt(dx ** 2 + dy ** 2)
    h = uav_pos[2]
    theta = math.atan(h / d_2d) * 180.0 / math.pi
    exp_term = -beta * (90.0 - theta - alpha)
    return 1.0 / (1.0 + alpha * math.exp(exp_term))

# ---------- 计算 UAV-任务点链路的平均路径损耗（dB） ----------
def path_loss(distance, los_prob, fc=2e9, eta_los=1, eta_nlos=20):
    c = 3e8
    base = 20.0 * math.log10(4.0 * math.pi * fc * distance / c)
    L_los = base + eta_los
    L_nlos = base + eta_nlos
    return los_prob * L_los + (1.0 - los_prob) * L_nlos

# ---------- 计算 UAV-任务点链路的下行数据速率（bps） ----------
def data_rate(path_loss_db, bandwidth=DEVICE_BANDWIDTH,
              dtp=DEVICE_TRANSMIT_POWER,
              np=NOISE_POWER):
    if path_loss_db <= 0:
        path_loss_db = 1e-6
    if bandwidth <= 0:
        bandwidth = DEVICE_BANDWIDTH
    L_linear = 10.0 ** (path_loss_db / 10.0)
    received_power = dtp / L_linear
    snr = received_power / np
    return bandwidth * math.log2(1.0 + snr)

# 移动设备时隙完成任务量
def device_computed_data():
    return (DEVICE_RESOURCE * UNITE_SLOT_LENGTH) / DEVICE_1BIT_CYCLE

# 无人机时隙完成任务量
def uav_computed_data():
    return (UAV_RESOURCE * UNITE_SLOT_LENGTH) / UAV_1BIT_CYCLE

# 移动设备本地计算时延计算
def device_compute_delay(subtask):
    return (subtask.data_length * DEVICE_1BIT_CYCLE) / DEVICE_RESOURCE

# 无人机卸载计算时延
def uav_compute_delay(subtask):
    return (subtask.data_length * UAV_1BIT_CYCLE) / UAV_RESOURCE

# 计算数据传输时延(设备到UAV)
def data_transmission_delay(subtask, device, uav):
    dist = distance_3d(uav.coordinate, device.coordinate)
    prob_los = los_probability(uav.coordinate, device.coordinate)
    pl_db = path_loss(dist, prob_los)
    rate = data_rate(pl_db)
    return subtask.data_length / rate

# 子任务最早开始时间（该子任务的前驱最晚结束时间）
def earliest_start_time(st: subtask) -> float:
    if not st.predecessors:
        return st.generate_time
    max_pred_finished = max(pred.finished_time for pred in st.predecessors)
    return max_pred_finished

# 移动设备队列等待时延
def device_queue_delay(device):
    return device.fun_calculate_queue_length() / DEVICE_RESOURCE

# 无人机队列等待时延
def uav_queue_delay(uav):
    return uav.fun_calculate_queue_length() / UAV_RESOURCE

# 无人机计算能耗
def uav_compute_energy(subtask, uav_resource=UAV_RESOURCE):
    return UAV_SWITCHING_CAPACITANCE * (uav_resource ** 3) * uav_compute_delay(subtask)

# 地面设备计算能耗
def device_compute_energy(subtask, device_resource=DEVICE_RESOURCE):
    return DEVICE_SWITCHING_CAPACITANCE * (device_resource ** 3) * device_compute_delay(subtask)

# 移动设备端总时延
def device_delay(subtask, device):
    total_delay = device_compute_delay(subtask) + \
                  max(device_queue_delay(device),
                      earliest_start_time(subtask))
    return total_delay

# 无人机段总时延
def uav_delay(subtask, device, uav):
    total_delay = uav_compute_delay(subtask) + \
                  max(uav_queue_delay(uav),
                      earliest_start_time(subtask),
                      data_transmission_delay(subtask, device, uav))
    return total_delay

# 无人机新坐标计算
def compute_uav_position(uav, ddpg_coordinate_decision):
    angle, _, d, h = ddpg_coordinate_decision
    dx = d * math.cos(angle)
    dy = d * math.sin(angle)
    x_new = uav.coordinate[0] + dx
    y_new = uav.coordinate[1] + dy
    return [x_new, y_new, h]

# 根据DDPG决策信息更新无人机位置
def update_uav_positions(uavs: List[uav],
                         uav_coordinate_decisions: List[tuple],
                         time_slot: int) -> None:
    for i, (uav, decision) in enumerate(zip(uavs, uav_coordinate_decisions)):
        if i >= len(uav_coordinate_decisions):
            break
        new_coordinate = compute_uav_position(uav, decision)
        uav.coordinate = new_coordinate
        uav.trajectory[time_slot] = new_coordinate.copy()

def ddpg_constrain_coord(uav: "uav", target_coord: tuple,
                         eps: float = 1e-3) -> bool:
    """
    检查目标坐标是否满足无人机移动约束。
    target_coord: (x, y, z)
    eps: 数值容差
    """
    x_new, y_new, z_new = target_coord

    # 高度范围
    if not (UAV_MIN_HIGH - eps <= z_new <= UAV_MAX_HIGH + eps):
        return False

    # 地图边界
    if not (-eps <= x_new <= GROUND_LENGTH + eps):
        return False
    if not (-eps <= y_new <= GROUND_WIDTH + eps):
        return False

    # 三维移动距离
    dx = x_new - uav.coordinate[0]
    dy = y_new - uav.coordinate[1]
    dz = z_new - uav.coordinate[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    max_allowed = UAV_MAX_SPEED * UNITE_SLOT_LENGTH

    if dist > max_allowed + eps:
        return False

    return True

# 约束校验（三维距离）
def ddpg_constrain(uav: "uav",
                   ddpg_coordinate_decision: tuple,
                   uav_max_speed: float = UAV_MAX_SPEED,
                   uav_max_high: float = UAV_MAX_HIGH,
                   uav_min_high: float = UAV_MIN_HIGH,
                   ground_length: float = GROUND_LENGTH,
                   ground_width: float = GROUND_WIDTH) -> bool:
    angle, _, d, h = ddpg_coordinate_decision

    if d < 0:
        return False
    if not (uav_min_high <= h <= uav_max_high):
        return False

    current_x, current_y, current_z = uav.coordinate
    dx = d * math.cos(angle)
    dy = d * math.sin(angle)
    x_new = current_x + dx
    y_new = current_y + dy
    z_new = h

    if not (0 <= x_new <= ground_length) or not (0 <= y_new <= ground_width):
        return False

    dx_total = x_new - current_x
    dy_total = y_new - current_y
    dz_total = z_new - current_z
    actual_3d_dist = math.sqrt(dx_total**2 + dy_total**2 + dz_total**2)
    max_allowed_dist = uav_max_speed * UNITE_SLOT_LENGTH
    if actual_3d_dist > max_allowed_dist:
        return False

    return True

# 对子任务进行拓扑排序
def topological_sort_dag(dag, num_nodes):
    in_degree = {i: 0 for i in range(num_nodes)}
    for u in dag:
        for v in dag[u]:
            in_degree[v] += 1
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in dag.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return result

# 随机生成卸载策略
def random_offload_decisions(tasks, uav_num, devices, uavs):
    if not tasks:
        return []
    all_decisions = []
    for task_idx, task in enumerate(tasks):
        subtasks = task.subtasks
        device = devices[task.device_id]
        dag = task._build_dag()
        topological_order = topological_sort_dag(dag, len(subtasks))
        task_decisions = [(0, 0, -1)] * len(subtasks)
        subtask_finished_times = [0.0] * len(subtasks)

        for subtask_idx in topological_order:
            st = subtasks[subtask_idx]
            earliest_start = st.generate_time
            if st.predecessors:
                max_pred_time = 0.0
                for pred in st.predecessors:
                    pred_idx = subtasks.index(pred)
                    pred_finish = subtask_finished_times[pred_idx]
                    if pred_finish > max_pred_time:
                        max_pred_time = pred_finish
                if max_pred_time > earliest_start:
                    earliest_start = max_pred_time

            possible_allocations = []
            local_compute_delay = device_compute_delay(st)
            local_queue_delay = device_queue_delay(device)
            local_start_time = max(earliest_start, local_queue_delay)
            local_finish_time = local_start_time + local_compute_delay
            possible_allocations.append((1, 0, -1, local_finish_time))

            for uav_idx in range(min(uav_num, len(uavs))):
                uav = uavs[uav_idx]
                trans_delay = data_transmission_delay(st, device, uav)
                uav_queue_delay_val = uav_queue_delay(uav)
                uav_compute_delay_val = uav_compute_delay(st)
                uav_start_time = max(earliest_start, trans_delay + uav_queue_delay_val)
                uav_finish_time = uav_start_time + uav_compute_delay_val
                possible_allocations.append((0, 1, uav_idx, uav_finish_time))

            possible_allocations.sort(key=lambda x: x[3])
            deadline = st.generate_time + st.max_finish_time
            selected_allocation = None
            for alloc in possible_allocations:
                lc, uc, uav_idx, finish_time = alloc
                if finish_time <= deadline:
                    selected_allocation = alloc
                    break
            if selected_allocation is None:
                selected_allocation = possible_allocations[0]

            lc, uc, uav_idx, finish_time = selected_allocation
            task_decisions[subtask_idx] = (lc, uc, uav_idx)
            subtask_finished_times[subtask_idx] = finish_time

        all_decisions.append(task_decisions)

    try:
        if ac_constrain(tasks, all_decisions, devices, uavs):
            print("--->随机卸载决策满足所有约束")
        else:
            print("===>随机卸载决策不满足约束")
    except Exception as e:
        print(f"约束检查出错: {e}")

    flat_decisions = []
    for task_decisions in all_decisions:
        flat_decisions.extend(task_decisions)
    return flat_decisions

# ac约束函数
def ac_constrain(
        tasks: List["task"],
        ac_offload_decisions: List[List[Tuple[int, int, int]]],
        devices: List["device"],
        uavs: List["uav"],
        slot_device_energy_budget: float = DEVICE_ENERGY_BUDGET,
        slot_uav_energy_budget: float = UAV_ENERGY_BUDGET) -> bool:

    total_device_energy = 0.0
    total_uav_energy = 0.0

    for task_idx, (task, decision) in enumerate(zip(tasks, ac_offload_decisions)):
        subtasks = task.subtasks
        if len(decision) != len(subtasks):
            print(f"决策信息长度违法！")
            return False

        for i, (lc, uc, uav_idx) in enumerate(decision):
            if lc + uc != 1:
                return False
            if uc == 1 and (uav_idx < 0 or uav_idx >= len(uavs)):
                return False
            st = subtasks[i]
            st.local_compute = lc
            st.uav_compute = uc
            st.uav_offload_location = uav_idx if uc else None

        dag = task._build_dag()
        topo_order = topological_sort_dag(dag, len(subtasks))

        for subtask_idx in topo_order:
            st = subtasks[subtask_idx]
            lc, uc, uav_idx = decision[subtask_idx]

            if lc:
                comp_d = device_compute_delay(st)
                queue_d = device_queue_delay(devices[task.device_id])
                trans_d = 0.0
                energy = device_compute_energy(st)
                total_device_energy += energy
            else:
                comp_d = uav_compute_delay(st)
                queue_d = uav_queue_delay(uavs[uav_idx])
                trans_d = data_transmission_delay(st, devices[task.device_id], uavs[uav_idx])
                energy = uav_compute_energy(st)
                total_uav_energy += energy

            if not st.predecessors:
                earliest_start = st.generate_time
            else:
                max_pred_finished = max(pred.finished_time for pred in st.predecessors)
                earliest_start = max_pred_finished

            start_time = max(earliest_start, trans_d + queue_d)
            finished = start_time + comp_d
            st.finished_time = finished

            deadline = st.generate_time + st.max_finish_time
            if finished > deadline:
                print(f"时延约束违法！！！ 子任务 {subtask_idx}: finished={finished}, deadline={deadline}")
                return False

        for st in subtasks:
            earliest_start = earliest_start_time(st)
            if st.finished_time - st.compute_delay < earliest_start:
                print(f"依赖约束违法！！！ 子任务 {st.subtask_id}")
                return False

    if total_device_energy > slot_device_energy_budget:
        print(f"本地计算能耗约束违法！！！")
        return False
    if total_uav_energy > slot_uav_energy_budget:
        print(f"无人机计算能耗约束违法！！！")
        return False

    return True

# 任务完成时间和能耗计算
def task_time_energy(
        tasks: List["task"],
        devices: List["device"],
        all_uavs: List["uav"],
        ac_offload_decisions: List[List[Tuple[int, int, int]]]
) -> None:
    for task, decision in zip(tasks, ac_offload_decisions):
        subtasks = task.subtasks
        device = devices[task.device_id]
        total_data = task.data_length
        offloaded_data = 0.0
        max_finished = 0.0
        total_energy = 0.0
        uav_set = set()

        for i, (lc, uc, uav_idx) in enumerate(decision):
            st = subtasks[i]

            if lc:
                comp_d = device_compute_delay(st)
                queue_d = device_queue_delay(device)
                trans_d = 0.0
                energy = device_compute_energy(st)
            else:
                if uav_idx < 0 or uav_idx >= len(all_uavs):
                    print(f"警告：无效的无人机索引 {uav_idx}，转为本地计算")
                    lc, uc = 1, 0
                    uav_idx = -1
                    comp_d = device_compute_delay(st)
                    queue_d = device_queue_delay(device)
                    trans_d = 0.0
                    energy = device_compute_energy(st)
                else:
                    comp_d = uav_compute_delay(st)
                    queue_d = uav_queue_delay(all_uavs[uav_idx])
                    trans_d = data_transmission_delay(st, device, all_uavs[uav_idx])
                    energy = uav_compute_energy(st)
                    offloaded_data += st.data_length
                    uav_set.add(uav_idx)

            if not st.predecessors:
                earliest_start = st.generate_time
            else:
                max_pred_finished = 0.0
                for pred in st.predecessors:
                    if hasattr(pred, 'finished_time'):
                        max_pred_finished = max(max_pred_finished, pred.finished_time)
                earliest_start = max_pred_finished

            start_time = max(earliest_start, trans_d + queue_d)
            finished = start_time + comp_d

            st.local_compute = lc
            st.uav_compute = uc
            st.uav_offload_location = uav_idx if uc else None
            st.compute_delay = comp_d
            st.transmission_delay = trans_d
            st.finished_time = finished
            st.device_compute_energy = energy if lc else 0.0
            st.uav_compute_energy = energy if uc else 0.0

            max_finished = max(max_finished, finished)
            total_energy += energy

        if total_data > 0:
            task.offload_ratio = offloaded_data / total_data
        else:
            task.offload_ratio = 0.0

        task.finished_time = max_finished
        task.compute_energy = total_energy

# 设备入队
def device_enqueue(device, subtask):
    if isinstance(subtask, list):
        subtask = subtask[0]
    device.task_queue.append(subtask)

# 无人机入队
def uav_enqueue(uav, subtask):
    if isinstance(subtask, list):
        subtask = subtask[0]
    uav.task_queue.append(subtask)

# 设备出队
def device_dequeue(device, time_slot: int):
    slot_end_sec = (time_slot + 1) * UNITE_SLOT_LENGTH
    for i in range(len(device.task_queue) - 1, -1, -1):
        if device.task_queue[i].finished_time <= slot_end_sec:
            del device.task_queue[i]

# 无人机出队
def uav_dequeue(uav, time_slot: int):
    slot_end_sec = (time_slot + 1) * UNITE_SLOT_LENGTH
    for i in range(len(uav.task_queue) - 1, -1, -1):
        if uav.task_queue[i].finished_time <= slot_end_sec:
            del uav.task_queue[i]

# 任务出入队操作
def enqueue_dequeue(tasks, devices, all_uavs, ac_offload_decisions, time_slot):
    task_time_energy(tasks, devices, all_uavs, ac_offload_decisions)
    for task, decision in zip(tasks, ac_offload_decisions):
        for i, (lc, uc, uav_idx) in enumerate(decision):
            st = task.subtasks[i]
            if lc:
                if 0 <= task.device_id < len(devices):
                    device_enqueue(devices[task.device_id], st)
            else:
                if 0 <= uav_idx < len(all_uavs):
                    uav_enqueue(all_uavs[uav_idx], st)
    for dev in devices:
        device_dequeue(dev, time_slot)
        dev.queue_record[time_slot + 1] = dev.fun_calculate_queue_length()
    for uav in all_uavs:
        uav_dequeue(uav, time_slot)
        uav.queue_record[time_slot + 1] = uav.fun_calculate_queue_length()

def calculate_uav_task_stats(tasks, ac_offload_decisions):
    stats = {}
    for task_idx, task_decision in enumerate(ac_offload_decisions):
        if task_idx >= len(tasks):
            continue
        task = tasks[task_idx]
        for subtask_idx, (lc, uc, uav_idx) in enumerate(task_decision):
            if subtask_idx >= len(task.subtasks):
                continue
            subtask = task.subtasks[subtask_idx]
            if uc == 1 and uav_idx >= 0:
                if uav_idx not in stats:
                    stats[uav_idx] = {'data': 0.0, 'count': 0}
                data_mb = subtask.data_length / (1024 * 1024)
                stats[uav_idx]['data'] += data_mb
                stats[uav_idx]['count'] += 1
    return stats

# 计算移动设备和无人机在每个时隙的总能耗
def compute_energy_per_slot(devices: List["device"],
                            uavs: List["uav"],
                            max_slot: int = TOTAL_TIME_SLOT):
    for dev in devices:
        dev.compute_energy = {t: 0.0 for t in range(1, max_slot + 1)}
    for uav in uavs:
        uav.compute_energy = {t: 0.0 for t in range(1, max_slot + 1)}

    for dev in devices:
        for task in dev.task.values():
            if task is None:
                continue
            for st in task.subtasks:
                if st.local_compute:
                    compute_start = st.finished_time - st.compute_delay
                    compute_end = st.finished_time
                    start_slot = int(compute_start // UNITE_SLOT_LENGTH) + 1
                    end_slot = int(compute_end // UNITE_SLOT_LENGTH) + 1
                    if start_slot == end_slot and 1 <= start_slot <= max_slot:
                        dev.compute_energy[start_slot] += st.device_compute_energy
                    else:
                        total_compute_time = st.compute_delay
                        for slot in range(start_slot, end_slot + 1):
                            if 1 <= slot <= max_slot:
                                slot_start = max(compute_start, (slot - 1) * UNITE_SLOT_LENGTH)
                                slot_end = min(compute_end, slot * UNITE_SLOT_LENGTH)
                                slot_compute_time = max(0, slot_end - slot_start)
                                energy_ratio = slot_compute_time / total_compute_time
                                slot_energy = st.device_compute_energy * energy_ratio
                                dev.compute_energy[slot] += slot_energy

    for uav in uavs:
        uav_idx = uavs.index(uav)
        for dev in devices:
            for task in dev.task.values():
                if task is None:
                    continue
                for st in task.subtasks:
                    if st.uav_compute and st.uav_offload_location == uav_idx:
                        compute_start = st.finished_time - st.compute_delay
                        compute_end = st.finished_time
                        start_slot = int(compute_start // UNITE_SLOT_LENGTH) + 1
                        end_slot = int(compute_end // UNITE_SLOT_LENGTH) + 1
                        if start_slot == end_slot and 1 <= start_slot <= max_slot:
                            uav.compute_energy[start_slot] += st.uav_compute_energy
                        else:
                            total_compute_time = st.compute_delay
                            for slot in range(start_slot, end_slot + 1):
                                if 1 <= slot <= max_slot:
                                    slot_start = max(compute_start, (slot - 1) * UNITE_SLOT_LENGTH)
                                    slot_end = min(compute_end, slot * UNITE_SLOT_LENGTH)
                                    slot_compute_time = max(0, slot_end - slot_start)
                                    energy_ratio = slot_compute_time / total_compute_time
                                    slot_energy = st.uav_compute_energy * energy_ratio
                                    uav.compute_energy[slot] += slot_energy

# 检查DDPG坐标决策违反（修改后版本，适配连续角度）
def check_constraints_detail(uav, ddpg_coordinate_decision):
    """
    详细检查DDPG坐标决策违反的约束。
    ddpg_coordinate_decision : (angle, _, d, h)
    """
    angle, _, d, h = ddpg_coordinate_decision
    violations = []

    # 1. 角度范围
    if angle < 0 or angle >= 2 * math.pi:
        violations.append(f"角度={angle} 不在 [0, 2π) 范围内")

    # 2. 高度范围
    if not (UAV_MIN_HIGH <= h <= UAV_MAX_HIGH):
        violations.append(f"高度h={h} 不在 [{UAV_MIN_HIGH}, {UAV_MAX_HIGH}] 范围内")

    # 3. 距离非负
    if d < 0:
        violations.append(f"距离d={d} 为负数")

    # 4. 计算目标坐标
    x_new, y_new, z_new = compute_uav_position(uav, ddpg_coordinate_decision)

    # 5. 地图边界
    if not (0 <= x_new <= GROUND_LENGTH):
        violations.append(f"x坐标={x_new} 超出[0,{GROUND_LENGTH}]范围")
    if not (0 <= y_new <= GROUND_WIDTH):
        violations.append(f"y坐标={y_new} 超出[0,{GROUND_WIDTH}]范围")

    # 6. 三维移动距离约束
    dx = x_new - uav.coordinate[0]
    dy = y_new - uav.coordinate[1]
    dz = z_new - uav.coordinate[2]
    actual_3d_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_dist = UAV_MAX_SPEED * UNITE_SLOT_LENGTH
    if actual_3d_dist > max_dist:
        violations.append(f"三维移动距离{actual_3d_dist:.2f}m 超过最大允许距离{max_dist}m")

    if violations:
        print(f"UAV约束违反详情: {', '.join(violations)}")

    return len(violations) == 0

# 获得DDPG reward
def get_ddpg_reword(devices: List["device"],
                    tasks: List["task"],
                    all_uavs: List["uav"],
                    uav_coordinate_decisions: List[tuple],
                    ac_offload_decisions: List[List[Tuple[int, int, int]]],
                    penalty: float = 0.0) -> float:
    """
    计算DDPG奖励：
        - 平均传输速率（Mbps）经对数压缩并缩放，使其与AC奖励尺度对齐（典型值0~0.2）
        - 加上探索奖励0.01
        - 减去违规惩罚
    不再进行[-1,1]截断，保留原始计算值以便DDPG获得梯度信息。
    """
    total_transmission_rate = 0.0
    transmission_count = 0

    for task_idx, (task, task_decisions) in enumerate(zip(tasks, ac_offload_decisions)):
        if task_idx >= len(tasks) or task_idx >= len(ac_offload_decisions):
            continue
        device = devices[task.device_id]
        subtasks = task.subtasks
        for subtask_idx, (lc, uc, uav_idx) in enumerate(task_decisions):
            if subtask_idx >= len(subtasks):
                continue
            if uc == 1 and uav_idx >= 0:
                if uav_idx >= len(all_uavs):
                    continue
                uav_obj = all_uavs[uav_idx]
                dist = distance_3d(uav_obj.coordinate, device.coordinate[:2])
                prob_los = los_probability(uav_obj.coordinate, device.coordinate[:2])
                pl_db = path_loss(dist, prob_los)
                rate = data_rate(pl_db)
                rate_mbps = rate / 1e6
                total_transmission_rate += rate_mbps
                transmission_count += 1

    if transmission_count == 0:
        # 没有卸载任务时，给予一个较小的正奖励（鼓励探索），并扣除惩罚
        reward = 0.1 - penalty
    else:
        avg_rate_mbps = total_transmission_rate / transmission_count
        # 对数压缩：log(1 + avg_rate_mbps)，除以缩放因子使数值减少
        compressed_rate = np.log1p(avg_rate_mbps) / 15.0
        exploration_bonus = 0.01
        reward = compressed_rate + exploration_bonus - penalty

    # 可选：添加一个宽松的边界防止极端值（例如-10~10），但非必需
    # reward = max(-10.0, min(10.0, reward))
    return reward

# 获得AC reward
def get_ac_reword(tasks: List["task"],
                  offload_decisions: List[List[Tuple[int, int, int]]],
                  devices: List["device"],
                  uavs: List["uav"],
                  alpha: float = ALPHA,
                  beta: float = BETA,
                  gamma: float = 0.002,
                  delta: float = 0.002,
                  target_offload_ratio: float = TARGET_OFFLOAD_RATIO) -> float:
    if not tasks:
        return 0.0

    total_delay = 0.0
    total_energy = 0.0
    total_subtasks = 0
    completed_subtasks = 0
    total_data = 0.0
    offloaded_data = 0.0
    total_overtime = 0.0

    task_time_energy(tasks, devices, uavs, offload_decisions)

    for task, decision in zip(tasks, offload_decisions):
        subtasks = task.subtasks
        dev = devices[task.device_id]
        total_data += task.data_length
        for i, (lc, uc, uav_idx) in enumerate(decision):
            if i >= len(subtasks):
                continue
            st = subtasks[i]
            total_subtasks += 1
            if uc == 1 and uav_idx >= 0 and uav_idx < len(uavs):
                offloaded_data += st.data_length
            deadline = st.generate_time + st.max_finish_time
            if st.finished_time <= deadline:
                completed_subtasks += 1
            else:
                overtime = st.finished_time - deadline
                total_overtime += overtime
            if lc:
                comp_d = device_compute_delay(st)
                queue_d = device_queue_delay(dev)
                trans_d = 0.0
                energy = device_compute_energy(st)
            else:
                if uav_idx < 0 or uav_idx >= len(uavs):
                    continue
                comp_d = uav_compute_delay(st)
                queue_d = uav_queue_delay(uavs[uav_idx])
                trans_d = data_transmission_delay(st, dev, uavs[uav_idx])
                energy = uav_compute_energy(st)
            if not st.predecessors:
                earliest_start = st.generate_time
            else:
                max_pred_finished = 0.0
                for pred in st.predecessors:
                    if hasattr(pred, 'finished_time'):
                        max_pred_finished = max(max_pred_finished, pred.finished_time)
                earliest_start = max_pred_finished
            dependency_wait_d = max(0, st.finished_time - st.compute_delay - earliest_start)
            wait_d = max(queue_d, dependency_wait_d, trans_d)
            total_delay += comp_d + wait_d
            total_energy += energy

    avg_delay = total_delay / total_subtasks if total_subtasks > 0 else 0.0
    avg_energy = total_energy / total_subtasks if total_subtasks > 0 else 0.0
    completion_rate = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
    offload_ratio = offloaded_data / total_data if total_data > 0 else 0.0

    offload_ratio_reward = np.exp(-((offload_ratio - target_offload_ratio) ** 2) / (2 * 0.2 ** 2))
    avg_overtime = total_overtime / total_subtasks if total_subtasks > 0 else 0.0
    overtime_penalty = -0.1 * min(avg_overtime, 5.0)

    try:
        if not ac_constrain(tasks, offload_decisions, devices, uavs):
            constraint_penalty = -0.003
            print(f"AC约束违反，施加软惩罚: {constraint_penalty:.3f}")
        else:
            constraint_penalty = 0.0
    except Exception as e:
        print(f"AC约束检查出错: {e}")
        constraint_penalty = -0.2

    if total_delay > 0 and total_energy > 0:
        delay_cost = alpha * total_delay
        energy_cost = beta * total_energy
        total_cost = delay_cost + energy_cost
        base_reward = 1.0 / (1.0 + total_cost)
    else:
        base_reward = 0.0

    completion_reward = gamma * completion_rate
    offload_reward = delta * offload_ratio_reward
    total_reward = base_reward + completion_reward + offload_reward + overtime_penalty + constraint_penalty
    total_reward = max(-0.3, min(1.0, total_reward))

    print(f"  平均时延: {avg_delay:.4f}s, 平均能耗: {avg_energy:.4f}J")
    print(f"  完成率: {completion_rate:.2%}, 卸载比例: {offload_ratio:.2%}")
    print(f"  超时惩罚: {overtime_penalty:.3f}, 约束惩罚: {constraint_penalty:.3f}")
    print(f"  基础奖励: {base_reward:.3f}, 完成奖励: {completion_reward:.3f}, 卸载奖励: {offload_reward:.3f}")
    print(f"  总奖励: {total_reward:.3f}")

    return total_reward