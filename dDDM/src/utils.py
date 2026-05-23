import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from simEnv import *
import math

# 无人机飞行能耗计算
def flight_energy_consumption(current_coord,
                              next_coord,
                              uav_quality = UAV_QUALITY,
                              unite_slot_length = UNITE_SLOT_LENGTH
                              ):
    """
    计算无人机飞行的能耗

    参数:
        current_coord (tuple): 当前坐标 (x_current, y_current)
        next_coord (tuple): 下一个坐标 (x_next, y_next)
        UAV_QUALITY (float): 无人机质量 (kg)
        UAV_MAX_SPEED (float): 无人机最大飞行速度 (m/s)
        UAV_PENALTY_FLIGHT (float): 飞行能耗惩罚系数

    返回:
        float: 飞行能耗 (J)
    """
    # 计算无人机的速度向量
    x_current, y_current = current_coord
    x_next, y_next = next_coord
    velocity_vector = (x_next - x_current, y_next - y_current)

    # 计算速度向量的大小
    velocity_magnitude = math.sqrt((velocity_vector[0] ** 2) + (velocity_vector[1] ** 2))

    # 计算无人机的飞行速度
    flight_speed = velocity_magnitude / unite_slot_length

    # 计算飞行能耗
    energy_consumption = 0.5 * (flight_speed ** 2) * uav_quality

    return energy_consumption




# PSO算法目标函数
def object_function_trajectory(tasks,
                               devices,
                               uav,
                               next_coord
                               ):
    # tasks_sum为该时隙总任务量大小
    tasks_sum, data_ratio_sum = 0, 0

    for i in range(len(tasks)):
        tasks_sum += tasks[i].data_length


    for i in range(len(tasks)):
        data_ratio_sum += (tasks[i].data_length / tasks_sum) * (math.sqrt(
            (devices[tasks[i].device_id].coordinate[0] - next_coord[0])**2
            +
            (devices[tasks[i].device_id].coordinate[1] - next_coord[1])**2
            +
            uav.high ** 2)
        )

    flight_energy = flight_energy_consumption(uav.coordinate, next_coord)


    return data_ratio_sum + (UAV_PENALTY_FLIGHT * flight_energy)


# 计算信道增益
def channel_gain(device, uav):
    # 获取设备坐标和无人机坐标
    device_coordinate = np.array([device.coordinate[0], device.coordinate[1], 0])
    uav_coordinate = np.array([uav.coordinate[0], uav.coordinate[1], uav.high])
    # 计算设备和无人机之间的距离
    distance = np.linalg.norm(device_coordinate - uav_coordinate)

    channel_gain = CHANNEL_POWER_GAIN / (distance ** 2)
    return channel_gain

# 计算上行数据传输速率:0-25（20） 25-50（10） 50-75（5） 75-100（2）Mbps
def data_rate(device,uav):
    transmission_rate = device.bandwidth \
           * \
           np.log2(1 + ((channel_gain(device,uav) * DEVICE_TRANSMIT_POWER) / (NOISE_POWER ** 2)))

    return transmission_rate


# 移动设备时隙完成任务量
def device_computed_data():
    return (DEVICE_RESOURCE * UNITE_SLOT_LENGTH) / DEVICE_1BIT_CYCLE


# 无人机时隙完成任务量
def uav_computed_data(uav_compute_resource):
    return (uav_compute_resource * UNITE_SLOT_LENGTH) / UAV_1BIT_CYCLE


# 移动设备本地计算时延计算
def device_compute_delay(task,
                         task_offload_ratio
                         ):
    return ((1 - task_offload_ratio) * task.data_length * DEVICE_1BIT_CYCLE) / DEVICE_RESOURCE



# 无人机计算时延
def uav_compute_delay(task, task_offload_ratio, uav_compute_resource):
    if uav_compute_resource <= 0:
        # 如果无人机资源为0，返回一个很大的延迟值（表示不可行）
        return float('inf')
    return (task_offload_ratio * task.data_length * UAV_1BIT_CYCLE) / uav_compute_resource


# 计算数据传输时延
def data_transmission_delay(task,task_offload_ratio,device,uav):
    return (task_offload_ratio * task.data_length) / data_rate(device,uav)


# 移动设备队列等待时延
def device_queue_delay(device):
    return device.fun_calculate_queue_length() / DEVICE_RESOURCE


# 无人机队列等待时延
def uav_queue_delay(uav, uav_compute_resource):
    if uav_compute_resource <= 0:
        # 如果无人机资源为0，返回一个很大的延迟值（表示不可行）
        return float('inf')
    return uav.fun_calculate_queue_length() / uav_compute_resource

# 无人机计算能耗——按任务所需时间计算
def uav_compute_energy(task,tasks_offload_ratio,uav_compute_resource):
    return  SWITCHING_CAPACITANCE * (uav_compute_resource ** 3) * uav_compute_delay(task,tasks_offload_ratio,uav_compute_resource)

# 无人机计算能耗——按资源分配量计算各时隙的能耗(该函数通过各时隙资源分配量计算无人机在时隙的计算能耗)
def uav_compute_energy_time(uav_compute_resource):
    return SWITCHING_CAPACITANCE * (uav_compute_resource ** 3) * UNITE_SLOT_LENGTH

# 无人机接收数据能耗
def uav_receive_energy(task, task_offload_ratio, device, uav):
    return UAV_RECEIVED_POWER * data_transmission_delay(task,task_offload_ratio,device,uav)

# 卸载与资源分配卸载决策目标函数
def objective_function_Off_Res_Decision(tasks,
                                        devices,
                                        uav,
                                        tasks_offload_ratio,
                                        uav_compute_resource,
                                        alpha= ALPHA,
                                        beta= BETA):
    # 加权和
    weighted_sum_delay, weighted_sum_energy = 0, 0

    for i in range(len(tasks)):
        # 时延和
        weighted_sum_delay += device_compute_delay(tasks[i],
                                                   tasks_offload_ratio[i])\
        +\
        device_queue_delay(devices[tasks[i].device_id])\
        +\
        uav_compute_delay(tasks[i], tasks_offload_ratio[i],uav_compute_resource)\
        +\
        max(data_transmission_delay(tasks[i],tasks_offload_ratio[i],devices[tasks[i].device_id],uav),
            uav_queue_delay(uav,uav_compute_resource))\
        +\
        uav_compute_delay(tasks[i],tasks_offload_ratio[i],uav_compute_resource)

        # 能耗和
        weighted_sum_energy += uav_compute_energy(tasks[i],tasks_offload_ratio[i],uav_compute_resource)\
        +\
        uav_receive_energy(tasks[i],tasks_offload_ratio[i],devices[tasks[i].device_id],uav)



    return alpha * weighted_sum_delay + beta * weighted_sum_energy






# 判断是否满足约束
def constrain(tasks, devices, uav, tasks_offload_ratio, uav_compute_resource):
    """
    检查所有约束是否满足

    参数:
        tasks: 任务列表。
        devices: 设备列表。
        uav: 无人机对象。
        tasks_offload_ratio: 任务卸载比例数组。
        device_compute_resource: 移动设备分配的计算资源数组。
        uav_compute_resource: 无人机分配的计算资源。

    返回:
        bool: 如果满足所有约束，返回 True；否则返回 False。
    """
    try:
        # 检查无人机资源约束
        if uav_compute_resource > UAV_MAX_RESOURCE:
            return False

        # 如果无人机资源为0，检查是否所有任务都是本地计算
        if uav_compute_resource <= 0:
            # 如果有任何任务卸载比例大于0，则不满足约束
            if any(ratio > 0 for ratio in tasks_offload_ratio):
                return False


        # 检查每个任务的完成时间约束
        for i, task in enumerate(tasks):
            device_obj = devices[task.device_id]

            # 计算总延迟，处理除零情况
            device_comp_delay = device_compute_delay(task, tasks_offload_ratio[i])
            device_queue_dly = device_queue_delay(device_obj)

            # 如果卸载比例为0，不需要计算传输和无人机延迟
            if tasks_offload_ratio[i] == 0:
                total_delay = device_comp_delay + device_queue_dly
            else:
                # 卸载比例大于0，需要检查无人机资源是否足够
                if uav_compute_resource <= 0:
                    return False

                trans_delay = data_transmission_delay(task, tasks_offload_ratio[i], device_obj, uav)
                uav_queue_dly = uav_queue_delay(uav, uav_compute_resource)
                uav_comp_delay = uav_compute_delay(task, tasks_offload_ratio[i], uav_compute_resource)

                total_delay = device_comp_delay + device_queue_dly + max(
                    trans_delay, uav_queue_dly
                ) + uav_comp_delay

            if total_delay > task.max_finish_time:
                return False

        return True

    except (ZeroDivisionError, ValueError) as e:
        # 捕获除零错误和其他数值错误
        return False

# 任务完成时间和能耗计算
def task_time_energy(task, device, uav, task_offload_ratio, uav_compute_resource):
    if task_offload_ratio > 0: # 卸载比率大于0，需要无人机辅助计算
        # 任务完成时间计算
        task.compute_delay = device_compute_delay(task,task_offload_ratio) + device_queue_delay(device)+max(
                                data_transmission_delay(task,task_offload_ratio,device,uav),
                                uav_queue_delay(uav,uav_compute_resource)
                                ) + uav_compute_delay(task,task_offload_ratio,uav_compute_resource)
        task.finished_time = task.generate_time + task.compute_delay

        # 任务各端完成的时隙计算
        task.finish_time_device = task.generate_time + device_compute_delay(task,task_offload_ratio) + device_queue_delay(device)
        task.finish_time_uav = task.generate_time + max(
                                                        data_transmission_delay(task,task_offload_ratio,device,uav),
                                                        uav_queue_delay(uav,uav_compute_resource)
                                                        ) + uav_compute_delay(task,task_offload_ratio,uav_compute_resource)
        # 任务传输时延
        task.transmission_delay = data_transmission_delay(task,task_offload_ratio,device,uav)
        # 任务传输能耗
        task.transmission_energy = uav_receive_energy(task,task_offload_ratio,device,uav)
        # 任务在无人机的处理能耗
        task.compute_energy = uav_compute_energy(task,task_offload_ratio,uav_compute_resource)
    else:   # 不需要无人机辅助计算
        task.compute_delay = device_compute_delay(task,task_offload_ratio) + device_queue_delay(device)
        task.finished_time = task.generate_time + task.compute_delay
        task.transmission_delay = 0
        task.transmission_energy = 0
        task.compute_energy = 0



# 设备入队(右端队尾)函数
def device_enqueue(device, task):
    if isinstance(task, list):
        task = task[0]  # 解包任务对象
    device.task_queue.append(task)


# 无人机入队（右端队尾）函数
def uav_enqueue(uav, task):
    if isinstance(task, list):
        task = task[0]  # 如果任务是列表，取出第一个元素
    uav.task_queue.append(task)




# 设备出队（左端队头）函数
def device_dequeue(device,time_slot):
    i = 0
    while i < len(device.task_queue):
        task = device.task_queue[i]
        if task.finish_time_device >= time_slot and task.finish_time_device < (time_slot + 1):
            del device.task_queue[i]
        else:
            i += 1


# 无人机出队（左端队头）函数
def uav_dequeue(uav,time_slot):
    i = 0
    while i < len(uav.task_queue):
        task = uav.task_queue[i]
        if task.finish_time_device >= time_slot and task.finish_time_device < (time_slot + 1):
            del uav.task_queue[i]  # 使用del来删除元素
        else:
            i += 1



def enqueue_dequeue(tasks, decision, devices, uav, time_slot):
    """
    遍历任务，将决策值记录，并进行任务入队和出队操作。

    参数:
        tasks: 任务列表。
        decision: 决策值（包括卸载比例和无人机计算资源）。
        devices: 设备列表。
        uav: 无人机对象。
        time_slot: 当前时间时隙。
    """
    for t, task in enumerate(tasks):
        task.offload_ratio = decision[0][t]  # 更新任务的卸载比例
        # UAV计算资源赋值并记录
        uav.compute_resource = decision[1].item()
        uav.compute_record[time_slot] = decision[1].item()

        if task.offload_ratio > 0:  # 如果任务需要卸载
            # 设备入队
            device_enqueue(devices[task.device_id], task)
            # UAV入队
            uav_enqueue(uav, task)
            # 计算任务的时延和能耗
            task_time_energy(task,
                             devices[task.device_id],
                             uav,
                             task.offload_ratio,
                             uav.compute_resource)
        else:
            # 任务完全不卸载，设备入队
            device_enqueue(devices[task.device_id], task)
            # 计算任务的时延和能耗
            task_time_energy(task,
                             devices[task.device_id],
                             uav,
                             task.offload_ratio,
                             uav.compute_resource)

    # 任务出队（设备）
    for j, device in enumerate(devices):
        device_dequeue(device, time_slot)

    # UAV任务出队
    uav_dequeue(uav, time_slot)


def uav_energy_record(uav, devices, time=TOTAL_TIME_SLOT):
    """
    计算无人机在每个时隙的总能耗，并分别记录飞行能耗、任务传输能耗和计算能耗，
    最后将各类能耗在各时隙累加到 energy 中。

    参数:
        uav: 无人机对象
        devices: 设备列表
        time: 总时隙数，默认为 TOTAL_TIME_SLOT
    """
    # 初始化各类能耗记录
    uav.transmission_energy = {t: 0 for t in range(1, time + 1)}
    uav.compute_energy = {t: 0 for t in range(1, time + 1)}
    uav.energy = {t: 0 for t in range(1, time + 1)}

    # 任务传输能耗记录
    for device in devices:
        for task in device.task.values():
            if task and 1 <= task.generate_time <= time:
                timeslot = task.generate_time
                uav.transmission_energy[timeslot] += task.transmission_energy

    # 计算能耗记录
    for time_slot, compute_resource in uav.compute_record.items():
        # 使用 uav_compute_energy_time 计算计算能耗
        compute_energy = uav_compute_energy_time(compute_resource)
        # 确保计算能耗加到对应的时隙中
        if 1 <= time_slot <= time:
            uav.compute_energy[time_slot] += compute_energy

    # 汇总能耗记录到 energy（总能耗=计算能耗+传输能耗）
    for t in range(1, time + 1):
        uav.energy[t] = (
            uav.transmission_energy.get(t)
            + uav.compute_energy.get(t)
        )


















