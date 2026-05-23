
from simEnvParameter import *
from typing import *
import numpy as np
from collections import deque





# 任务类
class task(object):
    # 移动设备id
    device_id: int
    # 创建的时隙
    generate_time: float
    # 任务数据大小
    data_length: float
    # 任务最大完成时间
    max_finish_time: float
    # 任务卸载比例
    offload_ratio: float
    # 任务计算时延
    compute_delay: float
    # 任务传输时延
    transmission_delay: float
    # 任务完成的时隙
    finished_time: float
    # 任务在device端完成的时隙
    finish_time_device = float
    # 任务在uav端完成的时隙
    finish_time_uav = float
    # 任务传输消耗能耗
    transmission_energy: float
    # 任务在无人机计算总能耗
    compute_energy: float
    #   任务初始化函数，初始化任务信息
    def __init__(self,
                 device_id: int,
                 time_slot: int):
        self.device_id = device_id
        self.generate_time = time_slot
        # 随机生成任务大小，单位bit，范围1~2Mbits（bit,byte,kb,mb）
        self.data_length = np.random.rand() * (DEVICE_TASK_UPPER_LIMIT - 1) * 10**6 + 1 * 10**6
        # 随机生成任务完成时间，单位s
        self.max_finish_time = np.random.choice([1, 2])
        # 默认全部本地计算
        self.offload_ratio = 0
        # 初始任务完成时隙为最大完成时间
        self.finished_time = self.max_finish_time + self.generate_time
        self.finish_time_device = 0
        self.finish_time_uav = 0
        self.compute_delay = 0
        self.transmission_delay = 0
        # 初始化任务传输能耗和无人机计算能耗
        self.transmission_energy = 0
        self.compute_energy = 0
        return None





# 移动设备类
class device(object):

    # 移动设备id
    id: int
    # 当前移动设备坐标,coordinate[0][0]为x，coordinate[0][1]为y
    coordinate: List[float]
    # 移动设备坐标记录,用字典保存
    trajectory: {}
    # 移动设备任务记录
    task: {}
    # 移动设备分配资源量
    compute_resource: float
    # 移动设备分配资源记录
    compute_record: {}
    # 移动设备分配到的带宽
    bandwidth: float
    # 移动设备最后产生任务的时隙
    final_task_slot: int
    # 移动设备队列长度
    queue_data_length : int
    # 移动设备任务等待队列
    task_queue = None


    # 移动设备初始化方法
    def __init__(self, device_id: int):
        self.id = device_id
        # 初始化坐标列表
        self.coordinate = []
        # 初始化坐标记录
        self.trajectory = {}
        # 初始化任务记录
        self.task = {}
        # 初始化分配资源量
        self.compute_resource = 0
        # 初始化分配资源记录
        self.compute_record = {}
        #初始化设备分配的带宽
        self.bandwidth = DEVICE_BANDWIDTH
        # 初始化最后产生任务的时隙
        self.final_task_slot = 0
        # 初始化设备队列长度
        self.queue_data_length = 0
        # 初始化设备任务等待队列;  使用collections.deque队列，支持访问所有元素
        self.task_queue = deque()


    # 移动设备信息输出方法
    def fun_print_device_info(self):
        print(
            f"device_id = {self.id}\t info:||"
            f"coordinate = {self.coordinate},"
            f"trajectory = {self.trajectory},"
            f"task = {self.task},"
            f"computed_task_volumn = {self.computed_task_volumn},"
            f"task_queue = {self.task_queue}\n"
              )

    # 任务产生方法
    """
    self:设备对象
    time_slot:当前时隙
    poisson_rate:泊松到达率
    """
    def fun_generate_task(self, time_slot, poisson_rate = TASK_POISSON_RATE):
        # 时间间隔 = 时隙间隔 * 单位时隙的时间长度
        interval = (time_slot - self.final_task_slot) * (TOTAL_TIME / TOTAL_TIME_SLOT)
        pro_generate = 1 - np.exp(-poisson_rate * interval)
        if np.random.rand() < pro_generate:
            newTask = task(self.id, time_slot)
            self.task[time_slot] = (newTask)
            self.final_task_slot = time_slot
            return True
        else:
            self.task[time_slot] = None
            # 时隙未产生任务，保持计算量，并记录
            self.compute_record[time_slot] = self.compute_resource
            return False


    # 计算队列长度
    def fun_calculate_queue_length(self):
        if not self.task_queue:  # 判断队列是否为空
            return 0
        total_data_length = 0
        for task_item in self.task_queue:
            total_data_length += task_item.data_length * (1 - task_item.offload_ratio)
        return total_data_length







# 无人机类
class uav(object):
    # 当前设备坐标
    coordinate: list[float]
    # 坐标记录
    trajectory: {}
    # 飞行能耗记录
    flight_energy: {}
    # 传输能耗记录
    transmission_energy: {}
    # 计算能耗记录
    compute_energy: {}
    # 能耗记录（传输能耗+计算能耗）
    energy: {}
    # 重量
    quality: float
    # 高度
    high: float
    # 无人机分配资源量
    compute_resource: float
    # 无人机分配资源记录
    compute_record: {}
    # 最大可用计算资源
    available_max_resource: float
    # 分配的带宽
    bandwidth: float
    # 无人机最大移动速度
    max_speed: float
    # 无人机接收功率
    received_power: float
    # 无人机队列长度
    queue_data_length: int
    # 任务等待队列
    task_queue = None

    def __init__(self):
        self.coordinate = []
        self.trajectory = {}
        self.flight_energy = {}
        self.transmission_energy = {}
        self.compute_energy = {}
        self.energy = {}
        self.quality = UAV_QUALITY
        self.high = UAV_HIGH
        self.compute_resource = 0
        self.compute_record = {}
        self.available_max_resource = UAV_MAX_RESOURCE
        self.bandwidth = UAV_BANDWIDTH
        self.max_speed = UAV_MAX_SPEED
        self.received_power = UAV_RECEIVED_POWER
        self.task_queue = deque()

    # 采用固定的初始位置
    def fun_random_point_on_circle(self):
        self.coordinate = UAV_START_COORDINATE
        self.trajectory[0] = self.coordinate
        return True


    # 计算队列长度
    def fun_calculate_queue_length(self):
        if not self.task_queue:  # 判断队列是否为空
            return 0
        total_data_length = 0
        for task_item in self.task_queue:
            total_data_length += task_item.data_length * task_item.offload_ratio
        return total_data_length












