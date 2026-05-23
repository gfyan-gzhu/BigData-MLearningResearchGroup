from simEnvParameter import *
from typing import *
import numpy as np
from collections import deque


# 子任务类
class subtask(object):
    # 移动设备id
    device_id: int
    # 子任务id
    subtask_id: int
    # 子任务前驱任务列表
    predecessors: List['subtask']
    # 创建的时隙
    generate_time: float
    # 任务数据大小
    data_length: float
    # 任务最大完成时间
    max_finish_time: float
    # 任务本地处理参数
    local_compute: int
    # 任务UAV处理参数
    uav_compute: int
    # 任务卸载到哪台UAV参数
    uav_offload_location: int
    # 任务计算时延
    compute_delay: float
    # 任务传输时延
    transmission_delay: float
    # 任务完成的时隙
    finished_time: float
    # 任务在本地计算能耗
    device_compute_energy: float
    # 任务在无人机计算能耗
    uav_compute_energy: float

    #   任务初始化函数，初始化任务信息
    def __init__(self,
                 device_id: int,
                 time_slot: int,
                 data_length: float,
                 max_finish_time: float,
                 subtask_id: int):
        self.device_id = device_id
        self.subtask_id = subtask_id
        # 前驱子任务对象列表
        self.predecessors: List["subtask"] = []
        self.generate_time = time_slot
        # 随机生成任务大小，单位bytes
        self.data_length = data_length
        # 随机生成任务完成时间，单位s
        self.max_finish_time = max_finish_time
        # 默认全部本地计算
        self.local_compute = 1
        # 默认UAV不计算
        self.uav_compute = 0
        self.uav_offload_location = None
        # 初始任务完成时隙为最大完成时间
        self.finished_time = self.max_finish_time + self.generate_time
        self.compute_delay = 0
        self.transmission_delay = 0
        # 初始化任务本地计算能耗和无人机计算能耗
        self.device_compute_energy = 0
        self.uav_compute_energy = 0
        return None




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
    # 任务完成的时隙
    finished_time: float
    # 任务的计算能耗
    compute_energy: float
    # 子任务列表
    subtasks: List[subtask]

    #   任务初始化函数，初始化任务信息
    def __init__(self,
                 device_id: int,
                 time_slot: int):
        self.device_id = device_id
        self.generate_time = time_slot
        # 随机生成任务大小，单位bit（bit,byte,kb,mb）
        self.data_length = np.random.rand() * \
                           (DEVICE_TASK_UPPER_LIMIT - DEVICE_TASK_LOWER_LIMIT) * 1e6 + \
                           DEVICE_TASK_LOWER_LIMIT * 1e6
        # 生成任务完成时间，单位s
        self.max_finish_time = 2
        # 默认全部本地计算
        self.offload_ratio = 0
        # 初始任务完成时隙为最大完成时间
        self.finished_time = self.max_finish_time + self.generate_time
        # 初始化任务计算能耗
        self.compute_energy = 0
        # 初始化子任务列表
        self.subtasks = []
        self._generate_subtasks()
        return None

    # DAG：7 个子任务固定依赖
    def _build_dag(self) -> Dict[int, List[int]]:
        return {
            0: [1, 2],
            1: [3],
            2: [4, 5],
            3: [6],
            4: [6],
            5: [6],
            6: []
        }

    @staticmethod
    def _topo_sort(dag: Dict[int, List[int]]) -> List[int]:
        """
        对 7 个子任务构成的 DAG 进行拓扑排序。
        参数:
            dag: 邻接表，key 为子任务编号(0~6)，value 为该节点直接指向的后继列表。
        返回:
            topo: 拓扑序列表，保证任意子任务出现在其所有前驱之后。
        """
        # 1. 统计每个节点的入度
        indeg = {i: 0 for i in range(7)}
        for u in dag:  # 遍历每条出边
            for v in dag[u]:
                indeg[v] += 1  # 终点入度+1

        # 2. 入度为 0 的节点先入队
        q = deque([i for i in range(7) if indeg[i] == 0])

        topo = []  # 保存拓扑序结果
        while q:
            u = q.popleft()  # 取出当前无前驱的节点
            topo.append(u)  # 加入结果

            # 3. 删除 u 的所有出边：后继入度减 1
            for v in dag[u]:
                indeg[v] -= 1
                if indeg[v] == 0:  # 新的入度 0 节点加入队列
                    q.append(v)

        # 若 DAG 合法，topo 长度必为 7；否则图中有环（本实现假设输入无环）
        return topo

    def _generate_subtasks(self):
        dag = self._build_dag()
        topo_order = self._topo_sort(dag)

        # 构建反向依赖图
        reverse_dag = {i: [] for i in range(7)}
        for u in dag:
            for v in dag[u]:
                reverse_dag[v].append(u)

        # 首先按照固定范围生成子任务的最大完成时间
        max_finish_times = [0.0] * 7

        # 1. 子任务0: 0.5-0.75s
        max_finish_times[0] = np.random.uniform(0.5, 0.75)

        # 2. 子任务1: 0.9-1.25s，且必须大于子任务0
        min_time_1 = max(0.9, max_finish_times[0] + 0.05)  # 至少比子任务0晚0.05s
        max_time_1 = 1.25
        if min_time_1 <= max_time_1:
            max_finish_times[1] = np.random.uniform(min_time_1, max_time_1)
        else:
            max_finish_times[1] = min_time_1

        # 3. 子任务2: 0.9-1.25s，且必须大于子任务0
        min_time_2 = max(0.9, max_finish_times[0] + 0.05)  # 至少比子任务0晚0.05s
        max_time_2 = 1.25
        if min_time_2 <= max_time_2:
            max_finish_times[2] = np.random.uniform(min_time_2, max_time_2)
        else:
            max_finish_times[2] = min_time_2

        # 4. 子任务3: 1.5-1.7s，且必须大于子任务1
        min_time_3 = max(1.5, max_finish_times[1] + 0.05)  # 至少比子任务1晚0.05s
        max_time_3 = 1.7
        if min_time_3 <= max_time_3:
            max_finish_times[3] = np.random.uniform(min_time_3, max_time_3)
        else:
            max_finish_times[3] = min_time_3

        # 5. 子任务4: 1.5-1.7s，且必须大于子任务2
        min_time_4 = max(1.5, max_finish_times[2] + 0.05)  # 至少比子任务2晚0.05s
        max_time_4 = 1.7
        if min_time_4 <= max_time_4:
            max_finish_times[4] = np.random.uniform(min_time_4, max_time_4)
        else:
            max_finish_times[4] = min_time_4

        # 6. 子任务5: 1.5-1.7s，且必须大于子任务2
        min_time_5 = max(1.5, max_finish_times[2] + 0.05)  # 至少比子任务2晚0.05s
        max_time_5 = 1.7
        if min_time_5 <= max_time_5:
            max_finish_times[5] = np.random.uniform(min_time_5, max_time_5)
        else:
            max_finish_times[5] = min_time_5

        # 7. 子任务6: 固定为2.0s
        max_finish_times[6] = 2.0

        # 8. 验证和调整依赖关系
        # 确保子任务6大于其所有前驱任务
        preds_of_6 = reverse_dag[6]  # [3, 4, 5]
        for pred in preds_of_6:
            if max_finish_times[pred] >= max_finish_times[6]:
                # 调整前驱任务的时间
                max_finish_times[pred] = max_finish_times[6] - np.random.uniform(0.05, 0.1)

        # 确保所有依赖关系
        for v in range(7):
            for u in reverse_dag[v]:
                if max_finish_times[u] >= max_finish_times[v]:
                    # 调整u的时间，使其小于v
                    diff = np.random.uniform(0.05, 0.1)
                    max_finish_times[u] = max_finish_times[v] - diff

        # 确保所有时间在指定范围内
        # 子任务0: 0.5-0.75
        max_finish_times[0] = np.clip(max_finish_times[0], 0.5, 0.75)

        # 子任务1、2: 0.9-1.25
        for i in [1, 2]:
            max_finish_times[i] = np.clip(max_finish_times[i], 0.9, 1.25)

        # 子任务3、4、5: 1.5-1.7
        for i in [3, 4, 5]:
            max_finish_times[i] = np.clip(max_finish_times[i], 1.5, 1.7)

        # 子任务6: 2.0
        max_finish_times[6] = 2.0

        # 现在，按照每个子任务的可用执行时间来分配数据量
        # 可用执行时间 = 最大完成时间 - max(前驱任务的最大完成时间)
        # 对于没有前驱的任务，可用执行时间 = 最大完成时间

        # 计算每个子任务的可用执行时间
        available_execution_times = [0.0] * 7

        for i in range(7):
            if not reverse_dag[i]:  # 没有前驱任务
                available_execution_times[i] = max_finish_times[i]
            else:
                # 找到所有前驱任务中最大的完成时间
                max_pred_time = max(max_finish_times[pred] for pred in reverse_dag[i])
                available_execution_times[i] = max_finish_times[i] - max_pred_time

        # 确保可用执行时间为正数
        for i in range(7):
            if available_execution_times[i] <= 0:
                # 如果可用执行时间为负或零，调整前驱任务的时间
                # 这是一个简化处理，实际上应该重新调整时间安排
                available_execution_times[i] = 0.05  # 最小执行时间

        # 考虑任务复杂度权重
        complexity_weights = {
            0: 1.0,  # 入口任务，相对简单
            1: 1.5,  # 中间处理
            2: 1.5,  # 中间处理
            3: 2.0,  # 数据处理
            4: 2.0,  # 数据处理
            5: 2.0,  # 数据处理
            6: 1.5  # 结果汇总
        }

        # 计算加权可用执行时间 = 可用执行时间 * 复杂度权重
        weighted_available_times = [
            available_execution_times[i] * complexity_weights[i]
            for i in range(7)
        ]

        # 计算总加权可用执行时间
        total_weighted_available_time = sum(weighted_available_times)

        # 基础数据量分配比例 = 加权可用执行时间 / 总加权可用执行时间
        base_data_ratios = [
            weighted_time / total_weighted_available_time
            for weighted_time in weighted_available_times
        ]

        # 添加随机扰动，但保持合理性
        perturbation = np.random.normal(0, 0.03, 7)  # 减小扰动幅度
        data_ratios = np.clip(base_data_ratios + perturbation, 0.05, 0.3)
        data_ratios = data_ratios / np.sum(data_ratios)

        # 计算子任务数据量
        sub_data = self.data_length * data_ratios

        # 验证数据量分配的合理性
        # 确保每个子任务的数据量与其可用执行时间成正比
        # 计算数据量与可用执行时间的比例
        data_per_time = [
            sub_data[i] / max(available_execution_times[i], 0.001)
            for i in range(7)
        ]

        # 如果某个任务的数据密度过高，重新调整
        avg_data_per_time = np.mean(data_per_time)
        std_data_per_time = np.std(data_per_time)

        for i in range(7):
            # 如果数据密度超过平均值2个标准差，调整
            if data_per_time[i] > avg_data_per_time + 2 * std_data_per_time:
                # 减少该任务的数据量
                reduction_factor = (avg_data_per_time + std_data_per_time) / data_per_time[i]
                sub_data[i] *= reduction_factor

        # 重新调整数据量，确保总和正确
        total_sub_data = sum(sub_data)
        if abs(total_sub_data - self.data_length) > 1:
            # 按比例调整
            scale_factor = self.data_length / total_sub_data
            sub_data = [data * scale_factor for data in sub_data]

        # 创建子任务
        subtasks = []
        for i in range(7):
            st = subtask(device_id=self.device_id,
                         time_slot=self.generate_time,
                         data_length=sub_data[i],
                         max_finish_time=max_finish_times[i],
                         subtask_id=i)
            subtasks.append(st)

        # 设置前驱关系
        sid2st = {st.subtask_id: st for st in subtasks}
        for u in dag:
            for v in dag[u]:
                sid2st[v].predecessors.append(sid2st[u])

        self.subtasks = subtasks
        return None


# 移动设备类
class device(object):

    # 移动设备id
    id: int
    # 当前移动设备坐标,coordinate[0][0]为x，coordinate[0][1]为y
    coordinate: List[float]
    # 移动设备坐标记录,用字典保存
    trajectory: {}
    # 移动设备计算能耗记录
    compute_energy: {}
    # 记录每个时隙队列长度
    queue_record: {}
    # 移动设备任务记录
    task: {}
    # 移动设备分配资源量
    compute_resource: float
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
        # 初始化移动设备计算能耗记录
        self.compute_energy = {}
        # 初始化队列长度记录
        self.queue_record = {0:0}
        # 初始化任务记录
        self.task = {}
        # 初始化分配资源量
        self.compute_resource = DEVICE_RESOURCE
        #初始化设备分配的带宽
        self.bandwidth = DEVICE_BANDWIDTH
        # 初始化最后产生任务的时隙
        self.final_task_slot = 0
        # 初始化设备队列长度
        self.queue_data_length = 0
        # 初始化设备任务等待队列;  使用collections.deque队列，支持访问所有元素
        self.task_queue = deque()

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
            return False


    # 计算队列长度
    def fun_calculate_queue_length(self):
        if not self.task_queue:  # 判断队列是否为空
            return 0
        total_data_length = 0
        for task_item in self.task_queue:
            total_data_length += task_item.data_length
        return total_data_length * DEVICE_1BIT_CYCLE


# 无人机类
class uav(object):
    # 无人机id
    uav_id: int
    # 当前设备坐标
    coordinate: List[float]
    # 坐标记录
    trajectory: {}
    # 计算能耗记录
    compute_energy: {}
    # 记录每个时隙队列长度
    queue_record: {}
    # 重量
    quality: float
    # 高度
    high: float
    # 无人机分配资源量
    compute_resource: float
    # 分配的带宽
    bandwidth: float
    # 无人机最大移动速度
    max_speed: float
    # 无人机队列长度
    queue_data_length: int
    # 任务等待队列
    task_queue = None

    def __init__(self,uav_id):
        self.uav_id = uav_id
        self.coordinate = []
        self.trajectory = {}
        self.compute_energy = {}
        self.queue_record = {0:0}
        self.quality = UAV_QUALITY
        self.high = np.random.uniform(UAV_MIN_HIGH, UAV_MAX_HIGH)
        self.compute_resource = UAV_RESOURCE
        self.bandwidth = UAV_BANDWIDTH
        self.max_speed = UAV_MAX_SPEED
        self.task_queue = deque()

    # 采用固定的初始位置（含高度）
    def fun_random_coordinate(self, num: int) -> bool:
        x, y = UAV_START_COORDINATE[num]  # 仅返回二维
        z = self.high  # 保留已有高度（__init__ 已随机）
        self.coordinate = [x, y, z]  # 三维
        self.trajectory[0] = self.coordinate.copy()
        return True


    # 计算队列长度
    def fun_calculate_queue_length(self):
        if not self.task_queue:  # 判断队列是否为空
            return 0
        total_data_length = 0
        for task_item in self.task_queue:
            total_data_length += task_item.data_length
        return total_data_length * UAV_1BIT_CYCLE












