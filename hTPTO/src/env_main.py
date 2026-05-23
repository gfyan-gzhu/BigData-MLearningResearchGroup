from coordinateManage import get_coordinate, change_coordinate
from UAVNumberOptimize import optimize_uav_number
from uavTrajectoryDecision import *
from utils import *
from simEnvParameter import *


# 根据AC的卸载决策获得DDPG的状态
def get_ddpg_state(devices, tasks, using_uavs, offload_decisions):
    ddpg_state_space = []
    ddpg_state_space.append(devices)
    ddpg_state_space.append(tasks)
    ddpg_state_space.append(using_uavs)
    ddpg_state_space.append(offload_decisions)
    return ddpg_state_space


# 根据DDPG的无人机位置决策获得AC的状态
def get_ac_state(devices, tasks, using_uavs, uavs_coordinate_decisions):
    ac_state_space = []
    ac_state_space.append(devices)
    ac_state_space.append(tasks)
    ac_state_space.append(using_uavs)
    ac_state_space.append(uavs_coordinate_decisions)
    return ac_state_space


# 强化学习环境类
class MultiUavEnv:
    def __init__(self, time_slot=1):
        """
        初始化环境：包括初始化所有设备，无人机，无人机数量等
        关键修改：初始化所有4台无人机，并通过优化算法选择当前时隙使用的无人机
        """
        # 初始化设备
        self.devices = []
        # 初始化保存时隙产生任务的设备
        self.hasTaskDevices = []
        for i in range(DEVICE_NUM):
            self.devices.append(device(i))
        # 初始化设备坐标
        get_coordinate(self.devices)

        # 在初始化时生成初始任务
        self.tasks_record = {}  # 各时隙任务记录
        self.tasks = []  # 当前时隙生成的任务
        for i in range(DEVICE_NUM):
            if self.devices[i].fun_generate_task(time_slot):
                self.tasks.append(self.devices[i].task[time_slot])
                self.hasTaskDevices.append(self.devices[i])

        self.tasks_record[time_slot] = self.tasks

        # 初始化所有无人机（4台）
        self.all_uavs = []  # 保存所有无人机
        for i in range(UAV_MAX_NUM):
            uav_obj = uav(i)
            uav_obj.fun_random_coordinate(i)  # 为每台无人机分配初始位置
            self.all_uavs.append(uav_obj)

        # 通过优化算法选择当前时隙使用的无人机
        self.uav_num_record = {}  # 无人机各时隙部署数量记录
        self.uav_num, self.selected_uav_indices = optimize_uav_number(
            self.hasTaskDevices, self.all_uavs, time_slot
        )

        # 获取被选中的无人机对象
        self.using_uavs = [self.all_uavs[i] for i in self.selected_uav_indices]
        self.uav_num_record[time_slot] = self.uav_num

        # 环境状态
        self.ddpg_state_space = []
        self.ac_state_space = []
        self.ddpg_action = []
        self.ac_action = []

    def reset(self, time_slot=1):
        """
        将环境回退到任意指定时隙的初始状态：
        1. 清空任务历史
        2. 重置所有设备/无人机内部状态
        3. 重新生成坐标与任务
        4. 重新优化无人机数量
        5. 清空上一步动作/状态缓存
        不重新实例化对象，避免内存泄漏和随机种子错乱。
        """
        # 任务相关清空
        self.tasks_record.clear()
        self.tasks = []
        self.hasTaskDevices = []

        # 重置设备
        self.devices = []
        for i in range(DEVICE_NUM):
            self.devices.append(device(i))
        # 初始化设备坐标
        get_coordinate(self.devices)

        # 生成初始任务
        self.tasks = []
        for i in range(DEVICE_NUM):
            if self.devices[i].fun_generate_task(time_slot):
                self.tasks.append(self.devices[i].task[time_slot])
                self.hasTaskDevices.append(self.devices[i])

        self.tasks_record[time_slot] = self.tasks

        # 重置所有无人机
        self.all_uavs = []
        for i in range(UAV_MAX_NUM):
            uav_obj = uav(i)
            uav_obj.fun_random_coordinate(i)
            self.all_uavs.append(uav_obj)

        # 通过优化算法选择当前时隙使用的无人机
        self.uav_num, self.selected_uav_indices = optimize_uav_number(
            self.hasTaskDevices, self.all_uavs, time_slot
        )

        # 获取被选中的无人机对象
        self.using_uavs = [self.all_uavs[i] for i in self.selected_uav_indices]
        self.uav_num_record[time_slot] = self.uav_num

        # 清空上层算法缓存
        self.ddpg_state_space = []
        self.ac_state_space = []
        self.ddpg_action = []
        self.ac_action = []

    def step(self, ddpg_trajectory_decisions, ac_offload_decisions, time_slot):
        """
        ddpg_trajectory_decisions 现在为坐标列表 [(x,y,z), ...]，长度等于 UAV_MAX_NUM。
        """
        done = False

        # ----- 计算违规惩罚（基于坐标决策） -----
        penalty = 0.0
        for uav_idx in self.selected_uav_indices:
            target_coord = ddpg_trajectory_decisions[uav_idx]
            uav_obj = self.all_uavs[uav_idx]
            if not ddpg_constrain_coord(uav_obj, target_coord):  # 使用新的约束检查函数
                penalty += 0.2  # 惩罚系数加大

        # 1. 更新被选中无人机的位置（直接使用坐标）
        for idx, uav_idx in enumerate(self.selected_uav_indices):
            if idx < len(ddpg_trajectory_decisions):
                target_coord = ddpg_trajectory_decisions[uav_idx]
                uav_obj = self.all_uavs[uav_idx]
                # 直接赋值（已经过安全处理）
                uav_obj.coordinate = list(target_coord)
                uav_obj.trajectory[time_slot] = list(target_coord)


        # 2. 任务时延能耗计算（使用所有无人机）
        task_time_energy(self.tasks, self.devices, self.all_uavs, ac_offload_decisions)

        # 3. 地面设备和所有无人机队列更新
        enqueue_dequeue(self.tasks, self.devices, self.all_uavs, ac_offload_decisions, time_slot)

        # 4. 根据DDPG决策计算ddpg_reward（传入惩罚）
        ddpg_reward = get_ddpg_reword(self.devices, self.tasks, self.all_uavs,
                                      ddpg_trajectory_decisions, ac_offload_decisions, penalty)

        # 5. 根据AC决策计算ac_reward
        ac_reward = get_ac_reword(self.tasks, ac_offload_decisions, self.devices, self.all_uavs)

        done = True

        # 6. 时隙进一步
        time_slot += 1

        # 7. 地面设备移动
        change_coordinate(self.devices, time_slot)

        # 8. 地面设备生成新任务
        self.hasTaskDevices = []
        self.tasks = []
        for i in range(DEVICE_NUM):
            if self.devices[i].fun_generate_task(time_slot):
                self.tasks.append(self.devices[i].task[time_slot])
                self.hasTaskDevices.append(self.devices[i])

        self.tasks_record[time_slot] = self.tasks

        # 9. 重新优化无人机选择
        self.uav_num, self.selected_uav_indices = optimize_uav_number(
            self.hasTaskDevices, self.all_uavs, time_slot
        )
        self.using_uavs = [self.all_uavs[i] for i in self.selected_uav_indices]

        return ddpg_reward, ac_reward, done