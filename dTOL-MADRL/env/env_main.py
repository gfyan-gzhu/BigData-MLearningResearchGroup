import random
import math

from env.vehicles import Vehicles
from env.task import Task
from env.availability_SV import AvailabilitySV
from env.transmission_rate import calculate_data_rate

# 生成车辆所用参数
RSU_coverage = 300  # RSU覆盖范围（单位：米）
road_length = 600  # 道路长度（单位：米）
num_lanes = 2  # 车道数
lane_width = 3  # 车道宽度（单位：米）
vehicle_num_TV = 10  # 单个车道的任务车辆数 5-15
vehicle_num_SV = 5  # 单个车道的服务车辆数
min_vehicle_speed = 10  # 最小车速（单位：米/秒）
max_vehicle_speed = 20  # 最大车速（单位：米/秒）

# 构建csv所用参数
availability_threshold = 0.5  # 是否成为服务车辆的阈值
max_communication = 300  # 车辆之间的最大通信范围
sigma = 0.3  # 保证任务执行成功的因素

# 计算能力、带宽、数据传输速率等参数
RSU_channel_bandwidth = 30  # RSU的总带宽
V2I_channel_bandwidth = RSU_channel_bandwidth / (vehicle_num_TV * num_lanes)  # V2I带宽
V2V_channel_bandwidth = 5  # V2V带宽
vehicle_pow = 600  # 车辆传输功率  600mW
data_rate_V2I = calculate_data_rate(RSU_coverage / 2, vehicle_pow, V2I_channel_bandwidth)  # V2I传输速率
data_rate_V2V = calculate_data_rate(max_communication / 2, vehicle_pow, V2V_channel_bandwidth)  # V2V传输速率
F_MEC = 15   # MEC服务器计算能力 Ghz
F_SV = 2   # SV计算能力 Ghz
F_TV = 1   # TV计算能力 Ghz

# 状态空间、动作空间
"""
卸载位置
[LOC,SV1,SV2,...,SV10,MEC]
[0,1,2,...,10,11][0-11]
卸载比例
[0-1]
"""
action_space = []
for i in range(vehicle_num_SV * num_lanes + 2):
    action_space.append(i)

# 效用函数中的常数
punish_H = -0.5  # 高优先级惩罚值
reward_L = 1  # 低优先级奖励值
c = 1  # 低优先级效用函数中的常数


class VecEnv:
    def __init__(self, num_T, T):
        """
        :type num_T: 时隙个数
        :type T:一个时隙大小
        """
        super(VecEnv, self).__init__()
        # 初始化车辆类
        self.vehicle_TV = Vehicles(vehicle_num_TV, road_length, num_lanes, lane_width, min_vehicle_speed
                                   , max_vehicle_speed)
        self.vehicle_SV = Vehicles(vehicle_num_SV, road_length, num_lanes, lane_width, min_vehicle_speed
                                   , max_vehicle_speed)
        self.TV_num = vehicle_num_TV * num_lanes  # 任务车辆数
        self.vehicle_TVs = []
        self.vehicle_SVs = []
        self.SVs_effective = []  # 记录所有服务车辆在哪个时刻空闲 初始[0,0,……,0]->[0,0.5,0.6],
        # 初始化任务
        self.task = Task()
        self.task_generate_times = []
        # 时隙
        self.num_T = num_T  # 时隙个数
        self.T = T  # 一个时隙大小
        # 初始化csv
        self.availabilitySV = AvailabilitySV(availability_threshold, max_communication,
                                             sigma)
        self.state_space1 = []  # 卸载位置算法的状态空间1
        self.state_space2 = []  # 卸载位置算法的状态空间2
        self.state_space3 = []  # 卸载比例算法的状态空间
        self.action_space = action_space  # 动作空间,卸载位置
        self.task_messages = []  # 临时存储任务信息
        self.TV_count = 0  # TV个数计数
        self.pre_action = []

    """
    S1: [[任务所需cpu周期数,任务大小,任务最大可容忍时延,任务优先级],[TV坐标x,TV坐标y,TV速度],[csv...]]
                0                                                     1               2   
    S2: []
    """

    def reset(self):
        self.state_space1.clear()  # 重置状态空间
        self.state_space2.clear()  # 重置状态空间
        self.state_space3.clear()
        # self.availabilitySV.csvs.clear()  # 重置候选服务车辆列表
        self.SVs_effective.clear()  # 记录所有服务车辆在哪个时刻空闲
        self.task.task_queue_mec.clear()  # 重置在MEC队列
        self.task.queue_time = 0  # 重置MEC队列时延
        self.pre_action.clear()  # 重置前一时隙存储的决策
        self.task_generate_times = self.task.task_generate_time(
            vehicle_num_TV * num_lanes * self.num_T, self.num_T, self.T).tolist()  # 任务生成时间
        self.vehicle_TV.vehicles.clear()  # 重置任务车辆集合
        self.vehicle_TVs = self.vehicle_TV.initialization_vehicle()  # 初始化任务车辆
        self.vehicle_SV.vehicles.clear()  # 重置服务车辆集合
        self.vehicle_SVs = self.vehicle_SV.initialization_vehicle()  # 初始化服务车辆
        for _ in range(vehicle_num_SV * num_lanes):
            self.SVs_effective.append(0)
        self.TV_count = 0

        # 构建状态空间S1
        # 生成任务
        self.task_messages = self.task.task_generate()
        self.state_space1.append(self.task_messages)  # 添加任务信息 0
        self.state_space1.append(self.vehicle_TVs[0])  # TV信息 1
        # 构建csv
        csv = self.availabilitySV.csv(self.vehicle_TVs[0], self.vehicle_SVs,
                                      self.SVs_effective, vehicle_pow, data_rate_V2V,
                                      self.task_generate_times[0], self.task_messages)
        self.state_space1.append(csv)  # 2
        # 构建状态空间S2
        total = 1
        for _ in range(vehicle_num_TV * num_lanes):
            temp_decides = []
            decide_location1 = random.randint(0, 11)
            decide_location2 = random.randint(0, 11)
            decide_location3 = random.randint(0, 11)
            decide_rate1 = random.triangular(0, total, 0)
            decide_rate2 = random.triangular(0, total - decide_rate1, 0)
            decide_rate3 = total - decide_rate1 - decide_rate2
            temp_decides.extend([decide_location1, decide_rate1, decide_location2,
                                 decide_rate2, decide_location3, decide_rate3])
            self.state_space2.append(temp_decides)
        state1 = sum(self.state_space1, [])
        state2 = self.state_space2
        return state1, state2

    # 返回算法2的状态decides[位置,位置,位置]
    # S3[[卸载位置],[节点计算能力],[任务信息],[MEC队列时延]]
    def get_state3(self, decides):
        self.state_space3.clear()
        self.state_space3.append(decides)  # 卸载位置
        capacity_temp = []
        for idx in range(len(decides)):
            if decides[idx] == 0:  # 本地计算
                capacity_temp.append(F_TV)
            elif decides[idx] == 11:  # MEC计算
                capacity_temp.append(F_MEC)
            else:
                capacity_temp.append(F_SV)
        self.state_space3.append(capacity_temp)  # 节点计算能力
        self.state_space3.append(self.task_messages)  # 任务信息
        self.state_space3.append([self.task.queue_time])  # MEC队列时延
        s3 = sum(self.state_space3, [])
        return s3

    """
    action: [[位置,比例],[位置,比例],[位置,比例]]
    卸载位置
    [LOC,SV1,SV2,...,SV10,MEC]
    [0,1,2,...,10,11][0-11]
    卸载比例
    [0-1]
     S1: [[任务所需cpu周期数,任务大小,任务最大可容忍时延,任务优先级],[TV坐标x,TV坐标y,TV速度],[csv...]]
                0                                                     1               2   
    S2: []
    """

    def step(self, action):
        cost_time = [0, 0, 0]
        done = False
        s1 = self.state_space1
        s1_ = s1  # 存储下一个状态
        s2 = self.state_space2
        s2_ = s2  # 存储下一个状态
        self.pre_action.append(sum(action, []))  # 存储决策
        # 计算任务执行时间
        for action_idx in range(len(action)):
            subtask_cycle = s1[0][0] * action[action_idx][1]
            subtask_size = s1[0][1] * action[action_idx][1]
            if action[action_idx][0] == 0:  # 本地计算
                time_cost, arrival_time, exe_time = self.task.cal_time('LOC', F_TV, subtask_cycle, subtask_size, 0.1,
                                                                       self.task_generate_times[0])
                cost_time[action_idx] += time_cost  # 若卸载位置相同,则时间累加
            elif action[action_idx][0] == 11:  # MEC计算
                time_cost, arrival_time, exe_time = self.task.cal_time('MEC', F_MEC, subtask_cycle, subtask_size,
                                                                       data_rate_V2I, self.task_generate_times[0])
                cost_time[action_idx] += time_cost
            else:  # SV计算
                sv_idx = action[action_idx][0] - 1
                time_cost, arrival_time, exe_time = self.task.cal_time('V2V', F_SV, subtask_cycle, subtask_size,
                                                                       data_rate_V2V, self.task_generate_times[0])
                if s1[2][sv_idx] == 0:  # 所选的车辆不可用
                    time_cost = s1[0][2] + 0.0001  # 
                    cost_time[action_idx] += time_cost  # 任务完成耗时
                else:
                    if arrival_time >= self.SVs_effective[sv_idx]:  # 任务到达时刻大于空闲时刻
                        cost_time[action_idx] += time_cost  # 任务完成耗时
                        self.SVs_effective[sv_idx] = arrival_time + exe_time  # 更新空闲时刻
                    else:
                        cost_time[action_idx] += (time_cost + self.SVs_effective[sv_idx]
                                                  - arrival_time)  # 任务完成耗时
                        self.SVs_effective[sv_idx] += exe_time  # 更新空闲时刻

        sum_time = max(cost_time)  # 任务执行总时间
        # 根据优先级计算效用
        if s1[0][3] == 3:  # 高优先级
            if sum_time <= s1[0][2]:
                utility = math.log(1 + s1[0][2] - sum_time)
            else:
                utility = punish_H
        elif s1[0][3] == 2:  # 中优先级
            if sum_time <= s1[0][2]:
                utility = math.log(1 + s1[0][2] - sum_time)
            else:
                utility = 0
        else:  # 低优先级
            if sum_time <= s1[0][2]:
                utility = reward_L
            else:
                utility = reward_L * math.exp(-c * (sum_time - s1[0][2]))
        reward = utility

        self.TV_count += 1
        # 下一个时隙
        if self.TV_count >= vehicle_num_TV * num_lanes:
            s2_ = self.pre_action
            self.TV_count = 0
            self.pre_action = []
            self.vehicle_TVs = self.vehicle_TV.vehicles_move(self.T)  # 车辆移动
            self.vehicle_SVs = self.vehicle_SV.vehicles_move(self.T)
            done = True
        # 更新任务到达时间
        self.task_generate_times.remove(self.task_generate_times[0])
        # 更新状态
        # 更新任务信息
        self.task_messages = self.task.task_generate()
        s1_[0] = self.task_messages  # 0
        s1_[1] = self.vehicle_TVs[self.TV_count]  # TV信息 1
        csv = self.availabilitySV.csv(self.vehicle_TVs[self.TV_count], self.vehicle_SVs,
                                      self.SVs_effective, vehicle_pow, data_rate_V2V,
                                      self.task_generate_times[0], self.task_messages)
        s1_[2] = csv  # csv信息 2
        self.state_space1 = s1_  # 更新状态
        self.state_space2 = s2_
        s1_ = sum(s1_, [])

        return s1_, s2_, reward, done, sum_time


