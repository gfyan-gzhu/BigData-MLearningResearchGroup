import math

from env.task import Task


class AvailabilitySV:
    def __init__(self, availability_threshold,
                 max_communication, sigma):
        """
        :param availability_threshold: 能否成为服务车辆的阈值
        :param vehicle_SVs: 服务车辆集合[(x,y,speed),...]
        :param max_communication:最大通信范围
        :param sigma: 保证任务成功执行的因素
        """
        super(AvailabilitySV, self).__init__()
        self.availability_threshold = availability_threshold
        self.max_communication = max_communication
        self.sigma = sigma

    # 构造CSV模型
    def csv(self, vehicle_TV, vehicle_SVs, SVs_effective, sv_computing_power,
            v2v_data_rate, generate_time, task_messages):
        """
        :param vehicle_TV:需要构建CSV模型的任务车辆(x,y,speed)
        :param vehicle_SVs: 服务车辆集合[(x,y,speed)....]
        :param SVs_effective: 记录所有服务车辆在哪个时刻空闲 初始[0,0,……,0] [0,0.5,0.6],
        再和任务到达时间比较，判断该时刻该服务车辆是否可以提供服务
        :param sv_computing_power: 服务车辆计算能力
        :param v2v_data_rate: v2v传输速率
        :param generate_time: 任务生成时间
        :param task_messages: 任务信息
        所需的cpu转数、任务数据大小、任务最大可容忍时延、任务优先级[c,d,t,p]
        :return: csv列表 [0,1,1,1....]
        """
        csvs = []
        # print("1", csvs)
        x_i = vehicle_TV[0]  # 任务车辆横坐标
        y_i = vehicle_TV[1]  # 任务车辆横坐标
        speed_i = vehicle_TV[2]  # 任务车辆速度
        for idx, (x_g, y_g, speed_g) in enumerate(vehicle_SVs):
            distance_i_g = math.sqrt((x_i - x_g) ** 2
                                     + (y_i - y_g) ** 2)  # 任务车辆和服务车辆的距离
            if distance_i_g > self.max_communication:  # 超出通信范围
                P_cand = 0
            else:
                # 求TV和SV的最大通信时间
                if speed_i == speed_g:
                    T_max = 10
                else:
                    theta = math.asin(abs(y_i - y_g) / self.max_communication)
                    if x_i == x_g:
                        delta = (speed_i - speed_g) / abs(speed_i - speed_g)
                    else:
                        delta = ((x_i - x_g) * (speed_i - speed_g)) / (abs(x_i - x_g)
                                                                       * abs(speed_i - speed_g))
                    # TV和SV最大通信时间
                    T_max = ((self.max_communication * math.cos(theta) -
                              distance_i_g * math.cos(theta) * delta)) / abs(speed_i - speed_g)
                # 通信时间的影响 P_t
                task = Task()
                t_sv, arrival_time, exe_time = task.cal_time('V2V', sv_computing_power,
                                                             task_messages[0] * 0.5, task_messages[1] * 0.5,
                                                             v2v_data_rate, generate_time)
                if T_max > task_messages[2]:
                    P_t = 1
                elif t_sv + self.sigma <= T_max <= task_messages[2]:
                    P_t = T_max / task_messages[2]
                else:
                    P_t = 0
                # 是否在为其它车辆服务
                if arrival_time >= SVs_effective[idx]:  # 任务到达时刻 >= SV空闲时刻
                    P_f = 1
                else:
                    P_f = 0
                # 判断是否能成为CSV
                if P_t * P_f >= self.availability_threshold:
                    P_cand = 1
                else:
                    P_cand = 0
            csvs.append(P_cand)
        return csvs
