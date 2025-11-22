import random
import numpy as np


class Task:
    def __init__(self, ):
        super(Task, self).__init__()
        self.task_queue_mec = []
        self.queue_time = 0

    # 生成任务生成时间
    def task_generate_time(self, time_num, num_T, T):
        """
        :param time_num: 时隙总数
        :param num_T: 时隙个数
        :param T: 一个时隙大小
        :return: 生成任务时间的numpy列表
        """
        lam = 10
        # 生成泊松分布的到达时间
        arrival_times = np.random.exponential(1 / lam, time_num+1)
        arrival_times = np.cumsum(arrival_times)

        # 将生成时间缩放到0-1秒之间
        max_time = arrival_times[-1]
        arrival_times = arrival_times / max_time

        # 缩放到0-x秒之间
        task_generate_times = 0 + arrival_times * (T * num_T - 0)
        return task_generate_times

    # 随机生成任务,所需的cpu转数、任务数据大小、任务最大可容忍时延、任务优先级[c,d,t,p]
    # 任务优先级 高：3 (0.5s) 中：2 (1s) 低：1 (2s,4s)
    def task_generate(self, ):
        cycle = 500  # 500 cycle/byte
        subtask_size = round(random.uniform(2, 8), 1)  # 任务数据大小 2-8M
        subtask_cycle = subtask_size * 1e6 * cycle * 1e-9  # 任务所需的cpu周期数1-4GHz
        subtask_max_time = random.choice([0.5, 1, 2, 4])  # 任务最大可容忍时延
        if subtask_max_time <= 0.5:
            priority = 3
        elif subtask_max_time >= 2:
            priority = 1
        else:
            priority = 2
        task_messages = [subtask_cycle, subtask_size, subtask_max_time, priority]
        return task_messages

    def cal_time(self, action_decide, computing_power, subtask_cycle, subtask_size,
                 data_rate, generate_time):
        """
        :param action_decide: 卸载位置
        :param computing_power: CPU计算能力
        :param subtask_cycle: 子任务所需CPU周期数
        :param subtask_size: 子任务大小
        :param data_rate: 数据传输速率
        :param generate_time: 任务生成时间
        :return: 子任务时延, 任务到达时间, 任务执行时间
        """
        SV_effective_time = 0  # 服务车辆空闲的时刻
        exe_time = subtask_cycle / computing_power  # 任务执行时间
        transmission_time = subtask_size / data_rate  # 任务传输时间
        if action_decide == 'LOC':
            time_cost = exe_time
            arrival_time = generate_time

        elif action_decide == 'MEC':
            arrival_time = generate_time + transmission_time  # 任务到达时间
            self.task_queue_mec.append([arrival_time, exe_time])  # 将执行的任务加入MEC队列
            self.queue_time = self.cal_queue_time(self.task_queue_mec)
            time_cost = transmission_time + exe_time + self.queue_time

        else:  # V2V
            arrival_time = generate_time + transmission_time  # 任务到达时间
            time_cost = exe_time + transmission_time
            SV_effective_time = arrival_time + exe_time

        return time_cost, arrival_time, exe_time

    def cal_queue_time(self, task_queue):

        """
        计算队列时间
        :param task_queue: 任务队列[[任务到达时间,执行时间]]
        :return: 队列时延
        """
        exe_sum_time = 0
        length = len(task_queue)
        if length == 0:
            return 0
        for row in range(length - 1):
            exe_sum_time += task_queue[row][1]
        queue_time = exe_sum_time - task_queue[length - 1][0]
        if queue_time < 0:
            queue_time = 0
        return queue_time
