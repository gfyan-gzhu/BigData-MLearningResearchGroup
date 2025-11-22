import random

"""生成车辆的速度和坐标"""


class Vehicles:
    def __init__(self, vehicles_num, road_length, num_lanes,
                 lane_width, min_vehicle_speed, max_vehicle_speed):
        """
        :param vehicles_Num: 每个车道车辆数
        :param road_length: 道路长度
        :param num_lans: 车道数
        :param lane_width: 车道宽度
        :param min_vehicle_speed: 车辆最小速度
        :param max_vehicle_speed: 车辆最大速度
        """
        super(Vehicles, self).__init__()
        self.vehicles_num = vehicles_num
        self.road_length = road_length
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.min_vehicle_speed = min_vehicle_speed
        self.max_vehicle_speed = max_vehicle_speed
        self.vehicles = []

    def initialization_vehicle(self):
        # 以起始点为原点建坐标系
        for lane in range(self.num_lanes):
            position_y = self.lane_width * lane + self.lane_width / 2  # 车辆纵坐标
            for _ in range(self.vehicles_num):
                position_x = random.randint(0, self.road_length)  # 车辆横坐标
                speed = random.randint(self.min_vehicle_speed, self.max_vehicle_speed)
                self.vehicles.append([position_x, position_y, speed])
        return self.vehicles

    def vehicles_move(self, time_step):
        for i in range(len(self.vehicles)):
            position_x, position_y, speed = self.vehicles[i]
            position_x += speed * time_step
            if position_x > self.road_length:
                position_x = position_x - self.road_length

            # 处理车辆交汇和边界条件，例如超车、变道等
            # 更新车辆位置
            self.vehicles[i] = [position_x, position_y, speed]
        return self.vehicles


