from pymobility.models.mobility import random_waypoint
from simEnvParameter import *




rw = random_waypoint(
                    DEVICE_NUM,
                    dimensions=(GROUND_LENGTH, GROUND_WIDTH),
                    velocity=(1, DEVICE_MAX_SPEED),
                    wt_max=DEVICE_MAX_RETENTION_TIME)



# 为所有设备设置初始坐标
def get_coordinate(devices):
    try:
        all_coords = next(rw).tolist()  # 获取所有设备的初始坐标
        for i, device in enumerate(devices):
            device.coordinate = all_coords[i]  # 设置每个设备的初始坐标
            device.trajectory[0] = device.coordinate
        print(f"坐标初始化成功!")
        return True
    except Exception as e:
        print(f"Error in get_coordinate: {e}")
        return False

# 更新所有设备的下一步坐标
def change_coordinate(devices, time_slot):
    try:
        all_coords = next(rw).tolist()  # 获取所有设备的下一步坐标
        for i, device in enumerate(devices):
            device.coordinate = all_coords[i]  # 更新每个设备的坐标
            device.trajectory[time_slot] = device.coordinate
        return True
    except Exception as e:
        print(f"Error in change_coordinate: {e}")
        return False







