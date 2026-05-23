# train_models.py - 训练所有设备数量的模型
import os
import copy
import numpy as np
from simEnv import device, uav
from coordinateManage import get_coordinate
from ddpg_slover import initialize_ddpg_agent, run_ddpg_training
from RATddpg import initialize_rat_agent, run_rat_training
import simEnvParameter


def train_models_for_device_num(target_device_num, num_episodes=1000):
    """
    为指定设备数量训练模型
    """
    print(f"\n{'=' * 60}")
    print(f"开始训练设备数量为 {target_device_num} 的模型")
    print(f"{'=' * 60}")

    # 设置设备数量
    simEnvParameter.DEVICE_NUM = target_device_num
    simEnvParameter.DEVICE_BANDWIDTH = simEnvParameter.BANDWIDTH / (target_device_num + 1)
    simEnvParameter.UAV_BANDWIDTH = simEnvParameter.BANDWIDTH / (target_device_num + 1)

    # 设置随机种子
    np.random.seed(77)

    # 初始化设备
    devices = [device(i) for i in range(target_device_num)]
    get_coordinate(devices)

    # 初始化无人机
    uav_instance = uav()
    uav_instance.fun_random_point_on_circle()

    # ====================== DDPG 模型训练 ======================
    print(f"开始训练DDPG模型 (设备数量: {target_device_num})...")

    try:
        ddpg_agent, training_rewards, completion_rates, offload_ratios = run_ddpg_training(
            devices, uav_instance, num_episodes=num_episodes, total_time_slots=simEnvParameter.TOTAL_TIME_SLOT
        )
        print(f"DDPG模型训练完成! 训练平均奖励: {np.mean(training_rewards):.4f}")
    except Exception as e:
        print(f"DDPG模型训练失败 (设备数量: {target_device_num}): {e}")

    # ====================== RATddpg 模型训练 ======================
    print(f"\n开始训练RATddpg模型 (设备数量: {target_device_num})...")

    try:
        rat_agent, training_rewards, completion_rates = run_rat_training(
            devices, uav_instance, num_episodes=num_episodes, total_time_slots=simEnvParameter.TOTAL_TIME_SLOT
        )
        print(f"RATddpg模型训练完成! 训练平均奖励: {np.mean(training_rewards):.4f}")
    except Exception as e:
        print(f"RATddpg模型训练失败 (设备数量: {target_device_num}): {e}")

    print(f"\n设备数量 {target_device_num} 的模型训练完成")


def main():
    """
    主函数：训练所有设备数量的模型
    """
    device_nums = [20, 30, 40, 50, 60]

    print("开始批量训练不同设备数量的模型")
    print(f"设备数量列表: {device_nums}")

    # 训练轮数
    training_episodes = 2000

    for device_num in device_nums:
        train_models_for_device_num(device_num, num_episodes=training_episodes)

    print("\n所有设备数量的模型训练已完成！")
    print("生成的模型文件:")
    for device_num in device_nums:
        print(f"  - ddpg_best_model_{device_num}.pth")
        print(f"  - rat_best_model_{device_num}.pth")


if __name__ == '__main__':
    main()