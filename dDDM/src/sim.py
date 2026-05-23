# 实验对比
import simEnvParameter
from uavTrajectoryDecision import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ddpg_slover import *
from RATddpg import *
from OffloadingResourceDecision import *
from draw import *

# 设置随机种子，获得可复现结果
np.random.seed(77)

if __name__ == '__main__':

    # 初始化设备
    devices: List[device] = [device(i) for i in range(DEVICE_NUM)]
    # 初始化坐标位置
    get_coordinate(devices)

    devices_KMPSO_IBKA = copy.deepcopy(devices)
    devices_KMPSO_BKA = copy.deepcopy(devices)
    devices_KMPSO_random = copy.deepcopy(devices)
    devices_HOLD = copy.deepcopy(devices)

    devices_PSO_IBKA = copy.deepcopy(devices)
    devices_random_IBKA = copy.deepcopy(devices)

    devices_MATS = copy.deepcopy(devices)

    print("初始化无人机！")
    # 初始化无人机
    uav = uav()

    # 初始化经验池
    experience_pool = []

    # 初始化无人机位置
    uav.fun_random_point_on_circle()
    print(f"uav.coordinate:{uav.coordinate}")

    uav_KMPSO_BKA = copy.deepcopy(uav)
    uav_KMPSO_IBKA = copy.deepcopy(uav)
    uav_KMPSO_random = copy.deepcopy(uav)
    uav_HOLD = copy.deepcopy(uav)

    uav_PSO_IBKA = copy.deepcopy(uav)
    uav_random_IBKA = copy.deepcopy(uav)

    uav_MATS = copy.deepcopy(uav)

    # 迭代记录对象
    iterative_recording_PSO = {}
    iterative_recording_KMPSO = {}

    iterative_recording_BKA = {}
    iterative_recording_IBKA = {}

    distance_recording = {}



    # ====================== DDPG 实验运行 ======================
    print("开始运行DDPG算法实验...")

    # 根据当前设备数量选择模型
    current_device_num = simEnvParameter.DEVICE_NUM
    ddpg_model_filename = f'ddpg_best_model_{current_device_num}.pth'

    # 检查模型文件是否存在
    if not os.path.exists(ddpg_model_filename):
        raise FileNotFoundError(f"DDPG模型文件 {ddpg_model_filename} 不存在！请先运行 train_models.py 训练模型。")

    # 初始化并加载对应设备数量的最佳模型
    ddpg_agent = initialize_ddpg_agent(device_num=current_device_num)
    print(f"正在加载DDPG模型: {ddpg_model_filename}")
    ddpg_agent.load(ddpg_model_filename, device_num=current_device_num)
    print(f"成功加载DDPG模型: {ddpg_model_filename}")

    # 运行实验
    devices_DDPG, uav_DDPG, experiment_rewards, experiment_data, avg_delay = run_ddpg_experiment(
        devices, uav, ddpg_agent, total_time_slots=TOTAL_TIME_SLOT
    )

    # ====================== RATddpg 实验运行 ======================
    print("\n\n开始运行RATddpg算法实验...")

    # 根据当前设备数量选择模型
    rat_model_filename = f'rat_best_model_{current_device_num}.pth'

    # 检查模型文件是否存在
    if not os.path.exists(rat_model_filename):
        raise FileNotFoundError(f"RATddpg模型文件 {rat_model_filename} 不存在！请先运行 train_models.py 训练模型。")

    # 初始化并加载对应设备数量的最佳模型
    rat_agent = initialize_rat_agent(device_num=current_device_num)
    print(f"正在加载RATddpg模型: {rat_model_filename}")
    rat_agent.load(rat_model_filename)
    print(f"成功加载RATddpg模型: {rat_model_filename}")

    # 运行RATddpg实验
    devices_RAT, uav_RAT, rat_experiment_rewards, rat_experiment_data, avg_delay = run_rat_experiment(
        devices, uav, rat_agent, total_time_slots=TOTAL_TIME_SLOT
    )

    # ====================== 非强化学习方法的实验运行 ======================
    # 时隙遍历
    for i in range(1,TOTAL_TIME_SLOT+1):
        # 更新地面设备坐标
        change_coordinate(devices, i)

        # 深拷贝坐标信息到对应的设备
        for j, device in enumerate(devices):
            devices_KMPSO_BKA[j].coordinate = copy.deepcopy(device.coordinate)
            devices_KMPSO_BKA[j].trajectory[i] = devices_KMPSO_BKA[j].coordinate

            devices_KMPSO_IBKA[j].coordinate = copy.deepcopy(device.coordinate)
            devices_KMPSO_IBKA[j].trajectory[i] = devices_KMPSO_IBKA[j].coordinate

            devices_KMPSO_random[j].coordinate = copy.deepcopy(device.coordinate)
            devices_KMPSO_random[j].trajectory[i] = devices_KMPSO_random[j].coordinate

            devices_HOLD[j].coordinate = copy.deepcopy(device.coordinate)
            devices_HOLD[j].trajectory[i] = devices_HOLD[j].coordinate

            devices_PSO_IBKA[j].coordinate = copy.deepcopy(device.coordinate)
            devices_PSO_IBKA[j].trajectory[i] = devices_PSO_IBKA[j].coordinate

            devices_random_IBKA[j].coordinate = copy.deepcopy(device.coordinate)
            devices_random_IBKA[j].trajectory[i] = devices_random_IBKA[j].coordinate



            devices_MATS[j].coordinate = copy.deepcopy(device.coordinate)
            devices_MATS[j].trajectory[i] = devices_MATS[j].coordinate


        # 任务列表，用于保存各时隙产生的任务
        tasks = []
        for j, device in enumerate(devices):
            if device.fun_generate_task(i):
                tasks.append(device.task[i])

                devices_KMPSO_BKA[j].task[i] = copy.deepcopy(device.task[i])
                devices_KMPSO_IBKA[j].task[i] = copy.deepcopy(device.task[i])
                devices_KMPSO_random[j].task[i] = copy.deepcopy(device.task[i])
                devices_HOLD[j].task[i] = copy.deepcopy(device.task[i])

                devices_PSO_IBKA[j].task[i] = copy.deepcopy(device.task[i])
                devices_random_IBKA[j].task[i] = copy.deepcopy(device.task[i])

                devices_MATS[j].task[i] = copy.deepcopy(device.task[i])

        tasks_KMPSO_BKA = [devices_KMPSO_BKA[j].task[i] for j in range(len(devices_KMPSO_BKA))
                           if i in devices_KMPSO_BKA[j].task]
        tasks_KMPSO_IBKA = [devices_KMPSO_IBKA[j].task[i] for j in range(len(devices_KMPSO_IBKA))
                            if i in devices_KMPSO_IBKA[j].task]
        tasks_KMPSO_random = [devices_KMPSO_random[j].task[i] for j in range(len(devices_KMPSO_random))
                              if i in devices_KMPSO_random[j].task]
        tasks_HOLD = [devices_HOLD[j].task[i] for j in range(len(devices_HOLD))
                              if i in devices_HOLD[j].task]

        tasks_PSO_IBKA = [devices_PSO_IBKA[j].task[i] for j in range(len(devices_PSO_IBKA))
                          if i in devices_PSO_IBKA[j].task]
        tasks_random_IBKA = [devices_random_IBKA[j].task[i] for j in range(len(devices_random_IBKA))
                          if i in devices_random_IBKA[j].task]

        tasks_MATS = [devices_MATS[j].task[i] for j in range(len(devices_MATS))
                      if i in devices_MATS[j].task]
        print(f"**********************************************************************************\n"
              f"时隙{i},共产生{len(tasks)}个任务!")


        if len(tasks) > 0:

            # UAV轨迹决策
            uav_KMPSO_IBKA.coordinate = kmpso_trajectory_optimization(tasks_KMPSO_IBKA,
                                                            devices_KMPSO_IBKA,
                                                            uav_KMPSO_IBKA,
                                                            i,
                                                            iterative_recording_KMPSO)
            uav_KMPSO_IBKA.trajectory[i] = uav_KMPSO_IBKA.coordinate

            # 同轨迹优化策略，将轨迹和飞行能耗添加到对应无人机
            uav_KMPSO_BKA.coordinate = uav_KMPSO_IBKA.coordinate
            uav_KMPSO_BKA.trajectory[i] = copy.deepcopy(uav_KMPSO_IBKA.trajectory.get(i))
            uav_KMPSO_BKA.flight_energy[i] = copy.deepcopy(uav_KMPSO_IBKA.flight_energy.get(i))

            uav_KMPSO_random.coordinate = uav_KMPSO_IBKA.coordinate
            uav_KMPSO_random.trajectory[i] = uav_KMPSO_random.coordinate
            uav_KMPSO_random.flight_energy[i] = copy.deepcopy(uav_KMPSO_IBKA.flight_energy.get(i))



            uav_MATS.coordinate = uav_KMPSO_IBKA.coordinate
            uav_MATS.trajectory[i] = uav_MATS.coordinate
            uav_MATS.flight_energy[i] = copy.deepcopy(uav_KMPSO_IBKA.flight_energy.get(i))


            # 不同轨迹优化策略
            uav_PSO_IBKA.coordinate = pso_trajectory_optimization(tasks_PSO_IBKA,
                                                                devices_PSO_IBKA,
                                                                uav_PSO_IBKA,
                                                                i,
                                                                iterative_recording_PSO)
            uav_PSO_IBKA.trajectory[i] = uav_PSO_IBKA.coordinate

            coordinate_HOLD, decision_HOLD = UAV_HOLD_trajectory_optimization(tasks_HOLD,
                                                                          devices_HOLD,
                                                                          uav_HOLD,
                                                                          i)
            uav_HOLD.coordinate = coordinate_HOLD
            uav_HOLD.trajectory[i] = uav_HOLD.coordinate


            uav_random_IBKA.coordinate = random_trajectory(uav_random_IBKA,i)
            uav_random_IBKA.trajectory[i] = uav_random_IBKA.coordinate
        else:
            continue

        enqueue_dequeue(tasks_HOLD,
                        decision_HOLD,
                        devices_HOLD,
                        uav_HOLD,
                        i)

        # 卸载和任务分配决策
        decision_KMPSO_BKA = BKA_offloading_resource_decision(tasks_KMPSO_BKA,
                                     devices_KMPSO_BKA,
                                     uav_KMPSO_BKA,
                                     i,
                                    iterative_recording_BKA)


        # 遍历任务，将决策值记录，并入队
        enqueue_dequeue(tasks_KMPSO_BKA,
                        decision_KMPSO_BKA,
                        devices_KMPSO_BKA,
                        uav_KMPSO_BKA,
                        i)

        print(f"_______________________________________________________________________\n"
              f"IBKA——时隙{i},共产生{len(tasks)}个任务!")
        decision_KMPSO_IBKA = IBKA_offloading_resource_decision(tasks_KMPSO_IBKA,
                                                            devices_KMPSO_IBKA,
                                                            uav_KMPSO_IBKA,
                                                            i,
                                                            experience_pool,
                                                            iterative_recording_IBKA)

        # 遍历任务，将决策值记录，并入队,出队
        enqueue_dequeue(tasks_KMPSO_IBKA,
                        decision_KMPSO_IBKA,
                        devices_KMPSO_IBKA,
                        uav_KMPSO_IBKA,
                        i)

        print(f"=======================================================================\n"
              f"random——时隙{i},共产生{len(tasks)}个任务!")
        decision_KMPSO_random = random_offloading_resource(tasks_KMPSO_random,
                                                     devices_KMPSO_random,
                                                     uav_KMPSO_random,
                                                   )
        enqueue_dequeue(tasks_KMPSO_random,
                        decision_KMPSO_random,
                        devices_KMPSO_random,
                        uav_KMPSO_random,
                        i)

        print(f"=======================================================================\n"
              f"MATS—时隙{i},共产生{len(tasks)}个任务!")
        decision_MATS = MATS_offloading_resource_decision(tasks_MATS,
                                                         devices_MATS,
                                                         uav_MATS)

        enqueue_dequeue(tasks_MATS,
                        decision_MATS,
                        devices_MATS,
                        uav_MATS,
                        i)

        # 同卸载和资源分配，不同轨迹策略，将decision_KMPSO_IBKA决策值作为PSO、random的决策值
        enqueue_dequeue(tasks_PSO_IBKA,
                        decision_KMPSO_IBKA,
                        devices_PSO_IBKA,
                        uav_PSO_IBKA,
                        i)

        enqueue_dequeue(tasks_random_IBKA,
                        decision_KMPSO_IBKA,
                        devices_random_IBKA,
                        uav_random_IBKA,
                        i
                        )

    # 时隙结束
    # print(f"iterative_recording_PSO = {iterative_recording_PSO}\n"
    #       f"iterative_recording_KMPSO = {iterative_recording_KMPSO}\n")
    # 累加无人机各时隙能耗
    uav_energy_record(uav_KMPSO_BKA, devices_KMPSO_BKA)
    uav_energy_record(uav_KMPSO_IBKA, devices_KMPSO_IBKA)
    uav_energy_record(uav_KMPSO_random, devices_KMPSO_random)
    uav_energy_record(uav_HOLD, devices_HOLD)

    uav_energy_record(uav_random_IBKA, devices_random_IBKA)
    uav_energy_record(uav_PSO_IBKA, devices_PSO_IBKA)



    # 绘图：PSO|KMPSO、BKA|IBKA迭代比较适应度值
    plot_average_convergence(iterative_recording_PSO,
                             iterative_recording_KMPSO,
                             "PSO",
                             "KMPSO")
    plot_average_convergence(iterative_recording_BKA,
                             iterative_recording_IBKA,
                             "BKA",
                             "IBKA")

    """遍历所有设备的所有任务，保存信息到excel中"""

    # 不同带宽下，random、PSO、KMPSO、HOLD、DDPG、RATddpg的传输时延比较
    # save_task_delay_data(devices_PSO_IBKA,"devices_PSO BANDWIDTH=40MHz task info.xlsx")
    # save_task_delay_data(devices_KMPSO_IBKA, "devices_KMPSO BANDWIDTH=40MHz task info.xlsx")
    # save_task_delay_data(devices_random_IBKA, "devices_random BANDWIDTH=40MHz task info.xlsx")
    # save_task_delay_data(devices_HOLD, "devices_HOLD BANDWIDTH=40MHz task info.xlsx")
    # save_task_delay_data(devices_DDPG, "devices_DDPG BANDWIDTH=40MHz task info.xlsx")
    # save_task_delay_data(devices_RAT, "devices_RAT BANDWIDTH=40MHz task info.xlsx")

    # 不同无人机飞行速度下，random、PSO、KMPSO、HOLD、DDPG、RATddpg的飞行能耗比较
    # save_uav_flight_energy_data(uav_random_IBKA, "uav_random speed=40m flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_PSO_IBKA, "uav_PSO speed=40m flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_KMPSO_IBKA, "uav_KMPSO speed=40m flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_HOLD, "uav_HOLD speed=40m flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_DDPG, "uav_DDPG speed=40m flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_RAT, "uav_RAT speed=40m flight energy record.xlsx")

    # 不同设备数量下，random、BKA、IBKA、HOLD、MATS、DDPG、RATddpg的比较
    # save_UAV_energy_and_task_delay_data(devices_KMPSO_random, "devices_random DEVICES=50 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_KMPSO_BKA, "devices_BKA DEVICES=50 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_KMPSO_IBKA, "devices_IBKA DEVICES=50 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_HOLD, "devices_HOLD DEVICES=50 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_MATS, "devices_MATS DEVICES=50 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_DDPG, "devices_DDPG DEVICES=50 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_RAT, "devices_RAT DEVICES=50 task info.xlsx")

    # 不同ALPHA参数下，random、BKA、IBKA、HOLD、DDPG、RATddpg的时延比较
    # save_task_delay_data(devices_KMPSO_random, "devices_random ALPHA=0.2 BETA=0.8 task info.xlsx")
    # save_task_delay_data(devices_KMPSO_BKA, "devices_BKA ALPHA=0.2 BETA=0.8 task info.xlsx")
    # save_task_delay_data(devices_KMPSO_IBKA, "devices_IBKA ALPHA=0.2 BETA=0.8 task info.xlsx")
    # save_task_delay_data(devices_HOLD, "devices_HOLD ALPHA=0.2 BETA=0.8 task info.xlsx")
    # save_task_delay_data(devices_MATS, "devices_MATS ALPHA=0.2 BETA=0.8 task info.xlsx")
    # save_task_delay_data(devices_DDPG, "devices_DDPG ALPHA=0.2 BETA=0.8 task info.xlsx")
    # save_task_delay_data(devices_RAT, "devices_RAT ALPHA=0.2 BETA=0.8 task info.xlsx")

    # 不同BETA参数下，random、BKA、IBKA、HOLD、DDPG、RATddpg的无人机计算能耗比较
    # save_UAV_energy_data(devices_KMPSO_random, "devices_random ALPHA=0.2 BETA=1.0 task info.xlsx")
    # save_UAV_energy_data(devices_KMPSO_BKA, "devices_BKA ALPHA=0.2 BETA=1.0 task info.xlsx")
    # save_UAV_energy_data(devices_KMPSO_IBKA, "devices_IBKA ALPHA=0.2 BETA=1.0 task info.xlsx")
    # save_UAV_energy_data(devices_HOLD, "devices_HOLD ALPHA=0.2 BETA=1.0 task info.xlsx")
    # save_UAV_energy_data(devices_MATS, "devices_MATS ALPHA=0.2 BETA=1.0 task info.xlsx")
    # save_UAV_energy_data(devices_DDPG, "devices_DDPG ALPHA=0.2 BETA=1.0 task info.xlsx")
    # save_UAV_energy_data(devices_RAT, "devices_RAT ALPHA=0.2 BETA=1.0 task info.xlsx")

    # 不同飞行惩罚eta参数下，random、PSO、KMPSO、HOLD、DDPG、RATddpg的飞行能耗比较
    # save_uav_flight_energy_data(uav_random_IBKA, "uav_random eta=0.0005 flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_PSO_IBKA, "uav_PSO eta=0.0005 flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_KMPSO_IBKA, "uav_KMPSO eta=0.0005 flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_HOLD, "uav_HOLD eta=0.0005 flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_DDPG, "uav_DDPG eta=0.0005 flight energy record.xlsx")
    # save_uav_flight_energy_data(uav_RAT, "uav_RAT eta=0.0005 flight energy record.xlsx")

    # 不同任务到达率lambda参数下，random、BKA、IBKA、HOLD、MATS、DDPG、RATddpg的比较
    # save_UAV_energy_and_task_delay_data(devices_KMPSO_random, "devices_random lambda_i=1.0 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_KMPSO_BKA, "devices_BKA lambda_i=1.0 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_KMPSO_IBKA, "devices_IBKA lambda_i=1.0 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_HOLD, "devices_HOLD lambda_i=1.0 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_MATS, "devices_MATS lambda_i=1.0 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_DDPG, "devices_DDPG lambda_i=1.0 task info.xlsx")
    # save_UAV_energy_and_task_delay_data(devices_RAT, "devices_RAT lambda_i=1.0 task info.xlsx")

    # 初始位置画图
    # plot_initial_positions(devices, uav)

    # 保存选定时隙的任务分布和无人机部署位置
    # save_selected_time_slots_data(
    #     devices,  # 原始设备列表
    #     [
    #         uav_random_IBKA,
    #         uav_PSO_IBKA,
    #         uav_HOLD,
    #         uav_DDPG,
    #         uav_RAT,
    #         uav_KMPSO_IBKA
    #     ],  # 无人机对象列表
    #     [
    #     "random",
    #      "PSO",
    #      "HOLD",
    #      "DDPG",
    #      "RAT",
    #      "dDDM"
    #     ],  # 无人机名称列表
    #     [5,
    #      10,
    #      20,
    #      30
    #      ],  # 要保存的时隙列表
    #     filename_prefix="task distribution and uav deployment"  # 文件名前缀
    # )

    # 时隙间移动比较——热点图

    # plot_trajectory_with_kde(devices,
    #                 uav_random_IBKA,
    #                 uav_PSO_IBKA,
    #                 uav_HOLD,
    #                 uav_DDPG,
    #                 uav_RAT,
    #                 uav_KMPSO_IBKA,
    #                 5,
    #                 "random",
    #                 "PSO",
    #                 "HOLD",
    #                 "DDPG",
    #                 "RAT",
    #                 "dDDM")
    # plot_trajectory_with_kde(devices,
    #                          uav_random_IBKA,
    #                          uav_PSO_IBKA,
    #                          uav_HOLD,
    #                          uav_DDPG,
    #                          uav_RAT,
    #                          uav_KMPSO_IBKA,
    #                          10,
    #                          "random",
    #                          "PSO",
    #                          "HOLD",
    #                          "DDPG",
    #                          "RAT",
    #                          "dDDM")
    # plot_trajectory_with_kde(devices,
    #                          uav_random_IBKA,
    #                          uav_PSO_IBKA,
    #                          uav_HOLD,
    #                          uav_DDPG,
    #                          uav_RAT,
    #                          uav_KMPSO_IBKA,
    #                          20,
    #                          "random",
    #                          "PSO",
    #                          "HOLD",
    #                          "DDPG",
    #                          "RAT",
    #                          "dDDM")
    # plot_trajectory_with_kde(devices,
    #                          uav_random_IBKA,
    #                          uav_PSO_IBKA,
    #                          uav_HOLD,
    #                          uav_DDPG,
    #                          uav_RAT,
    #                          uav_KMPSO_IBKA,
    #                          30,
    #                          "random",
    #                          "PSO",
    #                          "HOLD",
    #                          "DDPG",
    #                          "RAT",
    #                          "dDDM")

    # 所有绘图函数调用完成后，阻塞程序直到所有图像窗口被关闭
    plt.show(block=True)
