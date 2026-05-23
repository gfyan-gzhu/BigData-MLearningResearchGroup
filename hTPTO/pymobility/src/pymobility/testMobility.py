import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mobility import random_waypoint

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mobility import random_waypoint


def test1():
    # 模拟参数
    width, height = 100, 100  # 区域大小为 100x100
    velocity_range = (10, 25)  # 速度范围为 10 至 25
    wt_max = 10   # 停留时间最大为 5 分钟 (以秒为单位)
    total_time = 60  # 模拟 60 秒
    num_nodes = 3  # 模拟 1 个节点

    # 创建 RandomWaypoint 模型，每个节点都有独立的运动模型
    rws = [random_waypoint(1, dimensions=(width, height), velocity=velocity_range, wt_max=wt_max) for _ in range(num_nodes)]

    # 记录所有节点的移动轨迹
    position_lists = [[] for _ in range(num_nodes)]  # 为每个节点创建一个空的列表
    time_changes = [[] for _ in range(num_nodes)]  # 用于记录每个节点的位置变化时间点

    # 初始化上一个位置
    last_positions = [None] * num_nodes  # 每个节点的上一个位置

    # 运行模型直到模拟时间结束
    for i in tqdm(range(int(total_time))):  # 每次更新1秒
        for node_idx in range(num_nodes):
            position = next(rws[node_idx]).tolist()  # 获取当前位置
            position = position[0]  # 去除额外的嵌套结构，只保留[x, y]

            print("positon = ", position)

            # 如果位置发生变化，记录时间
            if last_positions[node_idx] is None or np.any(position != last_positions[node_idx]):
                time_changes[node_idx].append(i)
                last_positions[node_idx] = position

            position_lists[node_idx].append(position)  # 记录位置

    print("position_lists[0] = ", position_lists[0])

    # 可视化所有节点的运动轨迹
    plt.figure(figsize=(6, 6))

    for node_idx in range(num_nodes):
        # 获取当前节点的所有坐标
        x_positions = [pos[0] for pos in position_lists[node_idx]]  # pos[0] 为 x 坐标
        y_positions = [pos[1] for pos in position_lists[node_idx]]  # pos[1] 为 y 坐标

        # 绘制节点的轨迹
        plt.plot(x_positions, y_positions, marker='o', markersize=3, label=f"Node {node_idx + 1} Path")

        # 标注起始位置（红色）
        plt.scatter(x_positions[0], y_positions[0], color='r', zorder=5)
        plt.text(x_positions[0], y_positions[0], f'Start Node {node_idx + 1}', fontsize=12, ha='right', color='r')

        # 标注结束位置（黑色）
        plt.scatter(x_positions[-1], y_positions[-1], color='k', zorder=5)
        plt.text(x_positions[-1], y_positions[-1], f'End Node {node_idx + 1}', fontsize=12, ha='left', color='k')

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.title("Random Waypoint Mobility Simulation")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.show()





def test2():
    # 导入所需模块
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import time
    from tqdm import tqdm
    from mobility import random_waypoint
    import numpy as np
    import pandas as pd

    # 模型参数设置
    nr_nodes = 1
    dimensions = (100, 100)
    total_time = 30
    time_slots = total_time * 4

    # 初始化移动模型-路点模型
    model = random_waypoint(nr_nodes, dimensions, velocity=(0.1, 10.0), wt_max=2)

    # 初始化节点坐标数组
    positions = np.zeros((nr_nodes, time_slots, 2))


    # 节点颜色设置
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#F08080']

    # matplotlib可视化初始化
    fig, ax = plt.subplots()
    ax.set_xlim(0, dimensions[0])
    ax.set_ylim(0, dimensions[1])

    # 循环模拟各个时隙
    for t in tqdm(range(time_slots)):

        # 更新节点坐标
        pos = next(model)
        for i in range(nr_nodes):
            positions[i, t] = pos[i]

        # 绘制当前时隙节点
        ax.clear()
        for i in range(nr_nodes):
            ax.plot(positions[i, :t + 1, 0], positions[i, :t + 1, 1], colors[i % len(colors)])

        # 设置标题
        ax.set_title('Time = {}s'.format(t * 5))

        # 更新画布
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)
    #end of time_slot

    # 生成节点颜色图例
    labels = []
    for i in range(nr_nodes):
        color = colors[i % len(colors)]
        label = mpatches.Patch(color=color, label='Node {}'.format(i + 1))
        labels.append(label)

    ax.legend(handles=labels)

    # 保存结果到Excel
    df = pd.DataFrame()
    for t in range(time_slots):
        df['t' + str(t)] = range(1, nr_nodes + 1)
        for i in range(nr_nodes):
            df.at[i, 'x' + str(t)] = positions[i, t, 0]
            df.at[i, 'y' + str(t)] = positions[i, t, 1]

    df.to_csv('nodes_100sec.csv', index=False)

    print("结果已保存")
    plt.show()



if __name__ == '__main__':
    # test2()
    test1()