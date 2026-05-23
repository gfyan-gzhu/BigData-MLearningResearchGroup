"""
系统默认参数
"""
# 模拟区域大小(m)
GROUND_LENGTH: float = 500
GROUND_WIDTH: float = 500

# 任务泊松到达率(tasks/s)
TASK_POISSON_RATE: float = 1

# 总模拟时间(s) 规定：1秒1个时隙，一个时隙1秒
TOTAL_TIME = 15
# 单位时隙长度(s),1个单位表示1秒，1/4表示0.25秒
UNITE_SLOT_LENGTH = 1
# 总时隙
TOTAL_TIME_SLOT:int = int(TOTAL_TIME / UNITE_SLOT_LENGTH)


# 带宽(Hz):20MHz（默认），25MHz，30MHz，35MHz，40MHz
BANDWIDTH = 2e7 # 2e7, 2.5e7, 3e7, 3.5e7, 4e7

# 噪声功率(w):-100dbm = 1e-13w
NOISE_POWER = 1e-13


ALPHA = 0.2 # 0.1
BETA = 0.8 # 2
"""
移动设备默认参数

"""
# 地面设备数量
DEVICE_NUM: int = 50
# 地面设备高度(m)
DEVICE_HIGH: int = 0

# 移动设备最大移动速度(m/s)：2m/s
DEVICE_MAX_SPEED: float = 2

# 移动设备最大停留时间(s): 1分钟 = 60秒 = 60步
DEVICE_MAX_RETENTION_TIME = 20

# MD发射功率(w):200mW=0.2w
DEVICE_TRANSMIT_POWER = 0.2

# MD带宽(Hz)
DEVICE_BANDWIDTH = BANDWIDTH / (DEVICE_NUM + 4)

# MD计算能力(cycle/s): 100MC/s
DEVICE_RESOURCE = 1e8

# MD单时隙可用能耗: 2J(焦耳)
DEVICE_ENERGY_BUDGET = 2

# MD处理1bit所要的CPU周期数:200 cycles/bit
DEVICE_1BIT_CYCLE = 2e2

# MD产生任务数量量(bit):0~2M = (0,2) * (8e6)：(1, 2)(默认), (2, 3), (3, 4), (4, 5), (5, 6)
DEVICE_TASK_UPPER_LIMIT = 2

DEVICE_TASK_LOWER_LIMIT = 1

# MD有效开关电容(F):10^-26F
DEVICE_SWITCHING_CAPACITANCE = 1e-26

# 正贡献度权重
W_POSITIVE = 0.7
# 符贡献度权重
W_NEGATIVE = 0.3

"""
无人机默认参数
"""
# 无人机初始位置(场景顶点)
UAV_START_COORDINATE = [[0, 0],
                        [0, GROUND_WIDTH],
                        [GROUND_LENGTH, 0],
                        [GROUND_LENGTH, GROUND_WIDTH]]

# 无人机最大移动速度(m/s)
UAV_MAX_SPEED: float = 30

# 无人机最大数量
UAV_MAX_NUM: int = 4

# 无人机最小高度(m)
UAV_MIN_HIGH: int = 20

# 无人机最大高度(m)
UAV_MAX_HIGH: int = 45

# UAV计算能力(cycle/s)：1、2(默认)、3、4、5GC/s
UAV_RESOURCE = 2e9

# UAV单时隙可用能耗: 5J(焦耳)
UAV_ENERGY_BUDGET = 5

# UAV带宽(Hz)
UAV_BANDWIDTH = BANDWIDTH / DEVICE_NUM

# UAV处理1bit所要的CPU周期数: 100 cycles/bit
UAV_1BIT_CYCLE = 1e2

# UAV的接收功率(w):100mW=0.1w
UAV_RECEIVED_POWER = 0.1

# UAV质量(kg)
UAV_QUALITY = 10

# 无人机有效开关电容(F):10^-28F
UAV_SWITCHING_CAPACITANCE = 1e-28



"""
强化学习参数
"""
# 总训练次数
TOTAL_EPISODES= 1000

# 上下层交替间隔：5、10、15、20、25
ALTERNATE_INTERCAL = 10
ALTERNATE_UPDATE_INTERVAL = 2

# 经验回放缓冲区容量
BUFFER_CAPACITY = int(1e5)

# 训练批次大小
BATCH_SIZE = int(128)

# 奖励折扣因子
GAMMA = 0.99

# -----------MADDPG------------------

# Actor网络学习率
DDPG_LR_ACTOR = 1e-4

# Critic网络学习率
DDPG_LR_CRITIC = 1e-3


# 目标网络软更新系数
DDPG_TAU = 0.005


# 训练时动作噪声探索标准差
DDPG_NOISE_SCALE = 1e-1

# 噪声衰减系数
DDPG_NOISE_DECAY = 0.9995

# 噪声最小值
DDPG_MIN_NOISE_SCALE = 0.05


# -----------AC------------------

# Actor网络学习率
AC_LR_ACTOR = 1e-4

# Critic网络学习率
AC_LR_CRITIC = 1e-3

# 贪心策略探索概率
AC_EPSILON = 0.4

# 探索概率衰减系数
AC_EPSOLON_DECAY = 0.999

# 探索概率最小值
AC_MIN_EPSILON = 0.1

# 目标卸载比例: 0.2 0.4 0.6 0.8 1.0
TARGET_OFFLOAD_RATIO = 0.6



#-----------PPO------------------
PPO_LR_ACTOR = 1e-4
PPO_LR_CRITIC = 1e-3
PPO_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_BATCH_SIZE = 64
PPO_UPDATE_EPOCHS = 10









