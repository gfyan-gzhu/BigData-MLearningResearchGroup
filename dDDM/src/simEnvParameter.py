"""
系统默认参数
"""
# 模拟区域大小(m)
GROUND_LENGTH: float = 500
GROUND_WIDTH: float = 500

# 任务泊松到达率(tasks/s):0.2,0.4,0.6,0.8,1.0(默认值)
TASK_POISSON_RATE: float = 1.0

# 总模拟时间(s) 规定：1秒1个时隙，一个时隙1秒
TOTAL_TIME = 30
# 单位时隙长度(s),1个单位表示1秒，1/4表示0.25秒
UNITE_SLOT_LENGTH = 1
# 总时隙
TOTAL_TIME_SLOT:int = int(TOTAL_TIME / UNITE_SLOT_LENGTH)

# 1m处信道功率增益(w): -50db = 1e-5w
CHANNEL_POWER_GAIN: float = 1e-5

# 带宽(Hz):20MHz（默认），25MHz，30MHz，35MHz，40MHz
BANDWIDTH = 2e7 # 2e7, 2.5e7, 3e7, 3.5e7, 4e7

# 噪声功率(w):-100dbm = 1e-13w
NOISE_POWER = 1e-13

# 有效开关电容(F):10^-28F
SWITCHING_CAPACITANCE = 1e-28

# 时延加权系数：0.2（默认）、0.4、0.6、0.8、1.0
ALPHA = 0.2
# 能耗加权系数：0.2、0.4、0.6、0.8（默认）、1.0
BETA = 0.8
"""
移动设备默认参数

"""
# 地面设备数量 20、30、40、50(默认)、60
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
DEVICE_BANDWIDTH = BANDWIDTH / (DEVICE_NUM + 1)

# MD计算资源总量(cycle/s): 200MC/s
DEVICE_RESOURCE = 2e8

# MD处理1bit所要的CPU周期数:200 cycles/bit
DEVICE_1BIT_CYCLE = 2e2

# MD产生任务数量量(bit):1~2 Mbit = (0,2) * (1e6)
DEVICE_TASK_UPPER_LIMIT = 2

"""
无人机默认参数
"""
# 无人机初始位置
UAV_START_COORDINATE = [0, 0]

# 无人机最大移动速度(m/s) 20, 25, 30（默认）, 35, 40
UAV_MAX_SPEED: float = 30

# 无人机高度(m)
UAV_HIGH: int = 45

# UAV计算资源总量(cycle/s)：3GC/s
UAV_MAX_RESOURCE = 3e9

# UAV带宽(Hz)
UAV_BANDWIDTH = BANDWIDTH / (DEVICE_NUM + 1)

# UAV处理1bit所要的CPU周期数: 100 cycles/bit
UAV_1BIT_CYCLE = 1e2

# UAV的接收功率(w):100mW=0.1w
UAV_RECEIVED_POWER = 0.1

# UAV质量(kg)
UAV_QUALITY = 10

# UAV飞行能耗惩罚系数：0.0001, 0.0003, 0.0005(默认值), 0.0007, 0.0009
UAV_PENALTY_FLIGHT = 0.0005

