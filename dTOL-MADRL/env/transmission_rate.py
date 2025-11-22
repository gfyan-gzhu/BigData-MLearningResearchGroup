import math

"""
计算数据传输速率
"""


def calculate_data_rate(distance, power, channel_bandwidth):
    noise_square = 1e-10
    channel_gain = 128.1 + 37.6 * math.log10(distance)
    signal_to_noise_ratio = (power * channel_gain) / noise_square
    # 计算最大可用调制阶数
    max_modulation_order = math.log2(1 + signal_to_noise_ratio)
    # 计算最大数据速率（单位：bps）
    max_data_rate = channel_bandwidth * max_modulation_order
    # 将速率转换为 Mbps 并返回
    return max_data_rate


# P = 600  # mW
# B = 1.5
# d = 150
# rate = calculate_data_rate(d, P, B)
# print(rate)
