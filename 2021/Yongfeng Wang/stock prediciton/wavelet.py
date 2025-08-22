#模块调用
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sqrt
import pywt
from math import log

#sgn函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(list):
    data = list # 将np.ndarray()转为列表
    # print(len(data))
    w = pywt.Wavelet('coif3')#选择dB10小波基

    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # print("maximun level is " + str(maxlev))
    ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(data, w, level=maxlev)  # 3层小波分解
    # ca5 = ca5.squeeze(axis=0) #ndarray数组减维：(1，a)->(a,)
    
    length1 = len(cd1)
    length0 = len(data)
    # print("数据长度：" + str(length0))
    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = []
    usecoeffs.append(ca4)

    #软阈值方法

    for k in range(length1):
        if (abs(cd1[k]) >= lamda/np.log2(2)):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda/np.log2(2))
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda/np.log2(3)):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda/np.log2(3))
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda / np.log2(4)):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda / np.log2(4))
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda / np.log2(5)):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - lamda / np.log2(5))
        else:
            cd4[k] = 0.0

    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs

#提取数据
def denosing(data):
    data['open'] = wavelet_noising(data['open'].tolist())#调用小波阈值方法去噪
    data['high'] = wavelet_noising(data['high'].tolist())#调用小波阈值方法去噪
    data['low'] = wavelet_noising(data['low'].tolist())#调用小波阈值方法去噪
    data['close'] = wavelet_noising(data['close'].tolist())#调用小波阈值方法去噪
    data['preclose'] = wavelet_noising(data['preclose'].tolist())#调用小波阈值方法去噪
    # data['volume'] = wavelet_noising(data['volume'].tolist())#调用小波阈值方法去噪
    # data['amount'] = wavelet_noising(data['amount'].tolist())#调用小波阈值方法去噪
    # data['pctChg'] = wavelet_noising(data['pctChg'].tolist())#调用小波阈值方法去噪
    return data

# 信噪比性能
# origin_data = data['open'].tolist()
# data_up = np.sum(data_denoising ** 2)
# data_diff = np.sum((data_denoising - origin_data) ** 2)
# snr = 10 * np.log10(data_up / data_diff)
# mse = np.sum((origin_data - data_denoising) ** 2) / len(origin_data)
# rmse = sqrt(mse)
# print("信噪比SNR: " + str(snr))
# print("均方根误差RMSE: " + str(rmse))
