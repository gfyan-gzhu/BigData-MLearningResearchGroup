import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import time
import json
import seaborn as sns
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
# link = 'https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1h'

# 下载数据
def transform_data(file, index=0):
    f = open(file)
    f = f.read()
    data = json.loads(f)
    df = pd.DataFrame(data)
    # print(df)
    # 开盘时间， 开盘价， 最高价， 最低价， 收盘价， 成交量， 收盘时间， 成交额， 成交笔数， 主动买入成交量， 主动买入成交额， 忽略
    df.columns = ['opentime', 'open', 'high', 'low', 'close', 'volume', 'closetime', 'turnover', 'turnoversum', 'activevolume', 'activeturnover', 'ignore']
    time_list = df['opentime'].tolist()
    time_list = [time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(int(str(x)[:10]))) for x in time_list]
    # time_list = [datetime.strftime(df['opentime'], "%Y.%m.%d %H:%M:%S") for x in time_list]
    df['opentime'] = time_list
    df.drop(['closetime', 'turnover', 'turnoversum', 'activevolume', 'activeturnover', 'ignore'], axis=1, inplace=True)
    df = df.iloc[index:]
    init_money = 1000
    for item in df.iterrows():
        data_row = item[0]
        data = item[1]
        data_open = float(data.open)
        data_close = float(data.close)
        amplitude = (data_close - data_open) / data_open
        df.loc[data_row, 'amplitude'] = amplitude
        init_money = init_money * (1 + amplitude)
        df.loc[data_row, 'money'] = init_money
    return df

def plot_data(a, b):
    a['money'].plot(figsize=(10, 8), title='money')
    b['money'].plot(figsize=(10, 8), title='money')
    plt.show()
luna = transform_data('data/luna2.txt', 100)
lunc = transform_data('data/lunc.txt', 100)
plot_data(luna, lunc)

luna = transform_data('data/luna2.txt')
lunc = transform_data('data/lunc.txt')
plot_data(luna, lunc)

