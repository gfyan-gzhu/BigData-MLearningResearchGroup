import pandas as pd
from torch.utils.data.dataset import Dataset
import time
import json

# 下载数据
# def transform_data():
#     f = open('data/eth.txt')
#     f = f.read()
#     data = json.loads(f)
#     df = pd.DataFrame(data)
#     # 开盘时间， 开盘价， 最高价， 最低价， 收盘价， 成交量， 收盘时间， 成交额， 成交笔数， 主动买入成交量， 主动买入成交额， 忽略
#     df.columns = ['opentime', 'open', 'high', 'low', 'close', 'volume', 'closetime', 'turnover', 'turnoversum', 'activevolume', 'activeturnover', 'ignore']
#     time_list = df['opentime'].tolist()
#     time_list = [time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(int(str(x)[:10]))) for x in time_list]
#     df['opentime'] = time_list
#     df.drop(['ignore', 'closetime'], axis=1, inplace=True)
#     print(df)
#     return df

# 处理数据
# def deal_data(df):
#     # 计算涨跌
#     df = df.iloc[::-1]
#     prev_amplitude = 0
#     for item in df.iterrows():
#         data_row = item[0]
#         data = item[1]
#         df.loc[data_row, 'label'] = 1 if prev_amplitude > 0 else 0
#         prev_amplitude = float(data['open']) - float(data['close'])
#     df['label'] = df['label'].astype('int')
#     df = df.iloc[::-1]
#     df.to_csv('./data/eth_deal.csv')
#     print(df)

# 将本地csv转化为数据集
class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.drop(['opentime'], axis=1, inplace=True)
        print(df.columns.tolist())
        self.data = df
        print(self.data)
    def __getitem__(self, i):
        open = self.data['open'][i]
        high = self.data['high'][i]
        low = self.data['low'][i]
        close = self.data['close'][i]
        volume = self.data['volume'][i]
        turnover = self.data['turnover'][i]
        turnoversum = self.data['turnoversum'][i]
        activevolume = self.data['activevolume'][i]
        activeturnover = self.data['activeturnover'][i]
        label = self.data['label'][i]
        return [open, high, low, close, volume, turnover, turnoversum, activevolume, activeturnover], label

    def __len__(self):
        return len(self.data.index)

