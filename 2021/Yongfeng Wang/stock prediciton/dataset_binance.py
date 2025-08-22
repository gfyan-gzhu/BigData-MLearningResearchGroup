import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import time
import json
from sklearn.model_selection import train_test_split
# link = 'https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1h'



# 获取数据
def get_data(csv_path):
    f = pd.read_csv(csv_path)
    df = pd.DataFrame(f)
    # df1 = df.drop(['opentime'], axis=1).astype(float)
    # df1['opentime'] = df['opentime']
    # df1 = df1.insert(0, df1.pop())
    # print(df1)
    df = df.drop(['turnover', 'turnoversum', 'activevolume', 'activeturnover'], axis=1)
    print(df)
    return df

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.label, self.column_name = self.get_data()
        self.data = self.data.astype('float')
        print(self.column_name)
        self.data_num = self.data.shape[0] # 数组行数
        # self.train_num = int(self.data_num * config.train_data_rate) # 训练数量
        self.train_num = int(self.data_num) # 训练数量

        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean) / self.std # 归一化 axis=0，计算每一列的均值

    # 划分训练集和测试集
    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.label[:self.train_num]

        train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step + 1)]
        train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step + 1)]
        train_x, train_y = np.array(train_x), np.array(train_y)

        # feature_train, feature_valid, label_train, label_valid
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)  # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    # 获取验证数据
    def get_test_data(self, return_label_data=True):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.label[:self.train_num]

        test_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        test_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]

        return np.array(test_x), np.array(test_y)
    # 获取数据
    def get_data(self):
        f = open(self.config.csv_file)
        f = f.read()
        data = json.loads(f)
        df = pd.DataFrame(data)
        # 开盘时间， 开盘价， 最高价， 最低价， 收盘价， 成交量， 收盘时间， 成交额， 成交笔数， 主动买入成交量， 主动买入成交额， 忽略
        df.columns = ['opentime', 'open', 'high', 'low', 'close', 'volume', 'closetime', 'turnover', 'turnoversum',
                      'activevolume', 'activeturnover', 'ignore']
        time_list = df['opentime'].tolist()
        time_list = [time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(int(str(x)[:10]))) for x in time_list]
        df['opentime'] = time_list
        df.drop(['opentime', 'ignore', 'closetime'], axis=1, inplace=True)
        # delete the last row
        df.drop(df.tail(1).index, inplace=True)
        # 计算涨跌
        df = df.iloc[::-1]
        prev_amplitude = 0
        for item in df.iterrows():
            data_row = item[0]
            data = item[1]
            df.loc[data_row, 'label'] = 1 if prev_amplitude > 0 else 0
            prev_amplitude = float(data['close']) - float(data['open'])
        df['label'] = df['label'].astype('int')
        df = df.iloc[::-1]
        print(df)
        data_label = df['label']
        df.drop(['label'], axis=1, inplace=True)
        return df.values, data_label, df.columns.tolist()