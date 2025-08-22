import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import time
import json
import pandas as pd
from wavelet import denosing
from sklearn.model_selection import train_test_split
# link = 'https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1h'

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.label, self.column_name, self.denoing_data = self.get_data()
        # 降噪
        # print(self.denoing_data)
        self.data = self.data.astype('float')
        print(self.column_name)
        self.data_num = self.data.shape[0] # 数组行数
        # self.train_num = int(self.data_num * config.train_data_rate) # 训练数量
        self.train_num = int(self.data_num) # 训练数量
        print(self.data_num)
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean) / self.std # 归一化 axis=0，计算每一列的均值
        norm_data_df = pd.DataFrame(self.norm_data, columns=self.column_name)
        norm_data_df.to_csv("./norm_data1.csv", encoding="utf_8_sig", index=False)
        # 降噪数据
        self.de_mean = np.mean(self.denoing_data, axis=0)
        self.de_std = np.std(self.denoing_data, axis=0)
        self.de_norm_data = (self.denoing_data - self.de_mean) / self.de_std # 归一化 axis=0，计算每一列的均值
        print(self.de_norm_data)
        de_norm_data_df = pd.DataFrame(self.de_norm_data, columns=self.column_name)
        de_norm_data_df.to_csv("./de_norm_data1.csv", encoding="utf_8_sig", index=False)

    # 划分训练集和测试集
    def get_train_and_valid_data(self):
        feature_data = self.data[:]
        if self.config.model == 'model.alstmnet' and self.config.alstmnet['wavelet']:
            feature_data = self.de_norm_data[:]  # 经过离散小波处理的数据集
        label_data = self.label[:]

        train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step + 1)]
        train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step + 1)]
        train_x, train_y = np.array(train_x), np.array(train_y)
        # feature_train, feature_valid, label_train, label_valid
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)  # 划分训练和验证集，并打乱
        print(train_x.shape)
        # print(train_x.size())
        return train_x, valid_x, train_y, valid_y

    # 获取验证数据
    def get_test_data(self, return_label_data=True):
        train_num = int(self.config.train_data_rate * self.data_num) - 3
        test_num = self.data_num - train_num
        feature_data = self.norm_data[train_num:]
        label_data = self.label[train_num:]

        test_x = [feature_data[i:i + self.config.time_step] for i in range(test_num - self.config.time_step + 1)]
        test_y = [label_data[i:i + self.config.time_step] for i in range(test_num - self.config.time_step + 1)]

        return np.array(test_x), np.array(test_y)

    # 获取数据
    def get_data(self):
        df = pd.read_csv(self.config.csv_file)
        df = pd.DataFrame(df)
        # denose_df = denosing(df)
        # denose_df.to_csv("./data/dataset2_denose.csv",encoding="utf_8_sig", index=False)
        # print(df)
        # 计算涨跌
        df = df.iloc[::-1]
        # df.drop(['date', 'code'], axis=1, inplace=True)
        prev_amplitude = 0
        threshold = 0.055
        for item in df.iterrows():
            data_row = item[0]
            data = item[1]
            df.loc[data_row, 'label'] = 1 if prev_amplitude >= threshold else 0
            prev_amplitude = (float(data['close']) - float(data['preclose'])) / float(data['preclose']) * 100
        df['label'] = df['label'].astype('int')
        df = df.iloc[::-1]
        print(df)

        df.to_csv("./dataframe1.csv", encoding="utf_8_sig", index=False)
        data_label = df['label']
        df.drop(['label'], axis=1, inplace=True)
        return df.values, data_label, df.columns.tolist(), denosing(df)