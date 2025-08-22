from config import Config
import torch
import baostock as bs
import time
import datetime
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model.lstm import Model
import numpy as np
# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

class Data:
    def __init__(self, config, predict_time):
        self.config = config
        self.predict_time = predict_time
        self.data, self.label, self.column_name = self.get_higher_data()
        self.data = self.data.astype('float')
        self.data_num = self.data.shape[0] # 数组行数

        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean) / self.std # 归一化 axis=0，计算每一列的均值

    def get_test_data(self, return_label_data=True):
        feature_data = self.norm_data
        label_data = self.label

        test_x = [feature_data[i:i + self.config.time_step] for i in range(len(feature_data) - self.config.time_step + 1)]
        test_y = [label_data[i:i + self.config.time_step] for i in range(len(feature_data) - self.config.time_step + 1)]

        return np.array(test_x), np.array(test_y)

    # 获取数据
    def get_data(self):
        init_time = datetime.datetime.strptime(self.predict_time, "%Y-%m-%d")
        start_date = str(init_time - datetime.timedelta(days=90)).split(' ')[0] # 避免数据不足time_step
        end_date = str(init_time - datetime.timedelta(days=1)).split(' ')[0] # 取到预测前一天
        rs = bs.query_history_k_data_plus("sh.000001",
                                          "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                          start_date=start_date, end_date=end_date, frequency="d")
        # 打印结果集
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        # print(df)
        # 取时间步长度数据进行预测
        df = df[-self.config.time_step:]
        # print(df)
        return df

    # 获取提炼后的数据
    def get_higher_data(self):
        df = self.get_data()
        # 计算涨跌
        df = df.iloc[::-1]
        df.drop(['date', 'code'], axis=1, inplace=True)
        prev_amplitude = 0
        threshold = 0.055
        for item in df.iterrows():
            data_row = item[0]
            data = item[1]
            df.loc[data_row, 'label'] = 1 if prev_amplitude >= threshold else 0
            prev_amplitude = (float(data['close']) - float(data['preclose'])) / float(data['preclose']) * 100
        df['label'] = df['label'].astype('int')
        df = df.iloc[::-1]
        # print(df)
        data_label = df['label']
        df.drop(['label'], axis=1, inplace=True)
        return df.values, data_label, df.columns.tolist()


# 预测当天涨跌
def predict(predict_time, day=1):
    conf = Config()
    device = torch.device("cuda:0" if conf.use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU
    print(torch.cuda.is_available())
    # 获取股市交易时间
    init_time = datetime.datetime.strptime(predict_time, "%Y-%m-%d")
    start_date = str(init_time).split(' ')[0]
    end_date = str(init_time + datetime.timedelta(days=60)).split(' ')[0]
    trade = bs.query_history_k_data_plus("sh.000001",
                                      "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                      start_date=start_date, end_date=end_date, frequency="d")
    # 打印结果集
    data_list = []
    while (trade.error_code == '0') & trade.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(trade.get_row_data())
    trade = pd.DataFrame(data_list, columns=trade.fields)
    trade_time = trade.date.tolist()[:int(day)]
    for day in trade_time:
        predict_time = day
        data = Data(conf, predict_time)

        # 获取测试数据
        test_X, text_Y = data.get_test_data()
        test_X, test_Y = torch.from_numpy(test_X).float(), torch.from_numpy(text_Y).float()

        test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=1)
        # 加载模型
        device = torch.device("cuda:0" if conf.use_cuda and torch.cuda.is_available() else "cpu")
        model = Model(conf).to(device)
        model.load_state_dict(torch.load('./models_pth/model_3000.pth'))  # 加载模型参数

        # 先定义一个tensor保存预测结果
        result = torch.Tensor().to(device)
        # 预测过程
        model.eval()
        for _test_X, _test_Y in test_loader:
            _test_X, _test_Y = _test_X.to(device), _test_Y.to(device)
            out = model(_test_X)
            out_result = out.argmax(dim=1)
            result = "涨" if int(out_result.item()) == 1 else "跌"
            delta = datetime.timedelta(days=1)
            out_metrix = [[round(j, 3) for j in out.tolist()[i]] for i in range(len(out.tolist()))]
            print("预测{}涨跌为{},预测矩阵为{}".format(day, result, out_metrix))

if __name__ == '__main__':
    predict('2022-01-18', 9)
