from dataset import Data
# import numpy as np
# from dataset_f import DatasetFromCSV
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from sklearn.metrics import f1_score, matthews_corrcoef
import time
from tensorboardX import SummaryWriter
import torch
import warnings
from importlib import import_module
module = import_module(name=Config.model)
Model = getattr(module, "Model")
warnings.filterwarnings("ignore")


# 先清空logs文件夹

if __name__ == '__main__':
    conf = Config()
    data = Data(conf)

    device = torch.device("cuda:0" if conf.use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU
    print(torch.cuda.is_available())
    model = Model(conf).to(device)  # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中

    train_X, valid_X, train_Y, valid_Y = data.get_train_and_valid_data()
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y),
                              shuffle=False,
                              batch_size=conf.batch_size)  # DataLoader可自动生成可训练的batch数据

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y),
                              shuffle=False,
                              batch_size=conf.batch_size)

    model = Model(conf).to(device)  # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
    print(model)
    writer = SummaryWriter("logs")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # 这两句是定义优化器和loss

    train_correct = 0
    test_correct = 0
    train_total = 0
    test_total = 0

    best_mcc = {
        'value': 0
    }
    best_result = {
        'value': 0
    }
    for epoch in range(conf.epoch):
        time_total = 0
        # start_time = time.time()
        model.train()
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            print(_train_X.size()) #32, 10, 9
            print(_train_Y.size()) #32, 10
            optimizer.zero_grad()
            out = model(_train_X) # 64, 2
            _train_Y = torch.tensor(_train_Y[:, -1], dtype=torch.int64) #提取最后一个 64, 10
            loss = criterion(out, _train_Y)
            out = out.argmax(dim=1)
            train_correct += (out == _train_Y).sum().item()
            train_total += len(_train_Y)
            loss.backward()
            optimizer.step()
        print("epoch: {}, loss: {}".format(epoch + 1, loss.item()))
        print("训练集上的正确率: {}".format(train_correct / train_total))
        writer.add_scalar("train_accuracy", train_correct / train_total, epoch + 1)
        # if (epoch + 1) % 50 == 0:
        # 保存模型
        torch.save(model.state_dict(), "./models_pth/model_{}.pth".format(epoch + 1))
        writer.add_scalar("train_loss", loss.item(), epoch + 1)

        prob_all = []
        label_all = []
        # 测试步骤
        model.eval()  # pytorch中，预测时要转换成预测模式
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            out = model(_valid_X)
            _valid_Y = torch.tensor(_valid_Y[:, -1], dtype=torch.int64)

            loss = criterion(out, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            out = out.argmax(dim=1)
            test_correct += (out == _valid_Y).sum().item()
            test_total += len(_valid_Y)
            prob_all.extend(out)  # 求每一行的最大值索引
            label_all.extend(_valid_Y)
        prob_all = [i.item() for i in prob_all]
        label_all = [i.item() for i in label_all]
        end_time = time.time()
        valid_loss_cur = loss.item()
        # 评估指标
        Accuracy = test_correct / test_total
        F1_score = f1_score(label_all, prob_all)
        MCC = matthews_corrcoef(label_all, prob_all)
        print("测试集上的loss:{}".format(loss.item()))
        print("测试集上的正确率: {}".format(Accuracy))
        print("测试集上的F1-score:{:.3f}".format(F1_score))
        print("测试集上的MCC:{:.3f}".format(MCC))
        writer.add_scalar("test_MCC", MCC, epoch + 1)
        writer.add_scalar("test_F1-score", F1_score, epoch + 1)
        writer.add_scalar("test_loss", loss.item(), epoch + 1)
        writer.add_scalar("test_accuracy", Accuracy, epoch + 1)

        result = Accuracy + 1.15 * F1_score + 2.2 * MCC
        if matthews_corrcoef(label_all, prob_all) > best_mcc['value']:
            best_mcc['value'] = matthews_corrcoef(label_all, prob_all)
            best_mcc['epoch'] = epoch + 1
        if result > best_result['value']:
            best_result['value'] = result
            best_result['epoch'] = epoch + 1
print("最好的mcc：{}在epoch{}".format(best_mcc['value'], best_mcc['epoch']))
print("最好的result：{}在epoch{}".format(best_result['value'], best_result['epoch']))
writer.close()