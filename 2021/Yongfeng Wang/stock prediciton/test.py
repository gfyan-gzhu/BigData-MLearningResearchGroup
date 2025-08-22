from dataset import Data
from config import Config
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, matthews_corrcoef
from importlib import import_module
module = import_module(name=Config.model)
Model = getattr(module, "Model")
import pandas as pd
if __name__ == '__main__':
    conf = Config()
    data = Data(conf)
    device = torch.device("cuda:0" if conf.use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU
    print(torch.cuda.is_available())

    # 获取测试数据
    test_X, text_Y = data.get_test_data()

    test_X, test_Y = torch.from_numpy(test_X).float(), torch.from_numpy(text_Y).float()

    test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=1, shuffle=False)
    # 加载模型
    device = torch.device("cuda:0" if conf.use_cuda and torch.cuda.is_available() else "cpu")
    model = Model(conf).to(device)
    model.load_state_dict(torch.load('./models_pth/model_30.pth'))  # 加载模型参数
    # model.load_state_dict(torch.load('./best_models/dataset1/cnn/model_18.pth'))  # 加载模型参数

    df = pd.read_csv(conf.csv_file)
    df = pd.DataFrame(df)
    # 取时间步-
    df_index = df.date.tolist()[conf.time_step:]
    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)
    # 预测过程
    model.eval()
    correct = 0
    total = 0
    index = 0
    prob_all = []
    label_all = []
    for _test_X, _test_Y in test_loader:

        _test_X, _test_Y = _test_X.to(device), _test_Y.to(device)
        out = model(_test_X)
        if out.size()[0] == 2:
            out = torch.tensor([out.tolist()]).to(device) # 遇到ValueError则注释这句
        out_result = out.argmax(dim=1)
        _test_Y = torch.tensor(_test_Y[:, -1], dtype=torch.int64)
        pred_result = "涨" if int(out_result.item()) == 1 else "跌"
        true_result = "涨" if int(_test_Y.item()) == 1 else "跌"
        out_metrix = [[round(j,3) for j in out.tolist()[i]] for i in range(len(out.tolist()))]
        print("{}的预测结果是{},实际结果为{},预测矩阵为{}".format(df_index[index], pred_result, true_result, out_metrix))
        correct += (out_result == _test_Y).sum().item()
        total += len(_test_Y)
        index = index + 1

        prob_all.extend(out_result)  # 求每一行的最大值索引
        label_all.extend(_test_Y)

    prob_all = [i.item() for i in prob_all]
    label_all = [i.item() for i in label_all]
    print("F1-score:{:.3f}".format(f1_score(label_all, prob_all)))
    print("MCC:{:.3f}".format(matthews_corrcoef(label_all, prob_all)))
    print("accuracy:{}".format(correct / total))
