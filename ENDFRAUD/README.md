# ENDFRAUD

## 项目简介

本项目实现了一个面向多关系欺诈图的节点分类与欺诈检测模型。代码以 YelpChi 和 Amazon 等欺诈检测数据集为主要对象，利用节点属性、多关系邻接结构以及随机游走位置编码特征进行训练，并输出 F1-Macro、AUC、G-Mean 等评价指标。

项目中的核心模型为 ENDFRAUD。其训练流程主要包括数据读取、训练集/验证集/测试集划分、邻居采样、多关系聚合、对比约束训练、模型保存与测试评估。

## 项目结构

```text
ENDFRAUD/
├── main.py                         # 项目入口，读取配置并启动训练或多组实验
├── config/
│   ├── pcgnn\_yelpchi.yml            # YelpChi 数据集配置文件
│   └── pcgnn\_amazon.yml             # Amazon 数据集配置文件
├── data/
│   └── pe\_feature.py                # 随机游走位置编码特征生成脚本
├── src/
│   ├── model\_handler.py             # 数据加载、模型构建、训练、验证、测试与可视化流程
│   ├── model.py                     # ENDFRAUD 分类层与损失函数
│   ├── layers.py                    # 关系内聚合与关系间聚合模块
│   ├── graphsage.py                 # GraphSAGE 与 GCN 基线模型
│   ├── neigh\_gen\_model.py            # 邻域增强生成模块与对比学习模块
│   ├── neigh\_gen\_layers.py           # 图卷积基础层
│   ├── choose\_neighs.py              # 基于策略网络的邻居选择函数
│   ├── choose\_neighs2.py             # 第二类邻居选择函数
│   ├── policyNetwork.py              # 邻居选择策略网络
│   ├── data\_process.py               # 将稀疏邻接矩阵转换为邻接表文件
│   └── utils.py                      # 数据读取、采样、评价指标等工具函数
└── pytorch\_models/
    └── visualizations/              # 训练后可视化结果保存目录
```

## 运行环境

建议使用 Python 3.8 或 Python 3.9。主要依赖如下：

```bash
pip install torch numpy scipy scikit-learn pyyaml matplotlib
```

如果使用 GPU，需要安装与本机 CUDA 版本匹配的 PyTorch。若只使用 CPU，可以在配置文件中设置：

```yaml
no\_cuda: true
```

需要注意，`config/pcgnn\_yelpchi.yml` 中当前写有 `no\_cuda: Ture`，这里的 `Ture` 是拼写错误，会被 YAML 解析为字符串。建议改为：

```yaml
no\_cuda: true
```

如果希望启用 GPU，可改为：

```yaml
no\_cuda: false
cuda\_id: '0'
```

## 数据准备

数据文件没有包含在当前代码压缩包中，需要手动放入 `data/` 目录。代码默认支持 YelpChi、Amazon 和 t-finance 三类数据读取方式，其中配置文件已经提供了 YelpChi 和 Amazon 两个实验设置。

### YelpChi 数据

运行 YelpChi 实验时，`data/` 目录下至少需要包含：

```text
data/YelpChi.mat
data/pe\_features.mat
data/yelp\_homo\_adjlists.pickle
data/yelp\_rur\_adjlists.pickle
data/yelp\_rtr\_adjlists.pickle
data/yelp\_rsr\_adjlists.pickle
```

其中，`YelpChi.mat` 中需要包含以下字段：

```text
label
features
homo
net\_rur
net\_rtr
net\_rsr
```

如果还没有邻接表文件，可以运行：

```bash
python src/data\_process.py
```

当前 `src/data\_process.py` 默认处理 YelpChi 数据，并将 `net\_rur`、`net\_rtr`、`net\_rsr` 和 `homo` 转换为对应的 `.pickle` 邻接表文件。

### Amazon 数据

运行 Amazon 实验时，`data/` 目录下至少需要包含：

```text
data/Amazon.mat
data/amz\_homo\_adjlists.pickle
data/amz\_upu\_adjlists.pickle
data/amz\_usu\_adjlists.pickle
data/amz\_uvu\_adjlists.pickle
```

其中，`Amazon.mat` 中需要包含以下字段：

```text
label
features
homo
net\_upu
net\_usu
net\_uvu
```

Amazon 的 `pe\_features.mat` 不是强制项。若没有该文件，代码会自动构造一个默认的零矩阵作为位置编码特征。若需要使用真实的位置编码特征，可以运行 `data/pe\_feature.py` 生成随机游走位置编码，然后保存到 `data/pe\_features.mat`。

需要注意，`src/data\_process.py` 中 Amazon 邻接表生成部分目前是注释状态。如果需要重新生成 Amazon 的邻接表文件，需要取消对应代码注释后再运行脚本。

## 配置文件说明

配置文件位于 `config/` 目录下，主要字段含义如下：

|参数|含义|
|-|-|
|`data\_name`|数据集名称，可选 `yelp`、`amazon`、`t-finance`|
|`data\_dir`|数据目录，默认是 `./data/`|
|`train\_ratio`|训练集比例|
|`test\_ratio`|从剩余数据中划分测试集的比例|
|`save\_dir`|模型保存目录|
|`model`|模型名称，可选 `ENDFRAUD`、`GNN`、`SAGE`、`GCN`|
|`multi\_relation`|多关系聚合方式，当前配置为 `GNN`|
|`emb\_size`|节点表示维度|
|`thres`|根据预测概率生成类别标签的分类阈值|
|`rho`|邻居选择相关参数|
|`seed`|随机种子|
|`optimizer`|优化器名称，当前代码中使用 Adam|
|`lr`|学习率|
|`weight\_decay`|权重衰减系数|
|`batch\_size`|批大小|
|`num\_epochs`|训练轮数|
|`valid\_epochs`|每隔多少轮进行一次验证|
|`alpha`|对比学习损失权重|
|`no\_cuda`|是否禁用 GPU|
|`cuda\_id`|GPU 编号|

## 训练模型

在项目根目录下运行以下命令。

### YelpChi

```bash
python main.py -config config/pcgnn\_yelpchi.yml
```

### Amazon

```bash
python main.py -config config/pcgnn\_amazon.yml
```

训练过程中会打印当前配置、数据集划分信息、每个 epoch 的损失、验证集结果以及最终测试结果。输出指标包括：

```text
F1-Macro
F1-binary-1
F1-binary-0
AUC
G-Mean
TP / TN / FP / FN
```

## 模型保存与可视化

训练过程中，代码会根据验证集 AUC 保存当前最优模型。模型文件默认保存到：

```text
pytorch\_models/<timestamp>/<data\_name>\_<model>.pkl
```

其中，`<timestamp>` 为训练开始时的时间戳。

`main.py` 中包含训练后节点表示可视化逻辑。若可视化流程正常执行，t-SNE 可视化图片会保存到：

```text
pytorch\_models/visualizations/node\_embeddings\_<data\_name>\_<model>\_<timestamp>.png
```

如果训练完成后可视化失败，一般不会影响模型训练和测试结果。可检查 `ModelHandler.train()` 中是否将训练好的模型赋值给 `self.gnn\_model`，以及 `visualize\_embeddings()` 中是否对 `ENDFRAUD` 与 `GNN` 两种模型名称做了统一判断。

## 位置编码特征生成

`data/pe\_feature.py` 用于根据图结构生成随机游走位置编码特征。其主要流程为：

1. 读取 `.mat` 文件中的同质图邻接矩阵；
2. 计算随机游走矩阵；
3. 提取多步随机游走矩阵的对角线作为初始位置编码；
4. 通过随机映射得到指定维度的位置编码特征；
5. 将结果保存为 `.mat` 文件。

运行命令示例：

```bash
python data/pe\_feature.py
```

当前脚本示例默认读取 `data/Amazon.mat`。如果要处理 YelpChi，需要将脚本中的读取路径和邻接矩阵字段改为对应数据集，例如 `data/YelpChi.mat` 和 `homo`。

## 代码执行流程

整体训练流程如下：

1. `main.py` 读取配置文件，并设置随机种子；
2. `ModelHandler` 调用 `load\_data()` 读取节点特征、标签、位置编码和多关系邻接表；
3. 根据数据集名称划分训练集、验证集和测试集；
4. 构建 ENDFRAUD、GraphSAGE 或 GCN 模型；
5. 每个 epoch 中对训练节点进行欠采样，缓解类别不平衡问题；
6. 在每个 batch 中计算分类损失和对比学习损失；
7. 按照验证集 AUC 保存最优模型；
8. 训练结束后加载最优模型，在测试集上输出最终结果。

## 常见问题

### 1\. 找不到 `.mat` 或 `.pickle` 数据文件

请确认数据文件已经放入 `data/` 目录，并且文件名与 `src/utils.py` 中的读取路径一致。例如 YelpChi 对应 `YelpChi.mat`，Amazon 对应 `Amazon.mat`。

### 2\. 运行 YelpChi 时提示找不到邻接表文件

先运行：

```bash
python src/data\_process.py
```

如果仍然报错，请检查 `YelpChi.mat` 中是否包含 `net\_rur`、`net\_rtr`、`net\_rsr` 和 `homo` 字段。

### 3\. GPU 没有被使用

请检查配置文件中的设置：

```yaml
no\_cuda: false
cuda\_id: '0'
```

同时确认本机已经正确安装支持 CUDA 的 PyTorch。

### 4\. `no\_cuda: Ture` 是否正确

不正确。应修改为：

```yaml
no\_cuda: true
```

或：

```yaml
no\_cuda: false
```

### 5\. 可视化阶段报错

训练结果通常已经正常生成。可视化报错可能与模型对象未保存到 `self.gnn\_model` 或模型名称判断不一致有关。可以先忽略可视化，使用保存的 `.pkl` 模型结果进行实验记录。

## 参考运行命令汇总

```bash
# 安装依赖
pip install torch numpy scipy scikit-learn pyyaml matplotlib

# 生成 YelpChi 邻接表
python src/data\_process.py

# 训练 YelpChi
python main.py -config config/pcgnn\_yelpchi.yml

# 训练 Amazon
python main.py -config config/pcgnn\_amazon.yml

