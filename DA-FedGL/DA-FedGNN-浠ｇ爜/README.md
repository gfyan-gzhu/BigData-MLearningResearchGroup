# DA-FedGNN: Personalized Federated Graph Learning Framework

本项目是一个面向图分类任务的个性化联邦图学习实验框架，主要用于在多客户端、多数据集和数据异构场景下，对比不同联邦图学习算法的分类性能。

项目以 GIN（Graph Isomorphism Network）作为基础图神经网络模型，支持 Self-Training、FedAvg、FedProx、固定 α 的 APFL、DA-FedGNN、D-FedGNN、PENS 等方法。其中，DA-FedGNN 是本项目的核心方法：它为每个客户端构建本地个性化分支和全局共享分支，并通过 α 参数自适应融合本地知识与全局知识，以提升异构图数据场景下的个性化学习效果。

---

## 1. Project Overview

在传统联邦学习中，客户端通常共享同一个全局模型。但在图分类任务中，不同客户端可能对应不同图数据集，存在节点特征维度、图结构、标签分布和类别数量等方面的差异。直接使用统一全局模型，可能难以适应所有客户端的数据分布。

本项目围绕以下问题展开实验：

- 多数据集图分类场景下，不同联邦学习方法的性能差异；
- FedAvg/FedProx 在强异构图数据上的表现；
- 固定 α 融合策略与自适应 α 融合策略的差异；
- DA-FedGNN 中本地分支与全局分支如何共同提升个性化性能；
- 各算法在 Accuracy、F1-score、Recall、Precision、Loss 等指标上的综合表现。

---

## 2. Main Features

- 支持 TUDataset 图分类数据集；
- 支持 MultiDS 多数据集联邦学习设置；
- 支持节点特征维度统一，包括 PCA 降维和 zero-padding 补齐；
- 使用 GIN 作为基础图分类模型；
- 支持多种联邦学习算法对比；
- 支持 DA-FedGNN 的本地/全局双分支结构；
- 支持自适应 α 参数学习；
- 支持固定 α 对照实验；
- 支持记录每轮通信后的模型性能；
- 支持记录本地训练 loss 曲线；
- 支持保存实验 CSV 文件和最终对比报告；
- 支持多随机种子重复实验；
- 提供 `experiment_aggregator.py` 用于实验结果后处理、统计分析和可视化。

---

## 3. Repository Structure

```text
.
├── main.py                    # 实验主入口：参数解析、数据加载、算法调度和结果保存
├── runner.sh                  # 多随机种子重复实验脚本
├── requirements.txt           # 项目依赖环境说明
├── experiment_aggregator.py   # 实验结果聚合、统计分析和可视化工具
│
├── setup.py                   # 数据加载、数据划分、特征维度统一、客户端与服务器创建
├── models.py                  # GIN、DA-FedGNN、FedEgo、PENS 等模型定义
├── clients.py                 # 基础客户端、DA-FedGNN 客户端及其他算法客户端
├── server.py                  # 服务器端聚合、客户端相似性计算和数据统计工具
├── aggregation.py             # 统一联邦聚合模块
├── evaluation.py              # 统一评估模块
├── utils.py                   # 结果保存、指标聚合、最终报告和可视化工具
│
├── selftrain.py               # Self-Training 本地训练基线
├── dafedgnn.py                # DA-FedGNN 和固定 α APFL 训练流程
├── fedavg.py                  # FedAvg 训练流程
├── fedprox.py                 # FedProx 训练流程
├── dfedgnn.py                 # D-FedGNN 训练流程
├── pens.py                    # PENS 训练流程
├── fedego.py                  # FedEgo 训练流程
│
├── data/                      # 数据目录，可根据实际情况调整
└── outputs/                   # 实验结果输出目录
```

> Note: 如果当前项目目录中暂时没有 `fedavg.py`、`fedprox.py`、`dfedgnn.py`、`pens.py` 或 `fedego.py`，请根据实际代码情况补齐，或调整 `main.py` 中的导入与调用逻辑。

---

## 4. Environment Requirements

建议使用 Python 3.9 左右版本。当前 `requirements.txt` 中给出的是适配 Python 3.9.13、PyTorch 2.2.0 和 CUDA 12.2 环境的依赖版本。

主要依赖包括：

```text
torch
torchvision
torchaudio
torch-geometric
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
numpy
pandas
scipy
scikit-learn
dtaidistance
networkx
matplotlib
seaborn
tqdm
```

---

## 5. Installation

### 5.1 Create environment

```bash
conda create -n dafedgnn python=3.9
conda activate dafedgnn
```

### 5.2 Install PyTorch

如果使用 CUDA 12.1/12.2 环境，可参考：

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

如果只使用 CPU，可根据 PyTorch 官方安装命令调整。

### 5.3 Install PyG extensions

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### 5.4 Install PyTorch Geometric

```bash
pip install torch-geometric==2.4.0
```

### 5.5 Install remaining packages

```bash
pip install numpy pandas scipy scikit-learn dtaidistance networkx matplotlib seaborn tqdm
```

也可以根据 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```

> 如果直接使用 `pip install -r requirements.txt` 出现 PyTorch Geometric 扩展包编译问题，建议按照上面的分步方式安装 PyTorch 和 PyG 相关依赖。

---

## 6. Dataset Preparation

项目默认使用 `torch_geometric.datasets.TUDataset` 加载图分类数据集。

默认数据路径通常为：

```text
../data/TUDataset
```

在 `main.py` 中，数据加载逻辑类似：

```python
splitedData, df = setup.prepareData_multiDS(
    datapath='../data',
    group=args.dataset,
    batchSize=128,
    convert_x=False,
    seed=args.seed,
    target_dim=64
)
```

其中 `target_dim=64` 表示将不同数据集的节点特征统一到 64 维：

- 若原始特征维度大于 64，则使用 PCA 降维；
- 若原始特征维度小于 64，则使用 zero-padding 补齐；
- 若原始特征维度等于 64，则保持不变。

---

## 7. Supported Dataset Groups

`setup.py` 中支持以下数据集组：

| Group | Included Datasets |
|---|---|
| `molecules` | MUTAG, BZR, COX2, DHFR, PTC_MR, AIDS, NCI1 |
| `molecules_tiny` | molecules 的小规模版本 |
| `small` | MUTAG, BZR, COX2, DHFR, PTC_MR, ENZYMES, DD, PROTEINS |
| `mix` | molecules + bioinformatics + social networks |
| `mix_tiny` | mix 的小规模版本 |
| `biochem` | MUTAG, BZR, COX2, DHFR, PTC_MR, AIDS, NCI1, ENZYMES, DD, PROTEINS |
| `biochem_tiny` | biochem 的小规模版本 |

默认建议使用：

```bash
--dataset biochem
```

---

## 8. Quick Start

### 8.1 Single run on CPU

```bash
python3 main.py \
    --dataset biochem \
    --num_rounds 200 \
    --local_epoch 1 \
    --seed 1000 \
    --repeat 1 \
    --device cpu
```

### 8.2 Single run on GPU

```bash
python3 main.py \
    --dataset biochem \
    --num_rounds 400 \
    --local_epoch 1 \
    --seed 42 \
    --device cuda:0
```

### 8.3 Quick test with tiny dataset

```bash
python3 main.py \
    --dataset biochem_tiny \
    --num_rounds 20 \
    --local_epoch 1 \
    --seed 42 \
    --device cpu
```

### 8.4 Repeated experiments

项目提供 `runner.sh` 进行多随机种子重复实验：

```bash
bash runner.sh
```

默认脚本会使用多个随机种子重复运行实验，便于统计平均性能和稳定性。

---

## 9. Main Arguments

| Argument | Default | Description |
|---|---:|---|
| `--device` | `cuda:0` | 运行设备，如 `cpu` 或 `cuda:0` |
| `--dataset` | `biochem` | 数据集组名称 |
| `--model` | `gin` | 基础模型名称 |
| `--num_rounds` | `400` | 联邦通信轮数 |
| `--local_epoch` | `1` | 每轮本地训练 epoch 数 |
| `--hidden` | `64` | 隐藏层维度 |
| `--nlayer` | `3` | GIN 层数 |
| `--dropout` | `0.5` | dropout 概率 |
| `--lr` | `0.001` | 学习率 |
| `--weight_decay` | `5e-4` | 权重衰减 |
| `--num_clients` | `10` | 客户端数量，主要用于单数据集切分场景 |
| `--seed` | `42` | 随机种子 |
| `--repeat` | `None` | 重复实验编号 |

---

## 10. Algorithms

### 10.1 Self-Training

每个客户端仅使用自己的本地数据训练模型，不与其他客户端通信，也不进行参数聚合。该方法作为本地训练基线，用于判断联邦聚合是否带来性能提升。

### 10.2 FedAvg

FedAvg 是经典联邦平均算法。每轮中：

1. 每个客户端在本地数据上训练模型；
2. 客户端上传模型参数；
3. 服务器根据客户端训练样本数量进行加权平均；
4. 聚合后的全局模型广播回客户端。

### 10.3 FedProx

FedProx 的聚合方式与 FedAvg 类似，但在本地训练损失中加入近端项，用于限制客户端模型偏离全局模型过远。该方法适用于客户端数据分布存在明显异构的场景。

### 10.4 Fixed-α APFL

固定 α 实验用于分析不同本地/全局融合比例对模型性能的影响。项目默认测试：

```text
α = 0.2
α = 0.5
α = 0.8
```

其中：

- α 越大，模型越依赖本地个性化分支；
- α 越小，模型越依赖全局共享分支。

### 10.5 DA-FedGNN

DA-FedGNN 是本项目重点实现的方法。每个客户端包含两个分支：

| Component | Description |
|---|---|
| Local branch `v_i` | 本地个性化模型，不参与联邦聚合 |
| Global branch `w_i` | 全局共享模型，参与联邦聚合 |
| Alpha `α_i` | 控制本地分支与全局分支的融合比例 |

最终预测形式为：

```text
ŷ_i = α_i · v_i + (1 - α_i) · w_i
```

训练过程中：

1. 客户端使用训练集更新本地分支和全局分支；
2. 使用验证集更新 α；
3. 仅聚合全局共享分支；
4. 本地个性化分支保留在客户端内部；
5. 每个客户端根据自身数据分布自适应学习 α。

### 10.6 D-FedGNN

D-FedGNN 是去中心化联邦图学习方法。与传统服务器聚合不同，D-FedGNN 可以在客户端之间进行点对点参数交互或邻居聚合。

### 10.7 PENS

PENS 是一种结合去中心化机制和邻居选择策略的个性化联邦学习方法。项目中在 MultiDS 模式下会调用对应训练流程进行多数据集实验。

---

## 11. Experiment Flow

完整实验流程如下：

```text
1. 解析命令行参数
2. 设置随机种子
3. 创建输出目录
4. 加载 TUDataset 图分类数据
5. 统一节点特征维度
6. 构建客户端和服务器
7. 依次运行多个算法
8. 每隔若干轮评估客户端性能
9. 保存每个算法的 CSV 结果
10. 保存本地训练 loss 曲线
11. 生成最终性能对比报告
```

---

## 12. Output Files

实验结果默认保存到：

```text
./outputs/multiDS/<dataset>_run<repeat>_seed<seed>/
```

例如：

```text
./outputs/multiDS/biochem_run1_seed1000/
```

常见输出文件包括：

```text
selftrain_enhanced.csv
fedavg_enhanced.csv
fedavg_enhanced_local_loss.csv
fedprox_enhanced.csv
fedprox_enhanced_local_loss.csv
alpha_0.2_fixed.csv
alpha_0.2_fixed_local_loss.csv
alpha_0.5_fixed.csv
alpha_0.5_fixed_local_loss.csv
alpha_0.8_fixed.csv
alpha_0.8_fixed_local_loss.csv
dafedgnn_advanced.csv
dafedgnn_advanced_local_loss.csv
dfedgnn.csv
dfedgnn_local_loss.csv
pens.csv
final_comparison_report.csv
```

如果设置了 `--repeat`，结果会进入对应的重复实验目录，便于后续汇总分析。

---

## 13. Evaluation Metrics

项目主要记录以下指标：

| Metric | Description |
|---|---|
| `mean_accuracy` | 所有客户端平均准确率 |
| `std_accuracy` | 客户端准确率标准差 |
| `mean_f1` | 宏平均 F1-score |
| `std_f1` | F1-score 标准差 |
| `mean_recall` | 宏平均 Recall |
| `std_recall` | Recall 标准差 |
| `mean_precision` | 宏平均 Precision |
| `std_precision` | Precision 标准差 |
| `mean_loss` | 平均测试损失 |
| `std_loss` | 测试损失标准差 |
| `mean_alpha` | DA-FedGNN 中客户端 α 的平均值 |
| `std_alpha` | DA-FedGNN 中客户端 α 的标准差 |

---

## 14. Using `experiment_aggregator.py`

`experiment_aggregator.py` 是一个实验结果后处理工具，主要用于：

- 自动扫描实验结果目录；
- 聚合多次重复实验；
- 计算平均值、标准差、标准误和置信区间；
- 以某个基线算法为参照，计算性能提升；
- 执行统计显著性检验；
- 生成综合分析图表；
- 保存分析报告和 JSON 结果。

运行方式：

```bash
python3 experiment_aggregator.py
```

程序会提示输入原始实验结果目录，例如：

```text
./outputs/multiDS/biochem_run1_seed1000/repeats
```

默认配置中，分析器以 `selftrain` 作为基准算法，以 `test_acc` 作为主要指标。如果当前实验输出列名不是 `test_acc`，而是 `mean_accuracy` 或其他名称，需要在 `experiment_aggregator.py` 中修改：

```python
metric_column = 'mean_accuracy'
```

或根据实际 CSV 文件列名进行调整。

---

## 15. Secure Aggregation

项目中包含基于 ECDH 和伪随机掩码的安全聚合模块。如果安装了 `cryptography`，则可启用安全聚合；如果未安装该库，程序会自动关闭安全聚合，并继续执行普通联邦训练。

安全聚合的基本思想是：

```text
客户端之间生成成对随机掩码；
每个客户端上传带掩码的更新；
聚合时掩码相互抵消；
服务器只能看到聚合结果，而不能直接看到单个客户端的真实更新。
```

---

## 16. Notes

1. MultiDS 模式下，不同数据集的类别数可能不同，因此聚合时需要避免直接聚合 shape 不一致的参数。
2. 如果不同客户端参数 shape 不一致，聚合模块会跳过不兼容参数。
3. 如果使用 GPU，请确认 PyTorch、CUDA 和 PyTorch Geometric 版本匹配。
4. 如果运行时提示找不到 TUDataset，请检查数据路径是否正确。
5. 如果运行 `main.py` 时出现模块导入错误，请确认所有算法文件均位于项目根目录。
6. 如果内存不足，可以减少 `num_rounds`、降低 batch size，或使用 tiny 数据集组。
7. `experiment_aggregator.py` 属于后处理脚本，不影响主训练流程；若结果文件名或列名与默认设置不一致，需要相应修改分析配置。

---

## 17. Example Commands

### CPU single run

```bash
python3 main.py \
    --dataset biochem \
    --num_rounds 200 \
    --local_epoch 1 \
    --seed 1000 \
    --repeat 1 \
    --device cpu
```

### GPU single run

```bash
python3 main.py \
    --dataset biochem \
    --num_rounds 400 \
    --local_epoch 1 \
    --seed 42 \
    --device cuda:0
```

### Tiny dataset quick test

```bash
python3 main.py \
    --dataset biochem_tiny \
    --num_rounds 20 \
    --local_epoch 1 \
    --seed 42 \
    --device cpu
```

### Repeated experiments

```bash
bash runner.sh
```

### Result aggregation and analysis

```bash
python3 experiment_aggregator.py
```

---

## 18. Citation

If this project is used in academic research, please cite the corresponding paper, technical report, or project documentation associated with DA-FedGNN.

---

## 19. License

This project is intended for academic research and experimental comparison. Please check the licenses of all third-party datasets and dependencies before redistribution.
