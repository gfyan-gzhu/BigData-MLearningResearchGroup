# AST-VGAE

AST-VGAE 是一个面向动态图异常节点检测的实验代码项目。代码实现了基于 Beta 小波多尺度图门控循环单元的时空编码器，并结合变分图自编码器、时间感知先验、属性重构和结构重构来计算节点异常分数。项目可用于 JODIE 风格事件数据集，也可用于 `.npz` 动态图数据集。

## 1\. 项目结构

```text
AST-VGAE\\\\\\\\\\\\\\\_readme\\\\\\\\\\\\\\\_

├── ast\\\\\\\\\\\\\\\_vgae.py                # AST-VGAE 模型，主要用于事件流 CSV 数据
├── ast\\\\\\\\\\\\\\\_vgae\\\\\\\\\\\\\\\_dblp.py           # 支持可调 wavelet\\\\\\\\\\\\\\\_C 的 AST-VGAE 版本，主要用于 DBLP/STAR 数据
├── train\\\\\\\\\\\\\\\_event\\\\\\\\\\\\\\\_graph.py       # 事件流动态图训练入口，对应 Wikipedia、Reddit、MOOC 等 CSV 数据
├── train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py     # 动态图训练入口，支持 .npz 与部分 CSV 数据
├── event\\\\\\\\\\\\\\\_graph\\\\\\\\\\\\\\\_dataset.py     # 事件流 CSV 数据读取、快照划分、Node2Vec 特征生成与 PyGOD 异常注入
├── dynamic\\\\\\\\\\\\\\\_graph\\\\\\\\\\\\\\\_dataset.py   # CSV/.npz 动态图数据读取、划分与测试阶段异常注入
├── prepare\\\\\\\\\\\\\\\_star\\\\\\\\\\\\\\\_dataset.py    # .npz 数据预处理脚本
├── data/
│   └── DBLP3.npz              # 示例 DBLP 数据文件
├── README.md
└── .gitignore
```

## 2\. 环境依赖

建议使用 Python 3.9 或相近版本。项目主要依赖如下库：

```bash
pip install numpy pandas scikit-learn tqdm
pip install torch
pip install torch-geometric torch-scatter
pip install pygod
```

如果需要运行 `prepare\\\\\\\\\\\\\\\_star\\\\\\\\\\\\\\\_dataset.py` 中的 Node2Vec 特征生成模式，还需要安装：

```bash
pip install networkx gensim
```

注意：`torch-geometric` 和 `torch-scatter` 的安装方式与 PyTorch 版本、CUDA 版本有关。如果直接安装失败，建议根据本机的 PyTorch 和 CUDA 版本到 PyG 官方安装页面选择对应命令。

## 3\. 数据准备

### 3.1 JODIE 风格事件流 CSV 数据

`train\\\\\\\\\\\\\\\_event\\\\\\\\\\\\\\\_graph.py` 主要面向 Wikipedia、Reddit、MOOC 等事件流数据。默认数据路径为：

```text
data/mooc.csv
```

CSV 文件需要至少包含源节点、目标节点、时间戳和边特征等信息。代码中的 `event\\\\\\\\\\\\\\\_graph\\\\\\\\\\\\\\\_dataset.py` 会读取 CSV，将连续交互按照 `snap\\\\\\\\\\\\\\\_size` 划分为多个图快照，并利用训练快照的并集图训练 Node2Vec 作为节点特征。

常用数据放置方式如下：

```text
data/
├── wikipedia.csv
├── reddit.csv
└── mooc.csv
```

### 3.2 NPZ 数据

`train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py` 支持 `.npz` 格式动态图数据，默认路径为：

```text
data/DBLP3.npz
```

`.npz` 文件应包含以下字段：

|字段|含义|期望形状|
|-|-|-|
|`attmats`|节点属性矩阵序列|`\\\\\\\\\\\\\\\[N, T, F]`|
|`adjs`|邻接矩阵序列|`\\\\\\\\\\\\\\\[T, N, N]`|
|`labels`|原始标签，可选|代码中不直接作为异常标签训练|

其中，`N` 表示节点数，`T` 表示时间步数，`F` 表示节点特征维度。

## 4\. 快速运行

### 4.1 运行事件流数据实验

以 MOOC 数据为例：

```bash
python train\\\\\\\\\\\\\\\_event\\\\\\\\\\\\\\\_graph.py \\\\\\\\\\\\\\\\
  --dataset\\\\\\\\\\\\\\\_path data/mooc.csv \\\\\\\\\\\\\\\\
  --snap\\\\\\\\\\\\\\\_size 10000 \\\\\\\\\\\\\\\\
  --train\\\\\\\\\\\\\\\_ratio 0.7 \\\\\\\\\\\\\\\\
  --epochs 50 \\\\\\\\\\\\\\\\
  --lr 0.0001 \\\\\\\\\\\\\\\\
  --weight\\\\\\\\\\\\\\\_decay 1e-4 \\\\\\\\\\\\\\\\
  --device cuda
```

如果没有 GPU，可以将设备改为 CPU：

```bash
python train\\\\\\\\\\\\\\\_event\\\\\\\\\\\\\\\_graph.py --dataset\\\\\\\\\\\\\\\_path data/mooc.csv --device cpu
```

### 4.2 运行 DBLP 动态图实验

```bash
python train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py \\\\\\\\\\\\\\\\
  --dataset\\\\\\\\\\\\\\\_path data/DBLP3.npz \\\\\\\\\\\\\\\\
  --train\\\\\\\\\\\\\\\_ratio 0.7 \\\\\\\\\\\\\\\\
  --val\\\\\\\\\\\\\\\_ratio 0.15 \\\\\\\\\\\\\\\\
  --epochs 50 \\\\\\\\\\\\\\\\
  --lr 0.0005 \\\\\\\\\\\\\\\\
  --weight\\\\\\\\\\\\\\\_decay 0.0001 \\\\\\\\\\\\\\\\
  --h\\\\\\\\\\\\\\\_dim 64 \\\\\\\\\\\\\\\\
  --z\\\\\\\\\\\\\\\_dim 32 \\\\\\\\\\\\\\\\
  --wavelet\\\\\\\\\\\\\\\_C 2 \\\\\\\\\\\\\\\\
  --anomaly\\\\\\\\\\\\\\\_ratio 0.1 \\\\\\\\\\\\\\\\
  --device cuda
```

CPU 运行示例：

```bash
python train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py --dataset\\\\\\\\\\\\\\\_path data/DBLP3.npz --device cpu
```

## 5\. 主要参数说明

|参数|默认值|说明|
|-|-:|-|
|`--dataset\\\\\\\\\\\\\\\_path`|`data/mooc.csv` 或 `data/DBLP3.npz`|数据文件路径|
|`--snap\\\\\\\\\\\\\\\_size`|`10000`|CSV 事件流数据中每个快照包含的交互数|
|`--train\\\\\\\\\\\\\\\_ratio`|`0.7`|训练快照比例|
|`--val\\\\\\\\\\\\\\\_ratio`|`0.15`|验证快照比例，仅 `train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py` 显式提供|
|`--epochs`|`50`|训练轮数|
|`--lr`|`0.0001` 或 `0.0005`|学习率，不同入口默认值略有差异|
|`--weight\\\\\\\\\\\\\\\_decay`|`1e-4`|权重衰减|
|`--h\\\\\\\\\\\\\\\_dim`|`64`|隐藏层维度|
|`--z\\\\\\\\\\\\\\\_dim`|`64` 或 `32`|潜在表示维度|
|`--layer\\\\\\\\\\\\\\\_num`|`1`|BW-GRU 层数|
|`--wavelet\\\\\\\\\\\\\\\_C`|`2`|Beta 小波滤波器组阶数，仅 `train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py` 提供|
|`--anomaly\\\\\\\\\\\\\\\_ratio`|`0.1`|验证和测试阶段的异常注入比例，具体入口实现略有不同|
|`--struct\\\\\\\\\\\\\\\_clique\\\\\\\\\\\\\\\_size`|`5`|结构异常注入时的团大小|
|`--attr\\\\\\\\\\\\\\\_candidate\\\\\\\\\\\\\\\_k`|`50`|属性异常注入时的候选节点数量|
|`--random\\\\\\\\\\\\\\\_seed`|`7`|随机种子|
|`--device`|自动选择|`cuda` 或 `cpu`|

## 6\. 代码流程说明

### 6.1 数据处理流程

对于事件流 CSV 数据，代码流程如下：

1. 读取交互记录；
2. 根据 `snap\\\\\\\\\\\\\\\_size` 将连续交互划分为多个图快照；
3. 使用训练阶段的图结构训练 Node2Vec；
4. 将 Node2Vec 嵌入作为节点特征；
5. 训练集保持正常样本，验证和测试阶段按设定比例注入属性异常和结构异常；
6. 将每个时间步转换为 PyTorch Geometric 的 `Data` 对象。

对于 `.npz` 动态图数据，代码流程如下：

1. 读取 `attmats` 和 `adjs`；
2. 根据时间步划分训练集、验证集和测试集；
3. 对节点属性进行标准化；
4. 将邻接矩阵转换为 `edge\\\\\\\\\\\\\\\_index`；
5. 在验证和测试阶段注入异常；
6. 生成图快照序列并输入模型。

### 6.2 模型训练流程

模型主要包含以下部分：

1. `BetaWaveletFilter`：实现单个 Beta 小波图滤波器；
2. `BetaWaveletBank`：构建多尺度 Beta 小波滤波器组；
3. `BetaGraphGRUCell`：将 Beta 小波图卷积嵌入 GRU 的门控更新过程；
4. `Generative`：编码器，输出后验均值、后验标准差和潜在变量；
5. `TimeAwarePrior`：根据历史潜在变量生成时间感知先验；
6. `DualDecoder`：同时进行属性重构和结构重构；
7. `Model`：整合编码器、解码器、KL 项、重构损失和节点异常打分。

训练阶段优化目标主要由生成重构损失和 KL 散度组成：

```text
loss = gen\\\\\\\\\\\\\\\_loss + kl\\\\\\\\\\\\\\\_loss
```

验证和测试阶段，模型根据属性重构误差和结构重构误差得到节点异常分数。当前代码中，节点分数采用加权融合方式：

```text
score = 0.1 \\\\\\\\\\\\\\\* attribute\\\\\\\\\\\\\\\_score + 0.9 \\\\\\\\\\\\\\\* structure\\\\\\\\\\\\\\\_score
```

## 7\. 输出结果

训练过程中会输出每轮损失、验证 AUC 和最终测试 AUC。模型会根据验证集表现保存最佳参数，默认保存路径为：

```text
saved\\\\\\\\\\\\\\\_models/
```

事件流数据还会生成缓存，默认路径为：

```text
cache/
├── node2vec\\\\\\\\\\\\\\\_4/
└── pygod\\\\\\\\\\\\\\\_data\\\\\\\\\\\\\\\_4/
```

缓存用于避免重复训练 Node2Vec 和重复构造 PyGOD 注入后的数据。

## 8\. 常见问题

### 8.1 找不到数据文件

请确认运行命令中的 `--dataset\\\\\\\\\\\\\\\_path` 是否正确。例如：

```bash
python train\\\\\\\\\\\\\\\_dynamic\\\\\\\\\\\\\\\_graph.py --dataset\\\\\\\\\\\\\\\_path data/DBLP3.npz
```

如果使用 MOOC、Wikipedia 或 Reddit，需要先将对应 CSV 文件放入 `data/` 目录。

### 8.2 `torch\\\\\\\\\\\\\\\_scatter` 安装失败

`torch\\\\\\\\\\\\\\\_scatter` 需要与 PyTorch 和 CUDA 版本匹配。如果安装失败，建议先确认：

```bash
python -c "import torch; print(torch.\\\\\\\\\\\\\\\_\\\\\\\\\\\\\\\_version\\\\\\\\\\\\\\\_\\\\\\\\\\\\\\\_, torch.version.cuda)"
```

然后根据对应版本安装 PyG 相关依赖。

### 8.3 验证集或测试集 AUC 无法计算

AUC 需要同时包含正常节点和异常节点。如果某个划分中标签只有一种类别，代码会跳过 AUC 计算。可以适当调整 `--anomaly\\\\\\\\\\\\\\\_ratio`、`--train\\\\\\\\\\\\\\\_ratio` 或数据划分方式。

### 8.4 显存不足

结构解码部分会计算基于潜在变量的节点相似度矩阵，节点数较大时显存占用会明显增加。可以尝试：

```bash
--device cpu
```

或者降低节点数、快照规模、隐藏维度和潜在维度。

## 9\. 复现实验建议

为了保证结果尽量可复现，建议固定以下设置：

```bash
--random\\\\\\\\\\\\\\\_seed 7
--epochs 50
--train\\\\\\\\\\\\\\\_ratio 0.7
--val\\\\\\\\\\\\\\\_ratio 0.15
```

对于事件流数据，`event\\\\\\\\\\\\\\\_graph\\\\\\\\\\\\\\\_dataset.py` 中还固定了 Node2Vec 的训练参数，例如嵌入维度、随机游走长度、上下文窗口大小和训练轮数。若修改这些参数，需要删除旧缓存或更换缓存目录，避免读取旧结果。

