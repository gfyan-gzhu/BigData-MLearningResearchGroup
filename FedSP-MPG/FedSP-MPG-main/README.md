# FedSP-MPG

本项目是在 FedHGN 联邦异构图学习框架基础上扩展的 FedSP-MPG 实现，主要面向**联邦异构图节点分类任务**。项目在原有异构图神经网络编码器和联邦聚合流程上，引入了元路径视图构建、元路径视图编码以及节点级语义路径门控机制，用于提升不同客户端异构图结构下的表征学习能力。

---

## 1. 项目结构

```text
FedSP-MPG/
├── main.py                    # 项目入口：解析参数、加载配置、选择训练框架并运行实验
├── configs.yaml               # 数据集、模型和联邦框架的默认配置
├── FedSPMPG.py                # FedSP-MPG 主体：客户端、服务器、训练、聚合、评估与保存
├── mp_modules_addon.py        # 元路径枚举、元路径视图构建、视图编码器、节点级门控模块
├── HGNModel.py                # 异构图神经网络底座，目前主要实现 RGCN
├── Decoders.py                # 节点分类解码器
├── utils.py                   # 配置加载、数据读取、路径生成、随机种子、评估与结果保存
├── prepare_data.ipynb         # 数据预处理脚本，用于生成联邦划分后的图数据
├── collect_results.py         # 扫描 saves/ 下的 results.txt，并汇总实验结果
├── run_exp.sh                 # 超参数实验与消融实验批量运行脚本
├── run_fedspmpg_all.sh        # FedSP-MPG 多数据集/多客户端/多随机种子批量运行脚本
├── README.md                  # 项目说明文档
├── __pycache__/               # Python 自动生成缓存目录，可删除
└── baselines/                 # baseline 方法及其依赖文件
    ├── Central.py
    ├── FedAvg.py
    ├── Local.py
    ├── FedHGN.py
    ├── RGCN.py
    └── Decoders.py
```


---

## 2. 核心文件说明

### 2.1 入口文件

| 文件 | 作用 |
|---|---|
| `main.py` | 项目运行入口，负责解析命令行参数、加载配置、选择具体框架并执行训练与测试。 |
| `configs.yaml` | 存放数据集路径、模型参数、联邦训练参数和 FedSP-MPG 相关超参数。 |

### 2.2 FedSP-MPG 方法文件

| 文件 | 作用 |
|---|---|
| `FedSPMPG.py` | FedSP-MPG 的核心实现，包含客户端本地训练、服务器聚合、元路径门控分支、消融实验逻辑和 checkpoint 保存/加载。 |
| `mp_modules_addon.py` | FedSP-MPG 新增模块，包括元路径枚举、元路径视图构建、MPViewEncoder、DecoupledNodeWiseMPGating 和熵计算。 |
| `HGNModel.py` | 异构图神经网络编码器，当前主要支持 RGCN。 |
| `Decoders.py` | 节点分类任务的线性分类器。 |
| `utils.py` | 数据加载、配置合并、保存路径生成、随机种子设置、指标计算和结果保存等通用函数。 |

### 2.3 实验辅助文件

| 文件 | 作用 |
|---|---|
| `prepare_data.ipynb` | 将原始异构图数据处理成联邦客户端划分后的 `.bin` 文件。 |
| `run_fedspmpg_all.sh` | 批量运行 FedSP-MPG 主实验。 |
| `run_exp.sh` | 批量运行超参数实验和消融实验。 |
| `collect_results.py` | 汇总 `saves/` 目录下的实验结果，生成 CSV 统计表。 |

### 2.4 baseline 目录

`baselines/` 中保留了对比方法及其依赖文件，包括 `Central`、`FedAvg`、`Local`、`FedHGN` 等。

---

## 3. 环境依赖

建议使用 Python 3.9 左右的环境。主要依赖如下：

```text
python >= 3.9
torch
dgl
numpy
scipy
scikit-learn
pyyaml
tqdm
pandas
```

可使用如下命令安装常用依赖：

```bash
pip install torch dgl numpy scipy scikit-learn pyyaml tqdm pandas
```

如果使用 GPU，请根据自己的 CUDA 版本安装对应的 PyTorch 和 DGL 版本。

---

## 4. 数据准备

项目默认支持三个节点分类数据集：

```text
AIFB
MUTAG
BGS
```

数据路径在 `configs.yaml` 中配置，例如：

```yaml
AIFB:
  cname: "aifb-hetero"
  path: "./data/aifb-hetero"
```

运行实验前，需要先通过 `prepare_data.ipynb` 生成预处理后的联邦划分图数据。程序读取数据时，默认会在对应数据目录下查找如下格式的文件：

```text
{dataset_cname}_{split_strategy}_{num_clients}.bin
```

例如：

```text
./data/aifb-hetero/aifb-hetero_random-edges_5.bin
./data/aifb-hetero/aifb-hetero_random-etypes_5.bin
```

其中：

| 参数 | 含义 |
|---|---|
| `random-edges` | 按边随机划分客户端数据 |
| `random-etypes` | 按边类型划分客户端数据 |
| `num_clients` | 客户端数量，例如 3、5、10 |

---

## 5. 单次实验运行

### 5.1 运行 FedSP-MPG

以 AIFB 数据集、edges 划分、5 个客户端、GPU 0 为例：

```bash
python main.py \
  -d AIFB \
  -s edges \
  -f FedSP-MPG \
  -m RGCN \
  -c 5 \
  -g 0 \
  --random-seed 1000
```

如果使用 CPU：

```bash
python main.py -d AIFB -s edges -f FedSP-MPG -m RGCN -c 5 -g -1 --random-seed 1000
```

### 5.2 运行 baseline

运行 FedAvg：

```bash
python main.py -d AIFB -s edges -f FedAvg -m RGCN -c 5 -g 0 --random-seed 1000
```

运行 Local：

```bash
python main.py -d AIFB -s edges -f Local -m RGCN -c 5 -g 0 --random-seed 1000
```

运行 Central：

```bash
python main.py -d AIFB -s edges -f Central -m RGCN -g 0 --random-seed 1000
```

---

## 6. 常用命令行参数

| 参数 | 示例 | 说明 |
|---|---|---|
| `-d`, `--dataset` | `AIFB` | 数据集名称，可选 `AIFB`、`MUTAG`、`BGS`。 |
| `-s`, `--split-strategy` | `edges` | 数据划分方式，可选 `edges` 或 `etypes`。程序内部会转换为 `random-edges` 或 `random-etypes`。 |
| `-f`, `--framework` | `FedSP-MPG` | 训练框架，可选 `FedSP-MPG`、`FedHGN`、`FedAvg`、`FedProx`、`Local`、`Central`。 |
| `-m`, `--model` | `RGCN` | 异构图模型，目前主要使用 `RGCN`。 |
| `-c`, `--num-clients` | `5` | 客户端数量。 |
| `-g`, `--gpu` | `0` | GPU 编号；设置为 `-1` 时使用 CPU。 |
| `--random-seed` | `1000` | 随机种子。 |
| `--config-path` | `./configs.yaml` | 配置文件路径。 |
| `--exp-folder` | `normal` | 实验结果保存类别，可选 `normal`、`chaocan`、`xiaorong`。 |

---

## 7. FedSP-MPG 相关参数

FedSP-MPG 的默认参数主要在 `configs.yaml` 中设置：

```yaml
FedSP-MPG:
  align_reg: 0.5
  mp_max_len: 3
  mp_max_paths: 16
  mp_view_num_layers: 1
  mp_view_dropout: 0.1
  mp_num_gating_bases: 8
  lambda_stb: 0.0
```

| 参数 | 含义 |
|---|---|
| `mp_max_len` | 元路径最大长度。 |
| `mp_max_paths` | 每个客户端最多保留的元路径数量。 |
| `mp_view_num_layers` | 元路径视图编码器层数。 |
| `mp_view_dropout` | 元路径视图编码器 dropout。 |
| `mp_num_gating_bases` | 元路径门控基向量数量。 |
| `lambda_stb` | 语义熵稳定项权重。 |
| `align_reg` | FedHGN 底座中的对齐正则项权重。 |

也可以在命令行中覆盖部分 FedSP-MPG 超参数：

```bash
python main.py \
  -d AIFB \
  -s edges \
  -f FedSP-MPG \
  -c 5 \
  -g 0 \
  --random-seed 1000 \
  --mp-num-gating-bases 8 \
  --mp-max-paths 16
```

---

## 8. 消融实验

FedSP-MPG 当前支持以下消融设置：

| 消融项 | 含义 |
|---|---|
| `no_mp` | 去掉元路径分支，仅使用基础 HGN 表征。 |
| `uniform` | 使用均匀权重融合不同元路径视图。 |
| `static` | 使用客户端级静态元路径权重。 |
| `no_residual` | 去掉元路径融合后的残差连接。 |

示例：

```bash
python main.py \
  -d MUTAG \
  -s edges \
  -f FedSP-MPG \
  -a no_mp \
  -c 5 \
  -g 0 \
  --random-seed 1000 \
  --exp-folder xiaorong
```

---

## 9. 批量实验脚本

### 9.1 主实验脚本

`run_fedspmpg_all.sh` 用于批量运行 FedSP-MPG 主实验。运行方式：

```bash
bash run_fedspmpg_all.sh
```

该脚本中可以修改：

```bash
DATASETS=("AIFB" "BGS" "MUTAG")
SPLITS=("edges" "etypes")
CLIENTS=(3 5 10)
SEEDS=(1000 2000 3000)
GPU=0
```

当前脚本截图版本中默认主要运行 MUTAG，可根据需要自行打开其他数据集。

### 9.2 超参数与消融实验脚本

`run_exp.sh` 用于运行两类实验：

1. FedSP-MPG 超参数实验：
   - `mp_num_gating_bases`
   - `mp_max_paths`
2. FedSP-MPG 消融实验：
   - `no_mp`
   - `uniform`
   - `static`
   - `no_residual`

运行方式：

```bash
bash run_exp.sh
```

---

## 10. 结果保存路径

实验结果默认保存在 `saves/` 目录下。路径格式为：

```text
saves/
└── {exp_folder}/
    └── {framework_or_framework_ablation}/
        └── {model}/
            └── {dataset}_{split_strategy}_{num_clients}_seed{seed}_mpb{mp_num_gating_bases}_mpp{mp_max_paths}/
                └── {run_id}/
                    ├── results.txt
                    ├── configs.yaml
                    ├── server_encoder.pt
                    ├── server_decoder.pt
                    ├── client_0_encoder.pt
                    ├── client_0_decoder.pt
                    └── ...
```

例如：

```text
saves/normal/FedSP-MPG/RGCN/AIFB_random-edges_5_seed1000_mpb8_mpp16/1/results.txt
```

`results.txt` 中通常包含：

```text
accuracy    macro-f1    micro-f1    entropy
```

---

## 11. 结果汇总

完成实验后，可以使用 `collect_results.py` 汇总结果：

```bash
python collect_results.py \
  --saves-root ./saves \
  --out-dir ./summary \
  --fixed-gating-bases 8 \
  --fixed-max-paths 16
```

该脚本会自动扫描 `saves/` 下的 `results.txt`，并生成结果表。常见输出包括：

```text
summary/all_results_raw.csv
summary/chaocan_mpb_summary.csv
summary/chaocan_mpp_summary.csv
summary/xiaorong_long.csv
summary/xiaorong_accuracy_table.csv
summary/xiaorong_macro-f1_table.csv
summary/xiaorong_micro-f1_table.csv
summary/xiaorong_entropy_table.csv
```

---

## 12. 常见问题

### 找不到数据文件

如果报错提示找不到 `.bin` 文件，请先确认：

1. 是否已经运行 `prepare_data.ipynb`；
2. `configs.yaml` 中的数据路径是否正确；
3. 数据文件命名是否符合如下格式：

```text
{dataset_cname}_{split_strategy}_{num_clients}.bin
```

例如：

```text
aifb-hetero_random-edges_5.bin
```

```

---

## 13. 推荐运行流程

```text
1. 配置 Python 环境并安装依赖
2. 使用 prepare_data.ipynb 生成联邦划分后的数据
3. 运行单次实验，确认代码和数据路径无误
4. 使用 run_fedspmpg_all.sh 批量运行主实验
5. 使用 run_exp.sh 批量运行超参数实验和消融实验
6. 使用 collect_results.py 汇总结果
7. 根据 summary/ 下的 CSV 文件整理论文表格
```

---

## 14. 参考说明： FedSP-MPG 模块，用于在客户端私有异构图结构下进行元路径视图构建、路径语义编码和节点级动态门控融合。
