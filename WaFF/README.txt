# 基于方向搜索算法(DSSD)的模型融合工具

本项目实现了**带步长衰减的方向搜索(Directional Search with Step Decay, DSSD)** 模型融合算法。首先在不同数据集上预训练多个专用子模型，然后通过方向搜索策略自动寻找最优融合权重，将多个子模型的优势进行整合，最终提升目标任务的模型性能。

## 📋 项目概述
### 核心流程
1. **子模型预训练**：通过`modelbase#1.py`至`modelbase#4.py`，在不同领域数据集上训练4个专用子模型（包括掩码语言模型、序列分类模型等）。
2. **权重搜索与模型融合**：通过`dirsearch.py`（方向搜索核心）、`getscore.py`（融合效果评分）和`inifusion.py`（融合主程序）自动搜索子模型的最优权重组合，完成模型融合并评估效果。

### 核心特性
- 带步长衰减的方向搜索策略，平衡搜索效率与精度
- 权重约束机制（0 ≤ 权重 ≤ 1），保证融合结果的可行性
- 实时分数日志记录，根据搜索进度自动调整步长
- 支持时间阈值和步长阈值双重终止条件，灵活控制融合过程

## 🛠️ 环境依赖
运行代码前请安装所需依赖：
```bash
pip install torch transformers pandas numpy scikit-learn datasets evaluate matplotlib tqdm
```
- 代码默认使用GPU进行训练和推理，请确保CUDA环境可用；若无GPU，可将代码中`device = torch.device("cuda")`修改为`device = torch.device("cpu")`
- Python版本：3.8+（已在3.8/3.9/3.10版本测试通过）

## 📂 项目结构
| 文件名称            | 功能描述                                                                 |
|---------------------|--------------------------------------------------------------------------|
| `modelbase#1.py`    | 在`FinancialNews.csv`金融新闻数据集上预训练掩码语言模型(MLM)              |
| `modelbase#2.py`    | 在`FinancialComments.csv`金融评论数据集上预训练掩码语言模型(MLM)          |
| `modelbase#3.py`    | 在`AllNews.csv`新闻数据集上预训练10分类序列分类模型                       |
| `modelbase#4.py`    | 在`DBReviews.csv`评论数据集上预训练5分类序列分类模型                      |
| `dirsearch.py`      | 实现前向/后向方向搜索逻辑，用于权重优化                                   |
| `getscore.py`       | 根据给定权重融合模型，并在目标数据集上微调计算准确率（融合效果评分）       |
| `inifusion.py`      | 模型融合主入口：控制步长衰减、搜索循环和运行时间约束                       |

## 🚀 使用步骤

### 步骤1：数据集准备
准备以下数据集并放置在项目根目录：
- `FinancialNews.csv`：金融新闻数据，用于modelbase#1的MLM训练
- `FinancialComments.csv`：金融评论数据，用于modelbase#2的MLM训练
- `AllNews.csv`：新闻分类数据（10个类别），用于modelbase#3的序列分类训练
- `DBReviews.csv`：评论分类数据（5个类别），用于modelbase#4的序列分类训练
- `fine-tuning.csv`：目标任务数据集，用于融合模型的微调与评分计算

> 提示：可根据实际数据情况，修改`modelbase#*.py`中的数据集路径和划分比例。

### 步骤2：预训练子模型
依次运行以下脚本训练子模型（运行顺序无要求）：
```bash
# 训练掩码语言模型
python modelbase#1.py
python modelbase#2.py

# 训练序列分类模型
python modelbase#3.py
python modelbase#4.py
```
- 训练完成的子模型将保存至`./Model_<基础模型名>_EPT#<编号>/`目录（例如`./Model_bert-base-chinese_EPT#1/`）
- 自动生成训练损失曲线（图片）和损失日志（`Loss_*.csv`），用于监控训练过程

### 步骤3：配置融合参数
修改`inifusion.py`中以下未填充的关键参数（原代码中标记为`_`）：
```python
# 方向搜索的最小步长（推荐值：0.001）
MIN_step_size = 0.001
# 融合过程最大运行时间（推荐值：2小时）
MAX_exec_time = timedelta(hours=2)
```
其他可调整的核心参数：
- `step_size`：初始搜索步长（默认：0.1）
- `step_size_reduction_coefficient`：步长衰减系数（默认：0.5）
- `coefficient_decay_rate`：步长系数的衰减率（默认：0.2）

### 步骤4：运行模型融合
执行融合主程序：
```bash
python inifusion.py
```
#### 融合过程说明：
1. 初始化权重：`w_0=1.0, w_1=w_2=w_3=w_4=0.0`（初始仅使用第一个子模型）
2. 对每个步长，分别执行前向和后向方向搜索，寻找更优权重组合
3. 若当前步长下未找到更优权重，则减小步长继续搜索
4. 当步长小于`MIN_step_size`或运行时间超过`MAX_exec_time`时，终止搜索并输出最优权重

## 📊 输出文件说明
| 文件名称                  | 描述                                                                 |
|---------------------------|----------------------------------------------------------------------|
| `fusionModel.pt`          | 融合后的模型参数文件（评分计算过程中的临时文件）                       |
| `getScore_log.txt`        | 融合过程中所有权重组合及其对应准确率的汇总日志                         |
| `getScore_log_01_05__02.txt` | 每个权重组合的详细评分日志，包含精确到8位小数的准确率值             |
| `Model_<基础模型名>_EPT#<编号>/` | 预训练完成的子模型文件（来自modelbase#*.py）                     |
| `Loss_*.csv`              | 子模型训练过程的损失日志，包含轮次、迭代次数、损失值、困惑度/准确率等 |

## ⚠️ 注意事项
1. **基础模型替换**：将`modelbase#*.py`和`getscore.py`中的`'model'`替换为你使用的基础模型检查点（例如`'bert-base-chinese'`、`'roberta-base'`）。
2. **GPU设备指定**：修改`getscore.py`和`modelbase#2.py`中的`os.environ["CUDA_VISIBLE_DEVICES"]`，指定要使用的GPU编号（例如`"0"`表示使用第一块GPU）。
3. **数据集大小调整**：原代码中使用了数据下采样（例如modelbase#4.py中的`DATA.sample(500_000)`），请根据你的硬件资源调整采样大小。
4. **评分指标修改**：当前融合评分使用分类准确率，若需要其他指标（如F1值、精确率），请修改`getscore.py`中的指标计算部分。
5. **运行时间预估**：融合过程可能需要数小时（取决于步长设置和硬件性能），请合理设置`MAX_exec_time`避免运行时间过长。

## 📈 核心算法逻辑
### 方向搜索模块（dirsearch.py）
- 通过预定义的`masks`生成1-4维的权重变化组合
- 使用`isFeasible()`函数检查权重是否在[0,1]范围内
- 分别执行前向（`F`）和后向（`B`）搜索，调整权重并寻找最优组合

### 评分计算模块（getscore.py）
- 通过参数加权求和的方式融合子模型（`fuseModels()`函数）
- 在`fine-tuning.csv`数据集上微调融合模型，并在测试集上计算准确率
- 使用`scoresDB`缓存已计算的权重分数，避免重复计算