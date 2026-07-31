# Imitation Learning Experiments

模仿学习综合实验项目，涵盖自动驾驶行为克隆、DAgger 分布偏移修正、定量性能分析，以及基于 RoboMimic 官方数据集的机器人操作策略比较。

## 项目概述

本项目按照“行为克隆验证、分布偏移修正、机理分析、机器人任务迁移”的逻辑组织四个实验。实验一使用卷积神经网络从驾驶图像回归转向角；实验二在 CartPole 任务上比较 Expert、Behavior Cloning（BC）与 DAgger；实验三从回报、成功率、动作误差和状态覆盖等角度解释 DAgger 的改进来源；实验四使用 RoboMimic v1.5 官方低维 PH 数据集，对 BC、BC-RNN 和 BC-Transformer 进行统一的离线动作预测比较。

### 核心功能

- **自动驾驶行为克隆**：保留 CNN 训练代码、已训练模型、损失曲线和完整实验指标。
- **DAgger 复现**：实现专家策略查询、策略混合、数据聚合、迭代训练和 80 回合独立评估。
- **消融与误差分析**：比较不同 beta 衰减系数，并分析动作预测误差和策略访问状态分布。
- **官方 RoboMimic 数据实验**：覆盖 Lift、Can、Square 和 Transport 四个任务，比较三类策略结构。
- **论文级可视化**：提供学习曲线、成功率、状态覆盖、误差分解和模型比较图。
- **机器可读结果**：关键指标以 JSON、NPZ、Joblib 和 PyTorch checkpoint 形式保存。

## 项目结构

```text
src/
├── imation_learning/
│   ├── Experiment1_Behavior_Cloning/
│   │   ├── behavioral-cloning/       # CNN 训练、推理代码及 model.h5
│   │   ├── results/                  # 实验一指标与损失曲线
│   │   └── verify_experiment.py
│   ├── Experiment2_DAgger/
│   │   ├── dagger_cartpole.py        # BC、DAgger 与 beta 消融实验
│   │   ├── requirements.txt
│   │   └── results/
│   ├── Experiment3_Performance_Analysis/
│   │   ├── analyze_experiments.py    # 综合定量分析与绘图
│   │   ├── requirements.txt
│   │   └── results/
│   └── Experiment4_RoboMimic/
│       ├── official_benchmark.py      # 官方数据离线基准
│       ├── download_official_datasets.py
│       ├── data/official/             # RoboMimic v1.5 HDF5 数据
│       └── results/                   # 指标、训练历史与模型权重
├── images/                            # 实验报告图片
├── reports_completed.md               # 四个实验的完整报告
├── REPRODUCTION.md                    # 详细复现说明
└── README.md
```

> 目录名 `imation_learning` 为当前提交版中的实际名称，运行命令需保持这一拼写。

## 实验组成

| 实验 | 任务 | 主要方法 | 主要输出 |
| --- | --- | --- | --- |
| 实验一 | 自动驾驶转向角预测 | CNN Behavior Cloning | `model.h5`、训练/验证 MSE、损失曲线 |
| 实验二 | CartPole 控制 | Expert、BC、DAgger、beta 消融 | 回报、成功率、数据集增长、状态轨迹 |
| 实验三 | 性能与机理分析 | 动作误差、状态覆盖、策略分布比较 | 汇总 JSON、柱状图、散点图 |
| 实验四 | 机器人操作动作预测 | BC、BC-RNN、BC-Transformer | 四任务 MSE、置信区间、训练曲线、checkpoint |

## 数据集

| 实验 | 数据来源 | 规模 | 输入/输出 |
| --- | --- | --- | --- |
| 实验一 | Udacity 自动驾驶模拟器采集记录 | 7,698 条驾驶记录 | `32 x 128 x 3` 图像 / 连续转向角 |
| 实验二 | CartPole 专家策略在线标注 | 初始 1,000 条，DAgger 后 30,701 条 | 4 维状态 / 二值动作 |
| 实验四 Lift | RoboMimic v1.5 PH low-dim | 200 条轨迹，9,666 个样本 | 19 维观测 / 7 维动作 |
| 实验四 Can | RoboMimic v1.5 PH low-dim | 200 条轨迹，23,207 个样本 | 23 维观测 / 7 维动作 |
| 实验四 Square | RoboMimic v1.5 PH low-dim | 200 条轨迹，30,154 个样本 | 23 维观测 / 7 维动作 |
| 实验四 Transport | RoboMimic v1.5 PH low-dim | 200 条轨迹，93,752 个样本 | 59 维观测 / 14 维动作 |

实验四使用官方 HDF5 文件及其训练/验证掩码，数据清单记录了文件大小和 SHA-256 校验值。数据来源见 [RoboMimic v1.5 数据说明](https://robomimic.github.io/docs/datasets/robomimic_v1.5.html) 和 [官方 Hugging Face 数据仓库](https://huggingface.co/datasets/robomimic/robomimic_datasets)。

## 环境要求

- Windows 11
- Python 3.12.13
- NumPy 2.3.5
- Matplotlib 3.11.1
- scikit-learn 1.9.0
- PyTorch 2.12.1（CPU）和 h5py 3.15.1
- TensorFlow 2.15.0（仅实验一完整重训练需要）

实验二和实验四固定随机种子为 `20260730`。实验四已包含隔离依赖目录 `.deps`，也可通过安装脚本重新配置。

## 安装与使用

以下命令均在 `src` 目录下执行。

### 1. 安装实验二、三依赖

```powershell
python -m pip install -r .\imation_learning\Experiment2_DAgger\requirements.txt
```

### 2. 配置实验四环境

```powershell
.\imation_learning\Experiment4_RoboMimic\setup_official_benchmark.ps1
```

如果 `data\official` 中缺少 HDF5 文件，可重新下载并校验：

```powershell
python .\imation_learning\Experiment4_RoboMimic\download_official_datasets.py
```

### 3. 运行实验二 DAgger

```powershell
python .\imation_learning\Experiment2_DAgger\dagger_cartpole.py
```

快速检查：

```powershell
python .\imation_learning\Experiment2_DAgger\dagger_cartpole.py --quick
```

### 4. 运行实验四官方数据离线基准

```powershell
python .\imation_learning\Experiment4_RoboMimic\official_benchmark.py
```

快速检查：

```powershell
python .\imation_learning\Experiment4_RoboMimic\official_benchmark.py --quick
```

### 5. 生成实验三综合分析

实验三依赖实验一、二、四的结果文件，应在实验二和实验四完成后运行：

```powershell
python .\imation_learning\Experiment3_Performance_Analysis\analyze_experiments.py
```

### 6. 实验一完整重训练

当前提交包含已训练模型和结果，但不包含原始 Udacity 驾驶图像。完整重训练需要将 `driving_log.csv` 和 `IMG` 目录放入实验一数据目录，然后执行：

```powershell
cd .\imation_learning\Experiment1_Behavior_Cloning\behavioral-cloning
python model.py
```

## 算法说明

### Behavior Cloning

行为克隆将模仿学习转化为监督学习。给定专家数据集

$$
\mathcal{D}_E=\{(s_i,a_i)\}_{i=1}^{N},
$$

策略通过最小化预测动作与专家动作之间的损失进行训练：

$$
\theta^\ast=\arg\min_\theta\frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}\left(\pi_\theta(s_i),a_i\right).
$$

实验一对连续转向角使用均方误差；实验二对离散动作训练概率分类器；实验四对机器人连续动作使用标准化后的均方误差。

### DAgger

单纯 BC 只在专家状态分布上训练，部署时的微小误差可能使策略进入训练集未覆盖的状态。DAgger 在第 \(k\) 轮执行混合策略

$$
\pi_k=\beta_k\pi_E+(1-\beta_k)\pi_{\theta_k},
$$

并让专家对策略访问到的新状态重新标注，再聚合数据：

$$
\mathcal{D}_{k+1}=\mathcal{D}_k\cup
\{(s,\pi_E(s)):s\sim d_{\pi_k}\}.
$$

本项目比较 `0.9`、`0.5` 和 `0.1` 三种 beta 衰减，并记录每轮回报、成功率和数据集规模。

### 序列策略

实验四除单步 BC 外，还实现了基于历史窗口的 BC-RNN 和 BC-Transformer。二者使用长度为 10 的观测序列预测当前动作，用于检验时序建模能否改善不同机器人操作任务上的离线动作拟合。

## 实验结果

### 实验一：CNN Behavior Cloning

| 指标 | 结果 |
| --- | ---: |
| 训练样本数 | 6,158 |
| 验证样本数 | 1,540 |
| 模型参数量 | 972,225 |
| 最终训练 MSE | 0.0684 |
| 最终验证 MSE | 0.0144 |

![实验一训练损失](images/exp1_loss.png)

### 实验二：BC 与 DAgger

| 方法 | 平均回报 | 成功率 |
| --- | ---: | ---: |
| Expert | 500.00 | 100% |
| BC | 309.35 | 20% |
| DAgger | 500.00 | 100% |

DAgger 相对 BC 的平均回报提高 `190.65`，成功率提高 `80` 个百分点。聚合数据集由 `1,000` 条增长到 `30,701` 条，状态位置 \(x\) 的 1%--99% 覆盖宽度扩大约 `13.96` 倍。

![DAgger 学习曲线](images/exp2_learning_curve.png)

### 实验三：动作误差与状态分布

在 30,000 个独立策略访问状态上，BC 的动作概率 MSE 为 `0.221315`，DAgger 降至 `0.006405`，相对下降 `97.11%`。状态分布图显示，DAgger 采集的数据覆盖了 BC 部署时可能进入的偏移状态。

![策略状态分布](images/exp3_policy_state_distribution.png)

![综合性能分析](images/exp3_summary.png)

### 实验四：RoboMimic 官方数据离线基准

下表为四个任务上的轨迹平均动作 MSE，数值越低越好。

| 模型 | Lift | Can | Square | Transport | 四任务平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC | 0.02455 | 0.03302 | 0.03865 | 0.02812 | 0.03109 |
| BC-RNN | 0.02938 | 0.03432 | 0.03913 | 0.03266 | 0.03387 |
| BC-Transformer | **0.02074** | **0.02859** | **0.03240** | **0.02349** | **0.02631** |

BC-Transformer 在四个任务上均取得最低离线动作预测 MSE，四任务平均误差较 BC 下降约 `15.38%`。

![RoboMimic 模型比较](images/exp4_official_model_comparison.png)

## 主要发现

1. **BC 能完成专家行为拟合，但存在部署分布偏移。** CartPole 中 BC 的成功率仅为 20%，说明专家数据上的监督拟合不能直接保证闭环控制稳定性。
2. **DAgger 显著改善策略鲁棒性。** 聚合策略访问状态并添加专家标签后，成功率提升到 100%，动作概率 MSE 同时下降 97.11%。
3. **状态覆盖解释了性能提升来源。** DAgger 扩大了训练数据在关键状态维度上的覆盖范围，使模型能处理自身误差产生的偏移状态。
4. **序列建模在机器人离线动作预测中有效。** BC-Transformer 在 Lift、Can、Square 和 Transport 上均优于单步 BC 与 BC-RNN。

## 复现边界

- 实验一原始驾驶图像未随当前提交提供，因此默认复现范围是校验已有代码、模型、指标和图表；不能据此声称已重新训练该模型。
- 实验四使用 RoboMimic 官方 v1.5 PH low-dim 数据，但评估指标是离线动作预测 MSE，不包含 MuJoCo 环境 rollout，因而不报告机器人任务成功率。
- 完整配置、随机种子、指标定义和产物位置见 [REPRODUCTION.md](REPRODUCTION.md)。

## 实验报告

四个实验的设计、实现、结果和论文写作分析已整合到 [reports_completed.md](reports_completed.md)。原始 `reports.md` 未被修改。

## 参考资料

1. Pomerleau, D. A. *ALVINN: An Autonomous Land Vehicle in a Neural Network*. NeurIPS, 1989.
2. Bojarski, M. et al. *End to End Learning for Self-Driving Cars*. arXiv:1604.07316, 2016.
3. Ross, S., Gordon, G. J., and Bagnell, J. A. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html). AISTATS, 2011.
4. Mandlekar, A. et al. [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298). CoRL, 2021.
5. [Udacity Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
6. [RoboMimic Documentation](https://robomimic.github.io/)

## 许可证

上游数据集、框架和第三方组件遵循各自许可证。本目录中的实验代码与结果用于课程学习和研究复现。
