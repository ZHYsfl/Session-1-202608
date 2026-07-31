# SO-101 逆运动学：解析、数值、神经网络与混合法对比

实验9 —— 机械臂逆运动学解析与神经网络端到端实现方式对比。

## 一句话结论

在参数不准的3D打印SO-101机械臂上，对比四种IK方法，并用真机验证。
核心发现：**真机误差主体是机械误差（约34mm），与IK算法无关；
用关节空间闭环可将真机误差从34mm降到4mm（改善88%）。**

## 机械臂结构

SO-101 有6个电机，前5个参与IK：

| 关节 | 名称 | 作用 |
|------|------|------|
| 1 | shoulder_pan | 底座旋转 |
| 2 | shoulder_lift | 肩部抬升 |
| 3 | elbow_flex | 肘部弯曲 |
| 4 | wrist_flex | 腕部弯曲 |
| 5 | wrist_roll | 腕部旋转 |
| 6 | gripper | 夹爪（不参与IK） |

任务空间：(x, y, z, α, ψ) = 位置 + pitch + roll，5自由度。

## 四方法对比结果（同一批目标点）

| 方法 | 软件误差 | 真机开环 | 求解速度 | 说明 |
|------|---------|---------|---------|------|
| 解析IK | 243mm (4/50) | 无法测 | 0.01ms | 参数不准，基本解不出 |
| 数值IK | 0.9mm | 35.6mm | ~15ms | 软件最准但慢 |
| 神经网络 | 15mm | 35.9mm | 0.5ms | 最快，受数据量限制 |
| 混合法 | 0.9mm | 36.0mm | ~9ms | NN初值+数值微调，兼顾精度速度 |
| **关节闭环** | — | **4.0mm** | 多次迭代 | 编码器反馈补偿机械误差 |

## 核心流程

```
真机采集(600样本) → 数据转换 → 训练神经网络 → 四方法评估 → 真机验证 → 闭环补偿
collect_real_data   convert_real   train_network  evaluate_four  benchmark   closed_loop
```

## 关键文档

- [NEURAL_IK_PRINCIPLE.md](NEURAL_IK_PRINCIPLE.md) — 神经网络IK原理与数据转换公式推导（学习用）
- [数据来源与组长问题解答.md](数据来源与组长问题解答.md) — 训练数据来源澄清、各方法数据、常见疑问

## 目录结构

```
src/002/
├── kinematics/              # 运动学
│   ├── forward_kinematics_simple.py  # FK(含98mm把手偏移修复)
│   ├── analytical_ik.py              # 解析IK
│   └── numerical_ik.py               # 数值IK(阻尼最小二乘)
├── neural_network/          # 神经网络
│   ├── model.py                      # MLP模型+多分量损失
│   ├── dataset.py                    # 数据集
│   └── trainer.py                    # 训练器
├── utils/                   # 工具(配置/评估/数学/可视化)
├── configs/robot_config.yaml         # 机器人配置
├── models/                  # 训练产物
│   ├── ik_mlp_best.pth               # 最佳模型
│   └── training_history.json         # 训练曲线
├── data/ik_dataset_aug10.h5          # 训练数据集
└── real_robot_dataset.h5             # 真机采集原始数据
```

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `calibrate_table_height.py` | 桌面高度标定（防撞） |
| `verify_safe_height.py` | 静态安全高度验证 |
| `collect_real_data.py` | 真机数据采集（含安全防撞） |
| `convert_real_data.py` | 采集数据→训练格式转换 |
| `train_network.py` | 训练神经网络 |
| `evaluate_four_methods.py` | 四方法软件层评估 |
| `benchmark_real_robot.py` | 四方法真机统一对比 |
| `validate_ik_reaching.py` | IK验证(重复性/往返/真机到达) |
| `closed_loop_joint.py` | 关节空间闭环补偿 |
| `test_fk_validation.py` | FK真机验证 |

## 环境依赖

```bash
pip install numpy pyserial pyyaml h5py torch
```

## 安全说明

机械臂夹在桌沿、末端有98mm把手，采集/验证脚本均内置：
- 工作空间地板保护（末端Z≥192mm，基于桌面标定值66mm）
- 每次动作先经HOME中转（防路径扫桌）
- 低速运动 + 关节范围收窄
