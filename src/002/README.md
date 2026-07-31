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

## 四方法对比结果（N=50 同一批目标点，远初值 σ=0.3rad）

数字来自可复现脚本 `scripts/evaluation/evaluate_four_methods.py`（结果存 `four_methods_eval.json`）。

| 方法 | 软件误差 | 真机开环 | 求解速度 | 说明 |
|------|---------|---------|---------|------|
| 解析IK | 243mm (4/50) | 无法测 | 0.02ms | 用标称参数，几乎解不出 |
| 数值IK | 0.9mm | 35.6mm | ~33ms | 软件最准但慢 |
| 神经网络 | 17mm | 35.9mm | 0.8ms | 最快，受600样本数据量限制 |
| 混合法(3步) | 15mm | 36.0mm | ~6ms | NN初值+3步微调；步数是精度/延迟旋钮 |
| **关节闭环** | — | **4.0mm** | 多次迭代 | 编码器反馈补偿机械误差 |

> 混合法精度随微调步数变化：远初值下3步仅~15mm，到10步中位数才降到0.9mm
> （见 `scripts/evaluation/ablation_hybrid_steps.py` → `ablation_hybrid_steps.json`）。

## 核心流程

```
真机采集(600样本) → 数据转换 → 训练神经网络 → 四方法评估 → 真机验证 → 闭环补偿
collect_real_data   convert_real   train_network  evaluate_four  benchmark   closed_loop
```

## 关键文档

- [docs/NEURAL_IK_PRINCIPLE.md](docs/NEURAL_IK_PRINCIPLE.md) — 神经网络IK原理与数据转换公式推导（学习用）
- [docs/数据来源与组长问题解答.md](docs/数据来源与组长问题解答.md) — 训练数据来源澄清、各方法数据、常见疑问
- `docs/*.excalidraw` — 3张白板图：IK原理图、IK数学推导图（数形结合）、调试历程图
- 论文：[../../paper/002/main.tex](../../paper/002/main.tex)

## 目录结构

```
src/002/
├── kinematics/              # 运动学(可导入包)
│   ├── forward_kinematics_simple.py  # FK(含98mm把手偏移+关节链修复)
│   ├── analytical_ik.py              # 解析IK
│   └── numerical_ik.py               # 数值IK(阻尼最小二乘)
├── neural_network/          # 神经网络(可导入包)
│   ├── model.py / dataset.py / trainer.py
├── utils/                   # 工具(config_loader / math_utils)
├── robot_control/           # 舵机底层(扫描/连接测试)
├── scripts/                 # 可执行脚本(按功能分类)
│   ├── data_collection/     #   collect_real_data / convert_real_data
│   ├── training/            #   train_network
│   ├── evaluation/          #   evaluate_four_methods / benchmark_real_robot
│   │                        #   validate_ik_reaching / ablation_hybrid_steps
│   │                        #   closed_loop_joint
│   └── calibration/         #   calibrate_table_height / verify_safe_height / test_fk_validation
├── docs/                    # 文档+3张excalidraw图+图生成器
├── configs/robot_config.yaml
├── models/                  # ik_mlp_best.pth, so101_new_calib.urdf, training_history.json
├── data/ik_dataset_aug10.h5          # 训练数据集(6000样本)
├── real_robot_dataset.h5             # 真机采集原始数据(600样本)
├── tests/                            # 单元测试
└── *.json                            # 各实验结果(评估/闭环/消融/标定)
```

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `scripts/calibration/calibrate_table_height.py` | 桌面高度标定（防撞） |
| `scripts/calibration/verify_safe_height.py` | 静态安全高度验证 |
| `scripts/calibration/test_fk_validation.py` | FK真机验证 |
| `scripts/data_collection/collect_real_data.py` | 真机数据采集（含安全防撞） |
| `scripts/data_collection/convert_real_data.py` | 采集数据→训练格式转换 |
| `scripts/training/train_network.py` | 训练神经网络 |
| `scripts/evaluation/evaluate_four_methods.py` | 四方法软件层评估 |
| `scripts/evaluation/benchmark_real_robot.py` | 四方法真机统一对比 |
| `scripts/evaluation/validate_ik_reaching.py` | IK验证(重复性/往返/真机到达) |
| `scripts/evaluation/ablation_hybrid_steps.py` | 混合法微调步数消融 |
| `scripts/evaluation/closed_loop_joint.py` | 关节空间闭环补偿 |

## 环境依赖与运行

```bash
pip install numpy pyserial pyyaml h5py torch matplotlib
```

**统一从 `src/002/` 目录运行脚本**（脚本按此约定解析相对路径与包导入）：

```bash
cd src/002
python scripts/evaluation/evaluate_four_methods.py --points 50   # 纯软件,无需硬件
python scripts/evaluation/ablation_hybrid_steps.py               # 纯软件
# 真机脚本需连接机械臂(COM口),内置安全防撞
```

> Windows 控制台若报 `UnicodeEncodeError`（✓字符），前面加 `set PYTHONUTF8=1`。

## 安全说明

机械臂夹在桌沿、末端有98mm把手，采集/验证脚本均内置：
- 工作空间地板保护（末端Z≥192mm，基于桌面标定值66mm）
- 每次动作先经HOME中转（防路径扫桌）
- 低速运动 + 关节范围收窄
