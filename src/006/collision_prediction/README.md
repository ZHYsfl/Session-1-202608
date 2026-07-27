# 基于 PyTorch PPO 的机器人局部避障

本实验使用 36 路 360° 测距、相对局部目标和短时控制历史作为观测。PPO 策略直接输出前向速度与角速度，在静态随机障碍中完成局部绕行。训练环境使用 PyTorch 批量张量计算，Pygame 负责加载策略并显示实际运动轨迹；运行时不使用 A* 航点。

## 项目组成

- `rl_env.py`：GPU/CPU 通用的批量机器人环境、射线和碰撞计算。
- `policy.py`：零填充射线 CNN 与扁平 MLP Actor-Critic。
- `ppo.py`：GAE 和 PPO 裁剪更新。
- `train.py`：课程学习、checkpoint 和 TensorBoard 训练入口。
- `controller.py`：DIRECT、REACTIVE 和学习策略控制器。
- `planner.py`：仅用于离线地图检查和路径效率参考的 A*。
- `evaluate.py`：在固定种子任务上比较四类控制器。
- `main.py`：Pygame 自动避障演示。
- `config.yaml`：环境、奖励、模型和训练参数。

训练产物保存在 `src/006/collision_prediction/outputs/`，该目录仅供本地使用。

## 安装环境

激活已有 Conda 环境：

```bash
conda activate hands_on
```

安装依赖：

```bash
python -m pip install -r src/006/collision_prediction/requirements.txt
```

## 立即查看界面

REACTIVE 不需要模型，可先检查随机地图、36 条射线和回合逻辑：

```bash
python src/006/collision_prediction/main.py --controller reactive
```

窗口按键：

- `R`：生成新的局部绕行任务；
- `Space`：暂停或继续；
- `Esc`：退出。

## CNN-PPO 训练

### 1. 确认 CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期输出应包含 `True` 和显卡名称。本项目已在 RTX 4060、PyTorch 2.6 CUDA 12.4 环境完成冒烟验证。

### 2. 运行短程冒烟训练

这条命令只验证环境采样、PPO 更新和 checkpoint 保存，不会得到可靠的避障模型：

```bash
python src/006/collision_prediction/train.py --model cnn --device cuda --num-envs 64 --rollout-steps 32 --total-steps 8192 --no-noise
```

### 3. 运行长程 CNN 训练

当前建议单个随机种子最多训练 2000 万环境步：

```bash
python src/006/collision_prediction/train.py --model cnn --device cuda --num-envs 512 --rollout-steps 128 --total-steps 20000000
```

训练从课程阶段 0 开始；近期训练回合到达率达到 `0.75` 后依次进入阶段 1 和阶段 2。阶段 2 会加入遮挡目标、窄通道和测距噪声。终端会持续输出 `step`、`stage`、`success` 和平均奖励。

训练产物：

- `outputs/cnn_ppo.pt`：最新 CNN 策略；
- `outputs/cnn_training_metrics.json`：训练更新历史；
- `outputs/tensorboard_cnn/`：TensorBoard 日志。

另开一个终端查看训练曲线：

```bash
tensorboard --logdir src/006/collision_prediction/outputs/tensorboard_cnn
```

> 注意：当前可运行版本会在每次 PPO 更新后覆盖 `cnn_ppo.pt`，再次执行训练命令会从头开始，尚未实现优化器与课程阶段的完整断点续训。中断训练前应确认这一限制。

## 加载 CNN 策略演示

训练结束后启动 Pygame：

```bash
python src/006/collision_prediction/main.py --controller cnn --device cuda
```

运行噪声鲁棒性演示：

```bash
python src/006/collision_prediction/main.py --controller cnn --device cuda --noise
```

`R` 会生成新任务。建议连续观察不同障碍布局，而不是只检查一个成功回合。

## 固定种子评估

评估 CNN 的 500 个无噪声任务：

```bash
python src/006/collision_prediction/evaluate.py --controller cnn --device cuda --episodes 500
```

评估 CNN 的 500 个噪声任务：

```bash
python src/006/collision_prediction/evaluate.py --controller cnn --device cuda --episodes 500 --noise
```

运行非学习 REACTIVE 基线：

```bash
python src/006/collision_prediction/evaluate.py --controller reactive --device cuda --episodes 500
```

评估分别报告到达率、碰撞率、超时率、平均路径长度和相对离线 A* 参考路径的路径效率。A* 不参与控制，也不会向 PPO 提供路线。正式达标标准为无噪声到达率不低于 90%、碰撞率不高于 5%、超时率不高于 5%、平均路径比不高于 1.35。

### 当前版本限制

当前训练器依据近期训练回合到达率升级课程，尚未实现此前方案中的“固定 256 回合验证集、连续三次验证达标、完整断点续训和分阶段 checkpoint”。因此，跑满 2000 万步不自动等同于正式训练达标；必须执行上述固定种子评估并检查各项指标。

## 自动测试

```bash
python -m pytest src/006/collision_prediction/tests
```

当前阶段先完成 CNN-PPO 单随机种子训练。论文正式结果仍应在训练流程稳定后补充多个随机种子，并报告均值与标准差。
