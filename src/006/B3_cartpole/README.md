# B3：Pygame Cart-Pole 强化学习

本项目不依赖 Gym，直接实现 Cart-Pole 动力学，并用 Pygame 显示小车与倒立摆。你可以训练 **PyTorch DQN**，也可以用离散化后的 **表格 Q-Learning** 做对照。

## 先理解实验

每一步交互都是：

`状态 state → 决策函数 policy → 动作 action → 环境变化 → 奖励 reward`

- 状态空间：`[小车位置, 小车速度, 杆角度, 杆角速度]`，共 4 个连续值。
- 动作空间：`0` 表示向左施力，`1` 表示向右施力。
- 奖励：每坚持一步得 `+1`；提前倒下额外 `-10`；坚持 500 步额外 `+10`。
- 成功：杆在角度限制内、小车不越界，并坚持到 500 步。
- 决策函数：DQN 比较神经网络输出的两个 Q 值；Q-Learning 查询离散状态对应的 Q 表，选择 Q 值较大的动作。

DQN 能直接处理连续状态，但需要经验回放和目标网络稳定训练；表格 Q-Learning 更直观，但必须先把连续状态切成有限区间，精度受分箱数量限制。

## 环境与依赖

在仓库根目录执行：

```bash
conda activate hands_on
```

```bash
python -m pip install -r src/006/B3_cartpole/requirements.txt
```

配置集中在 `config.yaml`。初学阶段建议只改 `episodes`、`learning_rate`、`epsilon_decay_steps`，并一次只改一个变量。

## 训练

训练 DQN（默认 600 回合，4060 显卡会自动使用 CUDA）：

```bash
python src/006/B3_cartpole/train.py --algorithm dqn
```

训练表格 Q-Learning（默认 5000 回合，仅使用 CPU）：

```bash
python src/006/B3_cartpole/train.py --algorithm q_learning
```

快速检查代码能否运行：

```bash
python src/006/B3_cartpole/train.py --algorithm dqn --episodes 5 --device cpu
```

训练输出位于 `src/006/B3_cartpole/outputs/`，包含模型、逐回合 JSON 指标和 TensorBoard 日志。正式实验应固定配置和随机种子，不要用快速检查模型作为最终结果。

## 看训练曲线

```bash
tensorboard --logdir src/006/B3_cartpole/outputs/tensorboard
```

浏览器打开终端显示的网址，重点看：

- `episode/moving_mean_steps`：最近 100 回合平均坚持步数，越接近 500 越好。
- `episode/success`：单回合是否坚持满 500 步；应逐渐更常出现 1。
- `episode/return`：总奖励，通常随坚持步数上升。
- `exploration/epsilon`：随机探索概率，应从高到低平滑下降。
- `train/td_loss` 或 `mean_absolute_td_error`：预测 Q 值与 Bellman 目标的差距；会波动，不要求单调下降。

判断训练是否有效时，优先看留出地图的成功率和平均步数，不要只看 loss。

## 固定种子评估

默认在 100 个未用于训练的连续种子上评估：

```bash
python src/006/B3_cartpole/evaluate.py --algorithm dqn
```

```bash
python src/006/B3_cartpole/evaluate.py --algorithm q_learning
```

`success_rate` 越接近 1 越好，`mean_steps` 越接近 500 越好。固定评估种子使两种算法可以公平比较。

## Pygame 可视化

先训练，再运行模型：

```bash
python src/006/B3_cartpole/main.py --algorithm dqn
```

也可选择 `q_learning`、`random` 或 `manual`。手动模式用左右方向键施力；所有模式均支持 `R` 重置、空格暂停、`Esc` 退出。窗口右侧实时显示状态空间、动作、回报和成功/失败。

## 测试与代码检查

```bash
python -m pytest src/006/B3_cartpole/tests -q
```

```bash
python -m ruff check src/006/B3_cartpole
```

测试覆盖动力学接口、随机种子复现、DQN 张量形状、Q 表更新和无窗口渲染。生成的 `outputs/` 不应提交到 Git。
