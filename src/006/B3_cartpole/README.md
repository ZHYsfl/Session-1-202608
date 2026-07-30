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

DQN 能直接处理连续状态，但需要经验回放和目标网络稳定训练；表格 Q-Learning 更直观，但必须先把连续状态切成有限区间，精度受分箱数量限制。当前 v2 会归一化 DQN 输入并使用 Double DQN；Q-Learning 使用较粗的分箱，让有限训练数据能覆盖更多 Q 表状态。

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

训练 DQN（默认 1000 回合，4060 显卡会自动使用 CUDA）：

```bash
python src/006/B3_cartpole/train.py --algorithm dqn
```

训练表格 Q-Learning（默认 4000 回合，仅使用 CPU）：

```bash
python src/006/B3_cartpole/train.py --algorithm q_learning
```

快速检查代码能否运行：

```bash
python src/006/B3_cartpole/train.py --algorithm dqn --episodes 5 --device cpu
```

训练输出位于 `src/006/B3_cartpole/outputs/v2/`，包含最佳模型、逐回合 JSON 指标和 TensorBoard 日志。训练过程每隔一段时间关闭探索，在独立验证种子上测试，并自动保留平均步数最高的模型。正式实验应固定配置和随机种子，不要用快速检查模型作为最终结果。

## 看训练曲线

```bash
tensorboard --logdir src/006/B3_cartpole/outputs/v2/tensorboard
```

浏览器打开终端显示的网址，重点看：

- `episode/moving_mean_steps`：最近 100 回合平均坚持步数，越接近 500 越好。
- `episode/success`：单回合是否坚持满 500 步；应逐渐更常出现 1。
- `episode/return`：总奖励，通常随坚持步数上升。
- `exploration/epsilon`：随机探索概率，应从高到低平滑下降。
- `train/td_loss` 或 `mean_absolute_td_error`：预测 Q 值与 Bellman 目标的差距；会波动，不要求单调下降。
- `validation/mean_steps`：关闭探索后的验证平均步数，用来选择最佳模型。
- `validation/success_rate`：独立验证种子中坚持满 500 步的比例。

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

### v2 参考结果

使用默认随机种子 `42` 训练，并在随机种子 `10000~10099` 上关闭探索评估：

| 算法 | 最佳模型回合 | 平均步数 | 成功率 |
| --- | ---: | ---: | ---: |
| DQN v2 | 100 | 499.88 | 99% |
| Q-Learning v2 | 1000 | 369.51 | 27% |

旧版 DQN 和 Q-Learning 的对应结果分别为 `164.78 / 0%` 与 `336.57 / 16%`。DQN 后续训练可能出现暂时退化，因此程序保存的是独立验证集上表现最好的 checkpoint，而不是最后一个回合的网络。

## Pygame 可视化

先训练，再运行模型：

```bash
python src/006/B3_cartpole/main.py --algorithm dqn
```

也可选择 `q_learning`、`random` 或 `manual`。手动模式用左右方向键施力；所有模式均支持 `R` 重置、空格暂停、`Esc` 退出。窗口右侧实时显示状态空间、动作、回报和成功/失败。

## 正式实验流水线

`formal_experiments.yaml` 固定了论文实验协议：4组 DQN 消融、3组 Q-Learning 分箱、3个训练种子、每个模型70,000个环境步。正式流水线与前面的单模型教学命令相互独立。

查看完整矩阵（要训练哪些算法版本、每个版本使用哪些随机种子，以及每次训练运行多少环境交互步）：

```bash
python src/006/B3_cartpole/formal_runner.py list
```

训练全部21个学习模型：

```bash
python src/006/B3_cartpole/formal_runner.py train
```

训练中断后重新执行同一命令即可。已经完成且配置哈希、步数均匹配的任务会显示 `SKIP completed`；正在运行时被中断的单个任务会从头重跑，不会伪装成完整结果。

全部训练结束后运行开发评估：

```bash
python src/006/B3_cartpole/formal_runner.py evaluate --split development
```

生成逐种子 CSV、跨种子汇总 CSV、方法对比图、best/final 差异图和样本效率曲线：

```bash
python src/006/B3_cartpole/formal_runner.py report --split development
```

开发结果位于 `outputs/formal/reports/development/`。确认算法、超参数和图表都不再修改后，才允许一次性解封正式测试：

```bash
python src/006/B3_cartpole/formal_runner.py evaluate --split final --confirm-final-test
```

```bash
python src/006/B3_cartpole/formal_runner.py report --split final
```

正式测试不允许筛选部分算法或种子，并会先检查全部21个训练结果。开发评估使用 `10000~10099`，验证使用 `20000~20019`，封存正式测试使用 `30000~30099`。模型、日志和原始结果仍在 Git 忽略的 `outputs/` 中；确认后的表格和图片再放入 `paper/006/B3_cartpole/`。

## 测试与代码检查

```bash
python -m pytest src/006/B3_cartpole/tests -q
```

```bash
python -m ruff check src/006/B3_cartpole
```

测试覆盖动力学接口、随机种子复现、DQN 张量形状、Q 表更新和无窗口渲染。生成的 `outputs/` 不应提交到 Git。
