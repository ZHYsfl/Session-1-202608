# 基于 PyTorch PPO 的机器人局部避障

本实验使用 PyTorch PPO 训练二维差速机器人。策略接收 36 路 360° 测距、相对目标状态和短时控制历史，直接输出前进速度与角速度。Pygame 只负责演示；A* 仅用于离线校准和路径效率评估，不向策略提供地图、航点或动作。

## 项目文件

- `rl_env.py`：GPU/CPU 批量环境、程序化通道、射线和碰撞。
- `policy.py`：环形填充射线 CNN 与 MLP 对照模型。
- `ppo.py`：GAE 与 PPO 裁剪更新。
- `train.py`：课程学习、固定验证、checkpoint 和续训。
- `calibrate_terrain.py`：1000 地图校准与 36 图预览。
- `evaluate.py`：固定留出测试。
- `main.py`：Pygame 实时演示。
- `config.yaml`：全部实验参数。

训练产物位于 `src/006/collision_prediction/outputs/`，该目录仅供本地使用。

## 环境准备

```bash
conda activate hands_on
```

```bash
python -m pip install -r src/006/collision_prediction/requirements.txt
```

检查 RTX 4060 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 程序化三级课程

每张地图先生成弯曲通道，再用贯穿场地的可见分隔墙塑造入口，因此不能从上下边缘抄近路。

- Stage 0：2 道门、4 个障碍、1 次主要转向，至少训练 300 万步。
- Stage 1：3–4 道门、6–8 个障碍、1–2 次方向变化，至少训练 500 万步。
- Stage 2：4–6 道门、8–12 个障碍、2–4 次方向变化，至少训练 1000 万步，并加入测距噪声和射线丢失。

训练进入新阶段后，任务按 70% 当前阶段、30% 已通过阶段混合，避免遗忘。地图不包含死路、迷宫或隐藏碰撞边界。

正式训练前运行校准：

```bash
python src/006/collision_prediction/calibrate_terrain.py --device cpu --episodes 1000 --num-envs 64
```

当前固定种子结果：A* 可解率 100%、几何重复率 0%、A* 路径比 1.200–1.653、DIRECT 成功率 0%、REACTIVE 成功率 37%。预览图和报告分别保存为 `outputs/terrain_preview_v2.png`、`outputs/terrain_calibration_v2.json`。

## 正式训练

RTX 4060 推荐直接使用配置中的 512 个并行环境和 3000 万步上限：

```bash
python src/006/collision_prediction/train.py --model cnn --device cuda
```

训练中断后续训：

```bash
python src/006/collision_prediction/train.py --model cnn --device cuda --resume
```

每 10 次 PPO 更新会在 256 张固定验证地图上评估。达到阶段最低步数，并连续 3 次满足成功率不低于 85%、碰撞率不高于 10% 后才会晋级。Stage 2 必须达到成功率不低于 90%、碰撞率和超时率均不高于 5%，才会生成正式 `cnn_ppo_v2_best.pt`。

`cnn_ppo_v2_latest.pt` 保存模型、优化器、全局步数、课程阶段和随机状态；`cnn_ppo_v2_stage*_best.pt` 保存各阶段最佳策略。

查看曲线：

```bash
tensorboard --logdir src/006/collision_prediction/outputs/tensorboard_cnn_v2
```

终端显示 `TensorBoard ... at http://localhost:6006/` 后，在浏览器打开 `http://localhost:6006/`，进入 **Scalars** 页面。左侧勾选想看的指标；横轴选择 `STEP`，表示累计环境交互步数。曲线抖动是强化学习的正常现象，可把右上角 `Smoothing` 调到 `0.6–0.8` 观察整体趋势，但判断是否达标时仍以未经平滑的原始值为准。

### 新手建议先看这四组图

#### 1. 验证结果：最重要

- `validation/success_rate`：固定验证地图的成功率，越高越好。
- `validation/collision_rate`：碰撞结束的比例，越低越好。
- `validation/timeout_rate`：没有碰撞但规定步数内未到达的比例，越低越好。
- `validation/mean_path_ratio`：成功轨迹长度除以 A* 参考长度，接近 `1.0` 表示绕行较直接；例如 `1.30` 表示比参考路线长约 30%。
- `validation/mean_episode_reward`：一个完整验证回合的平均累计奖励，只作为辅助诊断。

验证每 10 次 PPO 更新运行一次，所以这些曲线的点比训练曲线少。如果一次验证没有任何成功回合，`mean_path_ratio` 会是 `NaN`，TensorBoard 可能不显示该点。Stage 0 和 Stage 1 连续 3 次满足成功率 ≥85%、碰撞率 ≤10% 才能升级；最终 Stage 2 要求成功率 ≥90%、碰撞率 ≤5%、超时率 ≤5%。

#### 2. 训练表现：观察学习方向

- `train/training_success_rate`：最近最多 500 个训练回合的成功比例。
- `train/mean_step_reward`：当前 rollout 中每一步的平均奖励。
- `train/curriculum_stage`：当前课程阶段，依次为 `0 → 1 → 2`。
- `train/stage_steps`：当前阶段累计训练步数，升级后会重新从 0 开始。

训练成功率来自带随机动作、随机地图、阶段混合和噪声的训练过程；验证成功率来自固定地图和确定性动作，两者不完全相同是正常的。课程升级后地图突然变难，成功率和 reward 短暂下降、value loss 上升也正常，之后应重新恢复。

#### 3. PPO 是否稳定：出现异常时再看

- `train/approximate_kl`：新旧策略变化幅度。接近 0 表示更新保守；长期大于约 `0.05` 通常表示更新过猛，需要检查学习率。
- `train/entropy`：动作随机性。训练初期较高、随后缓慢下降通常正常；过早接近 0 可能说明探索不足，一直很高则可能尚未收敛。
- `train/policy_loss`：策略更新目标。正负和短期震荡都正常，不能按“越小越好”判断。
- `train/value_loss`：价值网络的预测误差。课程切换时可能尖峰；若长期快速增大且成功率没有改善，训练可能不稳定。

#### 4. 进度信息

- `train/global_step`：所有并行环境累计产生的交互步数。
- `train/update`：完成的 PPO 更新次数。
- `validation/episodes`：每次固定验证使用的回合数，默认 256。
- `validation/stage`：本次验证所属课程阶段。

### 怎样判断训练得好不好

按以下顺序判断，不要只盯 reward：

1. `validation/success_rate` 是否总体上升；
2. `validation/collision_rate` 和 `validation/timeout_rate` 是否下降；
3. `validation/mean_path_ratio` 是否逐渐接近 1；
4. `approximate_kl` 和 `entropy` 是否保持合理，而不是突然失控；
5. 最后才用 reward 和两个 loss 帮助解释问题。

常见情况：

- reward 上升但成功率不升：机器人可能只学会靠近目标或安全停车，没有完成任务。
- 碰撞率下降但超时率上升：策略可能过于保守，经常停住。
- 训练成功率很高、验证成功率很低：可能对训练地图过拟合。
- 所有指标长时间不变：检查训练进程是否仍在运行、GPU 是否被使用，以及 TensorBoard 是否读取了正确目录。
- 页面没有曲线：先确认 `outputs/tensorboard_cnn_v2/` 中已经生成 `events.out.tfevents...` 文件；训练至少完成一次 PPO 更新后才会写入数据。

## 演示与评估

无需模型查看 REACTIVE 基线：

```bash
python src/006/collision_prediction/main.py --controller reactive --device cuda
```

正式训练完成后加载 CNN：

```bash
python src/006/collision_prediction/main.py --controller cnn --device cuda
```

按 `R` 生成新任务，按 `Space` 暂停，按 `Esc` 退出。

运行 500 张从未参与训练或调参的留出测试：

```bash
python src/006/collision_prediction/evaluate.py --controller cnn --device cuda --episodes 500
```

噪声鲁棒性测试：

```bash
python src/006/collision_prediction/evaluate.py --controller cnn --device cuda --episodes 500 --noise
```

最终重点查看成功率、碰撞率、超时率和成功回合路径比；累计奖励只用于训练诊断。

## 自动测试

```bash
python -m pytest src/006/collision_prediction/tests
```

```bash
python -m ruff check src/006/collision_prediction
```
