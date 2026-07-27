# 基于神经网络的扫地机器人碰撞预测

本实验使用 Pygame 构建带起点和终点的二维扫地机器人仿真。圆形机器人搭载 12 路 360° 测距传感器，PyTorch 多层感知机根据传感器距离、线速度和角速度，预测机器人在未来 `1.0 s` 内是否会发生碰撞。

## 项目组成

- `geometry.py`：圆形—矩形碰撞检测和射线测距。
- `simulation.py`：机器人运动学、随机场景和未来碰撞标签。
- `dataset.py`：生成正负样本近似均衡的仿真数据。
- `network.py`：碰撞预测多层感知机和模型检查点。
- `train.py`：训练、验证和测试。
- `evaluate.py`：比较神经网络与最小距离阈值基线。
- `main.py`：实时 Pygame 可视化与手动/自动控制。
- `config.yaml`：机器人、传感器、训练和场景参数。

所有运行结果保存在当前项目的 `outputs/` 中，不会在 `src` 或 `paper` 之外创建目录。

## 环境

```bash
conda activate hands_on
python -m pip install -r src/006/collision_prediction/requirements.txt
```

## 快速检查界面

未训练模型也可以先检查机器人、障碍物和传感器界面：

```bash
python src/006/collision_prediction/main.py
```

窗口控制：

- `↑/W`：加速；
- `↓/S`：减速；
- `←/A`、`→/D`：转向；
- `Space`：停止；
- `Tab`：切换自动避障模式；
- `P`：在 `ASTAR` 和 `DIRECT` 导航间切换，并从起点重新开始；
- `R`：重置；
- `Esc`：退出。

未训练时界面显示 `Model: NOT TRAINED`，碰撞概率显示 `N/A`。

## 起点、终点和单轮实验

起点和终点在 `config.yaml` 中配置：

```yaml
episode:
  start: [75.0, 75.0, 0.0]
  goal: [880.0, 570.0]
  goal_radius: 28.0
  max_duration_sec: 60.0
```

`start` 的三个值依次为水平坐标、垂直坐标和初始朝向角度。机器人进入终点半径、发生碰撞或运行超过最大时长时，本轮结束。界面会绘制起点、目标区域和实际运动轨迹，并显示运行时间、路径长度和剩余目标距离。按 `R` 开始新一轮。

自动模式采用配置好的安全途经点，不运行复杂路径规划或绕墙状态机。机器人依次到达 `episode.waypoints`，最后前往 `episode.goal`。界面会显示当前途经点序号，并绘制实际运动轨迹。

神经网络只负责预测并显示碰撞概率，不参与导航控制。终点也不输入碰撞预测网络，只用于控制和实验终止判断。

### 区分 A* 与碰撞预测

界面明确显示 `Prediction: DISPLAY ONLY`。按 `P` 可在同一张地形上切换：

- `DIRECT`：不使用 A*，机器人直接朝终点行驶，用于观察被隔墙阻挡以及模型何时发出碰撞预警；
- `ASTAR`：跟随规划路线依次穿过门洞，用于观察路径规划本身的效果。

因此，成功到达终点属于 A* 路径规划结果；神经网络使用 Accuracy、Recall、F1 和碰撞前预警时间单独评价，不能把两者混为同一个避障贡献。

## 随机地形

默认情况下，每次启动程序以及每次按 `R` 重置时都会生成新的分区地形，并重新规划路线。地形由多道贯穿上下边界的纵向隔墙组成，每道墙只保留一个门洞，门洞位置上下交错。因此机器人必须在场地内部连续穿过门洞，不能从顶部或底部绕过障碍。

生成器保证：

- 隔墙横向位置、厚度和门洞大小存在随机变化；
- 障碍物不会覆盖起点和终点；
- 隔墙真正分割起点和终点所在区域；
- A* 路线长度明显大于起终点直线距离；
- 规划时按照机器人半径膨胀障碍物，避免擦碰墙角。

灰蓝色线表示本轮 A* 规划路线，紫色线表示机器人实际轨迹。相关参数位于 `config.yaml` 的 `terrain` 部分。`seed: null` 表示每次使用不同地形；填写固定整数可复现实验地形。将 `randomize` 改为 `false` 后，程序会使用配置末尾的固定障碍物和途经点。

## 训练模型

先用较少样本确认流程：

```bash
python src/006/collision_prediction/train.py --samples 2000 --epochs 10
```

正式训练使用配置中的默认参数：

```bash
python src/006/collision_prediction/train.py
```

训练会生成 `outputs/collision_mlp.pt` 和 `outputs/training_metrics.json`。之后重新运行 `main.py`，界面会实时显示碰撞概率。

PyTorch 的 `BCEWithLogitsLoss` 将 Sigmoid 与二元交叉熵组合，适合当前二分类任务；模型使用 `state_dict` 方式保存，便于复现实验。

## 独立评估

```bash
python src/006/collision_prediction/evaluate.py --samples 4000
```

评估脚本使用不同随机种子生成新场景，同时报告神经网络和最小距离阈值规则的 Accuracy、Precision、Recall、F1 与混淆矩阵。结果写入 `outputs/evaluation_metrics.json`。

## 测试

```bash
python -m pytest src/006/collision_prediction/tests
```

论文实验建议比较不同传感器数量、预测时间窗口、隐藏层规模、训练样本量和告警阈值，并通过消融实验验证速度特征、角速度特征与多角度传感器的贡献。
