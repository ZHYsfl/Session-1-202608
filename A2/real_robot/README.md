# A2 真机 WebSocket Server

本目录是真机替换 001 Webots 仿真的实现，协议兼容 `A2/api.md` v1.1，并扩展 `human`/`human_confirm` 用于人工介入。

## 文件说明

| 文件 | 说明 |
|---|---|
| `A2_real_robot_server.py` | 真机 WebSocket server（核心） |
| `A2_test_client.py` | 简易测试 client，可手动验证 human-in-the-loop |
| `calibrate_lidar_front.py` | 雷达车头方向标定脚本 |
| `calibrate_velocity.py` | 速度标定脚本（relative 模式） |
| `movement_test.py` | 人工遥控 + 运动测试 |
| `teleop_keyboard.py` | 键盘遥控（record_goal 后开到起点用） |
| `start_server.sh` | Pi 上带 ROS2 环境启动 server 的包装脚本 |
| `run_a2_real_robot.sh` | Pi 上一键启动脚本（底盘+雷达+server） |
| `run_a2_calibration.sh` | Pi 上启动标定 server（relative 模式） |

## 启动（在 Pi 5 上）

```bash
ssh pi@192.168.43.114
bash ~/run_a2_real_robot.sh
```

脚本会：停止卖家 APP → 启动底盘 `car_base_node` → 启动 Delta-2G 雷达 → 启动 A2 server。

## 003 客户端对接

把 `A2/003/config.py` 里的 `WS_URI` 改为：

```python
WS_URI = "ws://192.168.43.114:8765"
```

并在训练循环里处理 `human` 消息（收到后暂停、提示线下操作、发 `human_confirm`）。

## 雷达车头方向标定

1. 确保底盘和雷达节点已启动。
2. 在车正前方 0.3~1.0 m 处放障碍物。
3. 运行：

```bash
source /opt/ros/jazzy/setup.bash
python3 ~/calibrate_lidar_front.py
```

4. 按输出建议值启动 server：

```bash
python3 ~/A2_real_robot_server.py --lidar-front-offset-deg <标定值>
```

### 当前小车的标定值

| 项目 | 值 |
|---|---|
| 雷达型号 | Delta-2G（杉川 3iRobotix） |
| 标定日期 | 2026-08-02 |
| `lidar_front_offset_deg` | **-61.88°** |
| 含义 | 雷达原始 0° 偏左约 62°，车头正前对应雷达 -61.88° |

因此本车启动命令固定为：

```bash
python3 ~/A2_real_robot_server.py --lidar-front-offset-deg -61.88
```

## 速度标定

用 `calibrate_velocity.py`（需 server 以 `--goal-mode relative` 启动）：

```bash
cd A2/003 && uv run python ../real_robot/calibrate_velocity.py --auto-confirm
```

- `--auto-confirm`：自动确认 human 提示（标定无需人工介入）。
- 默认测 0.05/0.1/0.15/0.2 m/s，每档跑 1.5s。
- 小车需放在开阔地面，正前方 2~3m 无障碍。

### 当前小车的标定结果（2026-08-02）

| 命令速度 | 实际速度(wall) | 比例 |
|---|---|---|
| 0.05 m/s | 0.059 m/s | 1.18x |
| 0.10 m/s | 0.105 m/s | 1.05x |
| 0.15 m/s | 0.163 m/s | 1.08x |
| 0.20 m/s | 0.219 m/s | 1.10x |

- 平均比例 **1.10x**：车实际比命令快约 10%，方向正确（正 v 前进）。
- 若想让命令速度 ≈ 实际速度，可在 server 发布 cmd_vel 时除以 1.10；但对 RL 训练并非必需（obs 里 `vel` 是 /odom 实测值，policy 可自适应）。

### 已修复的坑

1. **relative 模式 goal 坐标 bug**（2026-08-02）：旧代码把 `goal_relative` 直接加到世界坐标 x，未乘起点朝向旋转；车头不朝 x 正方向时目标点位置错误，导致"车前进反而离目标更远"。已改为 `goal = start + R(yaw) * goal_relative`。
2. **`start_app` 服务抢资源**：Pi 上卖家的 `start_app.service`/`start_app.timer` 会自动启动整包 car_app 节点（占满 CPU、抢 /cmd_vel），导致 /odom 中断、step 实际耗时 0.67s。已 `sudo systemctl disable start_app.timer`。若 Pi 重启后 car_app 又出现，重新执行禁用命令。

## 人工介入流程（默认 manual-drive）

```text
client 发 reset
server 回 human(record_goal)
  ↓ 人把车摆到目标点
client 发 human_confirm(record_goal)
server 回 human(drive_to_start)
  ↓ 人用键盘遥控把车开到起点（可任意旋转朝向）
client 发 human_confirm(drive_to_start)
server 回 obs(step_id=0)
  ↓ 进入 action ↔ obs 循环
```

**注意**：
- 人手把车搬到目标点或起点，/odom 不会更新，导致 goal 与 start 重合。
- 因此必须用**键盘遥控或程序控制**把车从目标点开到起点。
- 遥控脚本见 [`teleop_keyboard.py`](teleop_keyboard.py)。

## 源码同步（重要）

`/home/zane/session_1/A2/real_robot/` 是仓库里的**源码**，Pi 上 `/home/pi/` 是**执行副本**，两者不会自动同步。

修改源码后同步到 Pi：

```bash
scp /home/zane/session_1/A2/real_robot/*.py \
    /home/zane/session_1/A2/real_robot/*.sh \
    pi@192.168.43.114:/home/pi/
```

Pi 上临时调试后拷回仓库：

```bash
scp pi@192.168.43.114:/home/pi/A2_real_robot_server.py \
   /home/zane/session_1/A2/real_robot/
```
