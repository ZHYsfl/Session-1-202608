# A2 真机 WebSocket Server

本目录是真机替换 001 Webots 仿真的实现，协议兼容 `A2/api.md` v1.1，并扩展 `human`/`human_confirm` 用于人工介入。

## 文件说明

| 文件 | 说明 |
|---|---|
| `A2_real_robot_server.py` | 真机 WebSocket server（核心） |
| `A2_test_client.py` | 简易测试 client，可手动验证 human-in-the-loop |
| `calibrate_lidar_front.py` | 雷达车头方向标定脚本 |
| `run_a2_real_robot.sh` | Pi 上一键启动脚本 |

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

## 人工介入流程（默认 manual-drive）

```text
client 发 reset
server 回 human(record_goal)
  ↓ 人把车摆到目标点
client 发 human_confirm(record_goal)
server 回 human(drive_to_start)
  ↓ 人把车推到/开到起点（可任意旋转朝向）
client 发 human_confirm(drive_to_start)
server 回 obs(step_id=0)
  ↓ 进入 action ↔ obs 循环
```

**注意**：小车只能靠车轮移动让 /odom 更新位置。人手把车搬到目标点或起点，
/odom 不会更新，导致 goal 与 start 重合。因此必须先摆目标点，再**推车/开车**
到起点。

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
