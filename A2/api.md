# A2 项目共享接口文档（api.md）

适用人员：001（硬件/Webots 侧）、003（算法侧）、009（模型侧）
协议版本：`1.1`（任何字段改动必须升版本号并通知三方；v1.1 为语义澄清：`OBS_DIM` 明确为单帧观测维度，网络输入形状不再视为协议常量）

本文档是唯一权威接口定义。三方开发必须严格对齐本文档，字段一个不能少、一个不能多（多余字段接收方应忽略，但发送方不应产生）。

---

## 0. 全局约定

| 约定项 | 值 | 说明 |
|---|---|---|
| 长度单位 | 米 (m) | 所有距离、位置 |
| 角度单位 | 弧度 (rad) | 方位角、角速度 |
| 时间单位 | 秒 (s) | `t` 为 episode 内累计**仿真时间**，非墙钟时间 |
| 消息格式 | JSON，UTF-8，一条消息一个 JSON 对象 | 禁止发送裸 `inf`/`nan`（JSON 不支持） |
| 方位角 bearing 约定 | 车身坐标系，车头方向为 0，左正右负，取值 (-π, π] | |
| 动作范围 | v ∈ [-0.5, 0.5] m/s，w ∈ [-1.5, 1.5] rad/s | 发送方可发任意值，server 端 clip |
| 归一化职责 | **001 发原始物理量；003 负责归一化打包；009 的网络只见归一化向量** | 见 §4 |
| 连接保活 | 使用 WebSocket 库自带 ping/pong | 应用层不实现心跳；**真机模式下人工摆车可能阻塞数十秒，应关闭库级 ping/pong 或设置足够长的超时**，避免等待期间被误掐 |
| 断线策略 | v1 不支持断线重连：连接断开双方直接退出，重新启动 | |
| episode 终止原因 | 三种：`collision` / `goal_reached` / `timeout`，互斥 | 见 §5.4 |

### 全局常量（hello 中 server 也会下发，以此表为默认）

| 常量 | 默认值 | 说明 |
|---|---|---|
| `lidar_count` | 64 | 雷达一圈 360° 均匀打出的测距线数量（每 5.625° 一条），构成观测向量前 64 维。线数越多障碍物轮廓越精细，但网络输入越大、训练越慢；64 是够看清障碍物又不至于让 MLP 太大的折中。**真机若换成别的雷达/超声波阵列，此值必须跟着改。** |
| `lidar_max_range` | 3.5 m | 雷达最大量程。距离超过它或没有回波（打到无穷远）时，server 一律按 3.5 发送——JSON 传不了 inf。3.5 m 约为 4 m 场地对角线（5.66 m）的六成，保证车在场内任何位置都能至少看到最近的几面墙/障碍物。 |
| `control_dt` | 0.1 s | 决策周期：server 每收到一个 action，就把仿真推进 0.1 s（内部 10 个物理步 × 10 ms）。等价于网络以 10 Hz 的频率做决策。太大则动作粗糙、撞上了才发现；太小则每步位移太小、奖励信号弱，且 episode 步数变多、训练变慢。 |
| `max_episode_time` | 60.0 s | 单个 episode 的仿真时间上限，60 s ÷ 0.1 s = 600 步封顶，到时置 `timeout`。按 v_max=0.5 m/s 理论可走 30 m，约场地对角线的 5 倍——绕障往返、走错再回头都够用（v1.2 从 30 s 放宽：实测绕障路线常需 15~25 s，30 s 对绕远路线太紧）。设上限是为了防止策略"原地摆烂不动"也能无限混下去。 |
| `v_max` | 0.5 m/s | 线速度指令上限。同时是观测归一化除数（§4）。真机阶段的安全速度也以此为上界，真机实测时应从更低值开始。 |
| `min_linear_vel` | 0.0 m/s | 线速度指令下限：**禁止倒车**（clip 到 [0, +0.5]）。真机车壳遮挡雷达后向扇区（约 177°~277°），后方是盲区，倒车等于闭眼后退，仿真与真机对齐禁止。 |
| `w_max` | 1.5 rad/s | 角速度指令上限，约 86°/s，clip 范围 [-1.5, +1.5]。过大容易甩尾、仿真里一步转太狠导致观测跳变；过小则绕障转弯太慢、600 步内来不及到点。 |
| `goal_tolerance` | 0.15 m | 到达判定半径：目标点视为半径 0.15 m 的圆盘，车中心进入（`dist ≤ 0.15`）即置 `goal_reached`。车不可能精确压到一个数学点上，必须给容差；太大容易"没到也算到"蒙混过关，太小车会在目标边缘打转永远判定不了成功。 |
| `robot_radius` | 0.18 m | 车身轮廓的外接圆半径近似值。碰撞判定简化为**中位数滤波后** `min(lidar) < 0.18` 即 `collision`（滤波见 §5.3 注），免去车身多边形与障碍物的几何相交计算。取值 = 实际车身外接圆半径再略放大，留安全余量；Webots 车身尺寸改了此值要重算。 |
| `arena_size` | 4.0 m | 正方形场地边长（以世界文件为准）。它决定一系列其他参数的合理性：lidar_max_range（≈对角线六成）、目标点采样范围、§4 中 dist 归一化常数 10.0（≈对角线 2 倍，保证归一化后基本不超 1）。场地改大这三个都要重估。 |
| `OBS_DIM` | 132 | **观测维度** = 128（lidar 2 帧堆叠，64×2）+ 2（goal: dist、bearing）+ 2（vel: v、w）。hello 下发的 `obs_dim` 就是它，**协议锁定**；任何一项观测增减都要改它并升协议版本。2026-08-02 从 68（单帧）升级为 132：2 帧堆叠给网络"障碍在逼近还是远离"的趋势感。帧堆叠在 003 侧打包（server 照发单帧，003 维护上一帧，episode 首帧历史=当前帧），堆叠帧数是 003/009 的内部设计，协议消息内容不感知（详见 §4、§7.1）。 |
| `ACT_DIM` | 2 | 网络输出动作维度：(v, w) 两个数。若未来利用麦克纳姆轮加横向速度 vy，变为 3，属于不兼容改动，升协议主版本。 |

---

## 1. 系统组成与角色

```
┌─────────────┐  WebSocket   ┌─────────────┐  进程内 import  ┌─────────────┐
│ 001 硬件侧   │  server      │ 003 算法侧   │                │ 009 模型侧   │
│ Webots 仿真  │◀════════════▶│ SAC 客户端   │───────────────▶│ 网络定义     │
│ (env server)│   JSON 消息   │ (client)    │◀───────────────│ (model.py)  │
└─────────────┘              └─────────────┘                └─────────────┘
```

- **001**：Webots 世界 + 控制器，充当 WebSocket **server**（监听端口 `8765`）。
- **003**：WebSocket **client**，实现 SAC 训练/推理主循环；**唯一**与 001 通信的一方；进程内 import 009 的 model.py。
- **009**：纯 Python 模块，不联网、不碰仿真，只定义网络结构与权重存取。

后期 002/006 的真机驱动程序替换 001 的位置（同样的 server、同样的消息），003/009 零改动。当前真机实现位于 `A2/real_robot/`，运行说明见该目录 `run_a2_real_robot.sh`。由于真机没有 supervisor，真机 server 在 `reset` 流程中扩展了 `human` / `human_confirm` 消息用于人工介入（§2.8、§2.9、§3.4），003 需在真机模式下支持这两种消息。

---

## 2. WebSocket 消息总表

| 消息 type | 方向 | 触发时机 |
|---|---|---|
| `hello` | server → client | TCP/WebSocket 连接建立后，server 立即发送 |
| `reset` | client → server | 开始新 episode 前 |
| `obs` | server → client | ①reset 后的初始观测 ②每收到一个合法 action 推进后 |
| `action` | client → server | 对某条 obs 的回应，一一对应 |
| `all_finish` | client → server | 训练收敛/终止 |
| `bye` | server → client | 收到 all_finish 后回应，随后关连接 |
| `human` | server → client | 真机模式下需要线下人工操作时发送 |
| `human_confirm` | client → server | 线下人工操作完成后回复 server |
| `teleop` | client → server | 真机 `drive_to_start` 阶段遥控小车（临时扩展，不升协议版本） |
| `error` | 双向 | 任何非法消息/状态错误 |

### 2.1 hello（server → client）

连接建立后 server 立即发送，client 收到后必须校验 `obs_dim` 与模型侧 `OBS_DIM`（单帧观测维度）一致、`protocol_version` 兼容，否则发 error 并断开。若算法侧采用帧堆叠/循环结构，网络输入形状按 003/009 内部约定构造，不影响本校验（帧堆叠时网络输入 68×N，但 hello 的 `obs_dim` 恒为 68）。

```json
{
  "type": "hello",
  "protocol_version": "1.1",
  "env_name": "webots_diffbot_v1"  // 真机为 "real_diffbot_v1"
  "config": {
    "lidar_count": 64,
    "lidar_max_range": 3.5,
    "obs_dim": 68,
    "act_dim": 2,
    "control_dt": 0.1,
    "max_episode_time": 60.0,
    "v_max": 0.5,
    "w_max": 1.5,
    "goal_tolerance": 0.15,
    "robot_radius": 0.18,
    "arena_size": 4.0
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| type | string | 固定 "hello" |
| protocol_version | string | "主.次"，主版本不同必须拒绝连接 |
| env_name | string | 环境标识 |
| config | object | 全部全局常量，键固定为上表 11 项 |

### 2.2 reset（client → server）

**作用**：告诉 server "上一局结束了（或刚开始），请布置一个全新场景，并把车放回起点"。每个 episode 开始前必须发一次，且只能在 server 处于 WAIT_RESET 状态时发（即刚连上、或上一局 obs 的 `done=true` 之后）。server 布置完后会回一条**初始 obs**（`episode_id` 自增、`step_id=0`、`t=0.0`），这条 obs 就是新一局的第一步输入。

**示例 1：最常用——随机场景训练**

```json
{
  "type": "reset",
  "seed": -1,
  "config_override": null
}
```

server 自己随机抽一个种子，障碍物、起点、目标点都随机摆。训练时的常规用法。

**示例 2：固定种子——评估/复现**

```json
{
  "type": "reset",
  "seed": 10007,
  "config_override": null
}
```

`seed=10007` → server 用这个种子初始化随机数生成器，**障碍物布局、起点、目标点完全确定**：同一个 seed，不管什么时候发、发多少次，场景一模一样。评估（§6.7 固定 seed 跑 10 局）和调 bug 复现场景就靠它。`seed=-1` 是唯一表示"随机"的值，其余非负整数都是"我要复现这个场景"。

**示例 3：带覆盖——本局限时 15 秒**

```json
{
  "type": "reset",
  "seed": -1,
  "config_override": {"max_episode_time": 15.0}
}
```

`config_override` 是一个 JSON 对象，里面只放**这一局**想临时改的参数，键名与 hello 的 config 一致；这一局结束后失效，下一局回到默认值。目前唯一允许覆盖的键是 `max_episode_time`。其他键（`lidar_count`、`v_max` 等）写进来 server 必须回 `BAD_FIELD` 错误。不需要覆盖时填 `null`（不是省略这个字段）。

**为什么只有 `max_episode_time` 允许每局改？** 判断标准是：改动会不会改变网络输入/输出的数值含义。

- `max_episode_time` 只决定"这一局最多玩多久"。68 维观测向量里没有一位是它的函数（观测里没有"剩余时间"），网络根本不知道时限变了，所以随便改、每局改都无害。
- 反观 `lidar_count` 改一个数，网络输入维度直接变了，模型结构都对不上；`v_max` 改了，观测第 66 维 `v/v_max` 的数值含义就变了——网络昨天学到的"0.5 是半速"今天变成"0.5 是全速"，已学到的策略当场失效。所以这些键必须锁死。

**允许改它是为课程学习（curriculum learning）服务的**，目的是解决冷启动效率问题：训练前期策略什么都不会，60 s 的局基本全程乱撞，episode 又长又全是垃圾数据，训练效率极低。做法是前期用 `config_override` 把时限压短（如 30 s），让每局快速结束、快速重开，单位时间积累更多经验，等价于"先学短跑再学长跑"；成功率上来之后放宽回 60 s，让策略学会在完整时限内完成任务。它是**训练调度器（003）手里的教学旋钮**，不是环境的物理参数——这就是它和其他键的本质区别。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| type | string | 是 | 固定 "reset" |
| seed | int | 是 | 场景随机种子：`-1` = 随机（训练用）；非负整数 = 复现该种子对应的固定场景（评估/调试用）。**相同 seed 必须产生相同场景**（server 用其初始化 RNG） |
| config_override | object 或 null | 是 | 本局临时覆盖的参数，目前仅支持 `max_episode_time`（float，秒）；不需要时填 `null`，不允许省略字段，不允许包含其他键 |

server 收到后的内部流程：校验字段 → 用 seed 初始化 RNG → `simulationResetPhysics()` → 随机摆障碍物 → 采起点和目标点（间距 ≥2.0 m、各自离最近障碍 ≥0.4 m，不满足重采）→ `episode_id` 自增、`step_id=0`、`t=0.0` → 回初始 obs（详见 §5.2）。

### 2.3 obs（server → client）

```json
{
  "type": "obs",
  "episode_id": 3,
  "step_id": 17,
  "t": 1.7,
  "lidar": [0.83, 0.91, 3.5],
  "goal": {"dist": 2.31, "bearing": -0.44},
  "vel": {"v": 0.15, "w": 0.0},
  "flags": {"collision": false, "goal_reached": false, "timeout": false},
  "done": false
}
```

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| type | string | — | 固定 "obs" |
| episode_id | int | — | 从 1 开始自增 |
| step_id | int | — | episode 内从 0 开始自增 |
| t | float | s | episode 内累计仿真时间 = step_id × control_dt |
| lidar | float[64] | m | 距离值，逆时针排列，第 0 条为车头正前；**无效回波已替换为 lidar_max_range**（不得出现 inf/nan）；数值已 clip 到 [0, max_range]；**已过 3 邻域中位数滤波**（每根射线取与左右邻居的中位数，见 §5.3 注） |
| goal.dist | float | m | 目标到车中心距离，≥0 |
| goal.bearing | float | rad | 目标在车身坐标系的方位角，(-π, π] |
| vel.v | float | m/s | 实测线速度。当前 Webots 环境为**无轮运动学底盘**（(v,w) 直接积分驱动，见 §5.3），故等于指令值；换回轮式动力学或真机时才是编码器换算值 |
| vel.w | float | rad/s | 实测角速度，同上 |
| flags.collision | bool | — | 本步发生碰撞 |
| flags.goal_reached | bool | — | 本步到达目标 |
| flags.timeout | bool | — | 本步达到 max_episode_time |
| done | bool | — | 三个 flag 的或；true 后 server 进入等待 reset 状态 |

### 2.4 action（client → server）

```json
{
  "type": "action",
  "episode_id": 3,
  "step_id": 17,
  "v": 0.2,
  "w": 0.5
}
```

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| type | string | — | 固定 "action" |
| episode_id | int | — | 必须等于所回应 obs 的 episode_id |
| step_id | int | — | 必须等于所回应 obs 的 step_id |
| v | float | m/s | 期望线速度，server clip 到 [min_linear_vel, v_max] = [0, 0.5]（禁止倒车） |
| w | float | rad/s | 期望角速度，server clip 到 [-w_max, w_max] |

### 2.5 all_finish（client → server）

**作用**：告诉 server "整个训练任务彻底结束了，关机收工"。注意和 `reset` 区分：`reset` 是"这一局打完，开下一局"，训练要发成百上千次；`all_finish` 是"全部训练结束，环境可以关掉了"，**整个训练过程只发一次**。server 收到后会回一条 `bye`、关闭连接并结束仿真进程。

**三个发送时机，对应 reason 的三个取值**：

| reason | 什么时候发 | 举例 |
|---|---|---|
| `"converged"` | 收敛条件满足（§6.8：连续 3 次评估成功率 ≥90%），正常收工 | 训练循环跑到 break，发 `all_finish` |
| `"interrupted"` | 人为主动终止：Ctrl+C、手动停止脚本 | 003 的信号处理函数里捕获 SIGINT，先发 `all_finish` 再退出——别直接杀进程，否则 Webots 那边会傻等 |
| `"error"` | 训练侧自己出异常（NaN、bug），无法继续 | 003 的兜底 except 里发，detail 信息写进自己的日志 |

```json
{"type": "all_finish", "reason": "converged", "total_episodes": 1200}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| type | string | 是 | 固定 "all_finish" |
| reason | string | 是 | 三选一：`"converged"` / `"interrupted"` / `"error"`，见上表 |
| total_episodes | int | 是 | 本次训练累计跑过的 episode 总数（含最后这局），纯统计用途，server 记日志用，不影响行为 |

**server 收到后的内部流程**：停止推进（运动学模式无电机）→ 回 `bye` → 关闭 WebSocket 连接 → 结束仿真进程。之后想重新训练，需要重启 Webots/控制器。

**client 发出后的流程**：阻塞等待 server 的 `bye`（正常应立即收到）→ 保存最终 checkpoint 和日志 → 退出。等不到（如 server 已崩溃）则设 5 秒超时，超时直接保存退出。

**client 不发 `all_finish` 直接断线的情况**（进程被杀、网络断开）：server 检测到 WebSocket 断开后，按 §0 断线策略直接结束仿真进程。这是兜底路径；正常流程一律走 `all_finish` + `bye` 的礼貌关闭。

### 2.6 bye（server → client）

```json
{"type": "bye", "reason": "all_finish received"}
```

发送后 server 关闭连接并结束仿真进程；client 收到后保存权重并退出。

### 2.7 error（双向）

```json
{"type": "error", "code": "WRONG_STATE", "detail": "action received while waiting for reset"}
```

| code | 含义 |
|---|---|
| BAD_JSON | JSON 解析失败 |
| BAD_TYPE | type 未知 |
| BAD_FIELD | 缺字段/类型错/数组长度错 |
| WRONG_STATE | 状态机不允许的消息（如等待 reset 时收到 action） |
| VERSION_MISMATCH | 协议主版本不一致 |
| INTERNAL | 内部异常，detail 带堆栈摘要 |

收到 error 后双方记录日志并退出（v1 不做恢复）。

### 2.8 human（server → client）【真机扩展】

**仅真机 server 使用**。Webots 仿真环境由 supervisor 自动重置，不需要此消息；真机没有 supervisor，每次 `reset` 后必须线下人工摆放车辆/障碍物，因此 server 通过 `human` 消息暂停流程、提示当前需要人工做什么。

```json
{
  "type": "human",
  "action": "record_goal",
  "detail": "请把车放到目标点，摆好后发送 human_confirm('record_goal')"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| type | string | 固定 "human" |
| action | string | 当前需要人工执行的动作，目前定义：`record_goal`（记录目标点）、`drive_to_start`（把车遥控开到起点）、`record_start`（记录起点） |
| detail | string | 可读的线下操作提示 |

client（003 或任何人工操作客户端）收到后应**阻塞等待线下人员完成操作**，再回复对应的 `human_confirm`。训练脚本里可以把 detail 打印到屏幕或播放提示音。

**遥控开车到起点**：真机 `/odom` 由车轮编码器积分得到，只能跟踪车轮移动，不能跟踪人手搬车。因此必须先摆目标点，再遥控开车到起点，让 odom 记录真实位移。若两点间靠手搬，goal 与 start 的 odom 坐标会重合，导致 `goal_reached` 立即触发。

### 2.9 human_confirm（client → server）【真机扩展】

对 `human` 消息的确认，告诉 server 线下操作已完成。

```json
{"type": "human_confirm", "action": "record_goal"}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| type | string | 固定 "human_confirm" |
| action | string | 必须与收到的 `human.action` 一致 |

真机 reset 的完整时序见 §3.4。

### 2.10 teleop（client → server）【真机扩展】

仅在 `DRIVE_TO_START` 阶段有效，由人工遥控客户端向 server 发送实时速度，server 直接转发到 `/cmd_vel`。属于真机人工介入的临时通道，**不改变 `action` 的语义**，也不影响训练时 003 与 server 之间的 `action`/`obs` 循环。

```json
{"type": "teleop", "v": 0.2, "w": 0.0}
```

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| type | string | — | 固定 "teleop" |
| v | float | m/s | 期望线速度，server 会 clip 到 `[min_linear_vel, v_max]` = [0, 0.5]（禁止倒车） |
| w | float | rad/s | 期望角速度，server 会 clip 到 `[-w_max, w_max]` |

**注意**：`teleop` 只用于把车开到起点；进入 `RUNNING` 后必须使用 `action` 消息驱动。`teleop` 不会触发 `obs` 返回，也不会推进 episode 时间。

---

## 3. 状态机与完整运作流程

### 3.1 server（001）状态机

```
连接建立 → 发 hello → WAIT_RESET
WAIT_RESET ──收到 reset──▶ RUNNING（回初始 obs）
RUNNING ──收到 action──▶ 推进 0.1s ──▶ 回 obs ──▶ RUNNING
RUNNING ──obs.done=true──▶ WAIT_RESET
WAIT_RESET ──收到 all_finish──▶ 发 bye ──▶ 关闭
其余消息 → 回 error 并退出
```

**真机模式补充**：`reset` 后 server 进入 `RECORD_GOAL`，发送 `human(record_goal)`；收到 `human_confirm(record_goal)` 后进入 `DRIVE_TO_START`，发送 `human(drive_to_start)`（用户通过 `teleop` 消息遥控开车到起点）；收到 `human_confirm(drive_to_start)` 后进入 `RUNNING` 并发送初始 `obs`。详见 §3.4。

### 3.2 完整时序

```
client(003)                        server(001)
   │──── connect ─────────────────▶│
   │◀──────────── hello ───────────│  client 校验 obs_dim/版本
   │──── reset(seed=42) ──────────▶│  内部: RNG 初始化→随机障碍→放车→定目标
   │◀──────── obs(ep=1, s=0) ──────│
   │  打包观测向量→009 推理→缩放     │
   │──── action(ep=1, s=0) ───────▶│  内部: clip→运动学积分→推进0.1s→判终止
   │◀──────── obs(ep=1, s=1) ──────│
   │              ……                │
   │◀────── obs(done=true) ────────│  本 episode 结束
   │  存 buffer；SAC 采样更新 K 次   │  （期间 server 原地等待，无心跳）
   │──── reset(seed=-1) ──────────▶│
   │              ……                │
   │──── all_finish ──────────────▶│
   │◀──────────── bye ─────────────│  双方退出
```

### 3.3 训练主循环（003 伪代码）

```python
models = build_models(device)                      # 009 接口
ws = connect("ws://localhost:8765")
hello = recv(); assert_compatible(hello)           # §2.1
for episode in range(MAX_EPISODES):
    send(reset(seed=schedule(episode)))
    obs = recv()
    while not obs.done:
        vec = pack_obs(obs)                        # §4，归一化
        a01, _, _ = actor.sample(vec)              # 009 接口，[-1,1]²
        send(action(v=(a01[0]+1)/2*v_max, w=a01[1]*w_max))
        next_obs = recv()
        r = compute_reward(obs, action, next_obs)  # §6.4
        buffer.push(vec, a01, r, pack_obs(next_obs), done_mask(next_obs))  # §6.5
        obs = next_obs
    if len(buffer) >= WARMUP:
        for _ in range(obs.step_id + 1):           # K = 本 episode 步数，1:1 更新比
            sac_update(models, buffer)             # 003 内部
    log(episode)                                   # §6.6
    if episode % 50 == 0: evaluate_and_maybe_save()# §6.7
    if converged(): break                          # §6.8
send(all_finish(...)); recv()  # bye
save_checkpoint(...); exit()
```

### 3.4 真机模式下的 reset 时序（human-in-the-loop）

真机 server 替换 001 后，`reset` 不再由 supervisor 自动重置世界，而是需要线下人工摆放车辆与障碍物。因此时序扩展如下：

```
client(003/人工客户端)              server(真机 001-replacement)
   │──── connect ─────────────────▶│
   │◀──────────── hello ───────────│  env_name="real_diffbot_v1"
   │──── reset(seed=...) ─────────▶│
   │◀──── human(record_goal) ──────│  提示：把车放到目标点
   │      （线下摆车）              │
   │──── human_confirm ───────────▶│  server 记录当前 /odom 为 goal
   │◀──── human(drive_to_start) ─────│  提示：把车遥控开到起点
   │      （线下发送 teleop 消息）  │
   │──── human_confirm ───────────▶│  server 记录当前 /odom 为 start
   │◀──────── obs(ep=1, s=0) ──────│  开始这一局
   │──── action(ep=1, s=0) ───────▶│
   │              ……                │
```

**关键说明**：
- 真机 server 的 `hello.env_name` 为 `"real_diffbot_v1"`，与 Webots 的 `"webots_diffbot_v1"` 区分；`protocol_version` 仍为 `1.1`，`config` 与仿真一致，保证 003/009 的校验逻辑不变。
- `human` / `human_confirm` 是 001 真机侧的扩展消息。Webots 仿真 server 不发这两种消息；003 在收到时应判断：若 `type == "human"`，则暂停并提示线下操作，操作完成后再发 `human_confirm`。
- 遥控开车要求：`record_goal` 与 `drive_to_start` 之间必须靠车轮移动（推车或开车），不能手搬。`/odom` 只跟踪车轮编码器，不跟踪人手搬车；若手搬，goal 与 start 的 odom 坐标重合，会导致 `goal_reached` 在 step 0 触发。
- 目标点与起点均通过当前 `/odom` 位姿记录，后续 `goal.dist` / `goal.bearing` 均相对起点计算。

---

## 4. 观测打包与归一化（003 负责，009 只接受此向量）

输入向量 132 维（2 帧雷达堆叠 + 4 标量），顺序固定：

| 下标 | 来源字段 | 归一化公式 | 结果范围 |
|---|---|---|---|
| [0:64] | lidar[i]（当前帧） | `clip(x, 0, max_range) / max_range` | [0, 1] |
| [64:128] | lidar[i]（上一帧；episode 首帧=当前帧） | 同上 | [0, 1] |
| [128] | goal.dist | `min(d, 10.0) / 10.0` | [0, 1] |
| [129] | goal.bearing | `b / π` | (-1, 1] |
| [130] | vel.v | `clip(v, -v_max, v_max) / v_max` | [-1, 1]（禁止倒车后实际 ≥0） |
| [131] | vel.w | `clip(w, -w_max, w_max) / w_max` | [-1, 1] |

**动作映射（003 负责，2026-08-02 晚 v2 映射）**：网络输出一个二维动作向量，记作 `(a0, a1) ∈ (-1, 1)²`，然后映射到底盘真实速度范围：

```
v = (a0 + 1) / 2 * v_max      # a0 ∈ (-1,1) → v ∈ (0, 0.5) m/s
w = a1 * w_max                # a1 ∈ (-1,1) → w ∈ (-1.5, 1.5) rad/s
```

为什么从旧映射 `v = a0 * v_max` 改成仿射映射：禁止倒车后 server 把 v<0 钳成 0（`min_linear_vel=0`），旧映射里 a0 的整个负半轴都等于"刹车"——一半动作空间无物理效果、无梯度，策略漂进负半区后难以爬出（训练中"小车不动"的诱因之一）。仿射映射后每个动作都对应有效前进速度，停车只发生在 a0→-1 的极端。

**旧专家数据兼容**：旧映射下采集的演示动作在 `--preload` 预填时统一做 `a0' = 2·clip(a0,0,1) − 1` 变换（a0≥0 段严格可逆），见 `obs_pack.expert_action_v1_to_v2`。

记号说明：`(-1, 1)²` 右上角的 2 表示**二维空间**，来自区间与自身的笛卡尔积 `(-1,1) × (-1,1)`，即"两个分量各自都落在 (-1,1) 区间"，几何上是一个二维正方形区域。同理观测向量所在空间可记为 ℝ⁶⁸。

为什么动作天然落在 (-1,1)：策略网络（009 PolicyNet）的最后一层是 `tanh` 激活函数，tanh 把任意实数压到 (-1, 1)，所以两个输出分量都不可能越界——这给 SAC 提供了**有界动作空间**，保证发给 server 的 `v`、`w` 经缩放后必在 clip 范围内。注意 tanh 不含端点，所以是开区间 (-1,1) 而非 [-1,1]，实际使用中可当作闭区间处理。

**分工红线**：缩放置换（乘 v_max / w_max）只在 003 做。009 的网络永远只工作在 (-1,1)² 空间，不知道 v_max、w_max 是多少；001 收到的永远是物理单位的速度指令。任何一侧多做一次缩放，动作就错一倍。

**帧堆叠（现行实现，v1.1 起与协议解耦）**：本表定义的 132 维已是 003 侧 **2 帧堆叠**的产物——server 每帧照发单帧原始物理量（64 雷达 + goal + vel），003 的 `ObsPacker` 维护上一帧雷达拼成 128 维雷达段。若想换一种记忆结构，003 有两种做法：

- **改堆叠帧数**：如 N=3 时输入 64×3+4=196 维，`obs_dim` 同步改并升协议版本（hello 会下发，client 强校验）。
- **循环网络**（LSTM/GRU/Transformer）：每步仍喂当前帧，记忆在隐藏状态里。

两种做法都不改变协议消息内容。注意第 0 条 obs 之前没有历史：帧堆叠应在 `episode_id` 变化时清空缓冲（首帧可复制填充），循环策略则重置隐藏状态——这是 003 侧两三行的实现细节。

---

## 5. 001（硬件侧）内部实现清单

### 5.1 Webots 世界

- **无轮运动学机器人**（2026-08-02 起）：圆柱车身（r=0.16、h=0.12），**无轮子、无电机、无 physics**——机器人整体为运动学模式，由 supervisor 按 (v, w) 直接积分移动（§5.3），运动精确无打滑/钩挂/卡死。碰撞判定与 §5.3 一致：雷达滤波后 `min(lidar) < robot_radius`（机器人不参与物理碰撞，障碍物仍为物理体）。
- `Lidar` 节点：`horizontalFieldOfView=6.2832`、`numberOfLayers=1`、`resolution=64`、`maxRange=3.5`，装于车顶中心，记录安装高度。
- robot 节点 `supervisor TRUE`、`basicTimeStep=10`。
- 场地：4 m × 4 m 围墙；障碍物 5~8 个（box/cylinder 混合）；目标点放绿色标记柱便于肉眼调试。

### 5.1b pysim：纯 Python 仿真 server（异步 RL 用，2026-08-02 晚引入）

`003/tools/pysim_server.py`：本环境机器人是**理想运动学**（§5.3 无轮直接积分），Webots 里唯一的真实物理只剩 Lidar 射线检测——这部分用解析式射线检测（64 射线 vs 圆障碍 + 围墙）即可复刻。pysim 与 env_server 的关系和差异：

- **完全一致**：协议消息/状态机/error 码、运动学子步积分、reset 采样约束与死局 BFS、雷达中位数滤波 + 遮挡扇区（177°~277°）、碰撞判定（滤波后 min(lidar) < 0.18）、终止优先级。
- **已知差异（可接受，最终策略在 Webots/真机验收）**：障碍按外接圆（Webots 是 Box，正对盒面读数略小）、雷达无噪声/无运动尖峰、无 z 轴姿态。
- **用途**：异步 RL（train_async.py）的采集/评估实例。单实例吞吐比 Webots 高一个数量级，12+ 实例 CPU 开销可忽略；多 Webots 实例方案已弃用（2026-08-02 实测多实例全部卡死）。
- **保真度验收**：同种子 100 局脚本专家，Webots 94% / pysim 96% 成功率。
- 保活差异：env_server 断线即关仿真；pysim 没有仿真可关，`all_finish`/断线后回到 WAIT_RESET 等下一个客户端，进程由启动脚本统一管理。

### 5.2 reset 内部流程

1. 用 seed 初始化 RNG（numpy `default_rng(seed)`，seed=-1 时随机取）；
2. 随机采起点与目标点：两点间距 ≥ 2.0 m，不满足则重采（最多 200 次）；
3. 随机激活 5~8 个障碍物并摆放（supervisor `setSFVec3f`：离起点/目标表面 ≥ 0.4 m，**障碍物表面两两间距 ≥ 0.55 m**——车身外接圆直径 0.36 m，间距小于它就存在物理上过不去的缝；100 次采不到合法位置则弃用该障碍物）；未激活的障碍物沉到地板下；
4. 传送机器人（随机朝向，记录 yaw）与目标标记 → `simulationResetPhysics()` → 静置 3 个物理步；
5. `episode_id += 1`，`step_id = 0`，`t = 0.0`；回初始 obs。

### 5.3 action 内部流程（每收到一条）

1. 校验状态与 episode_id/step_id，不符回 error；
2. clip `(v, w)`；
3. 理想运动学积分（无轮）：每物理步 `yaw += w·dt`、`pos += v·dt·(cos yaw, sin yaw)`，supervisor `setSFVec3f/setSFRotation` 直接写位姿；
4. 推进 10 个物理步（共 0.1 s），**每个物理步**先积分移动再检查碰撞：该物理步**中位数滤波后**的 `min(lidar) < robot_radius` → 立即停推进，置 collision。
   - **注（中位数滤波）**：Webots Lidar 在运动中会偶发孤立单射线尖峰（读数 0.05~0.18 m，但对应方向 0.3~3 m 内无任何实物，静止时不出现）。server 对整圈扫描做 3 邻域中位数滤波（每根射线取与左右相邻射线三根的中位数）：真实近物在近距离必被 ≥2 根相邻射线同时看到所以保留，孤立尖峰被抹除。**碰撞判定、obs 下发、客户端奖励用的都是这份滤波后的扫描**，全链路语义一致；
5. 读取 lidar（`inf` → max_range → 中位数滤波）、supervisor 真值位姿（算 dist/bearing）；实测 v、w = 指令值（运动学模式无编码器）；
6. 判 goal_reached（`dist ≤ goal_tolerance`）与 timeout（`t ≥ max_episode_time`）；组装 obs 发送。

### 5.4 终止判定优先级

同一物理步内：`collision` > `goal_reached` > `timeout`。三者互斥，只置一个。

### 5.5 日志

每 episode 结束写一行 CSV：`episode_id, steps, outcome, min_lidar_ever, final_dist, seed`（前五列见上；`seed` 为附加列，记录本局实际使用的种子，便于复现场景）。

### 5.6 真机实现补充（A2/real_robot/）

真机 server 与 Webots 仿真的差异点：

1. **传感器来源**：`lidar[64]` 由 `/scan`（Delta-2G，288 点/圈，约 6.7 Hz）重采样而来；`vel{v,w}` 取自 `/odom.twist`；`goal` 由 `record_goal` 时记录的 `/odom` 位姿与 `drive_to_start` 时记录的 `/odom` 位姿相减得到。
2. **雷达车头方向标定**：`/scan` 的 0° 不一定与车头正前对齐，需运行 `calibrate_lidar_front.py` 得到 `lidar_front_offset_deg`，启动 server 时传入。当前标定值已记录在 `A2/real_robot/README.md`。
3. **重采样与中位数滤波**：288 点按角度最近邻重采样为 64 线；对 64 线结果做 3 邻域环形中位数滤波；`inf`/无效值替换为 `lidar_max_range`。
4. **动作执行**：收到 `(v,w)` 后 clip 到 `±v_max/±w_max`，以 50 Hz 向 `/cmd_vel` 发布 Twist，持续 `control_dt=0.1 s`，然后读取最新传感器数据并回 `obs`。
5. **坐标系约定**：`drive_to_start` 时把车所在位置视为该 episode 的局部坐标原点；目标点坐标为 `record_goal` 时的 `/odom` 位姿。因此 episode 内 `goal.dist` / `goal.bearing` 均相对于起点计算，依赖 `/odom` 的短时精度。
6. **目标点设定模式**：真机 server 启动参数 `--goal-mode` 控制。
   - `manual-drive`（默认，训练用）：`reset` 后先人工摆目标点 → `record_goal` → 遥控开车到起点 → `drive_to_start` → 开始 episode。
   - `relative`（测试用）：`reset` 后只需摆起点，`goal` 由 `start + (goal_relative_x, goal_relative_y)` 计算。
7. **遥控开车要求**：`record_goal` 与 `drive_to_start` 之间必须靠车轮移动（推车或开车），不能手搬。`/odom` 只跟踪车轮编码器，不跟踪人手搬车；手搬会导致 goal 与 start 的 odom 坐标重合，episode 在 step 0 即 `goal_reached`。
8. **启动依赖**：必须先启动底盘节点（`car_base_node`，串口 `/dev/ttyAMA0` @115200）和 Delta-2G 节点（`/dev/ttyUSB0` @115200），并停止卖家 `APP` 服务以避免 `/cmd_vel` 被覆盖。见 `run_a2_real_robot.sh`。

---

## 6. 003（算法侧）内部实现清单

### 6.1 连接与校验

连上后收 hello，校验 `protocol_version` 主版本、`config.obs_dim == OBS_DIM`（009 常量），失败报错退出。

### 6.2 观测打包与动作缩放

严格按 §4 表执行；这是 001 原始数据与 009 网络之间的唯一转换点。

### 6.3 SAC 实现要点

- replay buffer 容量 200 000，存 `(vec, a01, r, next_vec, done_mask)`；
- 双 Q 网络 + target 网络，soft update `τ=0.005`；α 自动调节（target entropy = −ACT_DIM）；
- `lr=3e-4`，`batch=256`，`γ=0.99`；buffer < 5000 条不更新（warmup）；
- 每 episode 结束后做 `K = 本 episode 步数` 次梯度更新（约 1:1 更新比）。

### 6.4 奖励函数（默认系数，可调，改动记录进实验日志）

```
r = 5.0 × (dist_{t-1} − dist_t)        # 朝目标靠近
    + 200   (若 goal_reached)
    − 200   (若 collision)
    − 0.1                              # 时间惩罚，每步
    − 8.0 × max(0, 0.5 − min(lidar_t)) # 障碍接近惩罚（v1.1 run2 起）
    − 0.5 × (a0 − a0_prev)² − 0.5 × (a1 − a1_prev)²   # 平滑惩罚
```

- 障碍接近惩罚：min(lidar) 进入 0.5 m（`D_SAFE`）内按侵入深度线性扣分，碰撞边界
  （robot_radius 0.18 m）处约 −2.56/步。稀疏 −200 只告诉策略"最后一步错了"，
  这一项给出"离碰撞还有多远"的连续梯度。阈值必须大于 robot_radius，留出可回旋区间。
- 系数约束：`W_DANGER` 必须大于 `W_APPROACH`——危险区内全速冲锋的惩罚梯度
  （W_DANGER×v_max×dt ≈ ×0.05 m/步）才能盖过接近奖励，局部激励才翻转为"绕开更划算"。
- 该惩罚是 003 侧内部奖励塑形，不改变任何协议消息；001/009 无感知。

### 6.5 终止掩码（SAC bootstrap 关键）

- `collision` 或 `goal_reached` → `done_mask = 1`（真终止，target 不 bootstrap）
- `timeout` → `done_mask = 0`（人为截断，target 照常 bootstrap）

### 6.6 日志

每 episode 写 CSV：`episode_id, steps, return, outcome, success(0/1)`；可选 TensorBoard。

### 6.7 评估与存盘

每 50 个训练 episode 做一次评估：**确定性策略（直接取 mean，不采样）**，固定种子 `seed=10000+i` 跑 10 局，记录成功率。每 50 episode 存一次 checkpoint；评估成功率新高另存 `best`。

### 6.8 收敛条件

连续 3 次评估成功率 ≥ 90% → 发 all_finish，存最终 checkpoint，退出。

---

## 7. 009（模型侧）内部实现清单

交付物：`model.py`（可 import，无第三方依赖之外的包，仅 torch + 标准库）。

### 7.1 常量

```python
OBS_DIM: int = 68   # 单帧观测维度（与 hello 的 obs_dim 一致）
ACT_DIM: int = 2
LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0
```

`OBS_DIM` 恒为单帧维度。若 003 采用帧堆叠 N 帧，实例化 `PolicyNet`/`QNet` 时传 `obs_dim=OBS_DIM*N`（如 204）即可，类签名不变；循环策略则保持 68。`build_models` 内部用同一 `obs_dim` 建全部网络，003 传入即可。

### 7.2 类与函数签名（签名不得改动，内部结构可调）

```python
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: tuple = (256, 256)) -> None: ...
    def forward(self, obs): ...
        # 输入:  float32 Tensor [B, obs_dim]
        # 输出:  (mean [B, act_dim], log_std [B, act_dim])
        # log_std 必须 clamp 到 [LOG_STD_MIN, LOG_STD_MAX]
    def sample(self, obs): ...
        # 重参数化采样 + tanh 压缩
        # 输出: (action [B, act_dim] ∈ (-1,1), log_prob [B], mean [B, act_dim])

class QNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: tuple = (256, 256)) -> None: ...
    def forward(self, obs, act): ...
        # 输入: obs [B, obs_dim], act [B, act_dim]
        # 输出: q 值 [B, 1]

def build_models(device) -> dict:
    # 返回 {"actor", "critic1", "critic2", "critic1_target", "critic2_target"}
    # target 与对应 critic 同权重初始化并 requires_grad_(False)

def save_checkpoint(dir_path: str, tag: str, models: dict,
                    optimizers: dict | None, meta: dict) -> str:
    # 保存为 {dir_path}/ckpt_{tag}.pt，返回完整路径

def load_checkpoint(path: str, models: dict,
                    optimizers: dict | None = None,
                    map_location: str = "cpu") -> dict:
    # 加载权重进 models（和可选 optimizers），返回 meta
```

### 7.3 checkpoint 文件格式

`ckpt_{tag}.pt`（torch.save 的 dict）：

| 键 | 内容 |
|---|---|
| actor / critic1 / critic2 / critic1_target / critic2_target | 各自 state_dict |
| optimizers | dict 或 None |
| meta | dict：`episode, update_step, obs_dim, act_dim, protocol_version, saved_at(ISO8601)` |

### 7.4 自测（交付前必须通过）

`python model.py --selftest`：构造 batch=4 随机 obs → sample → 断言输出形状、action 范围 (-1,1)、log_prob 有限；build_models → save → load → 断言权重一致。全绿打印 `SELFTEST OK`。

---

## 8. 联调顺序与开发桩

1. **009 先交** model.py + 自测通过（半天）；
2. **003 并行开发**：自写一个 20 行 echo server 桩（收到 action 就回一条伪造 obs），先把 SAC 主循环跑通；
3. **001 并行开发**：自写一个 20 行随机 client 桩（收到 obs 回随机 action），先把环境推进跑通；
4. 三方联调：hello 校验 → 单 episode 全流程 → 多 episode → 收敛测试；
5. 联调通过后，002/006 按 §1 末尾机制启动真机适配；真机 server 的 `hello.env_name` 为 `"real_diffbot_v1"`，并扩展 `human`/`human_confirm` 消息（§2.8、§2.9、§3.4、§5.6）。

## 9. 变更管理

任何字段、常量、流程的改动：改本文档 → 升 protocol_version 次版本号 → 群里通知三方 → 三方同步改代码。主版本号变更表示不兼容改动，client/server 必须拒绝连接。

**例外**：真机适配所需的 `human` / `human_confirm` 消息属于 env-specific 扩展，不改变 `hello`/`reset`/`obs`/`action`/`all_finish`/`bye` 的核心语义，也不改变观测/动作空间的数值含义，因此不升 protocol_version。003 可通过 `hello.env_name == "real_diffbot_v1"` 判断是否启用 human 处理。
