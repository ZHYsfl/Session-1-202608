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
| 连接保活 | 使用 WebSocket 库自带 ping/pong | 应用层不实现心跳 |
| 断线策略 | v1 不支持断线重连：连接断开双方直接退出，重新启动 | |
| episode 终止原因 | 三种：`collision` / `goal_reached` / `timeout`，互斥 | 见 §5.4 |

### 全局常量（hello 中 server 也会下发，以此表为默认）

| 常量 | 默认值 | 说明 |
|---|---|---|
| `lidar_count` | 64 | 雷达一圈 360° 均匀打出的测距线数量（每 5.625° 一条），构成观测向量前 64 维。线数越多障碍物轮廓越精细，但网络输入越大、训练越慢；64 是够看清障碍物又不至于让 MLP 太大的折中。**真机若换成别的雷达/超声波阵列，此值必须跟着改。** |
| `lidar_max_range` | 3.5 m | 雷达最大量程。距离超过它或没有回波（打到无穷远）时，server 一律按 3.5 发送——JSON 传不了 inf。3.5 m 约为 4 m 场地对角线（5.66 m）的六成，保证车在场内任何位置都能至少看到最近的几面墙/障碍物。 |
| `control_dt` | 0.1 s | 决策周期：server 每收到一个 action，就把仿真推进 0.1 s（内部 10 个物理步 × 10 ms）。等价于网络以 10 Hz 的频率做决策。太大则动作粗糙、撞上了才发现；太小则每步位移太小、奖励信号弱，且 episode 步数变多、训练变慢。 |
| `max_episode_time` | 30.0 s | 单个 episode 的仿真时间上限，30 s ÷ 0.1 s = 300 步封顶，到时置 `timeout`。按 v_max=0.5 m/s 理论可走 15 m，是场地对角线的近 3 倍——足够绕障到达；设上限是为了防止策略"原地摆烂不动"也能无限混下去。 |
| `v_max` | 0.5 m/s | 线速度指令上限，动作 clip 范围 [-0.5, +0.5]（允许倒车）。同时是观测归一化除数（§4）。真机阶段的安全速度也以此为上界，真机实测时应从更低值开始。 |
| `w_max` | 1.5 rad/s | 角速度指令上限，约 86°/s，clip 范围 [-1.5, +1.5]。过大容易甩尾、仿真里一步转太狠导致观测跳变；过小则绕障转弯太慢、300 步内来不及到点。 |
| `goal_tolerance` | 0.15 m | 到达判定半径：目标点视为半径 0.15 m 的圆盘，车中心进入（`dist ≤ 0.15`）即置 `goal_reached`。车不可能精确压到一个数学点上，必须给容差；太大容易"没到也算到"蒙混过关，太小车会在目标边缘打转永远判定不了成功。 |
| `robot_radius` | 0.18 m | 车身轮廓的外接圆半径近似值。碰撞判定简化为 `min(lidar) < 0.18` 即 `collision`，免去车身多边形与障碍物的几何相交计算。取值 = 实际车身外接圆半径再略放大，留安全余量；Webots 车身尺寸改了此值要重算。 |
| `arena_size` | 4.0 m | 正方形场地边长（以世界文件为准）。它决定一系列其他参数的合理性：lidar_max_range（≈对角线六成）、目标点采样范围、§4 中 dist 归一化常数 10.0（≈对角线 2 倍，保证归一化后基本不超 1）。场地改大这三个都要重估。 |
| `OBS_DIM` | 68 | **单帧观测维度** = 64（lidar）+ 2（goal: dist、bearing）+ 2（vel: v、w）。这是环境每帧返回的维度，hello 下发的 `obs_dim` 就是它，**协议锁定**；任何一项观测增减都要改它并升协议版本。注意它**不等于网络输入维度**：算法侧若做帧堆叠（如拼最近 3 帧），网络输入为 68×3=204；循环网络则每步仍喂 68。网络输入形状是 003/009 的内部设计，协议不感知（详见 §4、§7.1）。 |
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

后期 002/006 的真机驱动程序替换 001 的位置（同样的 server、同样的消息），003/009 零改动。

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
| `error` | 双向 | 任何非法消息/状态错误 |

### 2.1 hello（server → client）

连接建立后 server 立即发送，client 收到后必须校验 `obs_dim` 与模型侧 `OBS_DIM`（单帧观测维度）一致、`protocol_version` 兼容，否则发 error 并断开。若算法侧采用帧堆叠/循环结构，网络输入形状按 003/009 内部约定构造，不影响本校验（帧堆叠时网络输入 68×N，但 hello 的 `obs_dim` 恒为 68）。

```json
{
  "type": "hello",
  "protocol_version": "1.1",
  "env_name": "webots_diffbot_v1",
  "config": {
    "lidar_count": 64,
    "lidar_max_range": 3.5,
    "obs_dim": 68,
    "act_dim": 2,
    "control_dt": 0.1,
    "max_episode_time": 30.0,
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

**允许改它是为课程学习（curriculum learning）服务的**，目的是解决冷启动效率问题：训练前期策略什么都不会，30 s 的局基本全程乱撞，episode 又长又全是垃圾数据，训练效率极低。做法是前期用 `config_override` 把时限压短（如 15 s），让每局快速结束、快速重开，单位时间积累更多经验，等价于"先学短跑再学长跑"；成功率上来之后放宽回 30 s，让策略学会在完整时限内完成任务。它是**训练调度器（003）手里的教学旋钮**，不是环境的物理参数——这就是它和其他键的本质区别。

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
| lidar | float[64] | m | 原始距离，逆时针排列，第 0 条为车头正前；**无效回波已替换为 lidar_max_range**（不得出现 inf/nan）；数值已 clip 到 [0, max_range] |
| goal.dist | float | m | 目标到车中心距离，≥0 |
| goal.bearing | float | rad | 目标在车身坐标系的方位角，(-π, π] |
| vel.v | float | m/s | 由轮速换算的实测线速度（非指令值） |
| vel.w | float | rad/s | 由轮速换算的实测角速度 |
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
| v | float | m/s | 期望线速度，server clip 到 [-v_max, v_max] |
| w | float | rad/s | 期望角速度，server clip 到 [-w_max, w_max] |

### 2.5 all_finish（client → server）

**作用**：告诉 server "整个训练任务彻底结束了，关机收工"。注意和 `reset` 区分：`reset` 是"这一局打完，开下一局"，训练要发成百上千次；`all_finish` 是"全部训练结束，环境可以关掉了"，**整个训练过程只发一次**。server 收到后会停掉电机、回一条 `bye`、关闭连接并结束仿真进程。

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

**server 收到后的内部流程**：把电机速度置 0（车停下来，防止仿真空转时车还在动）→ 回 `bye` → 关闭 WebSocket 连接 → 结束仿真进程。之后想重新训练，需要重启 Webots/控制器。

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

### 3.2 完整时序

```
client(003)                        server(001)
   │──── connect ─────────────────▶│
   │◀──────────── hello ───────────│  client 校验 obs_dim/版本
   │──── reset(seed=42) ──────────▶│  内部: RNG 初始化→随机障碍→放车→定目标
   │◀──────── obs(ep=1, s=0) ──────│
   │  打包观测向量→009 推理→缩放     │
   │──── action(ep=1, s=0) ───────▶│  内部: clip→轮速换算→推进0.1s→判终止
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
        send(action(v=a01[0]*v_max, w=a01[1]*w_max))
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

---

## 4. 观测打包与归一化（003 负责，009 只接受此向量）

输入向量 68 维，顺序固定：

| 下标 | 来源字段 | 归一化公式 | 结果范围 |
|---|---|---|---|
| [0:64] | lidar[i] | `clip(x, 0, max_range) / max_range` | [0, 1] |
| [64] | goal.dist | `min(d, 10.0) / 10.0` | [0, 1] |
| [65] | goal.bearing | `b / π` | (-1, 1] |
| [66] | vel.v | `clip(v, -v_max, v_max) / v_max` | [-1, 1] |
| [67] | vel.w | `clip(w, -w_max, w_max) / w_max` | [-1, 1] |

**动作映射（003 负责）**：网络输出一个二维动作向量，记作 `(a0, a1) ∈ (-1, 1)²`，然后线性放大到底盘真实速度范围：

```
v = a0 * v_max      # a0 ∈ (-1,1) → v ∈ (-0.5, 0.5) m/s
w = a1 * w_max      # a1 ∈ (-1,1) → w ∈ (-1.5, 1.5) rad/s
```

记号说明：`(-1, 1)²` 右上角的 2 表示**二维空间**，来自区间与自身的笛卡尔积 `(-1,1) × (-1,1)`，即"两个分量各自都落在 (-1,1) 区间"，几何上是一个二维正方形区域。同理观测向量所在空间可记为 ℝ⁶⁸。

为什么动作天然落在 (-1,1)：策略网络（009 PolicyNet）的最后一层是 `tanh` 激活函数，tanh 把任意实数压到 (-1, 1)，所以两个输出分量都不可能越界——这给 SAC 提供了**有界动作空间**，保证发给 server 的 `v`、`w` 经缩放后必在 clip 范围内。注意 tanh 不含端点，所以是开区间 (-1,1) 而非 [-1,1]，实际使用中可当作闭区间处理。

**分工红线**：缩放置换（乘 v_max / w_max）只在 003 做。009 的网络永远只工作在 (-1,1)² 空间，不知道 v_max、w_max 是多少；001 收到的永远是物理单位的速度指令。任何一侧多做一次缩放，动作就错一倍。

**帧堆叠 / 循环结构（可选，003/009 内部设计，v1.1 起明确与协议解耦）**：本表定义的是**单帧**打包（68 维）。若想给网络动力学记忆（加速滞后、惯性滑行——本环境的 `vel` 反馈里就带着这些信息），003 有两种做法：

- **帧堆叠**：把最近 N 帧向量按本表各拼一次、拼接为 68×N 输入（各帧用同一套归一化）。如 N=3 时输入 204 维，PolicyNet/QNet 实例化传 `obs_dim=204` 即可（§7.1 签名不变）。
- **循环网络**（LSTM/GRU/Transformer）：每步仍喂 68 维，记忆在隐藏状态里，无需改维度。

两种做法都不改变协议消息内容与 hello 的 `obs_dim`。注意第 0 条 obs 之前没有历史：帧堆叠应在 `episode_id` 变化时清空缓冲（首帧可复制填充），循环策略则重置隐藏状态——这是 003 侧两三行的实现细节。

---

## 5. 001（硬件侧）内部实现清单

### 5.1 Webots 世界

- 差速小车：box 车身 + 左右驱动轮（RotationalMotor）+ 万向球轮；记录轮距 `L`、轮半径 `R`（写入代码注释，真机对齐时要用）。
- `Lidar` 节点：`horizontalFieldOfView=6.2832`、`numberOfLayers=1`、`resolution=64`、`maxRange=3.5`，装于车顶中心，记录安装高度。
- robot 节点 `supervisor TRUE`、`basicTimeStep=10`。
- 场地：4 m × 4 m 围墙；障碍物 5~8 个（box/cylinder 混合）；目标点放绿色标记柱便于肉眼调试。

### 5.2 reset 内部流程

1. 用 seed 初始化 RNG（numpy `default_rng(seed)`，seed=-1 时随机取）；
2. 随机采起点与目标点：两点间距 ≥ 2.0 m，不满足则重采（最多 200 次）；
3. 随机激活 5~8 个障碍物并摆放（supervisor `setSFVec3f`：互不重叠、离起点/目标表面 ≥ 0.4 m，100 次采不到合法位置则弃用该障碍物）；未激活的障碍物沉到地板下；
4. 传送机器人（随机朝向）与目标标记 → 电机置 0 → `simulationResetPhysics()` → 静置 3 个物理步；
5. `episode_id += 1`，`step_id = 0`，`t = 0.0`；回初始 obs。

### 5.3 action 内部流程（每收到一条）

1. 校验状态与 episode_id/step_id，不符回 error；
2. clip `(v, w)`；
3. 差速运动学换算轮速：`ω_r = (v + w·L/2) / R`，`ω_l = (v − w·L/2) / R`，设置电机速度；
4. 推进 10 个物理步（共 0.1 s），**每个物理步**检查碰撞：该物理步 `min(getRangeImage()) < robot_radius` → 立即停推进，置 collision；
5. 读取 lidar（`inf` → max_range）、supervisor 真值位姿（算 dist/bearing）、轮速反馈（换算实测 v、w：`v=(v_l+v_r)/2`，`w=(v_r−v_l)/L`）；
6. 判 goal_reached（`dist ≤ goal_tolerance`）与 timeout（`t ≥ max_episode_time`）；组装 obs 发送。

### 5.4 终止判定优先级

同一物理步内：`collision` > `goal_reached` > `timeout`。三者互斥，只置一个。

### 5.5 日志

每 episode 结束写一行 CSV：`episode_id, steps, outcome, min_lidar_ever, final_dist, seed`（前五列见上；`seed` 为附加列，记录本局实际使用的种子，便于复现场景）。

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
    − 3.0 × max(0, 0.5 − min(lidar_t)) # 障碍接近惩罚（v1.1 run2 起）
    − 0.5 × (a0 − a0_prev)² − 0.5 × (a1 − a1_prev)²   # 平滑惩罚
```

- 障碍接近惩罚：min(lidar) 进入 0.5 m（`D_SAFE`）内按侵入深度线性扣分，碰撞边界
  （robot_radius 0.18 m）处约 −0.96/步。稀疏 −200 只告诉策略"最后一步错了"，
  这一项给出"离碰撞还有多远"的连续梯度。阈值必须大于 robot_radius，留出可回旋区间。
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
5. 联调通过后，002/006 按 §1 末尾机制启动真机适配。

## 9. 变更管理

任何字段、常量、流程的改动：改本文档 → 升 protocol_version 次版本号 → 群里通知三方 → 三方同步改代码。主版本号变更表示不兼容改动，client/server 必须拒绝连接。
