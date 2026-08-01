# rl_chassis — A2 项目 001 硬件侧仿真环境

VOA（Vision-Other-Action）强化学习防碰撞项目的 Webots 仿真环境 + WebSocket server。
协议与参数的唯一权威定义见仓库 `src/001/A2/api.md`（v1.0），本目录代码与之对齐。

## 目录结构

```
rl_chassis/
├── worlds/
│   └── rl_arena.wbt            # 仿真世界：4x4 m 场地、差速小车、64 线雷达、8 个障碍物
├── controllers/
│   └── env_server/
│       ├── env_server.py       # WebSocket server（api.md 协议实现，001 的全部核心代码）
│       ├── runtime.ini         # 告诉 Webots 用哪个 python 启动控制器
│       └── logs/episodes.csv   # 每局结果日志（运行时生成）
├── tools/
│   ├── random_client.py        # 开发桩：随机动作客户端，验证全流程
│   ├── protocol_selftest.py    # 协议自测 A/B/C/D（验收用，见下文）
│   ├── probe_motion.py         # 开环运动探针：测量轮速符号/动力学响应
│   └── diag_goal.py            # seed=2008 到点导航诊断脚本
└── README.md
```

## 首次设置（只需一次）

```powershell
cd D:\webots_projects\rl_chassis
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

（依赖只有 numpy + websockets，详见 `requirements.txt`。`controller` 模块由 Webots 自带。）

`controllers/env_server/runtime.ini` 已指向该虚拟环境的 python，
Webots 启动控制器时会自动使用它（无需改动 Webots 全局设置）。

## 跑通全流程（5 分钟验收）

1. Webots 打开 `worlds\rl_arena.wbt`，按 ▶ 运行。
   控制台应出现：
   - `[env_server] lidar 标定: 方向=... 平均误差=... mm`（自动标定雷达数组顺序，误差应 < 80 mm）
   - `[env_server] WebSocket server 已启动: ws://127.0.0.1:8765`
2. 另开 PowerShell 跑开发桩：

   ```powershell
   .venv\Scripts\python.exe tools\random_client.py --episodes 3
   ```

   预期：client 打印 hello 摘要 → 每局打印初始雷达"前/左/后/右"四个读数
   （**对照 Webots 画面目测检查方向是否正确**）→ 小车随机乱走直至碰撞/超时 →
   最后 `bye 收到，全流程 OK`。
3. 查看 `controllers\env_server\logs\episodes.csv`，每局一行结果。
4. 协议自测（交付前必跑）：重开 Webots（server 设计为单客户端，退出即关仿真），再跑

   ```powershell
   .venv\Scripts\python.exe tools\protocol_selftest.py
   ```

   应输出 4 项 PASS：A. timeout+config_override；B. 同 seed 场景复现；
   C. goal_reached（确定性，seed=2008 直线通道，~100 步到达）；D. 错误处理 BAD_FIELD。

验收通过后，把 `ws://localhost:8765` 和 api.md 交给 003，算法侧即可开工。

## 参数对照表（api.md ↔ 实现）

| api.md 常量 | 值 | 实现位置 |
|---|---|---|
| arena_size | 4.0 m | .wbt 围墙 x/y = ±2.0 m |
| lidar_count / max_range | 64 / 3.5 m | .wbt Lidar：`fieldOfView 6.283185`、`horizontalResolution 64`、`maxRange 3.5`、`type "fixed"` |
| control_dt | 0.1 s | env_server.py：每次 action 推进 10 × basicTimeStep(10 ms) |
| robot_radius | 0.18 m | env_server.py 碰撞判定；车身外接圆实测 ~0.170 m |
| goal_tolerance | 0.15 m | env_server.py 到达判定；.wbt 绿色标记柱半径即 0.15 m |
| v_max / w_max | 0.5 m/s / 1.5 rad/s | env_server.py clip；电机 maxVelocity 15 rad/s 足够覆盖 |
| max_episode_time | 60 s | env_server.py；reset 的 config_override 可临时覆盖 |

底盘几何（真机对齐时用）：**轮半径 R = 0.05 m，轮距 L = 0.20 m，雷达安装高度 0.20 m**，
同时记录在 .wbt 注释与 env_server.py 常量区。

## 实现要点（读代码前先看）

- **雷达自动标定**：Webots 雷达数组的第 0 条朝向/排列方向没有稳定约定，
  env_server 启动时会把车传送到已知位姿、对比实测与解析距离，
  从 128 种候选重排中自动选出正确的一种（日志会打印误差）。
  **标定位姿必须同时偏离 x、y 两条对称轴**（现用 (1.5, 0.7)）：曾经放在 (1.5, 0)，
  左右墙等距、场景镜像对称，CCW/CW 两种重排拟合误差完全相同，方向蒙错后
  obs 里障碍物左右全是反的，"避障"变"瞄着撞"。修复后 CCW 369 mm vs CW 46 mm，
  方向悬殊可辨；日志同时打印两方向误差，相近时必须告警。
  若改了场地尺寸或雷达配置，留意标定误差是否变大。
- **雷达无效值与尖峰**：`inf` 一律替换为 3.5（max_range），JSON 传不了 inf。
  Webots Lidar 运动中还会偶发**孤立单射线尖峰**（读数 0.05~0.18 m，对应方向
  0.3~3 m 内无任何实物，静止时不出现）：server 对整圈扫描做 3 邻域中位数滤波，
  **碰撞判定、obs 下发、客户端奖励统一用滤波后的扫描**（真实近物必被 ≥2 根
  相邻射线同时看到所以保留，孤立尖峰被抹除；episode 日志有 `拦尖峰=` 计数）。
- **碰撞判定**：每个物理步（10 ms）判一次**滤波后** `min(lidar) < 0.18`，撞了立即停止本步推进。
- **终止优先级**：collision > goal_reached > timeout，互斥（api.md §5.4）。
- **场景随机化**：reset 时 5~8 个障碍物随机摆放（离起点/目标表面 ≥0.4 m、
  **障碍物表面两两间距 ≥0.55 m**——车身外接圆直径 0.36 m，间距不足会生成物理上
  过不去的缝），起点与目标间距 ≥2.0 m；`seed` 相同则场景完全相同（评估/复现用），
  `seed=-1` 时实际使用的种子会打印并记入 CSV。
- **目标点**：绿色圆盘纯视觉标记，无碰撞体，雷达看不见它（靠 supervisor 真值算 dist/bearing）。

## 动力学与碰撞（排障沉淀，改 .wbt 前必读）

小车是经过实测标定的标准动态差速模型，以下每条都踩过坑：

- **physics 放在 ROBOT 根节点**（Robot 是 Solid 子类）。根节点无 physics 会进入
  kinematic 模式：运动镜像失真且行为无法解释，坚决不要。
- **无 physics 的子 Solid（BODY）不要带 boundingObject**：会被 Webots 忽略
  （警告 "collisions will have no effect"）。碰撞几何集中在 ROBOT 的 boundingObject。
- **Webots 的 Cylinder 几何轴是局部 Y 轴**：轮子 endPoint 不需要任何 rotation；
  关节 `axis 0 1 0` 时正轮速 = 向 +x（车头）滚动，已经 probe_motion.py 实测。
- **四轮驱动底盘（现行）**：四个普通圆形车轮（HingeJoint + RotationalMotor），
  后轮位 (x=-0.08) 与前轮位 (x=+0.08) 各一对、y=±0.1，支撑面为矩形、姿态稳定。
  同侧前后轮由**同一个 (v,w) 指令**经差速运动学解出相同角速度——整车共享
  v 和 w，无独立轮速、非麦轮。视觉为黑轮胎+银轮毂双 Cylinder（宽 3 cm 探出车侧）；
  boundingObject 保持 h=0.02 r=0.05 圆柱不变。
- **轮子碰撞体必须包进车身碰撞盒**：ROBOT 根节点 boundingObject 为
  0.26×0.23×0.13 盒（z∈[0.02,0.15]），把四轮整个包进车身轮廓。轮子碰撞体
  凸出车身时，贴障滑行/转弯轮缘会钩住障碍棱边（μ=1 摩擦下电机拽不动，卡死到
  超时）；包进盒面后障碍只与光滑盒面接触，轮子只管滚地。外接圆 0.174 <
  robot_radius 0.18，判定半径不受影响。
- **历史教训（BallJoint 滚珠脚撑，已弃用）**：固定脚撑的静摩擦会锁死车身；
  滚珠球心悬空 5 mm 时车身加速后仰、滚珠拍地，运动中俯仰/横滚晃到 12~15°
  （GUI 里看到的"跑不平衡"）。换四轮矩形支撑后此类摇摆消失。
- **contactMaterial/ContactProperties 慎用于本车**：它只作用于"具名配对"，
  而 contactMaterial 还会被子 Solid 继承——曾把轮-地摩擦误砍导致全车"肉"。
  现版本全用默认摩擦（μ=1），不需要任何 ContactProperties。
- 动力学验收基准（probe_motion.py，seed=2008）：v=0.3 直线 ~1.5 s 加到指令速度；
  v=0 抱死即停；w=+1.0 原地旋转 yaw ≈ +0.09 rad/步（逆时针为正）。

## 常见问题

- **控制器报 `ModuleNotFoundError: websockets`**：venv 没建或没装依赖，回到"首次设置"。
- **Webots 用了错误的 python 解释器**：检查 `runtime.ini` 里的绝对路径是否存在。
- **client 连不上**：确认 Webots 已在运行（▶）且控制台打印了 server 启动行。
  注意 server 是**单客户端**设计：客户端断开（或测试脚本跑完）后 server 会关闭仿真，
  再跑任何 client 前都要先重开 Webots；也不要用"探测端口是否开放"的方式检查
  server——一次空连接再断开同样会触发关仿真。
- **想实时观看而不是快进仿真**：删掉 `env_server.py` 中的
  `simulationSetMode(Supervisor.SIMULATION_MODE_FAST)` 一行（训练时建议保留 FAST）。
- **雷达方向存疑**：看启动日志的标定误差；若换了雷达配置或场地，
  用 probe_motion.py / random_client.py 的四向读数对照画面复验。

## 给 002/006（真机适配）

真机驱动程序要替换的就是 env_server 的角色：同样的 WebSocket server、同样的消息格式。
需要对齐的物理量：轮半径 R、轮距 L、雷达安装高度与朝向、v_max/w_max、robot_radius。
麦克纳姆轮的 (v, w) → 四轮转速映射在真机侧驱动程序里做，协议层无感知。
