# rl_chassis — A2 项目 003 算法侧（SAC 训练/推理主循环）

VOA（Vision-Other-Action）强化学习防碰撞项目——线控底盘模型设计的**算法侧**交付。
协议与参数的唯一权威定义：`../api.md`（v1.1，三方共享）；网络接口约定：`Session-1_A2/api_doc.md`（009）。
本目录代码与两者严格对齐。

```
┌─────────────┐  WebSocket   ┌─────────────┐  进程内 import  ┌─────────────┐
│ 001 硬件侧   │   server     │ 003 算法侧   │                │ 009 模型侧   │
│ Webots 仿真  │◀════════════▶│ SAC 客户端   │───────────────▶│ 网络定义     │
│ (env server)│   JSON 消息   │ (client)    │◀───────────────│ (model.py)  │
└─────────────┘              └─────────────┘                └─────────────┘
```

003 是**唯一与 001 通信的一方**，进程内 import 009 的 `model.py`，只负责：
SAC 训练/推理主循环、观测归一化打包、动作缩放、奖励计算、经验存储、评估与收敛判定。

---

## 目录结构

```
003/
├── README.md            # 本文档
├── requirements.txt     # numpy / torch / websockets（网络仅 torch + 标准库，§7.1）
├── config.py            # 协议常量默认值 + 训练超参 + 奖励系数（唯一可调点）
├── obs_pack.py          # §4 观测打包/归一化 + 动作缩放
├── reward.py            # §6.4 奖励函数 + §6.5 终止掩码
├── buffer.py            # §6.3 replay buffer（环形数组，容量 200 000）
├── models.py            # 009 model.py 加载桥（可配置路径，缺失时回退内置桩）
├── sac.py               # §6.3 SAC 智能体（双 Q + soft update + α 自动调节）
├── train.py             # §3.3/§6 训练主循环（WebSocket client）
├── tools/
│   ├── model_stub.py    # 009 model.py 接口兼容桩（009 交付后删除，§7 全接口 + --selftest）
│   ├── echo_server.py   # §8.2 开发桩：假 env server（不依赖 Webots）
│   ├── selftest.py      # 003 离线自测（不连 server）
│   └── demo_run.py      # 桩 + 主循环的离线端到端联调
├── logs/                # 运行时生成：train.log / episodes.csv / eval.csv
└── checkpoints/         # 运行时生成：ckpt_{ep_N,best,final}.pt
```

## 首次设置与依赖

```powershell
cd D:\AllKindsofFiles\OfflinePractice_2026_Summer\Session-1-202608\A2\003
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

开发机已验证：Python 3.13 + torch 2.7（CPU）+ websockets 15 + numpy 2.1。

## 跑通流程（按顺序）

### 1. 离线自测（不依赖任何 server，30 秒）

```powershell
python tools/selftest.py
```

覆盖：观测打包（§4）、动作缩放（§4）、奖励（§6.4）、终止掩码（§6.5）、
buffer（§6.3）、SAC 单步更新 + checkpoint 存取（§7.3）。全绿打印 `SELFTEST OK`。

### 2. 桩联调（api.md §8.2，验证主循环全链路）

```powershell
python tools/demo_run.py --episodes 14
```

在事件循环内起 echo_server 桩（端口 8766，模拟 4×4 m 场地 + 障碍物 + 目标），
驱动完整训练主循环：hello 校验 → reset/obs/action → buffer → SAC 更新 →
周期性评估（确定性策略 + 固定种子）→ checkpoint 存盘 → all_finish/bye。
产物写 `logs/demo/`、`checkpoints/demo/`。

### 3. 真机联调（001 Webots）

001 在 Webots 中启动 `worlds\rl_arena.wbt`（server 监听 `ws://127.0.0.1:8765`），然后：

```powershell
python train.py --episodes 2000
```

课程学习默认开启（前 300 局 15 s 时限，之后放回 30 s，见 §2.2 config_override）；
`--no-curriculum` 关闭。训练中途 Ctrl+C 会先发 `all_finish(interrupted)` 再退出（§2.5）。

### 3b. 异步 RL（全 Python，2026-08-02 晚起，推荐）

多 Webots 实例方案已弃用（实测多实例全部卡死）。现在用纯 Python 仿真
`tools/pysim_server.py`（协议/物理与 env_server 对齐，api.md §5.1b），
一键起 12 个采集实例 + 1 个专职评估实例：

```bash
bash /home/zane/session_1/A2/real_robot/run_sim_async.sh        # 12 workers / 2000 局 / 从 0 + expert_v4 预填
bash /home/zane/session_1/A2/real_robot/run_sim_async.sh 12 2000 checkpoints/run_pysim/ckpt_best.pt   # 断点续训
```

要点：
- `train_async.py`：12 worker 并行采集共享 replay buffer，learner 独立线程
  1:1 UTD；learner 落后 >5000 条时 worker 背压暂停（不丢更新）。
- 评估走**独立实例**（默认 `ws://127.0.0.1:8873`）与采集并行，不再暂停 worker。
- 从 0 训练默认预填 `data/expert_v4.npz`（600 局 94% 成功、7.1 万条）；
  旧映射专家动作预填时自动做 `a0' = 2·clip(a0,0,1)−1` 变换（api.md §4）。
- checkpoint/日志在 `checkpoints/run_pysim` / `logs/run_pysim`。

**2026-08-02 晚架构/映射修复（v5）**：
- `LidarEncoder` 全局平均池化 `AdaptiveAvgPool1d(1)` 会抹掉障碍方向信息
  （v4 比 v2 差的根因），改为池化到 8 个方向 bin（45°/bin）后 flatten，
  雷达特征 32→256 维。
- 动作映射改仿射 `v=(a0+1)/2·v_max`：消除禁止倒车下 a0<0 全是"刹车"的死区。
- "小车不动"的奖励侧解释：不动 60 s 折扣后约 −50 ≫ 碰撞 −200，是理性局部
  最优；只有 critic 看得见到 +200 目标的路径才会动——方向信息修复是关键。

### 4. 真机物理小车微调（sim-to-real，08-02 起）

以 Webots 仿真收敛权重（`checkpoints/ckpt_sim_v2_best.pt`，68 维 MLP，
episode 1349 / update_step 146440 / 评估成功率 90%+，真机表现最好的基线）
为起点，在真机（`A2/real_robot` server，manual-drive 模式）上继续训练：

```bash
cd /home/zane/session_1/A2/003 && uv run python train.py --uri ws://192.168.43.114:8765 --resume checkpoints/ckpt_sim_v2_best.pt --warmup 256
```

> 08-02 晚注：v5（132 维 CNN + 仿射映射，纯 pysim 训练 100% 收敛）真机效果差，
> 已整套回退到 v2 的 68 维栈（model.py / obs_pack.py / config.py / 真机 server
> obs_dim=68）。v5 教训：纯 pysim（无噪声、解析雷达、理想响应）训出的策略
> 不过 Webots/真机——仿真收敛后必须先在 Webots 验收再上真机。

要点：
- 每局都是人工摆车（`record_goal` 摆目标点 → `drive_to_start` 遥控开到起点），
  不要用 `--auto-human`（会跳过等待）。
- `drive_to_start` 阶段已内嵌键盘遥控（08-02）：收到提示后直接在训练终端按
  W/S 前进后退、A/D 左右转、空格停止、Q 结束遥控，无需第二个终端
  （通过 server 的 teleop 消息驱动，不依赖 Pi 上的 ROS 环境）。
- `--warmup 256`：真机微调用仿真权重起步，critic 已训好，256 条经验即可开更，
  避免默认 5000 步空转（真机每步都要人工 reset，成本高）。
- 真机与仿真差异（雷达噪声、速度响应约 1.10x、转向动力学）由在线微调吸收，
  初期 SAC 探索噪声大，注意安全，随时 Ctrl+C。

## 各模块 ↔ api.md 映射

| api.md 章节 | 实现 | 说明 |
|---|---|---|
| §0 全局常量/默认值 | `config.py` | hello 下发的 config 才是权威，代码以 hello.config 为准 |
| §2.1 hello 校验 | `train.py: validate_hello` | 主版本一致 + obs_dim == 132，不符发 error 退出 |
| §2.2 reset / config_override | `train.py: reset_payload / curriculum_override` | 训练 seed=-1 随机；评估用固定种子；课程学习压短时限 |
| §2.4 action | `train.py: action_payload` | episode_id/step_id 严格回显所回应 obs |
| §2.5 all_finish / bye | `train.py: send_all_finish / _wait_bye` | 三时机 converged/interrupted/error；等 bye 5 s 超时 |
| §4 归一化打包 | `obs_pack.py: pack_obs / scale_action` | 唯一转换点；缩放只在 003 做（分工红线） |
| §6.3 SAC | `sac.py` + `buffer.py` | 见下节 |
| §6.4 奖励 | `reward.py: compute_reward` | 默认系数在 config.py，改动记录实验日志 |
| §6.5 终止掩码 | `reward.py: done_mask` | timeout → 0（bootstrap）；collision/goal → 1 |
| §6.6 日志 | `train.py: EpisodeLogger` | episodes.csv：`episode_id, steps, return, outcome, success` |
| §6.7 评估与存盘 | `train.py: evaluate / save_ckpt` | 每 50 局：确定性策略、seed 10000+i 跑 10 局；每 50 局存 ckpt；新高另存 best |
| §6.8 收敛 | `train.py` | 连续 3 次评估成功率 ≥90% → all_finish(converged) |
| §7 网络接口 | `models.py`（009 交付物） | 见「与 009 的协作」 |

## SAC 实现要点（§6.3）

- replay buffer 容量 200 000，存 `(vec, a01, r, next_vec, done_mask)`，float32 环形数组（`buffer.py`）
- 双 Q + target，soft update `τ=0.005`（`sac.py` 每步更新后执行）
- α 自动调节：`log_alpha` 走 Adam（lr 同 3e-4），target entropy = −ACT_DIM = −2
- `lr=3e-4`，`batch=256`，`γ=0.99`；**buffer < 5000 条不更新**（warmup）
- 每 episode 结束后做 `K = 本 episode 步数` 次梯度更新（§3.3，约 1:1 更新比）
- 奖励公式（§6.4，系数默认值可调）：

```
r = 5.0 × (dist_{t-1} − dist_t) + 200·[goal_reached] − 200·[collision]
    − 0.1 − 0.5·(a0−a0_prev)² − 0.5·(a1−a1_prev)²
```

## 观测打包（§4，003 唯一归一化点）

68 维向量，顺序固定（`obs_pack.py`）：

| 下标 | 公式 | 范围 |
|---|---|---|
| [0:64] | lidar[i] / lidar_max_range（先 clip） | [0, 1] |
| [64] | min(dist, 10) / 10 | [0, 1] |
| [65] | bearing / π | (-1, 1] |
| [66] | clip(v, ±v_max) / v_max | [-1, 1] |
| [67] | clip(w, ±w_max) / w_max | [-1, 1] |

动作映射（§4 分工红线，只在此缩放）：`v = a0·v_max`，`w = a1·w_max`。
**未采用帧堆叠/循环结构**（v1.1 与协议解耦，属 003/009 内部设计）；后续如需，
帧堆叠在 `train.py` 的 run_episode 内改两行即可，网络输入变为 68×N，hello 的 obs_dim 不受影响。

## 课程学习（§2.2 config_override）

前 `--cur-short-episodes`（默认 300）局用 `config_override={"max_episode_time": 15.0}`
压短时限，让每局快速结束、单位时间积累更多经验；之后放回默认 30 s。
评估局一律用完整时限（衡量真实任务表现）。只允许覆盖 `max_episode_time`（协议锁死其他键）。

## 与 009 的协作（model.py 加载）

**009 已交付**（`../009/model.py`，接口自测通过）。`models.py` 按优先级自动加载
（幂等，启动日志打印实际来源）：

1. 环境变量 `MODEL_MODULE=<path>` 显式指定
2. 003 根目录或上级目录存在 `model.py`
3. 同级 `009/model.py`（默认三方布局 `A2/{001,003,009}`，当前生效路径）
4. 回退 `tools/model_stub.py` 内置桩（接口与 api_doc.md §7 完全一致，含 `--selftest`；
   仅当 009 未交付或上述路径均不存在时使用，桩训练产物视为开发期预验证）

加载后校验 `OBS_DIM==68`、`ACT_DIM==2`（§6.1），不符拒绝启动。

**注意**：桩与正式 model.py 层名不同（`trunk.*` vs `feature_net.*`），
两者 checkpoint 不能互相 `--resume`；正式训练以 009 交付版为准，从零开始。

## 日志与 checkpoint

- `logs/train.log`：INFO 级全程记录（含每次评估、每次存盘、模型来源、hello 摘要）
- `logs/episodes.csv`：§6.6 每 episode 一行
- `logs/eval.csv`：每次评估一行（训练 ep、成功率、成功局数）
- `checkpoints/ckpt_{tag}.pt`：§7.3 格式（5 个 state_dict + optimizers + meta），
  tag：`ep_{N}`（每 50 局）/ `best`（成功率新高）/ `final`（退出时）
- 恢复训练：`python train.py --resume checkpoints/ckpt_best.pt`
  （从 meta 的 episode 之后继续，优化器状态一并恢复）

## 实验记录约定

每轮实验（改系数/超参/课程学习配置/网络结构）在 `logs/` 下建独立子目录并记录：
协议版本、模型来源（009 or 桩）、奖励系数、超参、课程学习开关、起止时间、结论。
`config.py` 中改动默认系数即视为开新实验。

### 记录

| run | 协议 | 模型 | 改动 | 课程学习 | 起止 | 结论 |
|---|---|---|---|---|---|---|
| run1 | v1.1 | 009 正式 | 无（api.md §6.4 原版奖励） | 前 300 局 15 s | 08-01 17:16 ~ 17:40 | **不收敛**。574 局后停：ep 301-460 碰撞率 94.1%，eval@49~449 连续 9 次成功率 0%。α 衰减至 0.068 探索枯竭；诊断：稀疏 −200 无梯度 + W_APPROACH=5 诱导直线冲锋穿障 |
| run2 | v1.1 | 009 正式 | 奖励 + 障碍接近惩罚 `−3.0×max(0, 0.5−min(lidar))` | 同 run1 | 08-01 17:41 ~ 17:52 | **仍不收敛**，ep ~400 停：eval@199~349 连续 4 次 0%，ep 301-400 碰撞率 95%。机理：W_DANGER=3 < W_APPROACH=5，危险区内冲锋近视收益仍为正（+0.25 − 0.15/步），α 塌缩前 Q 来不及纠偏 |
| run3 | v1.1 | 009 正式 | W_DANGER 3.0→**8.0**（> W_APPROACH，翻转局部激励） | 同 run1 | 08-01 17:5x ~ 18:2x | **无效实验（环境 bug 下无意义）**。214 局 0 成功。后续定位：run1-3 全部失败的主因不是奖励，而是四个环境 bug（见下表注） |
| run4 | v1.1 | 009 正式 | 环境四修复（见下注）+ obs 中位数滤波；奖励同 run3 | 同 run1 | 08-01 20:0x ~ 20:41 | **不收敛：冷启动不足**。376 局 0 成功，eval@99~349 连续 7 次 0%。起跑健康（前 100 局 77% 超时，run1 同期 94% 碰撞），但无成功样本 → 价值传播无梯度，判定必须预填演示数据 |
| run5 | v1.1 | 009 正式 | 四轮底盘 + 时限 30→60 s（课程短阶段 15→30 s）+ **演示数据预填**（demo_v1.npz 32186 条，77% 成功轨迹） | 前 300 局 30 s | 08-01 20:4x ~ 21:32 | **首次见收敛信号**：eval 0→40%→50%→**80%@199（峰值）**，后回落 60~70%。367 局后被轮子钩挂修复打断，best@199 权重保留 |
| run6 | v1.1 | 009 正式 | **碰撞盒屏蔽四轮**（0.26×0.23×0.13 盒包进轮子，障碍只碰光滑盒面）+ resume run5@199 + 预填 | 同 run5 | 08-01 21:3x ~ 21:55 | 换底盘 critic 重拟合：eval@249=40%→@299=60%→@349=40%。修复后脚本专家复验 **84%（+7pp）、碰撞 2→0**。180 局后被四驱改造打断，best@299 交 run7 |
| run7 | v1.1 | 009 正式 | **四驱底盘**（前轮加电机，四轮共享同一 (v,w) 指令；圆形车轮视觉，物理参数不变）+ resume run6@299 + 预填 | 同 run5 | 08-01 21:56 ~ 22:45 | **80% 平台**：eval 连续 5 次 0.8（90→50→70→80→60→90→80×4）。逐种子诊断（best 权重）：3 个硬种子稳定失败——10001 碰撞、10005/10009 超时。判定演示锚定在专家水平（专家自身 84%），改纯 RL |
| run8 | v1.1 | 009 正式 | **纯 RL 续训**（去演示预填，解除专家锚定）+ resume run7@349(90%) | 同 run5 | 08-01 22:51 ~ 23:17 | 未收敛。eval 0.7→0.6→0.9→0.8→0.8→0.5→0.6（0.5~0.9 振荡），350 局。振荡说明价值/策略在临界态反复横跳，转异步 RL 提速 |
| run9 | v1.1 | 009 正式 | **异步 RL**（train_async.py 4 worker 并行采集 + learner 线程 1:1 UTD，env_server 端口化 ENV_WS_PORT）纯 RL + resume run8@499(90%) | 同 run5 | 08-01 23:21 ~ 23:33 | 未收敛。eval 0.8→0.6→0.6→0.8，~55 局/分（单实例 5 倍）。ep690 处被夜间电脑休眠打断 |
| run10 | v1.1 | 009 正式 | 异步 4 worker 纯 RL + resume run9@649(80%) | 同 run5 | 08-02 09:10 ~ 09:12 | 未收敛。仅 1 次评估（0.8@499），被无轮底盘改造打断 |
| run11 | v1.1 | 009 正式 | **无轮运动学底盘**（圆柱机身，删除四轮/电机/physics，(v,w) 直接积分，碰撞仍雷达判定）+ 异步 4 worker 纯 RL + resume run10@499(80%) | 同 run5 | 08-02 09:18 ~ 09:24 | **收敛 ✓（warm-start 迁移验证）**。eval 0.8→**0.9×3 连续**（@549/@599/@649），654 局、18364 次更新后自动 all_finish。运动学底盘消除动力学噪声，策略平滑迁移即达 90% 门槛 |
| run12 | v1.1 | 009 正式 | **新世界从零训练**：无轮运动学底盘 + 新采集专家数据（1000 局 90.5% 成功、0 碰撞，14.35 万条过渡，`scripted_expert --dump`）+ **buffer 容量 20 万→50 万**（防演示主导采样）+ 异步 4 worker | 同 run5 | 08-02 10:16 ~ 10:45 | **收敛 ✓（从零复现）**。eval：0%→50%@199→10%@249/299（SACfD 早期波动）→70%@349→**90/100/90/100/90**（@399~@649 连续 3 次 ≥90%@549/599/649）。654 局、88756 次更新自动 all_finish。最终权重 `checkpoints/final_best.pt`（= run12/ckpt_best） |
| run13 (v5) | v1.1 | 009 正式 | **全 Python 异步 + 架构/映射修复**：pysim 12 worker + 独立评估实例 + 背压；LidarEncoder 全局池化→8 方向 bin（修复方向信息丢失，v4 不如 v2 的根因）；动作映射 v=(a0+1)/2·v_max（消除禁倒车死区）；expert_v4（600 局 94%、7.1 万条，死局修复后采集）预填 + 旧映射动作自动变换 | 固定难场景（5-8 障碍/2m 目标，无课程） | 08-02 22:23 ~ 22:45 | **收敛 ✓（22 分钟）**。eval：0%×3→10%@149→0%@199→70%@249→**90/90/100**（@299/349/399 连续 3 次 ≥90%）。422 局、69049 次更新自动停。最终权重 `checkpoints/ckpt_sim_v5_best.pt`（= run_pysim/ckpt_best） |

> **环境四根因（run1-3 共 1188 局 0 成功的实锤结论，001 侧修复）**：
> 1. 目标标记光柱对雷达可见 → 成功圈物理不可达（碰撞优先于 goal_reached），光柱抬出雷达平面；
> 2. 运动中孤立单射线尖峰（0.05~0.18 m 假回波）→ 3 邻域中位数滤波，碰撞/obs/奖励全链路统一；
> 3. **雷达数组左右镜像**：标定场景镜像对称（车放 (1.5,0)，左右墙等距），CCW/CW 拟合误差相同导致方向蒙错，obs 里障碍左右全反——标定改到 (1.5,0.7) 后 CW 46mm vs CCW 369mm 实锤；
> 4. 脚本专家走廊用角度邻域（±11°）量纲错误，只在 ≥0.92 m 处盖得住 0.18 m 车身 → 改米制矩形走廊（R_CLEAR=0.30 m）。
> 修复后脚本专家在线验收 **45/50 = 90%**（门槛 70%），碰撞全部复核为真实接触。

## 常见问题

- **连不上 server**：确认 001 Webots 已运行（▶）且控制台有 `WebSocket server 已启动`
  日志；server 是**单客户端**设计——任何 client 断开都会让它关仿真，测试后需重启 Webots。
  开发期用 `--uri ws://127.0.0.1:8766` 连桩。
- **动作偏大/飘**：确认 `pack_obs`/`scale_action` 用的是 hello.config 的
  v_max/w_max/lidar_max_range，而不是 config.py 默认值（两者当前一致，但真机可能不同）。
- **`ModelNotFoundError` 之外想调试 009 桩**：`python tools/model_stub.py --selftest`。
- **想实时观看仿真而不是快进**：001 侧删掉 env_server.py 的
  `SIMULATION_MODE_FAST` 一行（训练时建议保留 FAST）。
