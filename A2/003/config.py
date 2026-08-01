# -*- coding: utf-8 -*-
"""
config.py — A2 项目（VOA 强化学习防碰撞）003 算法侧全局配置

协议常量的权威定义在 api.md §0（server hello 会逐项下发），
本文件中的协议默认值仅用于离线开发/自测；连接后以 hello.config 为准。
训练超参与流程参数对应 api.md §6.3 ~ §6.8。

改动奖励系数 / 超参后，请记录进实验日志（README「实验记录」一节）。
"""

# ================= 协议常量默认值（api.md §0，与 hello.config 键名一致） =================
PROTOCOL_VERSION = "1.1"        # 与 server 握手的最低兼容版本（主版本必须一致）
ENV_NAME = "webots_diffbot_v1"  # 期望的环境标识，不符仅告警（以协议版本为准）

LIDAR_COUNT = 64
LIDAR_MAX_RANGE = 3.5
CONTROL_DT = 0.1
MAX_EPISODE_TIME = 30.0
V_MAX = 0.5
W_MAX = 1.5
GOAL_TOLERANCE = 0.15
ROBOT_RADIUS = 0.18
ARENA_SIZE = 4.0
OBS_DIM = 68
ACT_DIM = 2

# 观测归一化常数（api.md §4）
DIST_NORM = 10.0                # goal.dist 归一化除数（≈ 场地对角线 2 倍）

# ================= 训练超参（api.md §6.3，改动需做消融记录） =================
BUFFER_CAPACITY = 200_000       # replay buffer 容量
WARMUP = 5_000                  # buffer 少于该条数不做梯度更新
BATCH_SIZE = 256
LR = 3e-4                       # actor / critic / alpha 统一学习率
GAMMA = 0.99
TAU = 0.005                     # target 网络 soft update 系数
ALPHA_LOG_INIT = 0.0            # log_alpha 初值 → α = exp(0) = 1.0
TARGET_ENTROPY = -float(ACT_DIM)  # α 自动调节目标熵 = -ACT_DIM = -2

# ================= 奖励系数（api.md §6.4 默认值，改动记录进实验日志） =================
W_APPROACH = 5.0                # × (dist_{t-1} − dist_t)，朝目标靠近
R_GOAL_REACHED = 200.0          # 到达目标
R_COLLISION = -200.0            # 碰撞
R_TIME_STEP = -0.1              # 每步时间惩罚
W_SMOOTH = 0.5                  # × (a − a_prev)²，动作平滑惩罚（v、w 各一份）
# 密集障碍接近惩罚（run2 新增）：min(lidar) 进入 D_SAFE 内开始按深度线性惩罚，
# 给"离碰撞还有多远"一个连续梯度——run1 纯稀疏 -200 无法告诉策略撞之前哪步开始错的。
D_SAFE = 0.5                    # 危险区阈值（m）；> robot_radius 0.18，留出可回旋区间
W_DANGER = 3.0                  # × max(0, D_SAFE − min(lidar))，每步；碰撞边界处约 0.96/步

# ================= 训练流程（api.md §6.6 ~ §6.8） =================
MAX_EPISODES = 2_000            # 训练 episode 上限（收敛会提前 break）
EVAL_INTERVAL = 50              # 每 N 个训练 episode 评估一次
EVAL_EPISODES = 10              # 每次评估局数
EVAL_SEED_BASE = 10_000         # 评估固定种子 = EVAL_SEED_BASE + i（i = 0..EVAL_EPISODES-1）
CONVERGE_CONSECUTIVE = 3        # 连续多少次评估达到收敛线
CONVERGE_RATE = 0.9             # 成功率收敛线（≥90%）
CKPT_INTERVAL = 50              # 每 N 个训练 episode 存一次 checkpoint

# ================= 课程学习（api.md §2.2 config_override） =================
# 前期把时限压短（如 15 s），每局快速结束、单位时间积累更多经验；
# 过了前 CUR_SHORT_EPISODES 局后放宽回完整时限（30 s）。
CUR_SHORT_EPISODES = 300
CUR_SHORT_TIME = 15.0

# ================= 连接 =================
WS_URI = "ws://127.0.0.1:8765"
PING_INTERVAL = 20              # 应用层不做心跳（§0），仅保持库级 ping 参数
PING_TIMEOUT = 60
