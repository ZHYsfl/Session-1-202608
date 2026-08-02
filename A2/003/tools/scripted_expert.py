# -*- coding: utf-8 -*-
"""
scripted_expert.py — 脚本专家（纯追踪 + 反应式避障）基线诊断

用途：
  1) 环境验收：不经过任何学习算法，直接用几何控制律开车。
     若脚本专家成功率 ≥ ~70%，说明 env/物理/观测/奖励链路是通的，
     训不起来就是 SAC 侧的问题；若脚本专家都跑不到，先修环境。
  2) 冷启动数据采集（--dump）：把 (vec, a01, r, next_vec, done_mask)
     转存成 npz，供 train.py 预填 replay buffer（SACfD 思路）。

控制律（VFH：矩形走廊按米衡量，对 obs 逐步反应，无状态）：
  对每个候选航向 θ_k（|θ| ≤ HEADING_MAX），把所有雷达命中点投影到该航向
  坐标系，横向 |lat| < R_CLEAR（车身半径 0.18 + 余量）且纵向 lon > 0 的点
  限制可通行距离：clear_k = min(lon)。
    score_k = min(clear_k, D_SCORE) + GOAL_BIAS·cos(θ_k − bearing)
  取分最高的航向：w = K_W·θ；v 随 clear 缩放，clear < 0.22 m 时原地转。

  实战教训：走廊宽度必须按"米"衡量。曾经用角度邻域（±2 射线）当走廊，
  ±11° 只在 ≥0.92 m 处盖得住 0.18 m 的车身半径，更近时航向正前方没障碍
  就全速过，车身侧面撞上去——离线复现（同律同场景同撞）定位的这个 bug。

lidar 约定（api.md §4）：64 线，第 k 条角度 = k×5.625° 逆时针，index 0 = 正前方。
"""

import argparse
import asyncio
import json
import math

import numpy as np
import websockets

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WS_URI, V_MAX, W_MAX
from obs_pack import pack_obs
from reward import compute_reward, done_mask

K_W = 2.0                 # 追踪转向增益
V_SCALE = 0.6             # 全局限速（× v_max）：0.1s 控制周期 + 轮速滞后下留反应距离
D_SCORE = 1.2             # 净空评分截断距离（m）
GOAL_BIAS = 0.8           # 目标偏向权重（与净空同量纲）
R_CLEAR = 0.30            # 矩形走廊半宽（m）：车身半径 0.18 + 0.12 余量
HEADING_MAX = math.radians(100)   # 候选航向范围（不倒车）


def _signed_angle(k: int, n: int) -> float:
    """第 k 条射线的带符号机体系角度（rad），k>n/2 折到负侧。"""
    step = 2 * math.pi / n
    return k * step if k <= n // 2 else (k - n) * step


def _corridor_clearance(lidar, k: int) -> float:
    """沿候选航向 k 的自由距离：命中点投影到航向系，|lat|<R_CLEAR 且 lon>0 取 min(lon)。"""
    n = len(lidar)
    step = 2 * math.pi / n
    clear = D_SCORE
    for j in range(n):
        d = lidar[j]
        if d >= 3.5:              # LIDAR_MAX_RANGE：无回波
            continue
        diff = math.atan2(math.sin((j - k) * step), math.cos((j - k) * step))
        lon = d * math.cos(diff)
        lat = d * math.sin(diff)
        if lon > 0 and abs(lat) < R_CLEAR:
            if lon < clear:
                clear = lon
    return clear


def scripted_action(obs: dict, invert: bool = False) -> tuple:
    """返回 (v, w) 物理指令。invert=True 把航向左右反转（雷达镜像验证用）。"""
    lidar = obs["lidar"]
    bearing = obs["goal"]["bearing"]
    dist = obs["goal"]["dist"]
    n = len(lidar)

    best_k, best_s, best_clear = 0, -1e18, 0.0
    for k in range(n):
        th = _signed_angle(k, n)
        if abs(th) > HEADING_MAX:
            continue
        clear = _corridor_clearance(lidar, k)
        s = min(clear, D_SCORE) + GOAL_BIAS * math.cos(th - bearing)
        if s > best_s:
            best_s, best_k, best_clear = s, k, clear

    err = _signed_angle(best_k, n)
    if invert:
        err = -err
    w = max(-W_MAX, min(W_MAX, K_W * err))

    if best_clear < 0.22:
        v = 0.0                       # 贴太近：原地转出去
    else:
        v = V_SCALE * V_MAX * max(0.15, min(1.0, (best_clear - 0.25) / 0.5))
        v *= max(0.2, math.cos(err))  # 大转角时减速
    if dist < 0.4:                    # 近目标减速，防止冲过 0.15 m 容差圈
        v = min(v, max(0.15, 1.5 * dist))
    return v, w


async def run(args):
    outcomes = {}
    returns, finals = [], []
    dump = [] if args.dump else None

    async with websockets.connect(args.uri, ping_interval=20,
                                  ping_timeout=60, max_size=1 << 20) as ws:
        hello = json.loads(await ws.recv())
        assert hello.get("type") == "hello", f"首条不是 hello: {hello}"
        cfg = hello["config"]
        print(f"hello OK: {hello.get('env')} v{hello.get('protocol_version')}")

        for ep in range(args.episodes):
            seed = args.seed_base + ep
            await ws.send(json.dumps(
                {"type": "reset", "seed": seed, "config_override": None}))
            obs = json.loads(await ws.recv())
            assert obs["type"] == "obs", f"reset 后未收到 obs: {obs}"

            ret, steps = 0.0, 0
            a01_prev = np.zeros(2, dtype=np.float32)
            while not obs["done"]:
                v, w = scripted_action(obs, invert=args.invert)
                a01 = np.array([v / cfg["v_max"], w / cfg["w_max"]],
                               dtype=np.float32)
                await ws.send(json.dumps(
                    {"type": "action", "episode_id": obs["episode_id"],
                     "step_id": obs["step_id"], "v": v, "w": w}))
                nxt = json.loads(await ws.recv())
                assert nxt["type"] == "obs", f"action 后未收到 obs: {nxt}"

                if dump is not None:
                    r = compute_reward(obs, a01, a01_prev, nxt)
                    dump.append((pack_obs(obs, cfg), a01.copy(), r,
                                 pack_obs(nxt, cfg), done_mask(nxt)))
                    ret += r
                a01_prev = a01
                obs = nxt
                steps += 1

            f = obs["flags"]
            outcome = ("collision" if f["collision"]
                       else "goal_reached" if f["goal_reached"] else "timeout")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            finals.append(obs["goal"]["dist"])
            if ep % 10 == 0 or outcome != "goal_reached":
                print(f"ep {ep:3d} seed={seed} | {outcome:12s} | steps={steps} "
                      f"| final_dist={obs['goal']['dist']:.2f}", flush=True)

        await ws.send(json.dumps(
            {"type": "all_finish", "reason": "converged",
             "total_episodes": args.episodes}))
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except Exception:
            pass

    n = args.episodes
    succ = outcomes.get("goal_reached", 0)
    print(f"\n== 结果: {n} 局 | 成功 {succ} ({100*succ/n:.0f}%) "
          f"| 碰撞 {outcomes.get('collision', 0)} | 超时 {outcomes.get('timeout', 0)} "
          f"| 平均终距 {sum(finals)/n:.2f} m ==")

    if dump is not None:
        arr = {k: np.stack([t[i] for t in dump])
               for i, k in enumerate(("obs", "act", "rew", "next_obs", "done"))}
        np.savez_compressed(args.dump, **arr)
        print(f"演示数据已存: {args.dump} ({len(dump)} 条, "
          f"回报均值 {ret:.1f})" if args.episodes == 1 else
              f"演示数据已存: {args.dump} ({len(dump)} 条)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default=WS_URI)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=20000)
    p.add_argument("--dump", default=None,
                   help="把轨迹按 SAC buffer 格式转存 npz（冷启动预填用）")
    p.add_argument("--invert", action="store_true",
                   help="避障转向左右反转：若反转后成功率剧增，证明雷达数组镜像")
    asyncio.run(run(p.parse_args()))
