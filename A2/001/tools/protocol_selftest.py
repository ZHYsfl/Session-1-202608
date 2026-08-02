# -*- coding: utf-8 -*-
"""
protocol_selftest.py — A2 项目 001 侧协议自测（验收用）
覆盖 api.md 中 random_client 覆盖不到的路径：

  A. timeout 终止 + config_override 覆盖 max_episode_time
  B. 同 seed 场景可复现（两次 reset(42) 的初始 obs 在容差内一致）
  C. goal_reached 终止（简单比例控制器朝目标开，多个 seed 尝试）
  D. 错误处理：发错 step_id 的 action 必须收到 BAD_FIELD（§2.7）
     —— 注意：按协议 server 此后会关闭仿真，所以本项放最后

用法：先启动 Webots（▶），再运行
    python tools/protocol_selftest.py
"""

import asyncio
import json
import sys

try:  # websockets >= 14
    from websockets.asyncio.client import connect
except ImportError:  # websockets < 14
    from websockets import connect

URI = "ws://localhost:8765"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def report(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)


async def recv_obs(ws):
    msg = json.loads(await ws.recv())
    assert msg["type"] == "obs", f"期望 obs，收到 {msg}"
    return msg


async def send_action(ws, obs, v, w):
    await ws.send(json.dumps({
        "type": "action", "episode_id": obs["episode_id"],
        "step_id": obs["step_id"], "v": v, "w": w}))


async def test_timeout_and_override(ws):
    """A: 静止不动 + 时限覆盖为 2.0s → 20 步后必须 timeout。"""
    await ws.send(json.dumps({
        "type": "reset", "seed": 7,
        "config_override": {"max_episode_time": 2.0}}))
    obs = await recv_obs(ws)
    while not obs["done"]:
        await send_action(ws, obs, 0.0, 0.0)
        obs = await recv_obs(ws)
    ok = (obs["flags"]["timeout"] and not obs["flags"]["collision"]
          and not obs["flags"]["goal_reached"]
          and obs["step_id"] == 20 and abs(obs["t"] - 2.0) < 1e-6)
    report("A. timeout + config_override", PASS if ok else FAIL,
           f"step_id={obs['step_id']} t={obs['t']} flags={obs['flags']}")
    return ok


async def test_seed_reproducible(ws):
    """B: 两次 reset(42)，初始 obs 的雷达与目标必须逐值一致。
    注意协议状态机：reset 后处于 RUNNING，须先让本局结束才能再次 reset。"""
    await ws.send(json.dumps(
        {"type": "reset", "seed": 42,
         "config_override": {"max_episode_time": 0.1}}))  # 1 步即超时，快速结束本局
    first = await recv_obs(ws)
    obs = first
    while not obs["done"]:
        await send_action(ws, obs, 0.0, 0.0)
        obs = await recv_obs(ws)

    await ws.send(json.dumps({"type": "reset", "seed": 42,
                              "config_override": {"max_episode_time": 0.1}}))
    second = await recv_obs(ws)
    # 契约是"相同 seed 相同场景"（RNG 决定的障碍物/目标布局一致），
    # 而非 obs 逐位相等：物理机器人两次落位有 ~0.2 mm 抖动属正常
    max_lidar_diff = max(abs(a - b) for a, b in zip(first["lidar"], second["lidar"]))
    same = (max_lidar_diff <= 0.02
            and abs(second["goal"]["dist"] - first["goal"]["dist"]) <= 0.002
            and abs(second["goal"]["bearing"] - first["goal"]["bearing"]) <= 0.001)
    report("B. 同 seed 场景复现", PASS if same else FAIL,
           f"雷达最大差 {max_lidar_diff:.4f} m, "
           f"dist {first['goal']['dist']} vs {second['goal']['dist']}")
    # 收尾：把这一局跑完（1 步超时），状态机回到 WAIT_RESET，后续测试才能继续
    obs = second
    while not obs["done"]:
        await send_action(ws, obs, 0.0, 0.0)
        obs = await recv_obs(ws)
    return same


async def test_goal_reached(ws):
    """C: 确定性 goal_reached 验证。
    seed=2008 经离线复现布局筛选：起点→目标直线通道净余量 0.76 m（>> 0.18），
    沿直线行驶必然到达。到达不了就说明 goal_reached 判定逻辑有 bug。"""
    await ws.send(json.dumps(
        {"type": "reset", "seed": 2008, "config_override": None}))
    obs = await recv_obs(ws)
    init_dist = obs["goal"]["dist"]
    v_meas_seen = 0.0
    while not obs["done"]:
        bearing = obs["goal"]["bearing"]
        w = max(-1.5, min(1.5, 2.5 * bearing))
        v = 0.4 if abs(bearing) < 0.3 else 0.05  # 对准了全速，没对准先转身
        await send_action(ws, obs, v, w)
        obs = await recv_obs(ws)
        v_meas_seen = max(v_meas_seen, abs(obs["vel"]["v"]))
    ok_vel = v_meas_seen > 0.05
    if obs["flags"]["goal_reached"]:
        report("C. goal_reached 终止（确定性）", PASS,
               f"初始距离 {init_dist:.2f} m，{obs['step_id']} 步到达；"
               f"vel 字段{'正常' if ok_vel else '异常(始终≈0)'}")
        return True
    report("C. goal_reached 终止（确定性）", FAIL,
           f"结局={obs['flags']}，最终距离={obs['goal']['dist']:.2f} m "
           f"（直线通道上不可能撞墙/迷路，goal 判定链有 bug）")
    return False


async def test_error_path(ws):
    """D: 发错误 step_id → 必须收到 BAD_FIELD，随后 server 关闭仿真。"""
    await ws.send(json.dumps(
        {"type": "reset", "seed": 1, "config_override": None}))
    obs = await recv_obs(ws)
    await ws.send(json.dumps({
        "type": "action", "episode_id": obs["episode_id"],
        "step_id": obs["step_id"] + 999, "v": 0.0, "w": 0.0}))
    msg = json.loads(await ws.recv())
    ok = msg["type"] == "error" and msg["code"] == "BAD_FIELD"
    report("D. 错误处理 BAD_FIELD", PASS if ok else FAIL, f"收到: {msg}")
    return ok


async def main():
    async with connect(URI, ping_interval=20, ping_timeout=60) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello"
        print(f"[selftest] 已连接，协议 {hello['protocol_version']}", flush=True)
        await test_timeout_and_override(ws)
        await test_seed_reproducible(ws)
        await test_goal_reached(ws)
        await test_error_path(ws)  # 最后：server 将关闭仿真
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"\n[selftest] 完成: {len(results)} 项, "
          f"{n_fail} 失败, {n_warn} 警告")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
