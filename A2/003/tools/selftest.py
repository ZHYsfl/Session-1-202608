# -*- coding: utf-8 -*-
"""
selftest.py — 003 离线自测（不连 server、不依赖 Webots）

覆盖：观测打包（§4）、动作缩放（§4）、奖励（§6.4）、终止掩码（§6.5）、
replay buffer（§6.3）、SAC 单步更新、checkpoint 存取（§7.3）。

用法：python tools/selftest.py   （全绿打印 SELFTEST OK）
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # 003 根目录
sys.path.insert(0, str(Path(__file__).parent))         # tools/（model_stub）

import numpy as np
import torch

from buffer import ReplayBuffer
from config import (ACT_DIM, BATCH_SIZE, GAMMA, OBS_DIM, TAU)
from models import get_model_module
from obs_pack import pack_obs, scale_action
from reward import compute_reward, done_mask

MODEL = None

CONFIG = {"lidar_count": 64, "lidar_max_range": 3.5, "obs_dim": 68,
          "act_dim": 2, "control_dt": 0.1, "max_episode_time": 30.0,
          "v_max": 0.5, "w_max": 1.5, "goal_tolerance": 0.15,
          "robot_radius": 0.18, "arena_size": 4.0}


def fake_obs(lidar=None, dist=2.0, bearing=0.0, v=0.0, w=0.0,
             collision=False, goal_reached=False, timeout=False, ep=1, s=0):
    if lidar is None:
        lidar = [3.5] * 64
    return {"type": "obs", "episode_id": ep, "step_id": s, "t": s * 0.1,
            "lidar": lidar,
            "goal": {"dist": dist, "bearing": bearing},
            "vel": {"v": v, "w": w},
            "flags": {"collision": collision, "goal_reached": goal_reached,
                      "timeout": timeout},
            "done": collision or goal_reached or timeout}


def check(name, cond, detail=""):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name} {detail}")


def test_pack_obs():
    print("[1/6] 观测打包 §4 …")
    lidar = [0.5] * 64
    o = fake_obs(lidar=lidar, dist=2.3, bearing=-0.44, v=0.15, w=0.0)
    vec = pack_obs(o, CONFIG)
    check("维度 = 68", vec.shape == (68,), str(vec.shape))
    check("dtype float32", vec.dtype == np.float32, str(vec.dtype))
    check("lidar 段 = 0.5/3.5", abs(vec[0] - 0.5 / 3.5) < 1e-6)
    check("dist 段", abs(vec[64] - 0.23) < 1e-6)
    check("bearing 段", abs(vec[65] - (-0.44 / np.pi)) < 1e-6)
    check("vel.v 段", abs(vec[66] - 0.3) < 1e-6)
    check("vel.w 段", abs(vec[67]) < 1e-6)
    # 越界值被 clip
    o2 = fake_obs(lidar=[3.5] * 64, dist=50.0, bearing=3.2, v=9.0, w=-9.0)
    v2 = pack_obs(o2, CONFIG)
    check("dist 上限 1.0", v2[64] == 1.0)
    check("v 上限 1.0", v2[66] == 1.0)
    check("w 下限 -1.0", v2[67] == -1.0)


def test_scale_action():
    print("[2/6] 动作缩放 §4 …")
    v, w = scale_action(np.array([1.0, -1.0]), CONFIG)
    check("(1,-1) → (0.5,-1.5)", abs(v - 0.5) < 1e-9 and abs(w + 1.5) < 1e-9)
    v, w = scale_action(np.array([0.0, 0.0]), CONFIG)
    check("(0,0) → (0,0)", v == 0.0 and w == 0.0)


def test_reward():
    print("[3/6] 奖励函数 §6.4 …")
    o_prev = fake_obs(dist=2.0)
    a_prev = np.array([0.1, 0.1])
    # 靠近 0.1 m，动作不变
    o_next = fake_obs(dist=1.9, goal_reached=False)
    r = compute_reward(o_prev, a_prev, a_prev, o_next)
    check("靠近奖励 5×0.1 − 0.1 = 0.4", abs(r - 0.4) < 1e-9, str(r))
    # 到达目标
    o_next = fake_obs(dist=0.1, goal_reached=True)
    r = compute_reward(o_prev, a_prev, a_prev, o_next)
    check("到达 +200", abs(r - (5.0 * 1.9 + 200 - 0.1)) < 1e-6, str(r))
    # 碰撞
    o_next = fake_obs(dist=1.5, collision=True)
    r = compute_reward(o_prev, a_prev, a_prev, o_next)
    check("碰撞 −200", abs(r - (5.0 * 0.5 - 200 - 0.1)) < 1e-6, str(r))
    # 平滑惩罚：动作突变 0.1 → 0.9
    o_next = fake_obs(dist=1.95)
    a_new = np.array([0.9, 0.1])
    r = compute_reward(o_prev, a_new, a_prev, o_next)
    expect = 5.0 * 0.05 - 0.1 - 0.5 * 0.8 ** 2
    check("平滑惩罚 0.5×0.64", abs(r - expect) < 1e-9, str(r))


def test_done_mask():
    print("[4/6] 终止掩码 §6.5 …")
    check("collision → 1", done_mask(fake_obs(collision=True)) == 1.0)
    check("goal_reached → 1", done_mask(fake_obs(goal_reached=True)) == 1.0)
    check("timeout → 0", done_mask(fake_obs(timeout=True)) == 0.0)
    check("未终止 → 0", done_mask(fake_obs()) == 0.0)


def test_buffer():
    print("[5/6] replay buffer §6.3 …")
    buf = ReplayBuffer(100, OBS_DIM, ACT_DIM)
    vec = np.zeros(OBS_DIM, np.float32)
    for i in range(150):  # 超过容量 → 环形覆盖
        buf.push(vec, np.ones(ACT_DIM, np.float32), float(i),
                 vec + 1.0, 1.0)
    check("容量上限 100", len(buf) == 100)
    obs, act, rew, nxt, done = buf.sample(32)
    check("采样形状 (32,68)/(32,2)/(32,)", obs.shape == (32, 68)
          and act.shape == (32, 2) and rew.shape == (32,)
          and nxt.shape == (32, 68) and done.shape == (32,))
    check("环形覆盖生效", buf.reward[0] == 100.0
          and buf.reward[49] == 149.0 and buf.reward[99] == 99.0)


def test_sac_update():
    print("[6/6] SAC 单步更新 §6.3 …")
    device = torch.device("cpu")
    models = MODEL.build_models(device)
    from sac import SACAgent
    agent = SACAgent(device, models)

    buf = ReplayBuffer(4096, OBS_DIM, ACT_DIM)
    rng = np.random.default_rng(0)
    for _ in range(1024):
        o = rng.random(OBS_DIM).astype(np.float32)
        buf.push(o, rng.uniform(-1, 1, ACT_DIM).astype(np.float32), 1.0,
                 o, 0.0)

    target_before = [p.clone() for p in models["critic1_target"].parameters()]
    loss = agent.update(buf, batch_size=64)
    check("loss 有限", all(np.isfinite(v) for v in loss.values()), str(loss))
    check("update_step +1", agent.update_step == 1)
    target_after = list(models["critic1_target"].parameters())
    moved = any(not torch.allclose(a, b, atol=1e-7)
                for a, b in zip(target_before, target_after))
    check("soft update 生效", moved)
    check("α 可训练", np.isfinite(loss["alpha"]))

    # checkpoint 存取（§7.3）
    with tempfile.TemporaryDirectory() as td:
        meta = {"episode": 1, "update_step": 1, "obs_dim": OBS_DIM,
                "act_dim": ACT_DIM, "protocol_version": "1.1"}
        p = MODEL.save_checkpoint(td, "t", models, agent.optimizers_dict(), meta)
        models2 = MODEL.build_models(device)
        m2 = MODEL.load_checkpoint(p, models2)
        check("ckpt meta 回读", m2["episode"] == 1 and "saved_at" in m2)
        for k in models:
            for pa, pb in zip(models[k].parameters(), models2[k].parameters()):
                assert torch.allclose(pa, pb, atol=1e-7)
        check("ckpt 权重一致", True)


def main():
    global MODEL
    print("=" * 60)
    print("003 离线自测（不依赖 server / Webots）")
    print("=" * 60)
    MODEL = get_model_module()
    print(f"模型来源: {MODEL.__file__}")
    test_pack_obs()
    test_scale_action()
    test_reward()
    test_done_mask()
    test_buffer()
    test_sac_update()
    print("=" * 60)
    print("SELFTEST OK — 全部校验通过")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
