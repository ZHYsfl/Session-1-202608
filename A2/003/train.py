# -*- coding: utf-8 -*-
"""
train.py — A2 项目 003 算法侧 训练/推理主循环（WebSocket client）

按 api.md §3.3 伪代码实现；与 001（Webots env server，ws://127.0.0.1:8765）
通信，进程内 import 009 的 model.py（models.py 负责加载，缺失时用开发桩）。

用法：
    python train.py --episodes 2000                          # 默认训练
    python train.py --resume checkpoints/ckpt_best.pt        # 从 checkpoint 恢复
    python train.py --uri ws://127.0.0.1:8766 --episodes 6   # 对接开发桩（离线）
    python train.py --no-curriculum                          # 关闭课程学习

信号处理（api.md §2.5）：Ctrl+C 时先在连接内发 all_finish(reason=interrupted)
再退出——server 是单客户端设计，直接断线会导致它立刻关仿真（兜底路径）；
训练异常则发 all_finish(reason=error) 后退出。
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:  # 内嵌键盘遥控（真机 drive_to_start 阶段用）；Windows 无 termios 则回退普通输入
    import termios
    import tty

    _TELEOP_AVAILABLE = sys.stdin.isatty()
except ImportError:  # pragma: no cover — Windows
    termios = tty = None
    _TELEOP_AVAILABLE = False

try:  # websockets >= 14
    from websockets.asyncio.client import connect
except ImportError:  # websockets < 14
    from websockets import connect

from buffer import ReplayBuffer
from config import (ACT_DIM, CONVERGE_CONSECUTIVE, CONVERGE_RATE,
                    CUR_SHORT_EPISODES, CUR_SHORT_TIME, EVAL_EPISODES,
                    EVAL_INTERVAL, EVAL_SEED_BASE, MAX_EPISODES, OBS_DIM,
                    PING_INTERVAL, PING_TIMEOUT, SAVE_INTERVAL, WS_URI)
from models import get_model_module, model_source
from obs_pack import ObsPacker, scale_action
from reward import compute_reward, done_mask, outcome_of
from sac import SACAgent

log = logging.getLogger("train")

BUFFER_CAPACITY = 500_000     # api.md §6.3：replay buffer 容量（08-02 起 50 万，
                              # 与 train_async.py 一致；容纳 14.3 万演示 + 在线数据）
WARMUP = 5_000                # api.md §6.3：buffer 少于 5000 条不更新
BYE_TIMEOUT_S = 5.0           # api.md §2.5：等 bye 超时（秒）

# 内嵌遥控（真机 drive_to_start）：v/w 上限取 server v_max/w_max 内保守值
TELEOP_LINEAR = 0.2           # m/s
TELEOP_ANGULAR = 0.5          # rad/s
TELEOP_HZ = 20.0              # 遥控指令发送频率


class ProtocolError(RuntimeError):
    """server 回 error 消息或消息不合预期时抛出。"""


# ================= 协议消息 =================

async def recv_msg(ws) -> dict:
    """收一条消息；若 server 回 error 则抛 ProtocolError。"""
    raw = await ws.recv()
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"BAD_JSON: {raw!r}") from e
    if msg.get("type") == "error":
        raise ProtocolError(f"server error: code={msg.get('code')} "
                            f"detail={msg.get('detail')}")
    return msg


def human_confirm_payload(action: str) -> dict:
    """§2.9：对 human 消息的确认。"""
    return {"type": "human_confirm", "action": action}


def read_key() -> str:
    """读取单个按键（不回车）；仅类 Unix 终端可用。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def teleop_drive_to_start(ws, action: str, detail: str) -> None:
    """
    真机 drive_to_start 阶段：在训练终端直接键盘遥控（W/S/A/D/空格/Q），
    通过 server 的 teleop 消息驱动 /cmd_vel，免去第二个终端。
    遥控结束（Q）后发送 human_confirm。
    """
    print(f"\n>>> [需要人工操作] {detail}", flush=True)
    print("内嵌遥控：按住 W/S 前进后退，A/D 左右转，空格停止，Q 结束遥控")
    print("命令会以 20Hz 持续发给 server，server 再转发给 /cmd_vel\n", flush=True)

    teleop_state = {"v": 0.0, "w": 0.0, "running": True}
    last_key = ""

    async def sender_loop():
        while teleop_state["running"]:
            await ws.send(json.dumps({
                "type": "teleop",
                "v": teleop_state["v"],
                "w": teleop_state["w"],
            }))
            print(f"\r  teleop: v={teleop_state['v']:+.2f}  w={teleop_state['w']:+.2f}"
                  f"  |  last_key={last_key!r}  |  W/S/A/D/space/Q",
                  end="", flush=True)
            await asyncio.sleep(1.0 / TELEOP_HZ)
        await ws.send(json.dumps({"type": "teleop", "v": 0.0, "w": 0.0}))

    def key_reader():
        nonlocal last_key
        while teleop_state["running"]:
            key = read_key()
            last_key = key
            if key == "w" or key == "W" or key == "\x1b[A":      # 前进
                teleop_state["v"] = TELEOP_LINEAR
                teleop_state["w"] = 0.0
            elif key == "s" or key == "S" or key == "\x1b[B":    # 后退
                teleop_state["v"] = -TELEOP_LINEAR
                teleop_state["w"] = 0.0
            elif key == "a" or key == "A" or key == "\x1b[D":    # 左转
                teleop_state["v"] = 0.0
                teleop_state["w"] = TELEOP_ANGULAR
            elif key == "d" or key == "D" or key == "\x1b[C":    # 右转
                teleop_state["v"] = 0.0
                teleop_state["w"] = -TELEOP_ANGULAR
            elif key == " ":
                teleop_state["v"] = 0.0
                teleop_state["w"] = 0.0
            elif key == "q" or key == "Q":
                teleop_state["v"] = 0.0
                teleop_state["w"] = 0.0
                teleop_state["running"] = False
                return

    sender = asyncio.create_task(sender_loop())
    reader = asyncio.get_event_loop().run_in_executor(None, key_reader)
    await asyncio.gather(sender, reader)

    print("\n\n遥控结束，发送 human_confirm(drive_to_start)...")
    await ws.send(json.dumps(human_confirm_payload(action)))


async def recv_msg_handle_human(ws, auto_confirm: bool = False) -> dict:
    """
    收一条消息；若是真机 server 的 human 消息，则提示线下操作并回复确认，
    然后继续收下一条，直到拿到非 human 消息为止。
    """
    while True:
        msg = await recv_msg(ws)
        if msg.get("type") != "human":
            return msg
        action = msg.get("action", "unknown")
        detail = msg.get("detail", "")
        log.warning("[HUMAN ACTION REQUIRED] %s: %s", action, detail)
        if auto_confirm:
            log.warning("自动发送 human_confirm(%s)", action)
        elif action == "drive_to_start" and _TELEOP_AVAILABLE:
            # 真机：在训练终端直接键盘遥控，免开第二个终端
            await teleop_drive_to_start(ws, action, detail)
            continue
        else:
            # 交互式提示：等线下人员按回车；用 executor 避免阻塞 asyncio 事件循环
            try:
                prompt = f"\n>>> [需要人工操作] {detail}\n完成后按回车继续..."
                print(prompt, flush=True)
                await asyncio.get_event_loop().run_in_executor(None, input, "")
            except EOFError:
                log.warning("非交互终端，自动发送 human_confirm(%s)", action)
        await ws.send(json.dumps(human_confirm_payload(action)))


def validate_hello(hello: dict) -> dict:
    """§6.1 连接校验：协议主版本一致、obs_dim 与 009 常量一致，否则报错退出。"""
    if hello.get("type") != "hello":
        raise ProtocolError(f"首条消息不是 hello: {hello}")
    if hello.get("protocol_version", "").split(".")[0] != "1":
        raise ProtocolError(
            f"协议主版本不兼容: server={hello.get('protocol_version')}, "
            f"要求主版本 1")
    cfg = hello.get("config", {})
    if cfg.get("obs_dim") != OBS_DIM:
        raise ProtocolError(
            f"hello obs_dim={cfg.get('obs_dim')} 与模型 OBS_DIM={OBS_DIM} 不一致")
    for key in ("lidar_count", "lidar_max_range", "v_max", "w_max", "act_dim"):
        if key not in cfg:
            raise ProtocolError(f"hello.config 缺字段: {key}")
    return cfg


def reset_payload(seed: int, override: dict | None) -> dict:
    """§2.2 reset 消息；override 为 config_override（null 或 {"max_episode_time": T}）。"""
    return {"type": "reset", "seed": seed, "config_override": override}


def curriculum_override(episode: int, args) -> dict | None:
    """课程学习（api.md §2.2）：前 cur_short_episodes 局压短时限，之后放回默认。"""
    if not args.no_curriculum and episode < args.cur_short_episodes:
        return {"max_episode_time": args.cur_short_time}
    return None


def action_payload(obs: dict, v: float, w: float) -> dict:
    """§2.4：action 的 episode_id/step_id 必须等于所回应 obs 的。"""
    return {"type": "action", "episode_id": obs["episode_id"],
            "step_id": obs["step_id"], "v": v, "w": w}


def all_finish_payload(reason: str, total_episodes: int) -> dict:
    return {"type": "all_finish", "reason": reason,
            "total_episodes": total_episodes}


# ================= 日志（§6.6 / §6.7） =================

class EpisodeLogger:
    """§6.6：每 episode 一行 CSV（episode_id, steps, return, outcome, success）。"""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(log_dir / "episodes.csv", "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["episode_id", "steps", "return", "outcome", "success"])

    def log_episode(self, episode_id: int, steps: int, ret: float,
                    outcome: str, success: int):
        self.w.writerow([episode_id, steps, round(ret, 3), outcome, success])
        self.f.flush()

    def close(self):
        self.f.close()


class EvalLogger:
    """§6.7 评估结果 CSV（训练 ep, 成功率, 成功局数）。"""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(log_dir / "eval.csv", "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["train_episode", "success_rate", "successes",
                         "eval_episodes"])

    def log_eval(self, train_episode: int, rate: float, successes: int, n: int):
        self.w.writerow([train_episode, round(rate, 3), successes, n])
        self.f.flush()

    def close(self):
        self.f.close()


# ================= 单局执行 =================

async def run_episode(ws, cfg, agent, buffer, args, *, seed: int,
                      override: dict | None, deterministic: bool,
                      train: bool) -> dict:
    """跑一局，返回 {"steps", "return_", "outcome", "success"}。
    - train=True：采样动作、存 buffer（训练局）
    - train=False：确定性策略（取 mean）、不存 buffer（评估局，§6.7）
    """
    await ws.send(json.dumps(reset_payload(seed, override)))
    obs = await recv_msg_handle_human(ws, auto_confirm=args.auto_human)
    if obs.get("type") != "obs":
        raise ProtocolError(f"reset 后未收到 obs: {obs}")

    packer = ObsPacker(cfg)   # 帧堆叠打包器（每局新建，首帧历史=当前帧）
    steps = 0
    ret = 0.0
    prev_a01 = None
    while not obs["done"]:
        vec = packer.pack(obs)
        a01 = agent.select_action(vec, deterministic=deterministic)
        if prev_a01 is None:
            prev_a01 = a01.copy()  # 第一步无历史动作 → 平滑惩罚为 0（§6.4）
        v, w = scale_action(a01, cfg)
        await ws.send(json.dumps(action_payload(obs, v, w)))

        obs_next = await recv_msg(ws)
        if obs_next.get("type") != "obs":
            raise ProtocolError(f"action 后未收到 obs: {obs_next}")

        r = compute_reward(obs, a01, prev_a01, obs_next)
        if train:
            buffer.push(vec, a01, r, packer.pack(obs_next),
                        done_mask(obs_next))
        ret += r
        steps += 1
        prev_a01 = a01
        obs = obs_next

    return {"steps": steps, "return_": ret,
            "outcome": outcome_of(obs), "success": int(obs["flags"]["goal_reached"])}


async def evaluate(ws, cfg, agent, args) -> tuple:
    """§6.7 评估：确定性策略（直接取 mean，不采样），固定种子 EVAL_SEED_BASE+i 跑 10 局。"""
    succ = 0
    for i in range(args.eval_episodes):
        r = await run_episode(ws, cfg, agent, None, args,
                              seed=args.eval_seed_base + i,
                              override=None,
                              deterministic=True, train=False)
        succ += r["success"]
    return succ / args.eval_episodes, succ


async def save_ckpt(model_mod, save_dir: str, tag: str, models, agent,
                    episode: int, protocol_version: str) -> str:
    """§7.3 checkpoint：meta 含 episode / update_step / obs_dim / act_dim / protocol_version。"""
    meta = {"episode": episode, "update_step": agent.update_step,
            "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "protocol_version": protocol_version}
    path = model_mod.save_checkpoint(save_dir, tag, models,
                                     agent.optimizers_dict(), meta)
    log.info("checkpoint 已保存: %s", path)
    return path


async def send_all_finish(ws, reason: str, total_episodes: int) -> None:
    """§2.5 礼貌收工（best effort，异常不抛出）。"""
    try:
        await asyncio.wait_for(
            ws.send(json.dumps(all_finish_payload(reason, total_episodes))),
            timeout=2.0)
    except Exception as e:  # noqa: BLE001 — 尽力而为，失败交给 server 断线兜底
        log.warning("发送 all_finish(%s) 失败: %r", reason, e)


# ================= 主循环 =================

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "train.log", encoding="utf-8")])


async def amain(args) -> int:
    setup_logging(Path(args.log_dir))
    device = args.device or ("cuda" if _cuda_available() else "cpu")
    log.info("device=%s", device)

    model_mod = get_model_module()
    log.info("009 模型来源: %s", model_source())
    models = model_mod.build_models(device)
    agent = SACAgent(device, models)
    buffer = ReplayBuffer(BUFFER_CAPACITY, OBS_DIM, ACT_DIM)

    if args.preload:
        # SACfD 冷启动：用 scripted_expert --dump 采集的演示轨迹预填 buffer。
        # 演示数据不覆盖 warmup 计数逻辑——buffer 里有数据即开始更新。
        data = np.load(args.preload)
        # 教训：NpzFile 的 data["obs"] 每次访问都重新解压整包，直接写进循环是
        # O(n²)（3.2 万条卡 82s+）。先一次性取出数组再循环。
        obs_npz, act_npz = data["obs"], data["act"]
        rew_npz, nobs_npz, done_npz = data["rew"], data["next_obs"], data["done"]
        n = len(rew_npz)
        assert obs_npz.shape == (n, OBS_DIM), \
            f"preload obs 形状 {obs_npz.shape} 与 OBS_DIM={OBS_DIM} 不符"
        assert act_npz.shape == (n, ACT_DIM), \
            f"preload act 形状 {act_npz.shape} 与 ACT_DIM={ACT_DIM} 不符"
        for i in range(n):
            buffer.push(obs_npz[i], act_npz[i], float(rew_npz[i]),
                        nobs_npz[i], float(done_npz[i]))
        log.info("演示数据预填: %s → %d 条（回报均值 %.2f）", args.preload, n,
                 float(rew_npz.mean()))

    episode_start = 0
    if args.resume:
        meta = model_mod.load_checkpoint(args.resume, models,
                                         agent.optimizers_dict(),
                                         map_location=device)
        episode_start = int(meta.get("episode", 0)) + 1
        log.info("从 %s 恢复：episode=%d update_step=%d", args.resume,
                 meta.get("episode"), meta.get("update_step"))

    elog = EpisodeLogger(Path(args.log_dir))
    evlog = EvalLogger(Path(args.log_dir))
    best_rate = -1.0
    converge_streak = 0
    total_resets = 0        # 发给 server 的 reset 总数（含评估局，写进 all_finish）

    episode = episode_start - 1  # 供"达到上限"分支兜底引用（resume 起点 ≥ 上限时）
    try:
        log.info("连接 %s …", args.uri)
        try:
            ws_conn = connect(args.uri, ping_interval=PING_INTERVAL,
                              ping_timeout=PING_TIMEOUT)
            ws = await ws_conn
        except OSError as e:
            log.error("无法连接 %s：%s。请确认 001 Webots server 已启动，"
                      "或改用 --uri 指向开发桩（如 ws://127.0.0.1:8766）",
                      args.uri, e)
            return 2
        async with ws:
            hello = await recv_msg(ws)
            cfg = validate_hello(hello)
            log.info("hello OK: %s v%s | obs_dim=%d act_dim=%d | "
                     "v_max=%.2f w_max=%.2f dt=%.2f",
                     hello.get("env_name"), hello.get("protocol_version"),
                     cfg["obs_dim"], cfg["act_dim"],
                     cfg["v_max"], cfg["w_max"], cfg["control_dt"])

            try:
                for episode in range(episode_start, args.episodes):
                    # ---- 一局训练（随机场景 seed=-1，课程学习压短时限）----
                    r = await run_episode(
                        ws, cfg, agent, buffer, args, seed=-1,
                        override=curriculum_override(episode, args),
                        deterministic=False, train=True)
                    total_resets += 1
                    ret, steps, outcome, success = (r["return_"], r["steps"],
                                                    r["outcome"], r["success"])

                    # ---- 局后更新：K = 本 episode 步数（§3.3，约 1:1 更新比）----
                    if len(buffer) >= args.warmup:
                        loss_acc = {}
                        for _ in range(steps):
                            loss = agent.update(buffer)
                            for k, v in loss.items():
                                loss_acc[k] = loss_acc.get(k, 0.0) + v
                        n = max(steps, 1)
                        if episode % 10 == 0:
                            log.info(
                                "ep %4d | %-12s | steps=%3d | return=%8.2f | "
                                "α=%.3f | loss_actor=%8.1f loss_critic=%8.1f | "
                                "buffer=%d",
                                episode, outcome, steps, ret,
                                loss_acc["alpha"] / n,
                                loss_acc["loss_actor"] / n,
                                (loss_acc["loss_critic1"]
                                 + loss_acc["loss_critic2"]) / n / 2,
                                len(buffer))
                    else:
                        log.info("ep %4d | %-12s | steps=%3d | return=%8.2f | "
                                 "buffer=%d（warmup 中，需 ≥%d）",
                                 episode, outcome, steps, ret, len(buffer),
                                 args.warmup)

                    elog.log_episode(episode, steps, ret, outcome, success)

                    # ---- 定期存盘：每 save_interval 局存一次 ckpt_ep_N.pt ----
                    # （新名字、不覆盖；与评估解耦，Ctrl+C 最多丢 save_interval 局）
                    if (episode + 1) % args.save_interval == 0:
                        await save_ckpt(model_mod, args.save_dir,
                                        f"ep_{episode + 1}", models, agent,
                                        episode, hello["protocol_version"])

                    # ---- 周期性评估 + 存盘（§6.7）----
                    if (episode + 1) % args.eval_interval == 0:
                        rate, succ = await evaluate(ws, cfg, agent, args)
                        total_resets += args.eval_episodes
                        evlog.log_eval(episode, rate, succ, args.eval_episodes)
                        log.info("== 评估 @ ep %d：成功率 %.2f%% (%d/%d) ==",
                                 episode, 100 * rate, succ, args.eval_episodes)

                        if (episode + 1) % args.save_interval != 0:
                            await save_ckpt(model_mod, args.save_dir,
                                            f"ep_{episode + 1}", models, agent,
                                            episode, hello["protocol_version"])
                        if rate > best_rate + 1e-9:
                            best_rate = rate
                            await save_ckpt(model_mod, args.save_dir, "best",
                                            models, agent, episode,
                                            hello["protocol_version"])
                            log.info("== 新最佳成功率 %.2f%% ==", 100 * rate)
                        converge_streak = ((converge_streak + 1)
                                           if rate >= args.converge_rate else 0)

                        # ---- 收敛（§6.8）：连续 3 次评估成功率 ≥90% ----
                        # 注意：课程学习阶段（服务端 reset 计数）完成前不收敛，
                        # 否则会过早停止（只学了简单场景），--converge-min-episode 设门槛
                        if (converge_streak >= args.converge_consecutive
                                and episode >= args.converge_min_episode):
                            log.info("收敛：连续 %d 次评估 ≥%.0f%%",
                                     args.converge_consecutive,
                                     100 * args.converge_rate)
                            await send_all_finish(ws, "converged",
                                                  total_resets)
                            await _wait_bye(ws)
                            await save_ckpt(model_mod, args.save_dir,
                                            "final", models, agent, episode,
                                            hello["protocol_version"])
                            elog.close()
                            evlog.close()
                            return 0

                # ---- 达到上限未收敛：正常收工（reason=converged）----
                log.info("达到 %d 局上限，终止训练", args.episodes)
                await send_all_finish(ws, "converged", total_resets)
                await _wait_bye(ws)
                await save_ckpt(model_mod, args.save_dir, "final", models,
                                agent, episode, hello["protocol_version"])
                elog.close()
                evlog.close()
                return 0

            except asyncio.CancelledError:
                # Ctrl+C：先礼貌收工（reason=interrupted）再退出（§2.5）
                log.warning("收到中断信号，发送 all_finish(interrupted) …")
                await send_all_finish(ws, "interrupted", total_resets)
                raise
            except Exception as e:  # noqa: BLE001 — 兜底：礼貌关闭并记录日志
                log.exception("训练异常（发送 all_finish(error) 后退出）: %s", e)
                await send_all_finish(ws, "error", total_resets)
                raise
    finally:
        elog.close()
        evlog.close()


async def _wait_bye(ws):
    """§2.5：发完 all_finish 后阻塞等 bye，5 秒超时直接保存退出。"""
    try:
        async with asyncio.timeout(BYE_TIMEOUT_S):
            bye = await recv_msg(ws)
            if bye.get("type") != "bye":
                log.warning("期望 bye，收到: %s", bye)
            else:
                log.info("bye 收到，正常收工")
    except (asyncio.TimeoutError, TimeoutError):
        log.warning("%d 秒内未收到 bye，直接保存退出", BYE_TIMEOUT_S)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="003 SAC 训练主循环（api.md §3.3/§6）")
    ap.add_argument("--uri", default=WS_URI)
    ap.add_argument("--episodes", type=int, default=MAX_EPISODES)
    ap.add_argument("--device", default=None,
                    help="torch 设备（默认自动：cuda 可用则 cuda）")
    ap.add_argument("--resume", default=None,
                    help="从 checkpoint 恢复（ckpt_*.pt 路径）")
    ap.add_argument("--preload", default=None,
                    help="演示数据 npz（scripted_expert --dump 产物），开训前预填 buffer")
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--log-dir", default="logs")
    # 课程学习（api.md §2.2）
    ap.add_argument("--no-curriculum", action="store_true",
                    help="关闭课程学习（全程默认时限）")
    ap.add_argument("--cur-short-episodes", type=int, default=CUR_SHORT_EPISODES)
    ap.add_argument("--cur-short-time", type=float, default=CUR_SHORT_TIME)
    # 评估 / 收敛（§6.7 / §6.8）
    ap.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--eval-seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                    help="每隔多少局存一次 ckpt_ep_N.pt（每次新名字、不覆盖；"
                         "真机微调建议 10）")
    ap.add_argument("--converge-consecutive", type=int,
                    default=CONVERGE_CONSECUTIVE)
    ap.add_argument("--converge-rate", type=float, default=CONVERGE_RATE)
    ap.add_argument("--converge-min-episode", type=int, default=0,
                    help="课程学习完成前禁止收敛（仿真课程 3 阶段约 900 局，"
                         "传 900 保证学完全部难度再判定收敛）")
    # 真机模式（api.md §2.8 / §2.9）：自动回复 human_confirm，不暂停等人工
    ap.add_argument("--auto-human", action="store_true",
                    help="真机模式下自动发送 human_confirm，不暂停等线下操作（调试用）")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="buffer 少于该条数不做梯度更新（真机微调建议调小，如 256）")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        # asyncio.run 在取消主任务后会重新抛 KeyboardInterrupt
        sys.exit(130)


if __name__ == "__main__":
    main()
