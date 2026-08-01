# -*- coding: utf-8 -*-
"""
train_async.py — A2 项目 003 算法侧 异步 RL 训练（SAC，多环境并行采集）

与 train.py 的协议完全一致（api.md 不变），只是进程内部并行化：
- 同时连 N 个 Webots 实例（001 侧启动器为每个实例设不同 ENV_WS_PORT，
  本脚本连 ws://127.0.0.1:8765+i），每个实例一个采集协程独立跑局；
- 所有实例共享同一个 replay buffer，learner 在独立线程里按 1:1 UTD
  （每收集一条过渡做一次梯度更新）从 buffer 抽批更新；
- 每累计 eval_interval 局评估一次（暂停采集，在 worker 0 上确定性评估）；
- 收敛判据与 §6.8 一致：连续 3 次评估 ≥90% → 向所有 server 发 all_finish。

用法（先由 001 侧启动脚本起 N 个 webots 实例）：
    python train_async.py --workers 4 --episodes 2000 \
        --resume checkpoints/run8/ckpt_best.pt \
        --log-dir logs/run9 --save-dir checkpoints/run9

说明：SAC 是 off-policy 算法，多环境异步采集只改变数据到达时序，
不改变目标函数；共享权重的读写用一把线程锁保护（采集时 select_action
与 learner 的 update 互斥，代价可忽略）。
"""

import argparse
import asyncio
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import train as T  # 复用协议/单局/评估/存盘等全部逻辑
from buffer import ReplayBuffer
from config import (ACT_DIM, CONVERGE_CONSECUTIVE, CONVERGE_RATE,
                    EVAL_EPISODES, EVAL_INTERVAL, EVAL_SEED_BASE,
                    MAX_EPISODES, OBS_DIM, PING_INTERVAL, PING_TIMEOUT)
from models import get_model_module
from sac import SACAgent

log = logging.getLogger("train_async")

WARMUP = 5_000            # api.md §6.3：buffer 少于 5000 条不更新
BUFFER_CAPACITY = 200_000
MAX_PENDING = 5_000       # learner 落后上限：超了丢过期更新量，防数据无限过时


# ---------------- 线程安全的 agent 代理 ----------------
class LockedAgent:
    """select_action 与 update 互斥（读写同一套权重），事件循环线程和
    learner 线程共用这一个代理。"""

    def __init__(self, agent: SACAgent):
        self.agent = agent
        self.lock = threading.Lock()

    def select_action(self, vec, deterministic=False):
        with self.lock:
            return self.agent.select_action(vec, deterministic=deterministic)

    def update(self, buffer):
        with self.lock:
            return self.agent.update(buffer)

    def optimizers_dict(self):
        return self.agent.optimizers_dict()

    @property
    def update_step(self):
        return self.agent.update_step


# ---------------- 跨协程共享计数器 ----------------
class Counters:
    def __init__(self, start_episode: int):
        self.episodes = start_episode   # 已完成的训练局（聚合，跨 worker）
        self.active = 0                 # 正在跑的局数（eval 暂停用）
        self.steps_collected = 0        # 已收集过渡数（learner 用它算 pending）
        self.updates_done = 0           # 已做梯度更新数
        self.total_resets = start_episode  # 发给 server 的 reset 总数（含评估局）


# ---------------- learner：独立线程按 1:1 UTD 更新 ----------------
async def learner_task(agent: LockedAgent, buffer, counters: Counters,
                       stop_event: asyncio.Event):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        while not stop_event.is_set():
            pending = counters.steps_collected - counters.updates_done
            if pending > MAX_PENDING:  # 落后太多：丢过期更新，保持数据新鲜度
                counters.updates_done = counters.steps_collected - MAX_PENDING
                pending = MAX_PENDING
            if len(buffer) >= WARMUP and pending > 0:
                await loop.run_in_executor(ex, agent.update, buffer)
                counters.updates_done += 1
            else:
                await asyncio.sleep(0.005)


# ---------------- 单个 worker：一条 websocket 连一个实例，循环跑局 ----------------
async def worker(idx: int, uri: str, args, agent: LockedAgent, buffer,
                 counters: Counters, gate: asyncio.Event,
                 stop_event: asyncio.Event, holder: list,
                 elog: T.EpisodeLogger):
    try:
        ws = await T.connect(uri, ping_interval=PING_INTERVAL,
                             ping_timeout=PING_TIMEOUT)
    except OSError as e:
        log.error("worker%d 无法连接 %s: %s", idx, uri, e)
        stop_event.set()
        return
    async with ws:
        hello = await T.recv_msg(ws)
        cfg = T.validate_hello(hello)
        holder[idx] = {"ws": ws, "cfg": cfg,
                       "protocol_version": hello.get("protocol_version")}
        log.info("worker%d hello OK: %s v%s (obs=%d act=%d)",
                 idx, hello.get("env_name"), hello.get("protocol_version"),
                 cfg["obs_dim"], cfg["act_dim"])
        while not stop_event.is_set():
            await gate.wait()   # 评估期间暂停采集
            episode = counters.episodes
            if episode >= args.episodes:
                break
            counters.episodes += 1
            counters.active += 1
            try:
                r = await T.run_episode(
                    ws, cfg, agent, buffer, args, seed=-1,
                    override=T.curriculum_override(episode, args),
                    deterministic=False, train=True)
            finally:
                counters.active -= 1
            counters.steps_collected += r["steps"]
            counters.total_resets += 1
            elog.log_episode(episode, r["steps"], r["return_"],
                             r["outcome"], r["success"])
            if episode % 10 == 0:
                log.info("w%d ep %4d | %-12s | steps=%3d | return=%8.2f | "
                         "buffer=%d (pending=%d)",
                         idx, episode, r["outcome"], r["steps"], r["return_"],
                         len(buffer),
                         counters.steps_collected - counters.updates_done)


# ---------------- 评估调度：每 eval_interval 局评估一次 ----------------
async def eval_scheduler(args, agent: LockedAgent, models, model_mod,
                         counters: Counters, gate: asyncio.Event,
                         stop_event: asyncio.Event, holder: list,
                         evlog: T.EvalLogger) -> bool:
    best_rate = -1.0
    converge_streak = 0
    start = args.resume_episode + 1
    target = ((start + args.eval_interval - 1) // args.eval_interval) \
        * args.eval_interval
    while target <= args.episodes and not stop_event.is_set():
        while counters.episodes < target and not stop_event.is_set():
            await asyncio.sleep(1.0)
        if stop_event.is_set():
            break
        # 暂停采集，等所有在跑局收尾
        gate.clear()
        while counters.active > 0:
            await asyncio.sleep(0.1)
        while holder[0] is None and not stop_event.is_set():
            await asyncio.sleep(0.1)   # 等 worker0 就绪；worker0 挂了则退出
        if holder[0] is None:
            gate.set()
            return False
        ws0, cfg0 = holder[0]["ws"], holder[0]["cfg"]
        rate, succ = await T.evaluate(ws0, cfg0, agent, args)
        counters.total_resets += args.eval_episodes
        evlog.log_eval(target - 1, rate, succ, args.eval_episodes)
        log.info("== 评估 @ ep %d：成功率 %.2f%% (%d/%d) ==",
                 target - 1, 100 * rate, succ, args.eval_episodes)

        await T.save_ckpt(model_mod, args.save_dir, f"ep_{target}", models,
                          agent, target - 1,
                          holder[0]["protocol_version"])
        if rate > best_rate + 1e-9:
            best_rate = rate
            await T.save_ckpt(model_mod, args.save_dir, "best", models,
                              agent, target - 1,
                              holder[0]["protocol_version"])
            log.info("== 新最佳成功率 %.2f%% ==", 100 * rate)
        converge_streak = ((converge_streak + 1)
                           if rate >= args.converge_rate else 0)

        gate.set()
        if converge_streak >= args.converge_consecutive:
            log.info("收敛：连续 %d 次评估 ≥%.0f%%（@ep%d）",
                     args.converge_consecutive, 100 * args.converge_rate,
                     target - 1)
            stop_event.set()
            return True
        target += args.eval_interval
    gate.set()
    return False


# ---------------- 主流程 ----------------
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "train_async.log",
                                      encoding="utf-8")])


async def amain(args) -> int:
    setup_logging(Path(args.log_dir))
    device = args.device or ("cuda" if T._cuda_available() else "cpu")
    import torch
    torch.set_num_threads(args.torch_threads)  # 留核给 4 个 webots 实例
    log.info("device=%s workers=%d torch_threads=%d", device, args.workers,
             args.torch_threads)

    model_mod = get_model_module()
    models = model_mod.build_models(device)
    agent = LockedAgent(SACAgent(device, models))
    buffer = ReplayBuffer(BUFFER_CAPACITY, OBS_DIM, ACT_DIM)

    if args.preload:
        data = np.load(args.preload)
        obs_npz, act_npz = data["obs"], data["act"]
        rew_npz, nobs_npz, done_npz = data["rew"], data["next_obs"], data["done"]
        n = len(rew_npz)
        assert obs_npz.shape == (n, OBS_DIM), \
            f"preload obs 形状 {obs_npz.shape} 与 OBS_DIM={OBS_DIM} 不符"
        for i in range(n):
            buffer.push(obs_npz[i], act_npz[i], float(rew_npz[i]),
                        nobs_npz[i], float(done_npz[i]))
        log.info("演示数据预填: %s → %d 条", args.preload, n)

    resume_episode = -1
    if args.resume:
        meta = model_mod.load_checkpoint(args.resume, models,
                                         agent.optimizers_dict(),
                                         map_location=device)
        resume_episode = int(meta.get("episode", -1))
        log.info("从 %s 恢复：episode=%d update_step=%d", args.resume,
                 resume_episode, meta.get("update_step"))
    args.resume_episode = resume_episode

    counters = Counters(resume_episode + 1)
    gate = asyncio.Event()
    gate.set()
    stop_event = asyncio.Event()
    holder: list = [None] * args.workers

    elog = T.EpisodeLogger(Path(args.log_dir))
    evlog = T.EvalLogger(Path(args.log_dir))
    converged = False

    uris = [f"ws://{args.host}:{args.base_port + i}" for i in range(args.workers)]
    workers = [asyncio.create_task(worker(i, uris[i], args, agent, buffer,
                                          counters, gate, stop_event, holder,
                                          elog))
               for i in range(args.workers)]
    learner = asyncio.create_task(learner_task(agent, buffer, counters,
                                               stop_event))
    evaler = asyncio.create_task(eval_scheduler(args, agent, models, model_mod,
                                                counters, gate, stop_event,
                                                holder, evlog))

    try:
        await asyncio.gather(*workers)
        # workers 全退（局数跑满或 stop_event），等 learner 收尾（避免更新中断）
        stop_event.set()
        await learner
        converged = await evaler
    except asyncio.CancelledError:
        stop_event.set()
        await learner
        raise
    finally:
        # 向所有 server 礼貌道别（best effort）
        for h in holder:
            if h is not None:
                await T.send_all_finish(h["ws"], "converged" if converged
                                        else "interrupted",
                                        counters.total_resets)
    log.info("训练结束：episodes=%d updates=%d 收敛=%s",
             counters.episodes, counters.updates_done, converged)
    return 0 if converged else 1


def main():
    ap = argparse.ArgumentParser(description="异步 RL（SAC 多环境并行）")
    ap.add_argument("--workers", type=int, default=4, help="并行 webots 实例数")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8765)
    ap.add_argument("--episodes", type=int, default=MAX_EPISODES)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--preload", default=None)
    ap.add_argument("--log-dir", default="logs/runX")
    ap.add_argument("--save-dir", default="checkpoints/runX")
    ap.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--eval-seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--converge-rate", type=float, default=CONVERGE_RATE)
    ap.add_argument("--converge-consecutive", type=int,
                    default=CONVERGE_CONSECUTIVE)
    ap.add_argument("--no-curriculum", action="store_true",
                    help="关闭课程学习（前 cur_short_episodes 局压短时限）")
    ap.add_argument("--cur-short-episodes", type=int,
                    default=T.CUR_SHORT_EPISODES)
    ap.add_argument("--cur-short-time", type=float, default=T.CUR_SHORT_TIME)
    ap.add_argument("--device", default=None)
    ap.add_argument("--torch-threads", type=int, default=8,
                    help="torch 并行线程数（留核给 webots 实例）")
    args = ap.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
