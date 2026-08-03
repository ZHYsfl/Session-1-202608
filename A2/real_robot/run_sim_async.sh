#!/bin/bash
# 一键：全 Python 异步 RL（N 个 pysim 采集实例 + 1 个 pysim 专职评估实例）
# 2026-08-02 晚重写：弃用多 Webots 实例（卡死），场景本身是理想运动学+
# 解析雷达，pysim_server.py 与 env_server 协议/物理对齐，吞吐高一个量级。
#
# 用法：
#   bash run_sim_async.sh [WORKERS] [EPISODES] [RESUME]
#     WORKERS   采集实例数（默认 12）
#     EPISODES  训练局数上限（默认 2000）
#     RESUME    非空则从该 checkpoint 恢复（不预填专家数据）
set -u
WORKERS="${1:-12}"
EPISODES="${2:-2000}"
RESUME="${3:-}"
BASE=8765
EVAL_PORT=8873
DIR=/home/zane/session_1/A2/003

cleanup() {
    pkill -f "tools/pysim_server.py" 2>/dev/null
    sleep 1
}

cleanup

echo "[async] starting $WORKERS pysim collectors (ports $BASE..$((BASE+WORKERS-1))) + eval :$EVAL_PORT"
cd "$DIR"
for i in $(seq 0 $((WORKERS-1))); do
    (uv run python tools/pysim_server.py --port $((BASE+i)) > "/tmp/pysim_w${i}.log" 2>&1) &
done
(uv run python tools/pysim_server.py --port "$EVAL_PORT" > "/tmp/pysim_eval.log" 2>&1) &
sleep 3

COMMON="--workers $WORKERS --base-port $BASE --eval-uri ws://127.0.0.1:$EVAL_PORT \
    --episodes $EPISODES --no-curriculum \
    --log-dir logs/run_pysim_v6 --save-dir checkpoints/run_pysim_v6"
if [ -n "$RESUME" ]; then
    echo "[async] resume from $RESUME"
    # shellcheck disable=SC2086
    uv run python train_async.py $COMMON --resume "$RESUME"
else
    echo "[async] from scratch + preload data/expert_v4_68.npz"
    # shellcheck disable=SC2086
    uv run python train_async.py $COMMON --preload data/expert_v4_68.npz
fi
RC=$?

echo "[async] done (rc=$RC), closing pysim instances..."
cleanup
exit $RC
