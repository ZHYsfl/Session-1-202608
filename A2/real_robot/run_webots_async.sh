#!/bin/bash
# Webots 异步 RL：N 个 fast 采集实例 + 1 个 fast 专职评估实例
# 2026-08-02 晚：pysim 方案暴露 sim-sim 差距（pysim 100% / Webots 60%），
# 回到 Webots 训练——learner 才是瓶颈（~45 更新/s），4 个 Webots 实例
# 足以喂饱（run9 实测 4 实例 ~55 局/分），且训练分布 = 验收分布。
# 实例数不要超 5：8 实例实测全部卡死，4 实例是 run9-12 验证过的稳定配置。
#
# 用法：
#   bash run_webots_async.sh [WORKERS] [EPISODES] [RESUME]
#     WORKERS   采集实例数（默认 4，+1 评估实例 = 5 个 webots）
#     EPISODES  训练局数上限（默认 1500）
#     RESUME    非空则从该 checkpoint 恢复（不预填专家数据）
set -u
WORKERS="${1:-4}"
EPISODES="${2:-1500}"
RESUME="${3:-}"
TAG="${4:-run_webots_async}"
EXTRA="${5:-}"   # 额外透传给 train_async.py 的参数（如 "--converge-consecutive 999"）
WEBOTS="/mnt/d/Program Files/Webots/msys64/mingw64/bin/webots.exe"
WORLD='D:\webots_projects\rl_chassis\worlds\rl_arena.wbt'
BASE=8765
EVAL_PORT=8873
DIR=/home/zane/session_1/A2/003

cleanup() {
    powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force; \
        \$c = netstat -ano | Select-String ':87[0-9][0-9] ', ':8873 '; if (\$c) { \$c | ForEach-Object { \
        \$p = (\$_ -split '\s+')[-1]; Stop-Process -Id \$p -Force -EA SilentlyContinue } }" 2>/dev/null
    sleep 2
}

cleanup

echo "[wasync] starting $WORKERS collectors (ports $BASE..$((BASE+WORKERS-1))) + eval :$EVAL_PORT"
for i in $(seq 0 $((WORKERS-1))); do
    p=$((BASE + i))
    (WSLENV=ENV_WS_PORT:ENV_WS_HOST ENV_WS_PORT="$p" ENV_WS_HOST=0.0.0.0 \
        "$WEBOTS" --batch --mode=fast --stdout --stderr "$WORLD" > "/tmp/webots_w${i}.log" 2>&1) &
done
(WSLENV=ENV_WS_PORT:ENV_WS_HOST ENV_WS_PORT="$EVAL_PORT" ENV_WS_HOST=0.0.0.0 \
    "$WEBOTS" --batch --mode=fast --stdout --stderr "$WORLD" > "/tmp/webots_eval.log" 2>&1) &

echo "[wasync] waiting for instances to boot + lidar calib (up to 90s)..."
for i in $(seq 0 $((WORKERS-1))); do
    p=$((BASE + i))
    for t in $(seq 1 30); do
        sleep 3
        if powershell.exe -Command "netstat -ano | Select-String ':$p '" 2>/dev/null | grep -q LISTENING; then
            break
        fi
    done
    echo "[wasync] port $p: $(powershell.exe -Command "netstat -ano | Select-String ':$p '" 2>/dev/null | grep -c LISTENING) listener(s)"
done
for t in $(seq 1 30); do
    sleep 3
    if powershell.exe -Command "netstat -ano | Select-String ':$EVAL_PORT '" 2>/dev/null | grep -q LISTENING; then
        break
    fi
done
echo "[wasync] eval port $EVAL_PORT: $(powershell.exe -Command "netstat -ano | Select-String ':$EVAL_PORT '" 2>/dev/null | grep -c LISTENING) listener(s)"

cd "$DIR"
COMMON="--workers $WORKERS --base-port $BASE --eval-uri ws://127.0.0.1:$EVAL_PORT \
    --episodes $EPISODES --no-curriculum \
    --log-dir logs/$TAG --save-dir checkpoints/$TAG"
if [ -n "$RESUME" ]; then
    echo "[wasync] resume from $RESUME"
    # shellcheck disable=SC2086
    uv run python train_async.py $COMMON $EXTRA --resume "$RESUME"
else
    echo "[wasync] from scratch + preload data/expert_v4_68.npz"
    # shellcheck disable=SC2086
    uv run python train_async.py $COMMON $EXTRA --preload data/expert_v4_68.npz
fi
RC=$?

echo "[wasync] done (rc=$RC), closing webots..."
cleanup
exit $RC
