#!/bin/bash
# 一键：启动 N 个 Webots 仿真实例 + 异步 RL 训练（从 0 权重 + 专家数据预填）
# 用法：bash run_sim_async.sh [WORKERS] [EPISODES]
set -u
WORKERS="${1:-4}"
EPISODES="${2:-2000}"
WEBOTS="/mnt/d/Program Files/Webots/msys64/mingw64/bin/webots.exe"
WORLD='D:\webots_projects\rl_chassis\worlds\rl_arena.wbt'
BASE=8765

cleanup() {
    powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force; \
        \$c = netstat -ano | Select-String ':876[5-9]'; if (\$c) { \$c | ForEach-Object { \
        \$p = (\$_ -split '\s+')[-1]; Stop-Process -Id \$p -Force -EA SilentlyContinue } }" 2>/dev/null
    sleep 2
}

cleanup

# 启动 WORKERS 个 Webots 实例（端口 8765+i）
echo "[async] starting $WORKERS webots instances (ports $BASE..$((BASE+WORKERS-1)))..."
for i in $(seq 0 $((WORKERS-1))); do
    p=$((BASE + i))
    (WSLENV=ENV_WS_PORT:ENV_WS_HOST ENV_WS_PORT="$p" ENV_WS_HOST=0.0.0.0 \
        "$WEBOTS" --batch --mode=fast --stdout --stderr "$WORLD" > "/tmp/webots_w${i}.log" 2>&1) &
done
echo "[async] waiting 35s for instances to boot..."
sleep 35

# 确认各端口都在监听
for i in $(seq 0 $((WORKERS-1))); do
    p=$((BASE + i))
    cnt=$(powershell.exe -Command "netstat -ano | Select-String ':$p '" 2>/dev/null | grep -c LISTENING)
    echo "[async] port $p: $cnt listener(s)"
done

# 异步训练：从 0 权重（无 --resume），预填新专家数据
cd /home/zane/session_1/A2/003
echo "[async] train_async.py --workers $WORKERS --episodes $EPISODES --preload data/expert_v2.npz"
uv run python train_async.py --workers "$WORKERS" --episodes "$EPISODES" \
    --preload data/expert_v2.npz \
    --log-dir logs/run_v2 --save-dir checkpoints/run_v2
RC=$?

echo "[async] done (rc=$RC), closing webots..."
powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force" 2>/dev/null
exit $RC
