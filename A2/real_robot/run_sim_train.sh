#!/bin/bash
# 一键：启动 Webots 仿真 + 从 0 权重训练（预填新专家数据），训练结束自动关仿真
# 用法：bash run_sim_train.sh [EPISODES] [EXTRA_ARGS...]
set -u
EPISODES="${1:-2000}"
shift || true
WEBOTS="/mnt/d/Program Files/Webots/msys64/mingw64/bin/webots.exe"
WORLD='D:\webots_projects\rl_chassis\worlds\rl_arena.wbt'
URI="ws://127.0.0.1:8765"

cleanup() {
    powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force; \
        \$c = netstat -ano | Select-String ':8765'; if (\$c) { \$c | ForEach-Object { \
        \$p = (\$_ -split '\s+')[-1]; Stop-Process -Id \$p -Force -EA SilentlyContinue } }" 2>/dev/null
    sleep 2
}

cleanup

# 启动 Webots 仿真
(WSLENV=ENV_WS_PORT:ENV_WS_HOST ENV_WS_PORT=8765 ENV_WS_HOST=0.0.0.0 \
    "$WEBOTS" --batch --mode=fast --stdout --stderr "$WORLD" > /tmp/webots_train.log 2>&1) &
echo "[train] webots started, waiting 30s..."
sleep 30

# 从 0 权重训练：--preload 新专家数据，不带 --resume
cd /home/zane/session_1/A2/003
echo "[train] start: python train.py --uri $URI --preload data/expert_v2.npz --episodes $EPISODES $*"
uv run python train.py --uri "$URI" --preload data/expert_v2.npz --episodes "$EPISODES" "$@"
RC=$?

# 收尾
echo "[train] done (rc=$RC), closing webots..."
powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force" 2>/dev/null
exit $RC
