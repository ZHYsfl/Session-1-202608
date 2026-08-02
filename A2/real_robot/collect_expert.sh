#!/bin/bash
# 一键：启动 Webots 仿真 + 收集脚本专家数据（生命周期一致，收集完自动关仿真）
# 注意：env_server 单客户端设计——连接断开即退出并关仿真，因此脚本必须一次连接跑到底。
# 用法：bash collect_expert.sh [N_EPISODES] [DUMP_PATH]
set -u
N="${1:-600}"
DUMP="${2:-/home/zane/session_1/A2/003/data/expert_v2.npz}"
WEBOTS="/mnt/d/Program Files/Webots/msys64/mingw64/bin/webots.exe"
WORLD='D:\webots_projects\rl_chassis\worlds\rl_arena.wbt'
URI="${3:-ws://127.0.0.1:8765}"

cleanup() {
    powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force; \
        \$c = netstat -ano | Select-String ':8765'; if (\$c) { \$c | ForEach-Object { \
        \$p = (\$_ -split '\s+')[-1]; Stop-Process -Id \$p -Force -EA SilentlyContinue } }" 2>/dev/null
    sleep 2
}

# 0. 清理残留 webots / 占 8765 的孤儿进程
cleanup

# 1. 启动 Webots 仿真（0.0.0.0 监听，WSL 用 127.0.0.1 转发访问）
(WSLENV=ENV_WS_PORT:ENV_WS_HOST ENV_WS_PORT=8765 ENV_WS_HOST=0.0.0.0 \
    "$WEBOTS" --batch --mode=fast --stdout --stderr "$WORLD" > /tmp/webots_a0.log 2>&1) &
echo "[collect] webots started, waiting 30s for lidar calibration + server up..."
sleep 30

# 2. 跑脚本专家收集数据（全部局跑完才保存 npz；失败则清理重试一次）
RC=1
for attempt in 1 2; do
    echo "[collect] attempt $attempt: scripted_expert $N episodes -> $DUMP (uri=$URI)"
    cd /home/zane/session_1/A2/003
    uv run python /mnt/d/webots_projects/rl_chassis/A2/003/tools/scripted_expert.py \
        --uri "$URI" --episodes "$N" --dump "$DUMP"
    RC=$?
    if [ $RC -eq 0 ]; then
        break
    fi
    echo "[collect] attempt $attempt failed (rc=$RC), restarting webots..."
    cleanup
    (WSLENV=ENV_WS_PORT:ENV_WS_HOST ENV_WS_PORT=8765 ENV_WS_HOST=0.0.0.0 \
        "$WEBOTS" --batch --mode=fast --stdout --stderr "$WORLD" > /tmp/webots_a0.log 2>&1) &
    sleep 30
done

# 3. 收尾：关掉 webots
echo "[collect] expert done (rc=$RC), closing webots..."
powershell.exe -Command "Get-Process webots -EA SilentlyContinue | Stop-Process -Force" 2>/dev/null
exit $RC
