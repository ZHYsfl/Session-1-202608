#!/bin/bash
# 异步 RL：启动 N 个 Webots 实例，每个用环境变量 ENV_WS_PORT 指定端口
# （WSLENV 让 WSL→Windows 进程透传该变量；webots 起 controller 时继承）。
# 用法：./start_async_instances.sh [N] [BASE_PORT]   # 默认 4 个，8765 起
set -u
N="${1:-4}"
BASE="${2:-8765}"
WEBOTS="/mnt/d/Program Files/Webots/msys64/mingw64/bin/webots.exe"
WORLD="D:\\webots_projects\\rl_chassis\\worlds\\rl_arena.wbt"

powershell.exe -Command "Get-Process webots -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
sleep 2
for i in $(seq 0 $((N-1))); do
  p=$((BASE + i))
  (WSLENV=ENV_WS_PORT ENV_WS_PORT="$p" "$WEBOTS" --batch --mode=fast \
     --stdout --stderr "$WORLD" > "/tmp/webots_a${i}.log" 2>&1) &
done
echo "已启动 $N 个 webots 实例：端口 $BASE..$((BASE+N-1))，日志 /tmp/webots_a{i}.log"
wait
