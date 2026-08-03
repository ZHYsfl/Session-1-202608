#!/bin/bash
# 在 Pi 5 上启动 A2 真机 server 用于速度标定（relative 模式，goal 在前方 1.5m）
# 用法：bash run_a2_calibration.sh

set -e

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# 与 start_app.service 启动的底盘节点保持一致
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 确保底盘和雷达节点在跑（标定前已启动）
if ! pgrep -f "car_base_node" >/dev/null; then
    echo "错误：car_base_node 未启动，请先跑 run_a2_real_robot.sh 或手动启动底盘"
    exit 1
fi
if ! pgrep -f "delta2g_scan_node.py" >/dev/null; then
    echo "错误：delta2g_scan_node 未启动"
    exit 1
fi

# 停止 seller APP，避免抢 cmd_vel
if command -v APP >/dev/null 2>&1; then
    APP stop || true
    sleep 1
fi

LIDAR_OFFSET="-61.88"
echo "启动 A2 标定 server (goal_mode=relative, goal_relative_x=1.5) ..."
exec python3 ~/A2/real_robot/A2_real_robot_server.py \
    --lidar-front-offset-deg ${LIDAR_OFFSET} \
    --goal-mode relative \
    --goal-relative-x 1.5 \
    "$@"
