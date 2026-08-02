#!/bin/bash
# 在 Pi 5 上启动 A2 真机 server 所需的全套 ROS2 节点
# 用法：bash run_a2_real_robot.sh

set -e

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# 与 start_app.service 启动的底盘节点保持一致
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 1. 停止卖家 APP，避免 rplidar 节点抢串口
if command -v APP >/dev/null 2>&1; then
    echo "停止 APP..."
    APP stop || true
    sleep 1
fi

# 2. 确保底盘节点在跑（发布 /odom，订阅 /cmd_vel）
if ! pgrep -f "car_base_node" >/dev/null; then
    echo "启动底盘节点..."
    ros2 run car_base car_base_node --ros-args \
        -p usart_port_name:=/dev/ttyAMA0 \
        -p serial_baud_rate:=115200 &
    sleep 2
fi

# 3. 启动 Delta-2G 雷达节点（发布 /scan）
if ! pgrep -f "delta2g_scan_node.py" >/dev/null; then
    echo "启动 Delta-2G 雷达节点..."
    python3 ~/delta2g_scan_node.py &
    sleep 2
fi

# 4. 启动 A2 WebSocket server
# 默认 manual-drive 训练模式，已标定的 lidar 偏移
LIDAR_OFFSET="-61.88"
echo "启动 A2 真机 WebSocket server (goal_mode=manual-drive, lidar_offset=${LIDAR_OFFSET})..."
python3 ~/A2_real_robot_server.py \
    --lidar-front-offset-deg ${LIDAR_OFFSET} \
    --goal-mode manual-drive \
    "$@"
