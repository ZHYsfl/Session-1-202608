#!/bin/bash
# 在 Pi 5 上启动 A2 真机 WebSocket server（带 ROS2 环境）
# 用法：nohup bash start_server.sh > /tmp/a2_server.log 2>&1 &
# 重启：kill $(cat /tmp/a2_server.pid); 再启动

source /opt/ros/jazzy/setup.bash
source /home/pi/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo $$ > /tmp/a2_server.pid
cd /home/pi/A2/real_robot
exec python3 A2_real_robot_server.py "$@"
