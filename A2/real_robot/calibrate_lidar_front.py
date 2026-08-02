#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标定雷达车头方向。

步骤：
  1. 确保 car_base_node 和 delta2g_scan_node.py 在跑。
  2. 在车正前方 0.3~1.0 m 处放一个障碍物（纸箱、腿等）。
  3. 运行本脚本，看 "front_idx" 和 "suggested_offset_deg"。
  4. 用建议的偏移启动 A2 server：
       python3 A2_real_robot_server.py --lidar-front-offset-deg <offset>
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

N = 64


def normalize_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


class CalibNode(Node):
    def __init__(self):
        super().__init__("lidar_front_calib")
        self.sub = self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.got = False

    def on_scan(self, msg: LaserScan):
        if self.got:
            return
        ranges = list(msg.ranges)
        # 取最近点
        valid = [(i, r) for i, r in enumerate(ranges) if math.isfinite(r) and r > 0.05]
        if not valid:
            return
        idx, dist = min(valid, key=lambda x: x[1])
        angle_deg = idx * 360.0 / len(ranges)
        if angle_deg > 180.0:
            angle_deg -= 360.0

        # 重采样到 64 线后，最近点会落在哪个索引（假设 offset=0）
        front_idx = int(round(normalize_angle(math.radians(angle_deg)) / (2.0 * math.pi / N))) % N
        offset_deg = front_idx * 360.0 / N
        if offset_deg > 180.0:
            offset_deg -= 360.0

        print(f"\n原始 scan 最近点: idx={idx}/{len(ranges)}, dist={dist:.3f}m, angle={angle_deg:.2f}°")
        print(f"64 线对应索引: front_idx={front_idx}")
        print(f"建议 lidar_front_offset_deg={offset_deg:.2f}")
        print(f"启动命令: python3 A2_real_robot_server.py --lidar-front-offset-deg {offset_deg:.2f}\n")
        self.got = True


def main():
    rclpy.init()
    node = CalibNode()
    print("等待 /scan ...")
    while rclpy.ok() and not node.got:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
