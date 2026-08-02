#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘遥控小车，发布 /cmd_vel。

用法：
    source /opt/ros/jazzy/setup.bash
    python3 teleop_keyboard.py

控制：
    w / ↑     前进
    s / ↓     后退
    a / ←     左转
    d / →     右转
    空格      停止
    q / Ctrl+C 退出
"""

import sys
import termios
import threading
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


LINEAR_SPEED = 0.2   # m/s
ANGULAR_SPEED = 0.5  # rad/s
PUBLISH_RATE = 20.0  # Hz


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_keyboard")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.v = 0.0
        self.w = 0.0
        self.lock = threading.Lock()
        self.create_timer(1.0 / PUBLISH_RATE, self._publish)

    def set_speed(self, v: float, w: float):
        with self.lock:
            self.v = v
            self.w = w

    def _publish(self):
        with self.lock:
            v, w = self.v, self.w
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.pub.publish(twist)
        print(f"\r  v={v:+.2f}  w={w:+.2f}  |  w/s/a/d/space/q", end="", flush=True)


def read_key():
    """读取单个按键（不回车）"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = TeleopNode()

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    print("\n键盘遥控已启动：w/s 前进后退，a/d 左右转，空格停止，q 退出")
    try:
        while True:
            key = read_key()
            if key == "w" or key == "\x1b[A":  # ↑
                node.set_speed(LINEAR_SPEED, 0.0)
            elif key == "s" or key == "\x1b[B":  # ↓
                node.set_speed(-LINEAR_SPEED, 0.0)
            elif key == "a" or key == "\x1b[D":  # ←
                node.set_speed(0.0, ANGULAR_SPEED)
            elif key == "d" or key == "\x1b[C":  # →
                node.set_speed(0.0, -ANGULAR_SPEED)
            elif key == " ":
                node.set_speed(0.0, 0.0)
            elif key == "q":
                break
    finally:
        node.set_speed(0.0, 0.0)
        node._publish()
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        print("\n已退出")


if __name__ == "__main__":
    main()
