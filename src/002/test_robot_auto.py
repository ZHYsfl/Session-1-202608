"""SO-101 自动化测试脚本 - 非交互式版本"""

import sys
sys.path.insert(0, '.')

import serial
import time
import numpy as np
import argparse


class SO101SafeController:
    """SO-101 安全控制器 - Feetech SCS协议"""

    def __init__(self, port: str, baudrate: int = 1000000):
        self.port = port
        self.baudrate = baudrate
        self.servo_ids = [1, 2, 3, 4, 5, 6]
        self.MAX_SPEED = 50
        self.MAX_STEP = 100

        self.serial = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(0.5)
        print(f"✓ 已连接到 {port}")

    def _calculate_checksum(self, data):
        return (~(sum(data) % 256)) & 0xFF

    def _send_packet(self, servo_id, instruction, params=None):
        if params is None:
            params = []
        length = len(params) + 2
        packet_data = [servo_id, length, instruction] + params
        checksum = self._calculate_checksum(packet_data)
        packet = bytes([0xFF, 0xFF] + packet_data + [checksum])
        self.serial.reset_input_buffer()
        self.serial.write(packet)
        time.sleep(0.01)

    def _read_response(self, expected_length=6):
        response = self.serial.read(expected_length)
        return response

    def enable_torque(self, servo_id):
        self._send_packet(servo_id, 0x03, [0x28, 0x01])

    def disable_torque(self, servo_id):
        self._send_packet(servo_id, 0x03, [0x28, 0x00])

    def read_position(self, servo_id):
        self._send_packet(servo_id, 0x02, [0x38, 0x02])
        response = self._read_response(8)
        if len(response) >= 8 and response[0:2] == b'\xff\xff':
            pos_low = response[5]
            pos_high = response[6]
            position = pos_low + (pos_high << 8)
            return position
        return None

    def write_position(self, servo_id, position, speed=None):
        if speed is None:
            speed = self.MAX_SPEED
        speed = min(speed, self.MAX_SPEED)
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        speed_low = speed & 0xFF
        speed_high = (speed >> 8) & 0xFF
        self._send_packet(servo_id, 0x03,
                         [0x2A, pos_low, pos_high, speed_low, speed_high])

    def read_all_positions(self):
        positions = []
        for servo_id in self.servo_ids:
            pos = self.read_position(servo_id)
            positions.append(pos if pos is not None else -1)
        return positions

    def safe_move_single_joint(self, servo_id, target_position):
        current = self.read_position(servo_id)
        if current is None:
            print(f"  ✗ 无法读取舵机{servo_id}位置")
            return False
        step = abs(target_position - current)
        if step > self.MAX_STEP:
            print(f"  ⚠️ 步长过大 ({step} > {self.MAX_STEP}), 拒绝执行")
            return False
        print(f"  舵机{servo_id}: {current} → {target_position} (步长={step})")
        self.write_position(servo_id, target_position)
        return True

    def emergency_stop(self):
        print("\n⚠️ 紧急停止！")
        for servo_id in self.servo_ids:
            self.disable_torque(servo_id)
        print("  所有舵机扭力已关闭")

    def close(self):
        print("\n正在安全关闭...")
        for servo_id in self.servo_ids:
            self.disable_torque(servo_id)
        self.serial.close()
        print("✓ 连接已关闭")


def test_read_positions(controller):
    """读取所有关节位置"""
    print("\n" + "="*70)
    print("读取所有关节当前位置")
    print("="*70)

    positions = controller.read_all_positions()

    joint_names = [
        "底座旋转 (Shoulder Pan)",
        "肩部抬升 (Shoulder Lift)",
        "肘部弯曲 (Elbow Flex)",
        "腕部弯曲 (Wrist Flex)",
        "腕部旋转 (Wrist Roll)",
        "夹爪 (Gripper)"
    ]

    print("\n当前关节位置:")
    for i, (servo_id, pos, name) in enumerate(zip(controller.servo_ids, positions, joint_names)):
        if pos >= 0:
            angle_deg = (pos - 2048) * 240.0 / 4096
            print(f"  关节{servo_id} ({name}): {pos:4d} (约 {angle_deg:6.1f}°)")
        else:
            print(f"  关节{servo_id} ({name}): 读取失败")

    print("\n✓ 读取完成")
    return positions


def test_single_joint(controller, servo_id, move=False):
    """测试单关节"""
    print(f"\n{'='*70}")
    print(f"测试关节 {servo_id}")
    print('='*70)

    initial_pos = controller.read_position(servo_id)
    if initial_pos is None:
        print("✗ 无法读取初始位置")
        return False

    print(f"初始位置: {initial_pos}")

    if not move:
        print("  (仅读取，不移动)")
        return True

    print("\n⚠️ 开始移动...")
    controller.enable_torque(servo_id)
    time.sleep(0.3)

    try:
        # +50
        print("  → 正向移动 +50...")
        target1 = initial_pos + 50
        controller.safe_move_single_joint(servo_id, target1)
        time.sleep(1.5)
        actual1 = controller.read_position(servo_id)
        print(f"    实际: {actual1}")

        # 返回
        print("  → 返回初始位置...")
        controller.safe_move_single_joint(servo_id, initial_pos)
        time.sleep(1.5)
        actual2 = controller.read_position(servo_id)
        print(f"    实际: {actual2}")

        # -50
        print("  → 反向移动 -50...")
        target2 = initial_pos - 50
        controller.safe_move_single_joint(servo_id, target2)
        time.sleep(1.5)
        actual3 = controller.read_position(servo_id)
        print(f"    实际: {actual3}")

        # 最终返回
        print("  → 返回初始位置...")
        controller.safe_move_single_joint(servo_id, initial_pos)
        time.sleep(1.5)

        print("  ✓ 测试完成")
        return True

    finally:
        controller.disable_torque(servo_id)


def main():
    parser = argparse.ArgumentParser(description='SO-101 自动化测试')
    parser.add_argument('--port', type=str, required=True,
                       help='COM端口 (例如 COM23 或 COM24)')
    parser.add_argument('--test', type=str, default='read',
                       choices=['read', 'move1', 'move5'],
                       help='测试类型: read(仅读取), move1(测试关节1), move5(测试前5个关节)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"SO-101 自动化测试 - {args.port}")
    print("="*70)

    try:
        controller = SO101SafeController(args.port)

        if args.test == 'read':
            # 仅读取
            test_read_positions(controller)

        elif args.test == 'move1':
            # 测试关节1
            print("\n⚠️⚠️⚠️ 将开始移动测试！")
            print("  请确保机械臂周围无障碍物")
            print("  准备好随时断电")
            time.sleep(2)

            test_read_positions(controller)
            test_single_joint(controller, servo_id=1, move=True)

        elif args.test == 'move5':
            # 测试前5个关节
            print("\n⚠️⚠️⚠️ 将开始移动测试所有关节！")
            print("  请确保机械臂周围无障碍物")
            print("  准备好随时断电")
            time.sleep(2)

            test_read_positions(controller)

            for servo_id in [1, 2, 3, 4, 5]:
                test_single_joint(controller, servo_id, move=True)
                time.sleep(1)

        print("\n" + "="*70)
        print("✓ 测试完成")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        if 'controller' in locals():
            controller.emergency_stop()

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        if 'controller' in locals():
            controller.emergency_stop()

    finally:
        if 'controller' in locals():
            controller.close()


if __name__ == "__main__":
    main()
