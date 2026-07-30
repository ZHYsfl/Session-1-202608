"""SO-101 真机测试脚本 - 安全版本

⚠️ 安全警告：
1. 确保机械臂周围1米范围内无障碍物和人员
2. 准备好随时拔掉12V电源（紧急停止）
3. 首次测试使用最低速度和最小幅度
4. 不要在机械臂运动时触摸
"""

import sys
sys.path.insert(0, '.')

import serial
import time
import numpy as np


class SO101SafeController:
    """SO-101 安全控制器 - Feetech SCS协议"""

    def __init__(self, port: str, baudrate: int = 1000000):
        """初始化控制器

        Args:
            port: COM端口 (如 'COM23' 或 'COM24')
            baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.servo_ids = [1, 2, 3, 4, 5, 6]

        # 安全限制
        self.MAX_SPEED = 50  # 最大速度 (0-1023, 50很慢)
        self.MAX_STEP = 100   # 最大单步移动量 (舵机单位)

        # 连接
        self.serial = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(0.5)
        print(f"✓ 已连接到 {port}")

    def _calculate_checksum(self, data):
        """计算Feetech校验和"""
        return (~(sum(data) % 256)) & 0xFF

    def _send_packet(self, servo_id, instruction, params=None):
        """发送数据包"""
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
        """读取响应"""
        response = self.serial.read(expected_length)
        return response

    def enable_torque(self, servo_id):
        """使能舵机扭力"""
        # Feetech: 写入地址0x28(40), 值1
        self._send_packet(servo_id, 0x03, [0x28, 0x01])
        print(f"  使能舵机 {servo_id}")

    def disable_torque(self, servo_id):
        """关闭舵机扭力"""
        self._send_packet(servo_id, 0x03, [0x28, 0x00])
        print(f"  关闭舵机 {servo_id}")

    def read_position(self, servo_id):
        """读取当前位置

        Returns:
            位置值 (0-4095) 或 None
        """
        # Feetech: 读地址0x38(56), 长度2字节
        self._send_packet(servo_id, 0x02, [0x38, 0x02])

        response = self._read_response(8)
        if len(response) >= 8 and response[0:2] == b'\xff\xff':
            pos_low = response[5]
            pos_high = response[6]
            position = pos_low + (pos_high << 8)
            return position
        return None

    def write_position(self, servo_id, position, speed=None):
        """写入目标位置

        Args:
            servo_id: 舵机ID
            position: 目标位置 (0-4095)
            speed: 速度 (0-1023), None使用默认低速
        """
        if speed is None:
            speed = self.MAX_SPEED

        # 限制速度
        speed = min(speed, self.MAX_SPEED)

        # Feetech: 写地址0x2A(42), 位置2字节+速度2字节
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        speed_low = speed & 0xFF
        speed_high = (speed >> 8) & 0xFF

        self._send_packet(servo_id, 0x03,
                         [0x2A, pos_low, pos_high, speed_low, speed_high])

    def read_all_positions(self):
        """读取所有关节位置"""
        positions = []
        for servo_id in self.servo_ids:
            pos = self.read_position(servo_id)
            positions.append(pos if pos is not None else -1)
        return positions

    def safe_move_single_joint(self, servo_id, target_position):
        """安全移动单个关节

        Args:
            servo_id: 舵机ID (1-6)
            target_position: 目标位置 (0-4095)

        Returns:
            True if successful, False otherwise
        """
        # 读取当前位置
        current = self.read_position(servo_id)
        if current is None:
            print(f"  ✗ 无法读取舵机{servo_id}位置")
            return False

        # 检查步长
        step = abs(target_position - current)
        if step > self.MAX_STEP:
            print(f"  ⚠️ 步长过大 ({step} > {self.MAX_STEP}), 拒绝执行")
            return False

        # 执行移动
        print(f"  舵机{servo_id}: {current} → {target_position} (步长={step})")
        self.write_position(servo_id, target_position)
        return True

    def emergency_stop(self):
        """紧急停止 - 关闭所有舵机扭力"""
        print("\n⚠️ 紧急停止！")
        for servo_id in self.servo_ids:
            self.disable_torque(servo_id)
        print("  所有舵机扭力已关闭")

    def close(self):
        """安全关闭连接"""
        print("\n正在安全关闭...")
        for servo_id in self.servo_ids:
            self.disable_torque(servo_id)
        self.serial.close()
        print("✓ 连接已关闭")


def test_read_positions(controller):
    """测试1: 读取当前位置"""
    print("\n" + "="*70)
    print("测试1: 读取所有关节当前位置")
    print("="*70)

    positions = controller.read_all_positions()

    print("\n当前关节位置:")
    joint_names = [
        "底座旋转 (Shoulder Pan)",
        "肩部抬升 (Shoulder Lift)",
        "肘部弯曲 (Elbow Flex)",
        "腕部弯曲 (Wrist Flex)",
        "腕部旋转 (Wrist Roll)",
        "夹爪 (Gripper)"
    ]

    for i, (servo_id, pos, name) in enumerate(zip(controller.servo_ids, positions, joint_names)):
        if pos >= 0:
            angle_deg = (pos - 2048) * 240.0 / 4096  # 近似转换为角度
            print(f"  关节{servo_id} ({name}): {pos} (约 {angle_deg:.1f}°)")
        else:
            print(f"  关节{servo_id} ({name}): 读取失败")

    print("\n✓ 读取完成")
    return positions


def test_single_joint_small_move(controller, servo_id=1):
    """测试2: 单关节小幅度运动"""
    print("\n" + "="*70)
    print(f"测试2: 单关节小幅度运动 (关节{servo_id})")
    print("="*70)

    input("\n⚠️ 请确认:")
    input("  1. 机械臂周围无障碍物")
    input("  2. 准备好随时断电")
    input("  按Enter继续，或Ctrl+C取消...")

    # 读取初始位置
    initial_pos = controller.read_position(servo_id)
    if initial_pos is None:
        print("✗ 无法读取初始位置")
        return False

    print(f"\n初始位置: {initial_pos}")

    # 使能扭力
    controller.enable_torque(servo_id)
    time.sleep(0.5)

    try:
        # 小幅度移动: +50单位
        print("\n第1步: 正向移动 +50...")
        target1 = initial_pos + 50
        controller.safe_move_single_joint(servo_id, target1)
        time.sleep(2)

        actual1 = controller.read_position(servo_id)
        print(f"  实际位置: {actual1}")

        # 返回原位
        print("\n第2步: 返回初始位置...")
        controller.safe_move_single_joint(servo_id, initial_pos)
        time.sleep(2)

        actual2 = controller.read_position(servo_id)
        print(f"  实际位置: {actual2}")

        # 反向移动: -50单位
        print("\n第3步: 反向移动 -50...")
        target2 = initial_pos - 50
        controller.safe_move_single_joint(servo_id, target2)
        time.sleep(2)

        actual3 = controller.read_position(servo_id)
        print(f"  实际位置: {actual3}")

        # 最终返回原位
        print("\n第4步: 返回初始位置...")
        controller.safe_move_single_joint(servo_id, initial_pos)
        time.sleep(2)

        print("\n✓ 测试完成")
        return True

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断！")
        controller.emergency_stop()
        return False

    finally:
        controller.disable_torque(servo_id)


def test_all_joints_sequential(controller):
    """测试3: 逐个测试所有关节"""
    print("\n" + "="*70)
    print("测试3: 逐个测试所有关节")
    print("="*70)

    print("\n将依次测试每个关节（小幅度运动）")
    input("按Enter开始，或Ctrl+C取消...")

    for servo_id in controller.servo_ids[:5]:  # 前5个关节（不含夹爪）
        print(f"\n{'='*70}")
        print(f"测试关节 {servo_id}")
        print('='*70)

        success = test_single_joint_small_move(controller, servo_id)

        if not success:
            print(f"✗ 关节{servo_id}测试失败")
            break

        if servo_id < 5:
            input("\n按Enter继续测试下一个关节...")

    print("\n✓ 所有关节测试完成")


def main():
    """主测试流程"""
    print("\n" + "="*70)
    print("SO-101 真机安全测试")
    print("="*70)

    print("\n⚠️⚠️⚠️ 安全警告 ⚠️⚠️⚠️")
    print("  1. 确保机械臂周围1米范围内无障碍物")
    print("  2. 确保所有人员远离机械臂")
    print("  3. 准备好随时拔掉12V电源插头")
    print("  4. 不要在机械臂运动时触摸")
    print("  5. 听到异常声音立即断电")

    print("\n请选择要测试的机械臂:")
    print("  1. 主臂 (COM23)")
    print("  2. 从臂 (COM24)")

    choice = input("\n输入选择 [1/2]: ").strip()

    if choice == '1':
        port = 'COM23'
        print("\n✓ 选择: 主臂 (COM23)")
    elif choice == '2':
        port = 'COM24'
        print("\n✓ 选择: 从臂 (COM24)")
    else:
        print("✗ 无效选择")
        return

    input("\n最后确认: 按Enter开始测试，或Ctrl+C取消...")

    try:
        # 连接控制器
        controller = SO101SafeController(port)

        # 测试1: 读取位置
        positions = test_read_positions(controller)

        # 询问是否继续运动测试
        print("\n" + "="*70)
        response = input("\n是否继续运动测试? (y/n): ").strip().lower()

        if response == 'y':
            # 测试2: 单关节测试
            test_single_joint_small_move(controller, servo_id=1)

            # 询问是否测试所有关节
            response = input("\n是否测试所有关节? (y/n): ").strip().lower()
            if response == 'y':
                test_all_joints_sequential(controller)

        print("\n" + "="*70)
        print("测试完成！")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序退出")
