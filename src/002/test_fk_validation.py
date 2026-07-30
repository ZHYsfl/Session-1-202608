"""正向运动学真机验证

目的：
1. 读取机械臂当前关节角
2. 用FK计算末端位姿
3. 手动测量实际末端位置
4. 对比计算值与实测值，验证FK准确性
"""

import sys
sys.path.insert(0, '.')

import serial
import time
import numpy as np
from pathlib import Path

# 导入FK
sys.path.insert(0, str(Path(__file__).parent.parent))
from kinematics.forward_kinematics_simple import ForwardKinematics


class SO101Controller:
    """简化的SO-101控制器"""

    def __init__(self, port: str, baudrate: int = 1000000):
        self.port = port
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
        return self.serial.read(expected_length)

    def read_position(self, servo_id):
        """读取舵机位置 (0-4095)"""
        self._send_packet(servo_id, 0x02, [0x38, 0x02])
        response = self._read_response(8)
        if len(response) >= 8 and response[0:2] == b'\xff\xff':
            pos_low = response[5]
            pos_high = response[6]
            position = pos_low + (pos_high << 8)
            return position
        return None

    def read_all_positions(self):
        """读取前5个关节位置"""
        positions = []
        for servo_id in [1, 2, 3, 4, 5]:
            pos = self.read_position(servo_id)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    def servo_to_radians(self, servo_values):
        """舵机值转弧度

        舵机范围: 0-4095
        中心位置: 2048
        范围: ±240度 = ±4.189 rad
        """
        servo_array = np.array(servo_values)
        # (servo - 2048) / 4096 * 240° * π/180
        radians = (servo_array - 2048) / 4096.0 * (240.0 * np.pi / 180.0)
        return radians

    def close(self):
        self.serial.close()


def validate_fk(port='COM24', urdf_path='models/so101_new_calib.urdf'):
    """FK验证主流程"""

    print("\n" + "="*70)
    print("SO-101 正向运动学真机验证")
    print("="*70)

    # 初始化
    controller = SO101Controller(port)
    fk = ForwardKinematics(urdf_path)

    try:
        # 读取当前关节位置
        print("\n1. 读取当前关节位置...")
        servo_positions = controller.read_all_positions()

        if servo_positions is None:
            print("✗ 无法读取关节位置")
            return

        print("\n舵机位置:")
        for i, pos in enumerate(servo_positions, 1):
            print(f"  关节{i}: {pos:4d}")

        # 转换为弧度
        joint_angles_rad = controller.servo_to_radians(servo_positions)

        print("\n关节角度 (弧度):")
        for i, angle in enumerate(joint_angles_rad, 1):
            deg = np.degrees(angle)
            print(f"  关节{i}: {angle:7.4f} rad ({deg:6.2f}°)")

        # 计算FK
        print("\n2. 计算正向运动学...")
        position, rotation_matrix = fk.compute(joint_angles_rad)

        # 提取旋转角度 (欧拉角)
        # 这里使用简化的ZYX欧拉角提取
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

        if sy > 1e-6:
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            roll = 0

        print("\n计算结果:")
        print(f"  位置 (x, y, z): ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m")
        print(f"  转换为 mm:     ({position[0]*1000:.1f}, {position[1]*1000:.1f}, {position[2]*1000:.1f}) mm")
        print(f"  姿态 (roll, pitch, yaw):")
        print(f"    Roll  (绕X): {np.degrees(roll):7.2f}°")
        print(f"    Pitch (绕Y): {np.degrees(pitch):7.2f}°")
        print(f"    Yaw   (绕Z): {np.degrees(yaw):7.2f}°")

        print("\n旋转矩阵:")
        print(rotation_matrix)

        # 提示手动测量
        print("\n" + "="*70)
        print("3. 手动测量验证")
        print("="*70)
        print("\n请使用卷尺或游标卡尺测量末端执行器（夹爪中心）相对于：")
        print("  - 基座中心的水平距离 (X-Y平面距离)")
        print("  - 基座底面的高度 (Z)")
        print("\n测量说明:")
        print("  X轴: 机械臂正前方（初始朝向）")
        print("  Y轴: 左侧为正")
        print("  Z轴: 向上为正")
        print("  基座中心: 底座旋转轴心")

        print("\n" + "-"*70)
        print("实测值输入 (直接按Enter跳过)")
        print("-"*70)

        try:
            actual_x = input("实测 X (mm): ").strip()
            actual_y = input("实测 Y (mm): ").strip()
            actual_z = input("实测 Z (mm): ").strip()

            if actual_x and actual_y and actual_z:
                actual_x = float(actual_x)
                actual_y = float(actual_y)
                actual_z = float(actual_z)

                print("\n" + "="*70)
                print("对比结果")
                print("="*70)

                calc_x = position[0] * 1000
                calc_y = position[1] * 1000
                calc_z = position[2] * 1000

                error_x = calc_x - actual_x
                error_y = calc_y - actual_y
                error_z = calc_z - actual_z
                error_total = np.sqrt(error_x**2 + error_y**2 + error_z**2)

                print(f"\n{'':10s} {'计算值':>12s} {'实测值':>12s} {'误差':>12s}")
                print("-"*50)
                print(f"{'X (mm)':10s} {calc_x:12.1f} {actual_x:12.1f} {error_x:12.1f}")
                print(f"{'Y (mm)':10s} {calc_y:12.1f} {actual_y:12.1f} {error_y:12.1f}")
                print(f"{'Z (mm)':10s} {calc_z:12.1f} {actual_z:12.1f} {error_z:12.1f}")
                print("-"*50)
                print(f"{'总误差':10s} {error_total:12.1f} mm")

                print("\n分析:")
                if error_total < 10:
                    print("  ✓ 误差很小 (<10mm)，FK模型准确")
                elif error_total < 30:
                    print("  ⚠️ 误差中等 (10-30mm)，可能需要标定")
                else:
                    print("  ✗ 误差较大 (>30mm)，需要检查URDF参数")

        except ValueError:
            print("\n✗ 输入格式错误，跳过对比")
        except EOFError:
            print("\n(跳过手动测量)")

        # 保存记录
        print("\n" + "="*70)
        print("数据已记录")
        print("="*70)

        record = {
            'servo_positions': servo_positions,
            'joint_angles_rad': joint_angles_rad.tolist(),
            'position_m': position.tolist(),
            'rotation_matrix': rotation_matrix.tolist(),
            'euler_deg': [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
        }

        import json
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"fk_validation_{port}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(record, f, indent=2)

        print(f"✓ 数据保存到: {filename}")

    finally:
        controller.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SO-101 FK验证')
    parser.add_argument('--port', type=str, default='COM24',
                       help='COM端口 (默认 COM24)')
    parser.add_argument('--urdf', type=str,
                       default='models/so101_new_calib.urdf',
                       help='URDF文件路径')

    args = parser.parse_args()

    validate_fk(args.port, args.urdf)
