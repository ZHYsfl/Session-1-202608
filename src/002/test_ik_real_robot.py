"""IK算法真机验证

测试流程:
1. 读取当前关节角 θ_current
2. 计算当前末端位姿 P_current = FK(θ_current)
3. 用IK求解: θ_solution = IK(P_current)
4. 验证: FK(θ_solution) 应该 ≈ P_current
5. 发送到机械臂，测量实际到达位姿
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import time
import numpy as np
import json
import yaml
from typing import List, Tuple

from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.analytical_ik import AnalyticalIK
from kinematics.numerical_ik import NumericalIK


class SO101Controller:
    """SO-101控制器"""

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

    def write_position(self, servo_id, position, speed=50):
        speed = min(speed, 100)
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        speed_low = speed & 0xFF
        speed_high = (speed >> 8) & 0xFF
        self._send_packet(servo_id, 0x03,
                         [0x2A, pos_low, pos_high, speed_low, speed_high])

    def read_all_positions(self):
        positions = []
        for servo_id in [1, 2, 3, 4, 5]:
            pos = self.read_position(servo_id)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    def write_all_positions(self, positions, speed=50):
        """写入5个关节位置"""
        for servo_id, pos in enumerate(positions, 1):
            self.write_position(servo_id, int(pos), speed)

    def servo_to_radians(self, servo_values):
        servo_array = np.array(servo_values)
        radians = (servo_array - 2048) / 4096.0 * (240.0 * np.pi / 180.0)
        return radians

    def radians_to_servo(self, radians):
        rad_array = np.array(radians)
        servo = rad_array / (240.0 * np.pi / 180.0) * 4096.0 + 2048
        return np.clip(servo, 0, 4095).astype(int)

    def close(self):
        for servo_id in [1, 2, 3, 4, 5]:
            self.disable_torque(servo_id)
        self.serial.close()


def pose_error(pose1, pose2):
    """计算位姿误差"""
    pos_error = np.linalg.norm(pose1[:3] - pose2[:3])
    orient_error = np.linalg.norm(pose1[3:] - pose2[3:])
    return pos_error, orient_error


def test_analytical_ik(controller, fk, ik_solver, test_name="解析IK"):
    """测试解析IK"""
    print("\n" + "="*70)
    print(f"测试: {test_name}")
    print("="*70)

    # 读取当前关节角
    servo_pos = controller.read_all_positions()
    if servo_pos is None:
        print("✗ 无法读取关节位置")
        return None

    current_angles = controller.servo_to_radians(servo_pos)

    print("\n1. 当前配置:")
    print(f"  关节角 (deg): {np.degrees(current_angles)}")

    # 计算当前位姿 - 用task-space一致定义
    position, alpha, psi = fk.compute_task_space(current_angles)

    print(f"  末端位姿:")
    print(f"    位置 (m): ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})")
    print(f"    姿态 (deg): alpha(pitch)={np.degrees(alpha):.2f}°, psi(roll)={np.degrees(psi):.2f}°")

    # IK求解
    print("\n2. IK求解...")
    solutions = ik_solver.solve(position, alpha, psi)

    if solutions is None:
        print("  ✗ 无解 (目标超出简化模型可达范围)")
        return None

    # analytical solve 返回单个解或None
    best_solution = solutions
    dist = np.linalg.norm(best_solution - current_angles)

    print(f"\n  ✓ 解析解 (距离当前={dist:.4f} rad)")
    print(f"    解: {np.degrees(best_solution)}")

    # 验证FK
    verify_pos, verify_rot = fk.compute(best_solution)
    pos_error = np.linalg.norm(verify_pos - position)

    print(f"\n3. FK验证:")
    print(f"  原始位置: {position}")
    print(f"  解的位置: {verify_pos}")
    print(f"  位置误差: {pos_error*1000:.2f} mm")
    print(f"  (注: 解析IK用简化DH参数, 与真实URDF有差异, 误差偏大属正常)")

    if pos_error < 0.01:  # 10mm
        print("  ✓ IK解正确")
    else:
        print("  ⚠️ IK解误差较大 (简化模型所致)")

    return {
        'current_angles': current_angles.tolist(),
        'target_position': position.tolist(),
        'target_alpha': float(alpha),
        'target_psi': float(psi),
        'solution': best_solution.tolist(),
        'position_error_mm': pos_error * 1000
    }


def test_numerical_ik(controller, fk, ik_solver, test_name="数值IK", move_robot=False):
    """测试数值IK"""
    print("\n" + "="*70)
    print(f"测试: {test_name}")
    print("="*70)

    # 读取当前关节角
    servo_pos = controller.read_all_positions()
    if servo_pos is None:
        print("✗ 无法读取关节位置")
        return None

    current_angles = controller.servo_to_radians(servo_pos)

    print("\n1. 当前配置:")
    print(f"  关节角 (deg): {np.degrees(current_angles)}")

    # 计算当前位姿 - 必须用求解器一致的task-space定义 (alpha, psi)
    # 否则姿态误差永远无法归零，导致IK不收敛
    position, alpha, psi = fk.compute_task_space(current_angles)

    print(f"  末端位姿:")
    print(f"    位置 (m): ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})")
    print(f"    姿态 (deg): alpha(pitch)={np.degrees(alpha):.2f}°, psi(roll)={np.degrees(psi):.2f}°")

    # 创建目标位姿 (微小偏移) - 姿态保持当前值
    delta_pos = np.array([0.01, 0.00, 0.00])  # X方向+10mm
    target_position = position + delta_pos

    print(f"\n2. 目标位姿 (X+10mm):")
    print(f"    位置 (m): ({target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f})")
    print(f"    姿态 (deg): alpha={np.degrees(alpha):.2f}°, psi={np.degrees(psi):.2f}°")

    # IK求解 - 用task-space的alpha/psi
    print("\n3. 数值IK求解...")
    solution = ik_solver.solve(target_position, alpha, psi, initial_guess=current_angles)

    if solution is None:
        print("  ✗ 求解失败")
        return None

    print(f"  ✓ 求解成功")
    print(f"    解: {np.degrees(solution)}")

    # 验证FK
    verify_pos, verify_rot = fk.compute(solution)
    pos_error = np.linalg.norm(verify_pos - target_position)

    print(f"\n4. FK验证:")
    print(f"  目标位置: {target_position}")
    print(f"  解的位置: {verify_pos}")
    print(f"  位置误差: {pos_error*1000:.2f} mm")

    if pos_error < 0.01:
        print("  ✓ 数值IK解正确")
    else:
        print("  ⚠️ 数值IK解误差较大")

    # 真机执行
    result = {
        'current_angles': current_angles.tolist(),
        'target_position': target_position.tolist(),
        'target_alpha': float(alpha),
        'target_psi': float(psi),
        'solution': solution.tolist(),
        'position_error_mm': pos_error * 1000
    }

    if move_robot:
        print("\n5. 发送到机械臂...")
        print("  ⚠️⚠️⚠️ 准备移动！")
        time.sleep(2)

        # 使能扭力
        for servo_id in [1, 2, 3, 4, 5]:
            controller.enable_torque(servo_id)

        # 转换并发送
        target_servo = controller.radians_to_servo(solution)
        print(f"  目标舵机值: {target_servo}")

        controller.write_all_positions(target_servo, speed=30)
        print("  ✓ 已发送，等待到位...")
        time.sleep(3)

        # 读取实际位置
        actual_servo = controller.read_all_positions()
        actual_angles = controller.servo_to_radians(actual_servo)
        actual_pos, actual_rot = fk.compute(actual_angles)

        print(f"\n6. 实际到达:")
        print(f"  实际关节角: {np.degrees(actual_angles)}")
        print(f"  实际位置: {actual_pos}")

        actual_error = np.linalg.norm(actual_pos - target_position)
        print(f"  实际误差: {actual_error*1000:.2f} mm")

        result['actual_angles'] = actual_angles.tolist()
        result['actual_position'] = actual_pos.tolist()
        result['actual_error_mm'] = actual_error * 1000

        # 返回初始位置
        print("\n7. 返回初始位置...")
        controller.write_all_positions(servo_pos, speed=30)
        time.sleep(3)

        for servo_id in [1, 2, 3, 4, 5]:
            controller.disable_torque(servo_id)

        print("  ✓ 已返回")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SO-101 IK真机验证')
    parser.add_argument('--port', type=str, default='COM24')
    parser.add_argument('--urdf', type=str, default='models/so101_new_calib.urdf')
    parser.add_argument('--config', type=str, default='configs/robot_config.yaml')
    parser.add_argument('--test', type=str, default='all',
                       choices=['analytical', 'numerical', 'all'])
    parser.add_argument('--move', action='store_true',
                       help='实际移动机械臂（数值IK）')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SO-101 IK算法真机验证")
    print("="*70)

    # 初始化
    controller = SO101Controller(args.port)
    fk = ForwardKinematics(args.urdf)

    results = {}

    try:
        # 读取配置（解析IK和数值IK都要用）
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if args.test in ['analytical', 'all']:
            # 解析IK - 从config读取link长度
            ll = config['robot']['link_lengths']
            jl = config['robot']['joint_limits']
            jnames = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
            lower = [float(jl[n][0]) for n in jnames]
            upper = [float(jl[n][1]) for n in jnames]

            analytical_ik = AnalyticalIK(
                shoulder_height=float(ll['shoulder_height']),
                radial_offset=float(ll['radial_offset']),
                upper_arm_length=float(ll['upper_arm']),
                forearm_length=float(ll['forearm']),
                wrist_to_tcp=float(ll['wrist_to_tcp']),
                joint_limits=(lower, upper),
            )
            result = test_analytical_ik(controller, fk, analytical_ik)
            if result:
                results['analytical'] = result

        if args.test in ['numerical', 'all']:
            # 数值IK - 需要joint_limits
            joint_limits_dict = config['robot']['joint_limits']
            # 按照关节顺序提取限制
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
            lower_limits = np.array([joint_limits_dict[name][0] for name in joint_names], dtype=np.float64)
            upper_limits = np.array([joint_limits_dict[name][1] for name in joint_names], dtype=np.float64)

            numerical_ik = NumericalIK(fk, (lower_limits, upper_limits))
            result = test_numerical_ik(controller, fk, numerical_ik,
                                      move_robot=args.move)
            if result:
                results['numerical'] = result

        # 保存结果
        if results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ik_validation_{args.port}_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)

            print("\n" + "="*70)
            print(f"✓ 结果保存到: {filename}")
            print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        controller.close()


if __name__ == "__main__":
    main()
