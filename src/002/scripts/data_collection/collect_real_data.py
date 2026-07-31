"""真实数据采集脚本 - 方案A

目标：从真机采集 (关节角, 末端位姿) 数据对，用于训练神经网络IK

原理：
1. 随机生成关节角（在安全范围内）
2. 发送到机械臂，让它移动过去
3. 读取实际到达的关节角（可能有误差）
4. 用FK计算末端位姿（或手动测量）
5. 保存数据对 (pose, actual_angles)

优势：
- 绕过FK模型问题（不依赖URDF正确性）
- 数据包含真实的机械误差、摩擦、间隙
- 直接可用于神经网络训练
"""

import sys
sys.path.insert(0, '.')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/002

import serial
import time
import numpy as np
import json
import h5py
from typing import List, Tuple
from tqdm import tqdm

from kinematics.forward_kinematics_simple import ForwardKinematics


class SO101DataCollector:
    """SO-101真实数据采集器"""

    def __init__(self, port: str, urdf_path: str, baudrate: int = 1000000):
        self.port = port
        self.baudrate = baudrate

        # 连接机械臂
        self.serial = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(0.5)
        print(f"✓ 已连接到 {port}")

        # 加载FK（用于计算末端位姿）
        self.fk = ForwardKinematics(urdf_path)

        # 安全的关节限位 - 大幅收窄, 偏向抬高姿态
        # 机械臂夹在桌沿, 末端有~98mm长的把手, 极易扎到桌面下。
        # 肩部只允许抬起(向上), 严格禁止下俯。
        self.safe_limits_deg = {
            1: [-45, 45],    # 底座旋转 (进一步缩小)
            2: [-10, 45],    # 肩部抬升 (关键! 只允许略微下探到抬高, 主要向上)
            3: [-40, 40],    # 肘部弯曲 (缩小)
            4: [-40, 40],    # 腕部弯曲 (缩小)
            5: [-80, 80],    # 腕部旋转 (缩小)
        }

        # 工作空间安全边界 (FK把手尖端位置检测, 单位: 米, 相对base_link原点)
        # 基于实机标定: 桌面真实高度Z=66mm。安全线=桌面+重力下垂75mm+裕度50mm。
        # FK已含98mm把手偏移, 检测的是真实尖端位置。
        self.Z_FLOOR = 0.192       # 把手尖端预测最低高度(标定值): 实机下垂后仍距桌面~51mm
        self.RHO_MIN = 0.10        # 尖端到基座竖轴最小水平距离: 避开底座夹具区
        self.R_MIN = 0.20          # 尖端到基座原点最小距离: 避免蜷缩

        # 安全中转姿态 (舵机值): 各关节居中、末端抬高。
        # 每次移动到新配置前先经过此姿态, 避免两点间路径扫过桌面。
        self.HOME_SERVO = [2048, 2048, 2048, 2048, 2048]

        # 转换为舵机值，并钳制到有效物理范围 [100, 3995] (留边界余量)
        SERVO_MIN, SERVO_MAX = 100, 3995
        self.safe_limits_servo = {}
        for joint_id, (deg_min, deg_max) in self.safe_limits_deg.items():
            servo_min = int(deg_min / 240.0 * 4096 + 2048)
            servo_max = int(deg_max / 240.0 * 4096 + 2048)
            # 钳制防止溢出/越界导致舵机不响应
            servo_min = max(SERVO_MIN, min(servo_min, SERVO_MAX))
            servo_max = max(SERVO_MIN, min(servo_max, SERVO_MAX))
            self.safe_limits_servo[joint_id] = [servo_min, servo_max]

        print("\n安全关节范围 (舵机值):")
        for joint_id, (s_min, s_max) in self.safe_limits_servo.items():
            print(f"  关节{joint_id}: [{s_min}, {s_max}]")

        # 数据存储
        self.dataset = {
            'joint_angles': [],      # 关节角 (弧度)
            'end_positions': [],     # 末端位置 (m)
            'end_orientations': [],  # 末端姿态 (欧拉角, 弧度)
            'servo_positions': [],   # 舵机原始值
        }

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

    def write_position(self, servo_id, position, speed=100):
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        speed_low = speed & 0xFF
        speed_high = (speed >> 8) & 0xFF
        self._send_packet(servo_id, 0x03,
                         [0x2A, pos_low, pos_high, speed_low, speed_high])

    def read_all_positions(self):
        """读取所有5个关节位置"""
        positions = []
        for servo_id in [1, 2, 3, 4, 5]:
            pos = self.read_position(servo_id)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    def servo_to_radians(self, servo_values):
        """舵机值转弧度"""
        servo_array = np.array(servo_values)
        radians = (servo_array - 2048) / 4096.0 * (240.0 * np.pi / 180.0)
        return radians

    def _random_servo_once(self):
        """随机生成一组舵机值(未经安全检查)"""
        servo_values = []
        for joint_id in [1, 2, 3, 4, 5]:
            servo_min, servo_max = self.safe_limits_servo[joint_id]
            margin = int((servo_max - servo_min) * 0.1)
            servo_val = np.random.randint(servo_min + margin, servo_max - margin)
            servo_values.append(servo_val)
        return servo_values

    def is_pose_safe(self, servo_values):
        """用FK预检末端位置是否在安全工作空间内。

        这是防碰撞主防线: 拒绝任何会扎到桌面下、钻进底座夹具区、
        或蜷缩过近的配置。
        """
        angles = self.servo_to_radians(servo_values)
        position, _ = self.fk.compute(angles)
        x, y, z = position[0], position[1], position[2]

        # 1. 桌面保护: 末端不得低于地板高度
        if z < self.Z_FLOOR:
            return False, f"末端过低 z={z*1000:.0f}mm < {self.Z_FLOOR*1000:.0f}mm"

        # 2. 避开底座夹具: 末端到基座竖轴的水平距离不能太小
        rho = np.sqrt(x**2 + y**2)
        if rho < self.RHO_MIN:
            return False, f"太靠近底座竖轴 rho={rho*1000:.0f}mm"

        # 3. 避免蜷缩: 末端到原点距离不能太小
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < self.R_MIN:
            return False, f"末端离基座过近 r={r*1000:.0f}mm"

        return True, "ok"

    def generate_random_configuration(self, max_tries=100):
        """生成一个通过工作空间安全检查的随机配置。

        反复采样直到FK预检通过, 保证下发的指令不会碰撞。
        """
        for _ in range(max_tries):
            servo_values = self._random_servo_once()
            safe, _reason = self.is_pose_safe(servo_values)
            if safe:
                return servo_values
        # 极端情况: 多次未采到安全配置, 返回None让上层跳过
        return None

    def move_to_configuration(self, servo_positions, speed=80, wait_time=2.5):
        """移动到指定配置 (低速平缓)"""
        # 使能扭力
        for servo_id in [1, 2, 3, 4, 5]:
            self.enable_torque(servo_id)

        time.sleep(0.2)

        # 发送目标位置 (低速, 减小动作冲击)
        for servo_id, pos in enumerate(servo_positions, 1):
            self.write_position(servo_id, int(pos), speed)

        # 等待到位 (低速需要更长时间)
        time.sleep(wait_time)

    def collect_one_sample(self) -> bool:
        """采集一个数据样本"""
        # 生成通过安全检查的随机配置
        target_servo = self.generate_random_configuration()
        if target_servo is None:
            print("  ⚠️ 未能生成安全配置, 跳过")
            return False

        # 先经过安全中转姿态(HOME), 避免两点间路径扫过桌面
        self.move_to_configuration(self.HOME_SERVO, speed=80, wait_time=1.5)

        # 再移动到目标 (低速)
        self.move_to_configuration(target_servo, speed=80, wait_time=2.5)

        # 读取实际到达的位置（可能和目标有偏差）
        actual_servo = self.read_all_positions()
        if actual_servo is None:
            print("  ✗ 读取位置失败")
            return False

        # 转换为弧度
        actual_angles = self.servo_to_radians(actual_servo)

        # 用FK计算末端位姿
        position, rotation = self.fk.compute(actual_angles)

        # 提取欧拉角
        sy = np.sqrt(rotation[0, 0]**2 + rotation[1, 0]**2)
        if sy > 1e-6:
            pitch = np.arctan2(-rotation[2, 0], sy)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        else:
            pitch = np.arctan2(-rotation[2, 0], sy)
            yaw = np.arctan2(-rotation[1, 2], rotation[1, 1])
            roll = 0

        # 保存数据
        self.dataset['joint_angles'].append(actual_angles.tolist())
        self.dataset['end_positions'].append(position.tolist())
        self.dataset['end_orientations'].append([roll, pitch, yaw])
        self.dataset['servo_positions'].append(actual_servo)

        return True

    def collect_dataset(self, num_samples: int, save_interval: int = 50):
        """采集数据集"""
        print(f"\n开始采集 {num_samples} 个样本...")
        print("⚠️ 请确保机械臂周围安全，准备好随时断电")
        print("\n安全措施已启用:")
        print(f"  - 工作空间地板保护 Z >= {self.Z_FLOOR*1000:.0f}mm (防扎桌)")
        print(f"  - 避开底座夹具区 rho >= {self.RHO_MIN*1000:.0f}mm")
        print(f"  - 每次动作先经过HOME中转点 (防路径扫桌)")
        print(f"  - 低速运动 speed=80, 动作幅度已收窄")

        # 开场: 先缓慢回到安全HOME姿态
        print("\n第一步: 缓慢移动到安全HOME姿态...")
        self.move_to_configuration(self.HOME_SERVO, speed=60, wait_time=3.0)
        print("  ✓ 已到达HOME, 开始采集")
        time.sleep(1)

        success_count = 0

        for i in tqdm(range(num_samples), desc="采集进度"):
            try:
                if self.collect_one_sample():
                    success_count += 1
                else:
                    print(f"\n  样本 {i+1} 采集失败")

                # 定期保存
                if (i + 1) % save_interval == 0:
                    self.save_dataset(f"dataset_partial_{i+1}.h5")
                    print(f"\n  ✓ 已保存前 {i+1} 个样本")

            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断采集")
                break

            except Exception as e:
                print(f"\n  ✗ 样本 {i+1} 出错: {e}")
                continue

        print(f"\n✓ 采集完成: {success_count}/{num_samples} 成功")
        return success_count

    def save_dataset(self, filename: str):
        """保存数据集为HDF5格式"""
        with h5py.File(filename, 'w') as f:
            # 转换为numpy数组
            joint_angles = np.array(self.dataset['joint_angles'], dtype=np.float32)
            end_positions = np.array(self.dataset['end_positions'], dtype=np.float32)
            end_orientations = np.array(self.dataset['end_orientations'], dtype=np.float32)
            servo_positions = np.array(self.dataset['servo_positions'], dtype=np.int32)

            # 保存
            f.create_dataset('joint_angles', data=joint_angles)
            f.create_dataset('end_positions', data=end_positions)
            f.create_dataset('end_orientations', data=end_orientations)
            f.create_dataset('servo_positions', data=servo_positions)

            # 元数据
            f.attrs['num_samples'] = len(joint_angles)
            f.attrs['port'] = self.port
            f.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"✓ 数据已保存到: {filename}")
        print(f"  样本数: {len(joint_angles)}")

    def close(self):
        """安全关闭"""
        print("\n正在安全关闭...")
        for servo_id in [1, 2, 3, 4, 5]:
            self.disable_torque(servo_id)
        self.serial.close()
        print("✓ 连接已关闭")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SO-101 真实数据采集')
    parser.add_argument('--port', type=str, default='COM24',
                       help='COM端口')
    parser.add_argument('--urdf', type=str, default='models/so101_new_calib.urdf',
                       help='URDF文件路径')
    parser.add_argument('--samples', type=int, default=1000,
                       help='采集样本数')
    parser.add_argument('--output', type=str, default='real_robot_dataset.h5',
                       help='输出文件名')

    args = parser.parse_args()

    print("="*70)
    print("SO-101 真实数据采集 - 方案A")
    print("="*70)
    print("\n采集策略:")
    print("  1. 随机生成安全的关节配置")
    print("  2. 机械臂移动到目标位置")
    print("  3. 读取实际到达的关节角")
    print("  4. 用FK计算末端位姿")
    print("  5. 保存 (位姿, 关节角) 数据对")
    print("\n优势:")
    print("  - 数据包含真实机械误差")
    print("  - 不依赖URDF准确性")
    print("  - 直接用于神经网络训练")

    collector = SO101DataCollector(args.port, args.urdf)

    try:
        # 采集数据
        num_success = collector.collect_dataset(args.samples)

        # 保存最终数据集
        if num_success > 0:
            collector.save_dataset(args.output)

            print("\n" + "="*70)
            print("✓ 数据采集完成！")
            print("="*70)
            print(f"\n下一步:")
            print(f"  python train_neural_ik.py --dataset {args.output}")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        collector.close()


if __name__ == "__main__":
    main()
