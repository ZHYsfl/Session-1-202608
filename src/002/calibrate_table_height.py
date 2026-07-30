"""桌面高度标定

原理:
1. 关闭舵机扭力, 机械臂变"软"可手动搬动
2. 用户手动把把手尖端轻触桌面
3. 读取关节角, 用FK算出把手尖端Z = 桌面真实高度
4. 多点采样取平均, 得到稳健的桌面高度基准

输出: 桌面Z值, 用于设定绝对安全的采集高度线
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial, time, json
import numpy as np
from kinematics.forward_kinematics_simple import ForwardKinematics


class Ctrl:
    def __init__(self, port, baud=1000000):
        self.serial = serial.Serial(port, baud, timeout=0.5)
        time.sleep(0.5)
        print(f"[OK] connected {port}")

    def _ck(self, d): return (~(sum(d) % 256)) & 0xFF
    def _send(self, sid, inst, params=None):
        params = params or []
        pk = [sid, len(params)+2, inst] + params
        self.serial.reset_input_buffer()
        self.serial.write(bytes([0xFF, 0xFF] + pk + [self._ck(pk)]))
        time.sleep(0.01)
    def disable(self, sid): self._send(sid, 0x03, [0x28, 0x00])
    def read_pos(self, sid):
        self._send(sid, 0x02, [0x38, 0x02])
        r = self.serial.read(8)
        if len(r) >= 8 and r[0:2] == b'\xff\xff':
            return r[5] + (r[6] << 8)
        return None
    def read_all(self):
        return [self.read_pos(i) for i in [1, 2, 3, 4, 5]]
    def disable_all(self):
        for i in [1, 2, 3, 4, 5, 6]:
            self.disable(i)
    def close(self):
        self.serial.close()


def s2r(s):
    return (np.array(s) - 2048) / 4096.0 * (240.0 * np.pi / 180.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM23')
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    ap.add_argument('--samples', type=int, default=3, help='标定采样点数')
    args = ap.parse_args()

    fk = ForwardKinematics(args.urdf)
    ctrl = Ctrl(args.port)

    print("\n" + "="*60)
    print("桌面高度标定")
    print("="*60)
    print("步骤:")
    print("  1. 扭力将关闭, 机械臂变软")
    print("  2. 你手动把把手尖端轻触桌面")
    print("  3. 按Enter, 我读取并计算桌面高度")
    print(f"  4. 重复 {args.samples} 次(不同水平位置), 取平均\n")

    # 关闭扭力, 机械臂变软
    print(">>> 正在关闭所有舵机扭力, 机械臂将变软...")
    ctrl.disable_all()
    time.sleep(0.5)
    print(">>> 扭力已关闭, 现在可以手动搬动机械臂\n")

    table_zs = []
    tip_positions = []

    try:
        for i in range(args.samples):
            print(f"--- 标定点 {i+1}/{args.samples} ---")
            print("  请把把手尖端轻触桌面(可换不同水平位置), 扶稳后按Enter...")
            input()

            servos = ctrl.read_all()
            if None in servos:
                print("  [跳过] 读取失败, 重试这一点")
                continue

            angles = s2r(servos)
            pos, _ = fk.compute(angles)
            z_mm = pos[2] * 1000
            table_zs.append(z_mm)
            tip_positions.append(pos.tolist())
            print(f"  把手尖端: X={pos[0]*1000:.0f} Y={pos[1]*1000:.0f} Z={z_mm:.0f}mm")
            print(f"  舵机值: {servos}\n")

        if not table_zs:
            print("[失败] 未采集到有效标定点")
            return

        table_zs = np.array(table_zs)
        table_z_mean = table_zs.mean()
        table_z_std = table_zs.std()
        table_z_max = table_zs.max()  # 最保守的桌面高度估计

        print("="*60)
        print("标定结果")
        print("="*60)
        print(f"  各点桌面Z(mm): {[f'{z:.0f}' for z in table_zs]}")
        print(f"  平均桌面Z: {table_z_mean:.0f}mm")
        print(f"  标准差: {table_z_std:.0f}mm")
        print(f"  最高点(最保守): {table_z_max:.0f}mm")

        # 计算推荐安全线
        # 桌面高度 + 重力下垂余量(实测约50-75mm) + 安全裕度
        GRAVITY_SAG = 75   # 重力下垂: 实机比预测低这么多
        SAFETY_MARGIN = 50 # 额外安全裕度
        recommended_floor = table_z_max + GRAVITY_SAG + SAFETY_MARGIN

        print(f"\n  重力下垂余量: +{GRAVITY_SAG}mm")
        print(f"  安全裕度: +{SAFETY_MARGIN}mm")
        print(f"  >>> 推荐采集安全线 Z_FLOOR = {recommended_floor:.0f}mm <<<")
        print(f"      (预测Z高于此值, 实机下垂后仍高于桌面)")

        # 保存
        result = {
            'port': args.port,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'table_z_samples_mm': table_zs.tolist(),
            'table_z_mean_mm': float(table_z_mean),
            'table_z_max_mm': float(table_z_max),
            'gravity_sag_mm': GRAVITY_SAG,
            'safety_margin_mm': SAFETY_MARGIN,
            'recommended_floor_mm': float(recommended_floor),
            'tip_positions': tip_positions,
        }
        with open('table_calibration.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  已保存到 table_calibration.json")

    finally:
        ctrl.close()
        print("\n[完成] 连接已关闭")


if __name__ == "__main__":
    main()
