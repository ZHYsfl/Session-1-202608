"""静态安全高度验证 - 在正式采集前确认把手不再砸桌

只移动到几个"最低边界"的安全配置, 低速, 每点长停顿, 供目视确认。
不采集数据, 纯粹验证把手离桌面有足够余量。
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/002

import serial, time
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
    def enable(self, sid): self._send(sid, 0x03, [0x28, 0x01])
    def disable(self, sid): self._send(sid, 0x03, [0x28, 0x00])
    def read_pos(self, sid):
        self._send(sid, 0x02, [0x38, 0x02])
        r = self.serial.read(8)
        if len(r) >= 8 and r[0:2] == b'\xff\xff':
            return r[5] + (r[6] << 8)
        return None
    def write_pos(self, sid, pos, speed=60):
        pos = int(pos)
        self._send(sid, 0x03, [0x2A, pos & 0xFF, (pos >> 8) & 0xFF,
                               speed & 0xFF, (speed >> 8) & 0xFF])
    def read_all(self):
        return [self.read_pos(i) for i in [1, 2, 3, 4, 5]]
    def move_all(self, servos, speed=60, wait=3.0):
        for i in [1, 2, 3, 4, 5]:
            self.enable(i)
        time.sleep(0.2)
        for sid, p in enumerate(servos, 1):
            self.write_pos(sid, p, speed)
        time.sleep(wait)
    def close(self):
        for i in [1, 2, 3, 4, 5]:
            self.disable(i)
        self.serial.close()


def s2r(s):
    return (np.array(s) - 2048) / 4096.0 * (240.0 * np.pi / 180.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM24')
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    args = ap.parse_args()

    fk = ForwardKinematics(args.urdf)

    limits = {1: [-45, 45], 2: [-10, 45], 3: [-40, 40], 4: [-40, 40], 5: [-80, 80]}
    Z_FLOOR, RHO_MIN, R_MIN = 0.192, 0.10, 0.20  # 标定后的安全线(桌面66mm+下垂+裕度)

    def rand():
        s = []
        for j in [1, 2, 3, 4, 5]:
            lo, hi = limits[j]
            smin = int(lo/240.*4096+2048); smax = int(hi/240.*4096+2048)
            m = int((smax-smin)*0.1)
            s.append(np.random.randint(smin+m, smax-m))
        return s

    def safe(s):
        p, _ = fk.compute(s2r(s)); x, y, z = p
        return z >= Z_FLOOR and np.sqrt(x*x+y*y) >= RHO_MIN and np.linalg.norm(p) >= R_MIN

    # 找出几个把手尖端"最低"的安全配置(最接近地板线, 最危险的边界情况)
    np.random.seed(42)
    cands = []
    for _ in range(3000):
        s = rand()
        if safe(s):
            z = fk.compute(s2r(s))[0][2]
            cands.append((z, s))
    cands.sort()  # 按Z升序, 取最低的几个
    test_points = [c[1] for c in cands[:5]]

    HOME = [2048, 2048, 2048, 2048, 2048]

    print("\n" + "="*60)
    print("静态安全高度验证 - 测试5个最低边界配置")
    print("="*60)
    print("目的: 目视确认把手离桌面有余量, 不再砸桌")
    print("速度: 60 (很慢), 每点停顿4秒供观察\n")

    ctrl = Ctrl(args.port)
    try:
        print("先回HOME...")
        ctrl.move_all(HOME, speed=50, wait=3.5)
        p, _ = fk.compute(s2r(HOME))
        print(f"  HOME把手尖端预测Z={p[2]*1000:.0f}mm\n")

        for idx, tp in enumerate(test_points, 1):
            p, _ = fk.compute(s2r(tp))
            print(f"[{idx}/5] 目标: 把手尖端预测 X={p[0]*1000:.0f} Y={p[1]*1000:.0f} "
                  f"Z={p[2]*1000:.0f}mm (最低边界配置)")
            # 先经HOME中转
            ctrl.move_all(HOME, speed=60, wait=2.0)
            ctrl.move_all(tp, speed=60, wait=4.0)
            actual = ctrl.read_all()
            if None not in actual:
                pa, _ = fk.compute(s2r(actual))
                print(f"      实际到达: 把手尖端Z={pa[2]*1000:.0f}mm")
            print(f"      >>> 请目视检查把手离桌面距离 <<<\n")
            time.sleep(1.0)

        print("回HOME...")
        ctrl.move_all(HOME, speed=50, wait=3.5)
        print("\n[完成] 若全程把手未碰桌, 说明安全边界有效, 可正式采集")

    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
