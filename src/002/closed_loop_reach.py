"""闭环控制到达 - 真正降低真机误差

诊断已证明: 真机34mm误差主体是机械误差(重力下垂/间隙),非IK误差。
开环控制下, 命令关节角θ_cmd, 实际到达θ_act≠θ_cmd(尤其肩部下垂)。

闭环方案(关节空间反馈):
  1. 目标位姿 → IK → θ_target
  2. 命令 θ_cmd = θ_target, 移动, 读实际 θ_act
  3. 关节误差 Δθ = θ_target - θ_act
  4. 过冲补偿: θ_cmd ← θ_cmd + k·Δθ, 再移动
  5. 迭代N次, 逐步逼近

不需要外部测量, 用关节编码器反馈即可补偿稳态误差。
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial, time, json
import numpy as np
from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.numerical_ik import NumericalIK


class Ctrl:
    def __init__(self, port, baud=1000000):
        self.serial = serial.Serial(port, baud, timeout=0.5); time.sleep(0.5)
        print(f"[OK] connected {port}")
    def _ck(self, d): return (~(sum(d) % 256)) & 0xFF
    def _send(self, sid, inst, p=None):
        p = p or []; pk = [sid, len(p)+2, inst] + p
        self.serial.reset_input_buffer()
        self.serial.write(bytes([0xFF, 0xFF] + pk + [self._ck(pk)])); time.sleep(0.01)
    def enable(self, sid): self._send(sid, 0x03, [0x28, 0x01])
    def disable(self, sid): self._send(sid, 0x03, [0x28, 0x00])
    def read_pos(self, sid):
        self._send(sid, 0x02, [0x38, 0x02]); r = self.serial.read(8)
        return r[5]+(r[6]<<8) if len(r) >= 8 and r[0:2] == b'\xff\xff' else None
    def write_pos(self, sid, pos, speed=80):
        pos = int(pos); self._send(sid, 0x03, [0x2A, pos&0xFF, (pos>>8)&0xFF, speed&0xFF, (speed>>8)&0xFF])
    def read_all(self): return [self.read_pos(i) for i in [1,2,3,4,5]]
    def move_all(self, servos, speed=80, wait=2.5):
        for i in [1,2,3,4,5]: self.enable(i)
        time.sleep(0.2)
        for sid, p in enumerate(servos, 1): self.write_pos(sid, p, speed)
        time.sleep(wait)
    def close(self):
        for i in [1,2,3,4,5]: self.disable(i)
        self.serial.close()


def s2r(s): return (np.array(s)-2048)/4096.0*(240.0*np.pi/180.0)
def r2s(r): return np.clip(np.array(r)/(240.0*np.pi/180.0)*4096.0+2048, 100, 3995).astype(int)
HOME = [2048]*5
Z_FLOOR = 0.192


def load_limits(cfg='configs/robot_config.yaml'):
    import yaml
    c = yaml.safe_load(open(cfg))
    jl = c['robot']['joint_limits']; names = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
    return (np.array([float(jl[n][0]) for n in names]), np.array([float(jl[n][1]) for n in names]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM23')
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    ap.add_argument('--points', type=int, default=4)
    ap.add_argument('--iters', type=int, default=3, help='闭环迭代次数')
    ap.add_argument('--gain', type=float, default=0.8, help='反馈增益k')
    args = ap.parse_args()

    fk = ForwardKinematics(args.urdf)
    lo, hi = load_limits()
    numerical = NumericalIK(fk, (lo, hi))

    # 安全可达目标
    safe = {1:[-45,45],2:[-10,45],3:[-40,40],4:[-40,40],5:[-80,80]}
    def rand_safe():
        while True:
            s = [np.random.randint(int(safe[j][0]/240.*4096+2048)+80, int(safe[j][1]/240.*4096+2048)-80) for j in [1,2,3,4,5]]
            p, _ = fk.compute(s2r(s))
            if p[2] >= Z_FLOOR and np.sqrt(p[0]**2+p[1]**2) >= 0.10 and np.linalg.norm(p) >= 0.20:
                return s
    np.random.seed(11)
    targets = []
    for _ in range(args.points):
        st = rand_safe(); th = s2r(st)
        pos, alpha, psi = fk.compute_task_space(th)
        targets.append({'pos': pos, 'alpha': alpha, 'psi': psi, 'theta': th})

    print("\n" + "="*70)
    print(f"闭环控制到达实验 (迭代{args.iters}次, 增益{args.gain})")
    print("="*70)
    print("  ⚠️ 机械臂将移动, 请确保安全")

    ctrl = Ctrl(args.port)
    open_errs, closed_errs = [], []
    try:
        ctrl.move_all(HOME, speed=60, wait=3.0)
        for i, t in enumerate(targets):
            # IK求目标关节角(数值法, 保证软件层准确)
            q_target = numerical.solve(t['pos'], t['alpha'], t['psi'],
                                       initial_guess=np.array(s2r(HOME)))
            if q_target is None:
                print(f"  点{i+1}: IK无解, 跳过"); continue
            q_target = np.asarray(q_target)
            p_pred, _ = fk.compute(q_target)
            if p_pred[2] < Z_FLOOR:
                print(f"  点{i+1}: 预测Z低于安全线, 跳过"); continue

            print(f"\n  点{i+1}: 目标({t['pos'][0]*1000:.0f},{t['pos'][1]*1000:.0f},{t['pos'][2]*1000:.0f})mm")
            ctrl.move_all(HOME, speed=80, wait=1.5)

            # ---- 开环: 直接命令一次 ----
            q_cmd = q_target.copy()
            ctrl.move_all(r2s(q_cmd), speed=80, wait=3.0)
            act = ctrl.read_all()
            if None in act: print("    读取失败"); continue
            p_act, _ = fk.compute(s2r(act))
            e_open = np.linalg.norm(p_act - t['pos'])*1000
            open_errs.append(e_open)
            print(f"    开环: {e_open:.1f}mm")

            # ---- 闭环: 关节误差反馈迭代 ----
            e_closed = e_open
            for it in range(args.iters):
                q_act = s2r(ctrl.read_all())
                dtheta = q_target - q_act          # 关节角残差
                q_cmd = q_cmd + args.gain * dtheta  # 过冲补偿
                q_cmd = np.clip(q_cmd, lo, hi)
                # 安全校验
                p_c, _ = fk.compute(q_cmd)
                if p_c[2] < Z_FLOOR - 0.02:
                    print(f"    迭代{it+1}: 补偿后Z过低, 停止")
                    break
                ctrl.move_all(r2s(q_cmd), speed=80, wait=2.0)
                act = ctrl.read_all()
                if None in act: break
                p_act, _ = fk.compute(s2r(act))
                e_closed = np.linalg.norm(p_act - t['pos'])*1000
                print(f"    闭环迭代{it+1}: {e_closed:.1f}mm")
            closed_errs.append(e_closed)
        ctrl.move_all(HOME, speed=60, wait=3.0)
    finally:
        ctrl.close()

    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    if open_errs:
        print(f"  开环平均误差: {np.mean(open_errs):.1f}mm")
    if closed_errs:
        print(f"  闭环平均误差: {np.mean(closed_errs):.1f}mm")
        if open_errs:
            imp = (1 - np.mean(closed_errs)/np.mean(open_errs))*100
            print(f"  改善: {imp:.0f}%")

    json.dump({'open_mm': open_errs, 'closed_mm': closed_errs,
               'iters': args.iters, 'gain': args.gain},
              open('closed_loop_results.json','w'), indent=2)
    print("\n✓ 结果已保存: closed_loop_results.json")


if __name__ == '__main__':
    main()
