"""IK验证实验 - 单点到达 + 三方法对比 + 重复性

证明"IK算出来的是对的"的核心实验。无需外部测量设备,
用机械臂自身关节编码器闭环验证。

三个验证:
  1. 单点到达: 真机走到某姿态得真值P_true → IK反算θ → 真机执行 → 测实际到达误差
  2. 三方法对比: 同一目标, 解析/数值/神经网络各求解, 比精度和速度
  3. 重复性: 同一目标从多个初始姿态求解, 看解的离散度

安全: 所有目标位姿都经过Z>=192mm等安全过滤, 且移动前经HOME中转。
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial, time, json
import numpy as np

from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.analytical_ik import AnalyticalIK
from kinematics.numerical_ik import NumericalIK


# ---------- 机械臂控制 ----------
class Ctrl:
    def __init__(self, port, baud=1000000):
        self.serial = serial.Serial(port, baud, timeout=0.5)
        time.sleep(0.5)
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
        if len(r) >= 8 and r[0:2] == b'\xff\xff': return r[5] + (r[6] << 8)
        return None
    def write_pos(self, sid, pos, speed=80):
        pos = int(pos)
        self._send(sid, 0x03, [0x2A, pos & 0xFF, (pos >> 8) & 0xFF, speed & 0xFF, (speed >> 8) & 0xFF])
    def read_all(self):
        return [self.read_pos(i) for i in [1, 2, 3, 4, 5]]
    def move_all(self, servos, speed=80, wait=2.5):
        for i in [1, 2, 3, 4, 5]: self.enable(i)
        time.sleep(0.2)
        for sid, p in enumerate(servos, 1): self.write_pos(sid, p, speed)
        time.sleep(wait)
    def close(self):
        for i in [1, 2, 3, 4, 5]: self.disable(i)
        self.serial.close()


def s2r(s): return (np.array(s) - 2048) / 4096.0 * (240.0 * np.pi / 180.0)
def r2s(r): return np.clip(np.array(r) / (240.0*np.pi/180.0) * 4096.0 + 2048, 100, 3995).astype(int)

HOME = [2048, 2048, 2048, 2048, 2048]
Z_FLOOR = 0.192


def load_limits(config_path='configs/robot_config.yaml'):
    import yaml
    c = yaml.safe_load(open(config_path))
    jl = c['robot']['joint_limits']
    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    lo = np.array([float(jl[n][0]) for n in names])
    hi = np.array([float(jl[n][1]) for n in names])
    ll = c['robot']['link_lengths']
    return lo, hi, ll


# ---------- 神经网络IK封装 ----------
class NeuralIK:
    def __init__(self, model_path, joint_limits):
        import torch
        from neural_network.model import IKNet, prepare_input
        self.torch = torch
        self.prepare_input = prepare_input
        ckpt = torch.load(model_path, map_location='cpu')
        mc = ckpt['config']['training']['model']
        self.model = IKNet(input_dim=mc['input_dim'], hidden_dims=mc['hidden_dims'],
                           output_dim=mc['output_dim'], activation=mc['activation'],
                           use_batch_norm=mc['use_batch_norm'], dropout=mc['dropout'],
                           joint_limits=joint_limits)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    def solve(self, pos, alpha, psi, q_current):
        x = self.prepare_input(pos, alpha, psi, q_current)
        return self.model.predict(x)[0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM23')
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    ap.add_argument('--model', default='models/ik_mlp_best.pth')
    ap.add_argument('--config', default='configs/robot_config.yaml')
    ap.add_argument('--points', type=int, default=6, help='测试目标点数')
    ap.add_argument('--dry', action='store_true', help='仅软件, 不动真机')
    args = ap.parse_args()

    fk = ForwardKinematics(args.urdf)
    lo, hi, ll = load_limits(args.config)

    # 三种IK求解器
    analytical = AnalyticalIK(
        shoulder_height=float(ll['shoulder_height']), radial_offset=float(ll['radial_offset']),
        upper_arm_length=float(ll['upper_arm']), forearm_length=float(ll['forearm']),
        wrist_to_tcp=float(ll['wrist_to_tcp']), joint_limits=(lo.tolist(), hi.tolist()))
    numerical = NumericalIK(fk, (lo, hi))
    neural = NeuralIK(args.model, (lo, hi))

    # 生成测试目标: 从安全的随机姿态用FK得到可达位姿(带真值θ_true)
    safe_limits = {1: [-45, 45], 2: [-10, 45], 3: [-40, 40], 4: [-40, 40], 5: [-80, 80]}
    def rand_safe():
        while True:
            s = []
            for j in [1, 2, 3, 4, 5]:
                a, b = safe_limits[j]
                smin = int(a/240.*4096+2048); smax = int(b/240.*4096+2048)
                m = int((smax-smin)*0.1); s.append(np.random.randint(smin+m, smax-m))
            p, _ = fk.compute(s2r(s))
            if p[2] >= Z_FLOOR and np.sqrt(p[0]**2+p[1]**2) >= 0.10 and np.linalg.norm(p) >= 0.20:
                return s

    np.random.seed(7)
    targets = []
    for _ in range(args.points):
        servo_true = rand_safe()
        theta_true = s2r(servo_true)
        pos, alpha, psi = fk.compute_task_space(theta_true)
        targets.append({'servo_true': servo_true, 'theta_true': theta_true,
                        'pos': pos, 'alpha': alpha, 'psi': psi})

    print("\n" + "="*70)
    print("IK验证实验")
    print("="*70)

    results = {'analytical': [], 'numerical': [], 'neural': []}
    timing = {'analytical': [], 'numerical': [], 'neural': []}

    # ===== 验证A: 软件层三方法对比(FK-IK往返误差 + 速度) =====
    print("\n[验证A] 三方法软件对比: FK(IK(P)) vs P 往返误差 + 求解速度")
    print("-"*70)
    for i, t in enumerate(targets):
        pos, alpha, psi, theta_true = t['pos'], t['alpha'], t['psi'], t['theta_true']
        q_init = theta_true + np.random.normal(0, 0.1, 5)  # 初值加噪

        # 解析
        t0 = time.perf_counter()
        qa = analytical.solve(pos, alpha, psi)
        timing['analytical'].append((time.perf_counter()-t0)*1000)
        # 数值
        t0 = time.perf_counter()
        qn = numerical.solve(pos, alpha, psi, initial_guess=q_init)
        timing['numerical'].append((time.perf_counter()-t0)*1000)
        # 神经网络
        t0 = time.perf_counter()
        qm = neural.solve(pos, alpha, psi, q_init)
        timing['neural'].append((time.perf_counter()-t0)*1000)

        # 各自FK往返误差
        for name, q in [('analytical', qa), ('numerical', qn), ('neural', qm)]:
            if q is None:
                results[name].append(None)
            else:
                p2, _ = fk.compute(np.asarray(q))
                err = np.linalg.norm(p2 - pos) * 1000
                results[name].append(err)
        print(f"  点{i+1}: 目标({pos[0]*1000:.0f},{pos[1]*1000:.0f},{pos[2]*1000:.0f})mm  "
              f"解析={_fmt(results['analytical'][-1])} 数值={_fmt(results['numerical'][-1])} "
              f"神经={_fmt(results['neural'][-1])}")

    print("\n  --- 汇总(FK往返误差 mm / 求解时间 ms) ---")
    summary = {}
    for name in ['analytical', 'numerical', 'neural']:
        errs = [e for e in results[name] if e is not None]
        succ = len(errs)
        mean_err = np.mean(errs) if errs else float('nan')
        mean_t = np.mean(timing[name])
        summary[name] = {'success': succ, 'total': len(targets),
                         'mean_err_mm': mean_err, 'mean_time_ms': mean_t}
        print(f"  {name:12s}: 成功{succ}/{len(targets)}  "
              f"平均误差={mean_err:.2f}mm  平均耗时={mean_t:.3f}ms")

    # ===== 验证B: 重复性(神经网络, 同目标多初值) =====
    print("\n[验证B] 重复性: 同一目标从8个不同初值求解, 看解的离散度")
    print("-"*70)
    t = targets[0]
    sols = []
    for k in range(8):
        q_init = t['theta_true'] + np.random.normal(0, 0.15, 5)
        q = neural.solve(t['pos'], t['alpha'], t['psi'], q_init)
        sols.append(q)
    sols = np.array(sols)
    std_deg = np.degrees(sols.std(axis=0))
    print(f"  目标: ({t['pos'][0]*1000:.0f},{t['pos'][1]*1000:.0f},{t['pos'][2]*1000:.0f})mm")
    print(f"  8次解的各关节标准差(度): {np.round(std_deg,3)}")
    print(f"  最大标准差: {std_deg.max():.3f}° {'(高度一致,可复现)' if std_deg.max()<2 else '(离散较大)'}")

    # ===== 验证C: 真机单点到达, 对比神经网络 vs 混合法 =====
    real_nn, real_hybrid = [], []
    if not args.dry:
        print("\n[验证C] 真机单点到达: 神经网络 vs 混合法, 测实际到达误差")
        print("-"*70)
        print("  ⚠️ 机械臂将移动, 请确保安全")
        ctrl = Ctrl(args.port)
        try:
            ctrl.move_all(HOME, speed=60, wait=3.0)
            for i, t in enumerate(targets):
                q_init = np.array(s2r(HOME))
                # 纯神经网络解
                q_nn = np.asarray(neural.solve(t['pos'], t['alpha'], t['psi'], q_init))
                # 混合法解: NN初值 + 数值微调
                q_hy, _ = numerical.refine_solution(t['pos'], t['alpha'], t['psi'],
                                                     q_nn.copy(), num_steps=10)
                q_hy = np.asarray(q_hy)

                for label, q_pred, bucket in [('NN', q_nn, real_nn),
                                              ('混合', q_hy, real_hybrid)]:
                    p_pred, _ = fk.compute(q_pred)
                    if p_pred[2] < Z_FLOOR:
                        print(f"  点{i+1}[{label}]: [跳过] 预测Z={p_pred[2]*1000:.0f}mm 低于安全线")
                        continue
                    ctrl.move_all(HOME, speed=80, wait=1.5)
                    ctrl.move_all(r2s(q_pred), speed=80, wait=3.0)
                    actual = ctrl.read_all()
                    if None in actual:
                        print(f"  点{i+1}[{label}]: 读取失败"); continue
                    p_actual, _ = fk.compute(s2r(actual))
                    err = np.linalg.norm(p_actual - t['pos']) * 1000
                    bucket.append(err)
                    print(f"  点{i+1}[{label:>2s}]: 目标({t['pos'][0]*1000:.0f},"
                          f"{t['pos'][1]*1000:.0f},{t['pos'][2]*1000:.0f}) → 误差={err:.1f}mm")
            ctrl.move_all(HOME, speed=60, wait=3.0)
        finally:
            ctrl.close()
        print()
        if real_nn:
            print(f"  神经网络 真机到达: 平均{np.mean(real_nn):.1f}mm 中位{np.median(real_nn):.1f}mm")
        if real_hybrid:
            print(f"  混合法   真机到达: 平均{np.mean(real_hybrid):.1f}mm 中位{np.median(real_hybrid):.1f}mm")

    # 保存
    out = {'summary': summary,
           'repeatability_std_deg': std_deg.tolist(),
           'real_reach_nn_mm': real_nn,
           'real_reach_hybrid_mm': real_hybrid,
           'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}
    with open('ik_validation_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ 结果已保存: ik_validation_results.json")


def _fmt(e):
    return f"{e:.2f}mm" if e is not None else "无解"


if __name__ == '__main__':
    main()
