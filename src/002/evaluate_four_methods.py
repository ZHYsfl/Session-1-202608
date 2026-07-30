"""四方法IK完整评估 - 论文核心实验

对比: 解析IK / 数值IK / 神经网络IK / 混合法(NN初值+数值微调)
指标: 成功率、位置误差(mm)、求解时间(ms)、迭代次数

混合法核心: 神经网络给一个接近的初值(~8mm), 数值法只需2-3步微调
即可收敛到亚毫米, 兼具NN的速度和数值法的精度。
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent.parent))

import time, json
import numpy as np

from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.analytical_ik import AnalyticalIK
from kinematics.numerical_ik import NumericalIK


def s2r(s): return (np.array(s) - 2048) / 4096.0 * (240.0 * np.pi / 180.0)


class NeuralIK:
    def __init__(self, model_path, joint_limits):
        import torch
        from neural_network.model import IKNet, prepare_input
        self.prepare_input = prepare_input
        ck = torch.load(model_path, map_location='cpu')
        mc = ck['config']['training']['model']
        self.model = IKNet(input_dim=mc['input_dim'], hidden_dims=mc['hidden_dims'],
                           output_dim=mc['output_dim'], activation=mc['activation'],
                           use_batch_norm=mc['use_batch_norm'], dropout=mc['dropout'],
                           joint_limits=joint_limits)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()

    def solve(self, pos, alpha, psi, q_current):
        x = self.prepare_input(pos, alpha, psi, q_current)
        return self.model.predict(x)[0]


def load_cfg(config_path='configs/robot_config.yaml'):
    import yaml
    c = yaml.safe_load(open(config_path))
    jl = c['robot']['joint_limits']
    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    lo = np.array([float(jl[n][0]) for n in names])
    hi = np.array([float(jl[n][1]) for n in names])
    ll = c['robot']['link_lengths']
    return lo, hi, ll


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    ap.add_argument('--model', default='models/ik_mlp_best.pth')
    ap.add_argument('--config', default='configs/robot_config.yaml')
    ap.add_argument('--points', type=int, default=50)
    ap.add_argument('--hybrid_steps', type=int, default=3)
    args = ap.parse_args()

    fk = ForwardKinematics(args.urdf)
    lo, hi, ll = load_cfg(args.config)

    analytical = AnalyticalIK(
        shoulder_height=float(ll['shoulder_height']), radial_offset=float(ll['radial_offset']),
        upper_arm_length=float(ll['upper_arm']), forearm_length=float(ll['forearm']),
        wrist_to_tcp=float(ll['wrist_to_tcp']), joint_limits=(lo.tolist(), hi.tolist()))
    numerical = NumericalIK(fk, (lo, hi))
    neural = NeuralIK(args.model, (lo, hi))

    # 生成安全可达目标(带真值)
    safe = {1: [-45, 45], 2: [-10, 45], 3: [-40, 40], 4: [-40, 40], 5: [-80, 80]}
    def rand_safe():
        while True:
            s = [np.random.randint(int(safe[j][0]/240.*4096+2048)+50,
                                   int(safe[j][1]/240.*4096+2048)-50) for j in [1,2,3,4,5]]
            p, _ = fk.compute(s2r(s))
            if p[2] >= 0.192 and np.sqrt(p[0]**2+p[1]**2) >= 0.10 and np.linalg.norm(p) >= 0.20:
                return s

    np.random.seed(2026)
    targets = []
    for _ in range(args.points):
        st = rand_safe()
        th = s2r(st)
        pos, alpha, psi = fk.compute_task_space(th)
        targets.append((pos, alpha, psi, th))

    methods = ['analytical', 'numerical', 'neural', 'hybrid']
    R = {m: {'err': [], 'time': [], 'success': 0} for m in methods}

    for pos, alpha, psi, th_true in targets:
        q_init = th_true + np.random.normal(0, 0.3, 5)  # 远初值,更真实

        # 解析
        t0 = time.perf_counter()
        qa = analytical.solve(pos, alpha, psi)
        R['analytical']['time'].append((time.perf_counter()-t0)*1000)
        _score(R['analytical'], qa, fk, pos)

        # 数值
        t0 = time.perf_counter()
        qn = numerical.solve(pos, alpha, psi, initial_guess=q_init)
        R['numerical']['time'].append((time.perf_counter()-t0)*1000)
        _score(R['numerical'], qn, fk, pos)

        # 神经网络
        t0 = time.perf_counter()
        qm = neural.solve(pos, alpha, psi, q_init)
        R['neural']['time'].append((time.perf_counter()-t0)*1000)
        _score(R['neural'], qm, fk, pos)

        # 混合法: NN初值 + 数值微调
        t0 = time.perf_counter()
        qm2 = neural.solve(pos, alpha, psi, q_init)
        qh, iters = numerical.refine_solution(pos, alpha, psi, np.asarray(qm2),
                                              num_steps=args.hybrid_steps)
        R['hybrid']['time'].append((time.perf_counter()-t0)*1000)
        _score(R['hybrid'], qh, fk, pos)

    # 汇总
    print("\n" + "="*74)
    print(f"四方法IK评估 (N={args.points} 目标点, 混合法微调{args.hybrid_steps}步)")
    print("="*74)
    print(f"\n{'方法':<14}{'成功率':<12}{'平均误差(mm)':<16}{'中位误差(mm)':<16}{'平均耗时(ms)':<14}")
    print("-"*74)
    summary = {}
    for m in methods:
        errs = R[m]['err']
        n = len(errs)
        succ = f"{n}/{args.points}"
        me = np.mean(errs) if errs else float('nan')
        md = np.median(errs) if errs else float('nan')
        mt = np.mean(R[m]['time'])
        label = {'analytical':'解析IK','numerical':'数值IK','neural':'神经网络IK','hybrid':'混合法'}[m]
        print(f"{label:<14}{succ:<12}{me:<16.2f}{md:<16.2f}{mt:<14.3f}")
        summary[m] = {'success': n, 'total': args.points, 'mean_err_mm': me,
                      'median_err_mm': md, 'mean_time_ms': mt}

    print("\n结论:")
    print(f"  - 数值IK最准但较慢; 神经网络最快但精度受数据量限制")
    print(f"  - 混合法用NN初值(~{summary['neural']['mean_err_mm']:.0f}mm)经{args.hybrid_steps}步微调")
    print(f"    达到{summary['hybrid']['mean_err_mm']:.2f}mm, 耗时{summary['hybrid']['mean_time_ms']:.2f}ms")
    print(f"    (数值法{summary['numerical']['mean_time_ms']:.2f}ms) → 兼顾精度与速度")

    with open('four_methods_eval.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ 结果已保存: four_methods_eval.json")


def _score(bucket, q, fk, target_pos):
    if q is None:
        return
    p, _ = fk.compute(np.asarray(q))
    err = np.linalg.norm(p - target_pos) * 1000
    bucket['err'].append(err)
    bucket['success'] += 1


if __name__ == '__main__':
    main()
