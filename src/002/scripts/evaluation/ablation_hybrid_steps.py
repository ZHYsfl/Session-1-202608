"""Ablation: hybrid IK accuracy vs. number of DLS refinement steps.

Reuses the same seeded target set and far-initial-guess protocol as
evaluate_four_methods.py so the numbers are directly comparable. Reports
mean/median position error and solve time for n_steps in {0,1,2,3,5,10}.
n_steps=0 is the raw neural warm-start.

Run:  PYTHONUTF8=1 python ablation_hybrid_steps.py
Writes: ablation_hybrid_steps.json
"""
import sys
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/002

import json
import time

import numpy as np

from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.numerical_ik import NumericalIK
from evaluate_four_methods import NeuralIK, load_cfg, s2r


def main():
    fk = ForwardKinematics('models/so101_new_calib.urdf')
    lo, hi, _ = load_cfg()
    numerical = NumericalIK(fk, (lo, hi))
    neural = NeuralIK('models/ik_mlp_best.pth', (lo, hi))

    safe = {1: [-45, 45], 2: [-10, 45], 3: [-40, 40], 4: [-40, 40], 5: [-80, 80]}

    def rand_safe():
        while True:
            s = [np.random.randint(int(safe[j][0] / 240. * 4096 + 2048) + 50,
                                   int(safe[j][1] / 240. * 4096 + 2048) - 50)
                 for j in [1, 2, 3, 4, 5]]
            p, _ = fk.compute(s2r(s))
            if p[2] >= 0.192 and np.sqrt(p[0]**2 + p[1]**2) >= 0.10 and np.linalg.norm(p) >= 0.20:
                return s

    np.random.seed(2026)  # same seed as evaluate_four_methods
    targets = []
    for _ in range(50):
        th = s2r(rand_safe())
        pos, alpha, psi = fk.compute_task_space(th)
        targets.append((pos, alpha, psi, th))

    step_grid = [0, 1, 2, 3, 5, 10]
    out = {}
    print(f"\n{'n_steps':<10}{'mean_err(mm)':<16}{'median(mm)':<16}{'time(ms)':<12}")
    print("-" * 54)
    for ns in step_grid:
        errs, times = [], []
        for pos, alpha, psi, th_true in targets:
            q_init = th_true + np.random.normal(0, 0.3, 5)
            t0 = time.perf_counter()
            q0 = neural.solve(pos, alpha, psi, q_init)
            if ns == 0:
                q = np.asarray(q0)
            else:
                q, _ = numerical.refine_solution(pos, alpha, psi, np.asarray(q0), num_steps=ns)
            times.append((time.perf_counter() - t0) * 1000)
            if q is not None:
                p, _ = fk.compute(np.asarray(q))
                errs.append(np.linalg.norm(p - pos) * 1000)
        me, md, mt = float(np.mean(errs)), float(np.median(errs)), float(np.mean(times))
        out[ns] = {'mean_err_mm': me, 'median_err_mm': md, 'mean_time_ms': mt, 'n': len(errs)}
        print(f"{ns:<10}{me:<16.2f}{md:<16.2f}{mt:<12.3f}")

    with open('ablation_hybrid_steps.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nsaved: ablation_hybrid_steps.json")


if __name__ == '__main__':
    main()
