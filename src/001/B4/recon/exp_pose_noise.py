"""消融实验 2：位姿噪声对重建的影响 —— 直接拼接 vs TSDF 融合

模拟真实 SLAM 位姿误差：给每帧位姿的平移加高斯噪声、绕随机轴加小角度旋转，
观察两种方法在不同位姿误差下的地板 RMS。

输出: D:\webots_projects\my_room\data\recon_out\exp_pose_noise.csv
"""
import os
import sys
import glob
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion import merge_direct, fuse_tsdf, floor_rms

DATA = r"D:\webots_projects\my_room\data\run02"
OUT = r"D:\webots_projects\my_room\data\recon_out"

POSE_SIGMAS = [0.0, 0.002, 0.005, 0.01, 0.02]   # 平移噪声 sigma（米）


def perturb_pose(T, sigma_t, rng):
    """平移加噪 + 随机轴小角度旋转（角度标准差与平移同量级弧度/10）"""
    T2 = T.copy()
    T2[:3, 3] += rng.normal(0, sigma_t, 3)
    ang = rng.normal(0, sigma_t / 10)
    if abs(ang) > 1e-9:
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        T2[:3, :3] = dR @ T2[:3, :3]
    return T2


def main():
    intr = tuple(np.load(os.path.join(DATA, "intrinsics.npy")))
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "*_depth.npy"))):
        fid = os.path.basename(f)[:4]
        frames.append((np.load(f), np.load(os.path.join(DATA, f"{fid}_pose.npy"))))
    rng = np.random.default_rng(7)

    rows = ["pose_noise_mm,direct_rms_mm,tsdf_rms_mm,tsdf_verts"]
    for sigma in POSE_SIGMAS:
        perturbed = [(d, perturb_pose(T, sigma, rng)) for d, T in frames]

        pts_direct = merge_direct(perturbed, intr)
        rms_d, _ = floor_rms(pts_direct)

        verts, _, _ = fuse_tsdf(perturbed, intr)
        rms_t, _ = floor_rms(verts)

        rows.append(f"{sigma*1000:.0f},{rms_d*1000:.2f},{rms_t*1000:.2f},{len(verts)}")
        print(rows[-1])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "exp_pose_noise.csv")
    with open(path, "w") as f:
        f.write("\n".join(rows))
    print("saved:", path)


if __name__ == "__main__":
    main()
