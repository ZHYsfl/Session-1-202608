"""消融实验 1：深度噪声对重建的影响 —— 直接拼接 vs TSDF 融合

对 run02 的 36 帧深度图叠加不同强度的高斯噪声（模拟真实 RGBD 相机），
分别用两种方法重建，以地板 z 值 RMS（真值 z=0）为表面质量指标。

输出: D:\webots_data\recon_out\exp_noise.csv
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

NOISE_LEVELS = [0.0, 0.002, 0.005, 0.01, 0.02]   # 噪声 sigma（米）


def load_frames():
    intr = tuple(np.load(os.path.join(DATA, "intrinsics.npy")))
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "*_depth.npy"))):
        fid = os.path.basename(f)[:4]
        depth = np.load(f)
        T = np.load(os.path.join(DATA, f"{fid}_pose.npy"))
        frames.append((depth, T))
    return intr, frames


def add_noise(frames, sigma, rng):
    if sigma == 0:
        return frames
    noisy = []
    for depth, T in frames:
        d = depth.copy()
        valid = np.isfinite(d) & (d > 0)
        d[valid] += rng.normal(0, sigma, valid.sum())
        d[d < 0] = 0.0      # 负深度置无效
        noisy.append((d, T))
    return noisy


def main():
    intr, frames = load_frames()
    rng = np.random.default_rng(42)

    rows = ["noise_mm,direct_rms_mm,direct_pts,tsdf_rms_mm,tsdf_verts,tsdf_time_s"]
    for sigma in NOISE_LEVELS:
        noisy = add_noise(frames, sigma, rng)

        pts_direct = merge_direct(noisy, intr)
        rms_d, n_d = floor_rms(pts_direct)

        t0 = time.time()
        verts, mesh, _ = fuse_tsdf(noisy, intr)
        dt = time.time() - t0
        rms_t, _ = floor_rms(verts)

        rows.append(f"{sigma*1000:.0f},{rms_d*1000:.2f},{len(pts_direct)},"
                    f"{rms_t*1000:.2f},{len(verts)},{dt:.1f}")
        print(rows[-1])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "exp_noise.csv")
    with open(path, "w") as f:
        f.write("\n".join(rows))
    print("saved:", path)


if __name__ == "__main__":
    main()
