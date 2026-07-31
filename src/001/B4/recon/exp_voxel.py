"""消融实验 3：TSDF 体素尺寸对重建质量与耗时的影响

输出: D:\webots_projects\my_room\data\recon_out\exp_voxel.csv
"""
import os
import sys
import glob
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion import fuse_tsdf, floor_rms

DATA = r"D:\webots_projects\my_room\data\run02"
OUT = r"D:\webots_projects\my_room\data\recon_out"

VOXELS = [0.002, 0.005, 0.01, 0.02]


def main():
    intr = tuple(np.load(os.path.join(DATA, "intrinsics.npy")))
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "*_depth.npy"))):
        fid = os.path.basename(f)[:4]
        frames.append((np.load(f), np.load(os.path.join(DATA, f"{fid}_pose.npy"))))

    rows = ["voxel_mm,verts,tris,rms_mm,time_s"]
    for v in VOXELS:
        t0 = time.time()
        verts, mesh, _ = fuse_tsdf(frames, intr, voxel=v, trunc=4 * v)
        dt = time.time() - t0
        rms, _ = floor_rms(verts)
        rows.append(f"{v*1000:.0f},{len(verts)},{len(mesh.triangles)},"
                    f"{rms*1000:.2f},{dt:.1f}")
        print(rows[-1])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "exp_voxel.csv")
    with open(path, "w") as f:
        f.write("\n".join(rows))
    print("saved:", path)


if __name__ == "__main__":
    main()
