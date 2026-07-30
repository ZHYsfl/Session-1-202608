"""单帧反投影：深度图 + 位姿 -> 世界坐标点云

原理（前面讨论过的反投影公式，Webots FLU 设备约定：前=+x, 上=+z, 右=+y）：
    设备坐标: p_dev = (z, (u-cx)*z/fx, -(v-cy)*z/fy)
    世界坐标: p_world = R @ p_dev + t

用法: python backproject.py [帧号] [--radial-fix]
（注意：Webots RangeFinder planar 投影输出的是轴向深度 z，无需径向修正；
 --radial-fix 仅用于对比实验，会把地板"掰弯"，已实证）
"""
import os
import sys

import numpy as np

from pathlib import Path

DATA = str(Path(__file__).resolve().parents[1] / "data" / "run02")
OUT = str(Path(__file__).resolve().parents[1] / "data" / "recon_out")


def load_frame(frame_id):
    fx, fy, cx, cy = np.load(os.path.join(DATA, "intrinsics.npy"))
    depth = np.load(os.path.join(DATA, f"{frame_id:04d}_depth.npy"))
    T = np.load(os.path.join(DATA, f"{frame_id:04d}_pose.npy"))
    return (fx, fy, cx, cy), depth, T


def backproject(depth, intr, T, radial_fix=False):
    """深度图 -> 世界坐标点 (N,3)，返回点和有效像素坐标"""
    fx, fy, cx, cy = intr
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    d = depth[valid]

    du = (u - cx) / fx
    dv = (v - cy) / fy
    if radial_fix:
        # RangeFinder 输出的是沿射线的距离（斜边），转成沿光轴的 z
        z = d / np.sqrt(1.0 + du ** 2 + dv ** 2)
    else:
        z = d

    pts_dev = np.stack([z, -du * z, -dv * z], axis=1)  # (N,3) 设备坐标
    # 注意图像右 = fwd x up = 设备Y的反方向，所以是 -du（镜像修复）
    pts = (T[:3, :3] @ pts_dev.T).T + T[:3, 3]        # 世界坐标
    return pts, u, v


def main():
    frame_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    radial_fix = "--radial-fix" in sys.argv

    intr, depth, T = load_frame(frame_id)
    pts, u, v = backproject(depth, intr, T, radial_fix)

    print(f"frame {frame_id}: {len(pts)} points, radial_fix={radial_fix}")
    print(f"位置 t = {np.round(T[:3, 3], 3)}")

    # 地面校验：地板在世界 z=0 平面上，点云的 z 分布应该以 0 为主峰
    z = pts[:, 2]
    print(f"z 分布: 5%={np.percentile(z, 5):.3f}  50%={np.percentile(z, 50):.3f}  "
          f"95%={np.percentile(z, 95):.3f}")
    floor = np.abs(z) < 0.02
    print(f"|z|<2cm 的点占比: {floor.mean() * 100:.1f}%  (应主要是地板)")

    # 存 PLY（带颜色）
    from PIL import Image
    import open3d as o3d
    rgb = np.asarray(Image.open(os.path.join(DATA, f"{frame_id:04d}_rgb.png")),
                     dtype=np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb[v, u, :3])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"frame{frame_id:04d}.ply")
    o3d.io.write_point_cloud(out, pcd)
    print("saved:", out)


if __name__ == "__main__":
    main()
