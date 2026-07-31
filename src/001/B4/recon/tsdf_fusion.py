"""多帧 TSDF 融合：36 帧 RGBD -> 完整三维场景（网格 + 稠密点云）

管线（对应论文 Method 部分）：
    1. 每帧 RGB + 深度图 -> Open3D RGBDImage
    2. 位姿 T（世界<-设备）转成 Open3D 相机约定（世界->相机）的外参
    3. ScalableTSDFVolume 逐帧积分（体素内加权平均，消噪去重）
    4. marching cubes 提取三角网格

坐标约定转换（重点）：
    我们的设备是 FLU（前+x, 右+y, 上+z），Open3D 相机是（右+x, 下+y, 前+z）。
    设备坐标 p_dev 与 Open3D 相机坐标 p_cam 的关系：p_cam = M @ p_dev
    所以 Open3D 要的外参 extrinsic = inv(T @ M^{-1}) = M @ T^{-1}

用法: python tsdf_fusion.py [--voxel 0.005] [--trunc 0.02] [--show]
"""
import os
import sys
import glob

import numpy as np

from pathlib import Path

DATA = str(Path(__file__).resolve().parents[1] / "data" / "run02")
OUT = str(Path(__file__).resolve().parents[1] / "data" / "recon_out")

# 设备(FLU) -> Open3D相机(右x, 下y, 前z) 的换基矩阵
# 图像右 = fwd x up = -dev_y（不是 +dev_y，右手系推出来的，之前镜像 bug 在这）
M = np.array([[0, -1, 0],     # o3d_x(right) = -dev_y
              [0, 0, -1],     # o3d_y(down)  = -dev_z
              [1, 0, 0]],     # o3d_z(fwd)   = dev_x
             dtype=np.float64)


def main():
    voxel = 0.005
    trunc = 0.02
    show = "--show" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--voxel":
            voxel = float(sys.argv[i + 1])
        if a == "--trunc":
            trunc = float(sys.argv[i + 1])

    import open3d as o3d
    from PIL import Image

    fx, fy, cx, cy = np.load(os.path.join(DATA, "intrinsics.npy"))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    frames = sorted(glob.glob(os.path.join(DATA, "*_depth.npy")))
    print(f"fusing {len(frames)} frames, voxel={voxel}, trunc={trunc}")

    for f in frames:
        fid = os.path.basename(f)[:4]
        depth = np.load(f).astype(np.float32)          # 已是轴向深度(米)
        rgb = np.asarray(Image.open(os.path.join(DATA, f"{fid}_rgb.png")))
        T = np.load(os.path.join(DATA, f"{fid}_pose.npy"))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb[..., :3].copy()),
            o3d.geometry.Image(depth),
            depth_scale=1.0,          # 深度单位已是米
            depth_trunc=3.0,          # 超过 3m 丢弃（背景布）
            convert_rgb_to_intensity=False)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = M @ T[:3, :3].T          # M @ R^{-1}
        extrinsic[:3, 3] = -M @ T[:3, :3].T @ T[:3, 3]
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    os.makedirs(OUT, exist_ok=True)
    mesh_path = os.path.join(OUT, f"scene_v{int(voxel*1000)}mm.ply")
    pcd_path = os.path.join(OUT, f"scene_v{int(voxel*1000)}mm_pcd.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    o3d.io.write_point_cloud(pcd_path, pcd)

    vs = np.asarray(mesh.vertices)
    print(f"mesh: {len(vs)} vertices, {len(mesh.triangles)} triangles")
    print(f"pcd:  {len(pcd.points)} points")
    if len(vs):
        print(f"范围 x[{vs[:,0].min():.2f},{vs[:,0].max():.2f}] "
              f"y[{vs[:,1].min():.2f},{vs[:,1].max():.2f}] "
              f"z[{vs[:,2].min():.2f},{vs[:,2].max():.2f}]")
    print("saved:", mesh_path)
    print("saved:", pcd_path)

    if show:
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
