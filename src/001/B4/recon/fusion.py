"""可复用的融合模块：直接拼接(baseline) 与 TSDF 融合

坐标约定：
    设备 FLU（前+x, 右+y, 上+z），图像右 = fwd x up = -dev_y
    Open3D 相机（右+x, 下+y, 前+z），换基矩阵 M 见下
"""
import numpy as np

# 设备(FLU) -> Open3D相机
M = np.array([[0, -1, 0],
              [0, 0, -1],
              [1, 0, 0]], dtype=np.float64)


def backproject_pts(depth, intr, T):
    """单帧深度图 -> 世界坐标点 (N,3)。depth: 轴向深度(米), T: 世界<-设备 4x4"""
    fx, fy, cx, cy = intr
    valid = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[valid]
    du = (u - cx) / fx
    dv = (v - cy) / fy
    pts_dev = np.stack([z, -du * z, -dv * z], axis=1)
    return (T[:3, :3] @ pts_dev.T).T + T[:3, 3]


def merge_direct(frames, intr, voxel=0.005):
    """Baseline：所有帧反投影后直接拼接，再做体素下采样去重。返回点 (N,3)"""
    import open3d as o3d
    all_pts = []
    for depth, T in frames:
        all_pts.append(backproject_pts(depth, intr, T))
    pts = np.concatenate(all_pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points)


def fuse_tsdf(frames, intr, voxel=0.005, trunc=0.02):
    """TSDF 融合 -> (网格顶点 (V,3), 网格, 点云)"""
    import open3d as o3d
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        640, 480, intr[0], intr[1], intr[2], intr[3])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    gray = np.zeros((480, 640, 3), dtype=np.uint8)  # 融合几何不需要颜色
    for depth, T in frames:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(gray.copy()),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = M @ T[:3, :3].T
        extrinsic[:3, 3] = -M @ T[:3, :3].T @ T[:3, 3]
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    return np.asarray(mesh.vertices), mesh, volume.extract_point_cloud()


def floor_rms(pts, half=0.45, zmax=0.05):
    """地板平整度指标：场地内部(|x|,|y|<half)且 z<zmax 的点，
    取中位数附近 +-2cm 的内点（排除箱子侧面等物体点），RMS 越小越好。"""
    sel = (np.abs(pts[:, 0]) < half) & (np.abs(pts[:, 1]) < half) \
          & (np.abs(pts[:, 2]) < zmax)
    z = pts[sel, 2]
    if len(z) == 0:
        return float("nan"), 0
    z0 = np.median(z)
    inl = np.abs(z - z0) < 0.02
    z = z[inl]
    return float(np.sqrt(((z - z0) ** 2).mean())), int(inl.sum())
