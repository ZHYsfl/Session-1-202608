# my_room

Webots 室内场景 RGB-D 采集 + TSDF 三维重建。

## 环境

- Webots R2025a（world 里的 EXTERNPROTO 按此版本拉取）
- Python 依赖：`pip install -r requirements.txt`

## 结构

- `worlds/room.wbt` — 1m×1m 桌面场景（盒子、球、岩石）
- `controllers/collector/collector.py` — Supervisor 控制器：相机沿两个圆环轨迹采集 36 帧 RGB + 深度 + 位姿
- `recon/backproject.py` — 单帧反投影校验（深度图 + 位姿 → 点云）
- `recon/tsdf_fusion.py` — 36 帧 TSDF 融合 → 网格 + 稠密点云
- `data/` — 采集与重建产物（**不在仓库中**，运行后自动生成）

## 用法

1. Webots 打开 `worlds/room.wbt`，运行仿真，`collector` 控制器自动采集到 `data/run02/`
2. 校验单帧：`python recon/backproject.py 1`（|z|<2cm 应主要是地板）
3. 融合重建：`python recon/tsdf_fusion.py`（加 `--show` 弹窗查看网格）

产物在 `data/recon_out/`：`scene_v5mm.ply`（网格）、`scene_v5mm_pcd.ply`（稠密点云）。

## 坐标约定备忘

- Webots R2025a：世界 z 轴朝上；设备 FLU（前 +x，右 +y，上 +z）
- RangeFinder 输出轴向深度 z，无需径向修正
- 设备 → Open3D 相机的换基矩阵见 `recon/tsdf_fusion.py` 中的 `M`
