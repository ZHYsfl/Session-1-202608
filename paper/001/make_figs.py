"""从消融实验 CSV 生成论文图表（输出到 paper/draft/figures/）"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/mnt/d/webots_projects/my_room/data/recon_out"
FIG = "/home/zane/session_1/paper/draft/figures"
os.makedirs(FIG, exist_ok=True)


def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        rows = list(csv.DictReader(f))
    return rows


# 图1：深度噪声 + 位姿噪声（左右子图）
noise = read_csv("exp_noise.csv")
pose = read_csv("exp_pose_noise.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

x = [float(r["noise_mm"]) for r in noise]
axes[0].plot(x, [float(r["direct_rms_mm"]) for r in noise], "o-", label="Direct merge")
axes[0].plot(x, [float(r["tsdf_rms_mm"]) for r in noise], "s-", label="TSDF fusion")
axes[0].set_xlabel(r"Depth noise $\sigma$ (mm)")
axes[0].set_ylabel("Floor RMS residual (mm)")
axes[0].set_title("(a) Depth noise")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

x = [float(r["pose_noise_mm"]) for r in pose]
axes[1].plot(x, [float(r["direct_rms_mm"]) for r in pose], "o-", label="Direct merge")
axes[1].plot(x, [float(r["tsdf_rms_mm"]) for r in pose], "s-", label="TSDF fusion")
axes[1].set_xlabel(r"Pose noise $\sigma_t$ (mm)")
axes[1].set_ylabel("Floor RMS residual (mm)")
axes[1].set_title("(b) Pose noise")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_noise.pdf"))
fig.savefig(os.path.join(FIG, "fig_noise.png"), dpi=200)
print("fig_noise saved")

# 图2：体素尺寸（顶点数+耗时 vs RMS）
vox = read_csv("exp_voxel.csv")
x = [float(r["voxel_mm"]) for r in vox]
fig, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.plot(x, [float(r["rms_mm"]) for r in vox], "o-", color="tab:blue", label="Floor RMS")
ax1.set_xlabel("Voxel size (mm)")
ax1.set_ylabel("Floor RMS residual (mm)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(x, [float(r["verts"]) for r in vox], "s--", color="tab:red", label="Vertices")
ax2.set_ylabel("Mesh vertices", color="tab:red")
ax2.set_yscale("log")
ax2.tick_params(axis="y", labelcolor="tab:red")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_voxel.pdf"))
fig.savefig(os.path.join(FIG, "fig_voxel.png"), dpi=200)
print("fig_voxel saved")
