from controller import Supervisor
import numpy as np
import math
import os
from pathlib import Path

# 数据目录：项目根下的 data/，跟着项目走，拷给谁都能跑
OUT = str(Path(__file__).resolve().parents[2] / "data" / "run02")
os.makedirs(OUT, exist_ok=True)

# 场景实测参数（来自 room.wbt）：1m x 1m 桌面场地，物体在原点 +-0.5m 内、约 0.1m 高
# 采集轨迹：两个不同高度/半径的圆环，始终俯视场景中心，获得水平和垂直两个方向的视差
RINGS = [
    (0.65, 0.50, 18),   # (半径 m, 高度 m, 帧数) 高角度环
    (0.50, 0.25, 18),   # 低角度环
]
CENTER = np.array([0.0, 0.0, 0.05])   # 场景中心（z=0.05 约为物体半高）
WORLD_UP = np.array([0.0, 0.0, 1.0])  # Webots R2025a 世界坐标：z 轴朝上

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

cam = robot.getDevice("camera")
cam.enable(timestep)
rf = robot.getDevice("range-finder")
rf.enable(timestep)

# 预热：传感器 enable 后的头几步数据无效，丢弃
for _ in range(5):
    robot.step(timestep)

node = robot.getSelf()
t_field = node.getField("translation")
r_field = node.getField("rotation")

W, H = cam.getWidth(), cam.getHeight()
fov = cam.getFov()
fx = W / (2 * math.tan(fov / 2))
np.save(os.path.join(OUT, "intrinsics.npy"), np.array([fx, fx, W / 2, H / 2]))


def look_at_axis_angle(p, c):
    """计算相机从位置 p 看向目标 c 的 rotation 字段（轴角）。

    Webots R2025a 设备约定（FLU）：相机沿设备 +x 方向看，图像上方是设备 +z。
    构造旋转矩阵 R（世界 <- 设备），三列分别是设备 X/Y/Z 轴在世界中的方向，
    再转成轴角。
    """
    f = c - p
    f = f / np.linalg.norm(f)                 # 设备 X 在世界中的方向（视线）
    u = WORLD_UP - np.dot(WORLD_UP, f) * f    # 把世界上方投影到视线的垂面上
    u = u / np.linalg.norm(u)                 # 设备 Z 在世界中的方向（图像上方）
    y = np.cross(u, f)                        # 设备 Y（右手系：Y = Z x X）
    R = np.column_stack([f, y, u])

    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return [0, 0, 1, 0]
    if angle > math.pi - 1e-6:
        # 180° 旋转是轴角转换的奇点（sin(angle)=0，除法退化）。
        # 此时 R = 2·a·a^T - I，转轴 a 是 (R+I) 中模长最大的列
        M = R + np.eye(3)
        k = int(np.argmax([np.linalg.norm(M[:, j]) for j in range(3)]))
        axis = M[:, k] / np.linalg.norm(M[:, k])
        return [axis[0], axis[1], axis[2], angle]
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return [axis[0], axis[1], axis[2], angle]


i = 0
for radius, height, n in RINGS:
    for k in range(n):
        # 1. 把相机挪到圆环轨迹上的下一点
        ang = 2 * math.pi * k / n
        p = np.array([radius * math.cos(ang), radius * math.sin(ang), height])
        t_field.setSFVec3f(p.tolist())

        # 2. 转向场景中心
        r_field.setSFRotation(look_at_axis_angle(p, CENTER))

        # 3. 推进两步，确保传感器在新位姿刷新
        robot.step(timestep)
        robot.step(timestep)

        # 4. 采数据：RGB 图 + 深度数组 + 4x4 位姿矩阵
        cam.saveImage(os.path.join(OUT, f"{i:04d}_rgb.png"), 100)
        depth = np.array(rf.getRangeImage(), dtype=np.float32).reshape(H, W)
        pos = np.array(node.getPosition())
        Rw = np.array(node.getOrientation()).reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = Rw
        T[:3, 3] = pos
        np.save(os.path.join(OUT, f"{i:04d}_depth.npy"), depth)
        np.save(os.path.join(OUT, f"{i:04d}_pose.npy"), T)
        print(f"frame {i}: pos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
        i += 1

print("done.", i, "frames saved to", OUT)
