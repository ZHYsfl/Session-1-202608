# -*- coding: utf-8 -*-
"""分析 env_server DEBUG 日志：配对场景与碰撞，复核最短射线读数是否有实物对应。"""
import math
import re
import sys

LIDAR_COUNT = 64
WALL = 1.975          # 围墙内表面
MAX_RANGE = 3.5


def ray_circle(px, py, dx, dy, cx, cy, r):
    """射线 (px,py)+t(dx,dy) 与圆 (cx,cy,r) 的最近距离：命中返回 t，否则 inf。"""
    ox, oy = cx - px, cy - py
    t = ox * dx + oy * dy
    if t < 0:
        return float("inf")
    closest = math.hypot(ox - t * dx, oy - t * dy)
    if closest >= r:
        return float("inf")
    return max(0.0, t - math.sqrt(r * r - closest * closest))


def ray_wall(px, py, dx, dy):
    best = MAX_RANGE
    if abs(dx) > 1e-9:
        for wx in (WALL, -WALL):
            t = (wx - px) / dx
            if 0.01 < t < best and abs(py + t * dy) <= WALL:
                best = t
    if abs(dy) > 1e-9:
        for wy in (WALL, -WALL):
            t = (wy - py) / dy
            if 0.01 < t < best and abs(px + t * dx) <= WALL:
                best = t
    return best


def main(path):
    # 顺序扫描：场景行更新"当前场景"，碰撞行归属最近一个场景
    # （超时局没有碰撞打印，按序号配对会错位）
    events = []  # (obstacles, x, y, yaw, k, reading, pcx, pcy)
    scene_re = re.compile(r"DEBUG 场景: .*?障碍=\[(.*?)\]")
    goal_re = re.compile(r"episode \d+ 开始 .*?目标=\(([-\d.]+),([-\d.]+)\)")
    coll_re = re.compile(
        r"DEBUG 碰撞: 位姿=\(([-\d.]+),([-\d.]+),yaw=([-\d.]+)\).*?"
        r"最短射线=k(\d+) .*?读数=([-\d.]+) 点云最近点=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
    cur_obs, cur_goal = None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        mg = goal_re.search(line)
        if mg:
            cur_goal = (float(mg.group(1)), float(mg.group(2)))
        ms = scene_re.search(line)
        if ms:
            cur_obs = [(float(a), float(b), float(c))
                       for a, b, c in re.findall(r"\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)",
                                                 ms.group(1))]
            continue
        mc = coll_re.search(line)
        if mc and cur_obs is not None and cur_goal is not None:
            gx, gy = cur_goal
            events.append((cur_obs, gx, gy, float(mc.group(1)), float(mc.group(2)),
                           float(mc.group(3)), int(mc.group(4)),
                           float(mc.group(5)), float(mc.group(6)),
                           float(mc.group(7))))
    print(f"配对碰撞 {len(events)} 条")
    n_real, n_goal, n_phantom = 0, 0, 0
    for i, (obstacles, gx, gy, x, y, yaw, k, reading, pcx, pcy) in enumerate(events):
        # 点云最近点 → 世界坐标（不经过重排，雷达原始几何）
        wx = x + pcx * math.cos(yaw) - pcy * math.sin(yaw)
        wy = y + pcx * math.sin(yaw) + pcy * math.cos(yaw)
        d_goal = math.hypot(wx - gx, wy - gy)
        if d_goal <= 0.04 + 0.05:      # 幻影点就是目标柱（半径 0.04）
            n_goal += 1
            print(f"#{i:2d} 读数={reading:.3f} 点云世界=({wx:6.3f},{wy:6.3f}) "
                  f"距目标 {d_goal:.3f} → ** 撞的是目标柱 **")
            continue
        hit_obs = any(math.hypot(wx - cx, wy - cy) <= r + 0.05
                      for cx, cy, r in obstacles)
        hit_wall = (abs(abs(wx) - WALL) < 0.08 or abs(abs(wy) - WALL) < 0.08)
        if hit_obs or hit_wall:
            n_real += 1
        else:
            n_phantom += 1
            print(f"#{i:2d} 读数={reading:.3f} 点云世界=({wx:6.3f},{wy:6.3f}) "
                  f"距目标 {d_goal:.3f} → 其他幻影")
    print(f"\n目标柱碰撞 {n_goal} / 真实碰撞 {n_real} / 其他幻影 {n_phantom}")


if __name__ == "__main__":
    main(sys.argv[1])
