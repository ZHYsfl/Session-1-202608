import math
import numpy as np
from PIL import Image

class OccupancyGridMap:
    def __init__(self, width_m=20, height_m=20, resolution=0.05):
        self.resolution = resolution
        self.width_pix = int(width_m / resolution)
        self.height_pix = int(height_m / resolution)
        self.origin_x = self.width_pix // 2
        self.origin_y = self.height_pix // 2
        self.log_odds = np.zeros((self.height_pix, self.width_pix), dtype=np.float32)
        self.l_free = -0.4
        self.l_occ = 0.9
        self.l_min = -4.0
        self.l_max = 4.0

    def _world_to_grid(self, x, y):
        gx = int(x / self.resolution) + self.origin_x
        gy = int(y / self.resolution) + self.origin_y
        return gx, gy

    def _bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x0 += sx
            if e2 <= dx: err += dx; y0 += sy
        return points

    def update(self, robot_x, robot_y, robot_theta, scan_ranges, fov=360, max_range=10):
        rx, ry = self._world_to_grid(robot_x, robot_y)
        n = len(scan_ranges)
        angle_step = math.radians(fov / n)
        for i, r in enumerate(scan_ranges):
            if r <= 0 or r > max_range: continue
            theta = robot_theta + i * angle_step - math.radians(fov / 2)
            ex, ey = self._world_to_grid(robot_x + r*math.cos(theta), robot_y + r*math.sin(theta))
            if not (0 <= ex < self.width_pix and 0 <= ey < self.height_pix): continue
            for cx, cy in self._bresenham(rx, ry, ex, ey):
                if 0 <= cx < self.width_pix and 0 <= cy < self.height_pix:
                    self.log_odds[cy, cx] += self.l_free
            self.log_odds[ey, ex] += self.l_occ
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def to_image(self):
        probs = 1 - 1 / (1 + np.exp(self.log_odds))
        return Image.fromarray((255 * (1 - probs)).clip(0, 255).astype(np.uint8), 'L')

    def get_occupancy_prob(self):
        return 1 - 1 / (1 + np.exp(self.log_odds))
