import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
})

out_dir = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

boxes = [
    ('Input Image\nVideo/Camera', (0.6, 4.0), 1.95, 1.0),
    ('YOLO\nDetector', (3.0, 4.0), 1.65, 1.0),
    ('Distance\nEstimation', (5.45, 4.0), 1.95, 1.0),
    ('Warning\nDisplay', (8.05, 4.0), 1.7, 1.0),
]

for label, (x, y), w, h in boxes:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.025',
        linewidth=1.2,
        edgecolor='#2C6E8F',
        facecolor='#EAF5FB',
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=8.8, color='#1F4E79', linespacing=1.1)

for i in range(len(boxes) - 1):
    x1 = boxes[i][1][0] + boxes[i][2] + 0.04
    y1 = boxes[i][1][1] + boxes[i][3] / 2
    x2 = boxes[i + 1][1][0] - 0.04
    y2 = boxes[i + 1][1][1] + boxes[i + 1][3] / 2
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=11, linewidth=1.2, color='#4C78A8'))

panel = FancyBboxPatch((2.0, 1.35), 6.0, 0.85, boxstyle='round,pad=0.02', linewidth=0.9, edgecolor='#D0D7DE', facecolor='#F8FAFC')
ax.add_patch(panel)
ax.text(5.0, 1.78, 'Detection boxes → pixel size → distance → alert color', ha='center', va='center', fontsize=9.2, color='#4A4A4A')

fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.08)
fig.savefig(os.path.join(out_dir, 'system_architecture.png'), dpi=300, facecolor='white')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.1), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Camera body
ax.add_patch(Rectangle((0.8, 2.2), 0.9, 1.3, facecolor='#DDEBF7', edgecolor='#2C6E8F', linewidth=1.2))
ax.text(1.25, 2.85, 'Camera', ha='center', va='center', fontsize=9.2, color='#1F4E79')

# Pinhole and image plane
pinhole = (2.35, 3.0)
plane_x = 3.65
ax.scatter([pinhole[0]], [pinhole[1]], s=24, color='#1F4E79')
ax.plot([plane_x, plane_x], [1.3, 4.7], color='#2C6E8F', linewidth=1.4)

# Object on the right
obj_left = 7.0
obj_bottom_y = 2.2
obj_height = 2.0
ax.add_patch(Rectangle((obj_left, obj_bottom_y), 0.75, obj_height, facecolor='#FDEBD0', edgecolor='#C97A1D', linewidth=1.2))
ax.text(obj_left + 0.375, 4.8, 'Object', ha='center', va='center', fontsize=9.0, color='#8A4B08')

# Rays from object top and bottom through the pinhole
ax.plot([obj_left, pinhole[0]], [obj_bottom_y + obj_height, pinhole[1]], color='#C97A1D', linestyle='--', linewidth=1.2)
ax.plot([obj_left, pinhole[0]], [obj_bottom_y, pinhole[1]], color='#C97A1D', linestyle='--', linewidth=1.2)

# Projected image height p_h on the image plane
ax.plot([plane_x, plane_x], [2.75, 3.25], color='#4C78A8', linewidth=2.1)

# Labels for H, f, p_h, d
ax.add_patch(FancyArrowPatch((obj_left + 0.15, obj_bottom_y), (obj_left + 0.15, obj_bottom_y + obj_height), arrowstyle='->', mutation_scale=10, linewidth=1.3, color='#4C78A8'))
ax.text(obj_left + 0.28, obj_bottom_y + obj_height / 2, 'H', ha='left', va='center', fontsize=9.2, color='#1F4E79')

ax.add_patch(FancyArrowPatch((pinhole[0] + 0.12, pinhole[1] - 0.05), (plane_x - 0.08, pinhole[1] - 0.05), arrowstyle='->', mutation_scale=10, linewidth=1.3, color='#4C78A8'))
ax.text(2.95, 2.75, 'f', ha='center', va='center', fontsize=9.2, color='#1F4E79')

ax.add_patch(FancyArrowPatch((plane_x + 0.18, 2.75), (plane_x + 0.18, 3.25), arrowstyle='->', mutation_scale=10, linewidth=1.3, color='#4C78A8'))
ax.text(plane_x + 0.32, 3.0, 'p_h', ha='left', va='center', fontsize=9.2, color='#1F4E79')

ax.add_patch(FancyArrowPatch((pinhole[0], 1.55), (obj_left - 0.1, 1.55), arrowstyle='->', mutation_scale=10, linewidth=1.3, color='#4C78A8'))
ax.text(4.8, 1.55, 'd', ha='center', va='center', fontsize=9.2, color='#1F4E79')

# Formula caption
ax.text(5.0, 0.95, '$d = Hf/p_h$', ha='center', va='center', fontsize=10.2, color='#1F4E79')
ax.text(5.0, 1.45, 'similar triangles', ha='center', va='center', fontsize=8.5, color='#4A4A4A')

fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.08)
fig.savefig(os.path.join(out_dir, 'distance_geometry.png'), dpi=300, facecolor='white')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.1), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.add_patch(Rectangle((1.0, 1.0), 8.0, 4.0, facecolor='#F7F9FB', edgecolor='#C7CFD8', linewidth=1.0))

ax.add_patch(Rectangle((1.85, 2.2), 2.15, 1.25, facecolor='none', edgecolor='#E74C3C', linewidth=2.0))
ax.text(2.95, 3.65, 'Danger\n2.3 m', fontsize=9.3, color='#E74C3C', ha='center', va='center', linespacing=1.15)

ax.add_patch(Rectangle((4.55, 2.75), 1.75, 1.05, facecolor='none', edgecolor='#F1C40F', linewidth=1.8))
ax.text(5.425, 3.8, 'Warning\n4.1 m', fontsize=9.3, color='#F1C40F', ha='center', va='center', linespacing=1.15)

ax.add_patch(Rectangle((6.95, 1.85), 1.35, 1.05, facecolor='none', edgecolor='#2ECC71', linewidth=1.8))
ax.text(7.625, 3.0, 'Safe\n6.0 m', fontsize=9.3, color='#2ECC71', ha='center', va='center', linespacing=1.15)

ax.text(5.0, 0.55, 'Color-coded alert display for different distance ranges', ha='center', fontsize=8.8, color='#4A4A4A')

fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.08)
fig.savefig(os.path.join(out_dir, 'warning_demo.png'), dpi=300, facecolor='white')
plt.close(fig)

print('Generated figures in', out_dir)
