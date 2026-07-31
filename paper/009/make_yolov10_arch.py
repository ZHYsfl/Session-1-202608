import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

out_dir = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(out_dir, exist_ok=True)

COLORS = {
    'embed': '#F8D4D9',
    'residual': '#FFF5C8',
    'conv': '#D8EBFC',
    'attn': '#FCE8D2',
    'linear': '#E9E1F8',
    'sigmoid': '#DFF7E6',
    'frame': '#2F2F2F',
    'text': '#1F1F1F',
    'line': '#1B1B1B'
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.edgecolor': 'white'
})

def block(ax, x, y, w, h, text, color, edge=None, fontsize=8.5):
    if edge is None:
        edge = COLORS['frame']
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.03,rounding_size=0.12',
                         linewidth=1.1, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color=COLORS['text'])
    return box


def label_text(ax, x, y, text, fontsize=8.0):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=COLORS['text'])


def draw_module_stack(ax, x, y_top, w, block_h, items, title=None):
    padding = 0.12
    stack_h = len(items) * block_h + max(0, len(items)-1) * 0.08
    y_bottom = y_top - stack_h
    if title is not None:
        dashed_box(ax, x - padding, y_bottom - padding, w + padding*2, stack_h + padding*2, title)
    for idx, (label, color) in enumerate(items):
        y = y_top - (idx + 1) * block_h - idx * 0.08
        block(ax, x, y, w, block_h, label, color)
    return x + w, y_bottom + stack_h / 2


def arrow(ax, start, end, text=None, curved=False):
    style = 'arc3,rad=-0.28' if curved else 'arc3,rad=0.0'
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=style,
                                 arrowstyle='->', linewidth=1.1, color=COLORS['line']))
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.08, text, ha='center', va='center', fontsize=8, color=COLORS['text'])


def make_figure():
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Left backbone column
    x0 = 0.75
    y0 = 7.55
    block_h = 0.5
    gap = 0.16
    backbone = [
        ('Input', COLORS['embed']),
        ('Conv', COLORS['embed']),
        ('C2f', COLORS['conv']),
        ('Conv', COLORS['embed']),
        ('C2f', COLORS['conv']),
        ('SCDown', COLORS['conv']),
        ('C2f', COLORS['conv']),
        ('SCDown', COLORS['conv']),
        ('C2f', COLORS['conv']),
        ('SPPF', COLORS['conv']),
        ('PSA', COLORS['attn']),
    ]
    for idx, (label, color) in enumerate(backbone):
        y = y0 - idx * (block_h + gap)
        block(ax, x0, y, 2.2, block_h, label, color)
        if idx < len(backbone) - 1:
            arrow(ax, (x0 + 1.1, y - 0.03), (x0 + 1.1, y - block_h - gap + 0.03))

    # Branch lines to top right modules
    arrow(ax, (x0 + 2.2, y0 - 2*(block_h + gap) + block_h/2), (4.8, 6.9), curved=False)
    arrow(ax, (x0 + 2.2, y0 - 4*(block_h + gap) + block_h/2), (7.5, 6.9), curved=False)
    arrow(ax, (x0 + 2.2, y0 - 10*(block_h + gap) + block_h/2), (10.2, 6.9), curved=False)

    # Top right module stacks
    _, c2fcib_mid = draw_module_stack(ax, 4.8, 8.1, 2.0, block_h,
                                     [('Conv', COLORS['embed']), ('Split', COLORS['conv']), ('CIB', COLORS['conv']), ('Concat', COLORS['attn']), ('Conv', COLORS['embed'])],
                                     title='C2fCIB')
    _, sppf_mid = draw_module_stack(ax, 7.5, 8.1, 2.0, block_h,
                                    [('Conv', COLORS['embed']), ('MaxPool2d', COLORS['residual']), ('MaxPool2d', COLORS['residual']), ('MaxPool2d', COLORS['residual']), ('Concat', COLORS['attn']), ('Conv', COLORS['embed'])],
                                    title='SPPF')
    _, psa_mid = draw_module_stack(ax, 10.2, 8.1, 2.0, block_h,
                                   [('Split', COLORS['conv']), ('MHSA', COLORS['attn']), ('Add', COLORS['residual']), ('FFN', COLORS['conv']), ('Add', COLORS['residual']), ('Concat', COLORS['attn'])],
                                   title='PSA')
    label_text(ax, 12.5, 5.1, '×N', 12)

    arrow(ax, (5.8, 5.5), (5.8, 4.25), curved=True)
    arrow(ax, (8.5, 5.5), (8.5, 4.25), curved=True)
    arrow(ax, (11.2, 5.5), (11.2, 4.25), curved=True)

    # Right detection head path
    head_x = 5.5
    head_y = 4.6
    head_items = [
        ('Conv', COLORS['embed']),
        ('Concat', COLORS['attn']),
        ('C2f', COLORS['conv']),
        ('SCDown', COLORS['conv']),
        ('Concat', COLORS['attn']),
        ('C2fCIB', COLORS['conv']),
    ]
    for idx, (label, color) in enumerate(head_items):
        y = head_y - idx * (block_h + gap)
        block(ax, head_x, y, 2.0, block_h, label, color)
        if idx < len(head_items) - 1:
            arrow(ax, (head_x + 1.0, y - 0.03), (head_x + 1.0, y - block_h - gap + 0.03))

    # Detection output blocks
    detect_x = 9.7
    detect_y = 4.3
    outputs = ['v10Detect-P3', 'v10Detect-P4', 'v10Detect-P5']
    for idx, label in enumerate(outputs):
        y = detect_y - idx * (block_h + 0.3)
        block(ax, detect_x, y, 1.4, block_h, label, '#BEE7FF')
        arrow(ax, (head_x + 2.0, y + block_h/2), (detect_x - 0.05, y + block_h/2), curved=False)

    # Annotation labels
    label_text(ax, x0 + 1.1, y0 + 0.5, 'Backbone', 9)
    label_text(ax, 8.6, 8.7, 'Neck', 9)
    label_text(ax, 10.8, 4.8, 'Heads', 9)
    ax.text(5.5, 0.2, 'Figure 1: The YOLOv10 model architecture.', ha='center', va='center', fontsize=9.0, color=COLORS['text'])

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.08)
    out_png = os.path.join(out_dir, 'yolov10_arch.png')
    out_pdf = os.path.join(out_dir, 'yolov10_arch.pdf')
    fig.savefig(out_png, dpi=300, facecolor='white')
    fig.savefig(out_pdf, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('Saved', out_png, 'and', out_pdf)


if __name__ == '__main__':
    make_figure()
