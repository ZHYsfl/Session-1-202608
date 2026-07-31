"""Generate all figures for the SO-101 IK paper from real experiment data.

Reads result JSON / HDF5 produced by the experiments in src/002 and renders
publication-quality PDF figures into paper/002/figures/.

Run:  python gen_figs.py
"""
import json
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "002"))
OUT = os.path.dirname(__file__)


def load_json(name):
    # some result files contain GBK-encoded note fields; be tolerant.
    with open(os.path.join(SRC, name), "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return json.loads(raw.decode("utf-8", errors="replace"))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", name)


def fig_training_curve():
    """Fig: NN training/validation loss curves + LR schedule."""
    hist = load_json("models/training_history.json")
    tr, va, lr = hist["train_loss"], hist["val_loss"], hist["learning_rate"]
    ep = np.arange(1, len(tr) + 1)
    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.semilogy(ep, tr, label="Train loss", color="#1f77b4", lw=1.5)
    ax1.semilogy(ep, va, label="Val loss", color="#d62728", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)")
    ax1.legend(loc="upper right")
    best = int(np.argmin(va))
    ax1.scatter([best + 1], [va[best]], color="#d62728", zorder=5, s=30)
    ax1.annotate(f"best val={va[best]:.2e}\n@epoch {best + 1}",
                 (best + 1, va[best]), textcoords="offset points",
                 xytext=(-10, 25), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    save(fig, "training_curve.pdf")


def fig_software_vs_real():
    """Fig: per-method software error vs real-robot open-loop error (grouped bars)."""
    bench = load_json("benchmark_real_robot.json")["summary"]
    methods = ["numerical", "neural", "hybrid"]
    labels = ["Numerical\n(DLS)", "Neural\n(MLP)", "Hybrid"]
    sw = [bench[m]["sw_mean"] for m in methods]
    real = [bench[m]["open_mean"] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    b1 = ax.bar(x - w / 2, sw, w, label="Software error", color="#4c72b0")
    b2 = ax.bar(x + w / 2, real, w, label="Real-robot open-loop", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Position error (mm)")
    ax.legend()
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=8)
    ax.axhspan(30, 40, color="gray", alpha=0.12)
    ax.text(2.35, 35, "~34 mm\nmechanical\nfloor", fontsize=7.5,
            ha="center", va="center", color="dimgray")
    save(fig, "software_vs_real.pdf")


def fig_closed_loop():
    """Fig: joint-space closed-loop error convergence per iteration."""
    cl = load_json("closed_loop_joint_results.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for r in cl["results"]:
        seq = r["pos_err_seq"]
        ax.plot(range(len(seq)), seq, marker="o", ms=4,
                label=f"Point {r['point']}")
    ax.set_xlabel("Closed-loop iteration")
    ax.set_ylabel("Position error (mm)")
    ax.legend(ncol=2, fontsize=8)
    ax.axhline(cl["closed_mean"], ls="--", color="k", lw=1)
    ax.text(0.5, cl["closed_mean"] + 1.5,
            f"mean final = {cl['closed_mean']:.1f} mm", fontsize=8)
    ax.text(3.0, cl["open_mean"], f"open-loop mean = {cl['open_mean']:.1f} mm",
            fontsize=8, color="#c44e52", ha="right")
    save(fig, "closed_loop.pdf")


def fig_workspace():
    """Fig: real collected dataset workspace (top view + side view)."""
    with h5py.File(os.path.join(SRC, "real_robot_dataset.h5"), "r") as h:
        p = h["end_positions"][:] * 1000.0  # mm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a1.scatter(p[:, 0], p[:, 1], s=6, alpha=0.4, color="#4c72b0")
    a1.set_xlabel("x (mm)")
    a1.set_ylabel("y (mm)")
    a1.set_title("Top view (x-y)")
    a1.set_aspect("equal", "box")
    a2.scatter(np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2), p[:, 2], s=6,
               alpha=0.4, color="#55a868")
    a2.set_xlabel(r"radius $\rho=\sqrt{x^2+y^2}$ (mm)")
    a2.set_ylabel("z (mm)")
    a2.set_title("Side view")
    fig.suptitle(f"Real collected workspace (N={len(p)} samples)", fontsize=10)
    save(fig, "workspace.pdf")


def fig_error_decomposition():
    """Fig: software vs real errors on log scale showing mechanical floor."""
    ev = load_json("four_methods_eval.json")
    methods = ["analytical", "numerical", "neural", "hybrid"]
    labels = ["Analytical", "Numerical", "Neural", "Hybrid"]
    sw = [ev[m]["mean_err_mm"] for m in methods]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    colors = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"]
    bars = ax.bar(labels, sw, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Software position error (mm, log)")
    for b, v in zip(bars, sw):
        ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=8)
    save(fig, "error_decomposition.pdf")


def fig_hybrid_ablation():
    """Fig: hybrid position error and time vs. number of DLS refinement steps."""
    ab = load_json("ablation_hybrid_steps.json")
    steps = sorted(int(k) for k in ab.keys())
    mean = [ab[str(s)]["mean_err_mm"] for s in steps]
    med = [ab[str(s)]["median_err_mm"] for s in steps]
    tms = [ab[str(s)]["mean_time_ms"] for s in steps]
    fig, ax1 = plt.subplots(figsize=(5.4, 3.5))
    ax1.plot(steps, mean, "o-", color="#4c72b0", label="Mean error")
    ax1.plot(steps, med, "s--", color="#55a868", label="Median error")
    ax1.set_xlabel("DLS refinement steps $n$")
    ax1.set_ylabel("Position error (mm)")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(steps, tms, "^:", color="#c44e52", label="Solve time")
    ax2.set_ylabel("Solve time (ms)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.grid(False)
    ax1.annotate("median reaches\n0.9 mm at n=10",
                 (10, med[-1]), textcoords="offset points", xytext=(-70, 20),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
    save(fig, "hybrid_ablation.pdf")


def main():
    fig_training_curve()
    fig_software_vs_real()
    fig_closed_loop()
    fig_workspace()
    fig_error_decomposition()
    fig_hybrid_ablation()


if __name__ == "__main__":
    main()
