"""Plot TwoRoom training losses for the dissertation annex.

Reads the queue logs directly -- the Trainer's stdout is the only loss record this
branch keeps -- and writes vector PDFs next to the .tex source.

    python scripts/plot_tworoom_losses.py

Two figures:
  tworoom_loss_curves.pdf  every stabilized run, log-y, for the plateau argument
  tworoom_stability.pdf    the same configs before and after clipping + QK-norm

Re-run it as further runs land; it simply skips logs that do not exist yet.
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
## the .tex source lives in a separate checkout; override with --outdir elsewhere
OUT = Path(os.environ.get("TWOROOM_FIGURE_DIR", ROOT / "figures"))

## (log stem, legend label, colour) -- ordered high level then low level, K ascending
RUNS = [
    ("tworoom_hl_k2", r"high level, $K{=}2$ ($H/K{=}40$)", "#1b4f72"),
    ("tworoom_hl_k4", r"high level, $K{=}4$ ($H/K{=}20$)", "#2e86c1"),
    ("tworoom_hl_k8", r"high level, $K{=}8$ ($H/K{=}10$)", "#85c1e9"),
    ("tworoom_ll_k2", r"low level, $K{=}2$ ($H{=}3$)", "#7d3c98"),
    ("tworoom_ll_k4", r"low level, $K{=}4$ ($H{=}5$)", "#af7ac5"),
    ("tworoom_ll_k8", r"low level, $K{=}8$ ($H{=}9$)", "#d7bde2"),
    ("tworoom_flat", r"flat, $K{=}1$ ($H{=}80$)", "#b03a2e"),
]

LINE = re.compile(r"^(\d+):\s+([0-9.eE+-]+)\s*\|")


def read_log(path):
    """-> (steps, losses); empty arrays if the log is missing or has no loss lines."""
    if not path.exists():
        return np.array([]), np.array([])
    steps, losses = [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = LINE.match(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def smooth(y, w=5):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def plot_curves(logdir, out):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    any_data = False
    for stem, label, colour in RUNS:
        steps, losses = read_log(logdir / f"{stem}.log")
        if len(steps) == 0:
            continue
        any_data = True
        ax.plot(steps, losses, color=colour, alpha=0.22, lw=0.8)
        k = len(steps) - len(smooth(losses))
        ax.plot(steps[k:], smooth(losses), color=colour, lw=1.6, label=label)
    if not any_data:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_stability(before_dir, after_dir, out):
    """Side-by-side: the same configs, unstabilized and stabilized."""
    ## a log can exist but hold no loss lines -- a run that died at startup writes only
    ## a traceback, and listing it would put a label in the legend with no curve.
    stems = [s for s, _, _ in RUNS
             if len(read_log(before_dir / f"{s}.log")[0])
             and len(read_log(after_dir / f"{s}.log")[0])]
    if not stems:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (d, title) in zip(axes, [(before_dir, "no clipping, no QK-norm"),
                                     (after_dir, "clipping + QK-norm + dropout")]):
        for stem, label, colour in RUNS:
            if stem not in stems:
                continue
            steps, losses = read_log(d / f"{stem}.log")
            ax.plot(steps, losses, color=colour, lw=0.9, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel("training loss")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default=str(ROOT / "logs/queue"))
    p.add_argument("--before", default=str(ROOT / "logs/queue_precollapse"))
    p.add_argument("--outdir", default=str(OUT))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ok, name in [
        (plot_curves(Path(args.logdir), outdir / "tworoom_loss_curves.pdf"),
         "tworoom_loss_curves.pdf"),
        (plot_stability(Path(args.before), Path(args.logdir),
                        outdir / "tworoom_stability.pdf"), "tworoom_stability.pdf"),
    ]:
        print(f"{'wrote' if ok else 'skipped (no data)':>16}  {name}")


if __name__ == "__main__":
    main()
