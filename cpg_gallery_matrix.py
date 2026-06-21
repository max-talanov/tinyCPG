#!/usr/bin/env python3
"""
cpg_gallery_matrix.py
Gallery-style (converged-tail, E-solid / F-dashed) figures spanning the
full STDP-λ dimension:

  fig9  — speed × λ      force gallery   (3 speeds  × 3 λ)
  fig10 — ablation × λ   force gallery   (3 gains   × 3 λ)
  fig11 — weight profiles per λ          (CUT→RGE, BS→RGE, BS→RGF means)

All read the λ ∈ {1e-5, 1e-4, 1e-3} runs (120 s). The force galleries
show the converged tail (last `--zoom-ms`); the weight figure shows the
full-run mean-weight trajectories with the three λ overlaid.

Usage:
  python3 cpg_gallery_matrix.py --indir results/2026-06-20 --outdir plots/paper
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


LAMBDA_ORDER = ["lam1em5", "lam1em4", "lam1em3"]
_SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def _lam_label(tag: str) -> str:
    m = re.match(r"lam(\d+)em(\d+)", tag)
    if not m:
        return tag
    sup = "".join(_SUP.get(c, c) for c in f"-{m.group(2)}")
    return f"λ = {m.group(1)}·10{sup}"


# Speed conditions (Phase A) — file tag → (label, period ms)
SPEEDS = [
    ("06cms",   "slow walk\n≈6 cm/s (1200 ms)", 1200),
    ("13_5cms", "medium walk\n≈13.5 cm/s (520 ms)", 520),
    ("21cms",   "fast walk\n≈21 cm/s (350 ms)", 350),
]
# Ablation conditions (Phase B) — file tag → label
GAINS = [
    ("baseline", "baseline\nIa gain 1.0"),
    ("toe",      "toe stepping\nIa gain 0.5"),
    ("air",      "air stepping\nIa gain 0.1"),
]


def _find(indir: str, pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(indir, pattern)))
    return hits[0] if hits else None


def _force_panel(ax, path, color, zoom_ms, title=None):
    with h5py.File(path, "r") as h:
        t = np.asarray(h["times_ms"])
        fe = np.asarray(h["leg_L/force_e"]); ff = np.asarray(h["leg_L/force_f"])
        sim_ms = float(h.attrs.get("sim_ms", t.max() if t.size else 120000.0))
    z = max(0.0, sim_ms - zoom_ms)
    m = t >= z
    ax.plot(t[m], fe[m], color=color, linewidth=1.2, label="E")
    ax.plot(t[m], ff[m], color=color, linewidth=0.95, linestyle="--", alpha=0.7, label="F")
    ax.set_xlim(z, sim_ms); ax.grid(alpha=0.2)
    if title:
        ax.set_title(title, fontsize=9)


def _force_matrix(indir, rows, file_fn, row_colors, outpath, suptitle, zoom_ms):
    """rows: list of (tag, label); file_fn(tag, lam) -> path; 3 λ columns."""
    nr, nc = len(rows), len(LAMBDA_ORDER)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.1 * nr), squeeze=False)
    for r, (tag, lbl) in enumerate(rows):
        for c, lam in enumerate(LAMBDA_ORDER):
            ax = axes[r][c]
            f = file_fn(tag, lam)
            if f is None:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_xticks([]); ax.set_yticks([]); continue
            _force_panel(ax, f, row_colors[r], zoom_ms,
                         title=(_lam_label(lam) if r == 0 else None))
            if c == 0:
                ax.set_ylabel(f"{lbl}\nForce E / F (a.u.)", fontsize=9)
                ax.legend(loc="upper right", ncol=2, fontsize=7)
            if r == nr - 1:
                ax.set_xlabel("time (ms)")
    fig.suptitle(suptitle, fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[gallery] saved {outpath}")


def _weight_profiles(indir, anchor_speed, anchor_gain, outpath):
    """CUT->RGE, BS->RGE, BS->RGF mean-weight trajectories, 3 λ overlaid,
    for a speed anchor (top row) and an ablation anchor (bottom row)."""
    palette = {"lam1em5": "#1f77b4", "lam1em4": "#2ca02c", "lam1em3": "#d62728"}
    proj05 = [("cut->rge_mean", "CUT→RG-E"), ("bs->rge_mean", "BS→RG-E"),
              ("bs->rgf_mean", "BS→RG-F")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    rows = [
        ("medium walk (Ia intact)",
         lambda lam: _find(indir, f"cpg_speed_stdp_{anchor_speed}_{lam}_*.h5")),
        ("air stepping (Ia gain 0.1)",
         lambda lam: _find(indir, f"cpg_ablgrad_{anchor_gain}_{lam}_*.h5")),
    ]
    for r, (rlabel, file_fn) in enumerate(rows):
        for c, (key, plabel) in enumerate(proj05):
            ax = axes[r][c]
            for lam in LAMBDA_ORDER:
                f = file_fn(lam)
                if f is None:
                    continue
                with h5py.File(f, "r") as h:
                    wg = h["leg_L/weights"]
                    if key not in wg:
                        continue
                    w = np.asarray(wg[key]); t = np.asarray(h["times_ms"])[: w.size]
                ax.plot(t / 1000.0, w, color=palette[lam], linewidth=1.3,
                        label=_lam_label(lam))
            if key == "bs->rge_mean":
                ax.axhline(30.0, ls="--", color="grey", alpha=0.5, label="W$_{max}$=30")
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(plabel, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{rlabel}\nmean weight (pA)", fontsize=9)
            if r == 1:
                ax.set_xlabel("time (s)")
            if r == 0 and c == 2:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("STDP weight profiles across λ — descending/cutaneous projections\n"
                 "(top: medium-walk, Ia intact; bottom: air stepping, Ia gain 0.1)",
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[gallery] saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results/2026-06-20")
    ap.add_argument("--outdir", type=str, default="plots/paper")
    ap.add_argument("--zoom-ms", type=float, default=10000.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    speed_colors = [plt.get_cmap("viridis")(v) for v in (0.05, 0.5, 0.9)]
    gain_colors = ["#1f3b73", "#b5651d", "#c1272d"]

    speed_rows = [(tag, lbl) for tag, lbl, _per in SPEEDS]
    _force_matrix(
        args.indir, speed_rows,
        lambda tag, lam: _find(args.indir, f"cpg_speed_stdp_{tag}_{lam}_*.h5"),
        speed_colors, os.path.join(args.outdir, "fig9_speed_lambda_gallery.png"),
        "Speed × λ gait gallery — converged force profiles "
        f"(last {args.zoom_ms/1000:.0f} s; μ=3.5, CV=0.30)", args.zoom_ms)

    _force_matrix(
        args.indir, GAINS,
        lambda tag, lam: _find(args.indir, f"cpg_ablgrad_{tag}_{lam}_*.h5"),
        gain_colors, os.path.join(args.outdir, "fig10_ablation_lambda_gallery.png"),
        "Graded ablation × λ gait gallery — converged force profiles "
        f"(last {args.zoom_ms/1000:.0f} s; 520 ms; μ=3.5, CV=0.30)", args.zoom_ms)

    _weight_profiles(args.indir, "13_5cms", "air",
                     os.path.join(args.outdir, "fig11_weight_profiles.png"))


if __name__ == "__main__":
    main()
