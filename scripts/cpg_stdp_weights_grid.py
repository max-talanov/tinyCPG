#!/usr/bin/env python3
"""
cpg_stdp_weights_grid.py  (Results figure #2)
Combined STDP weight trajectories of all three plastic projections across the
locomotion modes (rows) x STDP rates lambda (columns) of the chosen --grid
(see scripts/paper_grid.py). Leg L. The primary projection (CUT->RG-E) on the
left axis (pA), the two auxiliary projections (BS->RG or Ia->RG) on the right
axis. Large fonts; every axis labelled with units.
"""
import argparse, glob, os
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_grid import GRIDS, LAMBDAS, add_grid_arg, find as _find

PRIMARY_COLOR = "#c1440e"
AUX_COLORS = ["#1f77b4", "#2ca02c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", default="paper/figures/fig_stdp_weights_grid.png")
    ap.add_argument("--dpi", type=int, default=170)
    add_grid_arg(ap)
    args = ap.parse_args()
    G = GRIDS[args.grid]
    MODES = [(lab, stem) for lab, stem, _w in G["modes"]]
    CUT = (G["primary"][0], G["primary"][1], PRIMARY_COLOR)
    IAS = [(k, lab, col) for (k, lab, _ym), col in zip(G["aux"], AUX_COLORS)]
    ymax_l, ymax_r = G["primary"][2], max(ym for _k, _l, ym in G["aux"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})

    nr, nc = len(MODES), len(LAMBDAS)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 2.5 * nr), squeeze=False, sharex=True)

    for r, (mlabel, stem) in enumerate(MODES):
        for c, (lam, llab) in enumerate(LAMBDAS):
            ax = axes[r][c]; axr = ax.twinx()
            f = _find(args.indir, stem, lam)
            if f is not None:
                with h5py.File(f, "r") as h:
                    t = np.asarray(h["times_ms"]) / 1000.0
                    if f"leg_L/weights/{CUT[0]}" in h:
                        ax.plot(t, np.asarray(h[f"leg_L/weights/{CUT[0]}"]),
                                color=CUT[2], lw=2.0, label=CUT[1])
                    for key, name, col in IAS:
                        if f"leg_L/weights/{key}" in h:
                            axr.plot(t, np.asarray(h[f"leg_L/weights/{key}"]),
                                     color=col, lw=1.6, ls="--", label=name)
            ax.set_xlim(0, 120); ax.set_ylim(0, ymax_l); axr.set_ylim(0, ymax_r)
            ax.grid(alpha=0.2); ax.tick_params(labelsize=12); axr.tick_params(labelsize=12)
            if c != 0:
                ax.set_yticklabels([])
            if c != nc - 1:
                axr.set_yticklabels([])
            if r == 0:
                ax.set_title(llab, fontsize=18, fontweight="bold")
            if c == 0:
                ax.set_ylabel(mlabel + f"\n\n{CUT[1]} (pA)", fontsize=15)
            if c == nc - 1:
                axr.set_ylabel(G["aux_axis"], fontsize=14, color="0.35")
            axr.tick_params(axis="y", labelcolor="0.35")
            if r == nr - 1:
                ax.set_xlabel("time (s)", fontsize=15)
    # panel index letters (row-major)
    _L = [chr(97 + i) if i < 26 else chr(96 + i // 26) + chr(97 + i % 26) for i in range(nr * nc)]
    for i, ax in enumerate(axes.flat):
        ax.text(0.03, 0.96, f"({_L[i]})", transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top", ha="left")

    # single combined legend
    h1 = [plt.Line2D([0], [0], color=CUT[2], lw=2.0, label=CUT[1] + " (left axis)")]
    h2 = [plt.Line2D([0], [0], color=c, lw=1.6, ls="--", label=n + " (right axis)")
          for _, n, c in IAS]
    axes[-1][-1].legend(handles=h1 + h2, loc="center right", ncol=1, fontsize=12,
                        frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[stdp-weights-grid] saved {args.out}")


if __name__ == "__main__":
    main()
