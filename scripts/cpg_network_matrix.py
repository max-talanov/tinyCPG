#!/usr/bin/env python3
"""
cpg_network_matrix.py
Full-circuit population activity over the locomotion modes of the chosen --grid
(see scripts/paper_grid.py), last few seconds of the run. Rows = both legs' RG,
reciprocal-inhibition (In), Ia interneurons and motor pools (16 populations).
No in-figure title (info -> LaTeX caption).
"""
import argparse, glob, os
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_grid import GRIDS, add_grid_arg, find as _find

ROWS = [
    ("leg_L/rgf", "L RG-F"), ("leg_L/rge", "L RG-E"),
    ("leg_L/inf", "L InF"),  ("leg_L/ine", "L InE"),
    ("leg_L/iaint_f", "L IaInt-F"), ("leg_L/iaint_e", "L IaInt-E"),
    ("leg_L/mus_f", "L M-F"), ("leg_L/mus_e", "L M-E"),
    ("leg_R/rgf", "R RG-F"), ("leg_R/rge", "R RG-E"),
    ("leg_R/inf", "R InF"),  ("leg_R/ine", "R InE"),
    ("leg_R/iaint_f", "R IaInt-F"), ("leg_R/iaint_e", "R IaInt-E"),
    ("leg_R/mus_f", "R M-F"), ("leg_R/mus_e", "R M-E"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--lambda-tag", default="lam1em3")
    ap.add_argument("--out", default="paper/figures/fig_network_matrix.png")
    ap.add_argument("--dpi", type=int, default=170)
    add_grid_arg(ap)
    args = ap.parse_args()
    # columns: (label, filename stem, display window ms = the last N ms of the run)
    COLS = list(GRIDS[args.grid]["modes"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    files = {c[1]: _find(args.indir, c[1], args.lambda_tag) for c in COLS}
    nr, nc = len(ROWS), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 0.92 * nr), squeeze=False, sharex="col")

    for r, (key, rlabel) in enumerate(ROWS):
        # shared y-scale across a row for comparability
        ymax = 1.0
        series = {}
        for clabel, stem, per in COLS:
            f = files[stem]
            if f is None:
                continue
            with h5py.File(f, "r") as h:
                if key not in h:
                    continue
                t = np.asarray(h["times_ms"]); y = np.asarray(h[key])
                sim = float(h.attrs.get("sim_ms", t.max()))
            z = max(0.0, sim - per)
            m = t >= z
            series[stem] = (t[m] - z, y[m], per)
            ymax = max(ymax, float(np.nanpercentile(y[m], 99)) * 1.1)
        for c, (clabel, stem, per) in enumerate(COLS):
            ax = axes[r][c]
            if stem in series:
                tt, yy, per = series[stem]
                ax.plot(tt, yy, color="#27408b", linewidth=0.9)
                ax.set_xlim(0, per)
            ax.set_ylim(0, ymax); ax.set_yticks([0, round(ymax)])
            if rlabel.startswith("R "):
                ax.set_facecolor("#f3f6fb")
            if c == 0:
                ax.set_ylabel(rlabel, rotation=0, ha="right", va="center",
                              fontsize=11, labelpad=26)
            if r == 0:
                ax.set_title(f"({chr(97 + c)})  {clabel}", fontsize=13, fontweight="bold")
            if r == nr - 1:
                ax.set_xlabel("time (ms, last cycles)", fontsize=11)
            ax.tick_params(labelsize=8)

    fig.supylabel("population rate  (spikes·neuron⁻¹·s⁻¹)", fontsize=14, x=0.005)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[network-matrix] saved {args.out}")


if __name__ == "__main__":
    main()
