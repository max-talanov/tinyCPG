#!/usr/bin/env python3
"""
cpg_network_activity.py
Zhang-style combined network-activity figure: population firing rates of
the recorded layers (rhythm generators + motor pools) for both legs,
across the three walking speeds — the tinyCPG analogue of Zhang et al.
2022 Fig. (spike rate per neuron vs time, columns = gait/speed).

Rows (per leg, F then E, matching Zhang's lf-F / lf-E ordering):
    L RG-F, L RG-E, R RG-F, R RG-E      (rhythm generators)
    L M-F,  L M-E,  R M-F,  R M-E        (motor pools, via muscle relays)
Columns: slow / medium / fast walk (1200 / 520 / 350 ms).
Each panel shows the last N_CYCLES cycles so burst shape is comparable.

Usage:
  python3 cpg_network_activity.py --indir results/2026-06-27 \\
      --lambda-tag lam1em3 --out plots/paper/fig_network_activity.png
"""

import argparse
import glob
import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np


SPEEDS = [("06cms", "α-slow · walk\n6 cm/s (1200 ms)", 1200),
          ("13_5cms", "α-med · walk/trot\n13.5 cm/s (520 ms)", 520),
          ("21cms", "α-fast · trot\n21 cm/s (350 ms)", 350)]

# (hdf5 key, row label) — F before E, mirroring Zhang's lf-F/lf-E layout
ROWS = [
    ("leg_L/rgf", "L  RG-F"), ("leg_L/rge", "L  RG-E"),
    ("leg_R/rgf", "R  RG-F"), ("leg_R/rge", "R  RG-E"),
    ("leg_L/mus_f", "L  M-F"), ("leg_L/mus_e", "L  M-E"),
    ("leg_R/mus_f", "R  M-F"), ("leg_R/mus_e", "R  M-E"),
]


def _find(indir, pat) -> Optional[str]:
    h = sorted(glob.glob(os.path.join(indir, pat)))
    return h[0] if h else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/2026-06-27")
    ap.add_argument("--lambda-tag", default="lam1em3")
    ap.add_argument("--n-cycles", type=int, default=5)
    ap.add_argument("--out", default="plots/paper/fig_network_activity.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    files = {tag: _find(args.indir, f"cpg_speed_stdp_{tag}_{args.lambda_tag}_*.h5")
             for tag, _l, _p in SPEEDS}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8})
    nr, nc = len(ROWS), len(SPEEDS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 1.05 * nr),
                             squeeze=False, sharex="col")

    # per-row shared y-limit for comparability across speeds
    for r, (key, rlabel) in enumerate(ROWS):
        ymax = 1.0
        series = {}
        for c, (tag, _lbl, per) in enumerate(SPEEDS):
            f = files[tag]
            if f is None:
                continue
            with h5py.File(f, "r") as h:
                t = np.asarray(h["times_ms"]); y = np.asarray(h[key])
                sim = float(h.attrs.get("sim_ms", t.max()))
            z = max(0.0, sim - args.n_cycles * per)
            m = t >= z
            series[tag] = (t[m] - z, y[m], per)
            ymax = max(ymax, float(np.nanpercentile(y[m], 99)) * 1.1)
        for c, (tag, _lbl, per) in enumerate(SPEEDS):
            ax = axes[r][c]
            if tag in series:
                tt, yy, per = series[tag]
                ax.plot(tt, yy, color="#27408b", linewidth=0.9)
                ax.set_xlim(0, args.n_cycles * per)
            ax.set_ylim(0, ymax)
            ax.set_yticks([0, round(ymax)])
            if c == 0:
                ax.set_ylabel(rlabel, rotation=0, ha="right", va="center",
                              fontsize=9, labelpad=22)
            if r == 0:
                ax.set_title(SPEEDS[c][1], fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("time (ms, last cycles)")
            ax.tick_params(labelsize=6)

    fig.suptitle(
        "tinyCPG network activity — population rates (spikes·neuron⁻¹·s⁻¹) "
        "across walking speed\n"
        f"rhythm generators (RG) and motor pools (M), both legs; "
        f"last {args.n_cycles} cycles; λ=1·10⁻³, μ=3.5, CV=0.30",
        fontsize=11, y=0.997)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[network] saved {args.out}")


if __name__ == "__main__":
    main()
