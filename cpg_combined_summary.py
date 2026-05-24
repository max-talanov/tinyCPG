#!/usr/bin/env python3
"""
cpg_combined_summary.py
Single-page combined summary figure for paced-gait MN5 sweep results.

Layout (rows top to bottom):
  1. Force-E leg L  — all sweep idxes overlaid (shows E phase + sweep-init robustness)
  2. Force-F leg L  — all sweep idxes overlaid (E vs F = within-leg counter-phase)
  3. Force-E leg R  — all sweep idxes (L vs R offset = trot gait 180°)
  4. Force-F leg R
  5. RG-E + RG-F population rates (one representative idx, both legs)
  6. STDP weight trends — cut->rge, bs->rge, bs->rgf (3 subpanels, all idxes)

Example:
  python3 cpg_combined_summary.py \
    --files cpg_bursting_paced_120s_idx*.h5 \
    --out plots/mn5_paced_120s/combined_summary.png \
    --representative-idx 4
"""

import argparse
import glob
import os
import re
from typing import List

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m_idx = re.search(r"idx(\d+)", base)
    m_mu = re.search(r"mu([\d.]+)", base)
    m_cv = re.search(r"cv([\d.]+)", base)
    parts = []
    if m_idx:
        parts.append(f"idx{m_idx.group(1)}")
    if m_mu:
        parts.append(f"μ={float(m_mu.group(1)):.2f}")
    if m_cv:
        parts.append(f"CV={float(m_cv.group(1)):.2f}")
    return " ".join(parts) if parts else base


def _sweep_idx_from_filename(path: str) -> int:
    m = re.search(r"idx(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if win <= 1 or x.size == 0:
        return x
    if np.isnan(x).any():
        idx = np.where(~np.isnan(x))[0]
        if idx.size:
            x = x.copy()
            x[: idx[0]] = x[idx[0]]
            for i in range(idx.size - 1):
                a, b = idx[i], idx[i + 1]
                if b > a + 1:
                    x[a + 1 : b] = x[a]
            x[idx[-1] + 1 :] = x[idx[-1]]
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="plots/combined_summary.png")
    ap.add_argument("--representative-idx", type=int, default=4,
                    help="Sweep idx used for the single-trace RG-rate panel.")
    ap.add_argument("--smooth-sec", type=float, default=1.0)
    args = ap.parse_args()

    # Expand glob patterns
    files: List[str] = []
    for pat in args.files:
        matches = sorted(glob.glob(pat))
        files.extend(matches if matches else [pat])
    files = sorted(set(files), key=_sweep_idx_from_filename)
    if not files:
        raise SystemExit("No files matched.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[combined] {len(files)} files → {args.out}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    runs = []
    for f in files:
        h5 = h5py.File(f, "r")
        runs.append({
            "h5": h5,
            "file": f,
            "idx": _sweep_idx_from_filename(f),
            "label": _label_from_filename(f),
            "times_ms": np.asarray(h5["times_ms"]),
            "dt_ms": float(h5.attrs.get("dt_ms", 100.0)),
            "sim_ms": float(h5.attrs.get("sim_ms", 30000.0)),
        })

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(runs)))

    rep = next((r for r in runs if r["idx"] == args.representative_idx), runs[0])
    print(f"[combined] representative idx for RG panel = {rep['idx']} ({rep['label']})")

    # ---- Build figure ----
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.45, wspace=0.25)

    # Rows 1-4: force E/F for both legs (each row spans all 3 cols)
    ax_fe_l = fig.add_subplot(gs[0, :])
    ax_ff_l = fig.add_subplot(gs[1, :], sharex=ax_fe_l)
    ax_fe_r = fig.add_subplot(gs[2, :], sharex=ax_fe_l)
    ax_ff_r = fig.add_subplot(gs[3, :], sharex=ax_fe_l)

    for r, col in zip(runs, colors):
        t = r["times_ms"]
        ax_fe_l.plot(t, r["h5"]["leg_L/force_e"][:], color=col, label=r["label"], linewidth=0.7, alpha=0.85)
        ax_ff_l.plot(t, r["h5"]["leg_L/force_f"][:], color=col, label=r["label"], linewidth=0.7, alpha=0.85)
        ax_fe_r.plot(t, r["h5"]["leg_R/force_e"][:], color=col, label=r["label"], linewidth=0.7, alpha=0.85)
        ax_ff_r.plot(t, r["h5"]["leg_R/force_f"][:], color=col, label=r["label"], linewidth=0.7, alpha=0.85)

    ax_fe_l.set_title("Muscle activity — Force-E leg L  (all sweep points overlaid)")
    ax_ff_l.set_title("Muscle activity — Force-F leg L  (within-leg E↔F counter-phase visible)")
    ax_fe_r.set_title("Muscle activity — Force-E leg R  (L vs R = 180° trot offset)")
    ax_ff_r.set_title("Muscle activity — Force-F leg R")
    for ax in (ax_fe_l, ax_ff_l, ax_fe_r, ax_ff_r):
        ax.set_ylabel("force (a.u.)")
        ax.grid(alpha=0.2)
    ax_fe_l.legend(loc="upper right", fontsize=7, ncol=3)

    # Row 5: RG-E and RG-F population rates for the representative idx
    # NOTE: do NOT sharex with the force panels — the zoom panel sets its own xlim,
    # which would propagate back via sharex and clip the force traces.
    ax_rg_l = fig.add_subplot(gs[4, :2])
    ax_rg_r = fig.add_subplot(gs[4, 2])

    t_rep = rep["times_ms"]
    ax_rg_l.plot(t_rep, rep["h5"]["leg_L/rge"][:], color="#1f77b4", label="RG-E rate", linewidth=0.6, alpha=0.9)
    ax_rg_l.plot(t_rep, rep["h5"]["leg_L/rgf"][:], color="#ff7f0e", label="RG-F rate", linewidth=0.6, alpha=0.9)
    ax_rg_l.set_title(f"RG activity — leg L  ({rep['label']})")
    ax_rg_l.set_ylabel("RG rate (Hz/neuron)")
    ax_rg_l.legend(loc="upper right", fontsize=8)
    ax_rg_l.grid(alpha=0.2)

    # RG zoom — last 5 s on the representative idx
    zoom_start = max(0.0, rep["sim_ms"] - 5000.0)
    mask = t_rep >= zoom_start
    ax_rg_r.plot(t_rep[mask], rep["h5"]["leg_L/rge"][:][mask], color="#1f77b4", label="RG-E", linewidth=0.8)
    ax_rg_r.plot(t_rep[mask], rep["h5"]["leg_L/rgf"][:][mask], color="#ff7f0e", label="RG-F", linewidth=0.8)
    ax_rg_r.set_title(f"RG activity (last 5 s zoom)")
    ax_rg_r.set_xlim(zoom_start, rep["sim_ms"])
    ax_rg_r.set_xlabel("time (ms)")
    ax_rg_r.grid(alpha=0.2)

    # Rows 6-7: STDP — three subpanels (cut->rge, bs->rge, bs->rgf) using leg L weights
    proj_keys = ["cut->rge", "bs->rge", "bs->rgf"]
    win = max(1, int(args.smooth_sec * 1000.0 / max(1.0, runs[0]["dt_ms"])))
    stdp_axes = [fig.add_subplot(gs[5:7, i]) for i in range(3)]

    for ax, proj in zip(stdp_axes, proj_keys):
        for r, col in zip(runs, colors):
            w = r["h5"]["leg_L/weights"]
            mk = f"{proj}_mean"
            sk = f"{proj}_std"
            if mk not in w:
                continue
            m = _moving_average(np.asarray(w[mk][:]), win)
            ax.plot(r["times_ms"], m, color=col, label=r["label"], linewidth=1.2)
            if sk in w:
                s = _moving_average(np.asarray(w[sk][:]), win)
                ax.fill_between(r["times_ms"], m - s, m + s, color=col, alpha=0.10)
        ax.set_title(f"STDP — {proj} (mean ± std, {args.smooth_sec:.1f}s MA)")
        ax.set_ylabel("weight (pA)")
        ax.set_xlabel("time (ms)")
        ax.grid(alpha=0.2)
    stdp_axes[0].legend(loc="lower right", fontsize=7)

    # Overall title
    fig.suptitle(
        f"tinyCPG paced-gait MN5 sweep — 120 s long-term stability "
        f"(N={len(runs)} sweep points)",
        fontsize=14, y=0.995,
    )

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] saved {args.out}")

    for r in runs:
        r["h5"].close()


if __name__ == "__main__":
    main()
