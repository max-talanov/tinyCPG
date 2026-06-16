#!/usr/bin/env python3
"""
cpg_speed_stdp_figure.py
Paper Figure — Phase A: speed × STDP learning-rate matrix.

3 walking speeds (Courtine/Lavrov range):
    6 cm/s   = 1200 ms cycle (slow walk)
    13.5 cm/s = 520 ms cycle  (medium walk)
    21 cm/s   = 350 ms cycle  (fast walk)

× 3 STDP learning rates (bio-plausible cortical range):
    λ = 5e-4 (slow)
    λ = 1e-3 (baseline, Morrison 2007)
    λ = 2e-3 (fast)

Layout (12 × 14 in, portrait — paper figure):
  Rows 1-3:  Force E+F leg L, last 5 s zoom, 3 rows × 3 cols
              Rows = speed (slow / medium / fast)
              Cols = λ      (slow / baseline / fast)
  Row 4:     Three quantitative summary panels across the 3×3 matrix:
              (a) measured cycle period (ms) heat-tile
              (b) Force-E peak amplitude (a.u.) heat-tile
              (c) E↔F counter-phase Pearson r heat-tile

Usage:
  python3 cpg_speed_stdp_figure.py \\
      --indir results/2026-06-16 --outdir plots/paper \\
      --out plots/paper/fig5_speed_stdp.png
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


SPEED_ORDER = ["06cms", "13_5cms", "21cms"]
SPEED_LABEL = {
    "06cms":   "slow walk\n≈6 cm/s  (1200 ms)",
    "13_5cms": "medium walk\n≈13.5 cm/s  (520 ms)",
    "21cms":   "fast walk\n≈21 cm/s  (350 ms)",
}
SPEED_PERIOD_MS = {"06cms": 1200, "13_5cms": 520, "21cms": 350}

LAMBDA_ORDER = ["lam5em4", "lam1em3", "lam2em3"]
LAMBDA_LABEL = {
    "lam5em4": "λ = 5·10⁻⁴\n(slow STDP)",
    "lam1em3": "λ = 1·10⁻³\n(baseline)",
    "lam2em3": "λ = 2·10⁻³\n(fast STDP)",
}
LAMBDA_VAL = {"lam5em4": 5e-4, "lam1em3": 1e-3, "lam2em3": 2e-3}


def _find_cycle_periods(t_ms: np.ndarray, signal: np.ndarray,
                        min_peak_height: float = 5.0,
                        min_gap_ms: float = 200.0) -> np.ndarray:
    if signal.size < 5:
        return np.array([])
    dt_med = float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 100.0
    min_gap_samples = max(1, int(min_gap_ms / max(1e-9, dt_med)))
    above = signal > min_peak_height
    peaks: List[int] = []
    last = -10**9
    for i in range(1, signal.size - 1):
        if above[i] and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last >= min_gap_samples:
                peaks.append(i)
                last = i
    if len(peaks) < 2:
        return np.array([])
    return np.diff(t_ms[np.asarray(peaks, dtype=int)])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or b.size < 4:
        return np.nan
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else np.nan


def _resolve_files(indir: str) -> Dict[Tuple[str, str], str]:
    """Map (speed_tag, lambda_tag) → file path."""
    files = sorted(glob.glob(os.path.join(indir, "cpg_speed_stdp_*.h5")))
    out: Dict[Tuple[str, str], str] = {}
    # Match any of the known speed tags (06cms, 13_5cms, 21cms) explicitly.
    # Using "[^_]+" fails for "13_5cms" because of the underscore inside.
    speed_re = "|".join(re.escape(s) for s in SPEED_ORDER)
    rgx = re.compile(rf"cpg_speed_stdp_({speed_re})_(lam\d+em\d+)_")
    for f in files:
        m = rgx.search(os.path.basename(f))
        if m:
            out[(m.group(1), m.group(2))] = f
    return out


def _load(path: str) -> Dict:
    h = h5py.File(path, "r")
    t = np.asarray(h["times_ms"])
    sim_ms = float(h.attrs.get("sim_ms", 30000.0))
    period_ms = float(h.attrs.get("step_period_ms", 1000.0))
    fe = np.asarray(h["leg_L/force_e"])
    ff = np.asarray(h["leg_L/force_f"])
    # Metrics on last 20 s
    mask = t >= max(0.0, sim_ms - 20000.0)
    fe_w = fe[mask]; ff_w = ff[mask]; t_w = t[mask]
    periods = _find_cycle_periods(t_w, fe_w, min_peak_height=5.0,
                                  min_gap_ms=max(150.0, period_ms * 0.4))
    return {
        "h5": h, "t": t, "fe": fe, "ff": ff, "sim_ms": sim_ms,
        "period_ms": period_ms,
        "measured_period_mean": float(np.mean(periods)) if periods.size else np.nan,
        "measured_period_std":  float(np.std(periods))  if periods.size else np.nan,
        "peak_e":  float(np.percentile(fe_w, 95)),
        "peak_f":  float(np.percentile(ff_w, 95)),
        "corr_ef": _safe_corr(fe_w, ff_w),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results/2026-06-16")
    ap.add_argument("--outdir", type=str, default="plots/paper")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = args.out or os.path.join(args.outdir, "fig5_speed_stdp.png")

    files = _resolve_files(args.indir)
    missing = [(s, l) for s in SPEED_ORDER for l in LAMBDA_ORDER if (s, l) not in files]
    if missing:
        print(f"[speed_stdp] WARNING: missing cells: {missing}")
    print(f"[speed_stdp] resolved {len(files)} files in {args.indir}")

    # Pre-load 3×3 matrix
    data: Dict[Tuple[str, str], Dict] = {}
    for s in SPEED_ORDER:
        for l in LAMBDA_ORDER:
            if (s, l) in files:
                data[(s, l)] = _load(files[(s, l)])

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    n_rows, n_cols = 3, 3
    fig = plt.figure(figsize=(13, 14))
    # 4 rows: 3 for traces, 1 for metrics. Metrics row a bit taller for ticks/colorbars.
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.25],
                          hspace=0.55, wspace=0.28)

    # Rows 1-3: Force traces (last 5 s zoom), rows = speed, cols = lambda
    for r, s in enumerate(SPEED_ORDER):
        for c, l in enumerate(LAMBDA_ORDER):
            ax = fig.add_subplot(gs[r, c])
            if (s, l) not in data:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            d = data[(s, l)]
            z = max(0.0, d["sim_ms"] - 5000.0)
            mask = d["t"] >= z
            ax.plot(d["t"][mask], d["fe"][mask], color="#1f77b4", linewidth=1.0, label="E")
            ax.plot(d["t"][mask], d["ff"][mask], color="#ff7f0e", linewidth=1.0, label="F")
            ax.set_xlim(z, d["sim_ms"])
            ax.grid(alpha=0.2)
            # Column headers on row 0
            if r == 0:
                ax.set_title(LAMBDA_LABEL[l], fontsize=9)
            # Row labels on col 0
            if c == 0:
                ax.set_ylabel(f"{SPEED_LABEL[s]}\nforce (a.u.)", fontsize=9)
                ax.legend(loc="upper right", fontsize=7, ncol=2)
            if r == n_rows - 1:
                ax.set_xlabel("time (ms)")

    # Row 4: three summary heat-tiles across the 3×3 matrix
    metric_keys = ["measured_period_mean", "peak_e", "corr_ef"]
    metric_titles = [
        "(a) Measured cycle period (ms)\nvs.\ commanded",
        "(b) Peak Force-E (95th pct., a.u.)",
        "(c) E↔F counter-phase r",
    ]
    metric_cmaps = ["viridis", "plasma", "RdBu"]
    metric_vlim = [None, None, (-1.0, 0.4)]

    for m_idx, (mk, mtitle, cmap_name, vlim) in enumerate(
            zip(metric_keys, metric_titles, metric_cmaps, metric_vlim)):
        ax = fig.add_subplot(gs[3, m_idx])
        M = np.full((n_rows, n_cols), np.nan)
        for r, s in enumerate(SPEED_ORDER):
            for c, l in enumerate(LAMBDA_ORDER):
                if (s, l) in data:
                    M[r, c] = data[(s, l)][mk]
        if vlim is not None:
            im = ax.imshow(M, aspect="auto", cmap=cmap_name, vmin=vlim[0], vmax=vlim[1])
        else:
            im = ax.imshow(M, aspect="auto", cmap=cmap_name)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([LAMBDA_LABEL[l].split("\n")[0] for l in LAMBDA_ORDER],
                           rotation=0, fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([SPEED_LABEL[s].split("\n")[0] for s in SPEED_ORDER], fontsize=8)
        ax.set_title(mtitle, fontsize=10)
        # Cell-value annotations
        for r in range(n_rows):
            for c in range(n_cols):
                if np.isfinite(M[r, c]):
                    val_str = f"{M[r, c]:.2f}" if mk == "corr_ef" else f"{M[r, c]:.1f}"
                    ax.text(c, r, val_str, ha="center", va="center",
                            color="white" if (cmap_name in ("viridis", "plasma") and r >= 1) else "black",
                            fontsize=9)
        # For cycle period (a) draw commanded value as a small marker on diagonal labels
        if mk == "measured_period_mean":
            # Annotate commanded period in row labels
            ax.set_yticklabels(
                [f"{SPEED_LABEL[s].split(chr(10))[0]}\n(cmd={SPEED_PERIOD_MS[s]} ms)"
                 for s in SPEED_ORDER],
                fontsize=7,
            )
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    fig.suptitle(
        "Phase A — Speed × STDP learning-rate matrix\n"
        "Self-organisation across rat locomotor range and bio-plausible λ"
        "  (μ=3.5, CV=0.30, 120 s each)",
        fontsize=12, y=0.997,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[speed_stdp] saved {out_path}")

    # Metrics CSV
    csv_path = os.path.splitext(out_path)[0] + "_metrics.csv"
    with open(csv_path, "w") as fh:
        fh.write("speed,lambda,commanded_period_ms,measured_period_ms,period_std_ms,"
                 "peak_force_e,peak_force_f,corr_ef\n")
        for s in SPEED_ORDER:
            for l in LAMBDA_ORDER:
                if (s, l) not in data:
                    continue
                d = data[(s, l)]
                fh.write(f"{s},{l},{SPEED_PERIOD_MS[s]},"
                         f"{d['measured_period_mean']:.2f},{d['measured_period_std']:.2f},"
                         f"{d['peak_e']:.3f},{d['peak_f']:.3f},{d['corr_ef']:.3f}\n")
    print(f"[speed_stdp] metrics CSV: {csv_path}")

    for d in data.values():
        d["h5"].close()


if __name__ == "__main__":
    main()
