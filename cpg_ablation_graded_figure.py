#!/usr/bin/env python3
"""
cpg_ablation_graded_figure.py
Paper Figure — Phase B: graded sensory ablation × STDP learning-rate matrix
(Courtine/Lavrov SCI rehabilitation paradigm).

3 Ia feedback gains:
    1.0  = baseline (full weight-bearing, intact afferent)
    0.5  = toe stepping (partial weight-bearing; Edgerton 2008; Cha 2007)
    0.1  = air stepping (limb unloaded, ≈deafferented; Lavrov 2008;
                         Hägglund 2013)

× 3 STDP learning rates:
    λ = 5e-4 (slow)
    λ = 1e-3 (baseline)
    λ = 2e-3 (fast)

Layout (13 × 14 in, portrait):
  Rows 1-3:  Force E+F leg L, last 5 s zoom, rows = Ia gain, cols = λ
  Row 4:     Three quantitative summary heat-tiles across the 3×3 matrix:
              (a) cycle period (ms)
              (b) Peak Force-E amplitude
              (c) E↔F counter-phase Pearson r

Usage:
  python3 cpg_ablation_graded_figure.py \\
      --indir results/2026-06-16 --outdir plots/paper \\
      --out plots/paper/fig6_ablation_graded.png
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


GAIN_ORDER = ["baseline", "toe", "air"]
GAIN_LABEL = {
    "baseline": "baseline\nIa gain 1.0\n(intact, weight-bearing)",
    "toe":      "toe stepping\nIa gain 0.5\n(partial weight)",
    "air":      "air stepping\nIa gain 0.1\n(unloaded ≈deaff.)",
}
GAIN_VAL = {"baseline": 1.0, "toe": 0.5, "air": 0.1}

LAMBDA_ORDER = ["lam5em4", "lam1em3", "lam2em3"]
LAMBDA_LABEL = {
    "lam5em4": "λ = 5·10⁻⁴\n(slow STDP)",
    "lam1em3": "λ = 1·10⁻³\n(baseline)",
    "lam2em3": "λ = 2·10⁻³\n(fast STDP)",
}
LAMBDA_VAL = {"lam5em4": 5e-4, "lam1em3": 1e-3, "lam2em3": 2e-3}


def _find_cycle_periods(t_ms: np.ndarray, signal: np.ndarray,
                        min_peak_height: float = 3.0,
                        min_gap_ms: float = 150.0) -> np.ndarray:
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
    """Map (gain_tag, lambda_tag) → file path.
    Files are named: cpg_ablgrad_<gain>_<lambda>_idx00_*.h5
    """
    files = sorted(glob.glob(os.path.join(indir, "cpg_ablgrad_*.h5")))
    out: Dict[Tuple[str, str], str] = {}
    rgx = re.compile(r"cpg_ablgrad_(baseline|toe|air)_(lam\d+em\d+)_")
    for f in files:
        m = rgx.search(os.path.basename(f))
        if m:
            out[(m.group(1), m.group(2))] = f
    return out


def _load(path: str) -> Dict:
    h = h5py.File(path, "r")
    t = np.asarray(h["times_ms"])
    sim_ms = float(h.attrs.get("sim_ms", 30000.0))
    period_ms = float(h.attrs.get("step_period_ms", 520.0))
    fe = np.asarray(h["leg_L/force_e"])
    ff = np.asarray(h["leg_L/force_f"])
    mask = t >= max(0.0, sim_ms - 20000.0)
    fe_w = fe[mask]; ff_w = ff[mask]; t_w = t[mask]
    periods = _find_cycle_periods(t_w, fe_w, min_peak_height=3.0,
                                  min_gap_ms=max(120.0, period_ms * 0.4))
    return {
        "h5": h, "t": t, "fe": fe, "ff": ff, "sim_ms": sim_ms,
        "period_ms": period_ms,
        "ia_gain": float(h.attrs.get("ia_feedback_gain", 1.0)),
        "stdp_lambda": float(h.attrs.get("stdp_lambda", 1e-3)),
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
    out_path = args.out or os.path.join(args.outdir, "fig6_ablation_graded.png")

    files = _resolve_files(args.indir)
    missing = [(g, l) for g in GAIN_ORDER for l in LAMBDA_ORDER if (g, l) not in files]
    if missing:
        print(f"[ablgrad] WARNING: missing cells: {missing}")
    print(f"[ablgrad] resolved {len(files)} files in {args.indir}")

    data: Dict[Tuple[str, str], Dict] = {}
    for g in GAIN_ORDER:
        for l in LAMBDA_ORDER:
            if (g, l) in files:
                data[(g, l)] = _load(files[(g, l)])

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    n_rows, n_cols = 3, 3
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.25],
                          hspace=0.55, wspace=0.28)

    # Rows 1-3: Force traces (last 5 s zoom), rows = gain, cols = lambda
    for r, g in enumerate(GAIN_ORDER):
        for c, l in enumerate(LAMBDA_ORDER):
            ax = fig.add_subplot(gs[r, c])
            if (g, l) not in data:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            d = data[(g, l)]
            z = max(0.0, d["sim_ms"] - 5000.0)
            mask = d["t"] >= z
            ax.plot(d["t"][mask], d["fe"][mask], color="#1f77b4", linewidth=1.0, label="E")
            ax.plot(d["t"][mask], d["ff"][mask], color="#ff7f0e", linewidth=1.0, label="F")
            ax.set_xlim(z, d["sim_ms"])
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(LAMBDA_LABEL[l], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{GAIN_LABEL[g]}\nforce (a.u.)", fontsize=9)
                ax.legend(loc="upper right", fontsize=7, ncol=2)
            if r == n_rows - 1:
                ax.set_xlabel("time (ms)")

    # Row 4: three summary heat-tiles
    metric_keys = ["measured_period_mean", "peak_e", "corr_ef"]
    metric_titles = [
        "(a) Cycle period (ms) — last 20 s",
        "(b) Peak Force-E (95th pct., a.u.)",
        "(c) E↔F counter-phase r",
    ]
    metric_cmaps = ["viridis", "plasma", "RdBu"]
    metric_vlim = [None, None, (-1.0, 0.4)]

    for m_idx, (mk, mtitle, cmap_name, vlim) in enumerate(
            zip(metric_keys, metric_titles, metric_cmaps, metric_vlim)):
        ax = fig.add_subplot(gs[3, m_idx])
        M = np.full((n_rows, n_cols), np.nan)
        for r, g in enumerate(GAIN_ORDER):
            for c, l in enumerate(LAMBDA_ORDER):
                if (g, l) in data:
                    M[r, c] = data[(g, l)][mk]
        if vlim is not None:
            im = ax.imshow(M, aspect="auto", cmap=cmap_name, vmin=vlim[0], vmax=vlim[1])
        else:
            im = ax.imshow(M, aspect="auto", cmap=cmap_name)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([LAMBDA_LABEL[l].split("\n")[0] for l in LAMBDA_ORDER],
                           rotation=0, fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([GAIN_LABEL[g].split("\n")[0] for g in GAIN_ORDER], fontsize=8)
        ax.set_title(mtitle, fontsize=10)
        for r in range(n_rows):
            for c in range(n_cols):
                if np.isfinite(M[r, c]):
                    val_str = f"{M[r, c]:.2f}" if mk == "corr_ef" else f"{M[r, c]:.1f}"
                    ax.text(c, r, val_str, ha="center", va="center",
                            color="white" if (cmap_name in ("viridis", "plasma") and r >= 1) else "black",
                            fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    fig.suptitle(
        "Phase B — Graded sensory ablation × STDP learning-rate matrix\n"
        "Courtine/Lavrov SCI paradigm: baseline → toe stepping → air stepping"
        "  (μ=3.5, CV=0.30, 520 ms period, 30 s each)",
        fontsize=12, y=0.997,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablgrad] saved {out_path}")

    csv_path = os.path.splitext(out_path)[0] + "_metrics.csv"
    with open(csv_path, "w") as fh:
        fh.write("ia_gain_tag,ia_gain_val,lambda_tag,lambda_val,"
                 "measured_period_ms,period_std_ms,peak_force_e,peak_force_f,corr_ef\n")
        for g in GAIN_ORDER:
            for l in LAMBDA_ORDER:
                if (g, l) not in data:
                    continue
                d = data[(g, l)]
                fh.write(f"{g},{GAIN_VAL[g]},{l},{LAMBDA_VAL[l]},"
                         f"{d['measured_period_mean']:.2f},{d['measured_period_std']:.2f},"
                         f"{d['peak_e']:.3f},{d['peak_f']:.3f},{d['corr_ef']:.3f}\n")
    print(f"[ablgrad] metrics CSV: {csv_path}")

    for d in data.values():
        d["h5"].close()


if __name__ == "__main__":
    main()
