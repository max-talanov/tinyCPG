#!/usr/bin/env python3
"""
cpg_cutforce_article_figure.py
Publication-style figures for the closed-loop force-triggered CUT extension
(--cut-trigger force --muscle-fatigue), matching the visual conventions of the
paper's existing figure scripts (cpg_force_stages.py, cpg_mode_lambda_summary.py).

Produces two figures:
  (a) fig_cutforce_example.png     -- example Force-E/Force-F traces for both
      legs at the confirmed operating point (tau=260ms, off-frac=0.35,
      cap=450ms), showing genuine cycle-to-cycle variability vs. the flat,
      perfectly-repeating waveform of a cap-dominated (disguised-clock) run.
  (b) fig_cutforce_robustness.png  -- Phase 3 seed/init robustness: four
      metrics (corr(Force-E,Force-F) per leg, corr(Force-E_L,Force-E_R),
      frac_at_cap, converged CUT->RG-E weight) plotted against the STDP
      initial-weight mean mu, mirroring the paper's own Fig 15/16
      "metric trends" convention.

Usage:
  python3 cpg_cutforce_article_figure.py \\
      --example-h5 results/2026-09-01/cpg_cutforce5_fat260_off0350_idx00_mu03.50_cv00.30_seed12345.h5 \\
      --robustness-dir results/2026-09-07 \\
      --outdir paper/figures
"""
import argparse
import glob
import os
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bout_stats(t_ms, cut_on, cap_ms, cap_tol_ms=110.0):
    """Exact bout durations + frac_at_cap from the ground-truth cut_on array
    (same logic as scripts/cpg_cutforce_diagnostics.py)."""
    stance = cut_on > 0.5
    trans = np.diff(stance.astype(int))
    onsets = t_ms[1:][trans == 1]
    offsets = t_ms[1:][trans == -1]
    durs = []
    for on in onsets:
        off = offsets[offsets > on]
        if len(off):
            durs.append(float(off[0] - on))
    durs = np.asarray(durs, dtype=float)
    if durs.size == 0:
        return np.nan
    return float(np.mean((durs >= cap_ms) & (durs <= cap_ms + cap_tol_ms)))


def fig_example(example_h5, out_path, dpi=200):
    with h5py.File(example_h5, "r") as f:
        t = np.asarray(f["times_ms"])
        cap_ms = float(f.attrs.get("cut_max_stance_ms", 0.0))
        data = {}
        for side in ("L", "R"):
            fe = np.asarray(f[f"leg_{side}/force_e"])
            ff = np.asarray(f[f"leg_{side}/force_f"])
            cut_on = np.asarray(f[f"leg_{side}/cut_on"])
            corr = float(np.corrcoef(fe, ff)[0, 1])
            at_cap = bout_stats(t, cut_on, cap_ms)
            data[side] = dict(fe=fe, ff=ff, corr=corr, at_cap=at_cap)
        fe_l, fe_r = data["L"]["fe"], data["R"]["fe"]
        corr_lr = float(np.corrcoef(fe_l, fe_r)[0, 1])
        tau = f.attrs.get("fatigue_tau_onset_ms", "?")
        off_frac = f.attrs.get("cut_force_off_frac", "?")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, side in zip(axes, ("L", "R")):
        d = data[side]
        ax.plot(t / 1000.0, d["fe"], color="tab:blue", lw=0.9, label="Force E")
        ax.plot(t / 1000.0, d["ff"], color="tab:orange", lw=0.9, label="Force F")
        ax.set_ylabel("force (a.u.)")
        ax.set_title(f"leg {side} — corr($F_E,F_F$)={d['corr']:.2f}, "
                     f"frac at cap={d['at_cap']:.2f}", fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Closed-loop force-triggered CUT — genuinely emergent gait "
                 f"($\\tau_{{fatigue}}$={tau}ms, off-frac={off_frac}, "
                 f"corr($F_{{E,L}},F_{{E,R}}$)={corr_lr:.2f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[cutforce-example] saved {out_path}")


def fig_robustness(robustness_dir, out_path, dpi=200):
    files = sorted(glob.glob(os.path.join(robustness_dir, "cpg_cutforce6_robustness_idx*.h5")))
    if not files:
        raise SystemExit(f"No cpg_cutforce6_robustness_idx*.h5 files found in {robustness_dir}")

    rows = []
    for fn in files:
        idx_match = re.search(r"idx(\d+)", os.path.basename(fn))
        idx = int(idx_match.group(1)) if idx_match else -1
        with h5py.File(fn, "r") as f:
            t = np.asarray(f["times_ms"])
            cap_ms = float(f.attrs.get("cut_max_stance_ms", 0.0))
            mu = float(f.attrs.get("winit_mu", np.nan))
            fe_l = np.asarray(f["leg_L/force_e"]); ff_l = np.asarray(f["leg_L/force_f"])
            fe_r = np.asarray(f["leg_R/force_e"]); ff_r = np.asarray(f["leg_R/force_f"])
            corr_l = float(np.corrcoef(fe_l, ff_l)[0, 1])
            corr_r = float(np.corrcoef(fe_r, ff_r)[0, 1])
            corr_lr = float(np.corrcoef(fe_l, fe_r)[0, 1])
            cut_on_l = np.asarray(f["leg_L/cut_on"]); cut_on_r = np.asarray(f["leg_R/cut_on"])
            at_cap_l = bout_stats(t, cut_on_l, cap_ms)
            at_cap_r = bout_stats(t, cut_on_r, cap_ms)
            cutv = np.asarray(f["leg_L/weights"]["cut->rge_mean"])
            cutv = cutv[np.isfinite(cutv)]
            cut_final = float(cutv[-1]) if cutv.size else np.nan
        rows.append(dict(idx=idx, mu=mu, corr_l=corr_l, corr_r=corr_r, corr_lr=corr_lr,
                          at_cap_l=at_cap_l, at_cap_r=at_cap_r, cut_final=cut_final))
    rows.sort(key=lambda r: r["idx"])
    mus = [r["mu"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    ax.plot(mus, [r["corr_l"] for r in rows], "o-", color="tab:blue", label="leg L")
    ax.plot(mus, [r["corr_r"] for r in rows], "s-", color="tab:orange", label="leg R")
    ax.axhspan(-0.8, -0.7, color="gray", alpha=0.15, label="recalibrated target")
    ax.set_ylabel("corr($F_E,F_F$)")
    ax.set_title("Counter-phase strength vs. initial weight", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(mus, [r["corr_lr"] for r in rows], "o-", color="tab:green")
    ax.set_ylabel("corr($F_{E,L}, F_{E,R}$)")
    ax.set_title("L/R anti-phase vs. initial weight", fontsize=10)
    ax.axhline(0, color="k", lw=0.7, ls=":")

    ax = axes[1, 0]
    ax.plot(mus, [r["at_cap_l"] for r in rows], "o-", color="tab:blue", label="leg L")
    ax.plot(mus, [r["at_cap_r"] for r in rows], "s-", color="tab:orange", label="leg R")
    ax.set_ylabel("frac. bouts at failsafe cap")
    ax.set_xlabel(r"STDP initial-weight mean $\mu$ (pA)")
    ax.set_title("Cap-domination check (want ~0)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(mus, [r["cut_final"] for r in rows], "o-", color="tab:purple")
    ax.set_ylabel(r"converged CUT$\to$RG-E (pA)")
    ax.set_xlabel(r"STDP initial-weight mean $\mu$ (pA)")
    ax.set_title("STDP set-point vs. initial weight", fontsize=10)

    fig.suptitle("Phase 3 — force-triggered CUT robustness across STDP initial-weight "
                 f"({len(rows)}/10 points, $\\tau_{{fatigue}}$=260ms, off-frac=0.35)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[cutforce-robustness] saved {out_path} ({len(rows)} points: "
          f"mu={sorted(mus)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example-h5", required=True,
                    help="HDF5 for the single-run example figure (round 5 winning config)")
    ap.add_argument("--robustness-dir", required=True,
                    help="Directory containing cpg_cutforce6_robustness_idx*.h5 (round 6)")
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig_example(args.example_h5, os.path.join(args.outdir, "fig_cutforce_example.png"), args.dpi)
    fig_robustness(args.robustness_dir, os.path.join(args.outdir, "fig_cutforce_robustness.png"), args.dpi)


if __name__ == "__main__":
    main()
