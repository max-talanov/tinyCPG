#!/usr/bin/env python3
"""
cpg_init_robustness.py
Initialisation robustness of force-triggered stepping (round 6,
run_cutforce_sweep6.sh): the 10-point STDP-init (mu, CV) grid, medium walk,
lambda = 1e-3, descending arm, 120 s.

  (a-c) weight trajectories of the three plastic projections (leg L), one
        line per initialisation point, coloured by mu
  (d)   per-point gait metrics on the steady state (t >= --steady-from-ms):
        corr(F_E,F_F) per leg, corr(F_E,L, F_E,R), and frac_at_cap (worst leg)

Usage:
  python3 scripts/cpg_init_robustness.py \
      --in results/2026-09-07/cpg_cutforce6_*.h5 results/2026-09-25/cpg_cutforce6_*.h5 \
      --out paper/figures/fig_init_robustness.png
"""
import argparse, os
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpg_cutforce_diagnostics import bouts_from_cut_on

PROJ = [("cut->rge_mean", "CUT→RG-E", 72), ("bs->rge_mean", "BS→RG-E", 30), ("bs->rgf_mean", "BS→RG-F", 30)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", default="paper/figures/fig_init_robustness.png")
    ap.add_argument("--steady-from-ms", type=float, default=30000.0)
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    runs = []
    for f in args.inp:
        with h5py.File(f, "r") as h:
            t = np.asarray(h["times_ms"])
            m = t >= args.steady_from_ms
            fe = {s: np.asarray(h[f"leg_{s}/force_e"]) for s in "LR"}
            ff = {s: np.asarray(h[f"leg_{s}/force_f"]) for s in "LR"}
            cap = float(h.attrs["cut_max_stance_ms"])
            runs.append(dict(
                mu=float(h.attrs["winit_mu"]), cv=float(h.attrs.get("winit_cv", np.nan)),
                t=t / 1000.0,
                w={k: np.asarray(h[f"leg_L/weights/{k}"]) for k, _l, _y in PROJ},
                corrF={s: float(np.corrcoef(fe[s][m], ff[s][m])[0, 1]) for s in "LR"},
                corrLR=float(np.corrcoef(fe["L"][m], fe["R"][m])[0, 1]),
                atcap=max(bouts_from_cut_on(t[m], np.asarray(h[f"leg_{s}/cut_on"])[m], cap, 110.0)["frac_at_cap"]
                          for s in "LR"),
            ))
    runs.sort(key=lambda r: r["mu"])
    mus = np.array([r["mu"] for r in runs])
    cmap = plt.cm.viridis
    col = lambda mu: cmap(mu / max(mus.max(), 1e-9))

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, (k, lab, ymax) in zip(axes[:3], PROJ):
        for r in runs:
            ax.plot(r["t"], r["w"][k], color=col(r["mu"]), lw=1.4)
        ax.set_xlim(0, 120); ax.set_ylim(0, ymax); ax.grid(alpha=0.2)
        ax.set_xlabel("time (s)"); ax.set_ylabel(f"{lab} weight (pA)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, mus.max()))
    fig.colorbar(sm, ax=axes[2], fraction=0.05, pad=0.02).set_label("initial mean μ (pA)")

    ax = axes[3]
    ax.plot(mus, [r["corrF"]["L"] for r in runs], "o-", color="#c1440e", label="corr($F_E,F_F$), leg L")
    ax.plot(mus, [r["corrF"]["R"] for r in runs], "s--", color="#c1440e", mfc="white", label="corr($F_E,F_F$), leg R")
    ax.plot(mus, [r["corrLR"] for r in runs], "^-", color="#1f6feb", label="corr($F_{E,L},F_{E,R}$)")
    ax.plot(mus, [r["atcap"] for r in runs], "d-", color="0.35", label="frac. stance at cap")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-1, 1); ax.grid(alpha=0.2)
    ax.set_xlabel("initial mean μ (pA)"); ax.set_ylabel("steady-state metric")
    ax.legend(fontsize=9, loc="upper right")
    for i, ax in enumerate(axes):
        ax.text(0.02, 0.97, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
    print(f"[init-robustness] {len(runs)} runs -> {args.out}")
    for r in runs:
        print(f"  mu={r['mu']:5.1f}  corrF L/R={r['corrF']['L']:+.2f}/{r['corrF']['R']:+.2f}  "
              f"corrLR={r['corrLR']:+.2f}  atcap={r['atcap']:.2f}  "
              f"CUT final={r['w']['cut->rge_mean'][np.isfinite(r['w']['cut->rge_mean'])][-1]:.1f}")


if __name__ == "__main__":
    main()
