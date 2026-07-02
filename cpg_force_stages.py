#!/usr/bin/env python3
"""
cpg_force_stages.py
Force profiles at three stages of STDP learning (beginning / middle / end),
one row per learning mode. Shows how the counter-phase force pattern sharpens
as the plastic weights converge over the 120 s run. Each panel is annotated
with the converged mean CUT->RG-E weight and the in-window corr(F_E,F_F).

Modes (representative canonical condition each):
  descending   cpg_speed_stdp    medium walk (13.5 cm/s)
  sensory      cpg_sensory_stdp  medium walk (13.5 cm/s)
  epidural-stim cpg_ablstim      baseline loading
  natural      cpg_ablgrad       baseline loading
  sensory-abl  cpg_ablsens       baseline loading
"""
import argparse, glob, os
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (row label, filename glob stem, [(weight-key, short-name), ...] to annotate)
MODES = [
    ("descending\n(BS+CUT plastic)",  "cpg_speed_stdp_13_5cms",  [("cut->rge_mean", "CUT")]),
    ("sensory\n(Ia→RG plastic)",  "cpg_sensory_stdp_13_5cms",[("ia->rge_mean", "Ia"), ("cut->rge_mean", "CUT")]),
    ("epidural-stim\n(CUT intact)",    "cpg_ablstim_baseline",    [("cut->rge_mean", "CUT")]),
    ("natural\n(CUT,Ia gated)",        "cpg_ablgrad_baseline",    [("cut->rge_mean", "CUT")]),
    ("sensory-abl\n(Ia gated)",        "cpg_ablsens_baseline",    [("ia->rge_mean", "Ia"), ("cut->rge_mean", "CUT")]),
]
STAGES = [("beginning", (4000, 9000)), ("middle", (40000, 45000)), ("end", (115000, 120000))]
WIN_LABEL = {"beginning": "early learning", "middle": "converging", "end": "converged"}


def _find(indir, stem, lam):
    h = sorted(glob.glob(os.path.join(indir, f"{stem}_{lam}_*.h5")))
    return h[0] if h else None


def _win(t, y, lo, hi):
    m = (t >= lo) & (t <= hi)
    return t[m], y[m], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--lambda-tag", default="lam1em4",
                    help="Use a slow-enough rate that convergence spans the 120 s run so the "
                         "three stages are distinct (at lam1em3 CUT->RG-E converges by ~8 s).")
    ap.add_argument("--out", default="plots/paper/fig_force_stages.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    nr, nc = len(MODES), len(STAGES)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 1.5 * nr), squeeze=False, sharey=True)

    for r, (mlabel, stem, wkeys) in enumerate(MODES):
        f = _find(args.indir, stem, args.lambda_tag)
        if f is None:
            for c in range(nc):
                axes[r][c].text(0.5, 0.5, f"missing:\n{stem}", ha="center", va="center",
                                fontsize=8, transform=axes[r][c].transAxes)
            continue
        with h5py.File(f, "r") as h:
            t = np.asarray(h["times_ms"])
            fe = np.asarray(h["leg_L/force_e"]); ff = np.asarray(h["leg_L/force_f"])
            ws = {kn[0]: np.asarray(h[f"leg_L/weights/{kn[0]}"])
                  for kn in wkeys if f"leg_L/weights/{kn[0]}" in h}
        for c, (sname, (lo, hi)) in enumerate(STAGES):
            ax = axes[r][c]
            tt, ee, m = _win(t, fe, lo, hi)
            _, fftt, _ = _win(t, ff, lo, hi)
            ax.plot((tt - lo) / 1000.0, ee, color="#c1440e", lw=1.1, label="F-E")
            ax.plot((tt - lo) / 1000.0, fftt, color="#1f6feb", lw=1.1, ls="--", label="F-F")
            corr = (np.corrcoef(ee, fftt)[0, 1]
                    if np.std(ee) > 1e-9 and np.std(fftt) > 1e-9 else float("nan"))
            # mean of each annotated learned weight over the window (nan-safe)
            parts = []
            for key, name in wkeys:
                if key in ws:
                    v = ws[key][m]; v = v[np.isfinite(v)]
                    if v.size:
                        parts.append(f"{name} {np.mean(v):.0f}")
            wtxt = "  ".join(parts)
            ax.set_title(f"{wtxt} pA   r={corr:+.2f}", fontsize=8)
            ax.set_xlim(0, (hi - lo) / 1000.0)
            ax.grid(alpha=0.2)
            if r == 0:
                ax.text(0.5, 1.32, f"{sname.upper()}  ({WIN_LABEL[sname]})", ha="center",
                        va="bottom", fontsize=11, fontweight="bold", transform=ax.transAxes)
            if c == 0:
                ax.set_ylabel(mlabel, rotation=0, ha="right", va="center", fontsize=9, labelpad=42)
            if r == nr - 1:
                ax.set_xlabel("time in window (s)")
    axes[0][0].legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.9)
    lamtxt = args.lambda_tag.replace("lam1em", "1·10⁻")
    fig.suptitle(f"tinyCPG — force profiles across three stages of STDP learning "
                 f"(leg L, λ={lamtxt}, μ=3.5, CV=0.30)\n"
                 "rate chosen so convergence spans the 120 s run; counter-phase sharpens "
                 "as the plastic weight grows (beginning → end)",
                 fontsize=12, y=1.005)
    fig.tight_layout(rect=(0.04, 0, 1, 0.975))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[force-stages] saved {args.out}")


if __name__ == "__main__":
    main()
