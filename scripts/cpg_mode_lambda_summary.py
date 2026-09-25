#!/usr/bin/env python3
"""
cpg_mode_lambda_summary.py
Extended comparison across the 5 locomotion modes x N STDP rates (lambda) of the
canonical sensory model. Auto-detects the lambda tags present (3 now, 5 after the
5x5 re-run). Produces three views of the same metrics:
  (a) metric heatmaps  -> <out>_heatmaps.png
  (b) line-plot trends -> <out>_trends.png
  (c) LaTeX + CSV table -> <out>_table.tex / .csv
Metrics per (mode, lambda) cell, last 20 s: corr(F_E,F_F), corr(RG-E,RG-F),
peak Force-E (95th pct), converged CUT->RG-E weight.
"""
import argparse, glob, os, re
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_grid import GRIDS, LAM_VAL, add_grid_arg, find as _find
from cpg_cutforce_diagnostics import bouts_from_cut_on

METRICS_BASE = [  # (key, label, cmap, vmin, vmax)
    ("corrF",  "corr($F_E,F_F$)",      "RdBu",     -1, 1),
    ("corrRG", "corr(RG-E,RG-F)",      "RdBu",     -1, 1),
    ("Fpk",    "peak Force-E (a.u.)",  "viridis",   0, 18),
    ("cut",    "converged CUT→RG-E (pA)", "magma",  0, 70),
]
# force-triggered grid adds L/R alternation and the failsafe-cap check
METRICS_FT = METRICS_BASE[:2] + [
    ("corrLR", "corr($F_{E,L},F_{E,R}$)", "RdBu",   -1, 1),
    ("atcap",  "frac. stance at cap",      "Reds",    0, 1),
    # force-triggered stance ends well before force saturates: peaks ~4-7 a.u., not ~17
    ("Fpk",    "peak Force-E (a.u.)",      "viridis", 0, 8),
    METRICS_BASE[3],
]


def metrics(f):
    with h5py.File(f, "r") as h:
        t = np.asarray(h["times_ms"]); sim = float(h.attrs["sim_ms"]); m = t >= sim - 20000
        g = lambda k: np.asarray(h["leg_L/" + k])[m]
        fe, ff, rge, rgf = g("force_e"), g("force_f"), g("rge"), g("rgf")
        cc = lambda x, y: (float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
        wg = h["leg_L/weights"]
        cutv = np.asarray(wg["cut->rge_mean"]); cutv = cutv[np.isfinite(cutv)]
        out = dict(corrF=cc(fe, ff), corrRG=cc(rge, rgf),
                   Fpk=float(np.nanpercentile(fe, 95)),
                   cut=float(cutv[-1]) if cutv.size else np.nan)
        out["corrLR"] = cc(fe, np.asarray(h["leg_R/force_e"])[m])
        cap = float(h.attrs.get("cut_max_stance_ms", 0.0))
        if "leg_L/cut_on" in h and cap > 0:
            # worst leg: a result is only genuine if neither leg is timer-driven
            out["atcap"] = max(bouts_from_cut_on(t[m], np.asarray(h[f"leg_{s}/cut_on"])[m],
                                                 cap, 110.0)["frac_at_cap"] for s in ("L", "R"))
        else:
            out["atcap"] = np.nan
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", default="plots/paper/fig_mode_lambda_summary")
    add_grid_arg(ap)
    args = ap.parse_args()
    G = GRIDS[args.grid]
    MODES = [(lab, stem) for lab, stem, _w in G["modes"]]
    METRICS = METRICS_FT if args.grid == "ftgrid" else METRICS_BASE
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # detect lambda tags actually present, order high->low rate
    lams = sorted({re.search(r"(lam1em\d)", os.path.basename(p)).group(1)
                   for _l, stem in MODES
                   for p in glob.glob(os.path.join(args.indir, f"{stem}_lam1em*_*.h5"))},
                  key=lambda L: -LAM_VAL[L])
    if not lams:
        raise SystemExit(f"no {args.grid} grid files in {args.indir}")
    lam_lbl = [f"$10^{{{int(np.log10(LAM_VAL[L]))}}}$" for L in lams]
    nM, nL = len(MODES), len(lams)

    # gather metric matrices
    data = {mk: np.full((nM, nL), np.nan) for mk, *_ in METRICS}
    for r, (mlabel, stem) in enumerate(MODES):
        for c, lam in enumerate(lams):
            f = _find(args.indir, stem, lam)
            if f is None:
                continue
            mt = metrics(f)
            for mk, *_ in METRICS:
                data[mk][r, c] = mt[mk]

    mode_short = [m[0].split("\n")[0] for m in MODES]

    # (a) heatmaps
    ncol = len(METRICS) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 3.6 * 2))
    for ax, (mk, mlab, cmap, vmin, vmax) in zip(axes.flat, METRICS):
        M = data[mk]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(nL)); ax.set_xticklabels([f"λ={l}" for l in lam_lbl], fontsize=9)
        ax.set_yticks(range(nM)); ax.set_yticklabels(mode_short, fontsize=9)
        ax.set_title(mlab, fontsize=12, fontweight="bold")
        for i in range(nM):
            for j in range(nL):
                if np.isfinite(M[i, j]):
                    val = f"{M[i,j]:.2f}" if (mk.startswith("corr") or mk == "atcap") else f"{M[i,j]:.0f}"
                    # white text on the diverging (corr) panels and on dark
                    # viridis/magma cells; black otherwise.
                    tcol = "white" if (cmap == "RdBu" or (cmap in ("viridis", "magma") and M[i, j] < vmax * 0.55)
                                       or (cmap == "Reds" and M[i, j] > 0.55)) else "black"
                    ax.text(j, i, val, ha="center", va="center", fontsize=9, color=tcol)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.out + "_heatmaps.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # (b) line-plot trends: metric vs mode, one line per lambda
    fig, axes = plt.subplots(2, ncol, figsize=(4.8 * ncol, 3.4 * 2))
    x = np.arange(nM)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, nL))
    for ax, (mk, mlab, *_r) in zip(axes.flat, METRICS):
        for c, (lam, col) in enumerate(zip(lams, colors)):
            ax.plot(x, data[mk][:, c], "o-", color=col, lw=1.8, label=f"λ={lam_lbl[c]}")
        ax.set_xticks(x); ax.set_xticklabels(mode_short, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(mlab, fontsize=11); ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, ncol=2, title="STDP rate")
    fig.suptitle("Metric trends across locomotion modes, one line per STDP rate λ",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out + "_trends.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # (c) table (CSV + LaTeX)
    with open(args.out + "_table.csv", "w") as fh:
        fh.write("mode,lambda," + ",".join(mk for mk, *_ in METRICS) + "\n")
        for r, (ml, _s) in enumerate(MODES):
            for c, lam in enumerate(lams):
                fh.write(f"{ml.split(chr(10))[0]},{LAM_VAL[lam]:.0e},"
                         + ",".join(f"{data[mk][r,c]:.3f}" for mk, *_ in METRICS) + "\n")
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \caption{Comparison across the " + f"{nM}" + r" locomotion modes $\times$ "
         + f"{nL}" + r" STDP rates $\lambda$ (" + G["model"] + r", last 20\,s, leg L"
         + (r"; L/R and at-cap columns use both legs" if args.grid == "ftgrid" else "") + r").}",
         r"  \label{tab:mode_lambda}", r"  \begin{tabular}{ll " + "r" * len(METRICS) + "}", r"    \toprule"]
    TEXHDR = {"corrF": r"corr$(F_E,F_F)$", "corrRG": "corr(RG)", "corrLR": r"corr$(F_{E,L},F_{E,R})$",
              "atcap": "at cap", "Fpk": r"peak $F_E$", "cut": r"CUT$\to$RG-E"}
    TEXFMT = {"corrF": ".2f", "corrRG": ".2f", "corrLR": ".2f", "atcap": ".2f", "Fpk": ".1f", "cut": ".0f"}
    L += [r"    Mode & $\lambda$ & " + " & ".join(TEXHDR[mk] for mk, *_ in METRICS) + r" \\",
          r"    \midrule"]
    for r, (ml, _s) in enumerate(MODES):
        for c, lam in enumerate(lams):
            L.append(f"    {ml.split(chr(10))[0]} & {lam_lbl[c]} & "
                     + " & ".join(f"${data[mk][r,c]:{TEXFMT[mk]}}$" for mk, *_ in METRICS) + r" \\")
        L.append(r"    \addlinespace[2pt]")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    open(args.out + "_table.tex", "w").write("\n".join(L) + "\n")
    print(f"[summary] {nM} modes x {nL} lambda -> {args.out}_heatmaps.png / _trends.png / _table.{{tex,csv}}")


if __name__ == "__main__":
    main()
