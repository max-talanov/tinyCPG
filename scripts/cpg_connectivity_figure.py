#!/usr/bin/env python3
"""
cpg_connectivity_figure.py
Connectivity-statistics figure from a --dump-connectivity HDF5:
distribution of synaptic WEIGHTS and DELAYS along every projection.

  (a) weight histograms for the plastic projections (lognormal init)
  (b) static synaptic weight per projection (signed bar — E/I structure)
  (c) conduction delay per projection (mean ± jitter), sorted

Also writes a CSV table (n, weight mean±std, delay mean±std) for the
Methods/Supplementary.

Usage:
  python3 cpg_connectivity_figure.py --in results/connectivity/conn_dump.h5 \\
      --out plots/paper/fig_connectivity.png
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="plots/paper/fig_connectivity.png")
    ap.add_argument("--learned", default="",
                    help="simulation HDF5 whose final full-weight snapshot (leg L) gives the "
                         "LEARNED distribution of each plastic projection. Panel (a) then "
                         "overlays initial vs learned, and panel (b) plots the learned "
                         "mean/s.d. for plastic projections (initial mean as an open marker). "
                         "Without it, all panels show the network as built at t=0.")
    ap.add_argument("--tex", default="",
                    help="also write the grouped supplementary LaTeX table (tab:delays) here")
    ap.add_argument("--tex-caption-extra", default="",
                    help="sentence appended to the table caption (e.g. which run the learned values come from)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    with h5py.File(args.inp, "r") as h:
        for k in h:
            g = h[k]
            rows.append(dict(name=g.attrs["projection"], n=int(g.attrs["n"]),
                             w=np.asarray(g["weight"]), d=np.asarray(g["delay"]),
                             wm=float(g.attrs["w_mean"]), ws=float(g.attrs["w_std"]),
                             dm=float(g.attrs["d_mean"]), ds=float(g.attrs["d_std"])))
    rows.sort(key=lambda r: r["dm"])
    # plastic projection name (as in the dump) -> full_weights key in a run HDF5
    FULLKEY = {"CUT->RG-E (plastic)": "cut_to_rge", "BS->RG-E": "bs_to_rge",
               "BS->RG-F": "bs_to_rgf", "Ia-E->RG-E": "ia_to_rge", "Ia-F->RG-F": "ia_to_rgf"}
    learned, learned_t = {}, None
    if args.learned:
        with h5py.File(args.learned, "r") as h:
            fw = h["leg_L/full_weights"]
            for nm, key in FULLKEY.items():
                if key in fw:
                    learned[nm] = np.asarray(fw[key]["w"][-1], dtype=float)
            learned_t = float(np.asarray(h["weights_times_ms"])[-1]) / 1000.0
        plastic_names = set(learned)
    else:
        # no run given: identify plastic projections by name (static synapses also
        # carry a lognormal distribution, so panel (a) shows only the plastic ones)
        plastic_names = {r["name"] for r in rows
                         if ("plastic" in r["name"]) or r["name"].startswith("BS->")
                         or r["name"] in ("Ia-E->RG-E", "Ia-F->RG-F")}
    ORDER = ["CUT->RG-E (plastic)", "BS->RG-E", "BS->RG-F", "Ia-E->RG-E", "Ia-F->RG-F"]
    plastic = sorted([r for r in rows if r["name"] in plastic_names],
                     key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER))
    for r in rows:  # panel (b) values: learned where available, else as built
        L = learned.get(r["name"])
        r["wm_b"], r["ws_b"] = ((float(L.mean()), float(L.std())) if L is not None
                                else (r["wm"], r["ws"]))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.25, 1.25], hspace=0.45)

    # (a) weight histograms for plastic projections
    ga = gs[0].subgridspec(1, max(1, len(plastic)), wspace=0.3)
    for i, r in enumerate(plastic):
        ax = fig.add_subplot(ga[0, i])
        L = learned.get(r["name"])
        if L is None:
            ax.hist(r["w"], bins=40, color="#4477aa", alpha=0.85)
            ax.set_title(f"{r['name']}\n(n={r['n']}, μ={r['wm']:.1f}, σ={r['ws']:.1f} pA)", fontsize=11)
        else:
            hi = max(float(r["w"].max()), float(L.max()))
            bins = np.linspace(0, hi * 1.02, 60)
            ax.hist(r["w"], bins=bins, color="#9aa8b8", alpha=0.9, label="t = 0 (initial)")
            ax.hist(L, bins=bins, color="#c1440e", alpha=0.8, label=f"t = {learned_t:.0f} s (learned)")
            ax.set_title(f"{r['name'].replace(' (plastic)', '')}\n"
                         f"init μ={r['wm']:.1f}  →  learned μ={L.mean():.1f}, σ={L.std():.1f} pA", fontsize=11)
            if i == 0:
                ax.legend(fontsize=9, loc="upper center")
        ax.set_xlabel("weight (pA)");
        if i == 0:
            ax.set_ylabel("count")
            ax.text(-0.28, 1.12, "(a)", transform=ax.transAxes, fontsize=15,
                    fontweight="bold", va="bottom", ha="left")
        ax.grid(alpha=0.2)

    # (b) weight per projection — signed bar with the per-connection spread (s.d.)
    ax = fig.add_subplot(gs[1])
    sr = sorted(rows, key=lambda r: r["wm_b"])
    names = [r["name"] + (" [learned]" if r["name"] in learned else "") for r in sr]
    wm = [r["wm_b"] for r in sr]; ws = [r["ws_b"] for r in sr]
    colors = ["#cc3311" if w < 0 else "#228833" for w in wm]
    y = np.arange(len(sr))
    ax.barh(y, wm, xerr=ws, color=colors, alpha=0.85,
            error_kw=dict(ecolor="#555", elinewidth=0.8, capsize=2))
    for yi, r in zip(y, sr):  # initial mean of plastic projections, for reference
        if r["name"] in learned:
            ax.plot(r["wm"], yi, "o", mfc="white", mec="black", ms=5, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("synaptic weight (pA): mean ± per-connection s.d.  —  green = excitatory, red = inhibitory")
    ax.set_title("(b) Weight by projection — all projections heterogeneous (lognormal CV≈0.5); "
                 "the 6:1 InF→RG-E (−48) vs InE→RG-F (−8) Zhang asymmetry preserved"
                 + ("\n[learned] = after STDP (open circle = initial mean)" if learned else ""))
    ax.grid(alpha=0.2, axis="x")

    # (c) delay per projection (mean ± std), sorted by delay
    ax = fig.add_subplot(gs[2])
    dm = [r["dm"] for r in rows]; ds = [r["ds"] for r in rows]
    nm = [r["name"] for r in rows]
    y = np.arange(len(rows))
    ax.errorbar(dm, y, xerr=ds, fmt="o", color="#9933aa", capsize=3, markersize=6)
    ax.set_yticks(y); ax.set_yticklabels(nm, fontsize=10)
    ax.set_xlabel("conduction + synaptic delay (ms; mean ± across-connection s.d.)")
    ax.set_title("(c) Delay by projection (length_velocity rat preset + 0.2 ms jitter)")
    ax.grid(alpha=0.2, axis="x")

    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[connectivity] saved {args.out}")

    # CSV table
    csv = os.path.splitext(args.out)[0] + "_table.csv"
    with open(csv, "w") as fh:
        fh.write("projection,n_connections,weight_mean_pA,weight_std_pA,"
                 "learned_weight_mean_pA,learned_weight_std_pA,delay_mean_ms,delay_std_ms\n")
        for r in sorted(rows, key=lambda r: r["name"]):
            lm, ls = ((f"{r['wm_b']:.3f}", f"{r['ws_b']:.3f}") if r["name"] in learned else ("", ""))
            fh.write(f"{r['name']},{r['n']},{r['wm']:.3f},{r['ws']:.3f},{lm},{ls},{r['dm']:.3f},{r['ds']:.3f}\n")
    print(f"[connectivity] table {csv}")

    if args.tex:
        write_tex(rows, learned, args.tex, args.tex_caption_extra)
        print(f"[connectivity] LaTeX table {args.tex}")


GROUPS = [
    ("Descending / supraspinal drive", ["BS->RG-E", "BS->RG-F", "CUT->RG-E (plastic)", "CUT->InE",
                                         "base->RG-E", "base->RG-F"]),
    ("Sensory afferent $\\to$ RG", ["Ia-E->RG-E", "Ia-F->RG-F", "flexAff->RG-F", "flexAff->InF"]),
    ("Rhythm-generator reciprocal core", ["RG-E->InE", "RG-F->InF", "InE->RG-F", "InF->RG-E"]),
    ("Ia proprioceptive interneuron pathway", ["Ia-E->IaInt-E", "Ia-F->IaInt-F", "IaInt-E->M-F",
                                                "IaInt-F->M-E", "Ia-E->InE", "Ia-F->InF"]),
    ("Motor output", ["RG-E->M-E", "RG-F->M-F", "M-E->M-F", "M-F->M-E", "M-E->mus-E", "M-F->mus-F"]),
    ("Commissural (interlimb; abstracts V0v/V0d/V2a/In1)", ["commiss E (L->R)", "commiss F (L->R)"]),
]


def write_tex(rows, learned, path, caption_extra):
    by = {r["name"]: r for r in rows}
    tex = lambda nm: nm.replace("->", "$\\to$")
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \caption{Distribution of synaptic weights and conduction delays along every",
         r"  projection, grouped by the architecture (Fig.~\ref{fig:schematic}). Static",
         r"  projections carry imposed lognormal heterogeneity (CV$=0.5$, mean and sign",
         r"  preserved; Song 2005; Buzs\'aki \& Mizuseki 2014). Plastic projections",
         r"  (\textsuperscript{p}) are initialised lognormal ($\mu=3.5$\,pA, CV$=0.3$) and",
         r"  reported at their \emph{learned} values" + (f" {caption_extra}" if caption_extra else "") + ".",
         r"  Delays follow the rat \texttt{length\_velocity} preset with $0.2$\,ms jitter.",
         r"  $n$ = connection count, leg L. Muscle$\to$Ia spindle feedback is rate-coded,",
         r"  not a synapse (no entry).}",
         r"  \label{tab:delays}", r"  \begin{tabular}{l r r r}", r"    \toprule",
         r"    Projection & $n$ & Weight (pA) & Delay (ms) \\", r"    \midrule"]
    for title, names in GROUPS:
        present = [n for n in names if n in by]
        if not present:
            continue
        L.append(r"    \multicolumn{4}{l}{\textit{" + title + r"}}\\")
        for n in present:
            r = by[n]
            p = n in learned
            wm, ws = (r["wm_b"], r["ws_b"])
            L.append(f"    \\quad {tex(n)} & {r['n']} & ${wm:+.1f}\\pm{ws:.1f}$"
                     + (r"\textsuperscript{p}" if p else "") + f" & ${r['dm']:.2f}\\pm{r['ds']:.2f}$ \\\\")
        L.append(r"    \addlinespace[2pt]")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
