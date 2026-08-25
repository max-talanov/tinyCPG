#!/usr/bin/env python3
"""
cpg_cutforce_diagnostics.py
Pass/fail diagnostic for --cut-trigger force sweep outputs.

Correlation alone (corr(Force-E,Force-F)) doesn't tell you whether a run is
genuinely force-threshold-driven or just the --cut-max-stance-ms/
--cut-max-swing-ms failsafe timer wearing a force-shaped costume: a Force-E
trace can look clean while every single stance-swing transition lands exactly
on the cap. This script reports both the correlation metrics AND the fraction
of stance bouts that hit the cap, which is the number that actually tells you
the mechanism is working.

Ground truth vs. reconstruction: runs written after the MOD_CUT_FORCE_TRIGGER
`cut_on` logging was added carry an exact per-chunk leg_{L,R}/cut_on array
(1.0 = CUT on / stance, 0.0 = off / swing), so bout durations and the at-cap
fraction are exact. Older files (e.g. the first sweep round, 2026-08-25) don't
have it, so this script falls back to reconstructing bouts from force_e
crossing a per-file-adaptive threshold -- flagged explicitly in the output
column ('exact' vs 'recon') because that reconstruction is threshold-sensitive
and was confirmed to give different at-cap verdicts on the same file depending
on the threshold chosen (a fixed absolute threshold also silently misdetects
bouts across runs with very different force amplitudes). Trust 'exact' rows;
treat 'recon' rows as indicative only.

Usage:
  python3 cpg_cutforce_diagnostics.py results/2026-08-25/*.h5
  python3 cpg_cutforce_diagnostics.py --stance-thresh-frac 0.4 results/2026-*/*.h5
"""

import argparse
import sys

import h5py
import numpy as np


def bouts_from_cut_on(t_ms: np.ndarray, cut_on: np.ndarray, cap_ms: float, cap_tol_ms: float):
    """Exact stance-bout durations from the ground-truth cut_on (0/1) array."""
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
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan, frac_at_cap=np.nan, exact=True)
    # The cap check only runs at --rate-update-ms granularity, so the measured
    # transition lands at the next tick at or after cap_ms, not exactly at it
    # (cap_tol_ms should be >= --rate-update-ms of the run; default 110ms
    # covers the 100ms ticks used by the sweep scripts).
    at_cap = np.mean((durs >= cap_ms) & (durs <= cap_ms + cap_tol_ms)) if cap_ms > 0 else np.nan
    return dict(n=int(durs.size), mean=float(durs.mean()), std=float(durs.std()),
                min=float(durs.min()), max=float(durs.max()), frac_at_cap=float(at_cap), exact=True)


def bouts_from_force_recon(t_ms: np.ndarray, force_e: np.ndarray, thresh_frac: float, cap_ms: float):
    """Fallback: reconstruct stance bouts from force_e crossing thresh_frac of
    this file's own peak. Threshold-sensitive -- see module docstring."""
    thresh = thresh_frac * float(force_e.max())
    stance = force_e > thresh
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
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan, frac_at_cap=np.nan, exact=False)
    at_cap = np.mean(durs >= (cap_ms - 50.0)) if cap_ms > 0 else np.nan
    return dict(n=int(durs.size), mean=float(durs.mean()), std=float(durs.std()),
                min=float(durs.min()), max=float(durs.max()), frac_at_cap=float(at_cap), exact=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="cpg_cutforce_*.h5 output files")
    ap.add_argument("--stance-thresh-frac", type=float, default=0.4,
                     help="fallback-only: fraction of this file's own force_e peak "
                          "above which a leg is counted as 'in stance' when the "
                          "ground-truth cut_on array isn't present (default 0.4)")
    ap.add_argument("--cap-tol-ms", type=float, default=110.0,
                     help="tolerance (ms) above cap_ms still counted as 'at cap' -- "
                          "should be >= the run's --rate-update-ms, since the cap "
                          "check only fires at that granularity (default 110)")
    args = ap.parse_args()

    header = (f"{'file':45s} {'corrL':>7s} {'corrR':>7s} {'corrLR':>7s} "
              f"{'mode':>6s} {'capL':>5s} {'durL':>13s} {'atCapL':>7s} "
              f"{'durR':>13s} {'atCapR':>7s}")
    print(header)
    print("-" * len(header))

    for fn in args.files:
        with h5py.File(fn, "r") as f:
            t = np.asarray(f["times_ms"])
            cap_stance = float(f.attrs.get("cut_max_stance_ms", 0.0))

            fe = {}
            corr = {}
            bouts = {}
            for side in ("L", "R"):
                fe[side] = np.asarray(f[f"leg_{side}/force_e"])
                ff = np.asarray(f[f"leg_{side}/force_f"])
                corr[side] = float(np.corrcoef(fe[side], ff)[0, 1])
                cut_on_key = f"leg_{side}/cut_on"
                if cut_on_key in f and np.asarray(f[cut_on_key]).size > 0:
                    bouts[side] = bouts_from_cut_on(t, np.asarray(f[cut_on_key]), cap_stance, args.cap_tol_ms)
                else:
                    bouts[side] = bouts_from_force_recon(t, fe[side], args.stance_thresh_frac, cap_stance)

            corr_lr = float(np.corrcoef(fe["L"], fe["R"])[0, 1])
            mode = "exact" if bouts["L"]["exact"] else "recon"

            def fmt_dur(b):
                if b["n"] == 0:
                    return "n/a"
                return f"{b['mean']:.0f}±{b['std']:.0f}ms"

            print(f"{fn:45s} {corr['L']:7.3f} {corr['R']:7.3f} {corr_lr:7.3f} "
                  f"{mode:>6s} {cap_stance:5.0f} {fmt_dur(bouts['L']):>13s} {bouts['L']['frac_at_cap']:7.2f} "
                  f"{fmt_dur(bouts['R']):>13s} {bouts['R']['frac_at_cap']:7.2f}")

    print("\nfrac_at_cap close to 1.0 means the failsafe timer is doing the work, not "
          "a genuine force-threshold crossing. 'recon' rows (no cut_on array in the "
          "file) are indicative only -- see module docstring.", file=sys.stderr)


if __name__ == "__main__":
    main()
