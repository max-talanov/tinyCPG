#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpg_plot_from_hdf5.py
Read the HDF5 produced by cpg_2legs_nest_to_hdf5.py and generate plots locally.

Example:
  python3 cpg_plot_from_hdf5.py --in cpg_run.h5 --smooth-sec 1.0
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def _fill_nans_forward(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if not np.isnan(x).any():
        return x
    idx = np.where(~np.isnan(x))[0]
    if idx.size == 0:
        return x
    x[:idx[0]] = x[idx[0]]
    for i in range(idx.size - 1):
        a, b = idx[i], idx[i + 1]
        if b > a + 1:
            x[a + 1:b] = x[a]
    x[idx[-1] + 1:] = x[idx[-1]]
    return x


def moving_average(x, win: int):
    x = _fill_nans_forward(np.asarray(x, dtype=float))
    win = int(max(1, win))
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input HDF5 file from HPC")
    ap.add_argument("--smooth-sec", type=float, default=1.0, help="Smoothing window for weights (seconds)")
    ap.add_argument("--save-prefix", type=str, default="", help="If set, save PNGs as <prefix>_legX_*.png")
    ap.add_argument("--show", action="store_true", help="Show plots interactively (default: false if saving)")
    args = ap.parse_args()

    with h5py.File(args.inp, "r") as h5:
        times_ms = np.array(h5["times_ms"])
        dt_ms = float(h5.attrs.get("dt_ms", np.median(np.diff(times_ms)) if len(times_ms) > 1 else 10.0))
        if "weights_times_ms" in h5:
            times_w = np.array(h5["weights_times_ms"])
        else:
            times_w = times_ms

        print("=== Run metadata ===")
        for k in ["created_utc", "nest_version", "sim_ms", "dt_ms", "phases", "bs_osc_hz", "local_threads", "mpi_processes"]:
            if k in h5.attrs:
                print(f"{k}: {h5.attrs[k]}")
        if "stats" in h5:
            s = h5["stats"].attrs
            print("--- stats ---")
            for k in sorted(s.keys()):
                print(f"{k}: {s[k]}")
        print("====================")

        win = max(1, int((args.smooth_sec * 1000.0) / dt_ms))
        print(f"[Plot] weight smoothing: {args.smooth_sec:.3f}s -> {win} samples")

        legs = sorted([k.split("_", 1)[1] for k in h5.keys() if k.startswith("leg_")])

        for side in legs:
            g = h5[f"leg_{side}"]
            w = g["weights"]

            # Debug/robustness: show what weight datasets exist for this leg
            try:
                print(f"[Plot] leg {side} weight datasets: {list(w.keys())}")
            except Exception:
                pass

            def maybe_save(fig, name):
                if args.save_prefix:
                    fig.savefig(f"{args.save_prefix}_leg{side}_{name}.png", dpi=160)

            # BS drive
            fig = plt.figure(figsize=(14, 5))
            plt.plot(times_ms, g["bs_e"][:], label="BS E")
            plt.plot(times_ms, g["bs_f"][:], label="BS F")
            plt.xlabel("time (ms)"); plt.ylabel("Hz")
            plt.title(f"Brainstem drive — leg {side}")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "bs")

            # Inputs learning (smoothed)
            fig = plt.figure(figsize=(14, 7))

            # Discover which weight series exist in this file for this leg
            available_keys = sorted([name[:-5] for name in w.keys() if name.endswith("_mean")])

            preferred = ["cut->rge", "bs->rge", "bs->rgf"]
            keys_to_plot = [k for k in preferred if f"{k}_mean" in w]
            if not keys_to_plot:
                # Plot any existing series except deprecated motor projections
                keys_to_plot = [k for k in available_keys if k not in ("rge->me", "rgf->mf")]

            if not keys_to_plot:
                plt.text(0.5, 0.5, f"No weight trends found for leg {side}", ha="center", va="center", transform=plt.gca().transAxes)
            else:
                # Smooth in weight-time coordinates
                dtw = float(np.median(np.diff(times_w)) if len(times_w) > 1 else dt_ms)
                win_w = max(1, int(round((args.smooth_sec * 1000.0) / max(1e-6, dtw))))

                for key in keys_to_plot:
                    try:
                        m = moving_average(w[f"{key}_mean"][:], win_w)
                    except KeyError:
                        # Some older files only store a subset of expected series
                        print(f"[Plot] WARNING: missing {key}_mean in leg {side}; skipping")
                        continue

                    if f"{key}_std" in w:
                        s = moving_average(w[f"{key}_std"][:], win_w)
                    else:
                        s = np.zeros_like(m)

                    plt.plot(times_w, m, label=f"{key} mean ({args.smooth_sec:.1f}s MA)")
                    plt.fill_between(times_w, m - s, m + s, alpha=0.15)

            plt.xlabel("time (ms)"); plt.ylabel("weight (pA)")
            plt.title(f"STDP learning — inputs trend (leg {side})")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "w_inputs")

            # Muscle
            fig = plt.figure(figsize=(14, 5))
            plt.plot(times_ms, g["mus_e"][:], label="mus-E rate")
            plt.plot(times_ms, g["mus_f"][:], label="mus-F rate")
            plt.xlabel("time (ms)"); plt.ylabel("Hz/neuron")
            plt.title(f"Muscle relay rates — leg {side}")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "mus_rate")

            # Activation
            fig = plt.figure(figsize=(14, 5))
            plt.plot(times_ms, g["act_e"][:], label="Activation E")
            plt.plot(times_ms, g["act_f"][:], label="Activation F")
            plt.xlabel("time (ms)"); plt.ylabel("a.u.")
            plt.title(f"Activation proxy — leg {side}")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "activation")

            # Force
            fig = plt.figure(figsize=(14, 5))
            plt.plot(times_ms, g["force_e"][:], label="Force E")
            plt.plot(times_ms, g["force_f"][:], label="Force F")
            plt.xlabel("time (ms)"); plt.ylabel("force (a.u.)")
            plt.title(f"Force proxy — leg {side}")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "force")

            # Length
            fig = plt.figure(figsize=(14, 5))
            plt.plot(times_ms, g["len_e"][:], label="Length E")
            plt.plot(times_ms, g["len_f"][:], label="Length F")
            plt.axhline(1.0, linestyle="--", linewidth=1)
            plt.xlabel("time (ms)"); plt.ylabel("length (a.u.)")
            plt.title(f"Length proxy — leg {side}")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "length")

            # Ia
            fig = plt.figure(figsize=(14, 5))
            plt.plot(times_ms, g["ia_e"][:], label="Ia-E rate")
            plt.plot(times_ms, g["ia_f"][:], label="Ia-F rate")
            plt.xlabel("time (ms)"); plt.ylabel("Hz")
            plt.title(f"Ia generator rates — leg {side}")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "ia")

        if args.show or not args.save_prefix:
            plt.show()


if __name__ == "__main__":
    main()