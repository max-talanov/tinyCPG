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
            for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                m = moving_average(w[f"{key}_mean"][:], win)
                s = moving_average(w[f"{key}_std"][:], win)
                plt.plot(times_ms, m, label=f"{key} mean ({args.smooth_sec:.1f}s MA)")
                plt.fill_between(times_ms, m - s, m + s, alpha=0.15)
            plt.xlabel("time (ms)"); plt.ylabel("weight (pA)")
            plt.title(f"STDP learning — inputs trend (leg {side})")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "w_inputs")

            # Motor synapses (smoothed)
            fig = plt.figure(figsize=(14, 6))
            m1 = moving_average(w["rge->me_mean"][:], win)
            m2 = moving_average(w["rgf->mf_mean"][:], win)
            plt.plot(times_ms, m1, label=f"rge->me mean ({args.smooth_sec:.1f}s MA)")
            plt.plot(times_ms, m2, label=f"rgf->mf mean ({args.smooth_sec:.1f}s MA)")
            plt.xlabel("time (ms)"); plt.ylabel("weight (pA)")
            plt.title(f"STDP learning — motor synapses trend (leg {side})")
            plt.legend(); plt.tight_layout()
            maybe_save(fig, "w_motor")

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