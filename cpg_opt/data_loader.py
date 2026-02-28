from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

SIGNAL_NAMES = ("mus_e", "mus_f", "act_e", "act_f", "force_e", "force_f", "len_e", "len_f")
LEGS = ("L", "R")


@dataclass
class CPGExperimentalData:
    signals: Dict[str, Dict[str, np.ndarray]]
    times_ms: np.ndarray
    duration_ms: float = 0.0
    sampling_rate_hz: float = 1000.0
    n_cycles: int = 0
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Union[str, Path], verbose: bool = True) -> "CPGExperimentalData":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return cls.from_hdf5(path, verbose=verbose)
        if suffix == ".npz":
            return cls.from_npz(path, verbose=verbose)
        if suffix == ".mat":
            return cls.from_matlab(path, verbose=verbose)
        if suffix in (".csv", ".tsv"):
            return cls.from_csv(path, verbose=verbose)
        if suffix == ".json":
            return cls.from_json(path, verbose=verbose)
        raise ValueError(f"Unsupported file format: {suffix}")

    @classmethod
    def from_hdf5(cls, path: Union[str, Path], verbose: bool = True) -> "CPGExperimentalData":
        import h5py
        path = Path(path)
        signals: Dict[str, Dict[str, np.ndarray]] = {}
        meta: Dict = {}
        with h5py.File(str(path), "r") as h5:
            times_ms = np.array(h5["times_ms"]) if "times_ms" in h5 else np.array([])
            for k in ("sim_ms", "dt_ms", "bs_osc_hz"):
                if k in h5.attrs:
                    meta[k] = float(h5.attrs[k])
            for side in LEGS:
                gname = f"leg_{side}"
                if gname not in h5:
                    continue
                g = h5[gname]
                leg_signals: Dict[str, np.ndarray] = {}
                for sig in SIGNAL_NAMES:
                    if sig in g:
                        leg_signals[sig] = np.array(g[sig])
                signals[gname] = leg_signals
        return cls._finalize(signals, times_ms, meta, path, verbose)

    @classmethod
    def from_npz(cls, path: Union[str, Path], verbose: bool = True) -> "CPGExperimentalData":
        path = Path(path)
        data = dict(np.load(str(path), allow_pickle=True))
        times_ms = data.pop("times_ms", np.array([]))
        signals = cls._parse_flat_keys(data)
        return cls._finalize(signals, times_ms, {}, path, verbose)

    @classmethod
    def from_csv(cls, path: Union[str, Path], verbose: bool = True) -> "CPGExperimentalData":
        path = Path(path)
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        raw = np.genfromtxt(str(path), delimiter=sep, names=True)
        times_ms = np.asarray(raw["times_ms"], dtype=float) if "times_ms" in raw.dtype.names else np.array([])
        flat = {n: np.asarray(raw[n], dtype=float) for n in raw.dtype.names if n != "times_ms"}
        signals = cls._parse_flat_keys(flat)
        return cls._finalize(signals, times_ms, {}, path, verbose)

    @classmethod
    def from_matlab(cls, path: Union[str, Path], verbose: bool = True) -> "CPGExperimentalData":
        from scipy.io import loadmat
        path = Path(path)
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=True)
        times_ms = np.asarray(mat["times_ms"], dtype=float) if "times_ms" in mat else np.array([])
        flat = {k: np.asarray(v, dtype=float).ravel() for k, v in mat.items() if not k.startswith("_") and k != "times_ms"}
        signals = cls._parse_flat_keys(flat)
        return cls._finalize(signals, times_ms, {}, path, verbose)

    @classmethod
    def from_json(cls, path: Union[str, Path], verbose: bool = True) -> "CPGExperimentalData":
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)
        times_ms = np.asarray(raw.get("times_ms", []), dtype=float)
        flat = {k: np.asarray(v, dtype=float) for k, v in raw.items() if k != "times_ms"}
        signals = cls._parse_flat_keys(flat)
        return cls._finalize(signals, times_ms, {}, path, verbose)

    @staticmethod
    def _parse_flat_keys(flat: Dict) -> Dict[str, Dict[str, np.ndarray]]:
        signals: Dict[str, Dict[str, np.ndarray]] = {}
        for key, arr in flat.items():
            for side in LEGS:
                prefix = f"leg_{side}_"
                if key.startswith(prefix):
                    signals.setdefault(f"leg_{side}", {})[key[len(prefix):]] = np.asarray(arr, dtype=float)
        return signals

    @classmethod
    def _finalize(cls, signals, times_ms, meta, path, verbose):
        duration_ms = float(meta.get("sim_ms", times_ms[-1] if len(times_ms) > 0 else 0))
        dt_ms = float(meta.get("dt_ms", np.median(np.diff(times_ms)) if len(times_ms) > 1 else 10.0))
        obj = cls(signals=signals, times_ms=times_ms, duration_ms=duration_ms,
                  sampling_rate_hz=1000.0 / max(1e-6, dt_ms), metadata=meta)
        obj.n_cycles = obj._estimate_n_cycles()
        if verbose:
            obj._print_summary(path)
        return obj

    def get_signal(self, side: str, name: str) -> Optional[np.ndarray]:
        return self.signals.get(f"leg_{side}", {}).get(name)

    def available_signals(self) -> Dict[str, List[str]]:
        return {k: sorted(v.keys()) for k, v in self.signals.items()}

    def dt_ms(self) -> float:
        if len(self.times_ms) > 1:
            return float(np.median(np.diff(self.times_ms)))
        return 1000.0 / max(1e-6, self.sampling_rate_hz)

    def _estimate_n_cycles(self) -> int:
        from .objective import _burst_intervals
        for sig_name in ("force_e", "act_e", "mus_e"):
            arr = self.get_signal("L", sig_name)
            if arr is not None and len(arr) > 10:
                ibis = _burst_intervals(arr, self.dt_ms() / 1000.0)
                return len(ibis) + 1 if len(ibis) > 0 else 0
        return 0

    def _print_summary(self, path: Path):
        avail = self.available_signals()
        print(f"\n{'=' * 50}")
        print(f"Loaded CPG data from {path.name}")
        print(f"  Duration: {self.duration_ms:.1f} ms  |  Samples: {len(self.times_ms)}  |  Cycles: ~{self.n_cycles}")
        for leg, sigs in avail.items():
            print(f"  {leg}: {', '.join(sigs)}")
        print(f"{'=' * 50}\n")

    def summary(self) -> str:
        lines = [f"Duration: {self.duration_ms:.1f} ms, Samples: {len(self.times_ms)}, Cycles: ~{self.n_cycles}"]
        for leg, sigs in self.available_signals().items():
            lines.append(f"  {leg}: {', '.join(sigs)}")
        return "\n".join(lines)
