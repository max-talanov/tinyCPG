from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_loader import CPGExperimentalData


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < win:
        return x
    return np.convolve(x, np.ones(win) / win, mode="same")


def _dominant_frequency(signal: np.ndarray, dt_s: float) -> float:
    if len(signal) < 8:
        return 0.0
    sig = signal - np.mean(signal)
    freqs = np.fft.rfftfreq(len(sig), d=dt_s)
    power = np.abs(np.fft.rfft(sig)) ** 2
    if len(power) > 1:
        power[0] = 0.0
    return float(freqs[np.argmax(power)])


def _phase_offset_rad(sig_a: np.ndarray, sig_b: np.ndarray, dt_s: float) -> float:
    if len(sig_a) < 8 or len(sig_b) < 8:
        return 0.0
    a = sig_a - np.mean(sig_a)
    b = sig_b - np.mean(sig_b)
    corr = np.correlate(a, b, mode="full")
    corr /= max(1e-30, np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
    mid = len(a) - 1
    freq = _dominant_frequency(sig_a, dt_s)
    if freq <= 0:
        return 0.0
    period_samples = int(round(1.0 / freq / dt_s))
    lo = max(0, mid - period_samples)
    hi = min(len(corr), mid + period_samples + 1)
    lag_samples = lo + int(np.argmax(corr[lo:hi])) - mid
    return float((2.0 * np.pi * lag_samples * dt_s * freq) % (2.0 * np.pi))


def _burst_intervals(signal: np.ndarray, dt_s: float, threshold_frac: float = 0.3) -> np.ndarray:
    if len(signal) < 4:
        return np.array([])
    thresh = np.min(signal) + threshold_frac * (np.max(signal) - np.min(signal))
    onsets = np.where(np.diff((signal > thresh).astype(int)) == 1)[0]
    if len(onsets) < 2:
        return np.array([])
    return np.diff(onsets) * dt_s


def _normalize_cycle_envelope(signal: np.ndarray, dt_s: float) -> np.ndarray:
    if len(signal) < 8:
        return np.array([])
    ibis = _burst_intervals(signal, dt_s)
    if len(ibis) < 1:
        return np.array([])
    thresh = np.min(signal) + 0.3 * (np.max(signal) - np.min(signal))
    onsets = np.where(np.diff((signal > thresh).astype(int)) == 1)[0]
    if len(onsets) < 2:
        return np.array([])

    n_bins = 100
    envelopes = []
    for i in range(len(onsets) - 1):
        seg = signal[onsets[i]:onsets[i + 1]]
        if len(seg) < 3:
            continue
        resampled = np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, len(seg)), seg)
        rng = float(np.max(resampled) - np.min(resampled))
        if rng > 1e-10:
            resampled = (resampled - np.min(resampled)) / rng
        envelopes.append(resampled)
    if not envelopes:
        return np.array([])
    return np.mean(envelopes, axis=0)


# ── Rule-based objective ─────────────────────────────────────────────

class CPGObjective:
    DEFAULT_WEIGHTS = {
        "ef_alternation": 2.0,
        "lr_antiphase": 1.5,
        "rhythm_regularity": 1.0,
        "frequency_match": 1.5,
        "activity_level": 3.0,
        "weight_health": 0.5,
    }

    def __init__(self, target_freq_hz: float = 1.0, weights: Optional[Dict[str, float]] = None,
                 wmax: float = 120.0, smooth_win_ms: float = 200.0):
        self.target_freq_hz = target_freq_hz
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.wmax = wmax
        self.smooth_win_ms = smooth_win_ms

    def __call__(self, data: Dict) -> float:
        return self.compute(data)

    def compute(self, data: Dict) -> float:
        scores = self.compute_all(data)
        return sum(self.weights.get(k, 0.0) * v for k, v in scores.items())

    def compute_all(self, data: Dict) -> Dict[str, float]:
        return {
            "ef_alternation": self._ef_alternation(data),
            "lr_antiphase": self._lr_antiphase(data),
            "rhythm_regularity": self._rhythm_regularity(data),
            "frequency_match": self._frequency_match(data),
            "activity_level": self._activity_level(data),
            "weight_health": self._weight_health(data),
        }

    def _dt_s(self, data: Dict) -> float:
        times = data.get("times_ms")
        if times is not None and len(times) > 1:
            return float(np.median(np.diff(times))) / 1000.0
        return data.get("dt_ms", 100.0) / 1000.0

    def _win(self, data: Dict) -> int:
        return max(1, int(round(self.smooth_win_ms / (self._dt_s(data) * 1000.0))))

    def _leg(self, data: Dict, side: str, key: str) -> np.ndarray:
        return np.asarray(data.get(f"leg_{side}", {}).get(key, np.array([])), dtype=float)

    def _discard_transient(self, data: Dict, frac: float = 0.15) -> Tuple[int, int]:
        n = len(data.get("times_ms", np.array([])))
        return int(n * frac), n

    def _ef_alternation(self, data: Dict) -> float:
        w = self._win(data)
        s, e = self._discard_transient(data)
        total, n_legs = 0.0, 0
        for side in ("L", "R"):
            me = _smooth(self._leg(data, side, "mus_e"), w)[s:e]
            mf = _smooth(self._leg(data, side, "mus_f"), w)[s:e]
            if len(me) < 8 or len(mf) < 8:
                total += 2.0; n_legs += 1; continue
            me_z, mf_z = me - np.mean(me), mf - np.mean(mf)
            norm = max(1e-30, np.sqrt(np.sum(me_z ** 2) * np.sum(mf_z ** 2)))
            total += 1.0 + float(np.sum(me_z * mf_z)) / norm
            n_legs += 1
        return total / max(1, n_legs)

    def _lr_antiphase(self, data: Dict) -> float:
        dt_s = self._dt_s(data)
        w = self._win(data)
        s, e = self._discard_transient(data)
        le = _smooth(self._leg(data, "L", "mus_e"), w)[s:e]
        re = _smooth(self._leg(data, "R", "mus_e"), w)[s:e]
        if len(le) < 8 or len(re) < 8:
            return 1.0
        return float(abs(_phase_offset_rad(le, re, dt_s) - np.pi) / np.pi)

    def _rhythm_regularity(self, data: Dict) -> float:
        dt_s = self._dt_s(data)
        w = self._win(data)
        s, e = self._discard_transient(data)
        cvs = []
        for side in ("L", "R"):
            fe = _smooth(self._leg(data, side, "force_e"), w)[s:e]
            if len(fe) < 8:
                cvs.append(2.0); continue
            ibis = _burst_intervals(fe, dt_s)
            if len(ibis) < 2:
                cvs.append(2.0); continue
            cvs.append(min(float(np.std(ibis) / max(1e-9, np.mean(ibis))), 2.0))
        return float(np.mean(cvs))

    def _frequency_match(self, data: Dict) -> float:
        dt_s = self._dt_s(data)
        w = self._win(data)
        s, e = self._discard_transient(data)
        freqs = []
        for side in ("L", "R"):
            fe = _smooth(self._leg(data, side, "force_e"), w)[s:e]
            freqs.append(_dominant_frequency(fe, dt_s) if len(fe) >= 8 else 0.0)
        if self.target_freq_hz <= 0:
            return 0.0
        err = ((float(np.mean(freqs)) - self.target_freq_hz) / self.target_freq_hz) ** 2
        return min(float(err), 4.0)

    def _activity_level(self, data: Dict) -> float:
        s, e = self._discard_transient(data)
        penalty = 0.0
        for side in ("L", "R"):
            for ch in ("act_e", "act_f"):
                a = self._leg(data, side, ch)[s:e]
                if len(a) == 0:
                    penalty += 5.0; continue
                peak, mean_a = float(np.max(a)), float(np.mean(a))
                if peak < 0.01:
                    penalty += 5.0
                elif peak < 0.05:
                    penalty += 2.0
                if mean_a > 1.0:
                    penalty += 2.0 * (mean_a - 1.0)
        return penalty

    def _weight_health(self, data: Dict) -> float:
        eps_lo, eps_hi = 0.5, self.wmax - 0.5
        bad = []
        for side in ("L", "R"):
            wg = data.get(f"leg_{side}", {}).get("weights", {})
            for proj in ("cut->rge", "bs->rge", "bs->rgf"):
                arr = np.asarray(wg.get(f"{proj}_mean", np.array([])), dtype=float)
                if len(arr) == 0:
                    continue
                v = float(arr[-1])
                bad.append(1.0 if (np.isnan(v) or v < eps_lo or v > eps_hi) else 0.0)
        return float(np.mean(bad)) * 2.0 if bad else 1.0


class CPGMultiObjective:
    def __init__(self, target_freq_hz: float = 1.0, wmax: float = 120.0):
        self.single = CPGObjective(target_freq_hz=target_freq_hz, wmax=wmax)

    def __call__(self, data: Dict) -> np.ndarray:
        s = self.single.compute_all(data)
        return np.array([s[k] for k in self.objective_names])

    @property
    def n_objectives(self) -> int:
        return 6

    @property
    def objective_names(self):
        return ["ef_alternation", "lr_antiphase", "rhythm_regularity",
                "frequency_match", "activity_level", "weight_health"]


# ── Supervised objective ─────────────────────────────────────────────

class SupervisedCPGObjective:
    COMPARED_SIGNALS = ("mus_e", "mus_f", "force_e", "force_f")

    DEFAULT_WEIGHTS = {
        "envelope_ks": 1.5,
        "ibi_ks": 1.0,
        "correlation": 1.0,
        "duty_cycle": 0.5,
        "activity_floor": 3.0,
    }

    def __init__(self, experimental_data: "CPGExperimentalData",
                 weights: Optional[Dict[str, float]] = None, smooth_win_ms: float = 200.0):
        self.data = experimental_data
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.smooth_win_ms = smooth_win_ms
        self._target_envelopes = self._precompute_envelopes()
        self._target_ibis = self._precompute_ibis()
        self._target_duty_cycles = self._precompute_duty_cycles()

    def _exp_dt_s(self) -> float:
        return self.data.dt_ms() / 1000.0

    def _precompute_envelopes(self) -> Dict[str, np.ndarray]:
        dt_s = self._exp_dt_s()
        out: Dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            for sig in self.COMPARED_SIGNALS:
                arr = self.data.get_signal(side, sig)
                if arr is not None and len(arr) > 10:
                    env = _normalize_cycle_envelope(arr, dt_s)
                    if len(env) > 0:
                        out[f"{side}_{sig}"] = env
        return out

    def _precompute_ibis(self) -> Dict[str, np.ndarray]:
        dt_s = self._exp_dt_s()
        out: Dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            for sig in ("force_e", "mus_e"):
                arr = self.data.get_signal(side, sig)
                if arr is not None and len(arr) > 10:
                    ibis = _burst_intervals(arr, dt_s)
                    if len(ibis) >= 2:
                        out[f"{side}_{sig}"] = ibis
        return out

    def _precompute_duty_cycles(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for side in ("L", "R"):
            arr = self.data.get_signal(side, "mus_e")
            if arr is None:
                arr = self.data.get_signal(side, "force_e")
            if arr is not None and len(arr) > 10:
                thresh = np.min(arr) + 0.3 * (np.max(arr) - np.min(arr))
                out[f"{side}_e"] = float(np.mean(arr > thresh))
        return out

    def __call__(self, data: Dict) -> float:
        return self.compute(data)

    def compute(self, data: Dict) -> float:
        scores = self.compute_all(data)
        return sum(self.weights.get(k, 0.0) * v for k, v in scores.items())

    def compute_all(self, data: Dict) -> Dict[str, float]:
        act = self._activity_floor(data)
        if act > 0:
            return {"envelope_ks": 100.0, "ibi_ks": 100.0, "correlation": 1.0,
                    "duty_cycle": 1.0, "activity_floor": act}
        return {
            "envelope_ks": self._envelope_ks(data),
            "ibi_ks": self._ibi_ks(data),
            "correlation": self._correlation_loss(data),
            "duty_cycle": self._duty_cycle_loss(data),
            "activity_floor": 0.0,
        }

    def _sim_dt_s(self, data: Dict) -> float:
        times = data.get("times_ms")
        if times is not None and len(times) > 1:
            return float(np.median(np.diff(times))) / 1000.0
        return data.get("dt_ms", 100.0) / 1000.0

    def _sim_leg(self, data: Dict, side: str, key: str) -> np.ndarray:
        return np.asarray(data.get(f"leg_{side}", {}).get(key, np.array([])), dtype=float)

    def _activity_floor(self, data: Dict) -> float:
        penalty = 0.0
        for side in ("L", "R"):
            for ch in ("mus_e", "mus_f"):
                arr = self._sim_leg(data, side, ch)
                if len(arr) == 0 or float(np.max(arr)) < 1e-6:
                    penalty += 10.0
        return penalty

    def _envelope_ks(self, data: Dict) -> float:
        from scipy.stats import ks_2samp
        dt_s = self._sim_dt_s(data)
        losses: List[float] = []
        for key, target_env in self._target_envelopes.items():
            side, sig = key.split("_", 1)
            sim_arr = self._sim_leg(data, side, sig)
            if len(sim_arr) < 10:
                losses.append(1.0); continue
            sim_env = _normalize_cycle_envelope(sim_arr, dt_s)
            if len(sim_env) == 0 or len(target_env) == 0:
                losses.append(1.0); continue
            try:
                stat, _ = ks_2samp(sim_env, target_env)
                losses.append(float(stat))
            except Exception:
                losses.append(1.0)
        return float(np.mean(losses)) * 100.0 if losses else 100.0

    def _ibi_ks(self, data: Dict) -> float:
        from scipy.stats import ks_2samp
        dt_s = self._sim_dt_s(data)
        losses: List[float] = []
        for key, target_ibis in self._target_ibis.items():
            side, sig = key.split("_", 1)
            sim_arr = self._sim_leg(data, side, sig)
            if len(sim_arr) < 10:
                losses.append(1.0); continue
            sim_ibis = _burst_intervals(sim_arr, dt_s)
            if len(sim_ibis) < 2 or len(target_ibis) < 2:
                losses.append(1.0); continue
            try:
                stat, _ = ks_2samp(sim_ibis, target_ibis)
                losses.append(float(stat))
            except Exception:
                losses.append(1.0)
        return float(np.mean(losses)) * 100.0 if losses else 100.0

    def _correlation_loss(self, data: Dict) -> float:
        sim_dt_s = self._sim_dt_s(data)
        win = max(1, int(round(self.smooth_win_ms / (sim_dt_s * 1000.0))))
        corrs: List[float] = []
        for side in ("L", "R"):
            for sig in ("mus_e", "mus_f"):
                target = self.data.get_signal(side, sig)
                sim = self._sim_leg(data, side, sig)
                if target is None or len(target) < 8 or len(sim) < 8:
                    corrs.append(0.0); continue
                sim_s = _smooth(sim, win)
                target_r = _smooth(np.interp(
                    np.linspace(0, 1, len(sim_s)),
                    np.linspace(0, 1, len(target)), target), win)
                if float(np.std(sim_s)) < 1e-10 or float(np.std(target_r)) < 1e-10:
                    corrs.append(0.0); continue
                r = float(np.corrcoef(sim_s, target_r)[0, 1])
                corrs.append(max(0.0, r) if not np.isnan(r) else 0.0)
        return 1.0 - (float(np.mean(corrs)) if corrs else 0.0)

    def _duty_cycle_loss(self, data: Dict) -> float:
        losses: List[float] = []
        for key, target_dc in self._target_duty_cycles.items():
            side = key.split("_")[0]
            sim = self._sim_leg(data, side, "mus_e")
            if len(sim) < 10:
                sim = self._sim_leg(data, side, "force_e")
            if len(sim) < 10:
                losses.append(1.0); continue
            thresh = np.min(sim) + 0.3 * (np.max(sim) - np.min(sim))
            losses.append((float(np.mean(sim > thresh)) - target_dc) ** 2)
        return float(np.mean(losses)) if losses else 1.0


# ── Hybrid objective ─────────────────────────────────────────────────

class HybridCPGObjective:
    def __init__(self, experimental_data: Optional["CPGExperimentalData"] = None,
                 supervised_weight: float = 0.5, target_freq_hz: float = 1.0, wmax: float = 120.0):
        self.rule_based = CPGObjective(target_freq_hz=target_freq_hz, wmax=wmax)
        self.supervised_weight = float(np.clip(supervised_weight, 0.0, 1.0))
        if experimental_data is not None:
            self.supervised: Optional[SupervisedCPGObjective] = SupervisedCPGObjective(
                experimental_data=experimental_data)
        else:
            self.supervised = None
            self.supervised_weight = 0.0

    def __call__(self, data: Dict) -> float:
        return self.compute(data)

    def compute(self, data: Dict) -> float:
        rule_loss = self.rule_based.compute(data)
        if self.supervised is not None and self.supervised_weight > 0:
            sup_loss = self.supervised.compute(data)
            return (1.0 - self.supervised_weight) * rule_loss + self.supervised_weight * sup_loss
        return rule_loss

    def compute_all(self, data: Dict) -> Dict[str, float]:
        components: Dict[str, float] = {}
        for name, value in self.rule_based.compute_all(data).items():
            components[f"rule_{name}"] = value
        if self.supervised is not None:
            for name, value in self.supervised.compute_all(data).items():
                components[f"sup_{name}"] = value
        components["total"] = self.compute(data)
        return components


# ── Factory ──────────────────────────────────────────────────────────

def create_cpg_objective(
    objective_type: str = "rule-based",
    experimental_data: Optional["CPGExperimentalData"] = None,
    supervised_weight: float = 0.5,
    target_freq_hz: float = 1.0,
    wmax: float = 120.0,
):
    if objective_type == "rule-based":
        return CPGObjective(target_freq_hz=target_freq_hz, wmax=wmax)
    elif objective_type == "supervised":
        if experimental_data is None:
            raise ValueError("supervised objective requires experimental_data")
        return SupervisedCPGObjective(experimental_data=experimental_data)
    elif objective_type == "hybrid":
        return HybridCPGObjective(experimental_data=experimental_data,
                                  supervised_weight=supervised_weight,
                                  target_freq_hz=target_freq_hz, wmax=wmax)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
