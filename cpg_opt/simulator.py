import json
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

CPG_SCRIPT = str(Path(__file__).resolve().parent.parent / "cpg_2legs_fast.py")


class CPGSimConfig:
    def __init__(
        self,
        sim_ms: float = 10000.0,
        dt_ms: float = 10.0,
        threads: int = 4,
        resolution_ms: float = 0.5,
        long_run: bool = True,
        save_weights: str = "final",
        nest_verbosity: str = "M_ERROR",
        work_dir: Optional[str] = None,
        python_bin: str = sys.executable,
        extra_args: Optional[list] = None,
    ):
        self.sim_ms = sim_ms
        self.dt_ms = dt_ms
        self.threads = threads
        self.resolution_ms = resolution_ms
        self.long_run = long_run
        self.save_weights = save_weights
        self.nest_verbosity = nest_verbosity
        self.work_dir = work_dir or tempfile.gettempdir()
        self.python_bin = python_bin
        self.extra_args = extra_args or []


def run_cpg_simulation(
    params: Dict[str, float],
    config: Optional[CPGSimConfig] = None,
    tag: str = "",
    cleanup: bool = True,
) -> Dict:
    if h5py is None:
        raise ImportError("h5py is required")

    config = config or CPGSimConfig()
    uid = tag or uuid.uuid4().hex[:8]
    work = Path(config.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    params_path = work / f"cpg_params_{uid}.json"
    h5_path = work / f"cpg_out_{uid}.h5"

    try:
        with open(params_path, "w") as f:
            json.dump(params, f)

        cmd = [
            config.python_bin, "-u", CPG_SCRIPT,
            "--out", str(h5_path),
            "--sim-ms", str(config.sim_ms),
            "--dt-ms", str(config.dt_ms),
            "--threads", str(config.threads),
            "--resolution-ms", str(config.resolution_ms),
            "--save-weights", config.save_weights,
            "--nest-verbosity", config.nest_verbosity,
            "--params-json", str(params_path),
        ]
        if config.long_run:
            cmd.append("--long-run")
        cmd.extend(config.extra_args)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            raise RuntimeError(
                f"cpg_2legs_fast.py failed (rc={result.returncode}):\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        if not h5_path.exists():
            raise FileNotFoundError(f"Expected HDF5 output not found at {h5_path}")

        return parse_cpg_hdf5(str(h5_path))

    finally:
        if cleanup:
            for p in (params_path, h5_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass


def parse_cpg_hdf5(path: str) -> Dict:
    data: Dict = {}
    with h5py.File(path, "r") as h5:
        data["times_ms"] = np.array(h5["times_ms"])
        data["sim_ms"] = float(h5.attrs.get("sim_ms", 0))
        data["dt_ms"] = float(h5.attrs.get("dt_ms", 10))
        data["bs_osc_hz"] = float(h5.attrs.get("bs_osc_hz", 1.0))

        for side in ("L", "R"):
            gname = f"leg_{side}"
            if gname not in h5:
                continue
            g = h5[gname]
            leg_data: Dict = {}
            for key in g.keys():
                if key == "weights":
                    w_dict: Dict = {}
                    for wk in g["weights"].keys():
                        w_dict[wk] = np.array(g["weights"][wk])
                    leg_data["weights"] = w_dict
                elif key == "full_weights":
                    continue
                else:
                    leg_data[key] = np.array(g[key])
            data[gname] = leg_data

    return data


def _eval_one(args: Tuple) -> Tuple[int, float, Optional[Dict]]:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    idx, params, config_dict, objective_pickle_path = args
    import pickle

    config = CPGSimConfig(**config_dict)
    with open(objective_pickle_path, "rb") as f:
        objective_fn = pickle.load(f)

    tag = f"worker_{idx}_{uuid.uuid4().hex[:6]}"
    try:
        data = run_cpg_simulation(params=params, config=config, tag=tag, cleanup=True)
        fitness = float(objective_fn(data))
        return idx, fitness, data if fitness < 1e5 else None
    except Exception:
        return idx, 1e6, None


def evaluate_population_parallel(
    population_params: List[Dict[str, float]],
    objective_fn,
    config: CPGSimConfig,
    max_workers: int = 4,
) -> List[float]:
    """Evaluate a list of parameter dicts in parallel using ProcessPoolExecutor."""
    import pickle, tempfile

    obj_path = Path(config.work_dir) / f"_obj_{uuid.uuid4().hex[:8]}.pkl"
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    with open(obj_path, "wb") as f:
        pickle.dump(objective_fn, f)

    config_dict = {
        "sim_ms": config.sim_ms,
        "dt_ms": config.dt_ms,
        "threads": config.threads,
        "resolution_ms": config.resolution_ms,
        "long_run": config.long_run,
        "save_weights": config.save_weights,
        "nest_verbosity": config.nest_verbosity,
        "work_dir": config.work_dir,
        "python_bin": config.python_bin,
        "extra_args": config.extra_args,
    }

    tasks = [(i, p, config_dict, str(obj_path)) for i, p in enumerate(population_params)]
    results = [1e6] * len(population_params)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_eval_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                idx, fitness, _ = future.result()
                results[idx] = fitness
    finally:
        try:
            obj_path.unlink(missing_ok=True)
        except Exception:
            pass

    return results
