#!/usr/bin/env python3
"""
GA optimization for the tinyCPG locomotor simulation.

  python run_cpg_opt.py --generations 10 --population 20 --sim-ms 10000 --reduced
  python run_cpg_opt.py --objective supervised --data-file ref.h5 --generations 20
  python run_cpg_opt.py --objective hybrid --data-file ref.h5 --supervised-weight 0.7
  python run_cpg_opt.py --workers 8 --generations 30 --population 40
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "optimization"))

from cpg_opt.parameters import get_cpg_parameter_space, get_reduced_cpg_parameter_space
from cpg_opt.objective import create_cpg_objective
from cpg_opt.simulator import run_cpg_simulation, CPGSimConfig, evaluate_population_parallel

from core.parameters import ParameterSpace
from methods.genetic import GeneticAlgorithm, GAConfig, GAResult


class CPGGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, param_space, cpg_objective, cpg_sim_config,
                 config=None, work_dir="/tmp/cpg_ga", workers=1):
        super().__init__(param_space=param_space, objective_fn=cpg_objective,
                         config=config or GAConfig())
        self.cpg_sim_config = cpg_sim_config
        self.work_dir = work_dir
        self.workers = workers
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        self._best_cpg_data = None

    def _evaluate(self, params_dict: Dict[str, float]) -> float:
        self.n_evaluations += 1
        tag = f"gen{len(self.history)}_eval{self.n_evaluations}"
        try:
            data = run_cpg_simulation(params=params_dict, config=self.cpg_sim_config,
                                      tag=tag, cleanup=True)
            fitness = self.objective_fn(data)
            if self.config.verbose and self.n_evaluations % 5 == 0:
                scores = self.objective_fn.compute_all(data)
                parts = " | ".join(f"{k}={v:.3f}" for k, v in scores.items())
                print(f"  [eval {self.n_evaluations}] fitness={fitness:.4f} ({parts})")
            if fitness < self._best_fitness:
                self._best_fitness = fitness
                self._best_params = params_dict.copy()
                self._best_cpg_data = data
                self._best_spike_data = {}
            return fitness
        except Exception as e:
            if self.config.verbose:
                print(f"  [eval {self.n_evaluations}] FAILED: {e}")
            return 1e6

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        if self.workers <= 1:
            return super()._evaluate_population(population)

        decoded = self._decode_population(population)
        results = evaluate_population_parallel(
            decoded, self.objective_fn, self.cpg_sim_config, max_workers=self.workers)

        for i, (params_dict, fitness) in enumerate(zip(decoded, results)):
            self.n_evaluations += 1
            if fitness < self._best_fitness:
                self._best_fitness = fitness
                self._best_params = params_dict.copy()
                self._best_spike_data = {}

        return np.array(results)


def _build_objective(args, wmax: float):
    target_freq = args.target_freq or 1.0
    experimental_data = None
    if args.data_file:
        from cpg_opt.data_loader import CPGExperimentalData
        experimental_data = CPGExperimentalData.from_file(args.data_file, verbose=True)
    return create_cpg_objective(
        objective_type=args.objective, experimental_data=experimental_data,
        supervised_weight=args.supervised_weight, target_freq_hz=target_freq, wmax=wmax)


def run_optimization(args) -> GAResult:
    param_space = get_reduced_cpg_parameter_space() if args.reduced else get_cpg_parameter_space()
    wmax = param_space.get_default().get("stdp_wmax", 120.0)
    cpg_objective = _build_objective(args, wmax)

    cpg_sim_config = CPGSimConfig(
        sim_ms=args.sim_ms, dt_ms=args.dt_ms, threads=args.threads,
        resolution_ms=args.resolution_ms, long_run=True, save_weights="final",
        nest_verbosity="M_ERROR",
        work_dir=str(Path(args.save_dir) / "_work") if args.save_dir else "/tmp/cpg_ga",
        python_bin=sys.executable)

    print("=" * 60)
    print("CPG Genetic Algorithm Optimization")
    print("=" * 60)
    print(f"  Parameters:  {param_space.n_params} ({'reduced' if args.reduced else 'full'})")
    print(f"  Objective:   {args.objective}")
    if args.data_file:
        print(f"  Data file:   {args.data_file}")
    if args.objective == "hybrid":
        print(f"  Sup. weight: {args.supervised_weight}")
    print(f"  Population:  {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Sim duration:{cpg_sim_config.sim_ms:.0f} ms")
    print(f"  Workers:     {args.workers}")
    print(f"  NEST threads:{cpg_sim_config.threads}")
    print(f"  Seed:        {args.seed}")
    print()

    for name, val in param_space.get_default().items():
        spec = param_space.get_spec(name)
        scale = " (log)" if spec and spec.log_scale else ""
        print(f"  {name}: {val}{scale}  [{spec.min_val}, {spec.max_val}]")
    print()

    ga_config = GAConfig(
        population_size=args.population, n_generations=args.generations,
        crossover_prob=0.9, mutation_prob=max(0.1, 1.0 / param_space.n_params),
        elite_size=max(1, args.population // 15), seed=args.seed, verbose=True)

    ga = CPGGeneticAlgorithm(
        param_space=param_space, cpg_objective=cpg_objective,
        cpg_sim_config=cpg_sim_config, config=ga_config,
        work_dir=str(Path(args.save_dir) / "_work") if args.save_dir else "/tmp/cpg_ga",
        workers=args.workers)

    return ga.optimize()


def save_results(args, result: GAResult, param_space: ParameterSpace):
    if not args.save_dir:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"cpg_ga_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "method": "cpg_ga",
        "timestamp": timestamp,
        "fitness": float(result.best_fitness),
        "best_params": result.best_params,
        "history": result.history,
        "n_evaluations": result.n_evaluations,
        "runtime_seconds": result.runtime_seconds,
        "cli_args": {
            "generations": args.generations, "population": args.population,
            "sim_ms": args.sim_ms, "threads": args.threads, "workers": args.workers,
            "reduced": args.reduced, "seed": args.seed,
            "target_freq": args.target_freq, "objective": args.objective,
            "data_file": args.data_file, "supervised_weight": args.supervised_weight,
        },
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    with open(save_dir / "best_params.json", "w") as f:
        json.dump(result.best_params, f, indent=2)
    print(f"\nResults:     {save_dir / 'results.json'}")
    print(f"Best params: {save_dir / 'best_params.json'}")
    print(f"  Re-run: python cpg_2legs_fast.py --params-json {save_dir / 'best_params.json'} --sim-ms 60000 --long-run")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        gens = [h["generation"] for h in result.history]
        ax.plot(gens, [h["best_fitness"] for h in result.history], "b-", lw=2, label="Best")
        ax.plot(gens, [h["mean_fitness"] for h in result.history], "r--", lw=1, alpha=0.7, label="Mean")
        ax.set(xlabel="Generation", ylabel="Fitness (lower = better)", title="CPG GA Convergence")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_dir / "convergence.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return save_dir


def main():
    p = argparse.ArgumentParser(description="GA optimization for CPG locomotor simulation")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reduced", action="store_true")
    p.add_argument("--sim-ms", type=float, default=10000.0)
    p.add_argument("--dt-ms", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=10, help="NEST threads per simulation")
    p.add_argument("--resolution-ms", type=float, default=0.5)
    p.add_argument("--target-freq", type=float, default=1.0)
    p.add_argument("--save-dir", "-s", type=str, default="results_cpg")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for population evaluation (1 = sequential)")
    p.add_argument("--objective", type=str, default="rule-based",
                   choices=["rule-based", "supervised", "hybrid"])
    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--supervised-weight", type=float, default=0.5)
    args = p.parse_args()

    result = run_optimization(args)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Best fitness: {result.best_fitness:.4f}")
    print(f"  Evaluations:  {result.n_evaluations}")
    print(f"  Runtime:      {result.runtime_seconds:.1f}s")
    print(f"\n  Best parameters:")
    for k, v in result.best_params.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    param_space = get_reduced_cpg_parameter_space() if args.reduced else get_cpg_parameter_space()
    save_results(args, result, param_space)


if __name__ == "__main__":
    main()
