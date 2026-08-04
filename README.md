# tinyCPG

A closed-loop spiking model of the **two-legged rat spinal central pattern
generator (sCPG)** in which the sensory synapses driving the locomotor rhythm
are not hand-tuned, but **self-organise through spike-timing-dependent
plasticity (STDP)**.

The model asks two questions: how does a spinal circuit tune its own synaptic
weights to produce a stable walking rhythm, and how well does that self-tuned
gait survive when sensory feedback is progressively withdrawn — the central
manipulation in spinal-cord-injury (SCI) rehabilitation.

## What the model contains

Each leg has an extensor and a flexor rhythm generator (RG-E, RG-F) coupled by
asymmetric reciprocal inhibition, driving motoneuron pools and Hill-like muscle
proxies whose force and length close the loop back through Ia afferents.
Left and right legs are coupled by commissural inhibition. Three projections
are plastic — cutaneous `CUT→RG-E` and the proprioceptive `Ia-E→RG-E` and
`Ia-F→RG-F` — while the brainstem drive is held tonic.

Extensions over the canonical computational sCPG lineage (Rybak 2006 →
Shevtsova 2015 → Danner 2017 → Zhang 2022):

1. STDP self-organisation of the descending and afferent synapses
2. a closed-loop Ia pathway into the reciprocal-inhibition core
3. explicit motoneuron pools and muscle proxies
4. externally paced heel→toe cutaneous drive during stance

## Experimental grid

Everything is reported on a **5 × 5 matrix**: five locomotion modes
(slow / medium=plantar=baseline / fast walk, toe stepping, air stepping)
crossed with five STDP learning rates (λ = 10⁻² … 10⁻⁶).

Main findings: the plastic weights converge to the same set-point regardless
of initialisation, leg or walking speed; gait quality peaks at an
**intermediate** learning rate rather than the fastest; and holding the
cutaneous drive intact under unloading rescues stepping that otherwise
collapses — a model of the epidural-stimulation effect in SCI.

## Repository layout

| Path | Contents |
|---|---|
| `cpg_2legs_fast.py` | The model — neurons, connectivity, simulation loop, HDF5 export |
| `run*.sh` | SLURM array scripts for the production sweeps (MN5) |
| `debug.sh` | Fast local single-config run (~30 s) |
| `scripts/` | Figure/analysis generators feeding `paper/figures/` |
| `scripts/legacy/` | Superseded generators, kept for reference |
| `paper/` | LaTeX manuscript (`main.tex`, `sections/`, `figures/`) |
| `results/` | Simulation output, one dated folder per production run |
| `validation/` | Literature-validation notes and data requests |

## Quick start

```bash
pip install -r requirements.txt     # see notes there re: NEST/NEURON
./debug.sh                          # fast local run -> results/debug.h5
python3 scripts/cpg_plot_from_hdf5.py --in results/debug.h5 --save-prefix debug
```

Production runs are submitted as SLURM arrays (`sbatch run_sensory_stdp.sh`);
see [`MN5_RUN.md`](MN5_RUN.md) for the full upload → submit → retrieve →
plot workflow, and [`CLAUDE.md`](CLAUDE.md) for model internals and the
tuning-knob reference.

## Implementations

The reduced model runs in **NEST** with Izhikevich neurons — cheap enough to
make the 5 × 5 sweeps tractable. A conductance-based **Hodgkin–Huxley**
implementation of the same circuit, used as a biophysical cross-check, lives
in the companion repository
[memCPG/CPG_STDP/py](https://github.com/max-talanov/memCPG/tree/main/CPG_STDP/py).

## Status

The manuscript is a work in progress: Methods, Results and Discussion are
drafted; Abstract and Introduction are not yet written. Figures are generated
from the committed scripts and are reproducible from the data in `results/`.

## License

MIT
