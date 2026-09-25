"""
paper_grid.py
Shared definition of the locomotion-mode x STDP-rate grids used by the paper
figure scripts (cpg_stdp_weight_matrix, cpg_stdp_weights_grid, cpg_force_stages,
cpg_network_matrix, cpg_mode_lambda_summary). Each script takes --grid.

  ftgrid : closed-loop force-triggered stepping, descending arm (BS->RG and
           CUT->RG-E plastic), 4 modes x 5 lambda -- run_forcetrig_grid.sh.
           Current paper main result.
  timer  : clock-paced stepping, sensory arm (BS frozen, CUT->RG-E and
           Ia->RG plastic), 5 modes x 5 lambda -- run_sensory_stdp.sh +
           run_ablation_sensory.sh. Previous paper version.

Per grid:
  modes   : (label, filename stem, display window ms for activity traces)
  primary : the main plastic projection, (weight key, label, y-max pA)
  aux     : the other plastic projections, same tuple, drawn on a second axis
  aux_axis: axis label for the aux projections
"""

LAMBDAS = [("lam1em2", "λ = 10⁻²"), ("lam1em3", "λ = 10⁻³"), ("lam1em4", "λ = 10⁻⁴"),
           ("lam1em5", "λ = 10⁻⁵"), ("lam1em6", "λ = 10⁻⁶")]
LAM_VAL = {"lam1em2": 1e-2, "lam1em3": 1e-3, "lam1em4": 1e-4, "lam1em5": 1e-5, "lam1em6": 1e-6}

GRIDS = {
    "ftgrid": dict(
        modes=[
            ("slow walk",                        "cpg_ftgrid_slow",   4000),
            ("medium walk\n(plantar, baseline)", "cpg_ftgrid_medium", 4000),
            ("fast walk",                        "cpg_ftgrid_fast",   4000),
            ("toe stepping\npartial unloading",  "cpg_ftgrid_toe",    4000),
        ],
        primary=("cut->rge_mean", "CUT→RG-E", 72),
        aux=[("bs->rge_mean", "BS→RG-E", 30), ("bs->rgf_mean", "BS→RG-F", 30)],
        aux_axis="BS→RG weight (pA)",
        model="force-triggered, descending arm",
    ),
    "timer": dict(
        modes=[
            ("slow walk\n6 cm/s",                     "cpg_sensory_stdp_06cms",   1200 * 5),
            ("medium / plantar\n13.5 cm/s (baseline)", "cpg_sensory_stdp_13_5cms", 520 * 5),
            ("fast walk\n21 cm/s",                    "cpg_sensory_stdp_21cms",   350 * 5),
            ("toe stepping\npartial unloading",       "cpg_ablsens_toe",          520 * 5),
            ("air stepping\nfull unloading",          "cpg_ablsens_air",          520 * 5),
        ],
        primary=("cut->rge_mean", "CUT→RG-E", 72),
        aux=[("ia->rge_mean", "Ia-E→RG-E", 12), ("ia->rgf_mean", "Ia-F→RG-F", 12)],
        aux_axis="Ia→RG weight (pA)",
        model="canonical sensory model",
    ),
}


def add_grid_arg(ap, default="ftgrid"):
    ap.add_argument("--grid", choices=sorted(GRIDS), default=default,
                    help="which mode x lambda grid to plot (see scripts/paper_grid.py)")


def find(indir, stem, lam):
    import glob, os
    h = sorted(glob.glob(os.path.join(indir, f"{stem}_{lam}_*.h5")))
    return h[0] if h else None
