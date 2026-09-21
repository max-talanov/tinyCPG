import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "/Users/maxtalanov/Projects/memCPG/tinyCPG"

# Shared color-by-mechanism-family scheme, consistent ACROSS both figures:
# the LTP family ("ltp") deliberately uses the same brown in Fig 1a and Fig 1b,
# because the whole point is that it is one mechanism appearing in both states.
colors = {
    # Palette taken from tinyHippo's timescale_axis.png: a muted fast->slow hue
    # ramp (blue -> teal -> green -> olive -> orange -> brick -> purple).
    # Six values are sampled exactly from that figure; "cote" is the one added
    # tone (dusty mauve), matched to the same saturation/lightness band because
    # this figure has one more mechanism family than the reference had rows.
    # One colour per family, shared across Fig 1a and Fig 1b -- the reference
    # does the same thing (its rows 5 and 6 share #C1642F).
    "induction":    "#2E6FA7",  # blue     - NMDAR/AMPAR induction + STP
    "serotonergic": "#1E8FA0",  # teal     - serotonergic CPG gating
    "grau":         "#4C8C3B",  # green    - spinal instrumental learning
    "ltp":          "#AD8A2E",  # olive    - two-phase LTP, motor circuits
    "wolpaw":       "#C1642F",  # orange   - H-reflex conditioning
    "homeostatic":  "#A6433F",  # brick    - homeostatic scaling
    "cote":         "#9C5A76",  # mauve    - activity-dependent step-training
    "tasklearn":    "#6B4A8A",  # purple   - task-specific locomotor learning
}

MS   = 1
SEC  = 1000
MIN  = 60 * SEC
HOUR = 60 * MIN
DAY  = 24 * HOUR
WEEK = 7 * DAY

def render(rows, legend_keys, title, outfile, hatch=None, xmax=12 * WEEK,
           ticks=None, tick_labels=None, height=8.0):
    fig = plt.figure(figsize=(16, height), dpi=150)
    ax = fig.add_axes([0.335, 0.20, 0.315, 0.72])

    y_positions = list(range(len(rows)))[::-1]
    for (label, start, end, key), y in zip(rows, y_positions):
        ax.barh(y, width=(end - start), left=start, height=0.5,
                color=colors[key],
                edgecolor=("white" if hatch else "none"),
                linewidth=(0.8 if hatch else 0), hatch=hatch)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_ylim(-0.75, len(rows) - 0.25)

    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=10.5)
    ax.set_xlim(1, xmax * 1.5)

    ax.set_xlabel("Time since induction (log scale)", fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(axis="x", which="major", color="lightgray", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=colors[k], label=lbl, hatch=hatch,
                              edgecolor=("white" if hatch else "none"))
               for k, lbl in legend_keys]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.02),
              fontsize=10, frameon=False)
    plt.savefig(f"{OUT}/{outfile}")
    plt.close(fig)
    print("wrote", outfile)


TICKS = [1, 10, 100, SEC, 10 * SEC, MIN, HOUR, DAY, WEEK, 4 * WEEK, 12 * WEEK]
TICK_LABELS = ["1ms", "10ms", "100ms", "1s", "10s", "1min", "1hr", "1day",
               "1wk", "4wk", "12wk"]

healthy_rows = [
    ("NMDAR/AMPAR-dependent induction", 1, 15, "induction"),
    ("Short-term plasticity (facilitation/depression)", 15, SEC, "induction"),
    ("Serotonergic gating -- acute (state-dependent)  [22,23]", SEC, 40 * SEC, "serotonergic"),
    ("Spinal instrumental learning -- contingent (adaptive)  [1,2]", MIN, HOUR, "grau"),
    ("Early-phase LTP, motor circuits (E-LTP)  [1,4,6]", 2 * MIN, 3 * HOUR, "ltp"),
    ("Late-phase LTP, motor circuits (L-LTP)  [1,4,7]", 3 * HOUR, 3 * DAY, "ltp"),
    ("H-reflex conditioning, Phase I (fast, small)  [8]", HOUR, 2 * DAY, "wolpaw"),
    ("Homeostatic AMPAR upscaling (deafferentation)  [24,27]", DAY, 5 * DAY, "homeostatic"),
    ("Step-training neurotrophin upregulation  [14,15]", 5 * DAY, 14 * DAY, "cote"),
    ("Serotonergic gating -- chronic, training-restored  [22,23]", 5 * DAY, 28 * DAY, "serotonergic"),
    ("H-reflex conditioning, Phase II (slow, multi-site)  [8,9]", WEEK, 49 * DAY, "wolpaw"),
    ("Task-specific spinal locomotor learning\n(de Leon/Roy/Edgerton, contested)  [11,12,13]", 14 * DAY, 84 * DAY, "tasklearn"),
]

# Legend follows the fast->slow ramp, i.e. order of first appearance in the
# chart, matching how the reference figure orders its own colours.
healthy_legend = [
    ("induction", "NMDAR/AMPAR induction / short-term plasticity"),
    ("serotonergic", "Serotonergic CPG gating"),
    ("grau", "Spinal instrumental learning (Grau)"),
    ("ltp", "Two-phase LTP, motor circuits (Grau; corticospinal)"),
    ("wolpaw", "H-reflex conditioning (Wolpaw)"),
    ("homeostatic", "Homeostatic scaling"),
    ("cote", "Activity-dependent step-training (Cote et al.)"),
    ("tasklearn", "Task-specific locomotor learning (de Leon/Roy/Edgerton)"),
]

pathological_rows = [
    ("Spinal instrumental learning -- non-contingent (maladaptive)  [1,3,5]", MIN, HOUR, "grau"),
    ("Homeostatic downscaling failure / KCC2 loss (spasticity)  [25,26]", HOUR, 28 * DAY, "homeostatic"),
    ("Serotonergic gating -- chronic, untreated (persists)  [22,23]", 5 * DAY, 28 * DAY, "serotonergic"),
]

pathological_legend = [
    ("serotonergic", "Serotonergic CPG gating"),
    ("grau", "Spinal instrumental learning (Grau)"),
    ("homeostatic", "Homeostatic scaling"),
]

render(healthy_rows, healthy_legend,
       "Healthy / adaptive spinal plasticity timescales",
       "spinal_timescale_healthy.png", hatch=None,
       ticks=TICKS, tick_labels=TICK_LABELS, height=8.6)

render(pathological_rows, pathological_legend,
       "Pathological motor-circuit plasticity timescales (after SCI)",
       "spinal_timescale_pathological.png", hatch="//",
       ticks=TICKS, tick_labels=TICK_LABELS, height=4.6)
