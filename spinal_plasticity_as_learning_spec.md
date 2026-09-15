# Spinal Cord Plasticity as a Learning Specification

Companion document to [`CLAUDE.md`](CLAUDE.md), written by analogy to
[`hippocampal_timescales_as_circuit_spec.md`](https://github.com/max-talanov/tinyHippo/blob/main/hippocampal_timescales_as_circuit_spec.md).
That document reads hippocampal synaptic-tagging-and-capture (STC) as a circuit
specification for a memristor-based memory device. This document asks the
mirror question for the current debug goal — "keep alternating when BS drops
from 60 to 20 Hz, relying on Ia closed-loop feedback instead of the
brainstem" — by treating **published spinal-cord plasticity data**, not
hippocampal data, as the specification for what a bio-plausible learning rule
in `cpg_2legs_fast.py` should look like.

The short version: the spinal cord does not need to *borrow* hippocampal STC.
It has its own, independently-documented version of the same computational
problem — a fast, decaying change that only becomes permanent if a separate,
slower gating signal arrives in time — worked out across four largely
non-overlapping literatures (dorsal-horn nociceptive LTP, isolated-spinal-cord
instrumental learning, H-reflex operant conditioning, and activity-dependent
step-training). §1 lays out the timescales each literature reports. §2
extracts the shared tag/gate structure and the falsification logic each
literature already used to prove it. §3 maps that structure onto the three
plastic pathways already in this model. Implementation is deliberately out of
scope here — see the separate plan.

## 1. Spinal plasticity timescales and their modeling constraints

This section was originally one combined table and figure with healthy and
pathological processes sorted only by speed, which made it hard to tell
which category a given row belonged to at a glance. §1.1 and §1.2 now split
it into two self-contained sets — same columns, same underlying numbers,
but never mixed in one chart or one table.

### 1.1 Healthy / adaptive processes

![Nine healthy spinal plasticity processes on one logarithmic time axis from 1ms to several weeks, sorted fastest-onset-first top to bottom and colored by literature source — blue for NMDAR/AMPAR induction and short-term plasticity, orange for Grau's contingent spinal instrumental learning, green for Wolpaw's two-phase H-reflex conditioning, purple for Côté's activity-dependent step-training, teal for serotonergic CPG gating, and gray for homeostatic AMPAR upscaling — bar position marks onset, length marks characteristic duration.](spinal_timescale_healthy.png)

*Fig. 1a — The healthy/adaptive half of the timescale table below, sorted by onset. These are baseline physiology (induction, short-term plasticity, acute serotonergic gating) plus every mechanism whose documented outcome in the cited literature is functional preservation or recovery.*

| Process | What it is | Time constant | Site / pathway | What it constrains for a spiking-network model |
|---|---|---|---|---|
| NMDAR/AMPAR-dependent synaptic induction | The basic ionotropic-glutamate-receptor gating that every mechanism below is built on top of. | ms-scale, same coincidence physics as everywhere else in CNS | Ubiquitous (dorsal horn, motoneuron) | Sets the same fast coincidence-detection floor as the hippocampal case — nothing spinal-specific here. |
| Short-term plasticity (facilitation/depression) | Transient, non-associative changes in synaptic efficacy that decay within a single behavioral episode, leaving no lasting trace. | ~10 ms–1 s | Ubiquitous (same NMDAR deactivation kinetics as the hippocampal case) | Not spinal-specific either, but still a real constraint: this is why a millisecond-scale coincidence window exists at all, underneath every pathway-specific mechanism below. |
| Serotonergic (5-HT) neuromodulatory gating of CPG excitability — acute | Moment-to-moment 5-HT release from descending brainstem projections that sets whether the CPG's rhythm-generating interneurons can burst at all, independent of any lasting weight change. | Seconds (state-dependent gating of whether rhythmic bursting can occur at all) | Brainstem-to-spinal monoaminergic projections onto CPG interneurons and motoneurons | The model's tonic `BS_REGULAR_HZ` drive is a generic reticulospinal proxy; the literature's actual candidate for "descending drive that gates whether the spinal rhythm-generator can run" is specifically serotonergic/noradrenergic. |
| Spinal instrumental learning — **contingent** (adaptive) | Response-contingent training (e.g., a limb flexion that terminates shock) that changes spinal reflex output for hours to days, entirely below a complete spinal transection. | Acquisition within a single session (tens of minutes); consolidation requires new protein synthesis over the following hours | Interneuron + motoneuron circuits caudal to a complete spinal transection — no brain involvement (Grau et al.) | The adaptive half of a bidirectional gate — see §1.2 for what the identical training produces when the outcome is uncontrollable instead. |
| H-reflex operant conditioning, Phase I | The first, small, rapidly-developing component of operant conditioning of the monosynaptic stretch reflex, visible within 1-2 days of daily training. | 1–2 days; small magnitude | Ia-afferent → motoneuron monosynaptic pathway + interneurons (Wolpaw) | The fast, labile component of a two-phase learning process running entirely on the pathway this model calls `Ia→RG`. |
| Homeostatic AMPAR **upscaling** after deafferentation / chronic inactivity | A compensatory, cell-wide increase in motoneuron AMPAR-mediated synaptic strength that partly offsets a chronic loss of afferent drive. | Hours–days; mediated by synaptic insertion of GluA2-lacking, Ca²⁺-permeable AMPA receptors | Motoneurons below an injury or period of inactivity | The literature's own mechanism for "what happens when descending/afferent drive is chronically reduced" is an active receptor-composition change with its own kinetics — not a static gain multiplier. Directly relevant to why the model's `--wmax-ia-unloaded` gain-based unloading-rescue attempts plateaued (see [CLAUDE.md](CLAUDE.md), "Core architecture fix" section). |
| Activity-dependent step-training neurotrophin upregulation | Repeated, task-specific locomotor training that raises BDNF/NT-3/NT-4 in the lumbar cord below an injury, with the training *type*, not just its amount, determining the outcome. | Daily training over days–weeks; **task-specific** (step-training and cycle-training produce different BDNF/NT-3/NT-4 profiles and different dorsal-horn/intermediate-gray neuron counts) | Lumbar spinal cord below a lesion (Côté, Azzam, Lemay, Zhukareva & Houlé 2011) | Argues against a single generic "activity level" gate — the training *modality*, not just its amount, sets what gets reinforced. Directly relevant to this model's distinction between `--cut-trigger force` (stance-loading-driven) and the paced-clock modes. |
| Serotonergic gating — chronic, **training-restored** | The same slow, injury-driven change in CPG serotonin sensitivity as §1.2's untreated case, but partly reversed by locomotor training and serotonergic agonists. | Days–weeks | Brainstem-to-spinal monoaminergic projections | Its *sensitivity*, not just its rate, changes with training — a second, slower plasticity axis this model does not yet represent (flagged here, not addressed by the plan in §3). |
| H-reflex operant conditioning, Phase II | The slow, large, multi-site component of H-reflex conditioning — altered motoneuron firing threshold, GABAergic terminal density, and interneuron properties — that consolidates over weeks of continued training. | 6–7 weeks; large, stable; **multi-site** | Same pathway | Demonstrated to correct locomotor asymmetry after spinal cord injury when combined with training — the closest published result to the model's own stated rehab goal. |

*Table 1a — The nine rows plotted in Fig. 1a above: baseline physiology plus every mechanism whose documented outcome in the cited literature is functional preservation or recovery. Two rows (spinal instrumental learning, serotonergic chronic axis) are one half of a bidirectional mechanism — their pathological counterpart is Table 1b, not absent, just not adaptive.*

### 1.2 Pathological processes (e.g., after spinal cord injury)

![Five pathological spinal plasticity processes on the same logarithmic time axis, hatched to distinguish them from the healthy set, colored by literature source — blue for Sandkühler's dorsal-horn E-/L-LTP now shown as a chronic-pain model, orange for Grau's non-contingent maladaptive suppression, teal for untreated chronic serotonergic dysregulation, and gray for KCC2-loss-driven homeostatic downscaling failure.](spinal_timescale_pathological.png)

*Fig. 1b — The pathological half, same axis and scale as Fig. 1a, so the two are directly comparable but never rendered as one mixed chart. Every row here has a same-timescale counterpart process in Fig. 1a (e.g. spinal instrumental learning, or the serotonergic chronic axis) — the pathology is a different *outcome* of the same paradigm or mechanism family, not a different timescale.*

| Process | What it is | Time constant | Site / pathway | What it constrains for a spiking-network model |
|---|---|---|---|---|
| Early-phase spinal LTP (E-LTP) | An NMDA-receptor-dependent potentiation of dorsal-horn transmission that forms within minutes of patterned afferent input and spontaneously decays within a few hours if nothing stabilizes it — in its own primary literature, the onset of central sensitization. | 1–3 h; NMDA-receptor-dependent; **protein-synthesis-independent** | Dorsal-horn C-fiber synapses, induced by afferent tetanic stimulation (Sandkühler & Liu 1998) | A transient, spontaneously-decaying potentiation that exists *before* any stabilizing signal arrives — the direct spinal analog of the hippocampal "tag," borrowed here as this document's molecular substrate even though its own field studies it to *block* it (see §2.2). |
| Spinal instrumental learning — **non-contingent** (maladaptive) | The same training paradigm as §1.1's contingent case, but with the shock uncontrollable instead of response-produced. | Same acquisition/consolidation window as the contingent case — the outcome, not the timing, differs | Same circuits (Grau et al.) | An active, protein-synthesis-dependent **suppression** of future learning capacity, not merely an absence of learning — a bidirectional gate, not a one-way accumulator. |
| Late-phase spinal LTP (L-LTP) | The protein-synthesis-dependent stabilization of E-LTP into a non-decaying potentiation — in its own primary literature, the cellular consolidation step for chronic pain. | Onset by ~3 h; **requires ongoing protein synthesis**; selectively induced/occluded by BDNF and spinal D1/D5 dopamine-receptor activation | Same dorsal-horn synapses | A discrete regime change from decaying to non-decaying — structurally identical to hippocampal "capture," but consolidating a pain state rather than a motor trace in the literature this mechanism is actually drawn from. |
| Homeostatic **downscaling failure** (KCC2 loss, spasticity) | Instead of excitation scaling down to compensate for hyperactivity, motoneuron KCC2 (which sets the Cl⁻ gradient underlying GABA/glycine inhibition) is chronically **downregulated** after SCI, producing spasticity. | Onset within hours of injury; partial training-driven recovery over weeks | Motoneuron membrane Cl⁻ transporters, below a spinal cord injury | The spinal-specific evidence that "homeostatic compensation" is not automatically adaptive — it can fail in the *opposite* direction from §1.1's upscaling row, and, critically, that failure is training-reversible (Boulenguez et al. 2010; see Fig. 2b), not fixed. |
| Serotonergic gating — chronic, **untreated** | The same injury-driven change in CPG serotonin sensitivity as §1.1's training-restored case, left to persist. | Days–weeks, and beyond without intervention | Brainstem-to-spinal monoaminergic projections | The pathological anchor for the same axis §1.1 shows can be treated — the two rows differ only in whether training happened, exactly like the KCC2 row above. |

*Table 1b — The five rows plotted in Fig. 1b above. Every row is a maladaptive outcome documented in the cited literature, not a hypothetical worst case — and three of the five (E-/L-LTP, instrumental learning, serotonergic axis) share a mechanism family with a Table 1a row; only the outcome differs, per §2.1/§2.2 below.*

### 1.3 Bidirectional homeostatic scaling: upscaling and downscaling

The upscaling row above is one direction of a bidirectional mechanism first
characterized in cortical culture (Turrigiano & Nelson 2004; Turrigiano 2008,
"the self-tuning neuron"): chronic silencing drives synaptic **upscaling**
(GluA1 Ser845 phosphorylation and insertion), while chronic hyperactivity
drives synaptic **downscaling** (stargazin dephosphorylation, GluA1 removal)
— the same AMPAR-trafficking toolkit running in opposite directions to hold
network activity near a set point. The spinal cord shows the upscaling side
cleanly (deafferentation → GluA2-lacking, Ca²⁺-permeable AMPAR insertion in
motoneurons, already in the table). The downscaling side is where the spinal
literature gets more interesting than a simple mirror image: after SCI, the
dominant documented homeostatic-*failure* mode is not "excitatory synapses
fail to scale down" but **inhibitory efficacy itself collapsing** — KCC2,
the potassium-chloride cotransporter that keeps the Cl⁻ reversal potential
hyperpolarized enough for GABA/glycine to inhibit, is downregulated in
motoneuron membranes after SCI (Boulenguez et al. 2010, *Nature Medicine*),
which is functionally equivalent to a failed downscaling response —
excitability stays elevated because the compensatory brake never engages —
and is a well-established mechanistic account of post-SCI spasticity.

**This closes the loop with the document's rehab framing rather than sitting
outside it**: the same Boulenguez-line literature reports that locomotor
training partially restores KCC2 expression and reduces spasticity, i.e. the
homeostatic failure is *training-reversible*, not fixed — precisely the
"does training restore lost function" question this whole document exists to
give a mechanistic answer to, on a completely different pathway (chloride
homeostasis) than the tag-and-capture story in §2.

![Two-panel chart: (a) synaptic AMPAR weight rising smoothly from baseline to a compensated plateau over about three days after deafferentation (healthy upscaling); (b) motoneuron KCC2/inhibitory efficacy dropping sharply at spinal cord injury, then either staying flat at the reduced floor with no training (dashed red, spasticity persists) or partially recovering toward baseline over several weeks with locomotor training (solid green) (pathological downscaling failure).](spinal_scaling_dynamics.png)

*Fig. 2 — Bidirectional homeostatic scaling. **(2a) Healthy** (left): the upscaling response to deafferentation — a single, reliably-reported trajectory; there is no "blocked" condition in the cited literature, so only one trace is shown. **(2b) Pathological** (right): KCC2 loss (downscaling failure) after SCI, training vs. none — both traces take the identical acute post-injury drop; they diverge only in whether locomotor training is applied afterward, which is the direct spinal analog of "capture" rescuing an otherwise-lost trace, the same normal-vs-blocked contrast the hippocampal document's Fig. 2 uses for capture. Curve shapes are illustrative (exponential fits to the qualitative time course each citation reports), not digitized data.*

## 2. Tag-and-capture and contingency-gated consolidation in spinal circuits

Three independent literatures converge on the same two-stage structure the
hippocampal document derives from STC — a labile, spontaneously-decaying
change, stabilized only by a second, slower, separately-gated signal — without
citing each other or the hippocampal literature:

**Molecular substrate (Sandkühler & Liu 1998; Yang, Chen, Zhang & Sandkühler 2004; Zhang & Sandkühler 2008).**
Dorsal-horn E-LTP behaves exactly like a decaying tag: NMDA-receptor-dependent,
induced by a single tetanic conditioning stimulus, present within minutes, and
gone within a few hours if nothing else happens. Its conversion to L-LTP is a
discrete regime change — selectively blocked by the protein-synthesis
inhibitors cycloheximide and anisomycin, which affect neither induction nor
baseline transmission — and that conversion is itself gated by a separate,
non-glutamatergic signal: activation of spinal D1/D5 dopamine receptors
*occludes only the late phase*, not the early phase, and exogenous BDNF is
independently sufficient to induce a late phase on its own. This is the
molecular-level proof that "labile tag, stabilized only if a distinct
neuromodulatory signal arrives before decay" is not a hippocampus-specific
trick — it is present, with the same pharmacological dissociation the
hippocampal document relies on (§2 of that document), at spinal synapses with
no cortex or hippocampus involved. **This substrate's own primary literature
framing is pathological, not adaptive**, worth flagging before treating it
purely as this document's tag/capture analogy: the identical NMDAR-dependent,
BDNF/dopamine-gated dorsal-horn potentiation is the standard cellular model
of central sensitization underlying hyperalgesia and chronic pain (Sandkühler
& Liu 1998; Ruscheweyh, Wilder-Smith, Drdla, Liu & Sandkühler 2011) — the same
mechanism the pain field studies specifically *to block it*. The retention
rule itself is not "good" or "bad"; §2.1/§2.2 below collect this alongside the
other three mechanisms above that are similarly two-directional.

**Behavioral/computational gate — contingency, not activity (Grau 2014; Crown & Grau 2001; Ferguson, Crown & Grau 2006).**
The isolated spinal cord (transected, brain removed from the loop) can be
trained: if shock is delivered *only* when a hindlimb is extended, the limb
learns to hold a flexed position, and this improvement requires ionotropic
glutamate receptors, NMDA-receptor-dependent plasticity, BDNF release, and new
protein synthesis to outlast the training session — the same molecular
signature as L-LTP above, but on a instrumental, response-contingent
timescale. Critically, the gate here is **bidirectional and is set by
contingency, not by how much afferent drive occurred**: uncontrollable
(non-contingent) shock at the same intensity and frequency produces the
*opposite* outcome — a protein-synthesis-dependent, actively **maladaptive**
suppression of the spinal cord's future capacity to learn the same task,
mediated in part through group-I metabotropic glutamate receptors acting via
protein kinase C (a genuine metaplasticity mechanism, not just "no capture
happened"). The falsification structure is already built into these papers:
block protein synthesis and the transient within-session change still occurs,
but it does not outlast the session — the same dashed-trace logic §2 of the
hippocampal document proposes as a memristor test, already run as a wet-lab
experiment on spinal tissue.

**Structural, two-phase consolidation on exactly this model's Ia pathway (Wolpaw; Chen, Chen, Liu & Wolpaw 2006; Thompson & Wolpaw 2014).**
H-reflex operant conditioning is the most direct match to `Ia→RG` specifically,
because it is conditioning of the monosynaptic Ia-afferent-to-motoneuron
reflex pathway itself, in awake, behaving animals (rats, monkeys, and humans),
sustained over weeks. It has an explicitly reported two-phase time course:
**Phase I**, a small change appearing within 1–2 days, and **Phase II**, a
much larger and more stable change developing over roughly 6–7 weeks and
distributed across multiple sites — altered motoneuron firing threshold and
axonal conduction velocity, changed density of GABAergic (and other) synaptic
terminals on the motoneuron, and interneuron-level changes. This is a
structural instantiation of "fast, labile component" and "slow, stable,
multi-site component" reached without invoking any tag/capture vocabulary at
all — and it is not merely a laboratory curiosity: H-reflex up-conditioning
has been shown to increase the soleus burst and **correct locomotor asymmetry
after spinal cord injury in rats**, and an equivalent protocol improves
walking in humans with incomplete SCI. This is the closest published result to
this model's own stated debug goal (self-sustained alternation under reduced
descending drive, carried by the sensory-afferent pathway) — achieved in real
spinal cords via training-induced consolidation of exactly the Ia-to-motor
pathway this model already represents.

**Synthesis.** All three literatures independently reproduce the same shape:
(1) a fast, NMDA-receptor-gated, spontaneously-decaying change; (2) a slower,
separately-gated conversion to a stable state, requiring protein synthesis and
a distinct signal (a neuromodulator, a behavioral-contingency outcome, or an
accumulating structural change) arriving before the fast component decays;
(3) the *sign* of the stable outcome is not guaranteed positive — a failed or
uncontrollable contingency produces active, protein-synthesis-dependent
suppression, not neutrality. None of this requires borrowing hippocampal
machinery; it is the spinal cord's own, independently-evolved version of the
same two-timescale plasticity problem, and it is better matched to a
locomotor CPG than the hippocampal case is, because two of the three
literatures (Wolpaw; Grau) were generated in exactly this kind of circuit.

Every mechanism introduced above is two-directional, and the prose so far
mostly narrated the adaptive direction. The retention rule (tag → decay →
gated capture) this document proposes is mechanism-neutral machinery, not an
inherently "good" plasticity rule — what makes a given instance of it
adaptive or pathological is which signal gates the capture and which synapse
it acts on, nothing else. §2.1 and §2.2 list the two directions as two
separate tables rather than side-by-side columns, matching the §1.1/§1.2
split above, so neither list is read as a footnote to the other.

### 2.1 Healthy / adaptive directions

| Mechanism | What the adaptive direction looks like |
|---|---|
| Homeostatic scaling (§1.1) | Upscaling after deafferentation restores excitability lost to reduced afferent drive |
| Spinal instrumental learning (Grau, §1.1) | Contingent (response-produced) outcome → NMDAR/BDNF/protein-synthesis-dependent potentiation that outlasts the session |
| Dorsal-horn E-/L-LTP (Sandkühler, §1.2) | Borrowed as this document's molecular-substrate analogy for a motor-consolidation tag/capture rule — not this mechanism's own primary framing, see §2.2 |
| Serotonergic gating, chronic axis (§1.1) | Partly restored by locomotor training and serotonergic agonists post-injury |
| H-reflex conditioning (Wolpaw, §1.1) | Up-conditioning corrects locomotor asymmetry after SCI, in both rats and humans |

*Table 2a — For each two-directional mechanism introduced in §1-2, the direction its cited literature reports as adaptive or rehab-positive. Pairs row-for-row with Table 2b except the last, which Table 2b leaves blank rather than inventing a counterpart.*

### 2.2 Pathological / maladaptive directions

| Mechanism | What the maladaptive direction looks like |
|---|---|
| Homeostatic scaling (§1.2) | **Downscaling failure**: KCC2 loss after SCI leaves inhibition too weak, producing spasticity (Boulenguez et al. 2010) — training partially reverses it |
| Spinal instrumental learning (Grau, §1.2) | Non-contingent (uncontrollable) outcome at the *identical* intensity → active, protein-synthesis-dependent **suppression** of future learning capacity (Ferguson, Crown & Grau 2006) — not merely an absence of learning |
| Dorsal-horn E-/L-LTP (Sandkühler, §1.2) | **Its own primary literature framing**: the standard cellular model of central sensitization underlying hyperalgesia and chronic pain (Ruscheweyh et al. 2011) |
| Serotonergic gating, chronic axis (§1.2) | Persistently altered CPG sensitivity to 5-HT if left untreated after injury |
| H-reflex conditioning (Wolpaw) | *No entry* — no specific pathological counterpart in the literature cited here. Left as an open question rather than asserted, unlike the four rows above (which is also why §1's Fig. 1b/Table 1b has no H-reflex row at all). |

*Table 2b — The maladaptive direction of the same mechanisms, where the cited literature documents one. Four of five rows are populated; the H-reflex row is deliberately left without an entry (§2's synthesis explains why, below) rather than filled with an invented failure mode.*

Four of five mechanisms have a confirmed pathological counterpart in the
cited literature; H-reflex conditioning is left asymmetric deliberately
rather than inventing one. The pattern that recurs across all four confirmed
pairs is the same one §2's synthesis above already generalizes across
mechanisms: gate signal and target synapse determine the outcome, not a
separate "adaptive plasticity" versus "maladaptive plasticity" machinery.
This is not only a biological aside — §3 below closes the loop back to
`cpg_2legs_fast.py` itself, where the same duality shows up as literal,
observed failure modes.

### 2.3 Maladaptive plasticity after spinal cord injury: synthesis

Table 2b's four populated rows are not four independent side-effects — after
SCI they are commonly reported together, and the literature has its own name
for the pattern: **maladaptive plasticity**, plasticity mechanisms that
persist (the retention rule doesn't switch off after injury) but now degrade
function instead of preserving or restoring it, precisely because the signal
gating capture and the synapse it acts on have both been changed by the
lesion, not because plasticity itself became a different kind of process
(Ferguson et al. 2012's own framing: **"maladaptive spinal plasticity opposes
spinal learning and recovery in spinal cord injury"** — the mechanism this
whole document borrows from Grau to explain adaptive spinal learning is, in
the same body of work, the explanation for why recovery sometimes fails to
happen at all).

Four convergent, commonly-co-occurring changes, each already a row in Table
1b/2b:

1. **Spasticity via chloride dysregulation.** KCC2 loss depolarizes the Cl⁻
   reversal potential, so GABA/glycine input that should inhibit a motoneuron
   instead barely restrains it — hyperreflexia and spasticity, not a single
   symptom but the direct electrophysiological consequence (Boulenguez et al.
   2010).
2. **Central/neuropathic pain via dorsal-horn LTP consolidation.** The same
   NMDA/BDNF/dopamine-gated capture mechanism this document borrows as a
   *motor*-consolidation analogy (§2) is, in its own field, understood as
   consolidating a *pain* state — hyperalgesia and allodynia that outlast the
   original noxious input by exactly the E-LTP→L-LTP transition described
   above (Sandkühler & Liu 1998; Ruscheweyh et al. 2011).
3. **Impaired capacity for further learning if training is uncontrollable.**
   Grau's own metaplasticity result: input the spinal cord cannot control
   (as opposed to input it can control, e.g. active stepping practice) leaves
   a protein-synthesis-dependent deficit that suppresses *subsequent*
   learning, not just the immediate trial — a mechanistic argument that
   passive or poorly-timed rehabilitation is not merely less helpful than
   active, contingent training, it can be actively counterproductive
   (Ferguson, Crown & Grau 2006).
4. **Altered descending neuromodulatory sensitivity.** The CPG's own
   response to serotonin — not just its resting drive — shifts after injury
   and, left untreated, stays shifted (§1 table).

**None of these four is a separate disease process bolted onto normal
plasticity** — each is the same class of mechanism as its Table 2a
counterpart (homeostatic scaling, tag/capture, contingency-gated learning,
neuromodulatory gating), running on the same retention rule, tipped toward
the maladaptive branch by what the lesion changed about the gating signal or
the target synapse. This is why §2's synthesis frames the retention rule as
neutral machinery rather than an inherently protective one: after a real
injury, several of these mechanisms plausibly tip in the same direction at
once, which is a *multiple-hit* picture, not a single lesion causing a
single problem, and matches the general SCI neuroplasticity literature's own
framing of concurrently-operating mechanisms across motoneurons,
interneurons, and afferents rather than one dominant cause.

**Consequence for this document's own proposal.** A `--consolidate`-style
mechanism (§2, implemented in `cpg_2legs_fast.py`, see [CLAUDE.md](CLAUDE.md))
is exactly this retention rule, and it inherited the same double edge: tuned
one way it reproduces Table 2a's outcomes (§3 below), tuned another way it
reproduces literal analogs of Table 2b's — cap-domination as a positive-
feedback runaway structurally like unchecked central sensitization, and
leg-synchronization as a loss of the differentiation a healthy pair of limbs
maintains. Building the mechanism was necessary but insufficient; which
regime it lands in is an empirical, per-configuration question, exactly as
it is in the biology this document is modeled on.

### 2.4 Active forgetting: erasing an already-consolidated trace

Everything decaying in this document so far is **passive**: an uncaptured
tag relaxes back toward baseline on its own because nothing sustains it
(§2's E-LTP decay; the hippocampal document's own Fig. 2, where the
blocked/dashed trace just drifts back down each cycle with no dedicated
mechanism driving it). The general active-forgetting literature (outside
the spinal cord) draws a sharp line between that and **active** forgetting:
a dedicated, triggerable process that dismantles a trace that would
otherwise persist — in *Drosophila*, specific dopaminergic "forgetting
cells" driving Rac1/cofilin-mediated actin remodeling in mushroom-body
neurons, with blocking those cells making memories last *longer*, not
shorter (Shuai et al. 2010; Cervantes-Sandoval et al. 2016's Scribble-Rac1-
Cofilin "forgetting signalosome"). The spinal cord has its own, independent
version of exactly this category — not borrowed from the fly literature, at
the same dorsal-horn synapses this document already uses as its molecular
substrate (§2):

**Opioid-triggered depotentiation (Drdla-Schütting, Benrath, Wunderbaldinger
& Sandkühler 2012, *Science*).** A brief, high-dose opioid-receptor
activation actively reverses already-established C-fiber LTP — not by
withdrawing support and letting it drift down, but through its own
Ca²⁺-dependent signaling cascade that normalizes AMPA-receptor
phosphorylation back to baseline, on demand. Critically, this **reverses
hyperalgesia in behaving animals**: it is not merely analgesia (the pain
signal is temporarily damped) but the erasure of the underlying synaptic
memory trace of pain, with the behavioral change outlasting the drug. This
is the most literal spinal counterpart to "active forgetting" available —
closer to it, in fact, than the hippocampal document's own STC-blocked
counterfactual, which only ever shows a trace that *failed to consolidate*,
never one that consolidated and was then actively taken back down.

**Maintenance-dependent erasure via PKMζ (Asiedu, Tillu, Melemedjian, Shy,
Sanoja, Bodell, Ghosh, Porreca & Price 2011, *J. Neurosci.*).** A second,
independent route to the same outcome: the spinal cord's consolidated
(late-phase) nociceptive sensitization is not a state that persists for
free once captured — it requires *ongoing* synthesis/activity of protein
kinase Mζ to remain maintained at all. Blocking PKMζ (with the
ζ-pseudosubstrate inhibitory peptide, ZIP) collapses the already-established
potentiation. This mirrors the well-known hippocampal PKMζ/ZIP maintenance
literature (Sacktor and colleagues) point for point, again without either
field citing the other — a second, independent line of evidence that
"capture" in real neural tissue is not the discrete permanent write this
document's tag/capture vocabulary (and the hippocampal circuit proposal it
mirrors) makes it sound like; maintenance itself can be an ongoing,
interruptible process with its own active-erasure failure mode.

**Consequence for `--consolidate` (this document's own proposal).** The
implemented mechanism (§2, `cpg_2legs_fast.py`) only has the *passive* half
of this picture: an uncaptured tag decays on its own (`consolidation_leak`),
but once a capture event freezes `baseline`, nothing in the current design
can ever move it back down again short of the *loading-dependent* logic
built for other reasons ([CLAUDE.md](CLAUDE.md), "Core architecture fix").
There is no analog of an actively-triggered erasure signal that could reset
an already-captured baseline — the spinal literature above says such a
signal (opioid receptor activation reversing a specific pathological
capture; a maintenance-kinase blockade collapsing another) is a real,
separate mechanism, not a hypothetical one, and this document does not yet
propose a model-level counterpart for it. Flagged here as an open gap for a
future pass, not addressed by the current plan.

## 3. Mapping onto the tinyCPG architecture

This model already has three standing plastic pathways
(`BS→RG`, `CUT→RG`, `Ia→RG` — see [CLAUDE.md](CLAUDE.md), "Core architecture
fix" and `MOD_IA_RG_STDP`), each currently a single-timescale STDP weight
bounded only by a fixed `Wmax`. §2's synthesis suggests each is missing a
second, slower component:

- **`Ia→RG-E`/`Ia→RG-F`** maps most directly onto Wolpaw's H-reflex
  conditioning: same pathway (afferent-to-motor-circuit), same fast/slow
  two-phase shape, and — uniquely among the three pathways — literature
  evidence that consolidating exactly this pathway restores locomotor
  symmetry under reduced/altered descending drive, which is this model's own
  debug goal. This is also the pathway currently deliberately kept
  weight-capped (`WMAX_IA=10`) to avoid saturating into tonic co-excitation
  (see CLAUDE.md's `MOD_IA_RG_STDP` note) — a fixed cap is exactly the kind of
  static limit that a genuine consolidation mechanism (a cap that *rises* only
  after demonstrated stable, successful use, per Phase I→II) would replace
  with something bio-plausible instead of hand-tuned.
- **`CUT→RG-E`** maps onto the dorsal-horn E-LTP/L-LTP data most directly
  (both are afferent, cutaneous/nociceptive-adjacent pathways with the same
  NMDA/BDNF/dopamine-gated late-phase structure) and onto the Grau
  contingency literature functionally — a stance bout that resolves via
  genuine force-threshold crossing is the "successful, contingent" case, and
  one that only ends via the failsafe timeout (`--cut-max-stance-ms`) is
  structurally the "forced/non-contingent" case those experiments show
  produces active suppression, not just an absence of reinforcement. That
  reframes the model's own `frac_at_cap` diagnostic — already used throughout
  the force-triggered-CUT tuning rounds to flag disguised-clock results — as a
  candidate for the literal biological gate signal, not just a
  post-hoc correctness check.
- **`BS→RG`** is the weakest fit: none of the four literatures describe
  descending drive itself as undergoing this kind of tag/capture-style
  plasticity in the same sense (the closest analog, the serotonergic
  acute/chronic receptor-sensitivity change after injury, is a separate,
  slower axis — see §1's table, not addressed further here). Applying the
  same mechanism to `BS→RG` for architectural uniformity is a modeling choice
  this document does not find direct literature support for, one way or the
  other — worth flagging explicitly before the plan commits to it.

Implementation (state variables, update equations, where in
`cpg_2legs_fast.py`'s sim loop this would live) is intentionally left to the
separate implementation plan, not this document.

**§2.1/§2.2's healthy/pathological duality is not just biological framing — it
recurs as literal, measured failure modes once `--consolidate` was actually
built and tuned** (see [CLAUDE.md](CLAUDE.md), "Tag-and-capture
consolidation"). Two of that section's four confirmed pathological
directions have a direct model-level counterpart, found empirically, not
predicted in advance by this document:

- **Cap-domination** (`frac_at_cap` near 1.0 — the failsafe timer, not
  genuine sensory feedback, driving every stance/swing transition) is the
  model's own version of the homeostatic-downscaling failure above: a
  positive-feedback loop (`CUT→RG-E→force_e→CUT`) that never releases,
  structurally the same shape as excitation that never gets scaled back down
  because the compensatory brake (there, KCC2; here, a capture gate tuned
  too permissively) never engages.
- **Leg-synchronization** (corr(F-E_L,F-E_R) flipping positive — both legs'
  `CUT→RG-E` consolidating to the same stable plateau instead of staying
  desynchronized) is a direct instance of over-consolidation: capturing too
  easily and too often erased exactly the kind of run-to-run,
  synapse-to-synapse asymmetry that real tag/capture leaves intact (§2's
  "labile, spontaneously-decaying" tag is *supposed* to preserve variability
  between reinforcement events, not average it away).

Both were found by tuning `--consolidate`'s gain/threshold constants after
implementation, the same way the biology's own pathological directions were
found by perturbing (not designing) real spinal circuits — the retention
rule was neutral machinery in both cases; the tuning (or the lesion) is what
picked adaptive or maladaptive.

## References

- Sandkühler, J. & Liu, X. (1998). [Induction of long-term potentiation at spinal synapses by noxious stimulation or nerve injury](https://onlinelibrary.wiley.com/doi/10.1046/j.1460-9568.1998.00278.x), *Eur. J. Neurosci.*
- Ruscheweyh, R., Wilder-Smith, O., Drdla, R., Liu, X.-G. & Sandkühler, J. (2011). [Long-term potentiation in spinal nociceptive pathways as a novel target for pain therapy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3078873/), *Mol. Pain* (dorsal-horn LTP's primary literature framing: a cellular model of pathological hyperalgesia/chronic pain, cited in §2 and §2.2 as the pathological counterpart to this document's tag/capture analogy).
- Zhang, H.-M. & Sandkühler, J. (2008) et al. — [Protein synthesis inhibition blocks the late-phase LTP of C-fiber evoked field potentials](https://journals.physiology.org/doi/full/10.1152/jn.01027.2002), *J. Neurophysiol.*
- Yang, Chen, Zhang & Sandkühler (2004). [Activation of spinal D1/D5 receptors induces late-phase LTP of C-fiber–evoked field potentials](https://journals.physiology.org/doi/full/10.1152/jn.01324.2004), *J. Neurophysiol.*
- [BDNF induces late-phase LTP of C-fiber evoked field potentials in rat spinal dorsal horn](https://sciencedirect.com/science/article/abs/pii/S0014488608002045), *Eur. J. Pain* / Neurosci. Lett.
- Grau, J.W. (2014). [Learning from the spinal cord: how the study of spinal cord plasticity informs learning theory](https://graulab.sites.tamu.edu/wp-content/uploads/sites/123/2018/09/Grau-2014-Learning-from-the-spinal-cord_-how-t.pdf).
- Crown, E.D. & Grau, J.W. (2001). [Preserving and restoring behavioral potential within the spinal cord using an instrumental training paradigm](https://pubmed.ncbi.nlm.nih.gov/11495955/).
- Ferguson, Crown & Grau (2006). [Group I metabotropic glutamate receptors control metaplasticity of spinal cord learning through a PKC-dependent mechanism](https://pmc.ncbi.nlm.nih.gov/articles/PMC2628285/).
- Ferguson, A.R., Huie, J.R., Crown, E.D., Baumbauer, K.M., Hook, M.A., Garraway, S.M., Lee, K.H., Hoy, K.C. & Grau, J.W. (2012). [Maladaptive spinal plasticity opposes spinal learning and recovery in spinal cord injury](https://pmc.ncbi.nlm.nih.gov/articles/PMC3468083/), *Front. Physiol.* (§2.3's synthesis reference: the same borrowed mechanism explaining adaptive spinal learning, in its own field's framing of why recovery sometimes fails).
- Drdla-Schütting, R., Benrath, J., Wunderbaldinger, G. & Sandkühler, J. (2012). [Erasure of a spinal memory trace of pain by a brief, high-dose opioid administration](https://pubmed.ncbi.nlm.nih.gov/22246779/), *Science* 335:235-238 (§2.4: active, triggered depotentiation of already-consolidated C-fiber LTP, reversing hyperalgesia in behaving animals).
- Asiedu, M.N., Tillu, D.V., Melemedjian, O.K., Shy, A., Sanoja, R., Bodell, B., Ghosh, S., Porreca, F. & Price, T.J. (2011). [Spinal protein kinase Mζ underlies the maintenance mechanism of persistent nociceptive sensitization](https://www.jneurosci.org/content/31/18/6646), *J. Neurosci.* (§2.4: the consolidated state requires ongoing PKMζ activity to persist at all — blocking it with ZIP collapses an already-established potentiation, mirroring the hippocampal PKMζ/ZIP maintenance literature independently).
- Shuai, Y., Lu, B., Hu, Y., Wang, L., Sun, K. & Zhong, Y. (2010). [Forgetting is regulated through Rac activity in Drosophila](https://www.cell.com/fulltext/S0092-8674(09)01630-4), *Cell* (§2.4: the canonical active-forgetting mechanism this section contrasts the spinal findings against — a dedicated Rac-dependent process, not passive decay).
- Cervantes-Sandoval, I. et al. (2016). [Scribble scaffolds a signalosome for active forgetting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4926877/), *Neuron* (§2.4: the dopamine-gated Scribble-Rac1-Cofilin "forgetting signalosome" in *Drosophila* mushroom body, cited for contrast with the spinal mechanisms above).
- Wolpaw, J.R. — overview of H-reflex operant conditioning and multi-site spinal plasticity: [Operant conditioning of H-reflex can correct a locomotor abnormality after SCI in rats](https://www.jneurosci.org/content/26/48/12537); [Memory traces in primate spinal cord produced by operant conditioning of H-reflex](https://journals.physiology.org/doi/abs/10.1152/jn.1989.61.3.563); [Operant conditioning of a spinal reflex can improve locomotion after SCI in humans](https://www.jneurosci.org/content/33/6/2365).
- Côté, M.-P., Azzam, G.A., Lemay, M.A., Zhukareva, V. & Houlé, J.D. (2011). [Activity-dependent increase in neurotrophic factors is associated with an enhanced modulation of spinal reflexes after spinal cord injury](https://pmc.ncbi.nlm.nih.gov/articles/PMC3037803/), *J. Neurotrauma*.
- Role of serotonin in locomotor CPG control and recovery after SCI: [The role of serotonin in the control of locomotor movements and strategies for restoring locomotion after SCI](https://pubmed.ncbi.nlm.nih.gov/24993627/); [The role of the serotonergic system in locomotor recovery after SCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC4321350/).
- Homeostatic/AMPAR-mediated plasticity after SCI: [AMPA receptor phosphorylation and synaptic colocalization on motor neurons drive maladaptive plasticity below complete SCI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4677690/); [Synaptic up-scaling preserves motor circuit output after chronic, natural inactivity](https://elifesciences.org/articles/30005).
- Turrigiano, G.G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. *Cell* 135(3):422-435 (bidirectional AMPAR up-/downscaling, the general framework §1.1 specializes to the spinal cord).
- Boulenguez, P. et al. (2010). [Down-regulation of the potassium-chloride cotransporter KCC2 contributes to spasticity after spinal cord injury](https://www.nature.com/articles/nm.2107), *Nature Medicine* (KCC2 loss as a homeostatic-downscaling failure; exercise/training partially restores KCC2 — the closest spinal-specific evidence that a homeostatic failure is training-reversible).
- For context, the hippocampal-side analogy this document mirrors: [Hippocampal Timescales as a Circuit Specification](https://github.com/max-talanov/tinyHippo/blob/main/hippocampal_timescales_as_circuit_spec.md).
