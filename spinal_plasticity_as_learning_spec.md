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
| Task-specific spinal locomotor learning (train-to-stand vs. train-to-step) | Complete-transection cats trained daily to either stand or step relearn specifically the trained task — stand-trained cats stand well but step poorly, and step-trained cats step well but stand poorly — the clearest behavioral demonstration that the isolated lumbar CPG itself, not just a single reflex pathway, can be shaped by training (de Leon, Hodgson, Roy & Edgerton 1998; framed explicitly as spinal motor learning by Edgerton, Roy, de Leon, Tillakaratne & Hodgson 1997). **Contested**: a later study found both standing and locomotion recover under non-task-specific stimulation, or with no training at all, attributing recovery to a general return of spinal circuit excitability rather than task-specific activity-dependent encoding (Harnie, Doelman, de Vette, Audet, Desrochers, Gaudreault & Frigon 2019) — an open controversy, not resolved here. | Daily training, ~8–12 weeks to a stable task-specific outcome | Lumbar locomotor CPG circuitry below a complete thoracic spinal transection (cat) | The strongest available evidence that a CPG core like this model's `RG-E`/`RG-F` — not just an afferent pathway — is a legitimate target for a "trained skill" framing. But the Harnie et al. 2019 contestation means this row shouldn't be read as an uncontested green light for CPG-level consolidation the way Wolpaw's Ia→motoneuron pathway is (§3's `Ia→RG` mapping stands on Wolpaw's result specifically, not on this one). |

*Table 1a — The ten rows plotted in Fig. 1a above: baseline physiology plus every mechanism whose documented outcome in the cited literature is functional preservation or recovery. Two rows (spinal instrumental learning, serotonergic chronic axis) are one half of a bidirectional mechanism — their pathological counterpart is Table 1b, not absent, just not adaptive. **Fig. 1a's image predates this row and does not yet depict it** — regenerating the figure is a small, not-yet-done follow-up, flagged here rather than left silently out of sync.*

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

The upscaling row above is one direction of a bidirectional mechanism —
chronic silencing drives synaptic **upscaling**, chronic hyperactivity drives
synaptic **downscaling**, the same AMPAR-trafficking toolkit running in
opposite directions to hold network activity near a set point (Turrigiano
2008). The rat spinal cord shows the upscaling side cleanly (deafferentation
→ GluA2-lacking, Ca²⁺-permeable AMPAR insertion in motoneurons, already in
the table). The downscaling side is more interesting than a simple mirror
image: after SCI, the dominant documented failure mode is not "excitatory
synapses fail to scale down" but **inhibitory efficacy itself collapsing** —
KCC2, the potassium-chloride cotransporter that keeps the Cl⁻ reversal
potential hyperpolarized enough for GABA/glycine to inhibit, is
downregulated in motoneuron membranes after SCI (Boulenguez et al. 2010),
functionally a failed downscaling response — excitability stays elevated
because the compensatory brake never engages — and a well-established
mechanistic account of post-SCI spasticity.

**This closes the loop with the document's rehab framing rather than sitting
outside it**: the same Boulenguez-line literature reports that locomotor
training partially restores KCC2 expression and reduces spasticity, i.e. the
homeostatic failure is *training-reversible*, not fixed — precisely the
"does training restore lost function" question this whole document exists to
give a mechanistic answer to, on a completely different pathway (chloride
homeostasis) than the tag-and-capture story in §2.

![Two-panel chart: (a) synaptic AMPAR weight rising smoothly from baseline to a compensated plateau over about three days after deafferentation (healthy upscaling); (b) motoneuron KCC2/inhibitory efficacy dropping sharply at spinal cord injury, then either staying flat at the reduced floor with no training (dashed red, spasticity persists) or partially recovering toward baseline over several weeks with locomotor training (solid green) (pathological downscaling failure).](spinal_scaling_dynamics.png)

*Fig. 2 — Bidirectional homeostatic scaling. **(2a) Healthy** (left): the upscaling response to deafferentation — a single, reliably-reported trajectory; there is no "blocked" condition in the cited literature, so only one trace is shown. **(2b) Pathological** (right): KCC2 loss (downscaling failure) after SCI, training vs. none — both traces take the identical acute post-injury drop; they diverge only in whether locomotor training is applied afterward, which is the direct spinal analog of "capture" rescuing an otherwise-lost trace, the same normal-vs-blocked contrast the hippocampal document's Fig. 2 uses for capture. Curve shapes are illustrative (exponential fits to the qualitative time course each citation reports), not digitized data.*

### 1.4 Which of Fig. 1a's rows are motor-skill-formation mechanisms specifically

Table 1a's scope statement is "baseline physiology plus every mechanism whose
documented outcome is functional preservation or recovery" — deliberately
broader than "motor skill formation." Checking the ten rows against the
narrower, spinal-specific definition of that term (a lasting,
practice-/contingency-dependent change in spinal circuit output that
persists without cortical involvement — distinct from cortical motor-skill
learning's own mechanism, new dendritic spines and map-level LTP/LTD in
motor cortex) sorts them into four groups, not one:

| Process | Category | Why |
|---|---|---|
| Spinal instrumental learning — contingent (Grau) | **Core skill-formation mechanism** | A trained, lasting, contingency-gated change in spinal reflex output — the literal definition. |
| H-reflex conditioning, Phase I | **Core skill-formation mechanism** | A trained, lasting change in a spinal reflex, by construction. |
| H-reflex conditioning, Phase II | **Core skill-formation mechanism** | The consolidated, multi-site version of the same trained change. |
| Task-specific spinal locomotor learning (de Leon/Roy/Edgerton) | **Core skill-formation mechanism — contested** | The only row directly about the CPG's own output pattern, not an afferent pathway or a withdrawal reflex; but its task-specificity claim is itself disputed (Harnie et al. 2019, see Table 1a) — counted here as core, flagged as unsettled, not a second confirmed Wolpaw-grade result. |
| Homeostatic AMPAR upscaling | Enabling/compensatory | Restores lost excitability after deafferentation; carries no information about *which* pattern was learned. |
| Serotonergic gating, chronic (training-restored) | Enabling/compensatory | Gates whether the CPG can burst at all; same reasoning as its acute counterpart. |
| Step-training neurotrophin upregulation (Côté) | Molecular correlate, not the mechanism itself | Row is framed as a BDNF/NT-3/NT-4 readout; the actual skill-refinement outcome — multisegmental network reorganization and reduced muscle co-contraction with training — isn't named. Flagged as a content gap here, not corrected in Table 1a itself. |
| NMDAR/AMPAR-dependent induction | Basic substrate | Generic coincidence-detection floor every mechanism above (skill-forming or not) runs on top of. |
| Short-term plasticity | Basic substrate | Same — a physics constraint, not a learning mechanism. |
| Serotonergic gating, acute | Basic substrate | Gates *whether* bursting can occur, not *which* pattern is learned. |

Net: **4 of the 10 rows are motor-skill-formation mechanisms in the strict
spinal-specific sense (one of them contested); the other 6 are the enabling
or basic-substrate processes those mechanisms operate within.** This
doesn't make Fig. 1a wrong — its stated scope already includes
prerequisites, not just the skill-encoding step — but a reader treating
"healthy spinal plasticity processes" as synonymous with "how the spinal
cord forms a motor skill" would still overcount, and one of the four core
rows shouldn't be leaned on as settled evidence either way.

**Caveat — a tempting addition, checked and left out.** Bizzi & Giszter's
spinal motor primitives / muscle synergies (force-field modules the spinal
cord combines to build movements, structurally close to this model's own
extensor/flexor half-center split) look like a natural missing row. Checked
directly before proposing it: the primitives themselves are reported as
largely fixed and conserved from early development rather than something
training forms anew — "motor primitives are determined in early
development and are then robustly conserved into adulthood"
([PNAS](https://www.pnas.org/doi/10.1073/pnas.1821455116)) — with
training/injury changing the excitatory/inhibitory *recruitment weighting*
of primitives, not the primitives' own structure
([Giszter 2013](https://nyaspubs.onlinelibrary.wiley.com/doi/abs/10.1111/nyas.12065)).
That reweighting is arguably already covered by the homeostatic-scaling and
serotonergic-gating rows already in the table. **Not added as a 10th row**
— it would misrepresent something conserved/fixed as a plasticity process
with its own induction timescale, the same category error this document
already treats carefully elsewhere (e.g. §2.2's careful separation of
"borrowed molecular substrate" from "the substrate's own primary framing").
It remains a useful *structural* analogy for the model's RG-E/RG-F
architecture, not a literature-grounded addition to Table 1a.

*(Broader locomotor-training review consulted for this check, beyond the
Côté citation already in Table 1a:
[A Review on Locomotor Training after SCI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4879237/),
which documents the multisegmental-reorganization/reduced-co-contraction
outcome flagged as the Côté-row gap above; also
[Cortical circuit dynamics underlying motor skill learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10641381/)
for the cortical-vs-spinal skill-formation contrast this section's opening
definition draws on.)*

## 2. Tag-and-capture and contingency-gated consolidation in spinal circuits

Three independent rat literatures converge on the same two-stage structure
the hippocampal document derives from STC — a labile, decaying change,
stabilized only by a separate, slower gating signal — without citing each
other or the hippocampal work:

**Molecular substrate.** Dorsal-horn E-LTP (Sandkühler & Liu 1998) behaves
like a decaying tag: NMDA-receptor-dependent, present within minutes, gone
within hours unless converted to L-LTP — a discrete regime change blocked by
protein-synthesis inhibitors and gated by spinal D1/D5 dopamine receptors or
exogenous BDNF (Yang et al. 2004; Zhang & Sandkühler 2008). **This
substrate's own primary framing is pathological, not adaptive**: the same
mechanism is the standard cellular model of central sensitization in chronic
pain (Ruscheweyh et al. 2011). Borrowed here purely as a motor-consolidation
analogy — the retention rule itself is neutral machinery, not "good" or
"bad" (§2.1/§2.2 below).

**Behavioral gate.** In transected rats, response-contingent training (a
limb flexion that terminates shock) produces NMDAR/BDNF/protein-synthesis-
dependent potentiation that outlasts the session (Grau 2014; Crown & Grau
2001). The gate is **bidirectional and set by contingency, not activity
level**: identical, uncontrollable shock produces the opposite outcome —
active, protein-synthesis-dependent suppression of future learning, via
group-I mGluR/PKC signaling, not merely an absence of reinforcement
(Ferguson, Crown & Grau 2006).

**Structural two-phase consolidation.** H-reflex operant conditioning in
rats — training the monosynaptic Ia-afferent-to-motoneuron reflex directly —
shows a fast, small Phase I (1–2 days) and a slow, large, multi-site Phase II
(6–7 weeks: motoneuron firing threshold, GABAergic terminal density,
interneuron changes) (Chen, Chen, Liu & Wolpaw 2006). Up-conditioning
**corrects locomotor asymmetry after spinal cord injury in rats** — the
closest published result to this model's own debug goal, achieved by
consolidating exactly the `Ia→RG` pathway this model represents.

**Synthesis.** All three reproduce the same shape: a fast, decaying change; a
slower, gated conversion to stability; and a sign that is not guaranteed
positive — an uncontrollable contingency actively suppresses, not just fails
to reinforce. This is the spinal cord's own version of the STC problem, a
better match for a locomotor CPG than the hippocampal case, since two of the
three literatures were generated in this kind of circuit.

Every mechanism above is two-directional. §2.1/§2.2 list the two directions
as separate tables, matching the §1.1/§1.2 split.

### 2.1 Healthy / adaptive directions

| Mechanism | What the adaptive direction looks like |
|---|---|
| Homeostatic scaling (§1.1) | Upscaling after deafferentation restores excitability lost to reduced afferent drive |
| Spinal instrumental learning (Grau, §1.1) | Contingent (response-produced) outcome → NMDAR/BDNF/protein-synthesis-dependent potentiation that outlasts the session |
| Dorsal-horn E-/L-LTP (Sandkühler, §1.2) | Borrowed as this document's molecular-substrate analogy for a motor-consolidation tag/capture rule — not this mechanism's own primary framing, see §2.2 |
| Serotonergic gating, chronic axis (§1.1) | Partly restored by locomotor training and serotonergic agonists post-injury |
| H-reflex conditioning (§1.1) | Up-conditioning corrects locomotor asymmetry after SCI in rats |

*Table 2a — The adaptive direction of each two-directional mechanism from §1-2. Pairs row-for-row with Table 2b except the last, which Table 2b leaves blank rather than inventing a counterpart.*

### 2.2 Pathological / maladaptive directions

| Mechanism | What the maladaptive direction looks like |
|---|---|
| Homeostatic scaling (§1.2) | **Downscaling failure**: KCC2 loss after SCI leaves inhibition too weak, producing spasticity (Boulenguez et al. 2010) — training partially reverses it |
| Spinal instrumental learning (Grau, §1.2) | Non-contingent (uncontrollable) outcome at the *identical* intensity → active, protein-synthesis-dependent **suppression** of future learning capacity (Ferguson, Crown & Grau 2006) |
| Dorsal-horn E-/L-LTP (Sandkühler, §1.2) | **Its own primary literature framing**: the standard cellular model of central sensitization underlying hyperalgesia and chronic pain (Ruscheweyh et al. 2011) |
| Serotonergic gating, chronic axis (§1.2) | Persistently altered CPG sensitivity to 5-HT if left untreated after injury |
| H-reflex conditioning | *No entry* — no specific pathological counterpart in the literature cited here (also why Table 1b has no H-reflex row) |

*Table 2b — The maladaptive direction, where the cited literature documents one. Gate signal and target synapse determine the outcome, not a separate "adaptive" vs. "maladaptive" machinery — §3 closes the loop with `cpg_2legs_fast.py`'s own observed failure modes.*

### 2.3 Maladaptive plasticity after spinal cord injury: synthesis

Table 2b's four rows are not independent side effects — in rat SCI models
they are commonly reported together, a pattern the literature calls
**maladaptive plasticity**: retention rules that persist but now degrade
function because injury changed the gating signal or target synapse, not
because plasticity itself changed kind (Ferguson et al. 2012: *"maladaptive
spinal plasticity opposes spinal learning and recovery in spinal cord
injury"* — the same mechanism this document borrows from Grau to explain
adaptive learning, in its own field's account of why recovery sometimes
fails).

1. **Spasticity** via KCC2 loss (Boulenguez et al. 2010).
2. **Central/neuropathic pain** via dorsal-horn LTP consolidation (Sandkühler
   & Liu 1998; Ruscheweyh et al. 2011).
3. **Impaired further learning** under uncontrollable training (Ferguson,
   Crown & Grau 2006).
4. **Altered serotonergic sensitivity** if left untreated.

None is a separate disease process — each is the Table 2a mechanism tipped
toward the maladaptive branch by what the lesion changed, matching the
general rat SCI literature's multiple-hit picture (motoneurons,
interneurons, and afferents concurrently, not one dominant cause).

**Consequence.** `--consolidate` (`cpg_2legs_fast.py`) is exactly this
retention rule, and inherited the same double edge — tuned one way it
reproduces Table 2a's outcomes, tuned another it reproduces Table 2b's, with
direct model-level counterparts found empirically (§3, closing paragraph).

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
- Chen, Y., Chen, X.Y., Liu, Z. & Wolpaw, J.R. (2006). H-reflex operant conditioning and its multi-site plasticity in the rat.
- Wolpaw, J.R. — [Operant conditioning of H-reflex can correct a locomotor abnormality after spinal cord injury in rats](https://www.jneurosci.org/content/26/48/12537).
- de Leon, R.D., Hodgson, J.A., Roy, R.R. & Edgerton, V.R. (1998). [Full weight-bearing hindlimb standing following stand training in the adult spinal cat](https://journals.physiology.org/doi/full/10.1152/jn.1998.80.1.83), *J. Neurophysiol.* (Table 1a/§1.4: the train-to-stand-vs-train-to-step task-specificity finding — the closest published evidence that the lumbar CPG itself, not just an afferent pathway, is trainable.)
- Edgerton, V.R., Roy, R.R., de Leon, R.D., Tillakaratne, N. & Hodgson, J.A. (1997). [Does motor learning occur in the spinal cord?](https://journals.sagepub.com/doi/10.1177/107385849700300510), *The Neuroscientist* (same group's own explicit framing of the above as spinal motor learning).
- Harnie, J., Doelman, A., de Vette, E., Audet, J., Desrochers, E., Gaudreault, N. & Frigon, A. (2019). [The recovery of standing and locomotion after spinal cord injury does not require task-specific training](https://elifesciences.org/articles/50134), *eLife* (the contestation flagged alongside the de Leon/Roy/Edgerton row in Table 1a/§1.4 — recovery attributed to a general return of spinal excitability, not task-specific encoding).
- Côté, M.-P., Azzam, G.A., Lemay, M.A., Zhukareva, V. & Houlé, J.D. (2011). [Activity-dependent increase in neurotrophic factors is associated with an enhanced modulation of spinal reflexes after spinal cord injury](https://pmc.ncbi.nlm.nih.gov/articles/PMC3037803/), *J. Neurotrauma*.
- Role of serotonin in locomotor CPG control and recovery after SCI: [The role of serotonin in the control of locomotor movements and strategies for restoring locomotion after SCI](https://pubmed.ncbi.nlm.nih.gov/24993627/); [The role of the serotonergic system in locomotor recovery after SCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC4321350/).
- Homeostatic/AMPAR-mediated plasticity after SCI: [AMPA receptor phosphorylation and synaptic colocalization on motor neurons drive maladaptive plasticity below complete SCI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4677690/); [Synaptic up-scaling preserves motor circuit output after chronic, natural inactivity](https://elifesciences.org/articles/30005).
- Turrigiano, G.G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. *Cell* 135(3):422-435 (bidirectional AMPAR up-/downscaling, the general framework §1.3 specializes to the rat spinal cord).
- Boulenguez, P. et al. (2010). [Down-regulation of the potassium-chloride cotransporter KCC2 contributes to spasticity after spinal cord injury](https://www.nature.com/articles/nm.2107), *Nature Medicine* (KCC2 loss as a homeostatic-downscaling failure; exercise/training partially restores KCC2 — the closest spinal-specific evidence that a homeostatic failure is training-reversible).
- [Cortical circuit dynamics underlying motor skill learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10641381/) (§1.4: cortical motor-skill learning's own mechanism — dendritic spine formation and map-level LTP/LTD — used as the contrast that defines the narrower, spinal-specific sense of "motor skill formation" §1.4 checks Table 1a's rows against).
- [Motor primitives are determined in early development and are then robustly conserved into adulthood](https://www.pnas.org/doi/10.1073/pnas.1821455116), *PNAS* (§1.4 caveat: the finding that ruled out adding Bizzi/Giszter spinal motor primitives as a 10th Table 1a row — primitives are largely fixed substrate, not something training forms).
- Giszter, S.F. (2013). [Motor primitives — new data and future questions](https://nyaspubs.onlinelibrary.wiley.com/doi/abs/10.1111/nyas.12065), *Ann. N.Y. Acad. Sci.* (§1.4 caveat: training/injury changes primitive *recruitment weighting*, not primitive structure itself).
- [A Review on Locomotor Training after SCI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4879237/) (§1.4: documents the multisegmental network reorganization and reduced muscle co-contraction outcome of step-training, flagged as missing from Table 1a's Côté row, which currently states only its BDNF/NT-3/NT-4 molecular correlate).
- For context, the hippocampal-side analogy this document mirrors: [Hippocampal Timescales as a Circuit Specification](https://github.com/max-talanov/tinyHippo/blob/main/hippocampal_timescales_as_circuit_spec.md).
