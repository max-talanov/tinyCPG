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

| Process | Time constant | Site / pathway | What it constrains for a spiking-network model |
|---|---|---|---|
| NMDAR/AMPAR-dependent synaptic induction | ms-scale, same coincidence physics as everywhere else in CNS | Ubiquitous (dorsal horn, motoneuron) | Sets the same fast coincidence-detection floor as the hippocampal case — nothing spinal-specific here. |
| Early-phase spinal LTP (E-LTP) | 1–3 h; NMDA-receptor-dependent; **protein-synthesis-independent** | Dorsal-horn C-fiber synapses, induced by afferent tetanic stimulation (Sandkühler & Liu 2003) | A transient, spontaneously-decaying potentiation that exists *before* any stabilizing signal arrives — the direct spinal analog of the hippocampal "tag." |
| Late-phase spinal LTP (L-LTP) | Onset by ~3 h; **requires ongoing protein synthesis**; selectively induced/occluded by BDNF and spinal D1/D5 dopamine-receptor activation | Same dorsal-horn synapses | A discrete regime change from decaying to non-decaying, gated by a *separate* neuromodulatory signal arriving before the tag decays — structurally identical to hippocampal "capture," demonstrated in spinal tissue with no hippocampus involved. |
| Spinal instrumental (contingency-based) learning | Acquisition within a single session (tens of minutes); consolidation requires new protein synthesis over the following hours; **the sign of what consolidates is set by response contingency**, not by activity level | Interneuron + motoneuron circuits caudal to a complete spinal transection — no brain involvement (Grau et al.) | Whether a training bout produces adaptive potentiation or an active, protein-synthesis-dependent *suppression* of future learning capacity depends on whether the outcome was behaviorally successful — a bidirectional gate, not a one-way accumulator. |
| H-reflex operant conditioning, Phase I | 1–2 days; small magnitude | Ia-afferent → motoneuron monosynaptic pathway + interneurons (Wolpaw) | The fast, labile component of a two-phase learning process running entirely on the pathway this model calls `Ia→RG`. |
| H-reflex operant conditioning, Phase II | 6–7 weeks; large, stable; **multi-site** (motoneuron firing threshold and axonal conduction velocity, GABAergic terminal density on the motoneuron, spinal interneuron changes) | Same pathway | The slow, structurally-consolidated component — days-to-weeks systems-consolidation territory, the same absolute regime the hippocampal document assigns to cortical redistribution, but reached entirely within the spinal cord. Demonstrated to correct locomotor asymmetry after spinal cord injury when combined with training — this is the closest published result to the model's own stated rehab goal. |
| Activity-dependent step-training neurotrophin upregulation | Daily training over days–weeks; **task-specific** (step-training and cycle-training produce different BDNF/NT-3/NT-4 profiles and different dorsal-horn/intermediate-gray neuron counts) | Lumbar spinal cord below a lesion (Côté, Azzam, Lemay, Zhukareva & Houlé 2011) | Argues against a single generic "activity level" gate — the training *modality*, not just its amount, sets what gets reinforced. Directly relevant to this model's distinction between `--cut-trigger force` (stance-loading-driven) and the paced-clock modes. |
| Serotonergic (5-HT) neuromodulatory gating of CPG excitability | Acute: seconds (state-dependent gating of whether rhythmic bursting can occur at all); chronic: days–weeks (5-HT receptor sensitivity changes after injury, partly restored by training + serotonergic agonists) | Brainstem-to-spinal monoaminergic projections onto CPG interneurons and motoneurons | The model's tonic `BS_REGULAR_HZ` drive is a generic reticulospinal proxy; the literature's actual candidate for "descending drive that gates whether the spinal rhythm-generator can run" is specifically serotonergic/noradrenergic, and its *sensitivity* itself changes with training — a second, slower plasticity axis this model does not yet represent (flagged here, not addressed by the plan in §3). |
| Homeostatic synaptic scaling after deafferentation / chronic inactivity | Hours–days; mediated by synaptic insertion of GluA2-lacking, Ca²⁺-permeable AMPA receptors | Motoneurons below an injury or period of inactivity | The literature's own mechanism for "what happens when descending/afferent drive is chronically reduced" is an active receptor-composition change with its own kinetics — not a static gain multiplier. Directly relevant to why the model's `--wmax-ia-unloaded` gain-based unloading-rescue attempts plateaued (see [CLAUDE.md](CLAUDE.md), "Core architecture fix" section): the biology's own fix for reduced afferent drive is itself a plasticity process with a time constant, not an instantaneously-applied cap relaxation. |

## 2. Tag-and-capture and contingency-gated consolidation in spinal circuits

Three independent literatures converge on the same two-stage structure the
hippocampal document derives from STC — a labile, spontaneously-decaying
change, stabilized only by a second, slower, separately-gated signal — without
citing each other or the hippocampal literature:

**Molecular substrate (Sandkühler & Liu 2003; Yang, Chen, Zhang & Sandkühler 2004; Zhang & Sandkühler 2008).**
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
no cortex or hippocampus involved.

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
  plasticity in the same sense (the closest analog, serotonergic
  receptor-sensitivity change after injury, is a separate, slower axis — see
  the last two rows of §1's table, not addressed further here). Applying the
  same mechanism to `BS→RG` for architectural uniformity is a modeling choice
  this document does not find direct literature support for, one way or the
  other — worth flagging explicitly before the plan commits to it.

Implementation (state variables, update equations, where in
`cpg_2legs_fast.py`'s sim loop this would live) is intentionally left to the
separate implementation plan, not this document.

## References

- Sandkühler, J. & Liu, X. (2003). Induction of long-term potentiation at spinal synapses by noxious stimulation or nerve injury. *Eur. J. Neurosci.*
- Zhang, H.-M. & Sandkühler, J. (2008) et al. — [Protein synthesis inhibition blocks the late-phase LTP of C-fiber evoked field potentials](https://journals.physiology.org/doi/full/10.1152/jn.01027.2002), *J. Neurophysiol.*
- Yang, Chen, Zhang & Sandkühler (2004). [Activation of spinal D1/D5 receptors induces late-phase LTP of C-fiber–evoked field potentials](https://journals.physiology.org/doi/full/10.1152/jn.01324.2004), *J. Neurophysiol.*
- [BDNF induces late-phase LTP of C-fiber evoked field potentials in rat spinal dorsal horn](https://sciencedirect.com/science/article/abs/pii/S0014488608002045), *Eur. J. Pain* / Neurosci. Lett.
- Grau, J.W. (2014). [Learning from the spinal cord: how the study of spinal cord plasticity informs learning theory](https://graulab.sites.tamu.edu/wp-content/uploads/sites/123/2018/09/Grau-2014-Learning-from-the-spinal-cord_-how-t.pdf).
- Crown, E.D. & Grau, J.W. (2001). [Preserving and restoring behavioral potential within the spinal cord using an instrumental training paradigm](https://pubmed.ncbi.nlm.nih.gov/11495955/).
- Ferguson, Crown & Grau (2006). [Group I metabotropic glutamate receptors control metaplasticity of spinal cord learning through a PKC-dependent mechanism](https://pmc.ncbi.nlm.nih.gov/articles/PMC2628285/).
- Wolpaw, J.R. — overview of H-reflex operant conditioning and multi-site spinal plasticity: [Operant conditioning of H-reflex can correct a locomotor abnormality after SCI in rats](https://www.jneurosci.org/content/26/48/12537); [Memory traces in primate spinal cord produced by operant conditioning of H-reflex](https://journals.physiology.org/doi/abs/10.1152/jn.1989.61.3.563); [Operant conditioning of a spinal reflex can improve locomotion after SCI in humans](https://www.jneurosci.org/content/33/6/2365).
- Côté, M.-P., Azzam, G.A., Lemay, M.A., Zhukareva, V. & Houlé, J.D. (2011). [Activity-dependent increase in neurotrophic factors is associated with an enhanced modulation of spinal reflexes after spinal cord injury](https://pmc.ncbi.nlm.nih.gov/articles/PMC3037803/), *J. Neurotrauma*.
- Role of serotonin in locomotor CPG control and recovery after SCI: [The role of serotonin in the control of locomotor movements and strategies for restoring locomotion after SCI](https://pubmed.ncbi.nlm.nih.gov/24993627/); [The role of the serotonergic system in locomotor recovery after SCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC4321350/).
- Homeostatic/AMPAR-mediated plasticity after SCI: [AMPA receptor phosphorylation and synaptic colocalization on motor neurons drive maladaptive plasticity below complete SCI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4677690/); [Synaptic up-scaling preserves motor circuit output after chronic, natural inactivity](https://elifesciences.org/articles/30005).
- For context, the hippocampal-side analogy this document mirrors: [Hippocampal Timescales as a Circuit Specification](https://github.com/max-talanov/tinyHippo/blob/main/hippocampal_timescales_as_circuit_spec.md).
