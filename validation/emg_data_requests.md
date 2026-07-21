# EMG data requests for tinyCPG model validation

Generated from the triage table (Google Sheet, `gid=51842667`). Filter applied:
papers ≤ 20 years old (2006–2026), EMG-relevant, spinal/SCI rat, **no dataset
publicly available** (rows already marked "Available" were excluded in the
sheet for independent reasons — single-leg data, uninjured animals, or
kinematics/synergies without raw EMG — so they don't help our validation
either way). The 1991 paper is out of the 20-year window and was already
flagged in the sheet as too old to pursue.

12 qualifying papers grouped into **8 emails** by corresponding author, to
avoid sending the same lab multiple redundant requests.

**Before sending:** these are drafts. Fill in `[Your name]` / `[affiliation]`
/ `[contact]` in the signature block, and adjust if you want to mention
IRCCS or a specific manuscript title. I have not sent anything — these are
text for you to review, personalize, and send yourself.

---

## 1. Grégoire Courtine (3 papers)
**To:** gregoire.courtine@epfl.ch
**Cc:** silvestro.micera@epfl.ch, nikolaus.wenger@charite.de
**Subject:** Data request — bilateral hindlimb EMG from your spinal rat locomotion studies

Dear Prof. Courtine,

I am writing regarding three of your published studies on spinal cord
injury (SCI) and neuromodulation in rats:

- (2014), *Closed-loop neuromodulation of spinal sensorimotor circuits
  controls refined locomotion after complete spinal cord injury*, Science
  Translational Medicine.
- (2016), *Mechanisms Underlying the Neuromodulation of Spinal Circuits
  for Correcting Gait and Balance Deficits after Spinal Cord Injury*,
  Neuron.
- (2009), *Transformation of nonfunctional spinal circuits into functional
  states after the loss of brain input*, Nature Neuroscience.

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, in which cutaneous
and proprioceptive synapses onto the rhythm-generating half-centres
self-organise via spike-timing-dependent plasticity. The model reproduces,
qualitatively, the graded loss and epidural-stimulation-assisted recovery
of hindlimb stepping across weight-bearing conditions (full support,
partial/toe-stepping support, and fully unloaded/air-stepping) that your
group has characterised experimentally.

We would like to validate the model's muscle-activity predictions
quantitatively, and are therefore asking whether bilateral hindlimb
flexor/extensor EMG recordings from any of the above studies — raw or
processed, ideally spanning more than one loading condition — might be
available to us for this purpose. We noted the data-availability statement
in the 2014 paper indicating that data can be made available under a
material transfer agreement, and would be glad to follow whatever process
you require.

Thank you very much for considering this request, and please let us know
if any further information about our project would be helpful.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 2. Marina Martinez & Marco Bonizzato (2 papers)
**To:** marina.martinez@umontreal.ca, marco.bonizzato@polymtl.ca
**Subject:** Data request — hindlimb EMG from your spinal cord injury / neuroprosthesis studies

Dear Dr. Martinez and Dr. Bonizzato,

I am writing regarding two of your recent papers:

- (2025), *Combining cortical and spinal stimulation maximizes the
  improvement of gait after spinal cord injury*, iScience.
- (2024), *Cortical neuroprosthesis-mediated functional ipsilateral
  control of locomotion in rats with spinal cord hemisection*, eLife.

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, in which sensory
synapses onto the spinal rhythm generators self-organise through
spike-timing-dependent plasticity. The model is aimed at understanding how
spinal circuits adapt to graded loss of descending and sensory input after
spinal cord injury, and how stimulation-based interventions might restore
coordinated stepping.

We would like to validate the model's predicted muscle-activity patterns
against real recordings, and are therefore asking whether bilateral
hindlimb EMG data (flexor/extensor pairs) from either study might be
available to us, even in a de-identified or partial form. We saw that the
iScience paper notes additional information is available from the lead
contact upon request, and would be very grateful if that could extend to
the underlying EMG traces.

Thank you for considering this, and please don't hesitate to ask if
further detail about our modelling work would help.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 3. V. Reggie Edgerton (2 papers)
**To:** vre@ucla.edu
**Subject:** Data request — hindlimb/forelimb EMG from your spinal rat stepping studies

Dear Prof. Edgerton,

I am writing regarding two of your studies on locomotor recovery in
spinal rats:

- (2013), *Neuromodulation of motor-evoked potentials during stepping in
  spinal rats*, Journal of Neurophysiology.
- (2012), *Forelimb EMG-based trigger to control an electronic spinal
  bridge to enable hindlimb stepping after a complete spinal cord lesion
  in rats*, Journal of NeuroEngineering and Rehabilitation.

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, in which cutaneous
and proprioceptive synapses self-organise via spike-timing-dependent
plasticity, and which reproduces the graded degradation and
epidural-stimulation-assisted rescue of stepping under reduced
weight-bearing that your group's work has been central in establishing.

We would very much like to validate the model's muscle-activity
predictions against real EMG recordings, and are writing to ask whether
hindlimb (and/or forelimb, for the 2012 study) EMG data from either paper
might be available to us for this purpose, in whatever form is convenient
to share.

Thank you for considering this request.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 4. Michał Zawiślak (1 paper)
**To:** michal.zawiski@pw.edu.pl
**Subject:** Data request — gait/EMG data from your spinal rat measurement system

Dear Dr. Zawiślak,

I am writing regarding your paper *The System for Measuring Gait
Parameters and Rehabilitation of Spinal Rats* (2024, Acta Physica Polonica
A).

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, self-organising via
spike-timing-dependent plasticity, and are looking to validate its
predictions against real gait and, if available, EMG recordings from
spinal-injured rats across different weight-bearing/support conditions.

Since your paper describes a system built specifically for measuring gait
parameters in spinal rats, we wanted to ask whether any recorded datasets
(gait kinematics and/or EMG) from that system might be available for us
to use as a validation reference, even in a limited or example form.

Thank you very much for considering this, and please let me know if I can
provide any further detail about our project.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 5. Mahima Sharma & M. Fallahrad (1 paper)
**To:** mahima.sharma0@gmail.com, mfallahrad@ccny.cuny.edu
**Subject:** Data request — spinal cord stimulation recordings from your ESAP study

Dear Dr. Sharma and Dr. Fallahrad,

I am writing regarding your paper *Novel Evoked Synaptic Activity
Potentials (ESAPs) Elicited by Spinal Cord Stimulation* (2023, eNeuro).

We are developing a closed-loop computational model (NEST simulator) of
the rat spinal central pattern generator, including an epidural
electrical stimulation analogue, and are seeking real physiological
recordings to validate the model's responses to spinal cord stimulation.

We noted that your custom MATLAB analysis code is available on request,
and wanted to ask whether the underlying evoked-response or EMG datasets
themselves might also be shareable for validation purposes in our
modelling work.

Thank you for considering this request.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 6. David Magnuson & Anastasia Keller (1 paper)
**To:** dsmagn01@louisville.edu, anastasia.keller@ucsf.edu
**Subject:** Data request — hindlimb EMG data from your rat SCI stretch-reflex study

Dear Dr. Magnuson and Dr. Keller,

I am writing regarding your paper *Electromyographic patterns of the rat
hindlimb in response to muscle stretch after spinal cord injury* (2018,
Spinal Cord).

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, in which
proprioceptive and cutaneous synapses self-organise via spike-timing-dependent
plasticity, and we are seeking real hindlimb EMG recordings from
spinal-injured rats to validate the model's flexor/extensor
activity patterns quantitatively.

As your study specifically recorded hindlimb EMG in response to muscle
stretch after SCI, we wanted to ask whether the underlying EMG traces
might be available to us for this purpose, in whatever form is convenient
to share.

Thank you very much for considering this, and please let us know if
further information about our project would help.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 7. Olivier Alluin & Serge Rossignol (1 paper)
**To:** olivier.alluin@aktantis.com, serge.rossignol@umontreal.ca
**Subject:** Data request — EMG/kinematic recordings from your spinal rat treadmill-training study

Dear Dr. Alluin and Prof. Rossignol,

I am writing regarding your paper *Inducing hindlimb locomotor recovery in
adult rat after complete thoracic spinal cord section using repeated
treadmill training with perineal stimulation only* (2015, Journal of
Neurophysiology).

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, self-organising via
spike-timing-dependent plasticity, and are seeking real hindlimb EMG
and/or kinematic recordings from spinal-transected rats across training
sessions to validate the model's predicted recovery of coordinated
stepping.

We understand from the published record that a movie of the locomotor
behaviour accompanies this study; we wanted to ask whether the underlying
EMG or kinematic datasets themselves might also be available for
validation purposes in our modelling work.

Thank you for considering this request.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## 8. Ronaldo M. Ichiyama (1 paper)
**To:** r.m.ichiyama@leeds.ac.uk
**Subject:** Data request — EMG data from your step-training spinal rat study

Dear Dr. Ichiyama,

I am writing regarding your paper *Step Training Reinforces Specific
Spinal Locomotor Circuitry in Adult Spinal Rats* (2008, Journal of
Neuroscience).

We are developing a closed-loop computational model (NEST simulator) of
the two-legged rat spinal central pattern generator, in which sensory
synapses onto the spinal rhythm generators self-organise via
spike-timing-dependent plasticity, and are seeking real hindlimb EMG
recordings from step-trained spinal rats to validate the model's
predicted muscle-activity patterns.

We would be very grateful if the EMG data underlying this study — or any
comparable dataset from your group's work on spinal locomotor training —
might be available to us for this purpose.

Thank you for considering this request.

Best regards,
[Your name]
[Your affiliation]
[Contact email]

---

## Papers excluded from the request list (for reference)

| Paper | Year | Reason |
|---|---|---|
| EMG patterns of rat ankle extensors and flexors during treadmill locomotion and swimming | 1991 | Outside the 20-year window; sheet notes "too old to obtain data" |
| Nanogenerator Neuromodulation... | — | Dataset available but single-leg only |
| EMUsort Rat Datasets (Dandi) | — | Dataset available but uninjured (intact) rats |
| Multi-pronged neuromodulation... (Nat Commun 2021) | — | Dataset available but single-leg only |
| AnMod_Neuro (GitHub homework repo) | — | Not a primary dataset; no EMG data |
| Spinal control of locomotion before/after SCI (Danner lab) | — | Dataset available (kinematics) but no EMG |
| Dataset of measured kinematics... (Zenodo) | — | Kinematics/synergies only, no EMG |
| Operation of spinal sensorimotor circuits controlling phase... | — | Dataset available but no EMG (per sheet) |
| Operation regimes of spinal circuits controlling locomotion | — | Collected from a non-rat species |
