# Where the research stands

10 November 2026. Read [MODEL_ODES.md](MODEL_ODES.md) first for the model itself.

## The project

This is a first computational variant of the neuron model sketched in:

> Gabriele Scheler, *Sketch of a novel approach to a neural model*, arXiv:2209.06865
> (also F1000Research 14:218).

Scheler's position paper proposes neuron-centric rather than synapse-centric plasticity.
A neuron has three layers of parameters:

- **external:** ion channels, which set the activation function;
- **internal:** second messengers, which filter signals and adapt the external parameters;
- **core:** long-term, "write-once" storage.

Its slogan is "Not every transmission event leaves a trace."

In this variant:
- **External** is AMPA and NMDA on a dendritic LIF.
- **Internal** is calcium only, standing in for the whole second-messenger layer.
  Its handling parameters are the knobs of interest:
  - buffer capacity B_tot;
  - extrusion time τ_c;
  - affinity K_d.
- **Core** is not modelled yet.

Plasticity is a per-compartment Bayesian layer. Each compartment learns what its
calcium normally looks like, and moves its AMPA conductance when calcium leaves that
range.

**The research question now:** what do the internal parameters, the buffer above all,
do to what a neuron can learn?

## How the model got here

Experiments before this package were run in the research repository. They are not
included here; ask the project owner for the full folder if you need them.

| Study | Date | What it established |
|---|---|---|
| exp1-exp4 | Jul-Aug | Built the calcium system, its coupling to the membrane and the NMDA AND gate, around a calcium latch ("switch") that has since been replaced. |
| exp5 | Sep | Buffering narrows the calcium distribution without moving its mean. |
| exp6 | Sep | Input rate moves the distribution right. Shown per buffer level in "panel A by buffer": higher buffer, narrower and better separated peaks. physio/p1 repeats it at physiological rates. |
| exp7, exp8 | Sep | Chose what plasticity reads: the 2 s rolling mean of free calcium, M. It is the only candidate that detects both increases and decreases. M is blind to buffer level at steady state, but raises alarms during the loading transient when buffer is added. exp8 introduced the exact binding step now in neuron/calcium.py. |
| exp9 | Sep | Built the Bayesian layer: gamma belief over M, predictive tails, bounded gbar rule. |
| vv | 1 Nov | Verification of the layer (details below). |
| physio | 3 Nov | Per-synapse reading at 1-64 Hz. In this package: [physio/FINDINGS.md](physio/FINDINGS.md). |
| branch | 3 Nov | Pooled-branch reading, step 1. In this package: [branch/FINDINGS.md](branch/FINDINGS.md). |
| likelihood | 8-10 Nov | Shape of calcium and of M over rate x buffer x window, and which likelihood the Bayesian layer should use. In this package: [likelihood/FINDINGS.md](likelihood/FINDINGS.md). |

What vv found:
- The model is first order in dt.
- The original 2-D belief grid aliased, so it was replaced by the exact belief now in
  neuron/plasticity.py.
- Under constant input, gbar does not stay at its start. It settles near 2.6 nS at
  100 Hz from any start, and cycles when τ_g ≤ 10 s.

All of exp5-exp9 and vv were in the old domain:
- 30-500 Hz into one compartment;
- J_max 0.5;
- the frozen u² step.

## What is established now

**The per-synapse reading** ([physio/FINDINGS.md](physio/FINDINGS.md)): one input stream
is one presynaptic neuron at 1-64 Hz.

- The only calibration change was J_max: 0.5 → 78.3 µM/ms. With it, one event at
  B_tot 44 µM (κ ≈ 20) gives a 0.5 µM peak.
- The exact u² step is required. The frozen step was 8-14% off. With the exact step,
  dt = 0.1 ms is within 2.1% of the dt → 0 limit everywhere.
- Mean calcium is linear in rate (0.15 µM per Hz) and independent of buffer to 0.19%.
- **On raw 1 ms calcium, the buffer is the lever for telling rates apart.** For
  4 → 8 Hz, AUC rises from 0.68 with no buffer to 0.93 at B_tot 300. Buffer saturation
  erodes this at high rates.
- **On M, the 2 s window already does the averaging, and buffer adds little.** It
  matters only where τ_eff approaches the window (B_tot 300) at ≤ 4 Hz.
- At 1-2 Hz, M holds only a few events and is multimodal. The likelihood study below
  explains why.
- One synapse cannot fire the soma, so there are no bAPs in this reading.

**The likelihood study** ([likelihood/FINDINGS.md](likelihood/FINDINGS.md)): 14 rates
(0.5-64 Hz) x 7 buffers x 6 observables (raw calcium, and M over 0.5-8 s windows).

- **M is a compound Poisson count:** (events in the window) x (calcium per event) / W.
  - CV(M - c0) = 1/√(rW) to within a few percent.
  - skew/CV, which is the Tweedie power p, is about 1.1 at low buffer. Poisson is 1 and
    gamma is 2.
  - Buffer pushes p toward gamma: a median of 1.6 at B_tot 300.
- **At low rates M is a comb with an atom at rest,** and a 2.5% lower tail only exists
  if the window holds rW > 3.7 events. With W = 2 s, input below about 1.8 Hz cannot
  produce a calibrated depression flag. No family fixes that; only the window does.
- **With known parameters, the current gamma-on-M likelihood is adequate once rW ≳ 4.**
  At 2 s it is calibrated in 83% of cells, against 86% for Tweedie and 85% for
  generalized gamma. Lognormal, inverse Gaussian and normal do much worse. The window
  matters more than the family: gamma on M is calibrated in 61% of cells at 0.5 s and
  97% at 8 s.
- **The layer as implemented under-flags, and leans toward depression.** At 2 s and
  2-16 Hz, it flags 2.0% for depression and 1.3% for potentiation, against 2.5% each.
  Only 10% of cells are calibrated. The shortfall is the 60 s learning memory, not the
  family. Longer windows make it worse unless τ_slow grows with W.
- Raw 1 ms calcium is far more skewed: p runs from 1.3 to above 4. Only the 3-parameter
  generalized gamma fits it in most cells.

**The branch reading** ([branch/FINDINGS.md](branch/FINDINGS.md)): a compartment is a
branch of 1000 synapses with a pooled gate, `pooled_gate=True`. This is the direction
chosen for building networks.

- The mean-field pooled gate matches 200 explicit synapses to 0.7%.
- Realistic irregular firing (1.8 Hz, ISI CV 1.2) needs **correlated** input. The
  baseline uses a shared OU rate modulation (σ 0.5, τ 20 ms) plus 1000 Hz somatic
  inhibition.
- With a firing soma, bAPs make branch calcium strongly supralinear: 27x from 1 to
  4 Hz per synapse, with 57% of calcium bAP-gated at 4 Hz.
- That calcium is partly non-local. A bystander branch's calcium doubles when another
  branch drives the neuron, entirely through bAPs. **Plasticity in this model will be
  heterosynaptic.**
- Each branch still reads its own input well on M: AUC 0.96 for a 2 → 2.5 Hz step.

## Open problems and decisions

1. **Plasticity has not been run in either new reading.** Every physio and branch run
   had gbar fixed. Expect heterosynaptic effects and positive feedback: more input →
   more spikes → more calcium everywhere → potentiation.
2. **The Bayesian window, likelihood and memory were sized for 100 Hz.** The likelihood
   study narrowed this to three decisions for the project owner (likelihood/FINDINGS.md,
   "Options"):
   - **Window:** size it in events, not seconds. W ≈ 8 / r̂ makes rW constant, which
     keeps every family calibrated and makes the shape of M rate-independent. The cost
     is a slower reaction at low rates.
   - **Family:** keeping gamma on M is defensible once the window holds enough events.
     Tweedie is the mechanistic alternative; families.py implements it. Lognormal,
     inverse Gaussian and normal are ruled out.
   - **Learned predictive:** the layer under-flags and leans toward depression even with
     the right family. Candidate fixes are to scale τ_slow with W, to score with plug-in
     parameters, or to calibrate q empirically. This needs a closed-loop test, and may
     explain vv's gbar drift.

   None of this has been decided or implemented. The model still uses 2 s, gamma on M
   and τ_slow 60 s.
3. **Absolute branch calcium is high:** 1.7 µM above rest at baseline. J_max was
   calibrated at rest, but the background holds branches near -47 mV, where the Mg
   unblock factor B is about 4x its resting value. Options:
   - dendritic inhibition;
   - less drive with more correlation;
   - recalibrating J_max for the branch.
4. **The calibration anchors are unverified.** They were quoted from memory:
   - spine κ ≈ 20 and unbuffered decay ≈ 12 ms (Sabatini, Oertner & Svoboda 2002);
   - single-event NMDA calcium of 0.5-1 µM.

   Check them against the literature before relying on absolute numbers.
5. **Missing biology:**
   - NMDA carries no current into voltage, so there are no dendritic NMDA spikes;
   - calcium is a branch average, so single spines never saturate;
   - the core layer is not modelled.
6. **The per-synapse reading still lacks somatic firing.** To get bAPs it needs a
   modelled source of spikes. `NMDANeuron.step(force_post=True)` exists in Python but
   not in the kernel.

## Where the project owner wants to go next

The stated aim, 8 November:

1. **Map the "modes" a neuron can take** from combinations of internal parameters, and
   the learning each mode supports. The two ends of the range are:
   - **low buffer:** fast but noisy plasticity;
   - **high buffer:** precise read-out of input rate, as in exp6 panel A by buffer.
2. **Add an optimisation gradient** that moves a neuron between modes according to what
   is useful at the time.

A plan was proposed but **not yet agreed**. Its key design point, from the data above:
**the buffer only creates distinct modes if nothing downstream does the averaging for
it.** On the 2 s mean M, buffer level barely matters (physio p1). So for buffer to be
the dial, plasticity probably has to read calcium on a timescale shorter than
τ_eff = τ_c(1 + κ), which is 12 ms to 1.6 s across B_tot 0-300. The buffer then *is*
the integrator.

Proposed sequence:
1. **Mode map.** On the branch model, sweep B_tot × K_d × τ_c, and measure for each
   combination:
   - speed: latency to detect a rate change;
   - precision: discriminability and false alarms;
   - selectivity: one strong event against many weak ones;
   - plasticity quality: gbar movement for real change against jitter without change.

   Plot the trade-off fronts.
2. **Environments.** Use stable input and input whose rate jumps at different hazard
   rates, and find the best mode for each. This decides whether the premise holds: if
   no environment prefers low buffer, rethink before building a gradient.
3. **An online gradient on B_tot**, from the layer's own surprise statistics:
   - surprises more frequent than expected: reduce buffer;
   - surprises rarer than expected: add buffer.

   This is the Kalman/volatility argument for adapting a learning rate. Caution:
   changing buffer itself triggers alarms while new buffer loads (exp7c/8c). The
   gradient must therefore be slow, or the layer must be told when buffer changes.
4. **A network** of about 30 neurons (24 E + 6 I sketched) with neurons in different
   modes.

Undecided:
- what plasticity reads (the 2 s M, or short-timescale calcium). If it is
  short-timescale calcium, note that raw calcium is far more skewed than M and needs a
  flexible family (likelihood/FINDINGS.md, point 8);
- whether the gradient's objective is self-supervised (the neuron's own input
  statistics) or task-defined.

## Not in this package

These are in the research repository:
- exp1-exp9 and their outputs;
- vv/'s long closed-loop runs;
- the old latch model;
- HANDOFF.md, the July handoff with the project's motivation, its ledger of committed
  vs placeholder choices, and the literature review. Its model status is out of date,
  but the framing still holds.
