# Where the research stands

10 November 2026. This page describes the current model and what is known about it.
The equations are in [MODEL_ODES.md](MODEL_ODES.md). The history is at the end.

## The model

This is a first computational variant of the neuron model in Gabriele Scheler, *Sketch
of a novel approach to a neural model*, arXiv:2209.06865 (F1000Research 14:218). That
paper proposes neuron-centric rather than synapse-centric plasticity. A neuron carries
three layers of parameters, and "not every transmission event leaves a trace". All three
layers are modelled here.

| Layer (Scheler) | In this model | Parameters |
|---|---|---|
| **External**: ion channels, which set the activation function | AMPA and NMDA on a dendritic LIF (dendritic compartments plus a soma, with bAPs) | the AMPA quantal conductance **gbar**, which the core writes |
| **Internal**: second messengers, which filter signals and adapt the external layer | calcium: a split NMDA gate (glutamate x Mg unblock with bAPs) feeding buffered free calcium | buffer capacity **B_tot**, extrusion **τ_c**, affinity **K_d**: the knobs under study |
| **Core**: long-term storage that decides what leaves a trace | **the Bayesian plasticity layer**, one per compartment | the learned belief about the compartment's normal calcium, and the tail rule that writes gbar |

How the core works:
- It reads M, the 2 s rolling mean of free calcium.
- It holds a belief over M's mean and CV. This is the stored expectation of "normal",
  with a memory of τ_slow = 60 s.
- Every 100 ms it asks whether M falls in the 2.5% tails of its own prediction.
- Only those departures move gbar: above the 97.5% point toward 4x its initial value,
  below the 2.5% point toward 0.25x.

Most events therefore leave no trace, which is Scheler's slogan in mechanistic form.

**The research question:** what do the internal parameters, the buffer above all, do to
what the core learns, and so to what the neuron can learn?

### Current defaults

These are the constructor defaults of `NMDANeuron` and the compiled `Sim` / `PlasticSim`.

| Part | Default |
|---|---|
| Numerics | dt 0.1 ms; exact u² quadrature; first order in dt; dt 0.1 is within 2.1% of the dt → 0 limit |
| NMDA gate | k_b 0.315 /ms, m = 2, τ_N 40 ms, Mg 2 mM; bAP waveform A_f 75 / A_s 25 mV |
| Calcium | c0 0.1 µM, τ_c 12 ms, K_d 2 µM, **B_tot 44 µM** (κ ≈ 20 at rest), **J_max 78.3 µM/ms**: one isolated event peaks at 0.5 µM |
| Core | reads M (2 s window of 1 ms samples), updates every 100 ms; gamma likelihood on M with the mean integrated out exactly and a 121-point CV grid; τ_slow 60 s; T_evidence 2 s; tails q = 0.025; gbar bounded to [0.25, 4] x initial with τ_g 60 s; 60 s warm-up; mode `feedback` (`observe-only` and `off` also exist) |

There are two readings of an input stream:
- **Per-synapse** (physio/, likelihood/): one stream is one presynaptic neuron at
  0.5-64 Hz.
- **Branch** (branch/, `pooled_gate=True`): a compartment is a dendritic branch of about
  1000 pooled synapses. **This is the chosen direction for networks.**

Verification ([VERIFICATION.md](VERIFICATION.md)):
- 50 tests pass;
- the compiled kernel matches the Python reference to 1e-9;
- an independent Radau solver confirms first-order convergence.

## What is known about the current model

### Internal layer: calcium

Per-synapse reading ([physio/FINDINGS.md](physio/FINDINGS.md)):
- Mean calcium is linear in rate, at 0.15 µM per Hz, and independent of buffer to 0.19%.
- **On raw 1 ms calcium, the buffer is the lever for telling rates apart.** For
  4 → 8 Hz, AUC rises from 0.68 with no buffer to 0.93 at B_tot 300. Saturation erodes
  this at high rates.
- **On M, the 2 s window does the averaging, and buffer adds little.** It matters only
  where τ_eff approaches the window (B_tot 300, ≤ 4 Hz).

Shape ([likelihood/FINDINGS.md](likelihood/FINDINGS.md)):
- **M is a compound Poisson count:** (events in the window) x (calcium per event) / W.
  - CV(M - c0) = 1/√(rW).
  - skew/CV, which is the Tweedie power p, is about 1.1 at low buffer. Poisson is 1 and
    gamma is 2. Buffer pushes it toward gamma, to about 1.6 at B_tot 300.
  - At low rates M is a comb of peaks with an atom at rest.
- Raw 1 ms calcium is far more skewed: p runs from 1.3 to above 4.

Branch reading ([branch/FINDINGS.md](branch/FINDINGS.md)):
- With a firing soma, bAPs make branch calcium strongly supralinear: 27x from 1 to
  4 Hz per synapse, with 57% bAP-gated at 4 Hz.
- That calcium is partly non-local. A bystander branch's calcium doubles when another
  branch drives the neuron. **Plasticity will be heterosynaptic.**
- Each branch still reads its own input on M: AUC 0.96 for a 2 → 2.5 Hz step.

### Core layer: the Bayesian plasticity layer

From [likelihood/FINDINGS.md](likelihood/FINDINGS.md). These are stationary input,
open-loop measurements; gbar was not fed back.

- **The core can only flag a decrease if its window holds rW > 3.7 events.** Below
  that, at least 2.5% of windows contain no event at all, so there is no lower tail.
  With the 2 s window, input below about 1.8 Hz cannot depress.
- **The likelihood family is adequate once the window holds about 4 or more events.**
  With known parameters, gamma on M hits both 2.5% tails in 83% of rate x buffer cells
  at 2 s, and 97% at 8 s. The mechanistic Tweedie family does slightly better: 86% at
  2 s. Lognormal, inverse Gaussian and normal do much worse.
- **The core as implemented under-flags, and leans toward depression.** At 2 s and
  2-16 Hz it flags 2.0% for depression and 1.3% for potentiation, against 2.5% each.
  Under constant input gbar should therefore drift down. The cause is learning with
  60 s of memory, not the family. Longer windows make it worse unless τ_slow grows
  with W.
- **The core has not yet run closed loop in either current reading.** Every physio,
  branch and likelihood run had gbar fixed.

### External layer: membrane and firing

- **Per-synapse:** one synapse cannot fire the soma, so there are no bAPs in this
  reading.
- **Branch:** realistic irregular firing (1.8 Hz, ISI CV 1.2) needs **correlated**
  input. The baseline is:
  - 3 branches x 1000 synapses at 2.5 Hz;
  - a shared OU rate modulation (σ 0.5, τ 20 ms);
  - 1000 Hz somatic inhibition at 2 nS;
  - B_tot 44.

  The mean-field pooled gate matches 200 explicit synapses to 0.7%.

## Open decisions and problems

1. **How the core reads and scores calcium.** The likelihood study (FINDINGS, "Options")
   reduced this to three choices. None has been made; the model still uses a 2 s window,
   gamma on M and τ_slow 60 s.
   - **Window:** size it in events, not seconds. W ≈ 8 / r̂ makes rW constant, so the
     shape of M is the same at every rate. The cost is a slower reaction at low rates.
   - **Family:** keep gamma on M, or move to Tweedie (implemented in
     likelihood/families.py).
   - **Learned predictive:** fix the under-flagging and the asymmetry. Candidates: scale
     τ_slow with W, score with plug-in parameters, or calibrate q empirically. This needs
     a closed-loop test.
2. **Close the loop.** Run the core with gbar feedback in the branch reading. Expect:
   - heterosynaptic effects through bAPs;
   - positive feedback: more input → more spikes → more calcium everywhere →
     potentiation;
   - possibly the downward drift predicted by the asymmetric flags.
3. **Absolute branch calcium is high:** 1.7 µM above rest at baseline. J_max was
   calibrated at rest, but branches sit near -47 mV, where Mg unblock is about 4x its
   resting value. Options:
   - dendritic inhibition;
   - less drive with more correlation;
   - recalibrating J_max for the branch.
4. **The calibration anchors are unverified.** They were quoted from memory:
   - spine κ ≈ 20 and unbuffered decay ≈ 12 ms (Sabatini, Oertner & Svoboda 2002);
   - single-event NMDA calcium of 0.5-1 µM.

   Check them against the literature before relying on absolute numbers.
5. **Missing biology:**
   - NMDA carries no current into voltage, so there are no dendritic NMDA spikes;
   - calcium is a branch average, so single spines never saturate.
6. **The per-synapse reading has no somatic firing.** `NMDANeuron.step(force_post=True)`
   exists in Python but not in the kernel.

## Where the project owner wants to go next

The stated aim, 8 November:

1. **Map the "modes" a neuron can take** from combinations of internal parameters, and
   the learning the core can do in each. The two ends of the range are:
   - **low buffer:** fast but noisy plasticity;
   - **high buffer:** precise read-out of input rate.
2. **Add an optimisation gradient** that moves a neuron between modes according to what
   is useful at the time. In layer terms, the core would then also write the internal
   parameters, not only gbar.

A plan was proposed but **not yet agreed**. Its key design point: **the buffer only
creates distinct modes if nothing downstream does the averaging for it.** On the 2 s
mean M, buffer level barely matters. So for buffer to be the dial, the core probably
has to read calcium on a timescale shorter than τ_eff = τ_c(1 + κ), which is 12 ms to
1.6 s across B_tot 0-300. The buffer then *is* the integrator. If it does, raw calcium
is far more skewed than M and needs a flexible family (likelihood/FINDINGS.md, point 8).

Proposed sequence:
1. **Mode map.** On the branch model, sweep B_tot x K_d x τ_c, and measure for each
   combination:
   - speed: latency to detect a rate change;
   - precision: discriminability and false alarms;
   - selectivity: one strong event against many weak ones;
   - plasticity quality: gbar movement for real change against jitter without change.

   Plot the trade-off fronts.
2. **Environments.** Use stable input and input whose rate jumps at different hazard
   rates, and find the best mode for each. If no environment prefers low buffer,
   rethink before building a gradient.
3. **An online gradient on B_tot**, from the core's own surprise statistics:
   - surprises more frequent than expected: reduce buffer;
   - surprises rarer than expected: add buffer.

   Changing buffer itself triggers alarms while new buffer loads, so the gradient must
   be slow, or the core must be told when buffer changes.
4. **A network** of about 30 neurons (24 E + 6 I sketched) with neurons in different
   modes.

Still undecided:
- what the core reads: the 2 s M, or short-timescale calcium;
- whether the gradient's objective is self-supervised (the neuron's own input
  statistics) or task-defined.

## History

Studies before this package ran in the research repository, which is not included. Ask
the project owner for it if needed. **exp1-exp9 and vv used an older calibration**
(30-500 Hz, J_max 0.5, the frozen u² step), so their numbers do not carry over to the
current model.

| Study | Date | Result that still matters |
|---|---|---|
| exp1-exp4 | Jul-Aug | Built the calcium system and the NMDA AND gate. They began with a calcium latch, later replaced by the Bayesian core. |
| exp5, exp6 | Sep | Buffer narrows calcium without moving its mean; rate moves it right. |
| exp7, exp8 | Sep | Chose M, the 2 s rolling mean, as what the core reads. M detects both increases and decreases, and raises alarms while new buffer loads. exp8 added the exact binding step. |
| exp9 | Sep | Built the Bayesian core: gamma belief, predictive tails, bounded gbar rule. |
| vv | 1 Nov | Verified the core. It replaced an aliasing 2-D belief grid with the exact belief now in use. Closed loop under constant 100 Hz input, gbar settled near 2.6 nS from any start, and cycled when τ_g ≤ 10 s. |
| physio | 3 Nov | Moved to the per-synapse reading and the current calibration (J_max 78.3, exact u² step). |
| branch | 3 Nov | Introduced the pooled branch reading and the correlated-input baseline. |
| likelihood | 8-10 Nov | Mapped the shape of calcium and M, and how the core's likelihood and memory perform. |

Also only in the research repository: vv's long closed-loop runs, the old latch model,
and HANDOFF.md. HANDOFF.md is the July handoff, with the project's motivation, a ledger
of committed versus placeholder choices, and the literature review.
