# Branch study (option A): findings, step 1

3 November 2026. Design: [DESIGN.md](DESIGN.md).

Setup:
- One neuron: 3 branches x 1000 synapses, w_syn = 1 nS, Btot 44 uM.
- Correlated excitatory background plus somatic inhibition.
- Plasticity off (gbar fixed at 1 nS).
- Exact u^2 step, per-spine J_max = 78.3 uM/ms from physio/.

## Summary

1. **The pooled branch is a faithful stand-in for many synapses.** The mean-field gate
   matches 200 explicitly simulated nonlinear synapses to within 0.7% on mean
   occupancy, with under 1% rms error, from 1 to 20 Hz per synapse (b0). Python and
   kernel agree to 1e-9 ([tests/test_kernel.py](../tests/test_kernel.py)).
2. **Realistic firing needs correlated input.** With independent Poisson input the
   soma is either silent or fires like a clock: somatic voltage sd is about 1 mV, so
   it is mean-driven. A shared rate fluctuation (sd 50%, tau 20 ms) gives irregular
   low-rate firing. Baseline: 2.5 Hz per synapse, 1000 Hz somatic inhibition, soma
   1.8 Hz, ISI CV 1.2 (b1).
3. **Once the soma fires, the AND-gate switches on and the rate-calcium relation turns
   strongly supralinear.** From 1 to 4 Hz per synapse on every branch, branch calcium
   rises 27x (0.27 to 7.3 uM), against 4x for a lone spine. At 4 Hz, 57% of branch
   calcium enters only because of bAPs (b2, S1).
4. **Even without bAPs, a branch is supralinear.** The background depolarises the branch,
   from -56 mV at 1 Hz to -40 mV at 4 Hz, which partly lifts the Mg block. Calcium then
   rises 12x over the same range, and at 1 Hz is already 1.8x a lone spine's.
5. **Branch calcium is no longer purely local.** In S2, one branch's input goes from
   2.5 to 8 Hz while the others stay at 2.5 Hz.
   - The bystanders' calcium doubles, 1.63 to 3.41 uM.
   - All of that rise comes through bAPs: the bAP-off twin stays flat at 1.37 to
     1.42 uM.
   - The bAP-gated term is a **Hebbian product: the branch's own NMDA occupancy x the
     neuron's output**.
   - The target still dominates, rising 10x (1.64 to 17.1 uM).
6. **Each branch still reads its own input precisely.** On the 2 s mean M, the target
   separates even a 25% step (2 to 2.5 Hz): AUC 0.96, or 0.99 without bAPs.
   - Through bAPs, the bystander also carries information about *another* branch's
     input: AUC 0.56-0.94.
   - Without bAPs it carries none (0.51-0.57).
7. **Pooling solves physio's precision problem.** CV(M) is 0.10-0.20 at 1-8 Hz per
   synapse, against 0.69 for one spine at 1 Hz.
   - What remains comes from the shared input fluctuation and from bAPs.
   - Both are common to all branches, so branch calcium traces are strongly correlated
     with each other (fig_b2_traces).

## b0: pooled gate vs explicit synapses

[b0_pooled_gate_check.py](b0_pooled_gate_check.py) · [results/b0_output.txt](results/b0_output.txt)

| rate per synapse | linear gate: mean ratio / rms | mean-field gate (production): mean ratio / rms |
|---|---|---|
| 1 Hz | 1.005 / 0.8% | 0.999 / 0.6% |
| 2.5 Hz | 1.014 / 1.5% | 0.999 / 0.6% |
| 5 Hz | 1.030 / 3.1% | 1.001 / 0.7% |
| 10 Hz | 1.060 / 6.1% | 1.002 / 0.6% |
| 20 Hz | 1.123 / 12.4% | 1.005 / 0.8% |

These rows are without OU modulation. With it the numbers are within 0.3 points of
these. The linear gate ignores a synapse's own earlier binding. The mean-field factor
(1 - s/K) restores it, because a Poisson synapse sees its own time-averaged occupancy
when it fires.

## b1: firing regime

[b1_firing_regime.py](b1_firing_regime.py) · [results/b1_output.txt](results/b1_output.txt) ·
[plots/fig_b1_regime.png](plots/fig_b1_regime.png)

| rate per synapse | independent, no inh. | independent, inh. 1000 Hz | correlated, no inh. | correlated, inh. 1000 Hz |
|---|---|---|---|---|
| 1.5 Hz | 0 | 0 | 3.1 Hz (CV 1.31) | 0 |
| 2.0 Hz | 0.06 | 0 | 18.7 (1.54) | 0.17 (0.93) |
| 2.5 Hz | 22.2 (0.64) | 0 | 39.6 (1.48) | **1.8 (1.20)** |
| 3.0 Hz | 64.1 (0.26) | 0.01 | 60.2 (1.40) | 6.3 (1.37) |
| 4.0 Hz | 111 (0.09) | 5.1 (0.94) | 96.0 (1.15) | 24.1 (1.54) |

The baseline neuron is steep: a 20% rise in input roughly triples the output.

## b2: the relation with a firing soma

[b2_relation.py](b2_relation.py) · [results/b2_output.txt](results/b2_output.txt) ·
[plots/fig_b2_relation.png](plots/fig_b2_relation.png) ·
[plots/fig_b2_traces.png](plots/fig_b2_traces.png)

6 seeds x 200 s per level. Every run is paired with a bAP-off twin that has the same
input and identical spikes.

**S1, every branch at the rate:**

| rate/syn | soma | branch excess, bAPs on | off | from bAPs | CV(M) |
|---|---|---|---|---|---|
| 1.0 Hz | 0 Hz | 0.27 uM | 0.27 | 0% | 0.10 |
| 2.0 | 0.16 | 0.94 | 0.92 | 2% | 0.13 |
| 2.5 | 1.8 | 1.66 | 1.38 | 17% | 0.19 |
| 3.0 | 6.5 | 2.94 | 1.92 | 35% | 0.20 |
| 4.0 | 25.5 | 7.32 | 3.17 | 57% | 0.17 |

**S2, branch 1 at the rate, branches 2 and 3 at 2.5 Hz:**

| branch-1 rate | soma | target: on / off | bystander: on / off | bystander from bAPs |
|---|---|---|---|---|
| 0.5 Hz | 0.01 Hz | 0.12 / 0.12 | 1.30 / 1.30 | 0% |
| 2.5 | 1.7 | 1.64 / 1.38 | 1.63 / 1.37 | 16% |
| 4.0 | 5.6 | 4.41 / 3.15 | 2.13 / 1.40 | 34% |
| 8.0 | 19.2 | 17.1 / 10.3 | 3.41 / 1.42 | 58% |

## Caveats

- **Absolute calcium is high.** At baseline the branch average is 1.66 uM above rest.
  J_max was calibrated for a single event at rest (B = 0.023), but the background
  holds branches near -47 mV, where B is about 4x larger. A more hyperpolarised
  operating point would lower this. Possible routes are dendritic inhibition or less
  excitatory drive with more input correlation; this needs a decision.
- The steep input-output curve makes somatic rate sensitive to every parameter of the
  background.
- The buffer acts on the branch average, so single spines saturating briefly are not
  represented.
- No NMDA current in voltage, so no dendritic NMDA spikes, which are the real
  branch-level AND-gate.

## What this means for the network

- **Plasticity will be heterosynaptic.** Each branch's Bayesian layer will see calcium
  rise when *other* branches drive the neuron harder. Rising input on one branch would
  push every branch toward potentiation. Whether that is wanted is a modelling choice.
  It is the strongest argument for checking plasticity on one neuron before building
  the network.
- **Positive feedback is built in.** More input drives more firing, which raises
  calcium in all branches, which drives potentiation, which drives more firing.
  Inhibition, and possibly a homeostatic term, will be needed for stability.
- **The external background must be correlated.** Independent Poisson drive will not
  give cortical-like firing in the network either.
