# Verification

What has been checked, what the result was, and how to rerun it. Run all commands from
this folder.

## 1. Unit and equivalence tests

`python -m unittest discover -s tests -v`: 50 tests, about 1 minute.

| File | What it establishes |
|---|---|
| [tests/test_lif.py](tests/test_lif.py) | **Voltage against analytic results:** <ul><li>steady states for equal and unequal drive</li><li>dendrite-soma attenuation with one or all compartments driven</li><li>exact EPSC decay</li><li>first-order Euler transient</li><li>refractory cap</li><li>somatic inhibition</li></ul> |
| [tests/test_model.py](tests/test_model.py) | **The gate and calcium:** <ul><li>a bAP alone gives exactly zero influx (the AND gate)</li><li>an event's calcium stays in its compartment</li><li>a bAP 2 ms after an event multiplies its influx about 4x</li><li>the exact u² step differs from the frozen one by φ = 0.906</li><li>an isolated event on a pooled branch matches the single-synapse gate</li></ul> **Calibration:** one event at B_tot 44 peaks at 0.5 µM, and `J_MAX_SPINE` equals physio's stored fit. **Calcium step:** <ul><li>against fine RK4 under a stiff square-pulse test: within 5% peak, 2% mean</li><li>binding conserves total calcium</li><li>buffer leaves mean calcium unchanged</li></ul> **Runs:** continuation, reset, and observe-only = off. |
| [tests/test_plasticity.py](tests/test_plasticity.py) | **The closed-form belief against direct numerical integration** of the gamma likelihood over the mean: posterior weights to 1e-7, predictive percentiles to 1e-6. This check shares no algebra with the closed form, and fails when that form is perturbed. Also: <ul><li>the score comes before learning</li><li>predictive interval and percentile agree</li><li>results do not depend on calcium units</li><li>95% band coverage on stationary data</li><li>causal rolling mean and tick times</li><li>drive mapping</li><li>conductance bounds and exact composition of updates</li></ul> |
| [tests/test_kernel.py](tests/test_kernel.py) | **The compiled kernel equals the Python model** (calcium rtol 1e-9, spike times 1e-8). Cases: <ul><li>per-synapse at B_tot 0-300 and 4-32 Hz</li><li>dt 0.2-0.025, and the frozen step</li><li>4 compartments with a firing soma</li><li>pooled branches with inhibition and a firing soma</li><li>the bAP-off twin, whose spikes are identical</li><li>zero inhibition = no inhibition</li></ul> **Closed loop:** `PlasticSim` equals `NMDANeuron(plasticity_mode='feedback')` in gbar, drive and percentile, per-synapse and pooled, including when runs are split at arbitrary points. |
| [tests/test_convergence.py](tests/test_convergence.py) | **A short convergence run against the independent Radau solver** ([tests/reference_solver.py](tests/reference_solver.py), written from MODEL_ODES.md). The scheme is first order; dt 0.1 is within 2% on mean calcium. The frozen step over-integrates by about 10%. |

## 2. Numerical studies

| Study | Result |
|---|---|
| [physio/p2_convergence.py](physio/p2_convergence.py) | **A: against the Radau reference**, 4 cases with dt from 0.2 to 0.00625: first order in every case. At dt 0.1 the exact step is within 0.4-1.5%; the frozen step is 9-14% off. **B: stationary statistics vs dt**, 4 rates x 4 buffers x 4 seeds x 200 s, against Richardson extrapolation. Every quantity in all 16 cells passes the pre-set criterion: whole 95% CI within ±5%. The worst case is 2.1%. Separation (AUC) is within 0.0002 of dt 0.0125. |
| [branch/b0_pooled_gate_check.py](branch/b0_pooled_gate_check.py) | **Mean-field pooled gate vs 200 explicitly simulated nonlinear synapses**, 1-20 Hz, with and without correlated input. Mean occupancy is within 0.5% and rms error under 1%. The simpler linear gate drifts to +12% at 20 Hz, which is why the mean-field term is used. |
| [physio/p0_retune.py](physio/p0_retune.py) | **Calibration.** The fitted J_max holds to 0.4% at dt 0.0125, so the calibration does not depend on dt under the exact step. |

## 3. Provenance

This package was cut from the research repository on 8 November 2026 and checked against
it:

- **Same model.** The handover `NMDANeuron` and kernel give bit-identical output to the
  research code on the same input: 117 array comparisons, 0 differences. The comparison
  covered:
  - per-synapse and pooled modes;
  - closed-loop plasticity;
  - all four buffer levels;
  - both u² steps.

  Only code paths outside the current model were removed:
  - the calcium latch, which was the old default mode;
  - the single-variable NMDA gate;
  - the 2-D grid belief.
- **Same results.** All six study scripts were rerun inside this folder: 12 minutes in
  total, with the times in README.md. Every text report and JSON result file is identical
  to the research repository's stored output. The only difference is the reference
  solver's wall-clock time in p2. The tables in physio/FINDINGS.md and branch/FINDINGS.md
  are therefore the results in this folder.

## 4. Not verified

- **Against biological data.** No data were fitted, and the calibration anchors are from
  memory (STATUS.md, open problem 4).
- **Closed-loop plasticity at physiological rates or in the branch model.** Long-run
  stability of the gbar rule was studied only in the earlier 30-500 Hz domain, in the
  research repository's vv/: gbar settles near 2.6 nS at 100 Hz and cycles when
  τ_g ≤ 10 s.
- **Anything with more than one neuron.**
