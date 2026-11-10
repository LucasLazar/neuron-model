# Calcium distribution families and the Bayesian likelihood: findings

Run 8-10 November 2026, as designed in [DESIGN.md](DESIGN.md):
- 98 cells: 14 rates (0.5-64 Hz) x 7 buffers (B_tot 0-300 µM);
- 12 seeds x 1000 s per cell;
- per-synapse reading with a silent soma.

Script: [l1_family_sweep.py](l1_family_sweep.py), with the report in
[l1_report.py](l1_report.py). Full tables: [results/l1_output.txt](results/l1_output.txt).
Per-cell data: `results/cells/`.

All three pre-set checks passed:
- The Tweedie density integrates to 1 within 2e-9 and matches Monte Carlo (KS 0.002).
- The vectorized replay matches `ExactGammaBelief` to 4e-16.
- The run reproduces physio p1's M skew bit for bit.

## Summary

1. **M is a compound Poisson count, not a gamma variable.** The excess y = M - c0 is
   roughly (number of input events in the window) x (calcium area per event) / W.
   - Its CV is 1/√(rW) to within 1-4% up to 16 Hz at B_tot ≤ 44.
   - Its skew/CV, which is the Tweedie power p, has median 1.07 at B_tot 0. Poisson is 1
     and gamma is 2.
   - Its kurtosis matches the Tweedie prediction p(2p-1).
2. **Buffer moves the shape from Poisson toward gamma.** The median p at W = 2 s, for
   rates ≤ 16 Hz:

   | B_tot (µM) | 0-15 | 44 | 100 | 200 | 300 |
   |---|---|---|---|---|---|
   | median p | 1.07 | 1.14 | 1.25 | 1.44 | 1.59 |

   A slow buffer spreads each event over several windows, so per-window contributions
   vary more.
3. **At low rates M is a comb with an atom at rest.** There is one peak per event count.
   A window with no event sits at rest, and this happens with probability about e^(-rW).
   At W = 2 s and B_tot 0, that is 35% of windows at 0.5 Hz, 11% at 1 Hz and 4% at
   1.5 Hz.
   - **A 2.5% lower tail exists only if rW > ln 40 ≈ 3.7 events per window.** Below
     that, no window can fall below "no event" often enough to be a 2.5% outlier.
   - With the current 2 s window, input below about 1.8 Hz therefore cannot produce a
     calibrated depression flag. No likelihood family can change this; only the window
     can.
4. **With known parameters, the current likelihood (gamma on M) is adequate wherever
   the window holds about 4 or more events.** At W = 2 s, the share of cells where both
   tails are calibrated:

   | Family | Calibrated cells |
   |---|---|
   | Tweedie | 86% |
   | generalized gamma | 85% |
   | **gamma on M (current)** | **83%** |
   | gamma on y | 62% |
   | normal | 36% |
   | inverse Gaussian | 32% |
   | lognormal | 30% |

   Its failures are the low-rate cells of point 3. Lognormal and inverse Gaussian are far
   too skewed, and normal is too symmetric until W = 8 s.
5. **The window matters more than the family.** The share of cells calibrated for gamma
   on M:

   | W | 0.5 s | 1 s | 2 s | 4 s | 8 s |
   |---|---|---|---|---|---|
   | calibrated cells | 61% | 70% | 83% | 92% | 97% |

   Across windows, calibration starts at about rW ≈ 4.
6. **The layer as implemented does not hit its nominal 2.5%, and it leans toward
   depression.** At W = 2 s and 2-16 Hz, the median flag rates are:
   - depress 2.0%;
   - potentiate 1.3%.

   Depress exceeds potentiate in 81% of cells. With stationary input, gbar should
   therefore drift down. Only 10% of cells are calibrated. The family is calibrated with
   known parameters, so the shortfall comes from learning with 60 s of memory, not from
   the family.
7. **Longer windows make the learned layer worse unless its memory grows with them.**
   With τ_slow fixed at 60 s and evidence_ms = W, the belief holds about τ_slow/W
   independent windows: 30 at 2 s, 7.5 at 8 s. Mean flag rates fall to 1.0/0.9% at 4 s
   and 0.2/0.2% at 8 s.
   - A model-free 60 s quantile tracker fails the other way: 3.0-3.2% at 2 s, and 4.5-4.8%
     at 8 s.
   - Window and memory are one design decision.
8. **Raw 1 ms calcium is a different animal.**
   - It is mostly at rest at low rates: 85% of ticks at 0.5 Hz and B_tot 0.
   - Its p runs from 1.3 to above 4, beyond gamma into lognormal and inverse Gaussian
     territory at high rates and buffers.
   - Only the 3-parameter generalized gamma is calibrated in most cells (80%), and it is
     the AIC winner in 58 of 98.

   If plasticity were to read short-timescale calcium, which is the buffer-as-integrator
   idea in STATUS.md, it would need a flexible family.

## Options for the Bayesian layer

These follow from the data, in order of impact. They are proposals for the project owner
to decide, not decisions.

**A. Size the window in events, not seconds (biggest single fix).** Every family and
both learners only work once the window holds about 4 or more events. A window of
k / r̂ with k ≈ 8 does this at every rate:
- r̂ is the rate implied by the belief's own mean, since mean excess ≈ 0.15 µM per Hz;
- at 0.5 Hz the window is 16 s, and at 64 Hz it is 125 ms.

A side effect is that rW is then constant, so the shape of M is nearly the same at
every rate, and one family with one CV fits all of them. The cost: at low rates the
layer reacts more slowly, because a 16 s window cannot see a change faster than that.

**B. The family: keep gamma on M, or move to Tweedie.**
- **Gamma on M** is already built and conjugate, and with enough events it does as well
  as anything here. This is the cheapest option.
- **Tweedie** (compound Poisson-gamma) is the mechanistic family. Its λ is the number of
  events per window, and it is the best calibrated at short windows (71% against 61% at
  0.5 s). It is not conjugate. Because it is an exponential dispersion model, though,
  the MLE of the mean is the sample mean, so a 2-D grid over (φ, p) would do.
  families.py has the density, CDF and quantiles.
- **Do not use** lognormal or inverse Gaussian: they are too skewed everywhere. Normal
  only works at W = 8 s.

**C. Fix the learned predictive (needed whatever A and B are).** The layer flags 1.3%
for potentiation against 2.0% for depression with the right family. Three candidates,
none tested yet:
- scale τ_slow with W, so that the belief always holds about 30 independent windows;
- score with plug-in parameters (posterior mean) instead of the full predictive;
- calibrate q empirically so that stationary input gives 2.5% on each side.

The depress/potentiate asymmetry is a stationary drift, so it should be tested closed
loop. The vv finding that gbar settles at about 2.6 nS rather than staying put may be
this effect.

**D. Model-free quantiles** need much more memory than 60 s. With 30 independent windows
they over-flag (3.0-3.2%). They are not a substitute for a family.

## What this does not cover

- **The branch reading.** There a compartment pools 1000 synapses, so rW is in the
  thousands and M is effectively Gaussian. The comb and the lower-tail limit are
  per-synapse problems. In the branch reading what matters instead is bAP-driven
  supralinearity and input correlation (branch/FINDINGS.md), which this study did not
  include.
- **Detection power for real rate changes.** This study measured stationary flag rates
  only. A longer or rate-scaled window trades power and latency against calibration;
  that needs a change-detection run.
- **Closed-loop plasticity.**

## Figures

| File | Shows |
|---|---|
| [plots/fig_l1_shapes.png](plots/fig_l1_shapes.png) | the distributions themselves: raw, M at 0.5, 2 and 8 s, five buffers, rates 0.5-64 Hz. The comb is visible at low rate and low buffer. |
| [plots/fig_l1_family_ladder.png](plots/fig_l1_family_ladder.png) | skew/CV (the Tweedie p) against CV for every cell and window, with the family lines |
| [plots/fig_l1_mechanism.png](plots/fig_l1_mechanism.png) | windows at rest against e^(-rW); CV against the Poisson count; Tweedie λ against rW |
| [plots/fig_l1_best_family.png](plots/fig_l1_best_family.png) | AIC winner per cell; faded cells have no clear winner |
| [plots/fig_l1_calibration_M2.png](plots/fig_l1_calibration_M2.png), [plots/fig_l1_calibration_M8.png](plots/fig_l1_calibration_M8.png) | tail rates per cell for each family, the layer and the tracker |
| [plots/fig_l1_calibration_summary.png](plots/fig_l1_calibration_summary.png) | the share of cells calibrated, family x window |
| [plots/fig_l1_qq.png](plots/fig_l1_qq.png) | QQ plots in four cells, with the 2.5% and 97.5% points marked |

## Notes on method

- **AIC.** The AIC winner is often not decisive at ≥ 2 Hz: Tweedie, generalized gamma
  and gamma on M trade places within ΔAIC_eff ≈ 10-60. Tail calibration is the measure
  that matters for the layer. At ≤ 1.5 Hz with low buffer, "normal" sometimes wins AIC.
  That is an artefact of censoring the at-rest windows; no continuous family describes
  a comb.
- **Fits.** Fits used 3000 ticks drawn at random from all seeds, left-censored at 1 nM.
  Tail rates are evaluated on every tick of every seed. These are in-sample for the
  families: 2-3 parameters on about 119k ticks, so the bias is negligible. Only the
  tracker is out of sample.
- **The layer replay.** The layer's flag rates use PlasticityConfig defaults with
  evidence_ms = W, scored after a 60 s warm-up, on input that starts at t = 8 s.
