# Calcium distribution families and the Bayesian likelihood: design

Written 2026-11-08, before any run in this study. One compartment, per-synapse reading,
as in physio/. The model and earlier results are not changed. Time in ms unless stated.

## Why

The Bayesian layer scores the 2 s rolling mean M of free calcium with a gamma likelihood
(MODEL_ODES.md section 4). That family was chosen in the old 30-500 Hz domain: exp3 found
calcium gamma-shaped by AIC and by the parameter-free test skew = 2 CV. physio p1 found
that at physiological rates M is not gamma-shaped. At 1-2 Hz it is multimodal, one peak
per number of events in the window, and its skew is 0.46-0.97 of the gamma value. STATUS.md
lists "a different likelihood family" and "a different window" as open decision 2.

This study maps the shape of calcium over the whole physiological rate x buffer grid. It
does so for raw calcium and for M at five window lengths, then tests candidate likelihoods
against it. Two kinds of test are used:
- the earlier ones: AIC, skew against CV, and QQ;
- tests of what the layer actually does with a family: how often it flags a
  stationary input as a departure.

## Grid

| | Values |
|---|---|
| Rates | 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 Hz: roughly half-octave steps over the physiological range, with 64 Hz as the burst-range reference |
| Buffers | B_tot 0, 5, 15, 44, 100, 200, 300 µM (κ at rest 0, 2.3, 6.8, 20, 45, 91, 136) |
| Every combination | 98 cells |
| Input | Homogeneous Poisson, physio_core.stationary_run, tag 91 |
| Model | Retuned J_max 78.3 µM/ms, exact u² step, dt 0.1 ms, soma silent, plasticity off |
| Length | 12 seeds x 1000 s per cell, after physio's burn-in |

The same seed gives the same spike train at every buffer. At p1's rates and seeds, the
first 400 s are p1's runs exactly, which gives a reproduction check.

## Observables

Ticks every 100 ms (the layer's update interval), from t = 8 s, so every observable is
read at the same instants:
- **raw**: free calcium c at the tick;
- **M_W** for W = 0.5, 1, 2, 4, 8 s: the mean of the last W of 1 ms samples. W = 2 s is
  the layer's current window.

Shape is judged on the excess y = M - c0, as in exp3 and exp5, because rest c0 = 0.1 µM
is the support boundary. The layer itself reads M, so the current likelihood is also
tested on M.

## Measures

**A. Shape, per cell and observable.**
- Statistics per seed:
  - mean;
  - CV of the excess and CV of M;
  - skew and excess kurtosis;
  - the fraction of ticks with y below 1 nM ("censored", in effect at rest);
  - the integrated autocorrelation time τ_int.
- Seed mean with t interval (physio convention).
- Family lines in the skew-CV plane (exp5's parameter-free test, extended). For a given
  CV of the excess, each family forces a skew:

  | Family | Forced skew |
  |---|---|
  | normal | 0 |
  | Poisson count | CV |
  | gamma | 2 CV |
  | inverse Gaussian | 3 CV |
  | lognormal | 3 CV + CV³ |

  The Tweedie family, var = φ μ^p, forces skew = p CV. So **p̂ = skew/CV** places each
  cell on one continuous axis:
  - 1 is Poisson;
  - between 1 and 2 is compound Poisson-gamma;
  - 2 is gamma.

  As a second parameter-free check, Tweedie predicts excess kurtosis / CV² = p(2p - 1).
- Poisson-count reference: if each event adds a fixed calcium area, CV(y) = 1/√(rate·W).

**B. Families with known parameters.** Each family is fitted by maximum likelihood to
5000 ticks drawn at random from all seeds. Excess below 1 nM is left-censored, so it
contributes P(y < 1 nM).

| Family | Variable | Parameters | Why it is a candidate |
|---|---|---|---|
| normal | y | 2 | conjugate; the large-count limit |
| **gamma_M** | **M** | 2 | **the current likelihood** |
| gamma | y | 2 | the current family with the known rest subtracted; still conjugate |
| lognormal | y | 2 | conjugate on log y |
| inverse Gaussian | y | 2 | positive, skew 3 CV |
| generalized gamma | y | 3 | flexible benchmark; nests gamma and Weibull, and lognormal in the limit |
| Tweedie, 1 < p < 2 | y | 3 | compound Poisson-gamma, the mechanism: a Poisson number of events per window, each adding a variable amount |

Reported per family:
- **ΔAIC per sample** against the cell's best family, with n_eff = ticks/τ_int.
  ΔAIC_eff = ΔAIC/n x n_eff approximates the evidence in independent windows.
- **Tail calibration**: the fraction of all ticks below the fitted 2.5% quantile and
  above the fitted 97.5% quantile. These are the layer's flagging thresholds, q = 0.025.
  The rates are computed per seed and given as a seed mean with t interval. They test the
  family's shape with its parameters known.
- **The Tweedie parameters**: p, λ (expected jumps per window, against rate·W) and the
  jump shape α. A large α means near-identical jumps, which is the comb.

**C. Learned with the layer's memory.** At stationary input:
- **The current layer**: ExactGammaBelief on M with its defaults (τ_slow 60 s, update
  100 ms, q = 0.025) and evidence_ms = W. It is replayed exactly by belief_replay.py,
  which is checked against the class. Report the fraction of ticks with U < 0.025
  (depress) and U > 0.975 (potentiate), after the 60 s warm-up.
- **A model-free reference with the same memory**: the empirical 2.5% and 97.5%
  quantiles of the previous 60 s of ticks, scored on the next tick.

The ideal is 2.5% on each side. Asymmetry between the two sides is a stationary drift in
gbar.

## Pre-set criteria

- A family is **calibrated** in a cell if both tail rates lie in [1.67%, 3.75%], within
  a factor 1.5 of 2.5%.
- A family is **indistinguishable from the best** if ΔAIC_eff < 10.
- The shape is **gamma-like** if the CI of p̂ includes 2, **Poisson-like** if it
  includes 1, and **normal-like** if the CI of skew includes 0.
- Checks that must pass before results count:
  - the Tweedie density integrates to 1 and matches Monte Carlo (families.self_check);
  - the replay matches ExactGammaBelief to 1e-12;
  - the p1 reproduction cell (4 Hz, B_tot 44, seed 0, first 400 s) matches p1's M skew
    to 1e-9.

## Questions

1. What shape does calcium take, raw and at each window, across rate x buffer? Where is
   it a comb, where compound Poisson, where gamma, and where near-normal?
2. Which family fits best (AIC), and which reproduces the 2.5% tails (calibration), in
   each cell?
3. At stationary input, what are the current layer's flag rates, and how asymmetric are
   they?
4. How does the window change 1-3? Which window and family combinations are calibrated
   across the physiological range?

## Not covered

- bAPs and somatic firing; the branch reading (branch/), where calcium is pooled over
  1000 synapses and bAP-gated.
- Bursty or correlated input.
- Detection power for real rate changes. This study measures behaviour at stationarity
  only.
- Closed-loop plasticity.
