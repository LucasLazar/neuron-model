# Physiological-rate study: findings

3 November 2026. Design, pre-set criteria and the amendment are in [DESIGN.md](DESIGN.md).

Setup:
- One compartment, homogeneous Poisson input. No somatic spikes occurred in any run.
- Exact u^2 step: `NMDANeuron(u_quadrature="exact")`, MODEL_ODES.md section 9.
- All runs use the compiled kernel (neuron/fast.py). [tests/test_kernel.py](../tests/test_kernel.py)
  holds it to the Python `NMDANeuron` at the retuned J_max under both steps (rtol 1e-9;
  4 to 32 Hz; every buffer; dt 0.2 to 0.025).
- exp5, exp6 and vv/ are earlier studies in the research repository at 30-500 Hz; see
  [STATUS.md](../STATUS.md). "Production" below means the model before this study.

## Summary

1. **The retune is one parameter.** J_max goes from 0.5 to **78.3 uM/ms** (x157), so that
   one isolated event at kappa = 20 raises free calcium by 0.5 uM. That is about 750
   Ca2+ ions into a 0.1 fL spine head. k_b, tau_N, Mg, weight and every voltage quantity
   keep their production values. J_max has no route back into voltage, so the retune
   changes calcium only.
2. **Without bAPs, calcium is close to a linear filter of the spike train.** A second
   event adds at most 11% more influx than an isolated one, and only at 2 ms spacing,
   because one event depolarises the compartment by just 8.7 mV. The only strong
   nonlinearity left is **buffer saturation**, which the retune switches on. At
   J_max 0.5 the buffer is never more than 11% occupied. Retuned, mean occupancy is
   0.82-0.84 at 64 Hz.
3. **Position: mean calcium is linear in rate and independent of buffer**, as
   conservation requires. It is 0.15 uM above rest per Hz: 0.25 uM at 1 Hz and
   11.7 uM at 64 Hz. It agrees across buffers to 0.19%.
4. **Raw 1 ms calcium: the buffer is what separates the peaks.** For a doubling from 4 to
   8 Hz, AUC rises from 0.68 with no buffer to 0.93 at Btot 300. Saturation costs part
   of this as rate rises. At 16 to 32 Hz the retuned model reaches 0.82 at Btot 44,
   against 0.96 at J_max 0.5. At Btot 300 the width turns back up above 16 Hz, and the
   32 to 64 Hz pair separates *less* well than 16 to 32 Hz.
5. **The 2 s rolling mean M (what the Bayesian layer reads): the window sets the
   separation, and buffer adds little.** AUC for a rate doubling is about 0.79-0.88 at
   1 to 2 Hz, 0.89-0.95 at 2 to 4, 0.95-0.98 at 4 to 8, and at least 0.99 from 8 Hz up.
   CV(M) follows shot noise, rate^-1/2. Buffer helps only at Btot 300 and only at 4 Hz
   or below.
6. **At low rates the distributions change shape.** With Btot 44 or less, the 1 ms
   distribution at 1-4 Hz is bimodal: rest plus transients. At 1 Hz, calcium is within
   1 nM of rest 72% of the time with no buffer and 20% at Btot 44. M at 1-2 Hz is
   multimodal, one peak per number of events in the window. In every cell, M's skew is
   0.46-0.97 times the 2 CV that a gamma distribution would have.
7. **Convergence: dt = 0.1 passes every pre-set criterion in this domain** under the exact
   u^2 step. Every absolute calcium quantity is within 2.1% of the dt -> 0 limit, and
   separation agrees to 0.0002. The historical frozen step failed every absolute
   quantity at 8-14%, which is why it was replaced.
8. **The NMDA AND-gate is intact but receives no second input.** With no somatic firing
   there are no bAPs. A bAP 2 ms after an event would multiply that event's calcium by
   about 4.

## p0: the retune

[p0_retune.py](p0_retune.py) · [results/p0_output.txt](results/p0_output.txt) ·
[plots/fig_p0_retune.png](plots/fig_p0_retune.png)

**Why J_max rather than k_b.** At J_max = 0.5 one event raises free calcium by only
0.017 uM (no buffer), 0.006 uM (Btot 15) and 0.003 uM (Btot 44). At 4 Hz the mean
excess would be 0.004 uM, so calcium would hardly leave rest.

The receptor side is already plausible. One release takes occupancy to s = 0.14, which
is unsaturated, consistent with Mainen et al. 1999 and Nimchinsky et al. 2004. k_b's
original constraint guarded against saturation at a 40 Hz background and is met
trivially here. So the shortfall is in calcium entry per open receptor, which is J_max.

| | value |
|---|---|
| J_max | 78.30 uM/ms (x157) |
| calcium entering per event | 12.5 uM total = 754 ions in 0.1 fL |
| V peak / Mg unblock for one event | -61.3 mV / 0.023 to 0.038 |
| same J_max at dt 0.0125 | peak 0.498 uM (-0.4% vs target) |
| same J_max with the frozen step at dt 0.1 | peak 0.556 uM (+11.2%) |

Under the exact step the calibration no longer depends on dt. The first fit, made under
the frozen step, gave J_max = 71.5. That value was compensating for the step's 10%
over-integration, and it landed 10% under target as dt -> 0. Calcium per event is the
same in both fits.

One event at each buffer (retuned):

| Btot | kappa | peak rise | time to peak | half-width | peak occupancy |
|---|---|---|---|---|---|
| 0 | 0 | 2.65 uM | 16 ms | 46 ms | -- |
| 15 | 6.8 | 1.12 uM | 42 ms | 111 ms | 0.38 |
| 44 | 20 | 0.50 uM | 73 ms | 232 ms | 0.23 |
| 100 | 45 | 0.24 uM | 102 ms | 457 ms | 0.15 |
| 300 | 136 | 0.086 uM | 141 ms | 1243 ms | 0.085 |

**Pairs (panel e).** Total influx for two events exceeds twice a single event's by at
most 11% at 2 ms, and by a few percent at most beyond 20 ms. The peak ratio goes above 2
because transients overlap and the buffer saturates, not because more calcium enters.

## p1: rate x buffer

[p1_rate_buffer_sweep.py](p1_rate_buffer_sweep.py) · [results/p1_output.txt](results/p1_output.txt) ·
[plots/fig_p1_panelA_raw.png](plots/fig_p1_panelA_raw.png) ·
[plots/fig_p1_panelA_rolling.png](plots/fig_p1_panelA_rolling.png) ·
[plots/fig_p1_summary.png](plots/fig_p1_summary.png)

7 rates (1-64 Hz, doubling) x 5 buffers x 2 tunings (retuned, and J_max = 0.5
"untuned"), 8 seeds x 400 s per cell. Each seed is the same spike train in every
buffer and tuning.

**Does exp5 still hold (buffer narrows without moving the mean)?** The mean holds to
0.19%. Narrowing holds at every rate, but the retuned buffer narrows less than the
untuned control as rate rises, because it fills (summary panels b and e). At Btot 300,
CV of the excess is 0.61, 0.36, 0.32, 0.36 and 0.41 at 1, 4, 16, 32 and 64 Hz. It falls,
then rises again once mean occupancy passes about 0.55.

**Does exp6 still hold (drive moves the distribution right and narrows it)?** It moves
right, linearly. It narrows as shot noise predicts only with no buffer. With a buffer,
narrowing stalls where saturation sets in.

**Peak separation (AUC for adjacent doublings, retuned):**

| Btot | 1 ms: 1-2 Hz | 4-8 | 16-32 | 32-64 | M: 1-2 Hz | 2-4 | 4-8 | 8-16 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.65 | 0.68 | 0.75 | 0.81 | 0.79 | 0.89 | 0.95 | 0.99 |
| 15 | 0.67 | 0.70 | 0.78 | 0.83 | 0.79 | 0.89 | 0.95 | 0.99 |
| 44 | 0.68 | 0.75 | 0.82 | 0.85 | 0.80 | 0.89 | 0.95 | 0.99 |
| 100 | 0.72 | 0.82 | 0.87 | 0.88 | 0.82 | 0.91 | 0.96 | 0.99 |
| 300 | 0.83 | 0.93 | 0.95 | 0.94 | 0.88 | 0.95 | 0.98 | 1.00 |

The seed-paired 95% CIs are at most +-0.02. The full tables, including the untuned
control, are in p1_output.txt.

**Reading it:**
- On raw calcium the buffer is the main lever, as in exp6's panel A. Saturation makes
  that lever weaker at exactly the rates where calcium is high.
- On M the 2 s window already does the averaging that the buffer does for raw calcium.
  Buffer then matters only where its own time constant approaches the window (Btot 300,
  tau_eff 1.6 s) and events are sparse. M's precision is set by how many events fall in
  the window: about 2 at 1 Hz and 8 at 4 Hz, against 200 at the old 100 Hz baseline.
  Shot noise scales as 1/sqrt(rate x window). To match the precision the 2 s window had
  at 100 Hz, a 4 Hz synapse would need about a 50 s window.
- AUC depends only on ranks, so it is blind to scale. On M, the untuned control
  separates as well as the retuned model, and a little better where the retuned buffer
  saturates. The retune is therefore not what makes rates distinguishable. What it sets
  is the absolute calcium level, and so how hard the buffer is loaded, plus anything
  downstream that reads absolute uM.

**Checks.**
- Mean excess agrees across buffers to 0.19%.
- With no buffer, calcium is linear in J_max, so the two tunings must give the same
  CV; the largest difference is 4e-12.
- Burn-in: the largest |second-half minus first-half| is 0.64 sd, in a heavily buffered
  untuned cell, where the sd is smallest. Per seed, the sign is shared across all
  buffers and both tunings, so it is the spike train's own count difference between
  halves, not relaxation.
- No somatic spikes in any of the 560 runs.

## p2: convergence

[p2_convergence.py](p2_convergence.py) · [results/p2_output.txt](results/p2_output.txt) ·
[plots/fig_p2_convergence.png](plots/fig_p2_convergence.png)

**A: against the independent Radau reference** ([tests/reference_solver.py](../tests/reference_solver.py)) (rtol 1e-10, J_max set to the retuned
value), dt 0.2 to 0.00625.

| case | exact step at dt 0.1: mean excess / peak | frozen step at dt 0.1 | observed order |
|---|---|---|---|
| 2 Hz, Btot 44 | +0.37% / +0.61% | +9.8% / +13.5% | 1.00-1.09 |
| 8 Hz, Btot 15 (peak 13 uM, buffer saturates) | +0.57% / +1.40% | +9.4% / +10.4% | 1.00-1.02 |
| 32 Hz, no buffer | +0.77% / +1.47% | +8.8% / +8.9% | 1.00-1.02 |
| 32 Hz, Btot 300 | +0.88% / +0.87% | +10.9% / +10.4% | 1.00-1.12 |

Both steps converge at first order in every case. The exact step's maximum error is
6-17x smaller.

**B: stationary quantities vs dt**, 4 rates x 4 buffers, 4 seeds x 200 s, paired
against Richardson(0.025, 0.0125).

| quantity at dt 0.1 | exact step, 16 cells | verdict (5% band) | frozen step |
|---|---|---|---|
| mean excess | +0.4 to +1.1% | pass in all 16 | +7.9 to +9.8% |
| sd, 1 ms | +0.4 to +2.1% | pass in all 16 | +7.5 to +14.1% |
| 99th percentile excess | +0.4 to +1.8% | pass in all 16 | +7.5 to +12.6% |
| sd of M | +0.4 to +1.6% | pass in all 16 | +6.6 to +10.7% |
| CV of M | 0.0 to +0.5% | pass in all 16 | -1.1 to +0.9% |
| AUC, 4x steps, 1 ms and M | within 0.0002 of dt 0.0125 | pass | within 0.004 |

**Reading it.**
- The frozen step's error was almost entirely one constant factor on calcium entry.
  (1 - e^-0.2)/0.2 = 0.906, so it over-integrates u^2 by 10.3%. That matches its 9.8%
  error at 1 Hz. The error shrinks toward 64 Hz as the occupancy term's (1 - s) factor
  saturates.
- That factor was not purely a scale: through buffer saturation it also changed the
  shape of raw calcium, up to 4.7 points beyond the mean's error at Btot 300 and 16 Hz.
- The exact step removes the factor. What remains is ordinary first-order splitting
  error, largest at 64 Hz (2.1% on the sd at Btot 300).
- dt = 0.1 is converged for everything measured here.

## The AND-gate without bAPs

The gate is `J = J_max · s · B(V + V_bAP)`. Glutamate opens s, and depolarisation
relieves the Mg block B. bAPs exist in the model but are triggered only by somatic
spikes. Here the soma's only input is this one synapse, which never brings it to
threshold at 1-64 Hz; the old domain fired only above about 150 Hz. The second input is
therefore only the synapse's own EPSP:

| | Mg unblock B |
|---|---|
| rest, -70 mV | 0.023 |
| peak of one EPSP, -61.3 mV | 0.038 |
| rest + bAP peak (+100 mV) | 0.92 |

Influx of one event with a bAP added to the gate at offset (post - pre). These values
are computed from the gate equation along one event's s and V traces, with the soma's
own reset ignored:

| offset | -40 ms | -10 | -2 | 0 | +2 | +5 | +10 | +20 | +40 | +80 | +150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| influx x | 1.14 | 1.63 | 2.70 | 3.86 | **4.13** | 3.84 | 3.38 | 2.74 | 2.01 | 1.37 | 1.06 |

The window is asymmetric and favours pre before post, because s outlasts the bAP. The
contrast is about 4x rather than B's 40x, because the fast bAP component lasts 3 ms
while s lasts tens of ms.

## What this means for the project

- **The model is now a per-synapse model with one changed value.** J_max x157 under the
  exact u^2 step. MODEL_ODES.md sections 8-10 record both. In this package both are the
  constructor defaults of `NMDANeuron`.
- **The Bayesian layer's 2 s window was sized for 100 Hz.** At physiological rates it
  holds a handful of events, M is multimodal at 1-2 Hz, and M's skew is well below the
  gamma likelihood's 2 CV. Before rerunning exp7-exp9 or vv D/F here, two things need
  revisiting:
  - the window, or a window that scales with the synapse's own rate;
  - the likelihood family.

  p1 suggests a doubling is reliably visible on M from about 4-8 Hz up. At 1.5x steps
  (exp9's 100 to 150 Hz) and low rates it will not be.
- **Somatic firing is the missing ingredient.** Under the per-synapse reading it comes
  mostly from the neuron's other inputs. `NMDANeuron.step(force_post=True)` already fires
  the bAP without touching voltages; the kernel (neuron/kernel.c) would need the same hook. What has to
  be decided is the somatic spike train: its rate, and how strongly it is correlated
  with this synapse.
- **Calibration anchors to verify.** These are quoted from memory:
  - spine kappa about 20 and an unbuffered decay of about 12 ms (Sabatini, Oertner &
    Svoboda 2002);
  - single-synapse NMDA calcium of order 0.5-1 uM.

  Change TARGET_PEAK_UM or REFERENCE_BUFFER in physio_core.py and rerun p0, p1 and p2 to
  regenerate everything; the study takes about 15 minutes on this machine. The studies
  read J_max from results/retune.json. Also copy the new value into J_MAX_SPINE in
  neuron/model.py: tests/test_model.py fails until the two agree.
