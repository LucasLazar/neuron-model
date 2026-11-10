# Physiological-rate study: design

> Copied unchanged from the research repository, where it was written before the runs. "Production" means the model before this study (J_max 0.5, frozen u^2 step); exp1-exp9 and vv/ are earlier studies in that repository, summarised in [STATUS.md](../STATUS.md). In this package the kernel is neuron/fast.py and its tests are in tests/.

Written 2026-11-03, before any run in this study. One compartment throughout, as in
exp5, exp6 and vv. Production code and earlier results are not changed. Time in ms
unless stated.

## Why

Every earlier experiment drove the compartment with one Poisson stream at 30-500 Hz,
and each event is one saturating glutamate kick at one AMPA/NMDA site with one gbar
(MODEL_ODES.md section 2). Read as one presynaptic neuron, those rates are far above
normal: cortical neurons mostly fire at 0.1-10 Hz, reach 20-50 Hz when strongly
driven, and exceed 100 Hz only inside bursts.

This study adopts the **per-synapse reading**: one input stream is the spike train of
one presynaptic neuron onto the modelled synapse. It reruns the rate x buffer sweep
(exp5 and exp6, including exp6's per-buffer panel A) and the time-step convergence
study (vv blocks A and B1) at physiological rates.

## The retune

At physiological rates the production model's calcium barely leaves rest. One isolated
event raises free calcium by 0.018 uM with no buffer, 0.006 uM at Btot = 15 and
0.003 uM at Btot = 44 uM. A 4 Hz train would therefore move mean calcium by about 1%.

**k_b is kept at 0.315 /ms.** One release takes receptor occupancy s from 0 to about
0.15. Single releases do not saturate spine NMDA receptors: a second release 10 ms
later adds about 80% more activation (Mainen, Malinow & Svoboda 1999), and fewer
than one NMDA receptor is open on average at the peak of a single-synapse response
(Nimchinsky et al. 2004). Its original constraint, s*_bg(40 Hz) <= 0.20, guarded
against background saturation in the high-rate domain and is met trivially here.

**J_max is the only parameter that changes.** It is set so that one isolated event,
from rest, at resting buffer capacity kappa = 20 (Btot = 44 uM, since
kappa = Btot Kd / (Kd + c0)^2), raises peak free calcium by **0.5 uM**, in the
production scheme at dt = 0.1 ms. The anchors are approximate and quoted from memory,
so they should be checked against the papers:

- Endogenous spine buffer capacity of about 20 and an unbuffered decay of about 12 ms
  (Sabatini, Oertner & Svoboda 2002). The model's tau_c = 12 ms already matches the
  decay.
- Single-synapse NMDA calcium transients in spines of order 0.5-1 uM.

The total calcium entering per event is reported as a cross-check: it should be a few
hundred to a few thousand ions in a 0.1 fL spine head.

**Unchanged and therefore inherited:**
- Weight 12 x gbar 1 nS, so one event opens 12 nS of AMPA in a 150 pF compartment.
- tau_N = 40 ms, 2 mM Mg, tau_c = 12 ms, Kd = 2 uM.
- No NMDA current in the voltage equations. J_max therefore changes calcium only; V, u
  and s are identical to production at any given input.

**Soma.** A single synapse at these rates cannot drive the soma to threshold (exp6:
silent below 150 Hz). This study is therefore bAP-free, like exp6's subthreshold rows.
Under the per-synapse reading, somatic firing comes mostly from the neuron's other
inputs. Adding it is the main open item and is out of scope here.

## Questions

1. **Retune (p0):** what J_max meets the target? What do single events, pairs and
   short trains then look like at each buffer, and how much does the buffer saturate?
2. **Rate x buffer (p1):** where does free calcium sit, how wide is it, and what shape
   does it have at each rate and buffer? Does buffer still narrow the distribution
   without moving its mean (exp5), and does drive still move it right (exp6)?
   **How well separated are adjacent rates, and how does separation change with
   buffer?** This is asked for raw 1 ms calcium and for the 2 s rolling mean M that
   the Bayesian layer reads.
3. **Convergence (p2):** in this domain, does the discrete model converge to the ODE
   sheet as dt shrinks, at what order, and are the quantities in (2) converged at the
   production dt = 0.1 ms?

## Grid

| | Values |
|---|---|
| Rates | 1, 2, 4, 8, 16, 32, 64 Hz. A doubling grid, so every adjacent pair is the same ratio and separation is comparable along the axis. 64 Hz is inside the burst range and is reported as an upper reference. |
| Buffers | Btot 0, 15, 44, 100, 300 uM (kappa 0, 6.8, 20, 45, 136) |
| Tunings | retuned J_max, and production J_max = 0.5 as a control |
| Input | Homogeneous Poisson in continuous time (vv's Events), as in exp6. Bursty trains are deferred. |

## Blocks

| Block | Purpose | Conditions | Simulated | Seeds |
|---|---|---|---|---|
| p0 | Calibrate J_max; single events, pairs, trains | 5 buffers; pair intervals 2-500 ms | seconds | deterministic |
| p1 | Stationary distributions and separation | 7 rates x 5 buffers x 2 tunings | 400 s | 8 |
| p2-A | Order against vv's independent Radau reference, run at the retuned J_max | 4 deterministic cases; dt 0.2 to 0.00625, production and exact u^2 | 2-4 s | 1 train each |
| p2-B | Stationary quantities vs dt | 4, 16, 64 Hz plus 1 Hz; buffers 0, 15, 44, 300; dt 0.2 to 0.0125, plus exact u^2 | 200 s | 4 |

The same seed gives the same input spike train for every buffer, tuning and dt, so
those comparisons are paired.

## Measures and pre-set criteria

**p1, per cell.** Mean, sd, CV of the excess over rest, skew, 1-99.9% quantiles,
fraction of time within 1 nM of rest, integrated autocorrelation time, mean and 99th
percentile of buffer occupancy b/Btot (sampled every 100 ms), and the mean, sd, CV and
skew of the 2 s rolling mean M.

**Separation.** For adjacent rates r and 2r: AUC = P(X at 2r > X at r), on 1 ms
samples and on M. A value of 0.5 means indistinguishable and 1 means fully separated.
AUC depends only on ranks, so log versus linear axes do not matter. The point estimate
pools all seeds. The interval is a t interval across seeds, each seed compared with the
same seed at the other rate.

**Burn-in.** 15 tau_eff at resting kappa (co.burn_ms). Saturation only shortens the
relaxation, so this is conservative. Each run is checked for drift using the
second-half mean minus the first-half mean, divided by the sd.

**Convergence, p2-B.** The reference is the Richardson extrapolation from dt = 0.025
and 0.0125, assuming first order (this is checked in p2-A). The production step counts
as converged for a quantity if its paired difference from the reference is within 5%,
or if the 95% CI includes zero and is narrower than the 5% band. For AUC the tolerance
is 0.01 absolute. This is the same rule as vv/DESIGN.md.

## Amendment, 2026-11-03, after the first full run

The first run used the historical frozen u^2 step. p2 found every absolute calcium
quantity 8-14% high at dt = 0.1, which fails the 5% criterion above. Almost all of that
is the step's 10.3% over-integration of binding. The user adopted the exact u^2 step as
the production scheme: NMDANeuron(u_quadrature="exact"), MODEL_ODES.md section 9. The
constructor default stays "frozen" so earlier runs reproduce.

p0, p1 and p2 were rerun under the exact step:
- J_max was refitted to the same target.
- p2 compares the frozen step at dt = 0.1 against the new scheme.
- The J_max = 0.5 control is now called "untuned".

Targets, grids, seeds and criteria are unchanged.

## Not covered

- bAPs and somatic firing driven by other inputs.
- Bursty or correlated input.
- The Bayesian layer and AMPA feedback (exp7-exp9 and vv D/F) in this domain. p1's
  separation of M decides whether those are worth rerunning.
- Validation against biological calcium data beyond the single calibration point.
