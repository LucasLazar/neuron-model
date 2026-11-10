# Branch study (option A): design

> Copied unchanged from the research repository, where it was written before the runs. "Production" means the model before this study (J_max 0.5, frozen u^2 step); exp1-exp9 and vv/ are earlier studies in that repository, summarised in [STATUS.md](../STATUS.md). In this package the kernel is neuron/fast.py and its tests are in tests/.

Written 2026-11-03, before the b1 and b2 runs. b0 and the regime probes described
below were run while building the model and are reported as such.

## Why

physio/ treated one input stream as one synapse. That synapse's calcium was nearly
linear in rate, and the soma never fired, so the NMDA AND-gate never got its bAP. Even
8 such compartments do not fire the soma below about 96 Hz per synapse.

Option A, chosen by the user: **a compartment is a dendritic branch carrying hundreds of
synapses.** Its voltage, calcium and gbar are branch quantities: one learned parameter
per branch, which is where the project's write addressing lives.

## Model additions (production code, opt-in; defaults unchanged)

1. **Pooled branch gate**, `NMDANeuron(pooled_gate=True, n_syn=K, w_syn=w)`.
   - Input is a count of events per branch.
   - AMPA adds gbar x w_syn per event.
   - NMDA: clefts are independent, so v = sum of u^m is linear in events. The summed
     occupancy s obeys ds/dt = k_eff v (1 - s/K) - s/tau_N.
   - k_eff makes one isolated event reproduce the nonlinear single-synapse gate.
   - (1 - s/K) is the mean-field self-saturation: a Poisson synapse sees its own
     time-averaged occupancy when it fires.
   - Calcium influx is (J_max/K) s B(V + V_bAP). Free calcium is therefore the branch
     average of spine calcium, and physio's per-spine J_max = 78.3 uM/ms carries over.
   - Requires split_gate and u_quadrature="exact".
2. **Somatic inhibition**, `DendriticLIF(EI=-75, tau_I=6, gbar_I=1)` driven by
   `inh_in`. GABA_A-type, onto the soma only.
3. Both are ported to vv's kernel (`Sim(..., pooled_gate=True)`, `run(w_inh=...)`).
   test_branch.py holds the kernel to Python at 1e-9. vv's and physio's suites, and a
   stored physio result, confirm the existing paths are unchanged.

## What was learned while building (probes, not pre-registered)

- One 1 nS synapse gives a 0.78 mV branch EPSP and a 0.17 mV somatic EPSP. That is in
  the range of unitary cortical EPSPs, so **w_syn = 1 nS** (gbar 1 nS).
- **Independent Poisson input cannot produce realistic firing.** With thousands of
  small inputs the somatic voltage sd is about 1 mV. The neuron is either silent or
  fires regularly at 20-150 Hz (ISI CV 0.04-0.65). Inhibition just shifts that edge,
  and the edge is razor-thin: 0 to 300 Hz of inhibition takes the soma from 22 to
  0.9 Hz.
- **Correlated input does.** A shared Ornstein-Uhlenbeck modulation of the excitatory
  rate (relative sd 0.5, tau 20 ms) stands in for population fluctuations in the
  presynaptic pool. It gives irregular firing at low rates (ISI CV 1.3-1.5). For
  example, 2,500 events/s per branch with 1,000 Hz somatic inhibition at 2 nS gives
  about 2 Hz.

## Baseline neuron

| | value |
|---|---|
| branches | 3, each K = 1000 synapses, w_syn = 1 nS |
| background | 2.5 Hz per synapse, shared OU modulation sigma 0.5, tau 20 ms |
| inhibition | 1000 Hz pooled at the soma, 2 nS per event |
| buffer | Btot 44 uM (kappa 20 at rest) |
| J_max | 78.3 uM/ms per spine (physio), so 0.0783 per branch event |
| step | dt 0.1 ms, exact u^2 |

## Blocks

**b0 (done while building): pooled gate vs explicit synapses.** 200 explicit nonlinear
synapses against the pooled gate, at 1-20 Hz per synapse, with and without OU
modulation. Reported for the linear gate and the mean-field gate. The production gate
is mean-field.

**b1: firing regime.**
- Grid: per-synapse rate 1, 1.5, 2, 2.5, 3, 4 Hz on every branch x OU sigma 0 or 0.5 x
  inhibition 0 or 1000 Hz.
- 3 seeds x 100 s.
- Reports somatic rate, ISI CV and mean branch voltage, and documents the baseline
  choice.

**b2: how the rate-calcium relation changes once the soma fires.**
- Every run is paired with a bAP-off twin: same input and identical spikes, since NMDA
  carries no current, but A_f = A_s = 0.
- The difference between the twins is calcium entering only because of bAPs.
- **S1, global:** every branch at 1, 1.5, 2, 2.5, 3, 3.5, 4 Hz per synapse.
- **S2, one branch:** branch 1 at 0.5, 1, 2, 2.5, 4, 8 Hz; branches 2 and 3 held at
  2.5 Hz.
- 6 seeds x 200 s, burn-in 15 tau_eff, Btot 44.

**Measures (b2).**
- somatic rate and ISI CV;
- per-branch mean free-calcium excess, bAP on and off;
- bAP share = 1 - excess(off) / excess(on);
- CV of 1 ms calcium and of the 2 s rolling mean M;
- in S2, AUC between consecutive branch-1 rates on M, for branch 1 (the target) and
  branch 2 (a bystander). This asks whether a branch's calcium reports its own input or
  the neuron's output.

## Not covered

- Plasticity feedback (gbar fixed at 1 nS).
- Dendritic inhibition.
- NMDA current in the voltage equations, so no NMDA spikes or plateaus.
- Per-spine buffer saturation: the buffer acts on the branch average.
- Networks.
