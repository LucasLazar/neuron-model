# Full neuron system

Dendritic LIF, split NMDA gate, buffered calcium and Bayesian AMPA plasticity. Current equation sheet · 8 November 2026.

This is a hybrid system: continuous biophysical ODEs, spike-triggered jumps, and discrete Bayesian and conductance updates. The Bayesian layer replaced an earlier calcium switch (latch); there is no switch state or switch equation in this model.

## 1 · Membrane and synaptic events

For compartments i = 1,...,N, V_i is dendritic voltage and V_s is somatic voltage. g_i is open AMPA conductance; gbar_i is the lasting quantal conductance that plasticity changes. Input weights w are dimensionless.

$$\frac{dg_i}{dt}=-\frac{g_i}{\tau_a}$$

$$C_d\frac{dV_i}{dt}=g_L(E_L-V_i)+g_i(E_a-V_i)+g_c(V_s-V_i)$$

$$C_s\frac{dV_s}{dt}=g_L(E_L-V_s)+\sum_{i=1}^{N}g_c(V_i-V_s)$$

At a presynaptic event of weight w at compartment i:

$$g_i(t^+)=g_i(t^-)+\bar g_i(t^-)w$$

When the nonrefractory soma reaches threshold, emit a spike, reset its voltage, and hold it for the refractory interval. Dendrites continue integrating during the hold.

$$V_s\geq V_{th}\quad\Rightarrow\quad V_s\leftarrow V_{reset}\quad\mathrm{for}\ t_{ref}$$

NMDA contributes calcium influx only in this implementation. Its current is not included in either voltage equation. AMPA feedback changes subsequent event increments; it does not rescale conductance already open.

Optional somatic inhibition (used in branch/): a GABA_A-type conductance on the soma adds g_I(E_I - V_s) to the somatic current, with E_I = -75 mV and tau_I = 6 ms. With no inhibitory input g_I stays 0.

$$\frac{dg_I}{dt}=-\frac{g_I}{\tau_I},\qquad g_I(t^+)=g_I(t^-)+\bar g_I w_I$$

## Coupling through the full system

Presynaptic input opens AMPA and supplies glutamate. Dendritic voltage and somatic spikes regulate NMDA calcium entry. Buffer binding and extrusion shape free calcium. Each compartment learns its own distribution of rolling calcium means. Predictive tail departures drive its AMPA quantal conductance, closing the loop through voltage and calcium.

Units: time ms, voltage mV, capacitance pF, conductance nS, current pA, calcium and buffer concentration µM. External magnesium is in mM.

<!-- pagebreak -->

# NMDA and buffered calcium

## 2 · Split ligand gate and voltage dependence

u_i is normalized cleft glutamate; s_i is receptor occupancy. They are distinct from the Bayesian percentile and plasticity drive introduced below. A positive presynaptic input triggers one saturating glutamate kick per integration step, independent of its weight.

$$\frac{du_i}{dt}=-\frac{u_i}{\tau_u},\qquad u_i(t^+)=u_i(t^-)+a_u[1-u_i(t^-)]$$

$$\frac{ds_i}{dt}=k_bu_i^m(1-s_i)-\frac{s_i}{\tau_N},\qquad m=2$$

The magnesium unblock factor sees local voltage plus a global back-propagating action-potential waveform. The waveform affects this gate only.

$$B(V)=\frac{1}{1+([\mathrm{Mg}^{2+}]/3.57)e^{-0.062V}}$$

$$V_{bAP}=y_f+y_s,\qquad \frac{dy_f}{dt}=-\frac{y_f}{\tau_{bf}},\quad \frac{dy_s}{dt}=-\frac{y_s}{\tau_{bs}}$$

$$\mathrm{somatic\ spike:}\quad y_f\leftarrow A_f,\quad y_s\leftarrow A_s$$

Each spike resets the waveform rather than adding another one. The influx scale absorbs the driving-force factor; no separate calcium reversal-potential term is used.

$$J_i=J_{max,i}s_i B(V_i+V_{bAP})$$

Pooled branch gate (pooled_gate=True, used in branch/). Compartment i is a branch of K_i synapses with independent clefts, and its input n_i counts events. AMPA adds gbar_i w_syn per event. v_i, the sum of u^m over clefts, is linear in events. s_i is the summed occupancy, saturating in mean field. k_eff makes one isolated event reproduce the single-synapse gate. Calcium is the branch average of spine calcium.

$$\frac{dv_i}{dt}=-\frac{m\,v_i}{\tau_u},\quad v_i(t^+)=v_i(t^-)+n_ia_u^m,\qquad \frac{ds_i}{dt}=k_{eff}v_i\left(1-\frac{s_i}{K_i}\right)-\frac{s_i}{\tau_N}$$

$$k_{eff}=\frac{1-e^{-k_ba_u^m\tau_u/m}}{a_u^m\tau_u/m},\qquad J_i=\frac{J_{max,i}}{K_i}\,s_i\,B(V_i+V_{bAP})$$

## 3 · Free calcium, binding and extrusion

c_i is free calcium, b_i is bound calcium, and B_tot,i is total buffer capacity. Capacity is fixed during the model run unless externally prescribed; there is no learned buffer controller yet.

$$R_i=k_{on}c_i(B_{tot,i}-b_i)-k_{off}b_i$$

$$\frac{dc_i}{dt}=J_i-\frac{c_i-c_0}{\tau_c}-R_i,\qquad \frac{db_i}{dt}=R_i$$

$$T_i=c_i+b_i,\qquad \frac{dT_i}{dt}=J_i-\frac{c_i-c_0}{\tau_c}$$

Binding redistributes calcium between free and bound pools. The estimator reads free c_i, not total T_i, occupancy s_i or calcium peaks.

<!-- pagebreak -->

# Bayesian estimate of normal activity

## 4 · Observations and likelihood

Sample c_i every delta = 1 ms. At inference ticks t_k, spaced Delta = 100 ms apart, use the last W = 2000 ms of samples. No incomplete window is scored. With these defaults the first tick is at 2000 ms.

$$L=W/\delta=2000,\qquad M_{i,k}=\frac{1}{L}\sum_{j=0}^{L-1}c_i(t_k-j\delta)$$

Each compartment has an independent joint belief over theta = (mu,v): the mean and coefficient of variation of rolling means. The gamma likelihood has shape a and rate r (not scale).

$$a=v^{-2},\qquad r=a/\mu,\qquad f(x\mid\theta)=\frac{r^a}{\Gamma(a)}x^{a-1}e^{-rx},\quad x>0$$

$$E[M\mid\theta]=\mu,\qquad \mathrm{Var}(M\mid\theta)=\mu^2v^2$$

The prior is uniform in log mu and uniform on a logarithmic grid of 121 points over v in [0.001,3]. The mean is integrated out exactly, because for fixed a a prior uniform in log rate is conjugate, so only v remains a grid. Inference then does not depend on the units of calcium. All sums below run over the v grid; omit i for readability.

## 5 · Forget, predict, then learn

The belief is three discounted statistics. Forget toward the prior with tau_slow = 60000 ms; superscript minus denotes the state before this tick's observation.

$$\rho=e^{-\Delta/\tau_{slow}},\qquad (n,S,L)_k^-=\rho\,(n,S,L)_{k-1}$$

The posterior over v, with a = 1/v^2, follows in closed form:

$$\ln p_k^-(v)=-n\ln\Gamma(a)+(a-1)L+\ln\Gamma(na)-na\ln S+\mathrm{const}$$

Predict from that belief before incorporating M_k. I is the regularized incomplete beta function:

$$F_k(x)=\sum_v p_k^-(v)\,I_{x/(x+S)}(a,\,na),\qquad \widehat M_k=\sum_v p_k^-(v)\frac{aS}{na-1}$$

$$[\ell_k,h_k]=[F_k^{-1}(q),F_k^{-1}(1-q)],\quad q=0.025$$

$$U_k=F_k(M_k),\qquad D_k=M_k-\widehat M_k$$

Then learn with a tempered likelihood. T_evidence defaults to 2000 ms; an independent baseline may set a longer value to account approximately for correlation.

$$\omega=\Delta/T_{evidence},\qquad n_k=n_k^-+\omega,\quad S_k=S_k^-+\omega M_k,\quad L_k=L_k^-+\omega\ln M_k$$

This is the same posterior as tempering f(M_k|theta)^omega on the joint belief, with n_0 = S_0 = L_0 = 0.

The first scored window also receives weight Delta/T_evidence, not W/T_evidence. n_k is discounted likelihood weight, not a certified count of independent observations.

<!-- pagebreak -->

# Signed plasticity and AMPA feedback

## 6 · Predictive tails supply the drive

The predictive interval describes a future rolling calcium mean, not a confidence interval for the parameter mu. Use the pre-learning percentile U_i,k to obtain the bounded signed drive d_i,k:

$$d_{i,k}=-(q-U_{i,k})/q\qquad\mathrm{if}\ U_{i,k}<q$$

$$d_{i,k}=0\qquad\mathrm{if}\ q\leq U_{i,k}\leq 1-q$$

$$d_{i,k}=[U_{i,k}-(1-q)]/q\qquad\mathrm{if}\ U_{i,k}>1-q$$

Thus -1 <= d <= 1. High calcium potentiates and low calcium depresses by modelling choice. This score is not the probability that input rate changed. The predictive band includes both expected observation variability and uncertainty about the parameters.

$$\mathrm{Var}_{pred}(M)=E_{p_k^-}[\mu^2v^2]+\mathrm{Var}_{p_k^-}(\mu)$$

## 7 · Bounded conductance updates

Let gbar_i,0 be initial quantal conductance. Bounds remain tied to that initial value, not to a moving baseline.

$$g_{min,i}=0.25\bar g_{i,0},\qquad g_{max,i}=4\bar g_{i,0}$$

$$g_{*,i,k}=g_{max,i}\ \mathrm{if}\ d_{i,k}>0;\qquad g_{*,i,k}=g_{min,i}\ \mathrm{if}\ d_{i,k}<0$$

For nonzero drive, after the 60000 ms learning-only period, the feedback mode applies this exact discrete update with tau_g = 60000 ms:

$$\bar g_{i,k}=g_{*,i,k}+(\bar g_{i,k-1}-g_{*,i,k})e^{-|d_{i,k}|\Delta/\tau_g}$$

For zero drive, during warmup (t_k <= 60000 ms), or in observe-only mode, gbar is unchanged. Belief learning continues during warmup. Between inference ticks, the simulator holds gbar fixed.

The exponential update is the interval solution of the following relaxation equation with drive and target held constant. It is an interpretation of the tick update, not an additional continuously integrated simulator state.

$$\frac{d\bar g_i}{dt}=\frac{|d_{i,k}|}{\tau_g}(g_{*,i,k}-\bar g_i)$$

Starting within the bounds guarantees containment. It does not guarantee feedback stability or zero stationary drift. Measured with one compartment in the earlier 30-500 Hz domain (J_max 0.5, frozen step): under constant input gbar does not stay at its start. It settles at about 2.6 nS at 100 Hz from any initial value, and turns into a repeating cycle at tau_g of 10 s or less. Learning a level shift as increased variability can temporarily mask a reversal; the present layer has no separate latent regime or change-point state. The layer has not yet been run closed-loop at physiological rates (STATUS.md).

<!-- pagebreak -->

# Defaults, initialization and execution

## 8 · Biophysical defaults

N = 4 compartments; integration timestep dt = 0.1 ms. Values below are the constructor defaults of NMDANeuron; they are model settings, not fitted biological constants.

Per-synapse reading (physio/, 3 November 2026): one input stream is one presynaptic neuron at physiological rates, with input weight w = 12. J_max = 78.3 µM/ms is fitted so that one isolated event at B_tot = 44 µM (kappa = 20) raises free calcium by 0.5 µM with the exact u^2 step (physio/p0_retune.py). The earlier 30-500 Hz domain used J_max = 0.50 and B_tot = 5 µM. In pooled mode (branch/), w_syn = 1 nS per event and influx is J_max/K per branch.

$$C_d=C_s=150\ \mathrm{pF},\quad g_L=g_c=10\ \mathrm{nS},\quad \bar g_{i,0}=1\ \mathrm{nS}$$

$$E_L=-70,\ E_a=0,\ V_{th}=-50,\ V_{reset}=-65\ \mathrm{mV}$$

$$\tau_a=3,\ t_{ref}=2,\ \tau_u=1,\ \tau_N=40\ \mathrm{ms};\quad a_u=1,\ m=2$$

$$k_b=0.315\ \mathrm{ms}^{-1},\quad J_{max,i}=78.3\ \mathrm{\mu M/ms},\quad [\mathrm{Mg}^{2+}]=2\ \mathrm{mM}$$

$$A_f=75,\ A_s=25\ \mathrm{mV};\qquad \tau_{bf}=3,\ \tau_{bs}=25\ \mathrm{ms}$$

$$c_0=0.1,\ B_{tot,i}=44\ \mathrm{\mu M};\quad \tau_c=12\ \mathrm{ms}$$

$$k_{on}=0.5\ \mathrm{\mu M}^{-1}\mathrm{ms}^{-1},\quad k_{off}=1\ \mathrm{ms}^{-1},\quad K_d=k_{off}/k_{on}=2\ \mathrm{\mu M}$$

## 9 · Initial conditions and numerical order

$$V_i=V_s=E_L,\quad g_i=u_i=s_i=y_f=y_s=0,\quad c_i=c_0$$

$$b_i=\frac{k_{on}c_0B_{tot,i}}{k_{off}+k_{on}c_0},\qquad \bar g_i=\bar g_{i,0},\quad (n,S,L)_i=0$$

Start with an empty observation ring buffer and zero evidence count. Each membrane step: decay AMPA and deliver input; advance voltages by simultaneous explicit Euler and apply soma events; decay/kick glutamate and update occupancy; decay/reset bAP; compute J; advance calcium with J fixed; then sample, predict, learn and, when enabled, update gbar.

Occupancy is advanced exactly for a binding rate held constant over the step. That rate is the step mean of k_b u^m while u decays freely from its post-kick value (u_quadrature="exact"):

$$A_i=k_b\,u_i^m\,\phi,\qquad \phi=\frac{1-e^{-m\,dt/\tau_u}}{m\,dt/\tau_u},\qquad s_i\leftarrow s_\infty+(s_i-s_\infty)e^{-(A_i+1/\tau_N)dt},\quad s_\infty=\frac{A_i}{A_i+1/\tau_N}$$

At dt = 0.1 ms, phi = 0.906. The historical option u_quadrature="frozen" sets phi = 1, which holds the post-kick u for the whole step; it is kept only to show why it was replaced (physio/p2).

Calcium uses two Strang substeps of 0.05 ms: exact influx/extrusion half-step with b fixed, exact binding step with T fixed, then another influx/extrusion half-step. This retains binding kinetics rather than imposing buffer equilibrium. Splitting and membrane discretization errors remain.

## 10 · Implementation and interpretation

This sheet specifies NMDANeuron (neuron/model.py) with its defaults: u_quadrature="exact" and plasticity_mode="feedback". Use "observe-only" for inference at fixed AMPA or "off" for biophysics alone. neuron/fast.py is the compiled equivalent (Sim for "off", PlasticSim for the Bayesian modes); tests/test_kernel.py holds it to the Python model.

Normalize beliefs in log space. Reject nonpositive or nonfinite observations. Prediction may discard at most 1e-12 mass. Flag at least 1% posterior mass on v-grid edges, or a mean outside [0.01,100] µM. Correlation weighting is approximate and does not guarantee calibrated uncertainty; measured coverage of the 95% band is about 0.97.

Numerical accuracy: the coupled scheme is first order in dt. With the exact u^2 step, calcium at dt = 0.1 ms is within about 2% of the dt limit at 1-64 Hz (physio/p2). The historical frozen step over-integrates binding by 10.3% at dt = 0.1 and put calcium 7-11% above the dt limit; k_b = 0.315 was tuned under it. Source of truth: neuron/model.py, lif.py, calcium.py and plasticity.py. Consecutive runs preserve states and clocks; reset_state() restores initial conductances, beliefs and empty observation history.
