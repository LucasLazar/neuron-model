"""The full neuron: dendritic LIF + split NMDA gate + buffered calcium + Bayesian
AMPA plasticity. MODEL_ODES.md is the specification; this file is its reference
implementation.

    Voltage (lif.py): AMPA conductance per compartment, dendrites coupled to a
    spiking soma; optional somatic GABA_A inhibition.

    NMDA gate (per compartment i), calcium only -- J never enters voltage:
        du/dt = -u/tau_u,                 pre event:  u += a_u (1 - u)
        ds/dt = k_b u^m (1 - s) - s/tau_N,           m = 2
        B(V)  = 1 / (1 + ([Mg]/3.57) exp(-0.062 V))   Jahr & Stevens 1990
        J     = J_max s B(V + V_bAP)
    so influx needs glutamate (s) AND depolarisation (B): an AND gate.

    Back-propagating AP, global, seen ONLY by the gate:
        V_bAP = y_f + y_s, decaying with tau_bf, tau_bs; a somatic spike resets
        y_f = A_f, y_s = A_s (stereotyped, not summed).

    Pooled branch gate (pooled_gate=True): a compartment is a branch of n_syn
    synapses with independent clefts; input counts events. v = sum of u^m is
    linear in events and the summed occupancy saturates in mean field:
        dv/dt = -m v/tau_u,  v += n a_u^m;   ds/dt = k_eff v (1 - s/n_syn) - s/tau_N
        J = (J_max/n_syn) s B(V + V_bAP)    (branch-average calcium)

    Calcium (calcium.py): free c and bound b with explicit binding kinetics,
        dc/dt = J - (c - c0)/tau_c - R,  db/dt = R,  R = kon c (Btot - b) - koff b

    Plasticity (plasticity.py): each compartment learns the distribution of its
    2 s rolling-mean calcium; predictive tail percentiles drive gbar toward
    0.25x or 4x its initial value.

Defaults are the current calibration: the per-synapse reading (physio/), where
one input stream is one presynaptic neuron and one isolated event at Btot = 44 uM
(kappa ~ 20) raises free calcium by 0.5 uM. physio/p0_retune.py refits J_max.

Units: ms, uM, mV, pF, nS, pA.
"""
import numpy as np

from .calcium import calcium_step
from .lif import DendriticLIF
from .plasticity import PlasticityConfig, RollingCalciumEstimator, conductance_update

# Fitted by physio/p0_retune.py (stored in physio/results/retune.json): one
# isolated event at Btot = 44 uM peaks 0.5 uM above rest, dt = 0.1, exact u^2 step.
J_MAX_SPINE = 78.29712874708824


def mg_block(V, Mg=2.0):
    """Jahr & Stevens (1990) magnesium unblock, B(V)."""
    return 1.0 / (1.0 + (Mg / 3.57) * np.exp(-0.062 * V))


class NMDANeuron:
    """DendriticLIF + per-compartment NMDA gate, buffered calcium and Bayesian
    plasticity. J = J_max * s * B(V_d + V_bAP)."""

    def __init__(
        self,
        n_dend=4,
        dt=0.1,
        # --- NMDA gate ---
        tau_N=40.0,     # receptor deactivation / coincidence window, ms (GluN2A range)
        tau_u=1.0,      # cleft glutamate clearance, ms (Clements 1992)
        a_u=1.0,        # cleft kick per pre event, saturating
        k_b=0.315,      # binding rate, /ms; one event takes s to 0.14
        m_coop=2,       # receptor cooperativity (two glutamates)
        u_quadrature="exact",  # "exact": binding rate is the step mean of k_b u^m
                        # under u's own decay (MODEL_ODES.md sec 9). "frozen" holds
                        # the post-kick u for the step and over-integrates by 10.3%
                        # at dt = 0.1; kept only to show why it was replaced (physio/p2)
        J_max=J_MAX_SPINE,  # influx at s = 1, B = 1, uM/ms
        Mg=2.0,         # external Mg2+, mM
        # --- bAP waveform seen only by the NMDA gate (Shouval 2002) ---
        A_f=75.0, A_s=25.0,      # fast/slow amplitudes, mV
        tau_bf=3.0, tau_bs=25.0, # fast/slow decays, ms
        # --- calcium, per compartment ---
        c0=0.1, tau_c=12.0,                # rest, extrusion
        Btot=44.0, kon=0.5, koff=1.0,      # buffer (Kd = 2 uM); 44 uM gives kappa ~ 20
        # --- plasticity ---
        plasticity_mode="feedback",  # "off", "observe-only" or "feedback"
        plasticity_config=None,
        # --- pooled branch gate (branch/) ---
        pooled_gate=False,
        n_syn=1.0,      # synapses per branch: calcium is their average, J_max/n_syn
        w_syn=1.0,      # AMPA weight per event in pooled mode (x gbar, nS)
        # --- everything else goes to the LIF ---
        **lif_kwargs,
    ):
        self.lif = DendriticLIF(n_dend=n_dend, dt=dt, **lif_kwargs)
        if plasticity_mode not in ("off", "observe-only", "feedback"):
            raise ValueError("unknown plasticity_mode")
        self.plasticity_mode = plasticity_mode
        self._initial_gbar = self.lif.gbar.copy()
        if plasticity_mode != "off" and (
                not np.all(np.isfinite(self._initial_gbar)) or np.any(self._initial_gbar <= 0)):
            raise ValueError("plasticity requires finite positive initial gbar")
        self.plasticity_config = plasticity_config or PlasticityConfig()
        ratio = self.plasticity_config.sample_ms / dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("sample_ms must be an integer multiple of dt")
        if min(tau_c, kon, koff) <= 0 or Btot < 0:
            raise ValueError("invalid calcium kinetics")
        self._sample_steps = int(round(ratio))
        self.estimator = None
        if plasticity_mode != "off":
            self.estimator = RollingCalciumEstimator(n_dend, self.plasticity_config)
        self.n_dend, self.dt = n_dend, dt
        self.tau_N, self.Mg = tau_N, Mg
        self.tau_u, self.a_u, self.k_b, self.m_coop = tau_u, a_u, k_b, int(m_coop)
        if u_quadrature not in ("frozen", "exact"):
            raise ValueError("unknown u_quadrature")
        self.u_quadrature = u_quadrature
        # mean of u(t)^m / u(0)^m over one step of free decay
        x = self.m_coop * dt / tau_u
        self._u_phi = (1.0 - np.exp(-x)) / x if u_quadrature == "exact" else 1.0
        self.pooled_gate = bool(pooled_gate)
        if self.pooled_gate:
            if u_quadrature != "exact":
                raise ValueError("pooled_gate needs u_quadrature='exact'")
            # v = sum over clefts of u^m decays at m/tau_u; one event adds a_u^m.
            # k_eff makes one isolated event's occupancy match the nonlinear
            # single-synapse gate: s1 = 1 - exp(-k_b a_u^m tau_u/m).
            area = a_u ** self.m_coop * tau_u / self.m_coop
            self.k_eff = (1.0 - np.exp(-k_b * area)) / area
            self._a_um = a_u ** self.m_coop
            self._dec_v = np.exp(-self.m_coop * dt / tau_u)
        self.n_syn = np.full(n_dend, float(n_syn))
        self.w_syn = float(w_syn)
        self.A_f, self.A_s = A_f, A_s
        self.c0, self.tau_c = c0, tau_c
        self.kon, self.koff = kon, koff

        # per-compartment, so a later controller can set them branch by branch
        self.J_max = np.full(n_dend, float(J_max))
        self.Btot = np.full(n_dend, float(Btot))
        self._J_branch = self.J_max / self.n_syn     # pooled mode: branch-average influx scale

        # exact per-step decay factors for the linear gates
        self._dec_u = np.exp(-dt / tau_u)
        self._dec_bf = np.exp(-dt / tau_bf)
        self._dec_bs = np.exp(-dt / tau_bs)

        self.reset_state()

    def b_rest(self):
        return self.kon * self.c0 * self.Btot / (self.koff + self.kon * self.c0)

    def kappa(self, Btot=None):
        """Buffer capacity at rest, dB/dc at c0."""
        Btot = self.Btot if Btot is None else Btot
        return Btot * self.kon * self.koff / (self.koff + self.kon * self.c0) ** 2

    def tau_eff(self, Btot=None):
        """Slowest calcium relaxation near rest, tau_c (1 + kappa), ms."""
        return self.tau_c * (1.0 + self.kappa(Btot))

    def reset_state(self):
        self.lif.reset_state()
        self.lif.gbar[:] = self._initial_gbar
        self._step_count = 0
        self.last_plasticity = None
        if self.estimator is not None:
            self.estimator.reset()
        self.s = np.zeros(self.n_dend)      # receptor occupancy (summed, in pooled mode)
        self.u = np.zeros(self.n_dend)      # cleft glutamate (v = sum of u^m, in pooled mode)
        self.c = np.full(self.n_dend, self.c0)
        self.b = self.b_rest().copy()
        self.y_f = 0.0                      # bAP components (global)
        self.y_s = 0.0
        self.J = np.zeros(self.n_dend)      # last influx, for recording

    # ------------------------------------------------------------------

    def step(self, w_in=None, force_post=False, inh_in=None):
        """One timestep of the full model.  Returns True on a somatic spike.

        w_in       : per-compartment input this step -- synaptic weight, or in
                     pooled mode the number of input events.
        force_post : trigger the bAP waveform as if the soma had spiked,
                     without touching the voltages.  Stands in for spikes
                     evoked by un-modelled drive in open-loop tests.
        inh_in     : inhibitory weight arriving at the soma this step.
        """
        dt = self.dt

        # 1. voltage pathway (AMPA only -- NMDA carries no current)
        if self.pooled_gate and w_in is not None:
            spiked = self.lif.step(np.asarray(w_in, dtype=float) * self.w_syn, inh_in=inh_in)
        else:
            spiked = self.lif.step(w_in, inh_in=inh_in)

        # 2. NMDA gates: glutamate binding (saturating) and the bAP
        if self.pooled_gate:
            # u holds v = sum over the branch's clefts of u^m.  Clefts are
            # independent, so v is linear in events.  s is the summed
            # occupancy; a synapse's own earlier binding is represented in
            # mean field by (1 - s/n_syn): with Poisson input a synapse sees
            # its own time-averaged occupancy when it fires (branch/ b0).
            self.u *= self._dec_v
            if w_in is not None:
                self.u += np.asarray(w_in, dtype=float) * self._a_um
            A = self.k_eff * self.u * self._u_phi
            rate = A / self.n_syn + 1.0 / self.tau_N
            s_inf = A / rate
            self.s = s_inf + (self.s - s_inf) * np.exp(-dt * rate)
        else:
            # cleft transient u, then occupancy s driven by u^m.  With the
            # binding rate held constant over the step, the s equation is linear
            # and integrated exactly.  "exact" sets that rate to the step mean
            # of k_b u^m under u's own decay; "frozen" uses the post-kick value.
            # A positive input of any weight gives one saturating cleft kick.
            self.u *= self._dec_u
            if w_in is not None:
                pre = np.asarray(w_in, dtype=float) > 0
                self.u[pre] += self.a_u * (1.0 - self.u[pre])
            A = self.k_b * self.u ** self.m_coop * self._u_phi   # binding rate this step
            rate = A + 1.0 / self.tau_N
            s_inf = A / rate
            self.s = s_inf + (self.s - s_inf) * np.exp(-dt * rate)
        self.y_f *= self._dec_bf
        self.y_s *= self._dec_bs
        if spiked or force_post:
            self.y_f, self.y_s = self.A_f, self.A_s   # stereotyped, not summed

        # 3. the AND gate: calcium influx
        V_eff = self.lif.V + self.y_f + self.y_s
        if self.pooled_gate:
            self.J = self._J_branch * self.s * mg_block(V_eff, self.Mg)
        else:
            self.J = self.J_max * self.s * mg_block(V_eff, self.Mg)

        # 4. calcium with J fixed over the step; then sample, infer, update gbar
        for i in range(self.n_dend):
            total, bound = calcium_step(float(self.c[i] + self.b[i]), float(self.b[i]),
                float(self.J[i]), float(self.Btot[i]), dt,
                self.c0, self.tau_c, self.kon, self.koff, nsub=2)
            self.c[i], self.b[i] = total - bound, bound
        self._step_count += 1
        self.last_plasticity = None
        if self.estimator is not None and self._step_count % self._sample_steps == 0:
            diag = self.estimator.sample(self.c)
            if diag is not None:
                cfg = self.plasticity_config
                if self.plasticity_mode == "feedback" and diag['time_ms'] > cfg.warmup_ms:
                    for i in range(self.n_dend):
                        self.lif.gbar[i] = conductance_update(self.lif.gbar[i],
                            self._initial_gbar[i], diag['drive'][i], cfg.update_ms, cfg)
                self.last_plasticity = dict(diag, gbar=self.lif.gbar.copy(), calcium=self.c.copy())
        return spiked

    # ------------------------------------------------------------------

    def run(self, duration, w_in=None, force_post=None, record_ms=None, inh_in=None):
        """Simulate for `duration` ms.

        w_in       : (n_steps, n_dend) synaptic weight per step (event counts in
                     pooled mode), or None.
        force_post : optional array of times (ms) at which a bAP is triggered
                     (see step()).
        inh_in     : optional (n_steps,) inhibitory weight at the soma per step.
        record_ms  : optional downsampling interval, aligned to the absolute
                     simulation step. None records every step.
        Returns a dict of traces and output spike times and, when plasticity is
        on, a 'plasticity' dictionary recorded only on inference ticks.
        Consecutive calls continue the same trajectory; reset_state() restarts.
        """
        n_steps = int(round(duration / self.dt))
        stride = 1 if record_ms is None else int(round(record_ms / self.dt))
        if stride < 1 or (record_ms is not None and not np.isclose(stride * self.dt, record_ms)):
            raise ValueError("record_ms must be an integer multiple of dt")
        start_step = int(round(self.lif.t / self.dt))
        n_records = (start_step + n_steps) // stride - start_step // stride
        if w_in is not None:
            w_in = np.atleast_2d(np.asarray(w_in, dtype=float))
            if w_in.shape[0] != n_steps:
                raise ValueError(f"w_in has {w_in.shape[0]} rows, expected {n_steps}")

        force_steps = set()
        if force_post is not None:
            t0 = self.lif.t
            force_steps = {int(round((t - t0) / self.dt)) - 1
                           for t in np.atleast_1d(force_post)}

        rec = {k: np.empty((n_records, self.n_dend))
               for k in ("s", "u", "J", "c", "b", "V", "g")}
        rec["t"] = np.empty(n_records)
        rec["Vs"] = np.empty(n_records)
        spikes = []
        plasticity = []
        j = 0

        for k in range(n_steps):
            fired = self.step(None if w_in is None else w_in[k], force_post=(k in force_steps),
                              inh_in=None if inh_in is None else inh_in[k])
            if (start_step + k + 1) % stride == 0:
                rec["t"][j] = self.lif.t
                rec["g"][j], rec["V"][j], rec["Vs"][j] = self.lif.g, self.lif.V, self.lif.Vs
                rec["s"][j], rec["u"][j], rec["J"][j] = self.s, self.u, self.J
                rec["c"][j], rec["b"][j] = self.c, self.b
                j += 1
            if self.last_plasticity is not None:
                plasticity.append(self.last_plasticity)
            if fired:
                spikes.append(self.lif.t)

        rec["spikes"] = np.array(spikes)
        if self.estimator is not None:
            rec["plasticity"] = ({key: np.array([d[key] for d in plasticity])
                                  for key in plasticity[0]} if plasticity else {})
        return rec


def spike_train(times, n_steps, dt, comp, n_dend, weight=1.0):
    """Deterministic input events at `times` (ms) onto compartment `comp`."""
    w = np.zeros((n_steps, n_dend))
    for t in np.atleast_1d(times):
        k = int(round(t / dt))
        if 0 <= k < n_steps:
            w[k, comp] += weight
    return w
