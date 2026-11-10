"""
Multi-compartment conductance-based leaky integrate-and-fire neuron.

N dendritic compartments, each with an AMPA conductance, coupled to a single
somatic compartment that carries the spike threshold.

Units throughout: pF, nS, mV, ms, pA.  These are chosen so no conversion
factors appear anywhere -- nS * mV = pA exactly, and pA / pF = mV/ms exactly.

    State (per dendritic compartment i):
        g[i]   AMPA conductance
        V[i]   membrane potential
    State (soma):
        Vs     membrane potential

    Dynamics:
        dg[i]/dt = -g[i] / tau_a
        dV[i]/dt = ( gL*(EL - V[i]) + g[i]*(Ea - V[i]) + gc*(Vs - V[i]) ) / Cd
        dVs/dt   = ( gL*(EL - Vs)   + sum_i gc*(V[i] - Vs) ) / Cs

    Events:
        presynaptic spike at compartment i:  g[i] += gbar[i] * w
        Vs >= Vth:  record spike, Vs = Vreset, hold for tref

    Optional somatic inhibition (branch/ study): a GABA_A-type conductance gI
    on the soma, adding gI*(EI - Vs) to the somatic current.
        dgI/dt = -gI / tau_I,   inhibitory events of weight wI:  gI += gbar_I * wI
    With no inhibitory input gI stays exactly 0.

    gbar is the QUANTAL conductance: what one unit-weight presynaptic spike
    opens.  It is per-compartment (shape (n_dend,)) so that plasticity can
    write to it -- g[i] itself decays with tau_a and cannot hold a lasting
    change.  A scalar at construction is broadcast to all compartments.

Analytic checks of this class are in tests/test_lif.py.
"""

import numpy as np


class DendriticLIF:

    def __init__(
        self,
        n_dend=1,
        Cd=150.0,       # dendritic capacitance, pF
        Cs=150.0,       # somatic capacitance, pF
        gL=10.0,        # leak conductance, nS (both compartments)
        EL=-70.0,       # leak reversal / resting potential, mV
        gc=10.0,        # dendrite-soma coupling conductance, nS
        Ea=0.0,         # AMPA reversal potential, mV
        tau_a=3.0,      # AMPA decay time constant, ms
        gbar=1.0,       # quantal conductance: what a unit-weight synapse
                        # opens, nS.  Scalar (broadcast) or (n_dend,).
                        # Per-compartment and mutable: the plastic variable.
        Vth=-50.0,      # spike threshold, mV
        Vreset=-65.0,   # reset potential, mV
        tref=2.0,       # refractory period, ms
        dt=0.1,         # integration timestep, ms
        EI=-75.0,       # somatic inhibitory (GABA_A) reversal, mV
        tau_I=6.0,      # inhibitory conductance decay, ms
        gbar_I=1.0,     # conductance opened by a unit-weight inhibitory event, nS
    ):
        self.n_dend = n_dend
        self.Cd, self.Cs = Cd, Cs
        self.gL, self.EL, self.gc = gL, EL, gc
        self.Ea, self.tau_a = Ea, tau_a
        # per-compartment so plasticity can potentiate one branch alone;
        # np.full broadcasts a scalar, so gbar=1.0 keeps the old behaviour
        self.gbar = np.full(n_dend, gbar, dtype=float)
        self.Vth, self.Vreset, self.tref = Vth, Vreset, tref
        self.EI, self.tau_I, self.gbar_I = EI, tau_I, gbar_I
        self.decay_I = np.exp(-dt / tau_I)
        self.dt = dt

        # Equation 1 has a closed-form solution, so decay g exactly rather
        # than with an Euler step.  Precomputed once, no accuracy cost.
        self.decay = np.exp(-dt / tau_a)

        self.reset_state()

    def reset_state(self):
        """Return all state variables to rest."""
        self.g = np.zeros(self.n_dend)
        self.gI = 0.0
        self.V = np.full(self.n_dend, self.EL)
        self.Vs = self.EL
        self.t = 0.0
        self.refractory_until = -np.inf

    # ------------------------------------------------------------------
    # single timestep
    # ------------------------------------------------------------------

    def step(self, w_in=None, g_clamp=None, inh_in=None):
        """Advance the model by one timestep.

        w_in     : array of shape (n_dend,), total synaptic weight arriving
                   at each compartment during this step.  None means no input.
        g_clamp  : if given, g is forced to this value instead of evolving.
                   Used for steady-state tests.
        inh_in   : total inhibitory weight arriving at the soma this step.

        Returns True if the soma spiked during this step.
        """
        dt = self.dt

        # --- equation 1': decay the synaptic conductance ---------------
        if g_clamp is None:
            self.g *= self.decay
        else:
            self.g = np.full(self.n_dend, float(g_clamp))
        self.gI *= self.decay_I

        # --- rule 4: deliver presynaptic spikes -----------------------
        # Decay first, then deliver, so that g is exactly gbar[i]*w in the
        # step a spike arrives rather than gbar[i]*w*exp(-dt/tau).
        if w_in is not None and g_clamp is None:
            self.g += self.gbar * np.asarray(w_in, dtype=float)
        if inh_in is not None:
            self.gI += self.gbar_I * float(inh_in)

        # --- equations 2' and 3' ---------------------------------------
        # Both derivatives are evaluated from the OLD voltages before
        # either is assigned.  Updating V before using it in the somatic
        # equation would break the antisymmetry of the coupling terms and
        # the pair would silently stop conserving charge.
        I_dend = (
            self.gL * (self.EL - self.V)         # leak
            + self.g * (self.Ea - self.V)        # synaptic
            + self.gc * (self.Vs - self.V)       # coupling to soma
        )
        I_soma = (
            self.gL * (self.EL - self.Vs)                 # leak
            + np.sum(self.gc * (self.V - self.Vs))        # coupling from dendrites
            + self.gI * (self.EI - self.Vs)               # somatic inhibition (0 unless driven)
        )

        dV = dt * I_dend / self.Cd
        dVs = dt * I_soma / self.Cs

        self.V = self.V + dV

        refractory = self.t < self.refractory_until
        if not refractory:
            self.Vs = self.Vs + dVs
        # During refractory the soma is clamped, but the dendrites keep
        # integrating and still feel the pull of the held-down soma.

        self.t += dt

        # --- rule 5: threshold test ------------------------------------
        spiked = False
        if (not refractory) and self.Vs >= self.Vth:
            spiked = True
            self.Vs = self.Vreset
            self.refractory_until = self.t + self.tref

        return spiked

    # ------------------------------------------------------------------
    # full simulation
    # ------------------------------------------------------------------

    def run(self, duration, w_in=None, g_clamp=None):
        """Simulate for `duration` ms.

        w_in : array of shape (n_steps, n_dend) giving the total synaptic
               weight arriving at each compartment on each step, or None.

        Returns a dict of recorded traces plus the output spike times.
        """
        n_steps = int(round(duration / self.dt))

        if w_in is None:
            w_in = np.zeros((n_steps, self.n_dend))
        else:
            w_in = np.atleast_2d(np.asarray(w_in, dtype=float))
            if w_in.shape[0] != n_steps:
                raise ValueError(
                    f"w_in has {w_in.shape[0]} rows, expected {n_steps}"
                )

        t_rec = np.empty(n_steps)
        g_rec = np.empty((n_steps, self.n_dend))
        V_rec = np.empty((n_steps, self.n_dend))
        Vs_rec = np.empty(n_steps)
        spikes = []

        for k in range(n_steps):
            fired = self.step(w_in[k], g_clamp=g_clamp)
            t_rec[k] = self.t
            g_rec[k] = self.g
            V_rec[k] = self.V
            Vs_rec[k] = self.Vs
            if fired:
                spikes.append(self.t)

        return {
            "t": t_rec,
            "g": g_rec,
            "V": V_rec,
            "Vs": Vs_rec,
            "spikes": np.array(spikes),
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def poisson_input(rate, duration, dt, n_dend=1, weight=1.0, rng=None):
    """Independent Poisson spike trains, one per compartment.

    rate : mean input rate per compartment, in Hz
    Returns an array of shape (n_steps, n_dend) of arriving weight.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_steps = int(round(duration / dt))
    p = rate * dt / 1000.0          # Hz * ms -> probability per step
    counts = rng.poisson(p, size=(n_steps, n_dend))
    return counts * weight


def attenuation(gc=10.0, gL=10.0, n_dend=1, n_active="all"):
    """Steady-state fraction of dendritic depolarisation reaching the soma.

    n_active : "all"  -- every compartment is driven equally
               1      -- only one compartment is driven, the rest quiet

    Two cases, and they differ more than you would guess:

        all driven together:   k = N*gc / (gL + N*gc)          rises with N
        one driven alone:      k = gc / (gL + gc + (N-1)*gc*gL/(gL+gc))

    The second is NOT gc/(gL + N*gc).  That would be the answer if the quiet
    compartments were voltage-clamped at rest, but they are not -- they are
    pulled up by the soma and so shunt it less than a naive count suggests.
    With gL = gc = 10 nS and N = 4 the naive formula gives 0.20 and the true
    value is 0.29.

    Either way the direction is the same: adding compartments makes the
    neuron easier to drive when they all fire together, and harder to drive
    from any single one of them.
    """
    if n_active == "all":
        return n_dend * gc / (gL + n_dend * gc)
    if n_active == 1:
        return gc / (gL + gc + (n_dend - 1) * gc * gL / (gL + gc))
    raise ValueError("n_active must be 'all' or 1; "
                     "for anything else use steady_state_general")


def steady_state_general(gd, gL=10.0, EL=-70.0, gc=10.0, Ea=0.0, Cs_gL=None):
    """Exact steady state for arbitrary per-compartment conductances.

    gd : array of shape (N,), the clamped AMPA conductance of each compartment.

    Solves the full (N+1)-dimensional linear system rather than assuming any
    symmetry.  Returns (V, Vs) with V of shape (N,).  Ignores the threshold.
    """
    gd = np.atleast_1d(np.asarray(gd, dtype=float))
    N = gd.size

    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # compartment i:  -(gL + gd_i + gc) Vi + gc Vs = -(gL EL + gd_i Ea)
    for i in range(N):
        A[i, i] = -(gL + gd[i] + gc)
        A[i, N] = gc
        b[i] = -(gL * EL + gd[i] * Ea)

    # soma:  gc sum(Vi) - (gL + N gc) Vs = -gL EL
    A[N, :N] = gc
    A[N, N] = -(gL + N * gc)
    b[N] = -gL * EL

    sol = np.linalg.solve(A, b)
    return sol[:N], sol[N]


def steady_state(n_dend=1, gd=0.0, gL=10.0, EL=-70.0, gc=10.0, Ea=0.0):
    """Analytic resting voltages for a constant dendritic conductance.

    Solves the two coupled equations with both derivatives set to zero.
    Returns (Vd, Vs) in mV.  Ignores the threshold, so the result may sit
    above Vth -- which simply means the neuron fires repetitively.
    """
    # Dendrite:  gL(EL - Vd) + gd(Ea - Vd) + gc(Vs - Vd) = 0
    # Soma:      gL(EL - Vs) + N*gc(Vd - Vs) = 0
    a = gL + gd + gc                       # coefficient on Vd, dendrite eq
    b = gL * EL + gd * Ea                  # constant term, dendrite eq
    c = gL + n_dend * gc                   # coefficient on Vs, soma eq
    d = gL * EL                            # constant term, soma eq
    # Vd = (b + gc*Vs) / a ;  Vs = (d + N*gc*Vd) / c
    Vs = (d + n_dend * gc * b / a) / (c - n_dend * gc * gc / a)
    Vd = (b + gc * Vs) / a
    return Vd, Vs
