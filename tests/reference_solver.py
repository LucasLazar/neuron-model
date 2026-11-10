"""Independent reference solution of the ODE sheet, for order verification.

The right-hand side is written from MODEL_ODES.md, sharing no code with the
simulator: one compartment, per-synapse split gate, plasticity off. Between events
it is integrated by Radau at rtol 1e-10. Callers put input events on the 0.2 ms
grid, so every tested step size can represent them exactly.

Used by tests/test_convergence.py and physio/p2_convergence.py.
"""
import numpy as np
from scipy.integrate import solve_ivp

WEIGHT = 12.0

P = dict(Cd=150., Cs=150., gL=10., gc=10., EL=-70., Ea=0., Vth=-50., Vreset=-65., tref=2., tau_a=3., gbar=1.,
         tau_u=1., tau_N=40., a_u=1., k_b=.315, J_max=78.29712874708824, Mg=2., A_f=75., A_s=25., tau_bf=3., tau_bs=25.,
         c0=.1, tau_c=12., kon=.5, koff=1.)
G, V, VS, U, S, YF, YS, C, B = range(9)


def rhs(t, y, capacity, soma_held):
    p = P
    dy = np.empty(9)
    dy[G] = -y[G] / p['tau_a']
    dy[V] = (p['gL'] * (p['EL'] - y[V]) + y[G] * (p['Ea'] - y[V]) + p['gc'] * (y[VS] - y[V])) / p['Cd']
    dy[VS] = 0.0 if soma_held else (p['gL'] * (p['EL'] - y[VS]) + p['gc'] * (y[V] - y[VS])) / p['Cs']
    dy[U] = -y[U] / p['tau_u']
    dy[S] = p['k_b'] * y[U] ** 2 * (1 - y[S]) - y[S] / p['tau_N']
    dy[YF] = -y[YF] / p['tau_bf']
    dy[YS] = -y[YS] / p['tau_bs']
    block = 1.0 / (1.0 + (p['Mg'] / 3.57) * np.exp(-0.062 * (y[V] + y[YF] + y[YS])))
    influx = p['J_max'] * y[S] * block
    binding = p['kon'] * y[C] * (capacity - y[B]) - p['koff'] * y[B]
    dy[C] = influx - (y[C] - p['c0']) / p['tau_c'] - binding
    dy[B] = binding
    return dy


def threshold(t, y, capacity, soma_held):
    return y[VS] - P['Vth']


threshold.terminal = True
threshold.direction = 1


def reference(event_times, duration, capacity):
    """Reference on the 1 ms grid. Every event and sample is a solver breakpoint.

    A state sampled at time t excludes an input arriving at t, matching the
    discrete model, where that input belongs to the following step.
    Returns (sample times, states, spike times).
    """
    bound = P['kon'] * P['c0'] * capacity / (P['koff'] + P['kon'] * P['c0'])
    y = np.array([0., P['EL'], P['EL'], 0., 0., 0., 0., P['c0'], bound])
    samples = np.arange(1.0, duration + .5, 1.0)
    out = np.empty((len(samples), 9))
    pending, t, held_until, spikes, written = list(event_times), 0.0, -np.inf, [], 0
    tolerance = 1e-9
    while written < len(samples):
        if abs(t - samples[written]) < tolerance:
            out[written] = y
            written += 1
            if written == len(samples):
                break
        while pending and pending[0] <= t + tolerance:
            pending.pop(0)
            y[G] += P['gbar'] * WEIGHT
            y[U] += P['a_u'] * (1 - y[U])
        held = held_until > t + tolerance
        breaks = [samples[written]]
        if pending:
            breaks.append(pending[0])
        if held:
            breaks.append(held_until)
        sol = solve_ivp(rhs, (t, min(breaks)), y, method='Radau', rtol=1e-10, atol=1e-11, args=(float(capacity), held),
                        events=None if held else threshold, first_step=1e-3)
        if not sol.success:
            raise RuntimeError(sol.message)
        if not held and len(sol.t_events[0]):
            t, y = float(sol.t_events[0][0]), sol.y_events[0][0].copy()
            spikes.append(t)
            y[VS] = P['Vreset']
            y[YF], y[YS] = P['A_f'], P['A_s']
            held_until = t + P['tref']
        else:
            t, y = float(sol.t[-1]), sol.y[:, -1].copy()
    return samples, out, np.array(spikes)
