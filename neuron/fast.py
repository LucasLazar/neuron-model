"""Compiled simulator: the same arithmetic as model.NMDANeuron, several hundred times faster.

Sim is NMDANeuron(plasticity_mode='off') in C (kernel.c). PlasticSim adds the
Bayesian layer in Python on the kernel's 1 ms samples, which makes it the compiled
equivalent of 'observe-only' or 'feedback'. tests/test_kernel.py holds both to the
Python model at rtol 1e-9. Parameters are read from an NMDANeuron instance, so
the kernel cannot drift from the model's defaults.

The kernel is compiled on first import (needs a C compiler, `cc`), and again
whenever kernel.c is newer than the library.
"""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np

from .model import NMDANeuron
from .plasticity import PlasticityConfig, RollingCalciumEstimator, conductance_update

HERE = Path(__file__).resolve().parent
MAXN = 16
WEIGHT = 12.0   # AMPA weight of one per-synapse input event (x gbar 1 nS), as in physio/


def _load_kernel():
    source, library = HERE / 'kernel.c', HERE / '_kernel.so'
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        # fp-contract off keeps results independent of fused multiply-add hardware.
        temporary = HERE / f'_kernel_{os.getpid()}.so'
        subprocess.run(['cc', '-O2', '-ffp-contract=off', '-shared', '-fPIC', '-o', str(temporary), str(source), '-lm'], check=True)
        os.replace(temporary, library)
    return ctypes.CDLL(str(library))


class Params(ctypes.Structure):
    _fields_ = [('n', ctypes.c_long), ('nsub', ctypes.c_long)] + [(name, ctypes.c_double) for name in (
        'dt', 'Cd', 'Cs', 'gL', 'EL', 'gc', 'Ea', 'decay', 'Vth', 'Vreset', 'tref',
        'dec_u', 'a_u', 'k_b', 'tau_N', 'dec_bf', 'dec_bs', 'A_f', 'A_s', 'mgc',
        'c0', 'tau_c', 'kon', 'koff', 'u_phi')] + [
        ('Jmax', ctypes.c_double * MAXN), ('Btot', ctypes.c_double * MAXN), ('pooled', ctypes.c_long)] + [
        (name, ctypes.c_double) for name in ('dec_v', 'k_eff', 'a_um', 'EI', 'dec_I', 'gbar_I')] + [
        ('wscale', ctypes.c_double * MAXN), ('n_syn', ctypes.c_double * MAXN)]


class State(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double * MAXN) for name in ('g', 'V', 'u', 's', 'T', 'b', 'gbar')] + [
        (name, ctypes.c_double) for name in ('Vs', 't', 'refr', 'yf', 'ys')] + [('steps', ctypes.c_long), ('gI', ctypes.c_double)]


_kernel = _load_kernel()
_kernel.run_chunk.restype = ctypes.c_long
_kernel.run_chunk.argtypes = [ctypes.POINTER(Params), ctypes.POINTER(State), ctypes.c_void_p, ctypes.c_long,
                              ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.c_void_p, ctypes.c_long,
                              ctypes.c_void_p]


class Sim:
    """Compiled equivalent of NMDANeuron(plasticity_mode='off') at any dt.

    Btot may be a scalar or per-compartment; None takes the model default. Other
    keyword arguments go to NMDANeuron (J_max, u_quadrature, pooled_gate, n_syn,
    w_syn, LIF parameters...). run() returns free calcium sampled every
    sample_ms, and somatic spike times. Consecutive runs continue the trajectory.
    The bAP can be removed for a counterfactual twin with
    sim.params.A_f = sim.params.A_s = 0 (spikes are unchanged: NMDA carries no current).
    """
    def __init__(self, n=1, dt=0.1, Btot=None, sample_ms=1.0, **model_kwargs):
        if n > MAXN:
            raise ValueError('too many compartments')
        ref = NMDANeuron(n_dend=n, dt=dt, plasticity_mode='off', **model_kwargs)
        lif = ref.lif
        ratio = sample_ms / dt
        if not np.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError('sample_ms must be an integer multiple of dt')
        self.n, self.dt, self.stride = n, dt, int(round(ratio))
        self.model = ref
        p = Params(n=n, nsub=2, dt=dt, Cd=lif.Cd, Cs=lif.Cs, gL=lif.gL, EL=lif.EL, gc=lif.gc, Ea=lif.Ea,
                   decay=float(lif.decay), Vth=lif.Vth, Vreset=lif.Vreset, tref=lif.tref,
                   dec_u=float(ref._dec_u), a_u=ref.a_u, k_b=ref.k_b, tau_N=ref.tau_N,
                   dec_bf=float(ref._dec_bf), dec_bs=float(ref._dec_bs), A_f=ref.A_f, A_s=ref.A_s, mgc=ref.Mg / 3.57,
                   c0=ref.c0, tau_c=ref.tau_c, kon=ref.kon, koff=ref.koff, u_phi=float(ref._u_phi),
                   pooled=int(ref.pooled_gate),
                   EI=lif.EI, dec_I=float(lif.decay_I), gbar_I=lif.gbar_I)
        if ref.pooled_gate:
            p.dec_v, p.k_eff, p.a_um = float(ref._dec_v), float(ref.k_eff), float(ref._a_um)
        capacity = np.broadcast_to(np.asarray(ref.Btot if Btot is None else Btot, dtype=float), (n,))
        for i in range(n):
            p.Jmax[i] = float(ref._J_branch[i] if ref.pooled_gate else ref.J_max[i])
            p.Btot[i] = float(capacity[i])
            p.wscale[i] = ref.w_syn if ref.pooled_gate else 1.0
            p.n_syn[i] = float(ref.n_syn[i])
        self.params = p
        self.capacity = capacity.copy()
        x = State(Vs=lif.EL, t=0.0, refr=-np.inf, yf=0.0, ys=0.0, steps=0, gI=0.0)
        for i in range(n):
            bound = ref.kon * ref.c0 * capacity[i] / (ref.koff + ref.kon * ref.c0)
            x.g[i], x.V[i], x.u[i], x.s[i] = 0.0, lif.EL, 0.0, 0.0
            x.T[i], x.b[i], x.gbar[i] = ref.c0 + bound, bound, float(lif.gbar[i])
        self.state = x
        self.gbar0 = np.array([x.gbar[i] for i in range(n)])

    def run(self, w, max_spikes=20000, w_inh=None):
        """w: (n_steps, n) input per step (weights, or event counts in pooled mode);
        w_inh: optional (n_steps,) somatic inhibitory weight.
        Returns (calcium samples (n_samples, n), spike times)."""
        w = np.ascontiguousarray(w, dtype=float)
        if w_inh is not None:
            w_inh = np.ascontiguousarray(w_inh, dtype=float)
            if w_inh.shape != (len(w),):
                raise ValueError('w_inh must have one entry per step')
        samples = np.empty((len(w) // self.stride + 1, self.n))
        spike_times = np.empty(max_spikes)
        written = ctypes.c_long(0)
        count = _kernel.run_chunk(ctypes.byref(self.params), ctypes.byref(self.state), w.ctypes.data, len(w), self.stride,
                                  samples.ctypes.data, ctypes.byref(written), spike_times.ctypes.data, max_spikes,
                                  None if w_inh is None else w_inh.ctypes.data)
        if count > max_spikes:
            raise RuntimeError('spike buffer overflow')
        return samples[:written.value], spike_times[:count]

    def set_gbar(self, i, value):
        self.state.gbar[i] = float(value)

    def get_gbar(self):
        return np.array([self.state.gbar[i] for i in range(self.n)])


class PlasticSim:
    """Sim plus the Bayesian layer: compiled NMDANeuron in 'feedback' (or, with
    feedback=False, 'observe-only') mode.

    The kernel runs in chunks that end on inference ticks, so a gbar change made
    at a tick acts from the next step, exactly as in the Python model.
    run() returns (calcium, spikes, ticks): ticks holds the estimator's
    diagnostics plus gbar, one row per inference tick.
    """
    def __init__(self, n=1, dt=0.1, Btot=None, feedback=True, plasticity_config=None, **model_kwargs):
        self.config = plasticity_config or PlasticityConfig()
        self.sim = Sim(n=n, dt=dt, Btot=Btot, sample_ms=self.config.sample_ms, **model_kwargs)
        self.estimator = RollingCalciumEstimator(n, self.config)
        self.feedback = feedback
        self.chunk = int(round(self.config.update_ms / dt))

    def run(self, w, w_inh=None):
        w = np.asarray(w, dtype=float)
        calcium, spikes, ticks = [], [], []
        cfg, sim, k = self.config, self.sim, 0
        while k < len(w):
            n_steps = min(self.chunk - sim.state.steps % self.chunk, len(w) - k)
            c, s = sim.run(w[k:k + n_steps], w_inh=None if w_inh is None else w_inh[k:k + n_steps])
            calcium.append(c)
            spikes.extend(s)
            for row in c:
                d = self.estimator.sample(row)
                if d is None:
                    continue
                if self.feedback and d['time_ms'] > cfg.warmup_ms:
                    for i in range(sim.n):
                        sim.set_gbar(i, conductance_update(sim.state.gbar[i], sim.gbar0[i], d['drive'][i],
                                                           cfg.update_ms, cfg))
                ticks.append(dict(d, gbar=sim.get_gbar()))
            k += n_steps
        ticks = {key: np.array([d[key] for d in ticks]) for key in ticks[0]} if ticks else {}
        return np.concatenate(calcium) if calcium else np.empty((0, sim.n)), np.array(spikes), ticks


class Events:
    """Poisson input in continuous time, so every dt grid sees the same spikes.

    Each chunk draws from its own stream position regardless of dt or of model
    state, which is what makes dt comparisons and paired runs pathwise.
    """
    def __init__(self, entropy):
        self.rng = np.random.default_rng(entropy)

    def chunk(self, rates_hz, chunk_ms, dt):
        rates_hz = np.atleast_1d(np.asarray(rates_hz, dtype=float))
        steps = int(round(chunk_ms / dt))
        w = np.zeros((steps, len(rates_hz)))
        for i, rate in enumerate(rates_hz):
            count = self.rng.poisson(rate * chunk_ms / 1000.0)
            positions = self.rng.uniform(0.0, chunk_ms, count)
            np.add.at(w[:, i], np.minimum((positions / dt).astype(int), steps - 1), WEIGHT)
        return w
