"""Shared setup for the branch (option A) study: branch/DESIGN.md.

A compartment is a dendritic branch carrying n_syn synapses with independent
clefts (NMDANeuron(pooled_gate=True)). Free calcium is the branch average of
spine calcium, so the per-spine calibration from physio/ carries over: J_max is
physio's retuned value and the model divides it by n_syn. Input is pooled event
counts with a shared Ornstein-Uhlenbeck modulation of rate (correlated presynaptic
activity), plus Poisson inhibition at the soma.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'physio'))
sys.dont_write_bytecode = True

from physio_core import C0, NMDANeuron, Sim, burn_ms as _burn_ms, mean_ci, retuned_j_max, rolling_mean, save_json, auc  # noqa: E402,F401

RESULTS = HERE / 'results'
PLOTS = HERE / 'plots'

N_BRANCH = 3
N_SYN = 1000.0          # synapses per branch
W_SYN = 1.0             # AMPA per event, x gbar 1 nS: 0.17 mV at the soma
W_INH = 2.0             # inhibitory event, nS
OU_TAU = 20.0           # ms, timescale of shared rate fluctuations
OU_SIGMA = 0.5          # relative sd of the shared excitatory rate
BASE_RATE = 2.5         # Hz per synapse
BASE_INH = 1000.0       # Hz, pooled somatic inhibition
BUFFER = 44.0           # uM, kappa ~ 20 at rest
DT = 0.1
CHUNK_MS = 100.0


def make_sim(buffer=BUFFER, n=N_BRANCH, n_syn=N_SYN, bap=True, dt=DT, sample_ms=1.0):
    sim = Sim(n=n, dt=dt, Btot=buffer, sample_ms=sample_ms, J_max=retuned_j_max(), u_quadrature='exact',
              pooled_gate=True, n_syn=n_syn, w_syn=W_SYN)
    if not bap:
        sim.params.A_f = sim.params.A_s = 0.0
    return sim


class Background:
    """Pooled event counts per branch and somatic inhibition, chunk by chunk.

    rates: per-synapse rate per branch (Hz). The excitatory rate of every branch is
    multiplied by the same OU factor (1 + sigma * x), x unit-variance with time
    constant tau, clipped at zero. Draws depend only on the entropy, so paired arms
    (bAP on and off, buffers) see identical input.
    """
    def __init__(self, entropy, sigma=OU_SIGMA, tau=OU_TAU, dt=DT):
        self.rng = np.random.default_rng(entropy)
        self.sigma, self.dt = sigma, dt
        self.a = np.exp(-dt / tau)
        self.b = np.sqrt(1 - self.a ** 2)
        self.state = np.array([self.rng.standard_normal() * self.a])

    def chunk(self, rates, inh_rate, chunk_ms=CHUNK_MS, n_syn=N_SYN):
        steps = int(round(chunk_ms / self.dt))
        z = self.rng.standard_normal(steps) * self.b
        x, self.state = lfilter([1.0], [1.0, -self.a], z, zi=self.state)
        factor = np.clip(1.0 + self.sigma * x, 0.0, None)
        self.last_factor = factor
        lam = np.outer(factor, np.asarray(rates, dtype=float) * n_syn) * self.dt / 1000.0
        counts = self.rng.poisson(lam).astype(float)
        inh = self.rng.poisson(inh_rate * self.dt / 1000.0, steps) * W_INH
        return counts, inh


def burn_ms(buffer=BUFFER):
    return _burn_ms(buffer)
