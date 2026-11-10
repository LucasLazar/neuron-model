"""Shared setup for the physiological-rate study (physio/DESIGN.md).

Per-synapse reading: one Poisson stream is one presynaptic neuron. The model is
NMDANeuron run through the compiled kernel (neuron/fast.py), with
u_quadrature="exact" (MODEL_ODES.md sec 9). J_max is fixed by p0_retune.py and
stored in results/retune.json; the model default J_MAX_SPINE is that value
(tests/test_model.py checks they agree).
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from neuron import NMDANeuron  # noqa: E402
from neuron.fast import Events, Sim, WEIGHT  # noqa: E402,F401

RESULTS = HERE / 'results'
PLOTS = HERE / 'plots'
RATES = [1, 2, 4, 8, 16, 32, 64]
BUFFERS = [0.0, 15.0, 44.0, 100.0, 300.0]
REFERENCE_BUFFER = 44.0      # kappa ~ 20 at rest, the spine estimate
TARGET_PEAK_UM = 0.5         # peak free-calcium rise for one isolated event
J_MAX_UNTUNED = 0.5         # the value before this study (30-500 Hz domain); kept as a control
SCHEME = 'exact'             # NMDANeuron u_quadrature; 'frozen' is the historical step
WINDOW_MS = 2000             # the Bayesian layer's rolling window
SAMPLE_MS = 1.0
NEAR_REST_UM = 1e-3
REF = NMDANeuron(n_dend=1, plasticity_mode='off')   # read for constants only
C0 = REF.c0


def retuned_j_max():
    return json.loads((RESULTS / 'retune.json').read_text())['J_max']


def tunings():
    return {'retuned': retuned_j_max(), 'untuned': J_MAX_UNTUNED}


def kappa(buffer):
    return REF.kappa(buffer)


def burn_ms(buffer):
    """15 tau_eff at resting kappa (at least 3 s), rounded up to whole 100 ms chunks."""
    return float(np.ceil(max(3000.0, 15.0 * REF.tau_eff(buffer)) / 100.0) * 100.0)


def state_trace(J, buffer, event_times, duration_ms, dt=0.1, scheme=SCHEME):
    """Deterministic run from rest, reading the full state after every step.

    Returns dict of arrays on the dt grid: t, V, s, c, b. For short traces only.
    """
    sim = Sim(n=1, dt=dt, Btot=buffer, sample_ms=dt, u_quadrature=scheme, J_max=J)
    steps = int(round(duration_ms / dt))
    w = np.zeros((steps, 1))
    for time_ms in event_times:
        w[int(round(time_ms / dt)), 0] += WEIGHT
    out = {k: np.empty(steps) for k in ('V', 's', 'c', 'b')}
    for k in range(steps):
        sim.run(w[k:k + 1])
        x = sim.state
        out['V'][k], out['s'][k] = x.V[0], x.s[0]
        out['c'][k], out['b'][k] = x.T[0] - x.b[0], x.b[0]
    out['t'] = (np.arange(steps) + 1) * dt
    return out


def calcium_trace(J, buffer, event_times, duration_ms, dt=0.1, scheme=SCHEME):
    """Free calcium on the dt grid for a fixed event list, from rest."""
    sim = Sim(n=1, dt=dt, Btot=buffer, sample_ms=dt, u_quadrature=scheme, J_max=J)
    steps = int(round(duration_ms / dt))
    w = np.zeros((steps, 1))
    for time_ms in event_times:
        w[int(round(time_ms / dt)), 0] += WEIGHT
    c, spikes = sim.run(w)
    return c[:, 0], spikes


def single_peak(J, buffer, dt=0.1, scheme=SCHEME):
    """Peak free-calcium rise for one event at 10 ms, from rest."""
    c, _ = calcium_trace(J, buffer, [10.0], 2500.0, dt, scheme)
    return float(c.max() - C0)


def stationary_run(J, buffer, rate, seed, run_s, dt=0.1, scheme=SCHEME, tag=91, occupancy_every_ms=100.0):
    """Burn in, then run run_s seconds of Poisson input.

    The input stream depends only on (tag, rate, seed), so buffers, tunings and dt
    levels sharing those see the same spike train. Returns 1 ms calcium samples,
    occupancy b/Btot every 100 ms, and the somatic spike count.
    """
    sim = Sim(n=1, dt=dt, Btot=buffer, u_quadrature=scheme, J_max=J)
    entropy = [tag, int(rate * 1000), seed]
    events, burn_events = Events(entropy), Events(entropy + [1])
    for _ in range(int(round(burn_ms(buffer) / 100.0))):
        sim.run(burn_events.chunk(float(rate), 100.0, dt))
    chunks = int(round(run_s * 1000 / occupancy_every_ms))
    calcium, occupancy, spikes = [], np.empty(chunks), 0
    for k in range(chunks):
        c, s = sim.run(events.chunk(float(rate), occupancy_every_ms, dt))
        calcium.append(c[:, 0])
        spikes += len(s)
        occupancy[k] = sim.state.b[0] / buffer if buffer > 0 else np.nan
    return np.concatenate(calcium), occupancy, spikes


def boxcar(x, n):
    """Causal mean of the last n samples; NaN before the first complete window."""
    out = np.full(len(x), np.nan)
    if n <= len(x):
        cs = np.concatenate(([0.0], np.cumsum(x)))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def rolling_mean(calcium, window_ms=WINDOW_MS, sample_ms=SAMPLE_MS):
    """The 2 s mean M the Bayesian layer reads, complete windows only."""
    n = int(round(window_ms / sample_ms))
    return boxcar(calcium, n)[n - 1:]


def tau_int(segs, max_lag=200_000):
    """Integrated autocorrelation time in samples, averaged within realisations and
    truncated at the first ACF crossing of 0.05."""
    acs = []
    for x in segs:
        y = x[np.isfinite(x)]
        y = y - y.mean()
        n = len(y)
        f = np.fft.rfft(y, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:min(max_lag, n)]
        acs.append(ac / ac[0])
    L = min(len(a) for a in acs)
    ac = np.mean([a[:L] for a in acs], axis=0)
    stop = np.argmax(ac < 0.05)
    stop = L if stop == 0 else stop
    return 1.0 + 2.0 * ac[1:stop].sum(), ac


def auc(high, low):
    """P(high > low) + 0.5 P(tie), by ranks."""
    from scipy.stats import rankdata
    high, low = np.asarray(high), np.asarray(low)
    ranks = rankdata(np.concatenate([high, low]))
    u = ranks[:len(high)].sum() - len(high) * (len(high) + 1) / 2
    return float(u / (len(high) * len(low)))


def mean_ci(x):
    from scipy import stats
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan
    return float(x.mean()), float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=1))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
