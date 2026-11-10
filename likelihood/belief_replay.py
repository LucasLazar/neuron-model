"""ExactGammaBelief replayed on a whole tick series at once (likelihood/DESIGN.md).

The belief's statistics (n, S, L) are discounted sums, so they follow linear
recursions and can be filtered in one pass. The percentile at each tick is then
the same mixture of regularized incomplete beta functions that
ExactGammaBelief.predictive evaluates, including its pruning of negligible CV
nodes. check() holds this to the class tick by tick.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.special import betainc, gammaln, logsumexp

from neuron.plasticity import ExactGammaBelief, PlasticityConfig


def percentiles(values, config=None, start=0, block=2000):
    """Pre-learning percentile U_k for every tick k >= start; ticks are update_ms apart."""
    c = config or PlasticityConfig()
    x = np.asarray(values, dtype=float)
    rho, weight = np.exp(-c.update_ms / c.slow_ms), c.update_ms / c.evidence_ms
    after = [lfilter([weight], [1.0, -rho], v) for v in (np.ones_like(x), x, np.log(x))]
    before = [rho * np.concatenate(([0.0], a[:-1])) for a in after]
    a = 1 / np.geomspace(c.cv_min, c.cv_max, c.cv_points) ** 2
    lg_a = gammaln(a)
    out = np.full(len(x), np.nan)
    for lo in range(max(start, 1), len(x), block):
        k = slice(lo, min(lo + block, len(x)))
        n, S, L = (b[k][:, None] for b in before)
        log_m = -n * lg_a + (a - 1) * L + gammaln(n * a) - n * a * np.log(S)
        w = np.exp(log_m - logsumexp(log_m, axis=1, keepdims=True))
        w = np.where(w > 1e-12 / a.size, w, 0.0)
        w /= w.sum(axis=1, keepdims=True)
        xk = x[k][:, None]
        out[k] = np.sum(w * betainc(a, n * a, xk / (xk + S)), axis=1)
    if start == 0:
        out[0] = 0.5     # the class scores nothing before its first observation
    return out


def check(n_ticks=3000, seed=5):
    """Largest |U| difference against ExactGammaBelief on a lognormal-ish series."""
    rng = np.random.default_rng(seed)
    x = np.exp(0.4 * rng.standard_normal(n_ticks)) * (1 + 0.5 * (np.arange(n_ticks) > n_ticks // 2))
    worst = 0.0
    for evidence in (2000.0, 8000.0):
        config = PlasticityConfig(evidence_ms=evidence)
        belief = ExactGammaBelief(config)
        belief.compute_interval = False
        exact = np.array([belief.observe(float(v))['percentile'] for v in x])
        worst = max(worst, float(np.abs(percentiles(x, config) - exact).max()))
    return worst
