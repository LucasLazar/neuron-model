"""Rolling calcium, discounted weighted gamma inference, and bounded AMPA.

Time is in ms, calcium in uM. Likelihood weighting is an approximation for
correlated observations, not an explicit temporal probability model.

The belief integrates the mean out exactly (ExactGammaBelief), leaving a grid
over the CV only. An earlier 2-D grid over (mean, CV) was dropped: at its default
resolution the belief about the mean was narrower than the grid spacing, so results
depended on where calcium fell between nodes. MODEL_ODES.md sections 4-7.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc, gammaln, logsumexp


@dataclass(frozen=True)
class PlasticityConfig:
    window_ms: float = 2000.0
    sample_ms: float = 1.0
    update_ms: float = 100.0
    slow_ms: float = 60000.0
    evidence_ms: float = 2000.0
    warmup_ms: float = 60000.0
    conductance_ms: float = 60000.0
    lower_factor: float = 0.25
    upper_factor: float = 4.0
    tail: float = 0.025
    mean_min: float = 0.01
    mean_max: float = 100.0
    cv_min: float = 0.001
    cv_max: float = 3.0
    cv_points: int = 121
    boundary_threshold: float = 0.01

    def __post_init__(self):
        for name in ('window_ms', 'sample_ms', 'update_ms', 'slow_ms',
                     'evidence_ms', 'conductance_ms', 'mean_min', 'cv_min'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('window_ms', 'update_ms'):
            ratio = getattr(self, name) / self.sample_ms
            if ratio < 1 or not np.isclose(ratio, round(ratio)):
                raise ValueError(f'{name} must be an integer multiple of sample_ms')
        if self.evidence_ms < self.update_ms:
            raise ValueError('evidence_ms must be >= update_ms')
        if not 0 < self.lower_factor <= 1 <= self.upper_factor < np.inf:
            raise ValueError('conductance factors must enclose one')
        if not 0 < self.tail < 0.5 or not np.isfinite(self.warmup_ms) or self.warmup_ms < 0:
            raise ValueError('invalid tail or warmup')
        if not self.mean_min < self.mean_max < np.inf or not self.cv_min < self.cv_max < np.inf:
            raise ValueError('invalid parameter limits')
        if int(self.cv_points) != self.cv_points or self.cv_points < 3:
            raise ValueError('cv_points must be an integer >= 3')
        if not 0 < self.boundary_threshold < 1:
            raise ValueError('invalid boundary threshold')

    def to_dict(self):
        return asdict(self)


def signed_drive(percentile, tail=0.025):
    return float(np.clip((percentile - (1 - tail)) / tail, 0, 1)
                 - np.clip((tail - percentile) / tail, 0, 1))


def conductance_update(g, initial, drive, dt_ms, config):
    if drive == 0:
        return float(g)
    target = initial * (config.upper_factor if drive > 0 else config.lower_factor)
    return float(target + (g - target) * np.exp(-abs(drive) * dt_ms / config.conductance_ms))


class ExactGammaBelief:
    """Gamma belief over (mean, CV) with the mean integrated out exactly.

    For fixed shape a, a prior uniform in log mean is uniform in log rate, which is
    conjugate. With discounted weights n, S = sum w x and L = sum w log x:

        rate | a, data ~ Gamma(n a, S)
        x / (x + S) | a, data ~ Beta(a, n a)
        log m(a) = -n lnGamma(a) + (a - 1) L + lnGamma(n a) - n a ln S

    Only the CV axis stays a grid. Forgetting toward the prior and tempering act on
    (n, S, L). Inference does not depend on the units of calcium. The prior's limits
    on the mean are not applied; the boundary diagnostic flags a mean outside them.
    Nothing is scored before the first observation: percentile 0.5, no interval.
    """
    compute_interval = True

    def __init__(self, config=None):
        self.config = config or PlasticityConfig()
        c = self.config
        cvs = np.geomspace(c.cv_min, c.cv_max, c.cv_points)
        self.shape = 1 / cvs ** 2
        self._gammaln_shape = gammaln(self.shape)
        self.boundary = np.zeros(c.cv_points, dtype=bool)
        self.boundary[[0, -1]] = True
        self.reset()

    def reset(self):
        self.n, self.S, self.L = 0.0, 0.0, 0.0
        self.effective_count = 0.0
        self.log_posterior = self._log_weights()

    def _log_weights(self):
        if self.n <= 0:
            return np.full(self.shape.size, -np.log(self.shape.size))
        a = self.shape
        log_m = -self.n * self._gammaln_shape + (a - 1) * self.L + gammaln(self.n * a) - self.n * a * np.log(self.S)
        return log_m - logsumexp(log_m)

    def _quantile(self, q, a, weights):
        y = brentq(lambda v: float(np.sum(weights * betainc(a, self.n * a, v))) - q, 0.0, 1.0, xtol=1e-14)
        return self.S * y / (1 - y) if y < 1 else np.inf  # early, heavy-tailed belief

    def predictive(self, observation=None):
        if self.n <= 0:
            return dict(predictive_mean=float(observation), predictive_low=np.nan, predictive_high=np.nan, percentile=0.5)
        weights = np.exp(self._log_weights())
        active = weights > 1e-12 / weights.size
        a, weights = self.shape[active], weights[active] / weights[active].sum()
        shape_total = self.n * a
        # E[mean | a] = a S / (n a - 1), finite once n a > 1.
        means = np.where(shape_total > 1, a * self.S / np.where(shape_total > 1, shape_total - 1, 1), self.S / self.n)
        result = dict(predictive_mean=float(np.sum(weights * means)))
        if self.compute_interval:
            result.update(predictive_low=self._quantile(self.config.tail, a, weights),
                          predictive_high=self._quantile(1 - self.config.tail, a, weights))
        if observation is not None:
            result['percentile'] = float(np.sum(weights * betainc(a, self.n * a, observation / (observation + self.S))))
        return result

    def observe(self, value, elapsed_ms=None):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('gamma observations must be finite and positive')
        c = self.config
        elapsed_ms = c.update_ms if elapsed_ms is None else elapsed_ms
        if not np.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError('elapsed_ms must be finite and positive')
        rho = np.exp(-elapsed_ms / c.slow_ms)
        self.n, self.S, self.L = rho * self.n, rho * self.S, rho * self.L
        result = self.predictive(value)
        result.update(rolling_mean=float(value),
                      departure=float(value - result['predictive_mean']),
                      drive=signed_drive(result['percentile'], c.tail),
                      predictive_effective_count=rho * self.effective_count)
        weight = elapsed_ms / c.evidence_ms
        self.n += weight
        self.S += weight * value
        self.L += weight * np.log(value)
        self.effective_count = rho * self.effective_count + weight
        self.log_posterior = self._log_weights()
        mass = float(np.exp(self.log_posterior[self.boundary]).sum())
        outside = not c.mean_min <= self.S / self.n <= c.mean_max
        result.update(effective_count=self.effective_count, boundary_mass=1.0 if outside else mass,
                      boundary_warning=outside or mass >= c.boundary_threshold)
        return result


class RollingCalciumEstimator:
    """One independent belief per compartment; sample calls are exactly sample_ms apart.

    Returns diagnostics only at complete-window update ticks; otherwise None.
    The ring buffer, clock and beliefs persist until reset().
    """
    def __init__(self, n_compartments=1, config=None):
        if int(n_compartments) != n_compartments or n_compartments < 1:
            raise ValueError('n_compartments must be a positive integer')
        self.config = config or PlasticityConfig()
        self.n_compartments = n_compartments
        self.beliefs = [ExactGammaBelief(self.config) for _ in range(n_compartments)]
        self.window_samples = round(self.config.window_ms / self.config.sample_ms)
        self.update_samples = round(self.config.update_ms / self.config.sample_ms)
        self.reset()

    def reset(self):
        self.buffer = np.zeros((self.window_samples, self.n_compartments))
        self.total = np.zeros(self.n_compartments)
        self.samples = 0
        self.last = None
        for belief in self.beliefs:
            belief.reset()

    def sample(self, calcium):
        calcium = np.asarray(calcium, dtype=float).reshape(-1)
        if calcium.shape != (self.n_compartments,) or not np.all(np.isfinite(calcium)) or np.any(calcium <= 0):
            raise ValueError('expected one finite positive calcium value per compartment')
        idx = self.samples % self.window_samples
        self.total += calcium - self.buffer[idx]
        self.buffer[idx] = calcium
        self.samples += 1
        if self.samples < self.window_samples or self.samples % self.update_samples:
            return None
        observations = self.total / self.window_samples
        values = [b.observe(float(x)) for b, x in zip(self.beliefs, observations)]
        self.last = {key: np.array([v[key] for v in values]) for key in values[0]}
        self.last['time_ms'] = self.samples * self.config.sample_ms
        return self.last
