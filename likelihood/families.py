"""Candidate likelihoods for the rolling calcium mean M (likelihood/DESIGN.md).

Each family is fitted by maximum likelihood to y = M - c0, the excess over rest,
except gamma_M, which is the Bayesian layer's current likelihood and models M itself.
Values of y below EPS_UM are left-censored: each contributes P(y < EPS_UM) under the
family, so windows holding no event are not placed on a log axis. Data are divided
by their mean before fitting, which makes every fit unit-free.

The Tweedie family (1 < p < 2) is compound Poisson-gamma: N ~ Poisson(lam) jumps,
each Gamma(alpha, gam). Its variance is phi mu^p and its skew is p CV, so p places
it between Poisson (p = 1, skew = CV) and gamma (p = 2, skew = 2 CV). The density is
the Dunn & Smyth (2005) series, summed around its largest term.
"""
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, gammainc, gammaln, logsumexp

EPS_UM = 1e-3          # excess below 1 nM counts as rest (physio NEAR_REST_UM)
FIT_CAP = 3000         # largest sample a fit uses
P_LIMIT = 7.0          # Tweedie p = 1 + expit(t), |t| <= P_LIMIT: 1.0009 <= p <= 1.9991
FAMILIES = ('normal', 'gamma_M', 'gamma', 'lognormal', 'invgauss', 'gengamma', 'tweedie')
N_PARAMS = dict(normal=2, gamma_M=2, gamma=2, lognormal=2, invgauss=2, gengamma=3, tweedie=3)
# skew / CV of the excess that each family forces (lognormal adds CV^2, gengamma is free)
SKEW_PER_CV = dict(normal=0.0, poisson=1.0, gamma=2.0, invgauss=3.0)


class Tweedie:
    """Tweedie distribution with 1 < p < 2, mean mu and variance phi mu^p."""

    def __init__(self, mu, phi, p):
        self.mu, self.phi, self.p = float(mu), float(phi), float(p)
        self.lam = mu ** (2 - p) / (phi * (2 - p))
        self.alpha = (2 - p) / (p - 1)
        self.gam = phi * (p - 1) * mu ** (p - 1)

    def _series(self, y):
        """log of each term of the density series that depends on the jump count j,
        on a window of j around the largest term for each y."""
        peak = np.minimum(y ** (2 - self.p) / (self.phi * (2 - self.p)), 20000.0)
        half = min(400, int(np.ceil(5 * np.sqrt(peak.max()) + 10)))   # exact while peak < 6000
        j = (np.maximum(1, np.floor(peak) - half).astype(int)[:, None] + np.arange(2 * half + 1)[None, :])
        table = np.arange(j.max() + 1, dtype=float)
        table = gammaln(table + 1) + gammaln(np.maximum(table, 1) * self.alpha)
        slope = np.log(self.lam) - self.alpha * np.log(self.gam) + self.alpha * np.log(y)
        return j * slope[:, None] - table[j]

    def logpdf(self, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        out = np.full(len(y), -np.inf)
        pos = y > 0
        if pos.any():
            yp = y[pos]
            out[pos] = logsumexp(self._series(yp), axis=1) - self.lam - np.log(yp) - yp / self.gam
        return out

    def _jumps(self):
        lo = max(1, int(self.lam - 12 * np.sqrt(self.lam) - 12))
        j = np.arange(lo, int(self.lam + 12 * np.sqrt(self.lam) + 12) + 1)
        return j, np.exp(stats.poisson.logpmf(j, self.lam))

    def cdf(self, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        j, weight = self._jumps()
        tail = gammainc(j[None, :] * self.alpha, np.maximum(y, 0)[:, None] / self.gam) @ weight
        return np.where(y < 0, 0.0, np.exp(-self.lam) + tail)

    def logcdf(self, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        j, weight = self._jumps()
        tail = gammainc(j[None, :] * self.alpha, np.maximum(y, 0)[:, None] / self.gam) @ weight
        with np.errstate(divide='ignore'):
            return np.where(y < 0, -np.inf, np.logaddexp(-self.lam, np.log(tail)))

    def ppf(self, q):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        sd = np.sqrt(self.phi * self.mu ** self.p)
        high = self.mu + 30 * sd
        out = np.empty(len(q))
        for i, qi in enumerate(q):
            if qi <= np.exp(-self.lam):
                out[i] = 0.0
            else:
                out[i] = optimize.brentq(lambda y: self.cdf(y)[0] - qi, 0.0, high, xtol=1e-12 * high)
        return out


def frozen(name, theta):
    """The family on the scaled variable z, from unconstrained parameters theta."""
    e = np.exp
    if name == 'normal':
        return stats.norm(theta[0], e(theta[1]))
    if name in ('gamma', 'gamma_M'):
        return stats.gamma(e(theta[0]), scale=e(theta[1]))
    if name == 'lognormal':
        return stats.lognorm(e(theta[1]), scale=e(theta[0]))
    if name == 'invgauss':
        mean, shape = e(theta[0]), e(theta[1])
        return stats.invgauss(mean / shape, scale=shape)
    if name == 'gengamma':
        return stats.gengamma(e(theta[0]), e(theta[1]), scale=e(theta[2]))
    if name == 'tweedie':
        return Tweedie(e(theta[0]), e(theta[1]), 1 + expit(np.clip(theta[2], -P_LIMIT, P_LIMIT)))
    raise ValueError(name)


def start(name, z, observed):
    m, v = z.mean(), z.var()
    if name == 'normal':
        return np.array([m, 0.5 * np.log(v)])
    if name in ('gamma', 'gamma_M', 'gengamma'):
        a = m * m / v
        theta = [np.log(a), np.log(m / a)]
        return np.array(theta[:1] + [0.0] + theta[1:]) if name == 'gengamma' else np.array(theta)
    if name == 'lognormal':
        logs = np.log(observed)
        return np.array([logs.mean(), np.log(logs.std() + 1e-3)])
    if name == 'invgauss':
        return np.array([np.log(m), np.log(m ** 3 / v)])
    if name == 'tweedie':
        p = float(np.clip(stats.skew(z) * m / np.sqrt(v), 1.05, 1.95))
        return np.array([np.log(m), np.log(v / m ** p), np.log((p - 1) / (2 - p))])
    raise ValueError(name)


@dataclass
class Fit:
    name: str
    theta: np.ndarray
    scale: float        # data were divided by this
    shift: float        # and this was added first (c0 for gamma_M, else 0)
    loglik: float       # log-likelihood of y, censored at EPS_UM
    n: int
    n_censored: int

    @property
    def dist(self):
        return frozen(self.name, self.theta)

    @property
    def aic(self):
        return 2 * N_PARAMS[self.name] - 2 * self.loglik

    def cdf(self, y):
        return self.dist.cdf((np.asarray(y) + self.shift) / self.scale)

    def ppf(self, q):
        return np.asarray(self.dist.ppf(q)) * self.scale - self.shift

    def pdf(self, y):
        return np.exp(self.dist.logpdf((np.asarray(y) + self.shift) / self.scale)) / self.scale

    def summary(self):
        out = dict(theta=self.theta.tolist(), scale=self.scale, shift=self.shift, loglik=self.loglik,
                   aic=self.aic, n=self.n, n_censored=self.n_censored)
        if self.name == 'tweedie':
            d = self.dist
            out.update(p=d.p, lam=d.lam, alpha=d.alpha, jump_mean=d.alpha * d.gam * self.scale)
        return out


def fit(name, y, c0):
    """Maximum likelihood fit of family `name` to the excess y, censored below EPS_UM."""
    y = np.asarray(y, dtype=float)
    shift = c0 if name == 'gamma_M' else 0.0
    x = np.maximum(y, 0.0) + shift
    scale = float(x.mean())
    z, cut = x / scale, (EPS_UM + shift) / scale
    censored = y < EPS_UM
    observed, n_cens = z[~censored], int(censored.sum())

    def nll(theta):
        d = frozen(name, theta)
        with np.errstate(all='ignore'):
            ll = d.logpdf(observed).sum() + (n_cens * float(np.squeeze(d.logcdf(cut))) if n_cens else 0.0)
        return -ll if np.isfinite(ll) else 1e300

    theta = start(name, z, observed)
    # Tweedie is an exponential dispersion model, so with nothing censored the MLE of
    # its mean is the sample mean (1 on the scaled axis) and only (phi, p) are searched.
    profile = name == 'tweedie' and n_cens == 0
    objective = (lambda rest: nll(np.concatenate(([0.0], rest)))) if profile else nll
    theta = theta[1:] if profile else theta
    for _ in range(2):      # a restart guards Nelder-Mead against early collapse
        res = optimize.minimize(objective, theta, method='Nelder-Mead',
                                options=dict(maxiter=1500, xatol=1e-4, fatol=1e-4, adaptive=True))
        if np.allclose(res.x, theta, atol=1e-3):
            break
        theta = res.x
    theta = np.concatenate(([0.0], res.x)) if profile else res.x
    loglik = -res.fun - len(observed) * np.log(scale)
    return Fit(name, theta, scale, shift, float(loglik), len(y), n_cens)


def fit_all(y, c0, rng=None):
    """Every family on the same sample, capped at FIT_CAP values."""
    y = np.asarray(y, dtype=float)
    if len(y) > FIT_CAP:
        rng = np.random.default_rng(0) if rng is None else rng
        y = y[np.sort(rng.choice(len(y), FIT_CAP, replace=False))]
    return {name: fit(name, y, c0) for name in FAMILIES}


def self_check(seed=3):
    """The Tweedie density and CDF against quadrature and Monte Carlo. Returns the worst errors."""
    from scipy.integrate import quad
    rng = np.random.default_rng(seed)
    worst = dict(mass=0.0, mean=0.0, var=0.0, ks=0.0)
    for mu, phi, p in [(1.0, 0.5, 1.2), (1.0, 0.05, 1.5), (2.0, 0.02, 1.05), (1.0, 0.8, 1.9), (0.3, 1.0, 1.4)]:
        d = Tweedie(mu, phi, p)
        f = lambda y, k: y ** k * np.exp(d.logpdf(y)[0])
        top = mu + 40 * np.sqrt(phi * mu ** p)
        points = [mu]
        mass = np.exp(-d.lam) + quad(f, 0, top, args=(0,), points=points, limit=400)[0]
        mean = quad(f, 0, top, args=(1,), points=points, limit=400)[0]
        var = quad(f, 0, top, args=(2,), points=points, limit=400)[0] - mean ** 2
        n = rng.poisson(d.lam, 200_000)
        sample = np.where(n > 0, rng.gamma(np.maximum(n, 1) * d.alpha, d.gam), 0.0)
        grid = np.quantile(sample, np.linspace(0.01, 0.99, 99))
        ks = np.abs(d.cdf(grid) - np.searchsorted(np.sort(sample), grid, side='right') / len(sample)).max()
        worst['mass'] = max(worst['mass'], abs(mass - 1))
        worst['mean'] = max(worst['mean'], abs(mean / mu - 1))
        worst['var'] = max(worst['var'], abs(var / (phi * mu ** p) - 1))
        worst['ks'] = max(worst['ks'], ks)
        for qi in (0.025, 0.975):
            yi = d.ppf([qi])[0]
            err = max(0.0, qi - d.cdf([0.0])[0]) if yi == 0 else abs(d.cdf([yi])[0] - qi)
            worst['ppf'] = max(worst.get('ppf', 0.0), err)
    return worst
