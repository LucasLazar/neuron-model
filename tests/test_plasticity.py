"""The Bayesian layer (neuron/plasticity.py): estimator, belief and conductance rule.

The closed-form belief is checked against direct numerical integration of the
gamma likelihood over the mean, which shares none of its algebra.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import gammainc, gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron import ExactGammaBelief, PlasticityConfig, RollingCalciumEstimator, conductance_update, signed_drive  # noqa: E402

SMALL = PlasticityConfig(window_ms=20, update_ms=10, evidence_ms=20)


def numerical_posterior(x, weights, shapes, points=4001):
    """Posterior over shape a and predictive CDF, integrating log rate on a grid.

    Prior uniform in log mean, i.e. in log rate r. For each a the integrand is
    prod_j f(x_j | a, r)^w_j, a gamma density in r, so a grid of +-12 sd around its
    mode holds all the mass. Returns (normalised log weights over a, cdf(X) function).
    """
    n, S = weights.sum(), np.sum(weights * x)
    log_m, grids = [], []
    for a in shapes:
        mode = n * a / S
        sd = np.sqrt(n * a) / S
        lo, hi = max(mode - 12 * sd, mode * 1e-6), mode + 12 * sd
        r = np.linspace(lo, hi, points)
        log_f = n * (a * np.log(r) - gammaln(a)) + (a - 1) * np.sum(weights * np.log(x)) - r * S
        # measure d(log r) = dr / r
        log_integrand = log_f - np.log(r)
        step = r[1] - r[0]
        log_m.append(logsumexp(log_integrand) + np.log(step))
        grids.append((r, np.exp(log_integrand - logsumexp(log_integrand))))
    log_m = np.array(log_m)
    log_w = log_m - logsumexp(log_m)

    def cdf(X):
        return float(sum(np.exp(lw) * np.sum(p * gammainc(a, r * X))
                         for lw, a, (r, p) in zip(log_w, shapes, grids) if lw > -40))
    return log_w, cdf


class Belief(unittest.TestCase):
    def observed(self, count=40, seed=12, cv=0.15):
        belief = ExactGammaBelief(SMALL)
        x = np.random.default_rng(seed).gamma(1 / cv ** 2, 0.23 * cv ** 2, count)
        for value in x:
            belief.observe(float(value))
        rho = np.exp(-SMALL.update_ms / SMALL.slow_ms)
        omega = SMALL.update_ms / SMALL.evidence_ms
        weights = omega * rho ** np.arange(count - 1, -1, -1)
        return belief, x, weights

    def test_closed_form_matches_numerical_integration(self):
        belief, x, weights = self.observed()
        self.assertAlmostEqual(belief.n, weights.sum(), places=12)
        active = belief.log_posterior > -30
        shapes = belief.shape[active]
        log_w, cdf = numerical_posterior(x, weights, shapes)
        closed = belief.log_posterior[active] - logsumexp(belief.log_posterior[active])
        np.testing.assert_allclose(np.exp(closed), np.exp(log_w), atol=1e-7)
        for X in np.quantile(x, [0.02, 0.3, 0.7, 0.98]):
            self.assertAlmostEqual(belief.predictive(X)['percentile'], cdf(X), places=6)

    def test_score_before_learning(self):
        belief = ExactGammaBelief(SMALL)
        for value in (.21, .25, .2, .24, .23, .22):
            belief.observe(value)
        rho = np.exp(-SMALL.update_ms / SMALL.slow_ms)
        n, S, L = belief.n, belief.S, belief.L
        belief.n, belief.S, belief.L = rho * n, rho * S, rho * L
        expected = belief.predictive(.5)['percentile']
        belief.n, belief.S, belief.L = n, S, L
        self.assertEqual(belief.observe(.5)['percentile'], expected)

    def test_interval_and_percentile_agree(self):
        belief, _, _ = self.observed(60, 13)
        d = belief.predictive()
        self.assertAlmostEqual(belief.predictive(d['predictive_low'])['percentile'], .025, places=9)
        self.assertAlmostEqual(belief.predictive(d['predictive_high'])['percentile'], .975, places=9)

    def test_ignores_calcium_units(self):
        x = np.random.default_rng(11).gamma(400, .23 / 400, 300)
        a, b = ExactGammaBelief(), ExactGammaBelief()
        for value in x:
            self.assertAlmostEqual(a.observe(float(value))['percentile'], b.observe(float(value) * 1.0371)['percentile'], places=9)

    def test_stationary_coverage_is_near_nominal(self):
        """Independent gamma draws: the 95% predictive band holds about 95%."""
        belief = ExactGammaBelief(PlasticityConfig(evidence_ms=100))
        u = np.array([belief.observe(float(v))['percentile'] for v in np.random.default_rng(4).gamma(50, .004, 4000)])
        inside = np.mean((u[500:] > .025) & (u[500:] < .975))
        self.assertGreater(inside, .93)
        self.assertLess(inside, .97)

    def test_invalid_inputs_and_boundary(self):
        for kwargs in ({'window_ms': .5}, {'update_ms': 1.5}, {'cv_min': 0}, {'cv_points': 1}, {'evidence_ms': 10}):
            with self.assertRaises(ValueError):
                PlasticityConfig(**kwargs)
        b = ExactGammaBelief(SMALL)
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                b.observe(value)
        self.assertTrue(b.observe(10000, elapsed_ms=20000)['boundary_warning'])


class Estimator(unittest.TestCase):
    def test_causal_rolling_mean_and_tick_times(self):
        x = np.random.default_rng(4).gamma(10, .02, 105)
        est = RollingCalciumEstimator(config=SMALL)
        ticks = []
        for i, value in enumerate(x):
            d = est.sample([value])
            if d is not None:
                self.assertAlmostEqual(d['rolling_mean'][0], x[i - 19:i + 1].mean(), places=13)
                ticks.append(i)
        self.assertEqual(ticks, list(range(19, 105, 10)))

    def test_compartments_are_independent(self):
        e = RollingCalciumEstimator(2, SMALL)
        for _ in range(100):
            e.sample([.2, .8])
        self.assertLess(e.last['predictive_mean'][0], e.last['predictive_mean'][1])


class Conductance(unittest.TestCase):
    def test_drive_mapping(self):
        for u, d in ((0, -1), (.0125, -.5), (.025, 0), (.5, 0), (.975, 0), (.9875, .5), (1, 1)):
            self.assertAlmostEqual(signed_drive(u), d, places=12)

    def test_bounds_and_exact_interval_composition(self):
        for drive in (-1, -.2, 0, .2, 1):
            g = 1.0
            for _ in range(500):
                g = conductance_update(g, 1, drive, 10000, SMALL)
                self.assertTrue(.25 <= g <= 4)
        half = conductance_update(1, 1, .7, 50, SMALL)
        self.assertAlmostEqual(conductance_update(1, 1, .7, 100, SMALL), conductance_update(half, 1, .7, 50, SMALL))


if __name__ == '__main__':
    unittest.main()
