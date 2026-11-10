"""Behaviour of the NMDA gate, calcium and calibration in the Python model.

Run: python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from neuron import J_MAX_SPINE, NMDANeuron, PlasticityConfig, calcium_step, mg_block, spike_train  # noqa: E402

SMALL = PlasticityConfig(window_ms=20, update_ms=10, evidence_ms=20, warmup_ms=25, conductance_ms=100)


def neuron(**kw):
    kw.setdefault('n_dend', 1)
    kw.setdefault('plasticity_mode', 'off')
    return NMDANeuron(**kw)


class Gate(unittest.TestCase):
    def test_silent_at_rest(self):
        n = neuron(n_dend=2)
        r = n.run(300.0)
        self.assertTrue(np.all(r['J'] == 0.0))
        np.testing.assert_allclose(r['c'], n.c0, atol=1e-12)
        np.testing.assert_allclose(r['Vs'], -70.0)

    def test_and_gate_bap_alone_gives_zero_influx(self):
        n = neuron(n_dend=2)
        r = n.run(1000.0, force_post=np.arange(50.0, 1000.0, 50.0))
        self.assertTrue(np.all(r['J'] == 0.0))
        np.testing.assert_allclose(r['c'], n.c0, atol=1e-12)

    def test_single_event_is_local_and_blocked(self):
        n = neuron(n_dend=4, Vth=1e9)
        r = n.run(500.0, spike_train(50.0, 5000, 0.1, comp=0, n_dend=4, weight=12.0))
        self.assertAlmostEqual(r['s'][:, 0].max(), 0.138, delta=0.002)
        np.testing.assert_allclose(r['c'][:, 1:], n.c0, atol=1e-12)
        self.assertLess(mg_block(r['V'][:, 0].max(), n.Mg), 0.04)

    def test_bap_multiplies_influx_of_a_paired_event(self):
        w = spike_train(50.0, 3000, 0.1, comp=0, n_dend=1, weight=12.0)
        alone = neuron(Vth=1e9).run(300.0, w)
        paired = neuron(Vth=1e9).run(300.0, w, force_post=[52.0])
        ratio = paired['J'].sum() / alone['J'].sum()
        self.assertAlmostEqual(ratio, 4.13, delta=0.1)   # physio/FINDINGS.md, offset +2 ms

    def test_quadrature_factor(self):
        """One event, no buffer: calcium is linear in J, so the integral scales by phi."""
        w = spike_train(10.0, 20000, 0.1, comp=0, n_dend=1, weight=12.0)
        excess = {}
        for scheme in ('exact', 'frozen'):
            r = neuron(Btot=0.0, Vth=1e9, u_quadrature=scheme).run(2000.0, w)
            excess[scheme] = np.sum(r['c'][:, 0] - 0.1)
        self.assertAlmostEqual(excess['exact'] / excess['frozen'], (1 - np.exp(-0.2)) / 0.2, delta=0.01)

    def test_pooled_isolated_event_matches_single_synapse(self):
        w = np.zeros((3000, 1))
        w[10, 0] = 1.0
        pooled = neuron(Btot=0.0, pooled_gate=True, n_syn=1000.0).run(300.0, w)
        single = neuron(Btot=0.0).run(300.0, w * 12.0)
        self.assertAlmostEqual(pooled['s'][:, 0].max() / single['s'][:, 0].max(), 1.0, delta=0.01)


class Calibration(unittest.TestCase):
    def test_one_event_peaks_half_micromolar_at_kappa_20(self):
        n = neuron(Btot=44.0, Vth=1e9)
        self.assertAlmostEqual(n.kappa()[0], 19.95, delta=0.01)
        r = n.run(1000.0, spike_train(10.0, 10000, 0.1, comp=0, n_dend=1, weight=12.0))
        self.assertAlmostEqual(r['c'][:, 0].max() - n.c0, 0.5, places=6)

    def test_default_matches_stored_calibration(self):
        path = ROOT / 'physio' / 'results' / 'retune.json'
        if not path.exists():
            self.skipTest('run physio/p0_retune.py first')
        self.assertEqual(json.loads(path.read_text())['J_max'], J_MAX_SPINE)


class Calcium(unittest.TestCase):
    @staticmethod
    def fine(J, B, h=0.002, dt=0.1, c0=0.1, tau_c=12.0, kon=0.5, koff=1.0):
        """Explicit RK4 on (c, b) at a tiny step: stable and independent of the splitting."""
        c, b = c0, kon * c0 * B / (koff + kon * c0)
        out = []
        f = lambda c, b, j: (j - (c - c0) / tau_c - (kon * c * (B - b) - koff * b), kon * c * (B - b) - koff * b)
        for j in J:
            for _ in range(int(round(dt / h))):
                k1 = f(c, b, j)
                k2 = f(c + h / 2 * k1[0], b + h / 2 * k1[1], j)
                k3 = f(c + h / 2 * k2[0], b + h / 2 * k2[1], j)
                k4 = f(c + h * k3[0], b + h * k3[1], j)
                c += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                b += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            out.append(c)
        return np.array(out)

    def test_strang_step_against_fine_rk4(self):
        """Sharp square influx is a deliberate stiff stress test. At Btot 300 the
        default splitting has about 5% peak error, under 2% on the mean."""
        J = np.tile(np.r_[np.zeros(100), np.full(100, 0.2)], 5)
        for B in (0.0, 15.0, 100.0, 300.0):
            total, bound = 0.1 + 0.05 * B / 1.05, 0.05 * B / 1.05
            c = []
            for j in J:
                total, bound = calcium_step(total, bound, j, B, 0.1, 0.1, 12.0, 0.5, 1.0)
                c.append(total - bound)
                self.assertTrue(0 <= bound <= B + 1e-12)
            c, ref = np.array(c), self.fine(J, B)
            self.assertLess(np.max(np.abs(c - ref)) / np.ptp(ref), 0.05)
            self.assertLess(abs(c.mean() - ref.mean()) / ref.mean(), 0.02)

    def test_binding_conserves_total_calcium(self):
        """Without influx or extrusion, binding only moves calcium between pools,
        so adding unbound buffer adds no calcium."""
        for B in (15.0, 300.0):
            total, bound = 0.6, 0.4
            for _ in range(1000):
                total, bound = calcium_step(total, bound, 0.0, B, 0.1, 0.1, 1e15, 0.5, 1.0)
            self.assertAlmostEqual(total, 0.6, places=9)
            c = total - bound
            self.assertAlmostEqual(0.5 * c * (B - bound), 1.0 * bound, places=9)   # binding equilibrium

    def test_buffer_leaves_mean_calcium_unchanged(self):
        """Conservation: in steady state mean excess = tau_c x mean influx, any Btot."""
        w = np.random.default_rng(3).poisson(16 * 1e-4, (60000, 1)) * 12.0
        means = []
        for B in (0.0, 44.0):
            r = neuron(Btot=B, Vth=1e9).run(6000.0, w, record_ms=1)
            means.append(r['c'][2000:, 0].mean())
        self.assertLess(abs(means[0] / means[1] - 1), 0.05)


class Runs(unittest.TestCase):
    def test_split_run_and_reset(self):
        args = dict(n_dend=2, Btot=100.0, plasticity_mode='feedback', plasticity_config=SMALL)
        one, split = NMDANeuron(**args), NMDANeuron(**args)
        rng = np.random.default_rng(3)
        w = np.concatenate([rng.poisson(.01, (500, 2)), rng.poisson(.1, (500, 2))]) * 12.   # a step, so gbar moves
        r = one.run(100, w, record_ms=1)
        a = split.run(43, w[:430], record_ms=1)
        b = split.run(57, w[430:], record_ms=1)
        for key in ('c', 'b', 'V', 'g', 't'):
            np.testing.assert_array_equal(r[key], np.concatenate([a[key], b[key]]))
        for key in r['plasticity']:
            np.testing.assert_array_equal(r['plasticity'][key], np.concatenate([a['plasticity'][key], b['plasticity'][key]]))
        self.assertTrue(np.all(r['plasticity']['gbar'][r['plasticity']['time_ms'] <= 25] == 1))
        self.assertFalse(np.all(r['plasticity']['gbar'] == 1))
        split.reset_state()
        np.testing.assert_array_equal(split.lif.gbar, [1, 1])
        np.testing.assert_array_equal(r['plasticity']['gbar'], split.run(100, w, record_ms=1)['plasticity']['gbar'])

    def test_observe_only_and_off_agree(self):
        w = np.random.default_rng(5).poisson(.04, (1000, 1)) * 12.
        a = NMDANeuron(n_dend=1, plasticity_mode='observe-only', plasticity_config=SMALL).run(100, w)
        b = NMDANeuron(n_dend=1, plasticity_mode='off', plasticity_config=SMALL).run(100, w)
        np.testing.assert_array_equal(a['c'], b['c'])
        self.assertTrue(np.all(a['plasticity']['gbar'] == 1))

    def test_feedback_stays_in_bounds(self):
        cfg = replace(SMALL, conductance_ms=20)
        n = NMDANeuron(n_dend=2, Btot=15, plasticity_mode='feedback', plasticity_config=cfg)
        r = n.run(300, np.random.default_rng(14).poisson(.05, (3000, 2)) * 12.)
        self.assertTrue(np.all(r['plasticity']['gbar'] >= .25) and np.all(r['plasticity']['gbar'] <= 4))

    def test_invalid_settings(self):
        for kw in (dict(plasticity_mode='legacy'), dict(u_quadrature='rk4'), dict(Btot=-1.0),
                   dict(pooled_gate=True, u_quadrature='frozen'), dict(dt=0.3)):
            with self.assertRaises(ValueError):
                NMDANeuron(**kw)


if __name__ == '__main__':
    unittest.main()
