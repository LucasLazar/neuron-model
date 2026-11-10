"""Holds the compiled kernel (neuron/fast.py) to the Python model at rtol 1e-9.

Every study runs on the kernel, so this is what makes their results results of
the model in MODEL_ODES.md. Covers the per-synapse reading at every buffer and
several dt, both u^2 steps, several compartments, the pooled branch gate with
somatic inhibition and a firing soma, the bAP-off twin, and closed-loop
plasticity (PlasticSim).

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from neuron import NMDANeuron, PlasticityConfig  # noqa: E402
from neuron.fast import WEIGHT, Events, PlasticSim, Sim  # noqa: E402

SMALL = PlasticityConfig(window_ms=20, update_ms=10, evidence_ms=20, warmup_ms=25, conductance_ms=200)


def pooled_input(seed, steps, rate_per_branch_hz, inh_hz, n=3):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rate_per_branch_hz * 1e-4, (steps, n)).astype(float)
    return counts, rng.poisson(inh_hz * 1e-4, steps) * 2.0


class PerSynapse(unittest.TestCase):
    def compare(self, n, dt, buffer, rate, duration_ms=2000.0, **kw):
        rng = np.random.default_rng([n, int(buffer), int(rate)])
        steps = int(round(duration_ms / dt))
        w = rng.poisson(rate * dt / 1000, (steps, n)) * WEIGHT
        result = NMDANeuron(n_dend=n, dt=dt, Btot=buffer, plasticity_mode='off', **kw).run(duration_ms, w, record_ms=1)
        calcium, spikes = Sim(n=n, dt=dt, Btot=buffer, **kw).run(w)
        np.testing.assert_allclose(calcium, result['c'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(spikes, result['spikes'], atol=1e-8)
        return len(spikes), float(result['c'].max())

    def test_all_buffers_physiological_rates(self):
        for buffer in (0., 15., 44., 300.):
            for rate in (4, 32):
                _, peak = self.compare(1, .1, buffer, rate)
                self.assertGreater(peak, 0.2, 'calcium should leave rest')

    def test_firing_soma_several_compartments(self):
        spikes, _ = self.compare(4, .1, 15., 700)
        self.assertGreater(spikes, 0)

    def test_other_steps_and_frozen_quadrature(self):
        for dt in (.2, .05, .025):
            self.compare(1, dt, 44., 16, duration_ms=1000.0)
        self.compare(1, .1, 15., 300, u_quadrature='frozen', J_max=0.5)


class Pooled(unittest.TestCase):
    kw = dict(pooled_gate=True, n_syn=1000.0, w_syn=1.0)

    def compare(self, buffer, rate, inh, duration_ms=1000.0):
        counts, w_inh = pooled_input([int(buffer), int(rate)], int(duration_ms * 10), rate, inh)
        result = NMDANeuron(n_dend=3, Btot=buffer, plasticity_mode='off', **self.kw).run(duration_ms, counts, record_ms=1,
                                                                                       inh_in=w_inh)
        calcium, spikes = Sim(n=3, Btot=buffer, **self.kw).run(counts, w_inh=w_inh)
        np.testing.assert_allclose(calcium, result['c'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(spikes, result['spikes'], atol=1e-8)
        return len(spikes)

    def test_firing_soma_with_inhibition(self):
        self.assertGreater(self.compare(44.0, 4500.0, 300.0), 5, 'test must exercise bAPs')

    def test_unbuffered_and_large_buffer(self):
        self.compare(0.0, 3000.0, 0.0, 600.0)
        self.compare(300.0, 3000.0, 2000.0, 600.0)

    def test_no_inhibition_is_bit_identical_to_zero_inhibition(self):
        counts, _ = pooled_input(8, 3000, 3000.0, 0.0)
        a, sa = Sim(n=3, **self.kw).run(counts)
        b, sb = Sim(n=3, **self.kw).run(counts, w_inh=np.zeros(len(counts)))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(sa, sb)

    def test_bap_off_twin_changes_calcium_not_spikes(self):
        counts, w_inh = pooled_input(9, 20000, 4500.0, 300.0)
        on = Sim(n=3, **self.kw)
        off = Sim(n=3, **self.kw)
        off.params.A_f = off.params.A_s = 0.0
        c_on, s_on = on.run(counts, w_inh=w_inh)
        c_off, s_off = off.run(counts, w_inh=w_inh)
        np.testing.assert_array_equal(s_on, s_off)
        self.assertGreater(len(s_on), 5)
        self.assertTrue(np.all(c_on >= c_off - 1e-12))
        self.assertGreater(c_on.mean(), c_off.mean())


class ClosedLoop(unittest.TestCase):
    def compare(self, n, w, chunks, w_inh=None, **kw):
        result = NMDANeuron(n_dend=n, plasticity_mode='feedback', plasticity_config=SMALL, **kw).run(
            len(w) / 10, w, record_ms=1, inh_in=w_inh)
        fast = PlasticSim(n=n, plasticity_config=SMALL, **kw)
        calcium, spikes, ticks = [], [], []
        for a, b in zip(chunks[:-1], chunks[1:]):    # uneven calls: ticks must still land on the right step
            c, s, t = fast.run(w[a:b], None if w_inh is None else w_inh[a:b])
            calcium.append(c)
            spikes.extend(s)
            ticks.append(t)
        np.testing.assert_allclose(np.concatenate(calcium), result['c'], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(spikes, result['spikes'], atol=1e-8)
        for key in ('gbar', 'drive', 'percentile', 'rolling_mean'):
            np.testing.assert_allclose(np.concatenate([t[key] for t in ticks if t]), result['plasticity'][key],
                                       rtol=1e-8, err_msg=key)
        self.assertFalse(np.all(result['plasticity']['gbar'] == 1.0), 'test must exercise feedback')

    # Inputs step up halfway, so the belief sees a change and gbar moves.
    def test_per_synapse_one_and_four(self):
        rng = np.random.default_rng(23)
        w = np.concatenate([rng.poisson(.002, (2000, 1)), rng.poisson(.03, (2000, 1))]) * WEIGHT
        self.compare(1, w, [0, 1234, 2000, 4000], Btot=15.0)
        w = np.concatenate([rng.poisson(.005, (1500, 4)), rng.poisson(.06, (1500, 4))]) * WEIGHT
        self.compare(4, w, [0, 3000], Btot=15.0)

    def test_pooled_with_inhibition(self):
        low, inh_low = pooled_input(24, 2000, 2000.0, 300.0)
        high, inh_high = pooled_input(25, 2000, 5000.0, 300.0)
        self.compare(3, np.concatenate([low, high]), [0, 777, 4000], w_inh=np.concatenate([inh_low, inh_high]),
                     pooled_gate=True, n_syn=1000.0, w_syn=1.0)

    def test_observe_only_leaves_gbar(self):
        _, _, ticks = PlasticSim(n=1, feedback=False, plasticity_config=SMALL).run(
            np.random.default_rng(1).poisson(.004, (3000, 1)) * WEIGHT)
        self.assertTrue(np.all(ticks['gbar'] == 1.0))


class Input(unittest.TestCase):
    def test_events_identical_across_grids(self):
        coarse = Events([1, 2]).chunk(300., 100., .1)
        fine = Events([1, 2]).chunk(300., 100., .025)
        self.assertEqual(coarse.sum(), fine.sum())
        np.testing.assert_allclose(coarse[:, 0], fine[:, 0].reshape(-1, 4).sum(axis=1))


if __name__ == '__main__':
    unittest.main()
