"""Analytic checks of the dendritic LIF (neuron/lif.py).

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron.lif import DendriticLIF, attenuation, steady_state, steady_state_general  # noqa: E402

DT = 0.1


def clamp_one(n, gd, steps):
    """Drive compartment 0 alone with a clamped conductance, by hand."""
    N = n.n_dend
    for _ in range(steps):
        n.g = np.zeros(N)
        n.g[0] = gd
        I_d = n.gL * (n.EL - n.V) + n.g * (n.Ea - n.V) + n.gc * (n.Vs - n.V)
        I_s = n.gL * (n.EL - n.Vs) + np.sum(n.gc * (n.V - n.Vs))
        n.V = n.V + DT * I_d / n.Cd
        n.Vs = n.Vs + DT * I_s / n.Cs


class LIF(unittest.TestCase):
    def test_silent_at_rest(self):
        r = DendriticLIF(dt=DT).run(100.0)
        self.assertTrue(np.allclose(r['V'], -70.0) and np.allclose(r['Vs'], -70.0))

    def test_epsc_decays_exactly(self):
        w = np.zeros((3000, 1))
        w[0, 0] = 1.0
        r = DendriticLIF(dt=DT).run(300.0, w)
        predicted = np.exp(-(r['t'] - r['t'][0]) / 3.0)
        self.assertAlmostEqual(r['g'][0, 0], 1.0, places=12)
        self.assertLess(np.max(np.abs(r['g'][:, 0] - predicted)), 1e-12)
        self.assertLess(r['V'].max(), -69.0)
        self.assertLess(abs(r['V'][-1, 0] + 70.0), 1e-3)

    def test_clamped_steady_state(self):
        for gd in (10.0, 20.0, 30.0):
            r = DendriticLIF(dt=DT, Vth=1e9).run(300.0, g_clamp=gd)
            Vd, Vs = steady_state(gd=gd)
            self.assertLess(abs(r['V'][-1, 0] - Vd), 1e-3)
            self.assertLess(abs(r['Vs'][-1] - Vs), 1e-3)

    def test_threshold_in_conductance(self):
        self.assertEqual(len(DendriticLIF(dt=DT).run(300.0, g_clamp=10.0)['spikes']), 0)
        self.assertGreater(len(DendriticLIF(dt=DT).run(300.0, g_clamp=30.0)['spikes']), 0)

    def test_uncoupled_soma_stays_at_rest(self):
        r = DendriticLIF(dt=DT, gc=0.0).run(200.0, g_clamp=50.0)
        self.assertTrue(np.allclose(r['Vs'], -70.0))

    def test_attenuation_one_compartment(self):
        for gc in (1.0, 10.0, 100.0):
            r = DendriticLIF(dt=DT, gc=gc, Vth=1e9).run(400.0, g_clamp=10.0)
            k = (r['Vs'][-1] + 70.0) / (r['V'][-1, 0] + 70.0)
            self.assertLess(abs(k - gc / (10.0 + gc)), 1e-3)

    def test_attenuation_many_compartments(self):
        for N in (1, 4, 10):
            r = DendriticLIF(n_dend=N, dt=DT, Vth=1e9).run(400.0, g_clamp=10.0)
            k = (r['Vs'][-1] + 70.0) / (r['V'][-1, 0] + 70.0)
            self.assertLess(abs(k - attenuation(n_dend=N, n_active='all')), 1e-3)
            n = DendriticLIF(n_dend=N, dt=DT, Vth=1e9)
            clamp_one(n, 10.0, 4000)
            k = (n.Vs + 70.0) / (n.V[0] + 70.0)
            self.assertLess(abs(k - attenuation(n_dend=N, n_active=1)), 1e-3)

    def test_general_solver_unequal_drive(self):
        gd = np.array([25.0, 8.0, 0.0, 3.0])
        n = DendriticLIF(n_dend=4, dt=DT, Vth=1e9)
        for _ in range(6000):
            n.g = gd.copy()
            I_d = n.gL * (n.EL - n.V) + n.g * (n.Ea - n.V) + n.gc * (n.Vs - n.V)
            I_s = n.gL * (n.EL - n.Vs) + np.sum(n.gc * (n.V - n.Vs))
            n.V = n.V + DT * I_d / n.Cd
            n.Vs = n.Vs + DT * I_s / n.Cs
        V, Vs = steady_state_general(gd)
        self.assertLess(np.max(np.abs(n.V - V)), 1e-3)
        self.assertLess(abs(n.Vs - Vs), 1e-3)

    def test_euler_transient_first_order(self):
        """A steady state cannot test order: Euler's fixed point does not depend on dt."""
        def transient(step):
            return DendriticLIF(dt=step, Vth=1e9).run(5.0, g_clamp=15.0)['Vs'][-1]
        v1, v2, v3 = transient(0.1), transient(0.05), transient(0.025)
        self.assertTrue(1.8 < (v1 - v2) / (v2 - v3) < 2.2)
        self.assertLess(abs(v1 - (2 * v3 - v2)), 0.2)

    def test_refractory_caps_rate(self):
        r = DendriticLIF(dt=DT).run(1000.0, g_clamp=500.0)
        self.assertGreaterEqual(np.diff(r['spikes']).min(), 2.0 - 1e-9)

    def test_somatic_inhibition(self):
        """Inhibition hyperpolarises toward E_I and is exactly absent without input."""
        a, b = DendriticLIF(dt=DT), DendriticLIF(dt=DT)
        for _ in range(100):
            a.step(np.zeros(1))
            b.step(np.zeros(1), inh_in=0.0)
        self.assertEqual(a.Vs, b.Vs)
        n = DendriticLIF(dt=DT)
        n.step(None, inh_in=5.0)
        for _ in range(50):
            n.step(None)
        self.assertLess(n.Vs, -70.0)
        self.assertGreater(n.Vs, n.EI)


if __name__ == '__main__':
    unittest.main()
