"""Time-step convergence against an independent ODE solver (tests/reference_solver.py).

A short version of physio/p2_convergence.py block A: one per-synapse case, 32 Hz
with the reference buffer. The full study is physio/p2 (about 5 minutes).

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import reference_solver as ref  # noqa: E402
from neuron.fast import WEIGHT, Sim  # noqa: E402

DURATION, BUFFER, RATE = 400.0, 44.0, 32


class AgainstRadau(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng([77, RATE, int(BUFFER)])
        slots = np.flatnonzero(rng.random(int(DURATION / .2)) < RATE * .2 / 1000)
        cls.events = (slots * .2)[(slots * .2) > 5.0]
        _, states, spikes = ref.reference(cls.events, DURATION, BUFFER)
        cls.c_true, cls.ref_spikes = states[:, ref.C], spikes

    def discrete(self, dt, scheme='exact'):
        w = np.zeros((int(round(DURATION / dt)), 1))
        for t in self.events:
            w[int(round(t / dt)), 0] += WEIGHT
        calcium, spikes = Sim(n=1, dt=dt, Btot=BUFFER, u_quadrature=scheme, J_max=ref.P['J_max']).run(w)
        self.assertEqual(len(spikes), len(self.ref_spikes))
        return calcium[:, 0]

    def excess_error(self, c):
        return (c.mean() - self.c_true.mean()) / (self.c_true.mean() - 0.1)

    def test_default_step_is_within_two_percent(self):
        c = self.discrete(0.1)
        self.assertGreater(len(self.events), 5)
        self.assertLess(abs(self.excess_error(c)), 0.02)
        self.assertLess(abs(c.max() - self.c_true.max()) / (self.c_true.max() - 0.1), 0.03)

    def test_first_order(self):
        errors = [np.max(np.abs(self.discrete(dt) - self.c_true)) for dt in (0.1, 0.05, 0.025)]
        for coarse, fine in zip(errors[:-1], errors[1:]):
            self.assertTrue(1.6 < coarse / fine < 2.4, errors)

    def test_frozen_step_over_integrates_by_ten_percent(self):
        self.assertTrue(0.07 < self.excess_error(self.discrete(0.1, 'frozen')) < 0.12)


if __name__ == '__main__':
    unittest.main()
