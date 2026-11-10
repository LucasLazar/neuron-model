"""b0: how good is the pooled gate? Compare it with explicitly simulated synapses.

K synapses, each with its own cleft u_k and occupancy s_k under the nonlinear split
gate (exact u^2 step), against the pooled linear gate driven by the summed event
counts. The voltage factor B(V) multiplies both identically, so total occupancy is
the whole approximation. Per-synapse Poisson input, optionally with the shared OU
rate modulation used in the experiments.

Run: python branch/b0_pooled_gate_check.py
Writes results/b0_output.txt and results/b0.json.
"""
import io
from contextlib import redirect_stdout

import numpy as np

from branch_core import OU_SIGMA, RESULTS, Background, NMDANeuron, save_json

K = 200
SECONDS = 20
RATES = [1.0, 2.5, 5.0, 10.0, 20.0]
DT = 0.1


def run(rate, sigma, seed=0):
    ref = NMDANeuron(n_dend=1, dt=DT, Btot=0.0, plasticity_mode='off', u_quadrature='exact',
                     pooled_gate=True, n_syn=K)
    bg = Background([31, int(rate * 10), int(sigma * 10), seed], sigma=sigma)
    u, s = np.zeros(K), np.zeros(K)
    v, linear, meanfield = 0.0, 0.0, 0.0
    dec_N = np.exp(-DT / ref.tau_N)
    explicit, lin, mf = [], [], []
    for _ in range(SECONDS * 10):
        bg.chunk([rate], 0.0)                     # advances the shared OU factor
        steps = len(bg.last_factor)
        # each synapse fires independently at the shared, modulated rate
        hits = bg.rng.random((steps, K)) < (bg.last_factor[:, None] * rate * DT / 1000.0)
        for k in range(steps):
            fired = hits[k]
            u *= ref._dec_u
            u[fired] += ref.a_u * (1.0 - u[fired])
            A = ref.k_b * u * u * ref._u_phi
            rate_s = A + 1.0 / ref.tau_N
            s_inf = A / rate_s
            s = s_inf + (s - s_inf) * np.exp(-DT * rate_s)
            n_ev = fired.sum()
            v = v * ref._dec_v + n_ev * ref._a_um
            A_p = ref.k_eff * v * ref._u_phi
            linear = A_p * ref.tau_N + (linear - A_p * ref.tau_N) * dec_N
            rate_mf = A_p / K + 1.0 / ref.tau_N
            meanfield = A_p / rate_mf + (meanfield - A_p / rate_mf) * np.exp(-DT * rate_mf)
            explicit.append(s.sum())
            lin.append(linear)
            mf.append(meanfield)
    explicit = np.array(explicit)
    row = dict(rate=rate, sigma=sigma, mean_occupancy_per_synapse=float(explicit.mean() / K))
    for name, pooled in (('linear', np.array(lin)), ('meanfield', np.array(mf))):
        row[name] = dict(mean_ratio=float(pooled.mean() / explicit.mean()),
                         rms_relative=float(np.sqrt(np.mean((pooled - explicit) ** 2)) / explicit.mean()),
                         corr=float(np.corrcoef(pooled, explicit)[0, 1]))
    return row


def main():
    rows = [run(r, sg) for sg in (0.0, OU_SIGMA) for r in RATES]
    save_json(RESULTS / 'b0.json', rows)
    text = io.StringIO()
    with redirect_stdout(text):
        print('b0: pooled gate vs K = 200 explicit nonlinear synapses, 20 s each')
        print('=' * 78)
        print('                              |     linear gate: pooled/explicit   | mean-field gate (production)')
        print(' OU sigma  rate/syn | s per syn |   mean ratio  rms diff/mean  corr |   mean ratio  rms diff/mean  corr')
        for r in rows:
            L, M = r['linear'], r['meanfield']
            print(f"   {r['sigma']:5.1f} {r['rate']:8.1f} Hz | {r['mean_occupancy_per_synapse']:9.4f} | "
                  f"{L['mean_ratio']:12.4f} {L['rms_relative']:14.4f} {L['corr']:6.4f} | "
                  f"{M['mean_ratio']:12.4f} {M['rms_relative']:14.4f} {M['corr']:6.4f}")
        print('\n linear: each event adds an isolated-synapse response; overestimates because it omits a\n'
              ' synapse\'s own earlier occupancy. mean-field: (1 - s/K), the occupancy a Poisson synapse\n'
              ' sees on average when it fires.')
    (RESULTS / 'b0_output.txt').write_text(text.getvalue())
    print(text.getvalue())


if __name__ == '__main__':
    main()
