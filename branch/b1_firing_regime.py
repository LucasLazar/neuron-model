"""b1: somatic firing regime of the 3-branch pooled neuron (branch/DESIGN.md).

Per-synapse rate on every branch x OU sigma (0, 0.5) x somatic inhibition (0, 1000 Hz),
3 seeds x 100 s. Reports somatic rate, ISI CV and mean branch voltage.

Run: python branch/b1_firing_regime.py [--workers 3]
Writes results/b1.json, results/b1_output.txt, plots/fig_b1_regime.png.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
from contextlib import redirect_stdout

import numpy as np

from branch_core import BASE_INH, BASE_RATE, CHUNK_MS, OU_SIGMA, PLOTS, RESULTS, Background, burn_ms, make_sim, save_json

RATES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
SIGMAS = [0.0, OU_SIGMA]
INHIBITION = [0.0, BASE_INH]
SEEDS = 3
RUN_S = 100


def job(args):
    rate, sigma, inh, seed = args
    sim = make_sim()
    bg = Background([41, int(rate * 10), int(sigma * 10), int(inh), seed], sigma=sigma)
    for _ in range(int(burn_ms() / CHUNK_MS)):
        counts, w_inh = bg.chunk([rate] * 3, inh)
        sim.run(counts, w_inh=w_inh)
    spikes, V = [], []
    for _ in range(int(RUN_S * 1000 / CHUNK_MS)):
        counts, w_inh = bg.chunk([rate] * 3, inh)
        _, s = sim.run(counts, w_inh=w_inh)
        spikes.extend(s)
        V.append(sim.state.V[0])
    isi = np.diff(spikes)
    return dict(rate=rate, sigma=sigma, inh=inh, seed=seed, soma_hz=len(spikes) / RUN_S,
                isi_cv=float(isi.std() / isi.mean()) if len(isi) > 10 else None, branch_V=float(np.mean(V)))


def summarise(rows):
    out = {}
    for r in RATES:
        for sg in SIGMAS:
            for inh in INHIBITION:
                cell = [x for x in rows if x['rate'] == r and x['sigma'] == sg and x['inh'] == inh]
                cvs = [x['isi_cv'] for x in cell if x['isi_cv'] is not None]
                out[(r, sg, inh)] = dict(soma_hz=np.mean([x['soma_hz'] for x in cell]),
                                         isi_cv=np.mean(cvs) if cvs else np.nan,
                                         branch_V=np.mean([x['branch_V'] for x in cell]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    jobs = [(r, sg, inh, s) for r in RATES for sg in SIGMAS for inh in INHIBITION for s in range(SEEDS)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(job, jobs))
    save_json(RESULTS / 'b1.json', rows)
    table = summarise(rows)
    text = io.StringIO()
    with redirect_stdout(text):
        print('b1: firing regime, 3 branches x 1000 synapses, w_syn 1 nS, Btot 44')
        print('=' * 78)
        print('  rate/syn | ' + ' | '.join(f'sigma {sg:.1f} inh {inh:4.0f}: Hz   CV    V' for sg in SIGMAS for inh in INHIBITION))
        for r in RATES:
            parts = []
            for sg in SIGMAS:
                for inh in INHIBITION:
                    c = table[(r, sg, inh)]
                    parts.append(f"{'':16s}{c['soma_hz']:6.2f} {c['isi_cv']:5.2f} {c['branch_V']:5.1f}")
            print(f'  {r:6.1f} Hz | ' + ' | '.join(parts))
        b = table[(BASE_RATE, OU_SIGMA, BASE_INH)]
        print(f"\n  baseline ({BASE_RATE} Hz/syn, sigma {OU_SIGMA}, inh {BASE_INH:g} Hz): soma {b['soma_hz']:.2f} Hz, "
              f"ISI CV {b['isi_cv']:.2f}, branch V {b['branch_V']:.1f} mV")
    (RESULTS / 'b1_output.txt').write_text(text.getvalue())
    print(text.getvalue())
    figure(table)


def figure(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {(0.0, 0.0): ('0.6', 'o--'), (0.0, BASE_INH): ('0.2', 's--'),
              (OU_SIGMA, 0.0): ('tab:orange', 'o-'), (OU_SIGMA, BASE_INH): ('tab:red', 's-')}
    for (sg, inh), (col, st) in styles.items():
        hz = [table[(r, sg, inh)]['soma_hz'] for r in RATES]
        cv = [table[(r, sg, inh)]['isi_cv'] for r in RATES]
        label = f'{"correlated" if sg else "independent"} input, inhibition {inh:g} Hz'
        ax[0].plot(RATES, np.maximum(hz, 0.01), st, color=col, label=label)
        ax[1].plot(RATES, cv, st, color=col, label=label)
    ax[0].set_yscale('log'); ax[0].set_ylim(0.01, 300)
    ax[0].axhspan(1, 10, color='0.92', zorder=0)
    ax[0].set_xlabel('rate per synapse (Hz), every branch'); ax[0].set_ylabel('somatic rate (Hz; 0 drawn at 0.01)')
    ax[0].set_title('(a) somatic rate (grey band: 1-10 Hz)', fontsize=10.5)
    ax[1].axhline(1, color='0.6', lw=0.8)
    ax[1].set_xlabel('rate per synapse (Hz), every branch'); ax[1].set_ylabel('ISI CV')
    ax[1].set_title('(b) irregularity (1 = Poisson-like)', fontsize=10.5)
    ax[0].legend(fontsize=8)
    fig.suptitle('b1: only correlated input gives low, irregular firing (3 branches x 1000 synapses)', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / 'fig_b1_regime.png', dpi=140)
    print(f"saved {PLOTS / 'fig_b1_regime.png'}")


if __name__ == '__main__':
    main()
