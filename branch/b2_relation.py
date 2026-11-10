"""b2: how the rate-calcium relation changes once the soma fires (branch/DESIGN.md).

Every run has a bAP-off twin: same input and identical spikes (NMDA carries no
current), with the bAP amplitudes set to zero. The difference is calcium that enters
only because the soma fired.
  S1 global:     every branch at 1-4 Hz per synapse.
  S2 one branch: branch 1 at 0.5-8 Hz, branches 2 and 3 at the 2.5 Hz baseline.

Run: python branch/b2_relation.py [--workers 3] [--replot]
Writes results/b2_rows.json, results/b2_samples.npz, results/b2_output.txt,
plots/fig_b2_relation.png, plots/fig_b2_traces.png.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
from contextlib import redirect_stdout
import json

import numpy as np
from scipy import stats

from branch_core import (BASE_INH, BASE_RATE, C0, CHUNK_MS, N_BRANCH, PLOTS, RESULTS, Background, auc, burn_ms,
                         make_sim, mean_ci, rolling_mean, save_json)

SWEEPS = {'S1': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'S2': [0.5, 1.0, 2.0, 2.5, 4.0, 8.0]}
SEEDS = 6
RUN_S = 200
TRACE_S = 3
PHYSIO_SLOPE = 0.1485          # uM excess per Hz, one spine at Btot 44 without bAPs (physio p1)


def rates_for(sweep, level):
    return [level] * N_BRANCH if sweep == 'S1' else [level] + [BASE_RATE] * (N_BRANCH - 1)


def job(args):
    sweep, level, seed = args
    rates = rates_for(sweep, level)
    on, off = make_sim(), make_sim(bap=False)
    bg = Background([51, 1 if sweep == 'S1' else 2, int(level * 100), seed])
    for _ in range(int(burn_ms() / CHUNK_MS)):
        counts, w_inh = bg.chunk(rates, BASE_INH)
        on.run(counts, w_inh=w_inh)
        off.run(counts, w_inh=w_inh)
    c_on, c_off, spikes = [], [], []
    for _ in range(int(RUN_S * 1000 / CHUNK_MS)):
        counts, w_inh = bg.chunk(rates, BASE_INH)
        a, s = on.run(counts, w_inh=w_inh)
        b, s2 = off.run(counts, w_inh=w_inh)
        if not np.array_equal(s, s2):
            raise RuntimeError('bAP-off twin diverged in spikes')
        c_on.append(a)
        c_off.append(b)
        spikes.extend(s)
    c_on, c_off = np.concatenate(c_on), np.concatenate(c_off)
    t0 = on.state.t - RUN_S * 1000
    isi = np.diff(spikes)
    row = dict(sweep=sweep, level=level, seed=seed, rates=rates, soma_hz=len(spikes) / RUN_S,
               isi_cv=float(isi.std() / isi.mean()) if len(isi) > 10 else None, branches=[])
    m_keep = {}
    for i in range(N_BRANCH):
        x_on, x_off = c_on[:, i] - C0, c_off[:, i] - C0
        m_on = rolling_mean(c_on[:, i])
        m_off = rolling_mean(c_off[:, i])
        row['branches'].append(dict(excess_on=float(x_on.mean()), excess_off=float(x_off.mean()),
                                    bap_share=float(1 - x_off.mean() / x_on.mean()),
                                    cv_on=float(x_on.std() / x_on.mean()), cv_off=float(x_off.std() / x_off.mean()),
                                    m_cv_on=float(m_on.std() / (m_on.mean() - C0)),
                                    m_cv_off=float(m_off.std() / (m_off.mean() - C0)),
                                    skew_on=float(stats.skew(c_on[:, i]))))
        m_keep[f'm_on_{i}'] = m_on[::100].astype(np.float32)
        m_keep[f'm_off_{i}'] = m_off[::100].astype(np.float32)
    trace = None
    if seed == 0 and level == BASE_RATE:
        n = TRACE_S * 1000
        trace = dict(c_on=c_on[:n].astype(np.float32), c_off=c_off[:n].astype(np.float32),
                     spikes=np.array([t - t0 for t in spikes if t - t0 < n], dtype=np.float32))
    return row, m_keep, trace


def key(sweep, level, seed):
    return f'{sweep}|{level:g}|{seed}'


def simulate(workers):
    jobs = [(sw, lv, s) for sw, levels in SWEEPS.items() for lv in levels for s in range(SEEDS)]
    print(f'  {len(jobs)} jobs x 2 twins x {RUN_S} s', flush=True)
    rows, samples, traces = [], {}, {}
    with ProcessPoolExecutor(workers) as pool:
        for k, (row, m, tr) in enumerate(pool.map(job, jobs)):
            rows.append(row)
            for name, v in m.items():
                samples[f"{key(row['sweep'], row['level'], row['seed'])}|{name}"] = v
            if tr is not None:
                for name, v in tr.items():
                    traces[f"{row['sweep']}|{name}"] = v
            if (k + 1) % 12 == 0:
                print(f'    {k + 1}/{len(jobs)}', flush=True)
    save_json(RESULTS / 'b2_rows.json', rows)
    np.savez_compressed(RESULTS / 'b2_samples.npz', **samples, **{f'trace|{k}': v for k, v in traces.items()})


def load():
    rows = json.loads((RESULTS / 'b2_rows.json').read_text())
    data = np.load(RESULTS / 'b2_samples.npz')
    return rows, {k: data[k] for k in data.files}


def cell(rows, sweep, level):
    return [r for r in rows if r['sweep'] == sweep and r['level'] == level]


def branch_stat(rows, sweep, level, i, field):
    return mean_ci([r['branches'][i][field] for r in cell(rows, sweep, level)])


def consecutive_auc(samples, sweep, branch, kind='on'):
    levels = SWEEPS[sweep]
    out = []
    for lo, hi in zip(levels[:-1], levels[1:]):
        per_seed = [auc(samples[f'{key(sweep, hi, s)}|m_{kind}_{branch}'], samples[f'{key(sweep, lo, s)}|m_{kind}_{branch}'])
                    for s in range(SEEDS)]
        m, ci = mean_ci(per_seed)
        out.append(dict(low=lo, high=hi, auc=m, ci=ci))
    return out


def report(rows, samples):
    print('b2: rate-calcium relation with a firing soma, 3 branches x 1000 synapses, Btot 44')
    print('=' * 78)
    print(f'  baseline {BASE_RATE} Hz/syn, somatic inhibition {BASE_INH:g} Hz; {SEEDS} seeds x {RUN_S} s; '
          'bAP-off twins share input and spikes')
    for sweep, label in (('S1', 'S1 GLOBAL: every branch at the rate'), ('S2', 'S2 ONE BRANCH: branch 1 at the rate, 2 and 3 at baseline')):
        print(f'\n  {label}')
        print('   rate | soma Hz  ISI CV | branch 1: excess on   off  bAP share  CV(M) on | '
              'branch 2: excess on   off  bAP share  CV(M) on')
        for lv in SWEEPS[sweep]:
            c = cell(rows, sweep, lv)
            hz = mean_ci([r['soma_hz'] for r in c])
            cv = np.mean([r['isi_cv'] for r in c if r['isi_cv'] is not None]) if any(r['isi_cv'] for r in c) else np.nan
            parts = []
            for i in (0, 1):
                g = lambda f: branch_stat(rows, sweep, lv, i, f)[0]
                parts.append(f"{g('excess_on'):13.4f} {g('excess_off'):6.4f} {g('bap_share'):9.1%} {g('m_cv_on'):9.3f}")
            print(f'   {lv:4.1f} | {hz[0]:6.2f}+-{hz[1]:4.2f} {cv:5.2f} | ' + ' | '.join(parts))
    print('\n  S2 separation on M (2 s rolling mean), AUC between consecutive branch-1 rates, seed-paired')
    for kind in ('on', 'off'):
        print(f'   bAPs {kind}:')
        for i, name in ((0, 'branch 1 (target)  '), (1, 'branch 2 (bystander)')):
            s = consecutive_auc(samples, 'S2', i, kind)
            print(f'    {name}: ' + '  '.join(f"{x['low']:g}->{x['high']:g} {x['auc']:.3f}+-{x['ci']:.3f}" for x in s))
    print(f'\n  reference: one spine at Btot 44 without bAPs (physio) has {PHYSIO_SLOPE:.4f} uM excess per Hz')


def figures(rows, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.8))

    a = ax[0, 0]
    for sweep, col, label in (('S1', 'k', 'S1: all branches'), ('S2', 'tab:blue', 'S2: branch 1 only')):
        lv = SWEEPS[sweep]
        hz = [mean_ci([r['soma_hz'] for r in cell(rows, sweep, x)]) for x in lv]
        a.errorbar(lv, [h[0] for h in hz], yerr=[h[1] for h in hz], fmt='o-', color=col, capsize=2, label=label)
    a.set_xlabel('rate per synapse (Hz)'); a.set_ylabel('somatic rate (Hz)')
    a.set_title('(a) output rate', fontsize=10.5); a.legend(fontsize=8)

    a = ax[0, 1]
    lv = np.array(SWEEPS['S1'])
    on = [branch_stat(rows, 'S1', x, 0, 'excess_on')[0] for x in lv]
    off = [branch_stat(rows, 'S1', x, 0, 'excess_off')[0] for x in lv]
    a.plot(lv, on, 'o-', color='tab:red', label='bAPs on')
    a.plot(lv, off, 's--', color='0.4', label='bAPs off (twin)')
    a.plot(lv, PHYSIO_SLOPE * lv, ':', color='tab:green', label='one spine, no bAPs (physio)')
    a.set_xlabel('rate per synapse (Hz), every branch'); a.set_ylabel('branch calcium excess (uM)')
    a.set_title('(b) S1: branch calcium vs rate', fontsize=10.5); a.legend(fontsize=8)

    a = ax[0, 2]
    for sweep, i, col, st, label in (('S1', 0, 'k', 'o-', 'S1, any branch'), ('S2', 0, 'tab:blue', 'o-', 'S2, branch 1 (target)'),
                                      ('S2', 1, 'tab:orange', 's--', 'S2, branch 2 (bystander)')):
        lvs = SWEEPS[sweep]
        a.plot(lvs, [branch_stat(rows, sweep, x, i, 'bap_share')[0] * 100 for x in lvs], st, color=col, label=label)
    a.set_xlabel('rate per synapse (Hz) of the swept branches'); a.set_ylabel('calcium from bAPs (%)')
    a.set_title('(c) share of branch calcium that needs a bAP', fontsize=10.5); a.legend(fontsize=8)

    a = ax[1, 0]
    lv = SWEEPS['S2']
    for i, col, label in ((0, 'tab:blue', 'branch 1 (target)'), (1, 'tab:orange', 'branch 2 (bystander)')):
        a.plot(lv, [branch_stat(rows, 'S2', x, i, 'excess_on')[0] for x in lv], 'o-', color=col, label=f'{label}, bAPs on')
        a.plot(lv, [branch_stat(rows, 'S2', x, i, 'excess_off')[0] for x in lv], 's:', color=col, label=f'{label}, off')
    a.set_xlabel('branch-1 rate per synapse (Hz)'); a.set_ylabel('branch calcium excess (uM)')
    a.set_title('(d) S2: target and bystander calcium', fontsize=10.5); a.legend(fontsize=7.5)

    a = ax[1, 1]
    lv = SWEEPS['S1']
    for sweep, i, col, label in (('S1', 0, 'k', 'S1'), ('S2', 0, 'tab:blue', 'S2 branch 1')):
        lvs = SWEEPS[sweep]
        a.plot(lvs, [branch_stat(rows, sweep, x, i, 'm_cv_on')[0] for x in lvs], 'o-', color=col, label=f'{label}, bAPs on')
        a.plot(lvs, [branch_stat(rows, sweep, x, i, 'm_cv_off')[0] for x in lvs], 's:', color=col, label=f'{label}, off')
    a.set_xlabel('rate per synapse (Hz)'); a.set_ylabel('CV of M (2 s mean)')
    a.set_title('(e) fluctuation of what the Bayesian layer reads', fontsize=10.5); a.legend(fontsize=7.5)

    a = ax[1, 2]
    centers = [np.sqrt(lo * hi) for lo, hi in zip(SWEEPS['S2'][:-1], SWEEPS['S2'][1:])]
    for i, col, label in ((0, 'tab:blue', 'branch 1 (target)'), (1, 'tab:orange', 'branch 2 (bystander)')):
        for kind, st in (('on', 'o-'), ('off', 's:')):
            s = consecutive_auc(samples, 'S2', i, kind)
            a.errorbar(centers, [x['auc'] for x in s], yerr=[x['ci'] for x in s], fmt=st, color=col, capsize=2,
                       label=f'{label}, bAPs {kind}')
    a.axhline(0.5, color='0.6', lw=0.8); a.set_ylim(0.3, 1.02); a.set_xscale('log')
    a.set_xticks(centers); a.set_xticklabels([f'{lo:g}-{hi:g}' for lo, hi in zip(SWEEPS['S2'][:-1], SWEEPS['S2'][1:])], fontsize=8)
    a.minorticks_off()
    a.set_xlabel('consecutive branch-1 rates (Hz)'); a.set_ylabel('AUC on M')
    a.set_title('(f) S2: does a branch report its own input?', fontsize=10.5); a.legend(fontsize=7)

    fig.suptitle('b2: branch calcium once the soma fires (3 branches x 1000 synapses, correlated background, Btot 44)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / 'fig_b2_relation.png', dpi=140)
    plt.close(fig)
    print(f"saved {PLOTS / 'fig_b2_relation.png'}")

    tr_on, tr_off, sp = samples['trace|S1|c_on'], samples['trace|S1|c_off'], samples['trace|S1|spikes']
    t = np.arange(len(tr_on)) / 1000.0
    fig, ax = plt.subplots(N_BRANCH, 1, figsize=(12, 6.5), sharex=True)
    for i in range(N_BRANCH):
        ax[i].plot(t, tr_on[:, i] - C0, color='tab:red', lw=0.9, label='bAPs on')
        ax[i].plot(t, tr_off[:, i] - C0, color='0.4', lw=0.9, label='bAPs off (same input)')
        for x in sp:
            ax[i].axvline(x / 1000.0, color='tab:blue', lw=0.6, alpha=0.5)
        ax[i].set_ylabel(f'branch {i + 1}\nexcess (uM)')
    ax[0].legend(fontsize=8, loc='upper right')
    ax[-1].set_xlabel('s (blue lines: somatic spikes)')
    fig.suptitle(f'b2 baseline, {BASE_RATE} Hz per synapse on every branch: each somatic spike lifts calcium in every branch',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / 'fig_b2_traces.png', dpi=140)
    plt.close(fig)
    print(f"saved {PLOTS / 'fig_b2_traces.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--replot', action='store_true')
    args = parser.parse_args()
    if not args.replot:
        simulate(args.workers)
    rows, samples = load()
    text = io.StringIO()
    with redirect_stdout(text):
        report(rows, samples)
    (RESULTS / 'b2_output.txt').write_text(text.getvalue())
    print(text.getvalue())
    figures(rows, samples)


if __name__ == '__main__':
    main()
