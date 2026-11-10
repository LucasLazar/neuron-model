"""p1: free-calcium distribution vs input rate and buffer, at physiological rates.

exp5 (buffer at fixed rate) and exp6 with its per-buffer panel A, rerun on the
per-synapse grid: 1-64 Hz x Btot 0-300 uM, retuned J_max and the untuned J_max = 0.5 as
control, 8 seeds x 400 s per cell. The same seed is the same spike train in every
buffer and tuning, so those comparisons are paired.

Separation of adjacent rates is AUC = P(X at 2r > X at r), on 1 ms calcium samples
and on the 2 s rolling mean M that the Bayesian layer reads.

Run: python physio/p1_rate_buffer_sweep.py [--workers 3] [--replot]
Writes results/p1_stats.json, results/p1_samples.npz, results/p1_output.txt, plots/fig_p1_*.png.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
from contextlib import redirect_stdout
import json
import time

import numpy as np
from scipy import stats

from physio_core import (BUFFERS, C0, NEAR_REST_UM, PLOTS, RATES, RESULTS, auc, kappa, mean_ci, rolling_mean,
                         save_json, stationary_run, tau_int, tunings)

N_SEEDS = 8
RUN_S = 400
RAW_EVERY = 25        # keep every 25th 1 ms sample for densities and AUC
M_EVERY = 100         # and every 100th rolling mean
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


def job(args):
    tuning, J, rate, buffer, seed = args
    c, occupancy, spikes = stationary_run(J, buffer, rate, seed, RUN_S)
    m = rolling_mean(c)
    excess = c - C0
    half = len(c) // 2
    tau, _ = tau_int([c])
    row = dict(tuning=tuning, rate=rate, buffer=buffer, seed=seed, spikes=spikes, soma_hz=spikes / RUN_S,
               mean=float(c.mean()), sd=float(c.std()), cv_excess=float(c.std() / excess.mean()),
               skew=float(stats.skew(c)), exkurt=float(stats.kurtosis(c)),
               near_rest=float(np.mean(excess < NEAR_REST_UM)), tau_int_s=float(tau / 1000.0),
               drift=float((c[half:].mean() - c[:half].mean()) / c.std()),
               quantiles=np.quantile(c, QUANTILES).tolist(),
               occupancy_mean=float(np.nanmean(occupancy)) if buffer > 0 else None,
               occupancy_p99=float(np.nanquantile(occupancy, .99)) if buffer > 0 else None,
               m_mean=float(m.mean()), m_sd=float(m.std()), m_cv_excess=float(m.std() / (m.mean() - C0)),
               m_skew=float(stats.skew(m)))
    return row, c[::RAW_EVERY].astype(np.float32), m[::M_EVERY].astype(np.float32)


def simulate(workers):
    tune = tunings()
    jobs = [(name, J, r, B, s) for name, J in tune.items() for r in RATES for B in BUFFERS for s in range(N_SEEDS)]
    print(f"  {len(jobs)} jobs: {len(tune)} tunings x {len(RATES)} rates x {len(BUFFERS)} buffers x {N_SEEDS} seeds "
          f"x {RUN_S} s; workers {workers}", flush=True)
    started = time.time()
    rows, raw, roll = [], {}, {}
    with ProcessPoolExecutor(workers) as pool:
        for k, (row, c, m) in enumerate(pool.map(job, jobs, chunksize=2)):
            rows.append(row)
            key = f"{row['tuning']}|{row['rate']:g}|{row['buffer']:g}|{row['seed']}"
            raw[key], roll[key] = c, m
            if (k + 1) % 40 == 0:
                print(f"    {k + 1}/{len(jobs)}  [{time.time() - started:.0f}s]", flush=True)
    print(f"  done in {time.time() - started:.0f}s")
    save_json(RESULTS / 'p1_stats.json', dict(tunings=tune, run_s=RUN_S, n_seeds=N_SEEDS, rows=rows))
    np.savez_compressed(RESULTS / 'p1_samples.npz', **{f'raw:{k}': v for k, v in raw.items()},
                        **{f'm:{k}': v for k, v in roll.items()})


def load():
    meta = json.loads((RESULTS / 'p1_stats.json').read_text())
    data = np.load(RESULTS / 'p1_samples.npz')
    raw, roll = {}, {}
    for name in data.files:
        kind, key = name.split(':', 1)
        (raw if kind == 'raw' else roll)[key] = data[name]
    return meta, raw, roll


def key(tuning, rate, buffer, seed):
    return f"{tuning}|{rate:g}|{buffer:g}|{seed}"


def cell(rows, tuning, rate, buffer):
    return [r for r in rows if r['tuning'] == tuning and r['rate'] == rate and r['buffer'] == buffer]


def summary(rows, tuning, rate, buffer, field):
    values = [r[field] for r in cell(rows, tuning, rate, buffer)]
    return mean_ci(values)


def separation(samples, tuning, buffer, n_seeds):
    """AUC for each adjacent rate pair: pooled point estimate and seed-paired CI."""
    out = []
    for lo, hi in zip(RATES[:-1], RATES[1:]):
        pooled = auc(np.concatenate([samples[key(tuning, hi, buffer, s)] for s in range(n_seeds)]),
                     np.concatenate([samples[key(tuning, lo, buffer, s)] for s in range(n_seeds)]))
        per_seed = [auc(samples[key(tuning, hi, buffer, s)], samples[key(tuning, lo, buffer, s)]) for s in range(n_seeds)]
        out.append(dict(low=lo, high=hi, auc=pooled, ci=mean_ci(per_seed)[1]))
    return out


def report(meta, raw, roll):
    rows, n = meta['rows'], meta['n_seeds']
    tune = meta['tunings']
    print('p1: free calcium vs input rate and buffer, per-synapse reading')
    print('=' * 78)
    print(f"  J_max retuned {tune['retuned']:.3f}, untuned {tune['untuned']:g} uM/ms; "
          f"{n} seeds x {meta['run_s']} s per cell; Poisson input, one compartment, soma silent unless noted")
    sep = {}
    for tuning in ('retuned', 'untuned'):
        print(f"\n######## {tuning.upper()} J_max = {tune[tuning]:g}")
        for B in BUFFERS:
            print(f"\n  Btot = {B:g} uM   (kappa at rest {kappa(B):.1f})")
            print('   rate  soma |    mean c   excess    CV(dc)   skew  near-rest  tau_int | occ mean  occ p99 |'
                  '  M CV(dc)  M skew |  drift')
            for r in RATES:
                g = lambda f: summary(rows, tuning, r, B, f)[0]
                occ = (f"{g('occupancy_mean'):8.3f} {g('occupancy_p99'):8.3f}" if B > 0 else f"{'--':>8s} {'--':>8s}")
                drift = max(abs(x['drift']) for x in cell(rows, tuning, r, B))
                print(f"   {r:4g} {g('soma_hz'):5.2f} | {g('mean'):9.4f} {g('mean') - C0:8.4f} {g('cv_excess'):9.3f} "
                      f"{g('skew'):6.2f} {g('near_rest'):9.3f} {g('tau_int_s'):7.2f}s | {occ} | "
                      f"{g('m_cv_excess'):9.3f} {g('m_skew'):7.2f} | {drift:6.3f}")
        for B in BUFFERS:
            sep[(tuning, B, 'raw')] = separation(raw, tuning, B, n)
            sep[(tuning, B, 'm')] = separation(roll, tuning, B, n)
        print(f"\n  SEPARATION of adjacent rates, AUC = P(X at 2r > X at r); +- is the seed-paired 95% CI")
        for kind, label in (('raw', '1 ms calcium'), ('m', '2 s rolling mean M')):
            print(f"   {label}")
            print('     Btot | ' + '  '.join(f"{lo:g}->{hi:g} Hz".rjust(14) for lo, hi in zip(RATES[:-1], RATES[1:])))
            for B in BUFFERS:
                print(f"    {B:5g} | " + '  '.join(f"{x['auc']:.3f}+-{x['ci']:.3f}".rjust(14) for x in sep[(tuning, B, kind)]))

    print('\n  CHECKS')
    for tuning in ('retuned', 'untuned'):
        worst = 0.0
        for r in RATES:
            means = [summary(rows, tuning, r, B, 'mean')[0] - C0 for B in BUFFERS]
            worst = max(worst, (max(means) - min(means)) / np.mean(means))
        print(f"   {tuning}: mean excess across buffers at fixed rate, worst relative spread {worst:.2%} "
              f"(conservation: <c> = c0 + tau_c <J>, buffer-independent)")
    worst = max(abs(summary(rows, 'retuned', r, 0.0, 'cv_excess')[0] - summary(rows, 'untuned', r, 0.0, 'cv_excess')[0])
                for r in RATES)
    print(f"   Btot 0: calcium is linear in J_max, so CV must match between tunings; worst |dCV| = {worst:.2e}")
    drift = max(abs(x['drift']) for x in rows)
    print(f"   burn-in: largest |drift| over all {len(rows)} runs = {drift:.3f} sd")
    print(f"   soma: total spikes over all runs = {sum(x['spikes'] for x in rows)}")
    return sep


def kde_panel(ax, samples, tuning, buffer, n, xlim, colors, legend):
    for i, r in enumerate(RATES):
        x = np.log10(np.maximum(np.concatenate([samples[key(tuning, r, buffer, s)] for s in range(n)]) - C0, NEAR_REST_UM))
        x = x[::max(1, len(x) // 20000)]
        grid = np.linspace(xlim[0], xlim[1], 500)
        if np.ptp(x) == 0:
            continue
        dens = stats.gaussian_kde(x)(grid)
        ax.plot(10 ** grid, dens / dens.max(), color=colors[i], lw=1.5, label=f'{r:g}')
    ax.set_xscale('log')
    ax.set_xlim(10 ** xlim[0], 10 ** xlim[1])
    ax.set_ylim(0, 1.12)
    ax.set_title(f'Btot = {buffer:g} uM   (kappa {kappa(buffer):.0f})', fontsize=10.5)
    if legend:
        ax.legend(title='rate (Hz)', fontsize=7, title_fontsize=7.5, ncol=2, loc='upper left', framealpha=0.85)


def auc_panel(ax, sep, tuning, kind, title):
    col = __import__('matplotlib.pyplot', fromlist=['cm']).cm.viridis(np.linspace(0.05, 0.85, len(BUFFERS)))
    centers = [np.sqrt(lo * hi) for lo, hi in zip(RATES[:-1], RATES[1:])]
    for i, B in enumerate(BUFFERS):
        s = sep[(tuning, B, kind)]
        ax.errorbar(centers, [x['auc'] for x in s], yerr=[x['ci'] for x in s], fmt='o-', color=col[i], ms=4,
                    capsize=2, label=f'Btot {B:g}')
    ax.set_xscale('log')
    ax.set_xticks(centers)
    ax.set_xticklabels([f'{lo:g}-{hi:g}' for lo, hi in zip(RATES[:-1], RATES[1:])], fontsize=8)
    ax.set_ylim(0.45, 1.01)
    ax.axhline(0.5, color='0.6', lw=0.8)
    ax.set_xlabel('adjacent rate pair (Hz)')
    ax.set_ylabel('AUC  P(X at 2r > X at r)')
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7.5, loc='lower right')


def figures(meta, raw, roll, sep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = meta['n_seeds']
    rows = meta['rows']
    colors = plt.cm.plasma(np.linspace(0.05, 0.88, len(RATES)))
    PLOTS.mkdir(parents=True, exist_ok=True)
    for kind, samples, name, what in (('raw', raw, 'fig_p1_panelA_raw.png', '1 ms free calcium'),
                                      ('m', roll, 'fig_p1_panelA_rolling.png', '2 s rolling mean M (what the Bayesian layer reads)')):
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.8), sharey=False)
        xlim = (np.log10(NEAR_REST_UM) - 0.1, 1.45) if kind == 'raw' else (-2.6, 1.3)
        for p, B in enumerate(BUFFERS):
            a = ax.flat[p]
            kde_panel(a, samples, 'retuned', B, n, xlim, colors, legend=(p == 0))
            a.set_xlabel('free calcium above rest (uM, log axis)' if kind == 'raw' else 'M above rest (uM, log axis)')
            a.set_ylabel('density of log excess, peak-normalised')
        auc_panel(ax.flat[5], sep, 'retuned', kind, f'adjacent-rate separation, {"1 ms calcium" if kind == "raw" else "M"}')
        floor = (f'Values within {NEAR_REST_UM * 1000:g} nM of rest are drawn at the left edge.' if kind == 'raw' else '')
        fig.suptitle(f'p1 panel A at physiological rates, retuned J_max: {what}\n'
                     f'Within a panel: rate (doubling grid). Across panels: buffer. {n} x {meta["run_s"]} s per curve. {floor}',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(PLOTS / name, dpi=140)
        plt.close(fig)
        print(f"saved {PLOTS / name}")

    col = plt.cm.viridis(np.linspace(0.05, 0.85, len(BUFFERS)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.8))
    rates = np.array(RATES, dtype=float)

    def series(tuning, B, field):
        return np.array([summary(rows, tuning, r, B, field)[0] for r in RATES], dtype=float)

    a = ax[0, 0]
    for tuning, style in (('retuned', 'o-'), ('untuned', 's--')):
        a.plot(rates, series(tuning, 0.0, 'mean') - C0, style, color='k' if tuning == 'retuned' else '0.55',
               label=f'{tuning} (every buffer)')
    a.plot(rates, (series('retuned', 0.0, 'mean')[0] - C0) * rates / rates[0], ':', color='tab:red', label='slope 1')
    a.axhline(C0, color='tab:blue', lw=0.8, ls=':')
    a.text(rates[0], C0 * 1.15, 'resting level c0', color='tab:blue', fontsize=8)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('rate (Hz)'); a.set_ylabel('mean excess <c> - c0 (uM)')
    a.set_title('(a) mean excess: linear in rate, buffer-independent', fontsize=10.5)
    a.legend(fontsize=8)

    for a, field, title in ((ax[0, 1], 'cv_excess', '(b) CV of the excess, 1 ms calcium'),
                            (ax[0, 2], 'm_cv_excess', '(c) CV of the excess, 2 s rolling mean M')):
        for i, B in enumerate(BUFFERS):
            a.plot(rates, series('retuned', B, field), 'o-', color=col[i], ms=4, label=f'Btot {B:g}')
            a.plot(rates, series('untuned', B, field), ':', color=col[i], lw=1)
        a.plot(rates, series('retuned', 0.0, field)[0] * (rates / rates[0]) ** -0.5, '--', color='tab:red', lw=1,
               label='shot noise, rate^-1/2')
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('rate (Hz)'); a.set_ylabel('sd / (mean - c0)')
        a.set_title(title + '\n(dotted: untuned J_max 0.5)', fontsize=10.5)
        a.legend(fontsize=7.5)

    a = ax[1, 0]
    for i, B in enumerate(BUFFERS):
        a.plot(rates, series('retuned', B, 'skew'), 'o-', color=col[i], ms=4, label=f'Btot {B:g}')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('rate (Hz)'); a.set_ylabel('skew, 1 ms calcium')
    a.set_title('(d) skew', fontsize=10.5)
    a.legend(fontsize=7.5)

    a = ax[1, 1]
    for i, B in enumerate(BUFFERS[1:], start=1):
        a.plot(rates, series('retuned', B, 'occupancy_mean'), 'o-', color=col[i], ms=4, label=f'Btot {B:g} mean')
        a.plot(rates, series('retuned', B, 'occupancy_p99'), '^:', color=col[i], ms=4, lw=1)
    a.axhline(C0 / (C0 + 2.0), color='0.6', lw=0.8)
    a.set_xscale('log'); a.set_ylim(0, 1); a.set_xlabel('rate (Hz)'); a.set_ylabel('buffer occupancy b / Btot')
    a.set_title('(e) buffer occupancy, retuned (triangles: 99th pct;\ngrey line: rest)', fontsize=10.5)
    a.legend(fontsize=7.5)

    a = ax[1, 2]
    centers = [np.sqrt(lo * hi) for lo, hi in zip(RATES[:-1], RATES[1:])]
    for i, B in enumerate(BUFFERS):
        a.plot(centers, [x['auc'] for x in sep[('retuned', B, 'm')]], 'o-', color=col[i], ms=4, label=f'Btot {B:g}')
        a.plot(centers, [x['auc'] for x in sep[('untuned', B, 'm')]], ':', color=col[i], lw=1.2)
    a.set_xscale('log'); a.set_xticks(centers)
    a.set_xticklabels([f'{lo:g}-{hi:g}' for lo, hi in zip(RATES[:-1], RATES[1:])], fontsize=8)
    a.set_ylim(0.45, 1.01); a.axhline(0.5, color='0.6', lw=0.8)
    a.set_xlabel('adjacent rate pair (Hz)'); a.set_ylabel('AUC on M')
    a.set_title('(f) separation on M: retuned (solid) vs untuned (dotted)', fontsize=10.5)
    a.legend(fontsize=7.5)

    fig.suptitle('p1 summary: position, width, shape, buffer load and separation vs rate', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS / 'fig_p1_summary.png', dpi=140)
    plt.close(fig)
    print(f"saved {PLOTS / 'fig_p1_summary.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--replot', action='store_true')
    args = parser.parse_args()
    if not args.replot:
        print('p1: running')
        simulate(args.workers)
    meta, raw, roll = load()
    text = io.StringIO()
    with redirect_stdout(text):
        sep = report(meta, raw, roll)
    (RESULTS / 'p1_output.txt').write_text(text.getvalue())
    print(text.getvalue())
    save_json(RESULTS / 'p1_separation.json',
              {f'{t}|{B:g}|{k}': v for (t, B, k), v in sep.items()})
    figures(meta, raw, roll, sep)


if __name__ == '__main__':
    main()
