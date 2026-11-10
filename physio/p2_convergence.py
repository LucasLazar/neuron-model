"""p2: time-step convergence in the physiological domain, at the retuned J_max.

A: order against the independent Radau reference (tests/reference_solver.py, which
   shares no code with the simulator), with its J_max set to the retuned value.
B: stationary quantities vs dt, paired by seed (continuous-time input, so every dt
   sees the same spikes). The reference is the Richardson extrapolation from
   dt = 0.025 and 0.0125, assuming the first order measured in A.
The production step is u_quadrature="exact"; the historical frozen step is run
alongside (every dt in A, dt = 0.1 in B) for comparison.

Run: python physio/p2_convergence.py [--workers 3] [--blocks A,B] [--replot]
Writes results/p2_A.json, results/p2_B.json, results/p2_output.txt, plots/fig_p2_convergence.png.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
from contextlib import redirect_stdout
import json
import sys
import time

import numpy as np

from physio_core import (C0, PLOTS, RESULTS, ROOT, WEIGHT, Sim, auc, mean_ci, retuned_j_max, rolling_mean, save_json,
                         stationary_run)

sys.path.insert(0, str(ROOT / 'tests'))   # reference_solver.py

A_CASES = [('isolated events, 2 Hz, Btot 44', 2, 44.0, 4000.0),
           ('8 Hz, Btot 15 (buffer saturates)', 8, 15.0, 3000.0),
           ('32 Hz, no buffer', 32, 0.0, 2000.0),
           ('32 Hz, Btot 300', 32, 300.0, 2000.0)]
A_DTS = [.2, .1, .05, .025, .0125, .00625]

B_RATES = [1, 4, 16, 64]
B_BUFFERS = [0.0, 15.0, 44.0, 300.0]
B_LEVELS = [(.2, 'exact'), (.1, 'exact'), (.05, 'exact'), (.025, 'exact'), (.0125, 'exact'), (.1, 'frozen')]
B_SEEDS = 4
B_RUN_S = 200
QUANTITIES = [('mean_excess', 'mean excess'), ('sd', 'sd, 1 ms'), ('p99_excess', '99th pct excess'),
              ('m_sd', 'sd of M'), ('m_cv', 'CV of M')]


def level_name(dt, scheme):
    return f"dt{dt:g}" + ('_frozen' if scheme == 'frozen' else '')


# ---------------- A ----------------
def a_case(args):
    label, rate, capacity, duration, J = args
    import reference_solver as ref
    ref.P['J_max'] = J
    rng = np.random.default_rng([77, rate, int(capacity)])
    slots = np.flatnonzero(rng.random(int(duration / .2)) < rate * .2 / 1000)
    events = (slots * .2)[(slots * .2) > 5.0]
    started = time.perf_counter()
    _, states, spikes = ref.reference(events, duration, capacity)
    seconds = time.perf_counter() - started
    c_true = states[:, ref.C]
    row = dict(case=label, rate_hz=rate, buffer=capacity, duration_ms=duration, events=len(events),
               reference_spikes=len(spikes), reference_seconds=seconds, mean_excess=float(c_true.mean() - C0),
               peak_excess=float(c_true.max() - C0), levels=[])
    for scheme in ('exact', 'frozen'):
        for dt in A_DTS:
            sim = Sim(n=1, dt=dt, Btot=capacity, u_quadrature=scheme, J_max=J)
            w = np.zeros((int(round(duration / dt)), 1))
            for t in events:
                w[int(round(t / dt)), 0] += WEIGHT
            calcium, discrete_spikes = sim.run(w)
            calcium = calcium[:, 0]
            row['levels'].append(dict(dt=dt, scheme=scheme,
                                      max_abs_error=float(np.max(np.abs(calcium - c_true))),
                                      max_error_over_range=float(np.max(np.abs(calcium - c_true)) / np.ptp(c_true)),
                                      mean_excess_error=float((calcium.mean() - c_true.mean()) / (c_true.mean() - C0)),
                                      peak_error=float((calcium.max() - c_true.max()) / (c_true.max() - C0)),
                                      spikes=len(discrete_spikes)))
    return row


# ---------------- B ----------------
def b_job(args):
    J, rate, buffer, dt, scheme, seed = args
    c, _, spikes = stationary_run(J, buffer, rate, seed, B_RUN_S, dt=dt, scheme=scheme, tag=93)
    m = rolling_mean(c)
    excess = c - C0
    return dict(rate=rate, buffer=buffer, dt=dt, scheme=scheme, seed=seed, spikes=spikes,
                mean_excess=float(excess.mean()), sd=float(c.std()), p99_excess=float(np.quantile(excess, .99)),
                m_sd=float(m.std()), m_cv=float(m.std() / (m.mean() - C0)),
                raw=c[::25].astype(np.float32), m=m[::100].astype(np.float32))


def sample_key(row, kind):
    return f"{kind}:{row['rate']:g}|{row['buffer']:g}|{level_name(row['dt'], row['scheme'])}|{row['seed']}"


def load_b():
    path = RESULTS / 'p2_B.json'
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    samples = np.load(RESULTS / 'p2_B_samples.npz')
    for row in rows:
        row['raw'], row['m'] = samples[sample_key(row, 'raw')], samples[sample_key(row, 'm')]
    return rows


def run_a(J, workers):
    print(f"  A: {len(A_CASES)} reference cases at J_max {J:.3f}", flush=True)
    with ProcessPoolExecutor(min(workers, len(A_CASES))) as pool:
        rows = list(pool.map(a_case, [case + (J,) for case in A_CASES]))
    save_json(RESULTS / 'p2_A.json', rows)


def run_b(J, workers):
    jobs = [(J, r, B, dt, scheme, s) for r in B_RATES for B in B_BUFFERS for dt, scheme in B_LEVELS for s in range(B_SEEDS)]
    # finest steps first so the long jobs do not trail at the end
    jobs.sort(key=lambda j: j[3])
    print(f"  B: {len(jobs)} jobs, {B_SEEDS} seeds x {B_RUN_S} s", flush=True)
    started, rows = time.time(), []
    with ProcessPoolExecutor(workers) as pool:
        for k, row in enumerate(pool.map(b_job, jobs)):
            rows.append(row)
            if (k + 1) % 32 == 0:
                print(f"    {k + 1}/{len(jobs)}  [{time.time() - started:.0f}s]", flush=True)
    np.savez_compressed(RESULTS / 'p2_B_samples.npz', **{sample_key(r, kind): r[kind] for r in rows for kind in ('raw', 'm')})
    save_json(RESULTS / 'p2_B.json', [{k: v for k, v in r.items() if k not in ('raw', 'm')} for r in rows])


# ---------------- analysis ----------------
def b_table(rows):
    """values[(rate, buffer, level)] -> list over seeds of row dicts, seed-ordered."""
    table = {}
    for r in rows:
        table.setdefault((r['rate'], r['buffer'], level_name(r['dt'], r['scheme'])), {})[r['seed']] = r
    return {k: [v[s] for s in sorted(v)] for k, v in table.items()}


def richardson(table, rate, buffer, field):
    fine = np.array([x[field] for x in table[(rate, buffer, 'dt0.0125')]])
    coarse = np.array([x[field] for x in table[(rate, buffer, 'dt0.025')]])
    return 2 * fine - coarse


def relative_error(table, rate, buffer, level, field):
    ref = richardson(table, rate, buffer, field)
    x = np.array([v[field] for v in table[(rate, buffer, level)]])
    return mean_ci((x - ref) / np.mean(ref))


def verdict(mean, ci, band=0.05):
    if abs(mean) + ci < band:
        return 'pass'
    if abs(mean) < band:
        return 'marginal'
    return 'FAIL'


def b_auc(table, buffer, level, kind):
    """AUC for 4x rate steps at one dt level, seed-paired mean."""
    out = []
    for lo, hi in zip(B_RATES[:-1], B_RATES[1:]):
        per_seed = [auc(np.array(h[kind]), np.array(l[kind]))
                    for h, l in zip(table[(hi, buffer, level)], table[(lo, buffer, level)])]
        out.append(np.array(per_seed))
    return out


def report(a_rows, b_rows):
    print('p2: time-step convergence at physiological rates, retuned J_max')
    print('=' * 78)
    if a_rows:
        print('\n  A: against the independent Radau reference (rtol 1e-10), events on the 0.2 ms grid')
        for row in a_rows:
            print(f"\n  {row['case']}: {row['events']} events, {row['reference_spikes']} reference spikes, "
                  f"mean excess {row['mean_excess']:.4f} uM, peak excess {row['peak_excess']:.3f} uM  "
                  f"[reference {row['reference_seconds']:.0f}s]")
            print('        dt | exact (production): max|dc|  order  mean-excess err  peak err |'
                  ' frozen (historical): max|dc|  order  mean-excess err  peak err')
            exact = [x for x in row['levels'] if x['scheme'] == 'exact']
            frozen = [x for x in row['levels'] if x['scheme'] == 'frozen']
            for i, (p, e) in enumerate(zip(exact, frozen)):
                op = np.log2(exact[i - 1]['max_abs_error'] / p['max_abs_error']) if i else np.nan
                oe = np.log2(frozen[i - 1]['max_abs_error'] / e['max_abs_error']) if i else np.nan
                print(f"   {p['dt']:7.5f} | {p['max_abs_error']:19.3e} {op:6.2f} {p['mean_excess_error']:+16.2%} "
                      f"{p['peak_error']:+9.2%} | {e['max_abs_error']:18.3e} {oe:6.2f} {e['mean_excess_error']:+16.2%} "
                      f"{e['peak_error']:+9.2%}")
    if b_rows:
        table = b_table(b_rows)
        print('\n  B: stationary quantities, relative error against Richardson(dt 0.025, 0.0125), '
              f'{B_SEEDS} seeds x {B_RUN_S} s, paired')
        print('     pass = whole 95% CI inside +-5%; marginal = point estimate inside only')
        levels = ['dt0.2', 'dt0.1', 'dt0.05', 'dt0.025', 'dt0.1_frozen']
        for field, label in QUANTITIES:
            print(f"\n   {label}")
            print('    rate  Btot | ' + ' '.join(f"{lv:>17s}" for lv in levels) + ' | dt0.1 verdict')
            for r in B_RATES:
                for B in B_BUFFERS:
                    cells = [relative_error(table, r, B, lv, field) for lv in levels]
                    v = verdict(*cells[1])
                    print(f"    {r:4g} {B:5g} | " + ' '.join(f"{m:+8.2%}+-{c:6.2%}" for m, c in cells) + f" | {v}")
        print('\n   AUC for 4x rate steps (seed mean); dt 0.1 vs dt 0.0125, tolerance 0.01 absolute')
        for kind, label in (('raw', '1 ms calcium'), ('m', 'M')):
            print(f"    {label}")
            for B in B_BUFFERS:
                coarse = b_auc(table, B, 'dt0.1', kind)
                fine = b_auc(table, B, 'dt0.0125', kind)
                parts = []
                for (lo, hi), a, b in zip(zip(B_RATES[:-1], B_RATES[1:]), coarse, fine):
                    d, ci = mean_ci(a - b)
                    parts.append(f"{lo:g}->{hi:g} Hz {a.mean():.3f} vs {b.mean():.3f} ({d:+.4f}+-{ci:.4f})")
                print(f"     Btot {B:5g}: " + '; '.join(parts))
        spikes = sum(x['spikes'] for x in b_rows)
        print(f"\n   somatic spikes over all B runs: {spikes}")


def figure(a_rows, b_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.8))
    a = ax[0, 0]
    colors = plt.cm.tab10(np.arange(len(a_rows)))
    for i, row in enumerate(a_rows):
        for scheme, style in (('exact', 'o-'), ('frozen', 's--')):
            lv = [x for x in row['levels'] if x['scheme'] == scheme]
            a.plot([x['dt'] for x in lv], [x['max_error_over_range'] for x in lv], style, color=colors[i], ms=4,
                   label=row['case'] if scheme == 'exact' else None)
    dts = np.array(A_DTS)
    a.plot(dts, 0.1 * dts / dts[0], ':', color='k', label='slope 1')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('dt (ms)'); a.set_ylabel('max |c - c_ref| / range')
    a.set_title('(a) A: error vs Radau reference\n(solid: exact u^2, production; dashed: frozen, historical)', fontsize=10.5)
    a.legend(fontsize=7)

    table = b_table(b_rows)
    rcol = plt.cm.plasma(np.linspace(0.05, 0.85, len(B_RATES)))
    prod_levels = ['dt0.2', 'dt0.1', 'dt0.05', 'dt0.025']
    for a, field, title in ((ax[0, 1], 'mean_excess', '(b) B: mean excess vs dt'),
                            (ax[0, 2], 'm_sd', '(c) B: sd of M vs dt')):
        for i, r in enumerate(B_RATES):
            for B, alpha in zip(B_BUFFERS, (0.35, 0.35, 1.0, 0.35)):
                y = [relative_error(table, r, B, lv, field)[0] for lv in prod_levels]
                a.plot([.2, .1, .05, .025], y, 'o-', color=rcol[i], alpha=alpha, ms=3,
                       label=f'{r:g} Hz' if B == 44.0 else None)
            ye = relative_error(table, r, 44.0, 'dt0.1_frozen', field)[0]
            a.plot([.1], [ye], 'x', color=rcol[i], ms=8)
        a.axhspan(-0.05, 0.05, color='0.9', zorder=0)
        a.set_xscale('log'); a.set_xlabel('dt (ms)'); a.set_ylabel('relative error vs dt -> 0')
        a.set_xticks([.025, .05, .1, .2]); a.set_xticklabels(['0.025', '0.05', '0.1', '0.2'])
        a.minorticks_off()
        a.set_title(title + '\n(Btot 44 bold, others faint; x: frozen step at 0.1)', fontsize=10.5)
        a.legend(fontsize=7.5)

    for a, field, title in ((ax[1, 0], 'mean_excess', '(d) dt 0.1 error: mean excess'),
                            (ax[1, 1], 'sd', '(e) dt 0.1 error: sd of 1 ms calcium'),
                            (ax[1, 2], 'm_sd', '(f) dt 0.1 error: sd of M')):
        grid = np.array([[relative_error(table, r, B, 'dt0.1', field)[0] for B in B_BUFFERS] for r in B_RATES])
        im = a.imshow(grid * 100, cmap='RdBu_r', vmin=-5, vmax=5, aspect='auto')
        for i in range(len(B_RATES)):
            for j in range(len(B_BUFFERS)):
                a.text(j, i, f'{grid[i, j]:+.1%}', ha='center', va='center', fontsize=8.5)
        a.set_xticks(range(len(B_BUFFERS))); a.set_xticklabels([f'{B:g}' for B in B_BUFFERS])
        a.set_yticks(range(len(B_RATES))); a.set_yticklabels([f'{r:g} Hz' for r in B_RATES])
        a.set_xlabel('Btot (uM)'); a.set_title(title, fontsize=10.5)
        fig.colorbar(im, ax=a, label='%')
    fig.suptitle('p2: convergence at physiological rates, retuned J_max (exact u^2 step unless marked)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / 'fig_p2_convergence.png', dpi=140)
    print(f"saved {PLOTS / 'fig_p2_convergence.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--blocks', default='A,B')
    parser.add_argument('--replot', action='store_true')
    args = parser.parse_args()
    J = retuned_j_max()
    if not args.replot:
        blocks = args.blocks.split(',')
        if 'A' in blocks:
            run_a(J, args.workers)
        if 'B' in blocks:
            run_b(J, args.workers)
    a_rows = json.loads((RESULTS / 'p2_A.json').read_text()) if (RESULTS / 'p2_A.json').exists() else []
    b_rows = load_b()
    text = io.StringIO()
    with redirect_stdout(text):
        report(a_rows, b_rows)
    (RESULTS / 'p2_output.txt').write_text(text.getvalue())
    print(text.getvalue())
    if a_rows and b_rows:
        figure(a_rows, b_rows)


if __name__ == '__main__':
    main()
