"""l1: calcium distribution families over rate x buffer x window (likelihood/DESIGN.md).

Every combination of 14 physiological rates and 7 buffers, 12 seeds x 1000 s each.
Raw calcium and the rolling mean M over 0.5-8 s windows are read at the layer's
100 ms ticks. Per cell and observable:
- the shape moments;
- maximum likelihood fits of seven candidate families (families.py), with their 2.5%
  tail rates;
- the flag rates of the current layer, replayed exactly (belief_replay.py);
- the flag rates of a model-free 60 s quantile tracker.

Run: python likelihood/l1_family_sweep.py [--workers 4] [--replot]
Each cell is written to results/cells/ as it finishes, so an interrupted run resumes.
The report writes results/l1_rows.json, results/l1_output.txt and plots/fig_l1_*.png.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'physio'))
from physio_core import C0, kappa, mean_ci, retuned_j_max, rolling_mean, save_json, stationary_run, tau_int  # noqa: E402
import belief_replay  # noqa: E402
import families as F  # noqa: E402
from neuron.plasticity import PlasticityConfig  # noqa: E402

RESULTS, PLOTS, CELLS = HERE / 'results', HERE / 'plots', HERE / 'results' / 'cells'
RATES = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
BUFFERS = [0.0, 5.0, 15.0, 44.0, 100.0, 200.0, 300.0]
WINDOWS_MS = [500, 1000, 2000, 4000, 8000]
OBSERVABLES = ['raw'] + [f'M{w / 1000:g}' for w in WINDOWS_MS]
N_SEEDS = int(os.environ.get('L1_SEEDS', 12))
RUN_S = float(os.environ.get('L1_RUN_S', 1000))
TICK_MS, FIRST_TICK_MS = 100, 8000      # every observable read at the same instants
WARMUP_TICKS = 600                      # the layer's 60 s warm-up, not scored
MEMORY_TICKS = 600                      # the model-free tracker's 60 s memory
Q = 0.025                               # the layer's tail, PlasticityConfig.tail
STORE_SEEDS = 2                         # ticks kept per cell for figures
P1_CHECK = (4, 44.0, 0)                 # first 400 s are p1's run exactly


def window_ms(observable):
    return None if observable == 'raw' else int(round(float(observable[1:]) * 1000))


def ticks(c):
    """Raw calcium and every M_W at ticks every 100 ms from t = 8 s (c is 1 ms samples)."""
    cs = np.concatenate(([0.0], np.cumsum(c)))
    k = np.arange(FIRST_TICK_MS - 1, len(c), TICK_MS)
    out = {'raw': c[k]}
    for w in WINDOWS_MS:
        out[f'M{w / 1000:g}'] = (cs[k + 1] - cs[k + 1 - w]) / w
    return out


def flag_rates(u):
    return float(np.mean(u < Q)), float(np.mean(u > 1 - Q))


def empirical_flags(x):
    """Each tick scored against the 2.5% and 97.5% quantiles of the previous 60 s of ticks."""
    past = np.lib.stride_tricks.sliding_window_view(x[:-1], MEMORY_TICKS)
    lo, hi = np.quantile(past, [Q, 1 - Q], axis=1)
    now = x[MEMORY_TICKS:]
    return float(np.mean(now < lo)), float(np.mean(now > hi))


def ci(values):
    m, h = mean_ci(values)
    return [m, h]


def analyse(observable, per_seed, rng):
    """Shape, family fits with tail rates, and learned flag rates for one observable."""
    ys = [v - C0 for v in per_seed]
    seed = dict(mean=[v.mean() for v in per_seed], mean_y=[y.mean() for y in ys],
                cv_y=[y.std() / y.mean() for y in ys], cv_m=[v.std() / v.mean() for v in per_seed],
                skew=[stats.skew(v) for v in per_seed], exkurt=[stats.kurtosis(v) for v in per_seed],
                censored=[np.mean(y < F.EPS_UM) for y in ys])
    seed['p_hat'] = [s / c for s, c in zip(seed['skew'], seed['cv_y'])]
    seed['kurt_ratio'] = [k / c ** 2 for k, c in zip(seed['exkurt'], seed['cv_y'])]
    tau, _ = tau_int(ys)
    n_ticks = sum(len(y) for y in ys)
    out = {k: ci(v) for k, v in seed.items()}
    out.update(tau_int_ticks=float(tau), n_ticks=n_ticks, n_eff=float(n_ticks / tau))

    pooled = np.concatenate(ys)
    fits = F.fit_all(pooled, C0, rng)
    best = min(f.aic for f in fits.values())
    out['fits'] = {}
    for name, f in fits.items():
        lo, hi = f.ppf([Q, 1 - Q])
        low, high = [np.mean(y < lo) for y in ys], [np.mean(y > hi) for y in ys]
        d_aic = (f.aic - best) / f.n
        out['fits'][name] = dict(f.summary(), d_aic_per_n=d_aic, d_aic_eff=d_aic * out['n_eff'],
                                 q_low=float(lo), q_high=float(hi), tail_low=ci(low), tail_high=ci(high))

    w = window_ms(observable)
    if w is not None:
        config = PlasticityConfig(evidence_ms=float(w))
        layer = [flag_rates(belief_replay.percentiles(v, config, start=WARMUP_TICKS)[WARMUP_TICKS:]) for v in per_seed]
        tracker = [empirical_flags(v) for v in per_seed]
        out['layer'] = dict(low=ci([a for a, _ in layer]), high=ci([b for _, b in layer]))
        out['tracker'] = dict(low=ci([a for a, _ in tracker]), high=ci([b for _, b in tracker]))
    return out


def job(cell):
    rate, buffer = cell
    started = time.time()
    J = retuned_j_max()
    series, check = {o: [] for o in OBSERVABLES}, None
    for seed in range(N_SEEDS):
        c, _, spikes = stationary_run(J, buffer, rate, seed, RUN_S)
        assert spikes == 0, 'the soma should be silent in the per-synapse reading'
        if (rate, buffer, seed) == P1_CHECK and RUN_S >= 400:
            check = float(stats.skew(rolling_mean(c[:400_000])))
        for o, v in ticks(c).items():
            series[o].append(v)
    sim_s = time.time() - started
    rng = np.random.default_rng([int(rate * 10), int(buffer)])
    result = dict(rate=rate, buffer=buffer, kappa=kappa(buffer), n_seeds=N_SEEDS, run_s=RUN_S, p1_check=check,
                  obs={o: analyse(o, series[o], rng) for o in OBSERVABLES})
    result.update(sim_s=sim_s, total_s=time.time() - started)
    stem = f'{rate:g}_{buffer:g}'
    save_json(CELLS / (stem + '.json'), result)
    np.savez_compressed(CELLS / (stem + '.npz'), **{o: np.concatenate(series[o][:STORE_SEEDS]).astype(np.float32)
                                                     for o in OBSERVABLES})
    return rate, buffer, sim_s, result['total_s']


def simulate(workers):
    CELLS.mkdir(parents=True, exist_ok=True)
    todo = [(r, B) for B in reversed(BUFFERS) for r in reversed(RATES)
            if not (CELLS / f'{r:g}_{B:g}.json').exists()]
    print(f'  {len(todo)} of {len(RATES) * len(BUFFERS)} cells to run: {N_SEEDS} seeds x {RUN_S:g} s; '
          f'workers {workers}', flush=True)
    started = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, (r, B, sim_s, total_s) in enumerate(pool.map(job, todo)):
            print(f'    {k + 1}/{len(todo)}  {r:g} Hz  Btot {B:g}  sim {sim_s:.0f}s  total {total_s:.0f}s  '
                  f'[{time.time() - started:.0f}s]', flush=True)
    print(f'  done in {time.time() - started:.0f}s', flush=True)


def load_cells():
    cells = {}
    for path in sorted(CELLS.glob('*.json')):
        cell = json.loads(path.read_text())
        cells[(cell['rate'], cell['buffer'])] = cell
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--replot', action='store_true')
    args = parser.parse_args()
    if not args.replot:
        print('l1: running', flush=True)
        simulate(args.workers)
    import l1_report
    l1_report.run(load_cells())


if __name__ == '__main__':
    main()
