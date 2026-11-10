"""l1 report: checks, tables and figures from results/cells (likelihood/DESIGN.md).

Called by l1_family_sweep.py; `python likelihood/l1_family_sweep.py --replot` reruns it alone.
"""
import io
from contextlib import redirect_stdout
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
import numpy as np  # noqa: E402

from l1_family_sweep import (BUFFERS, CELLS, OBSERVABLES, PLOTS, P1_CHECK, Q, RATES, RESULTS,  # noqa: E402
                             window_ms)   # first: it puts physio/ and the model on the path
import belief_replay  # noqa: E402
import families as F  # noqa: E402
from physio_core import C0, RESULTS as PHYSIO_RESULTS, save_json  # noqa: E402

PRIMARY = 'M2'
LOW_RATES = 8                              # "typical cortical" summary: rates <= 8 Hz
BAND = (Q / 1.5, Q * 1.5)                  # calibrated: both tails within 1.5x of 2.5%
CODE = dict(normal='N', gamma_M='G', gamma='g', lognormal='L', invgauss='I', gengamma='X', tweedie='T')
LABEL = dict(normal='normal (y)', gamma_M='gamma on M (current)', gamma='gamma (y)', lognormal='lognormal (y)',
             invgauss='inverse Gaussian (y)', gengamma='generalized gamma (y)', tweedie='Tweedie (y)')
# fixed categorical slots, the same in every figure
COLOR = dict(gamma_M='#2a78d6', tweedie='#eb6834', gamma='#1baf7a', normal='#eda100',
             lognormal='#e87ba4', invgauss='#008300', gengamma='#6250d6')
LIGHT = {'normal', 'gamma', 'lognormal'}       # fills that need dark ink
BLUES = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#184f95', '#0d366b']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
DIVERGING = LinearSegmentedColormap.from_list('tails', ['#184f95', '#6da7ec', '#f0efec', '#ef8a89', '#b52b2a'])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 9.5, 'axes.titlecolor': INK, 'figure.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False})


def obs_label(o):
    return 'raw calcium (1 ms)' if o == 'raw' else f'M, {o[1:]} s window'


def ok(low, high):
    return BAND[0] <= low <= BAND[1] and BAND[0] <= high <= BAND[1]


def tails(cell, o, who):
    """(low, high) tail or flag rates for a family name, 'layer' or 'tracker'."""
    d = cell['obs'][o]
    src = d['fits'][who] if who in d['fits'] else d.get(who)
    if src is None:
        return None
    if 'tail_low' in src:
        return src['tail_low'][0], src['tail_high'][0]
    return src['low'][0], src['high'][0]


def best_family(cell, o):
    fits = cell['obs'][o]['fits']
    return min(fits, key=lambda n: fits[n]['aic'])


def tail_char(low, high):
    """'+' calibrated; otherwise the worse side: L/H too many below/above, l/h too few."""
    if ok(low, high):
        return '+'
    errors = [(abs(np.log(max(low, 1e-4) / Q)), 'L' if low > Q else 'l'),
              (abs(np.log(max(high, 1e-4) / Q)), 'H' if high > Q else 'h')]
    return max(errors)[1]


def fit_object(summary, name):
    return F.Fit(name, np.array(summary['theta']), summary['scale'], summary['shift'], summary['loglik'],
                 summary['n'], summary['n_censored'])


# ---------------------------------------------------------------- text ----
def checks(cells):
    print('CHECKS')
    worst = F.self_check()
    print(f"  Tweedie density: |mass - 1| {worst['mass']:.1e}, mean {worst['mean']:.1e}, var {worst['var']:.1e} "
          f"(relative), KS vs 200k Monte Carlo {worst['ks']:.4f}, ppf {worst['ppf']:.1e}")
    print(f'  replay vs ExactGammaBelief: max |dU| = {belief_replay.check():.1e}')
    rate, buffer, seed = P1_CHECK
    p1 = json.loads((PHYSIO_RESULTS / 'p1_stats.json').read_text())
    ref = [r['m_skew'] for r in p1['rows']
           if r['tuning'] == 'retuned' and r['rate'] == rate and r['buffer'] == buffer and r['seed'] == seed][0]
    mine = cells[(rate, buffer)]['p1_check']
    print(f'  p1 reproduction ({rate} Hz, Btot {buffer:g}, seed {seed}, first 400 s): M skew {mine:.12f} vs p1 '
          f'{ref:.12f}, |diff| {abs(mine - ref):.1e}')
    n = next(iter(cells.values()))
    print(f"  {len(cells)} cells, {n['n_seeds']} seeds x {n['run_s']:g} s each; per-cell worker time, median "
          f"{np.median([c['sim_s'] for c in cells.values()]):.0f} s simulating and "
          f"{np.median([c['total_s'] for c in cells.values()]):.0f} s in all (wall clock)")


def primary_table(cells):
    o = PRIMARY
    print(f'\nSHAPE AND FAMILIES, {obs_label(o)} (the layer\'s current window)')
    print('  y = M - c0. p^ = skew/CV(y) with seed CI: 0 normal, 1 Poisson, 2 gamma, 3 inverse Gaussian.')
    print('  kurt/CV^2 against Tweedie\'s p(2p-1) at p^. rest = ticks with y < 1 nM. best = lowest AIC; '
          'dAICeff = (AIC - best)/n x n_eff.')
    print('  Tails: % of ticks below the fitted 2.5% / above the fitted 97.5% quantile (ideal 2.5 / 2.5).')
    print('  layer = ExactGammaBelief replayed; tracker = 60 s empirical quantiles. Flag % depress / potentiate.')
    for B in BUFFERS:
        print(f"\n  Btot = {B:g} uM  (kappa {cells[(RATES[0], B)]['kappa']:.1f})")
        print('   rate  rW |  CV(y)   skew      p^ (+-)   kurt/CV2 p(2p-1) | rest% | best dAICeff(G) |'
              ' gamma_M tails |  Tweedie p   lam   alpha |  layer flags | tracker flags')
        for r in RATES:
            d = cells[(r, B)]['obs'][o]
            ph = d['p_hat']
            tw = d['fits']['tweedie']
            gl, gh = tails(cells[(r, B)], o, 'gamma_M')
            ll, lh = tails(cells[(r, B)], o, 'layer')
            tl, th = tails(cells[(r, B)], o, 'tracker')
            p = min(max(ph[0], 0.0), 4.0)
            print(f"  {r:5g} {r * 2:4g} | {d['cv_y'][0]:6.3f} {d['skew'][0]:6.3f}  {ph[0]:5.2f} ({ph[1]:4.2f})  "
                  f"{d['kurt_ratio'][0]:7.2f} {p * (2 * p - 1):7.2f} | {100 * d['censored'][0]:5.1f} | "
                  f"  {CODE[best_family(cells[(r, B)], o)]}  {d['fits']['gamma_M']['d_aic_eff']:8.1f}  | "
                  f"  {100 * gl:5.2f} {100 * gh:5.2f} | {tw['p']:9.3f} {tw['lam']:6.1f} {tw['alpha']:7.2f} | "
                  f"{100 * ll:5.2f} {100 * lh:5.2f} | {100 * tl:5.2f} {100 * th:5.2f}")


def grid(cells, o, value, title):
    print(f'  {title}')
    print('   Btot | ' + ' '.join(f'{r:>4g}' for r in RATES))
    for B in BUFFERS:
        print(f'  {B:5g} | ' + ' '.join(f'{value(cells[(r, B)], o):>4s}' for r in RATES))


def window_grids(cells):
    print('\nBY OBSERVABLE: best family by AIC (letters), and tail calibration of the current likelihood,')
    print('the layer and the tracker. Codes: ' + ', '.join(f'{v} {k}' for k, v in CODE.items()) + '.')
    print("  Tail grids: '+' calibrated (both tails within 1.67-3.75%); otherwise the worse side:")
    print('  L / H = too many ticks below / above the band (over-flagging), l / h = too few (under-flagging).')
    for o in OBSERVABLES:
        print(f'\n {obs_label(o)}')
        grid(cells, o, lambda c, o: CODE[best_family(c, o)], 'best family')
        grid(cells, o, lambda c, o: tail_char(*tails(c, o, 'gamma_M')), 'gamma on M, known parameters')
        if o != 'raw':
            grid(cells, o, lambda c, o: tail_char(*tails(c, o, 'layer')), 'layer as implemented')
            grid(cells, o, lambda c, o: tail_char(*tails(c, o, 'tracker')), '60 s empirical tracker')
        grid(cells, o, lambda c, o: f"{100 * c['obs'][o]['censored'][0]:.0f}", '% of ticks at rest (y < 1 nM)')


def calibration_summary(cells):
    rows = list(F.FAMILIES) + ['layer', 'tracker']
    table = {}
    print('\nCALIBRATED CELLS (both tails within 1.67-3.75%), as a fraction of cells')
    for subset, rates in (('all rates', RATES), (f'rates <= {LOW_RATES} Hz', [r for r in RATES if r <= LOW_RATES])):
        print(f'  {subset} ({len(rates) * len(BUFFERS)} cells)')
        print('  ' + f"{'':24s}" + ''.join(f'{o:>8s}' for o in OBSERVABLES))
        for who in rows:
            line = []
            for o in OBSERVABLES:
                values = [tails(cells[(r, B)], o, who) for r in rates for B in BUFFERS]
                if values[0] is None:
                    line.append(None)
                    continue
                line.append(float(np.mean([ok(*v) for v in values])))
            table[(subset, who)] = line
            label = LABEL.get(who, {'layer': 'layer as implemented', 'tracker': '60 s empirical tracker'}.get(who))
            print('  ' + f'{label:24s}' + ''.join(f'{"--":>8s}' if v is None else f'{v:8.2f}' for v in line))
    return rows, table


def flag_balance(cells):
    print('\nLAYER AS IMPLEMENTED: mean flag rates over the grid (ideal 2.5 / 2.5)')
    for o in OBSERVABLES[1:]:
        low = np.array([[tails(cells[(r, B)], o, 'layer')[0] for r in RATES] for B in BUFFERS])
        high = np.array([[tails(cells[(r, B)], o, 'layer')[1] for r in RATES] for B in BUFFERS])
        print(f'  {o:5s}  depress {100 * low.mean():5.2f}%  potentiate {100 * high.mean():5.2f}%   '
              f'depress > potentiate in {np.mean(low > high) * 100:3.0f}% of cells')


# ------------------------------------------------------------- figures ----
def save(fig, name):
    path = PLOTS / name
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


def fig_shapes(cells):
    shown = [0.5, 1, 2, 4, 8, 16, 32, 64]
    rows, cols = ['raw', 'M0.5', 'M2', 'M8'], [0.0, 15.0, 44.0, 100.0, 300.0]
    edges = np.linspace(-3, 1.6, 185)
    mids = (edges[1:] + edges[:-1]) / 2
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(16, 11), sharex=True, sharey=True)
    for i, o in enumerate(rows):
        for j, B in enumerate(cols):
            a = ax[i, j]
            for k, r in enumerate(shown):
                y = np.load(CELLS / f'{r:g}_{B:g}.npz')[o].astype(float) - C0
                y = y[y >= F.EPS_UM]
                if len(y) < 50:
                    continue
                h, _ = np.histogram(np.log10(y), bins=edges, density=True)
                a.plot(mids, h / h.max(), color=BLUES[k], lw=1.3, label=f'{r:g} Hz')
            a.grid(color=GRID, lw=0.6)
            a.set_ylim(0, 1.08)
            if i == 0:
                a.set_title(f'Btot {B:g} µM (κ {cells[(1, B)]["kappa"]:.0f})')
            if j == 0:
                a.set_ylabel(f'{obs_label(o)}\ndensity of log y, peak = 1')
            if i == len(rows) - 1:
                a.set_xlabel('y = calcium above rest (µM, log10)')
    ax[0, 0].legend(title='input rate', fontsize=7.5, title_fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('What the distributions look like: each curve is one input rate (ticks at rest, y < 1 nM, '
                 'omitted; their share is in fig_l1_mechanism)', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'fig_l1_shapes.png')


def fig_ladder(cells):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    cv = np.geomspace(0.02, 6, 200)
    for a, o in zip(axes.ravel(), OBSERVABLES):
        top = max(4.4, max(cells[(r, B)]['obs'][o]['p_hat'][0] for r in RATES for B in BUFFERS) + 0.4)
        for line, label, style in ((0 * cv, 'normal', '-'), (1 + 0 * cv, 'Poisson', '--'), (2 + 0 * cv, 'gamma', '-'),
                                   (3 + 0 * cv, 'inverse Gaussian', '--'), (3 + cv ** 2, 'lognormal', ':')):
            a.plot(cv, line, color=INK2, lw=0.9, ls=style, zorder=1)
            if label == 'lognormal':     # label where the curve leaves the panel
                a.text(np.sqrt(top - 3.3) * 1.08, top - 0.3, label, color=INK2, fontsize=7, va='center')
            else:
                a.text(0.022, line[0] + 0.08, label, color=INK2, fontsize=7, va='bottom')
        for b, B in enumerate(BUFFERS):
            d = [cells[(r, B)]['obs'][o] for r in RATES]
            x = np.array([v['cv_y'][0] for v in d])
            y = np.array([v['p_hat'][0] for v in d])
            e = np.array([v['p_hat'][1] for v in d])
            a.errorbar(x, y, yerr=e, color=BLUES[b + 1], lw=1.2, marker='o', ms=3.5, elinewidth=0.8,
                       capsize=0, label=f'{B:g} µM', zorder=3)
        a.set_xscale('log')
        a.set_xlim(0.02, 6)
        a.set_ylim(-0.6, top)
        a.grid(color=GRID, lw=0.6)
        a.set_title(obs_label(o))
        a.set_xlabel('CV of the excess y (rate falls to the right)')
    for a in axes.ravel():
        a.set_ylabel('skew / CV  =  Tweedie power p')
    axes[0, 0].legend(title='Btot', fontsize=7.5, title_fontsize=8, frameon=False, loc='upper left', ncol=2)
    fig.suptitle('Which family? Each family forces skew = p·CV: normal 0, Poisson 1, gamma 2, inverse Gaussian 3, '
                 'lognormal 3 + CV²', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'fig_l1_family_ladder.png')


def fig_mechanism(cells):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    rw = np.geomspace(0.2, 600, 200)
    ax[0].plot(rw, np.exp(-rw), color=INK2, lw=1, ls='--', label='e^{-rW}: no event in the window')
    ax[0].axhline(Q, color=INK2, lw=0.8, ls=':')
    ax[0].text(0.22, Q * 1.15, '2.5%: below this a lower tail exists', color=INK2, fontsize=7.5)
    ax[1].axhline(1, color=INK2, lw=1, ls='--')
    ax[1].text(0.22, 1.04, 'Poisson count, fixed area per event', color=INK2, fontsize=7.5)
    ax[2].axhline(1, color=INK2, lw=1, ls='--')
    ax[2].text(0.22, 1.08, 'one jump per input event', color=INK2, fontsize=7.5)
    for b, B in enumerate(BUFFERS):
        x, rest, cvn, lam = [], [], [], []
        for o in OBSERVABLES[1:]:
            W = window_ms(o) / 1000
            for r in RATES:
                d = cells[(r, B)]['obs'][o]
                x.append(r * W)
                rest.append(d['censored'][0])
                cvn.append(d['cv_y'][0] * np.sqrt(r * W))
                lam.append(d['fits']['tweedie']['lam'] / (r * W))
        x = np.array(x)
        for a, y in zip(ax, (rest, cvn, lam)):
            a.scatter(x, y, s=12, color=BLUES[b + 1], edgecolor='white', linewidth=0.4, label=f'{B:g} µM', zorder=3)
    ax[0].set_yscale('symlog', linthresh=1e-3)
    ax[0].set_ylim(-1e-4, 1.1)
    ax[0].set_ylabel('fraction of windows at rest (y < 1 nM)')
    ax[1].set_ylabel('CV(y) × √(rW)')
    ax[1].set_ylim(0, 2.2)
    ax[2].set_yscale('log')
    ax[2].set_ylabel('fitted Tweedie λ / (rW)')
    titles = ('a  windows with no event: a lower tail needs rW > 3.7',
              'b  width against the Poisson count', 'c  Tweedie jumps per window against input events')
    for a, t in zip(ax, titles):
        a.set_xscale('log')
        a.set_xlim(0.2, 800)
        a.set_xlabel('expected input events per window, rW')
        a.grid(color=GRID, lw=0.6)
        a.set_title(t, loc='left')
    ax[0].legend(title='Btot', fontsize=7, title_fontsize=7.5, frameon=False, loc='upper right', ncol=2)
    fig.tight_layout()
    save(fig, 'fig_l1_mechanism.png')


def fig_best(cells):
    names = list(F.FAMILIES)
    cmap = ListedColormap([COLOR[n] for n in names])
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.6))
    for a, o in zip(axes.ravel(), OBSERVABLES):
        idx = np.array([[names.index(best_family(cells[(r, B)], o)) for r in RATES] for B in BUFFERS])
        a.imshow(idx, cmap=cmap, norm=BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names)), aspect='auto',
                 origin='lower')
        for i, B in enumerate(BUFFERS):
            for j, r in enumerate(RATES):
                fits = cells[(r, B)]['obs'][o]['fits']
                runner_up = sorted(f['d_aic_eff'] for f in fits.values())[1]
                if runner_up < 10:      # no clear winner: fade the cell
                    a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='white', alpha=0.6, lw=0))
                mark = CODE[names[idx[i, j]]] + ('' if fits['gamma_M']['d_aic_eff'] < 10 else '•')
                a.text(j, i, mark, ha='center', va='center', fontsize=7,
                       color=INK if names[idx[i, j]] in LIGHT or runner_up < 10 else 'white')
        a.set_xticks(range(len(RATES)), [f'{r:g}' for r in RATES], fontsize=7.5)
        a.set_yticks(range(len(BUFFERS)), [f'{B:g}' for B in BUFFERS], fontsize=7.5)
        a.set_xlabel('input rate (Hz)')
        a.set_ylabel('Btot (µM)')
        a.set_title(obs_label(o))
        a.spines[:].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[n]) for n in names]
    fig.legend(handles, [f'{CODE[n]}  {LABEL[n]}' for n in names], loc='lower center', ncol=7, frameon=False,
               fontsize=8)
    fig.suptitle('Best family by AIC. Faded: the runner-up is within ΔAIC_eff 10 (no clear winner). '
                 '•: the current likelihood (gamma on M) is clearly worse, ΔAIC_eff ≥ 10', fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig_l1_best_family.png')


def fig_calibration(cells, o=PRIMARY):
    rows = ['gamma_M', 'gamma', 'normal', 'lognormal', 'tweedie', 'gengamma', 'layer', 'tracker']
    names = dict(LABEL, layer='layer as implemented (gamma on M, learned)',
                 tracker='60 s empirical quantile tracker')
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 2.25 * len(rows)), constrained_layout=True)
    for i, who in enumerate(rows):
        for j, side in enumerate(('below the 2.5% point (depress)', 'above the 97.5% point (potentiate)')):
            a = axes[i, j]
            v = np.array([[tails(cells[(r, B)], o, who)[j] for r in RATES] for B in BUFFERS])
            im = a.imshow(np.log2(np.maximum(v, 1e-4) / Q), cmap=DIVERGING, vmin=-2.5, vmax=2.5, aspect='auto',
                          origin='lower')
            for bi in range(len(BUFFERS)):
                for ri in range(len(RATES)):
                    val = 100 * v[bi, ri]
                    dark = abs(np.log2(max(v[bi, ri], 1e-4) / Q)) > 1.6
                    a.text(ri, bi, f'{val:.1f}' if val < 9.95 else f'{val:.0f}', ha='center', va='center',
                           fontsize=6.3, color='white' if dark else INK)
            a.set_xticks(range(len(RATES)), [f'{r:g}' for r in RATES], fontsize=7)
            a.set_yticks(range(len(BUFFERS)), [f'{B:g}' for B in BUFFERS], fontsize=7)
            a.spines[:].set_visible(False)
            a.set_title(f'{names[who]}: % of ticks {side}', fontsize=8.5, loc='left')
            if j == 0:
                a.set_ylabel('Btot (µM)', fontsize=8)
            if i == len(rows) - 1:
                a.set_xlabel('input rate (Hz)')
    cbar = fig.colorbar(im, ax=axes, shrink=0.3, pad=0.01)
    cbar.set_ticks(np.log2(np.array([0.625, 1.25, 2.5, 5, 10]) / 2.5), labels=['0.6%', '1.2%', '2.5%', '5%', '10%'])
    cbar.set_label('tail rate (blue: too few flags, red: too many)')
    fig.suptitle(f'Tail calibration, {obs_label(o)}. Ideal is 2.5% on each side; families use known (fitted) '
                 'parameters, the last two rows learn with 60 s memory', fontsize=10.5, color=INK)
    save(fig, f'fig_l1_calibration_{o.replace(".", "p")}.png')


def fig_summary(rows, table):
    names = dict(LABEL, layer='layer as implemented', tracker='60 s empirical tracker')
    subsets = sorted({s for s, _ in table}, key=lambda s: s != 'all rates')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    cmap = LinearSegmentedColormap.from_list('seq', ['#f7f7f5', '#9ec5f4', '#3987e5', '#104281'])
    for a, subset in zip(axes, subsets):
        v = np.array([[np.nan if x is None else x for x in table[(subset, who)]] for who in rows])
        a.imshow(v, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):
                    a.text(j, i, f'{v[i, j]:.2f}', ha='center', va='center', fontsize=8,
                           color='white' if v[i, j] > 0.6 else INK)
        a.set_xticks(range(len(OBSERVABLES)), ['raw'] + [f'M {o[1:]} s' for o in OBSERVABLES[1:]])
        a.set_yticks(range(len(rows)), [names[w] for w in rows])
        a.set_title(f'fraction of cells calibrated, {subset}')
        a.spines[:].set_visible(False)
    fig.tight_layout()
    save(fig, 'fig_l1_calibration_summary.png')


def fig_qq(cells):
    picks = [(1, 0.0), (2, 44.0), (8, 44.0), (32, 300.0)]
    shown = ['gamma_M', 'gamma', 'normal', 'tweedie']
    probs = np.concatenate((np.geomspace(0.002, 0.05, 12), np.linspace(0.07, 0.93, 25), 1 - np.geomspace(0.05, 0.002, 12)))
    fig, axes = plt.subplots(1, len(picks), figsize=(16, 4.4))
    for a, (r, B) in zip(axes, picks):
        y = np.load(CELLS / f'{r:g}_{B:g}.npz')[PRIMARY].astype(float) - C0
        emp = np.quantile(y, probs)
        m, s = y.mean(), y.std()
        lim = [(emp[0] - m) / s - 0.3, (emp[-1] - m) / s + 0.3]
        a.plot(lim, lim, color=INK2, lw=0.8, ls='--', zorder=1)
        for name in shown:
            f = fit_object(cells[(r, B)]['obs'][PRIMARY]['fits'][name], name)
            fq = f.ppf(probs)
            a.plot((fq - m) / s, (emp - m) / s, color=COLOR[name], lw=1.4, marker='o', ms=2.6, label=LABEL[name])
            lo, hi = f.ppf([Q, 1 - Q])
            a.plot([(lo - m) / s, (hi - m) / s], np.quantile((y - m) / s, [Q, 1 - Q]), ls='none', marker='s',
                   ms=6, mfc='none', mec=COLOR[name])
        a.set_title(f'{r:g} Hz, Btot {B:g} µM, {obs_label(PRIMARY)}')
        a.set_xlabel('fitted quantile (sd from the mean)')
        a.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel('data quantile (sd from the mean)')
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper left')
    fig.suptitle('QQ against the fitted families; squares mark the 2.5% and 97.5% points the layer flags at '
                 '(on the dashed line = calibrated)', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'fig_l1_qq.png')


def run(cells):
    expected = len(RATES) * len(BUFFERS)
    if len(cells) < expected:
        raise SystemExit(f'only {len(cells)} of {expected} cells in {CELLS}; run the sweep first')
    PLOTS.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print('l1: calcium distribution families over rate x buffer x window, per-synapse reading')
        print('=' * 100)
        checks(cells)
        primary_table(cells)
        window_grids(cells)
        rows, table = calibration_summary(cells)
        flag_balance(cells)
    text = buffer.getvalue()
    print(text)
    (RESULTS / 'l1_output.txt').write_text(text)
    save_json(RESULTS / 'l1_rows.json', [dict(rate=c['rate'], buffer=c['buffer'], kappa=c['kappa'], obs=c['obs'])
                                         for c in cells.values()])
    fig_shapes(cells)
    fig_ladder(cells)
    fig_mechanism(cells)
    fig_best(cells)
    fig_calibration(cells, PRIMARY)
    fig_calibration(cells, 'M8')
    fig_summary(rows, table)
    fig_qq(cells)
