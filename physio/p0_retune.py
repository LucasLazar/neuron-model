"""p0: fix J_max for the per-synapse reading, and show what the retuned calcium does.

Target (DESIGN.md): one isolated event from rest, Btot = 44 uM (kappa ~ 20), raises
peak free calcium by 0.5 uM at dt = 0.1 ms with the exact u^2 step (MODEL_ODES.md sec 9).

Run: python physio/p0_retune.py
Writes results/retune.json, results/p0_output.txt and plots/fig_p0_retune.png.
"""
import io
from contextlib import redirect_stdout

import numpy as np
from scipy.optimize import brentq

from physio_core import (BUFFERS, C0, J_MAX_UNTUNED, PLOTS, REFERENCE_BUFFER, RESULTS, SCHEME, TARGET_PEAK_UM,
                         REF, Events, Sim, calcium_trace, kappa, save_json, single_peak, state_trace)

INTERVALS = [2, 5, 10, 20, 50, 100, 200, 500]
AVOGADRO = 6.022e23
SPINE_LITRES = 1e-16          # 0.1 fL


def calibrate():
    J = brentq(lambda j: single_peak(j, REFERENCE_BUFFER) - TARGET_PEAK_UM, J_MAX_UNTUNED, 5000.0, xtol=1e-6)
    return float(J)


def influx_per_event(J):
    """Total calcium entering, uM. With no buffer the calcium ODE is linear, so the
    integral of the excess equals tau_c times the integral of J."""
    c, _ = calcium_trace(J, 0.0, [10.0], 3000.0, dt=0.1)
    tau_c = REF.tau_c
    return float(np.sum(c - C0) * 0.1 / tau_c)


def pair_ratios(J, buffer, single_peak_value, single_integral):
    rows = []
    for gap in INTERVALS:
        c, _ = calcium_trace(J, buffer, [10.0, 10.0 + gap], 3000.0 + gap)
        rows.append(dict(interval_ms=gap, peak_ratio=float((c.max() - C0) / single_peak_value),
                         integral_ratio=float(np.sum(c - C0) * 0.1 / (2 * single_integral))))
    return rows


def main():
    J = calibrate()
    report = dict(J_max=J, J_max_untuned=J_MAX_UNTUNED, scale=J / J_MAX_UNTUNED,
                  target_peak_uM=TARGET_PEAK_UM, reference_buffer=REFERENCE_BUFFER,
                  reference_kappa=kappa(REFERENCE_BUFFER), dt=0.1)
    report['scheme'] = SCHEME
    report['peak_at_reference_dt0.0125'] = single_peak(J, REFERENCE_BUFFER, dt=0.0125)
    report['peak_at_reference_frozen_dt0.1'] = single_peak(J, REFERENCE_BUFFER, scheme='frozen')
    total = influx_per_event(J)
    report['influx_per_event_uM'] = total
    report['ions_per_event_in_0.1fL'] = total * 1e-6 * SPINE_LITRES * AVOGADRO

    trace = state_trace(J, REFERENCE_BUFFER, [10.0], 600.0)
    ref = REF
    report['V_peak_mV'] = float(trace['V'].max())
    report['s_peak'] = float(trace['s'].max())
    report['block_at_V_peak'] = float(1 / (1 + ref.Mg / 3.57 * np.exp(-0.062 * trace['V'].max())))
    report['block_at_rest'] = float(1 / (1 + ref.Mg / 3.57 * np.exp(-0.062 * ref.lif.EL)))

    per_buffer = []
    traces = {}
    for B in BUFFERS:
        row = dict(buffer=B, kappa=kappa(B))
        for name, j in (('untuned', J_MAX_UNTUNED), ('retuned', J)):
            tr = state_trace(j, B, [10.0], 2500.0)
            row[f'{name}_peak_uM'] = float(tr['c'].max() - C0)
            row[f'{name}_time_to_peak_ms'] = float(tr['t'][tr['c'].argmax()] - 10.0)
            excess = tr['c'] - C0
            above = np.flatnonzero(excess > 0.5 * excess.max())
            row[f'{name}_half_width_ms'] = float((above[-1] - above[0]) * 0.1)
            if B > 0:
                rest_occ = tr['b'][0] / B
                row[f'{name}_peak_occupancy'] = float(tr['b'].max() / B)
                row['rest_occupancy'] = float(rest_occ)
            if name == 'retuned':
                traces[B] = tr
        per_buffer.append(row)
    report['single_event'] = per_buffer

    pairs = {}
    for B in (0.0, REFERENCE_BUFFER, 300.0):
        c, _ = calcium_trace(J, B, [10.0], 3000.0)
        pairs[B] = pair_ratios(J, B, c.max() - C0, np.sum(c - C0) * 0.1)
    report['pairs'] = {f'{B:g}': rows for B, rows in pairs.items()}

    trains = {}
    for rate in (4, 32):
        for B in (15.0, 300.0):
            sim = Sim(n=1, dt=0.1, Btot=B, J_max=J, u_quadrature=SCHEME)
            ev = Events([92, rate])
            c = np.concatenate([sim.run(ev.chunk(float(rate), 100.0, 0.1))[0][:, 0] for _ in range(60)])
            trains[(rate, B)] = c
    save_json(RESULTS / 'retune.json', report)
    text = io.StringIO()
    with redirect_stdout(text):
        print_report(report)
    (RESULTS / 'p0_output.txt').write_text(text.getvalue())
    print(text.getvalue())
    figure(report, traces, pairs, trains)


def print_report(r):
    print('p0: J_max retune for the per-synapse reading')
    print('=' * 78)
    print(f"  target: one event from rest raises free calcium by {r['target_peak_uM']} uM at Btot "
          f"{r['reference_buffer']:g} uM (kappa {r['reference_kappa']:.1f}), dt 0.1 ms, u_quadrature={r['scheme']}")
    print(f"  J_max = {r['J_max']:.3f} uM/ms   ({r['scale']:.1f} x untuned {r['J_max_untuned']})")
    print(f"  same J_max at dt 0.0125: peak {r['peak_at_reference_dt0.0125']:.4f} uM "
          f"({r['peak_at_reference_dt0.0125'] / r['target_peak_uM'] - 1:+.1%} vs target)")
    print(f"  same J_max with the historical frozen u^2 step at dt 0.1: peak {r['peak_at_reference_frozen_dt0.1']:.4f} uM "
          f"({r['peak_at_reference_frozen_dt0.1'] / r['target_peak_uM'] - 1:+.1%})")
    print(f"  calcium entering per event: {r['influx_per_event_uM']:.2f} uM total "
          f"= {r['ions_per_event_in_0.1fL']:.0f} ions in a 0.1 fL spine head")
    print(f"  unchanged by the retune: V peak {r['V_peak_mV']:.1f} mV, s peak {r['s_peak']:.3f}, "
          f"Mg unblock {r['block_at_rest']:.4f} at rest -> {r['block_at_V_peak']:.4f} at V peak")
    print()
    print('  SINGLE EVENT from rest, per buffer')
    print('    Btot  kappa | untuned (0.5) peak   retuned peak  ratio | t-peak  half-width | occupancy rest -> peak')
    for row in r['single_event']:
        occ = (f"{row['rest_occupancy']:.3f} -> {row['retuned_peak_occupancy']:.3f}" if row['buffer'] > 0 else '--')
        print(f"   {row['buffer']:5.0f} {row['kappa']:6.1f} | {row['untuned_peak_uM']:14.4f} uM "
              f"{row['retuned_peak_uM']:10.4f} uM {row['retuned_peak_uM'] / row['untuned_peak_uM']:6.1f} | "
              f"{row['retuned_time_to_peak_ms']:5.1f} ms {row['retuned_half_width_ms']:7.1f} ms | {occ}")
    print()
    print('  PAIRS: second event after the interval; ratio to one event (peak) or to two isolated events (integral)')
    for B, rows in r['pairs'].items():
        print(f"    Btot {B:>3s}: " + '  '.join(f"{x['interval_ms']:g}ms {x['peak_ratio']:.2f}/{x['integral_ratio']:.2f}" for x in rows))
    print('    (peak ratio 2 = linear summation of peaks; integral ratio 1 = linear influx)')


def figure(r, traces, pairs, trains):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = plt.cm.viridis(np.linspace(0.05, 0.85, len(BUFFERS)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))

    a = ax[0, 0]
    tr = traces[REFERENCE_BUFFER]
    a.plot(tr['t'] - 10, tr['V'], color='k', lw=1.4, label='dendritic V')
    a.set_xlim(-5, 200); a.set_xlabel('ms after event'); a.set_ylabel('V (mV)')
    b = a.twinx()
    b.plot(tr['t'] - 10, tr['s'], color='tab:red', lw=1.4, label='NMDA occupancy s')
    b.set_ylabel('s', color='tab:red')
    a.set_title(f"(a) one event: voltage and occupancy\n(unchanged by the retune; s peak {r['s_peak']:.2f})", fontsize=10)

    a = ax[0, 1]
    for i, B in enumerate(BUFFERS):
        a.plot(traces[B]['t'] - 10, traces[B]['c'] - C0, color=col[i], lw=1.5, label=f'Btot {B:g} (kappa {kappa(B):.0f})')
    a.axhline(TARGET_PEAK_UM, color='0.5', ls=':', lw=1)
    a.set_xlim(-5, 600); a.set_xlabel('ms after event'); a.set_ylabel('free calcium above rest (uM)')
    a.set_title(f"(b) one event, retuned J_max = {r['J_max']:.1f} uM/ms\n(dotted: 0.5 uM target, met at kappa 20)", fontsize=10)
    a.legend(fontsize=8)

    a = ax[0, 2]
    for i, B in enumerate(BUFFERS):
        if B > 0:
            a.plot(traces[B]['t'] - 10, traces[B]['b'] / B, color=col[i], lw=1.5, label=f'Btot {B:g}')
    a.set_xlim(-5, 1500); a.set_ylim(0, 1); a.set_xlabel('ms after event'); a.set_ylabel('buffer occupancy b / Btot')
    a.set_title('(c) one event: buffer occupancy', fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 0]
    rows = r['single_event']
    x = np.arange(len(rows))
    a.bar(x - 0.2, [q['untuned_peak_uM'] for q in rows], 0.4, color='0.6', label='untuned J_max 0.5')
    a.bar(x + 0.2, [q['retuned_peak_uM'] for q in rows], 0.4, color='tab:blue', label=f"retuned J_max {r['J_max']:.1f}")
    a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels([f"{q['buffer']:g}" for q in rows])
    a.set_xlabel('Btot (uM)'); a.set_ylabel('peak rise for one event (uM)')
    a.set_title(f"(d) peak per buffer: J_max x{r['scale']:.0f}; the peak ratio\nexceeds that where one event saturates the buffer", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 1]
    for i, (B, rows) in enumerate(pairs.items()):
        gaps = [q['interval_ms'] for q in rows]
        a.plot(gaps, [q['peak_ratio'] for q in rows], 'o-', color=plt.cm.viridis(0.1 + 0.35 * i), label=f'peak ratio, Btot {B:g}')
    a.plot(gaps, [q['integral_ratio'] for q in pairs[0.0]], 's--', color='k', label='influx ratio (any buffer)')
    a.axhline(2, color='0.6', lw=0.8); a.axhline(1, color='0.6', lw=0.8)
    a.set_xscale('log'); a.set_xlabel('interval between two events (ms)'); a.set_ylabel('pair / single')
    a.set_title('(e) pairs: peak (vs one event) and total influx\n(vs two isolated events)', fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 2]
    t = np.arange(len(trains[(4, 15.0)])) * 1e-3
    for (rate, B), c in trains.items():
        style = '-' if B == 15.0 else '--'
        color = 'tab:orange' if rate == 32 else 'tab:purple'
        a.plot(t, c - C0, style, color=color, lw=1.0, label=f'{rate} Hz, Btot {B:g}')
    a.set_xlim(0, 6); a.set_xlabel('s'); a.set_ylabel('free calcium above rest (uM)')
    a.set_title('(f) Poisson trains, retuned (same seed per rate)', fontsize=10)
    a.legend(fontsize=8)

    fig.suptitle('p0: J_max retune for the per-synapse reading. Only J_max changes; V, u and s are unchanged.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / 'fig_p0_retune.png', dpi=140)
    print(f"saved {PLOTS / 'fig_p0_retune.png'}")


if __name__ == '__main__':
    main()
