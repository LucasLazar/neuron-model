# Neuron-centric model: handover package

A dendritic LIF neuron with:
- a split NMDA gate: an AND of glutamate and depolarisation, with back-propagating APs;
- buffered calcium in each compartment;
- a Bayesian plasticity layer, which learns each compartment's normal calcium and
  moves its AMPA conductance when calcium leaves that range.

It is a first computational variant of Scheler's neuron-centric model, with calcium
handling (buffer, extrusion) as the internal parameters under study.

This folder is self-contained. It holds:
- the current model;
- a compiled simulator that reproduces it exactly;
- the verification;
- the three most recent studies, which you can rerun.

Earlier history lives in the research repository; [STATUS.md](STATUS.md) summarises it.

## Read in this order

1. [MODEL_ODES.md](MODEL_ODES.md): the equations, defaults and numerical scheme.
   [ODE_sheet.pdf](ODE_sheet.pdf) is the same content typeset; regenerate it with
   `python make_ode_sheet.py`.
2. [STATUS.md](STATUS.md): what is established, what is open, and where the project is
   heading.
3. [VERIFICATION.md](VERIFICATION.md): what has been checked, and how to rerun it.
4. [physio/FINDINGS.md](physio/FINDINGS.md), [branch/FINDINGS.md](branch/FINDINGS.md)
   and [likelihood/FINDINGS.md](likelihood/FINDINGS.md): the three current studies.

## Setup

You need Python 3.9 or later, numpy, scipy and matplotlib, plus a C compiler available as
`cc` (Xcode command-line tools on macOS, gcc on Linux). The kernel compiles itself on
first import.

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m unittest discover -s tests -v      # 50 tests, about 1 minute
```

All commands run from this folder.

## Layout

| Path | Contents |
|---|---|
| `neuron/model.py` | `NMDANeuron`: the reference implementation of MODEL_ODES.md |
| `neuron/lif.py` | `DendriticLIF`: voltage, AMPA, optional somatic inhibition |
| `neuron/calcium.py` | exact-binding Strang step for buffered calcium |
| `neuron/plasticity.py` | rolling estimator, exact gamma belief, conductance rule |
| `neuron/fast.py`, `neuron/kernel.c` | `Sim` and `PlasticSim`: compiled equivalent, several hundred times faster |
| `tests/` | unit and equivalence tests; `reference_solver.py` is an independent Radau solver of the ODEs |
| `physio/` | per-synapse reading at 1-64 Hz: calibration (p0), rate x buffer sweep (p1), convergence (p2) |
| `branch/` | branch reading (1000 pooled synapses per compartment): gate check (b0), firing regime (b1), rate-calcium relation (b2) |
| `likelihood/` | shape of calcium and M over rate x buffer x window, and candidate likelihoods for the Bayesian layer (l1); `families.py` has a Tweedie density, and `belief_replay.py` a vectorized, exact replay of `ExactGammaBelief` |

Each study folder has a `DESIGN.md` (written before the runs), a `FINDINGS.md`,
scripts, `results/` and `plots/`.

## Using the model

Defaults are the current calibration:
- per-synapse J_max = 78.3 µM/ms;
- B_tot = 44 µM;
- exact u² step;
- dt = 0.1 ms;
- plasticity on.

```python
import numpy as np
from neuron import NMDANeuron
from neuron.fast import Sim, PlasticSim

# One synapse at 8 Hz for 2 s; each event has AMPA weight 12 (x gbar 1 nS), as in physio/.
w = np.random.default_rng(0).poisson(8 * 1e-4, (20000, 1)) * 12.0

r = NMDANeuron(n_dend=1, plasticity_mode='off').run(2000, w, record_ms=1)
r['c'][:, 0]            # free calcium, uM, one sample per ms; also 'V', 's', 'b', 'spikes', ...

calcium, spikes = Sim(n=1).run(w)              # the same run, compiled: identical to 1e-9

# A neuron of 3 branches x 1000 synapses with plasticity, compiled. w holds event counts per branch.
sim = PlasticSim(n=3, pooled_gate=True, n_syn=1000.0, w_syn=1.0)
counts = np.random.default_rng(1).poisson(3.0 * 1000 * 1e-4, (20000, 3)).astype(float)
calcium, spikes, ticks = sim.run(counts)       # ticks: drive, percentile, gbar... every 100 ms
```

`run()` continues from the current state; `reset_state()` restarts. Units are ms, mV,
nS, pF, pA and µM.

Inputs:
- **Per-synapse mode:** `w` is the AMPA weight per step. Any positive value is one
  glutamate release.
- **Pooled mode:** `w` is the event count per step.
- **Somatic inhibition:** `inh_in` (Python) or `w_inh` (compiled), in nS per step.

For realistic irregular firing in the branch model, use correlated input:
`branch/branch_core.py` has `Background`, a shared Ornstein-Uhlenbeck rate modulation.

The bAP-off counterfactual is `sim.params.A_f = sim.params.A_s = 0`. Spikes are
unchanged, so the difference in calcium is what bAPs add.

## Reproducing the studies

| Command | Takes* | Writes |
|---|---|---|
| `python physio/p0_retune.py` | 15 s | `results/retune.json`: the J_max fit used by everything after it |
| `python physio/p1_rate_buffer_sweep.py` | 5 min | rate x buffer sweep, AUC tables, panel-A plots |
| `python physio/p2_convergence.py` | 4 min | dt convergence against the Radau reference |
| `python branch/b0_pooled_gate_check.py` | 1 min | pooled gate vs 200 explicit synapses |
| `python branch/b1_firing_regime.py` | 25 s | firing regime: correlation x inhibition |
| `python branch/b2_relation.py` | 1.5 min | rate-calcium relation with bAP-off twins |
| `python likelihood/l1_family_sweep.py` | 1.5 h** | rate x buffer x window distributions, family fits, layer flag rates; resumes from `results/cells/`; `--replot` redraws only |

\* Measured on a 2017 MacBook Pro (dual-core i5), 3 worker processes. \*\* 4 worker processes.

Results are deterministic. Every run is seeded, and paired arms see identical input.
With the library versions in requirements.txt, the stored results reproduce bit for bit.
Other versions of numpy may change random streams, so results would then agree
statistically but not exactly. The large `*_samples.npz` files are left out (see
`.gitignore`); each script regenerates them, and the KDE plots need them.

## Ground rules this project has kept

- **Specification first.** MODEL_ODES.md is the specification; the code implements it.
  A change to the model goes into MODEL_ODES.md (and `make_ode_sheet.py`) in the same
  step as the code.
- **The Python model is the reference.** The kernel is an exact port of it. If you
  change `model.py`, change `kernel.c` and `fast.py` to match, and run
  `tests/test_kernel.py`.
- **Studies follow a fixed pattern.** Write a DESIGN.md with pre-set pass criteria
  before running. Then the script writes JSON and text results and plots, and
  FINDINGS.md interprets them.
- **Uncertainty comes from independent seeds.** Arms that share a seed are paired
  comparisons, never extra samples.
