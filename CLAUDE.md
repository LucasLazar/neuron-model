# Notes for Claude working on this repository

This repository is the working copy of a computational neuron model: a dendritic LIF
with an NMDA AND gate, buffered calcium and a Bayesian plasticity layer. It is a first
variant of Scheler's neuron-centric model.

Read these in order:
1. [README.md](README.md): layout, setup and ground rules.
2. [STATUS.md](STATUS.md): what is established, the open decisions, and where the
   project owner wants to go.
3. [MODEL_ODES.md](MODEL_ODES.md): the specification the code implements.

Each study folder has a DESIGN.md (written before the runs) and a FINDINGS.md.

## Setup on a new machine

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m unittest discover -s tests -v    # 50 tests
```

- A C compiler is needed as `cc`. On macOS, install the Xcode command-line tools with
  `xcode-select --install`.
- `neuron/_kernel.so` is not in git. The kernel compiles itself on first import, for
  whatever architecture the machine has.
- The large sample arrays `physio/results/*.npz` and `branch/results/*.npz` are not in
  git. Rerun the study script to regenerate them if a plot needs them.

## Decisions the project owner has made (keep to them unless told otherwise)

- **Per-synapse reading (3 Nov 2026).** In physio/ and likelihood/, one input stream is
  one presynaptic neuron at physiological rates (0.5-64 Hz). The exact u² step is
  production, and J_max = 78.3 µM/ms. Treat new single-compartment experiments as
  physiological-rate experiments.
- **Branch reading, "option A" (3 Nov 2026).** For networks, a compartment is a dendritic
  branch of about 1000 pooled synapses (`pooled_gate=True`), with branch-level calcium
  and gbar. The stated long-term goal is a toy network of about 30 neurons (24 E + 6 I).
  Baseline in branch/branch_core.py:
  - 3 branches x 1000 synapses at 2.5 Hz each;
  - OU rate modulation σ 0.5, τ 20 ms;
  - 1000 Hz somatic inhibition at 2 nS;
  - B_tot 44.
- **Handover (8 Nov 2026).** This folder was cut from a larger research repository
  (exp1-exp9, vv/, legacy latch model) so that it is self-contained. That repository is
  not here. If something refers to it, ask the project owner.

## Where the work stopped (10 Nov 2026)

- The likelihood study (likelihood/) is complete. FINDINGS.md lists options A-D for the
  Bayesian layer's window, family and learned predictive.
- None of those options has been chosen or implemented. The model still uses a 2 s
  window, gamma on M and τ_slow 60 s.
- The project owner's next aim, from STATUS.md, is "proposed, not agreed": map the
  buffer "modes", then an online gradient on B_tot, then the network.
- Ask before starting it, and check whether the likelihood options should be settled
  first.

## Working conventions

- **Studies follow a fixed pattern.**
  - Write DESIGN.md with pre-set criteria before running.
  - Scripts write JSON, text and plots.
  - FINDINGS.md interprets them.
  - Seeds give the uncertainty; arms that share a seed are paired, not extra samples.
- **The model and its kernel move together.** A model change goes into MODEL_ODES.md and
  `make_ode_sheet.py` in the same step. `neuron/model.py` is the reference, and
  `kernel.c` / `fast.py` must match it (tests/test_kernel.py).
- **Long sweeps run in the background**, with one file per cell so they can resume. The
  old 2017 MacBook ran the likelihood sweep at about 200 s per cell on 4 workers.
