"""Neuron-centric model: dendritic LIF, split NMDA gate, buffered calcium and
Bayesian AMPA plasticity. See README.md and MODEL_ODES.md.

    from neuron import NMDANeuron        # reference implementation (Python)
    from neuron.fast import Sim          # compiled equivalent, for long runs
"""
from .calcium import calcium_step
from .lif import DendriticLIF, attenuation, poisson_input, steady_state, steady_state_general
from .model import J_MAX_SPINE, NMDANeuron, mg_block, spike_train
from .plasticity import (ExactGammaBelief, PlasticityConfig, RollingCalciumEstimator, conductance_update,
                         signed_drive)
