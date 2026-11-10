"""Dependency-free scalar calcium step (MODEL_ODES.md sections 3 and 9).

Strang splitting: exact clearance/influx half steps around the exact
mass-action binding solution. State is (total calcium, bound calcium).
"""
from math import exp, sqrt


def calcium_step(total, bound, influx, capacity, dt, c0, tau_c, kon, koff,
                 nsub=2):
    """Advance a constant-influx step; adding capacity adds no calcium.

    Callers validate positive kinetic constants and integral nsub. No clipping
    or equilibrium approximation is used. Scalars keep long replay runs cheap.
    """
    h = dt / nsub
    decay = exp(-0.5 * h / tau_c)
    for _ in range(nsub):
        equilibrium = bound + c0 + influx * tau_c
        total = equilibrium + (total - equilibrium) * decay
        beta = kon * (total + capacity) + koff
        gamma = kon * total * capacity
        # Equivalent to beta**2 - 4*kon*gamma, without cancellation.
        speed = sqrt((kon * (total - capacity)) ** 2
                     + 2 * kon * koff * (total + capacity) + koff * koff)
        lower = 2 * gamma / (beta + speed)
        delta = speed / kon
        denominator = bound - lower - delta
        ratio = ((bound - lower) / denominator) * exp(-speed * h) if denominator else 0.0
        bound = lower - delta * ratio / (1 - ratio)
        equilibrium = bound + c0 + influx * tau_c
        total = equilibrium + (total - equilibrium) * decay
    return total, bound
