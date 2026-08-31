"""Lévy-distributed random steps, shared by several algorithms.

Reference:
    R. N. Mantegna, "Fast, accurate algorithm for numerical simulation
    of Lévy stable stochastic processes," Physical Review E, 49(5),
    4677-4683, 1994.
"""

import math

import numpy as np


def levy_flight(rng, size, beta=1.5, scale=1.0):
    """Draw ``size`` Lévy-distributed steps of exponent ``beta``.

    Uses Mantegna's algorithm: ``u / |v|^(1/beta)`` with normally
    distributed ``u`` and ``v`` reproduces a Lévy distribution, giving
    many small steps and rare very large ones. That heavy tail is what
    makes a Lévy walk cover ground a Gaussian walk cannot.

    Two details matter to callers. This draws from ``rng`` exactly
    twice, ``u`` before ``v``, and seeded reproducibility depends on
    that order. And ``scale`` is applied to the numerator rather than to
    the result, so a scaled call is bit-identical to writing
    ``scale * u / |v|^(1/beta)`` inline; the default 1.0 is exact and
    leaves unscaled callers unaffected.

    Args:
        rng: A ``numpy.random.Generator``.
        size: Number of steps to draw.
        beta: Lévy exponent in (0, 2]; lower means heavier tails.
        scale: Step size multiplier.

    Returns:
        numpy.ndarray: The steps.
    """
    sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
             / (math.gamma((1.0 + beta) / 2.0) * beta
                * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)
    u = rng.normal(0.0, sigma, size)
    v = rng.normal(0.0, 1.0, size)
    return scale * u / np.abs(v) ** (1.0 / beta)
