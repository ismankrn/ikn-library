"""Whale Optimization Algorithm for continuous search spaces.

Reference:
    S. Mirjalili and A. Lewis, "The Whale Optimization Algorithm,"
    Advances in Engineering Software, 95, 51-67, 2016.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class WhaleOptimizationAlgorithm(Algorithm):
    """Whale Optimization Algorithm (Mirjalili & Lewis, 2016).

    Modeled on the *bubble-net* feeding of humpback whales. Each whale
    flips a coin and either **swims toward** the best solution or
    **spirals around** it, and a second control decides whether the
    target is the best whale or a random one:

    | Coin \\(p\\) | \\(\\lvert A \\rvert\\) | Move |
    |---|---|---|
    | < 0.5 | \\(\\ge 1\\) | search — encircle a **random** whale |
    | < 0.5 | \\(< 1\\) | encircle the **best** whale |
    | \\(\\ge 0.5\\) | — | spiral toward the best whale |

    The spiral is the algorithm's signature: a logarithmic path around
    the incumbent, the same curve the
    Moth-Flame Optimization uses. What differs is that here it
    alternates with a straight-line approach on every individual, rather
    than being the only move.

    The coefficient \\(a\\) falls linearly from 2 to 0, which sets the
    range of \\(A\\) and so decides how often the search branch can fire
    at all. Once \\(a < 1\\) exploration switches itself off.

    Note that the encircling and search moves scale the target's
    **absolute coordinates** through \\(C\\), while the spiral does not.
    That mixture has a measurable consequence; see the algorithm page.

    Args:
        population_size: Number of whales.
        a_start: Initial value of the control coefficient ``a``.
        spiral_constant: Shape constant ``b`` of the logarithmic
            spiral. The paper uses 1.0; 2.0 measures better on shifted
            and rotated problems.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, a_start=2.0, spiral_constant=2.0,
                 seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if a_start <= 0:
            raise ValueError("a_start must be > 0")
        if spiral_constant <= 0:
            raise ValueError("spiral_constant must be > 0")
        self.a_start = float(a_start)
        self.spiral_constant = float(spiral_constant)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _spiral(self, best, whale, dimension):
        """Logarithmic bubble-net path around the best whale."""
        distance = np.abs(best - whale)
        spin = self.rng.uniform(-1.0, 1.0, dimension)
        return (distance * np.exp(self.spiral_constant * spin)
                * np.cos(2.0 * np.pi * spin) + best)

    def _encircle(self, target, whale, a, dimension):
        """Straight-line approach to (or retreat from) a target."""
        A = 2.0 * a * self.rng.random(dimension) - a
        C = 2.0 * self.rng.random(dimension)
        return target - A * np.abs(C * target - whale)

    def run_iteration(self, task, state):
        whales, fitness = state
        best = whales[np.argmin(fitness)].copy()
        # a falls to zero, which shuts the search branch off partway.
        a = self.a_start * (1.0 - self._progress(task))

        for i in range(self.population_size):
            if task.stopping_condition():
                break

            if self.rng.random() < 0.5:
                # |A| >= 1 sends the whale after a random peer instead.
                magnitude = abs(2.0 * a * self.rng.random() - a)
                if magnitude >= 1.0:
                    target = whales[self.rng.integers(self.population_size)]
                else:
                    target = best
                candidate = self._encircle(target, whales[i], a,
                                           task.dimension)
            else:
                candidate = self._spiral(best, whales[i], task.dimension)

            whales[i] = task.repair(candidate)
            fitness[i] = task.eval(whales[i])

        return whales, fitness
