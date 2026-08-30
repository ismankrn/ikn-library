"""Grey Wolf Optimizer for continuous search spaces.

Reference:
    S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey wolf optimizer,"
    Advances in Engineering Software, 69, 46-61, 2014.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class GreyWolfOptimizer(Algorithm):
    """Grey Wolf Optimizer (Mirjalili et al., 2014).

    Modeled on the social hierarchy and hunting tactics of grey wolves.
    The three best solutions are named **alpha**, **beta** and **delta**,
    and every other wolf (an *omega*) is repositioned by averaging three
    estimates of where the prey must be — one suggested by each leader:

    \\[
    X^{t+1} = \\frac{X_1 + X_2 + X_3}{3}
    \\]

    Following three leaders rather than one global best is the whole
    idea. A single attractor collapses the swarm onto one point; three
    disagreeing attractors keep it spread over the region they bracket,
    which is what preserves diversity here without any explicit
    diversity mechanism.

    The exploration/exploitation balance comes from one coefficient,
    ``a``, which falls linearly from 2 to 0 across the run. It sets the
    range of the random factor ``A``: while ``|A| > 1`` wolves are
    pushed *away* from a leader (search), and once ``|A| < 1`` they are
    pulled toward it (attack). Unusually for its era, this schedule is
    tied to the run's progress in the original formulation, so no step
    decay had to be added here.

    Args:
        population_size: Number of wolves in the pack.
        a_start: Initial value of the control coefficient ``a``. Falls
            linearly to ``a_end``; the paper uses 2.
        a_end: Final value of ``a``.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, a_start=2.0, a_end=0.0,
                 seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if a_start <= 0:
            raise ValueError("a_start must be > 0")
        if a_end < 0:
            raise ValueError("a_end must be >= 0")
        if a_end >= a_start:
            raise ValueError("a_end must be < a_start")
        self.a_start = float(a_start)
        self.a_end = float(a_end)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def run_iteration(self, task, state):
        wolves, fitness = state
        n, dim = wolves.shape

        # The pack's three best wolves lead the hunt.
        leaders = wolves[np.argsort(fitness)[:3]]

        # a falls linearly; |A| > 1 explores, |A| < 1 attacks.
        a = self.a_start - (self.a_start - self.a_end) * self._progress(task)
        A = 2.0 * a * self.rng.random((3, n, dim)) - a
        C = 2.0 * self.rng.random((3, n, dim))

        # Each leader proposes where the prey is; the wolf takes the mean.
        D = np.abs(C * leaders[:, None, :] - wolves[None, :, :])
        candidates = (leaders[:, None, :] - A * D).mean(axis=0)

        for i in range(n):
            if task.stopping_condition():
                break
            wolves[i] = task.repair(candidates[i])
            fitness[i] = task.eval(wolves[i])

        return wolves, fitness
