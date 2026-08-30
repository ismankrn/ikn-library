"""Bat Algorithm for continuous search spaces.

Reference:
    X.-S. Yang, "A new metaheuristic bat-inspired algorithm," in
    Nature Inspired Cooperative Strategies for Optimization (NICSO
    2010), Studies in Computational Intelligence 284, Springer, 2010.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class BatAlgorithm(Algorithm):
    """Bat Algorithm (BA), inspired by the echolocation of microbats.

    Each bat flies toward the best solution found so far with a
    randomly tuned frequency, giving every individual a different
    step size. Two echolocation-inspired controls balance the search:
    the **pulse rate** grows over time and increasingly triggers local
    random walks around the best solution (exploitation), while the
    **loudness** decays whenever a bat accepts an improvement, making
    acceptance more selective as the search converges. As a practical
    refinement for box-bounded problems, the local-walk step is scaled
    to the bound range and decays linearly with the evaluation budget.

    Args:
        population_size: Number of bats.
        loudness: Initial loudness ``A0`` (acceptance probability).
        pulse_rate: Final pulse emission rate ``r0``; the actual rate
            grows toward it as ``r0 * (1 - exp(-gamma * t))``.
        alpha: Loudness decay factor per accepted improvement, in (0, 1).
        gamma: Growth rate of the pulse rate over iterations.
        min_frequency: Lower bound of the frequency range.
        max_frequency: Upper bound of the frequency range.
        local_scale: Initial step of the local random walk around the
            best solution, as a fraction of each dimension's bound
            range; decays with search progress.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, loudness=1.0, pulse_rate=0.5,
                 alpha=0.9, gamma=0.9, min_frequency=0.0, max_frequency=2.0,
                 local_scale=0.05, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if loudness <= 0:
            raise ValueError("loudness must be > 0")
        if not 0.0 <= pulse_rate <= 1.0:
            raise ValueError("pulse_rate must be in [0, 1]")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if min_frequency >= max_frequency:
            raise ValueError("min_frequency must be < max_frequency")
        if local_scale <= 0:
            raise ValueError("local_scale must be > 0")
        self.loudness = float(loudness)
        self.pulse_rate = float(pulse_rate)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.local_scale = float(local_scale)

    def init_population(self, task):
        positions = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in positions])
        velocities = np.zeros_like(positions)
        loudness = np.full(self.population_size, self.loudness)
        return positions, velocities, fitness, loudness

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return task.evals / task.max_evals
        if np.isfinite(task.max_iters):
            return task.iters / task.max_iters
        return 0.0

    def run_iteration(self, task, state):
        positions, velocities, fitness, loudness = state
        best = task.best_x
        rate = self.pulse_rate * (1.0 - np.exp(-self.gamma * task.iters))
        span = task.upper - task.lower
        walk_scale = self.local_scale * max(1.0 - self._progress(task), 1e-3)

        for i in range(self.population_size):
            frequency = self.rng.uniform(self.min_frequency, self.max_frequency)
            velocities[i] += (best - positions[i]) * frequency
            candidate = positions[i] + velocities[i]
            if self.rng.random() > rate:
                # Local random walk around the current best solution.
                candidate = best + (walk_scale
                                    * self.rng.standard_normal(task.dimension) * span)
            candidate = task.repair(candidate)
            candidate_fitness = task.eval(candidate)
            if candidate_fitness <= fitness[i] and self.rng.random() < loudness[i]:
                positions[i] = candidate
                fitness[i] = candidate_fitness
                loudness[i] *= self.alpha

        return positions, velocities, fitness, loudness
