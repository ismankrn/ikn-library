"""Binary Ant Colony Optimization for subset-selection problems.

Each decision variable is a bit (1 = selected, 0 = not selected). A
pheromone value is maintained per (variable, bit-value) pair; ants build
bit strings by sampling each bit with a probability proportional to its
pheromone. Pheromone is updated in the hyper-cube framework: it
evaporates toward the bit values of the best solution found so far, and
is clamped to ``[tau_min, tau_max]`` to preserve exploration.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class BinaryAntColonyOptimization(Algorithm):
    """Binary ACO for feature selection and other subset problems.

    Solutions are 0/1 vectors, so the wrapped problem must interpret its
    input as a bit mask (e.g.
    :class:`~ikn_library.problems.FeatureSelectionProblem`).

    Args:
        population_size: Number of ants per iteration.
        evaporation: Pheromone evaporation/learning rate (rho) in (0, 1).
            Higher values converge faster toward the best solution.
        alpha: Pheromone importance exponent.
        tau_min: Lower pheromone limit, keeps every bit reachable.
        tau_max: Upper pheromone limit.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, evaporation=0.1, alpha=1.0,
                 tau_min=0.1, tau_max=0.9, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 < evaporation < 1.0:
            raise ValueError("evaporation must be in (0, 1)")
        if not 0.0 < tau_min < tau_max:
            raise ValueError("require 0 < tau_min < tau_max")
        self.evaporation = float(evaporation)
        self.alpha = float(alpha)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

    def _sample_ants(self, task, pheromone):
        weights = pheromone ** self.alpha
        p_one = weights[:, 1] / np.sum(weights, axis=1)
        ants = (self.rng.random((self.population_size, task.dimension)) < p_one).astype(float)
        # An all-zero ant selects nothing and is unevaluable as a subset;
        # repair it by switching one random bit on.
        for ant in ants:
            if not ant.any():
                ant[self.rng.integers(task.dimension)] = 1.0
        return ants

    def init_population(self, task):
        pheromone = np.full((task.dimension, 2), 0.5)
        ants = self._sample_ants(task, pheromone)
        for ant in ants:
            task.eval(ant)
        return pheromone

    def run_iteration(self, task, pheromone):
        ants = self._sample_ants(task, pheromone)
        for ant in ants:
            task.eval(ant)

        # Hyper-cube pheromone update toward the best-so-far solution.
        best_bits = task.best_x.astype(int)
        deposit = np.zeros_like(pheromone)
        deposit[np.arange(task.dimension), best_bits] = 1.0
        pheromone = (1.0 - self.evaporation) * pheromone + self.evaporation * deposit
        return np.clip(pheromone, self.tau_min, self.tau_max)
