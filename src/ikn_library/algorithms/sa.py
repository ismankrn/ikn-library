"""Simulated Annealing for continuous search spaces.

Reference:
    S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by
    simulated annealing," Science, 220(4598), 1983.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class SimulatedAnnealing(Algorithm):
    """Simulated Annealing (SA), a single-solution metaheuristic.

    Each iteration proposes one neighbor of the current solution by
    adding Gaussian noise. A better neighbor is always accepted; a worse
    one is accepted with probability ``exp(-delta / T)`` (the Metropolis
    criterion), where the temperature ``T`` decays geometrically. High
    temperatures allow uphill moves that escape local optima; as ``T``
    cools, the search settles into greedy refinement. The neighbor step
    shrinks with the temperature (proportional to ``sqrt(T / T0)``), so
    early iterations explore widely and late iterations refine locally.
    The task tracks the best solution ever visited, so wandering never
    loses it.

    Args:
        initial_temperature: Starting temperature ``T0``. Choose it
            relative to typical fitness differences of the problem:
            larger values accept more uphill moves early on.
        cooling: Geometric decay factor per iteration, in (0, 1).
            Values close to 1 cool slowly (more exploration, slower
            refinement).
        step_size: Initial neighbor step as a fraction of each
            dimension's bound range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, initial_temperature=1.0, cooling=0.995, step_size=0.1,
                 seed=None):
        super().__init__(population_size=1, seed=seed)
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be > 0")
        if not 0.0 < cooling < 1.0:
            raise ValueError("cooling must be in (0, 1)")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self.initial_temperature = float(initial_temperature)
        self.cooling = float(cooling)
        self.step_size = float(step_size)

    def init_population(self, task):
        x = self.rng.uniform(task.lower, task.upper)
        return x, task.eval(x), self.initial_temperature

    def run_iteration(self, task, state):
        x, fx, temperature = state
        step_fraction = np.sqrt(max(temperature / self.initial_temperature, 1e-12))
        scale = self.step_size * (task.upper - task.lower) * step_fraction
        candidate = task.repair(x + self.rng.normal(0.0, scale))
        f_candidate = task.eval(candidate)
        if f_candidate < fx:
            x, fx = candidate, f_candidate
        else:
            delta = f_candidate - fx
            if np.isfinite(delta) and self.rng.random() < np.exp(-delta / temperature):
                x, fx = candidate, f_candidate
        return x, fx, temperature * self.cooling
