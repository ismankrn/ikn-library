"""Sine Cosine Algorithm for continuous search spaces.

Reference:
    S. Mirjalili, "SCA: a Sine Cosine Algorithm for solving
    optimization problems," Knowledge-Based Systems, 96, 120-133, 2016.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class SineCosineAlgorithm(Algorithm):
    """Sine Cosine Algorithm (Mirjalili, 2016).

    One of the simplest algorithms in this library, and one of the few
    built on a mathematical function rather than an animal. Every
    solution moves toward the best one found so far, by a displacement
    scaled by a **sine or a cosine** chosen at random:

    \\[
    x \\leftarrow x + r_1 \\sin(r_2)\\,\\lvert r_3 P - x \\rvert
    \\quad\\text{or}\\quad
    x + r_1 \\cos(r_2)\\,\\lvert r_3 P - x \\rvert
    \\]

    The trigonometric factor is the whole idea. Because
    \\(\\sin\\) and \\(\\cos\\) range over \\([-1, 1]\\), a solution can
    move toward the destination, away from it, or past it, and the
    proportion of each is fixed by the geometry rather than by a tuned
    probability. The amplitude \\(r_1\\) falls linearly to zero across
    the run, so those excursions shrink and the population converges.

    Note that \\(r_3\\) multiplies the destination's **absolute
    coordinates**, which — as with the Grey Wolf Optimizer — makes the
    update depend on where the origin happens to be. See the algorithm
    page.

    Args:
        population_size: Number of solutions.
        amplitude: Initial value of \\(r_1\\), the movement amplitude.
            Falls linearly to zero. The paper uses 2.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, amplitude=2.0, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if amplitude <= 0:
            raise ValueError("amplitude must be > 0")
        self.amplitude = float(amplitude)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def run_iteration(self, task, state):
        population, fitness = state
        shape = (self.population_size, task.dimension)
        destination = population[np.argmin(fitness)].copy()

        # r1 sets how far a solution may travel, and decays to zero.
        r1 = self.amplitude * (1.0 - self._progress(task))
        r2 = self.rng.uniform(0.0, 2.0 * np.pi, shape)
        r3 = self.rng.uniform(0.0, 2.0, shape)
        use_sine = self.rng.random(shape) < 0.5

        # sin and cos both span [-1, 1], so a solution can move toward the
        # destination, away from it, or beyond it.
        swing = np.where(use_sine, np.sin(r2), np.cos(r2))
        candidates = population + r1 * swing * np.abs(
            r3 * destination - population)

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            population[i] = task.repair(candidates[i])
            fitness[i] = task.eval(population[i])

        return population, fitness
