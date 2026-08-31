"""Moth-Flame Optimization for continuous search spaces.

Reference:
    S. Mirjalili, "Moth-flame optimization algorithm: a novel
    nature-inspired heuristic paradigm," Knowledge-Based Systems, 89,
    228-249, 2015.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class MothFlameOptimization(Algorithm):
    """Moth-Flame Optimization (Mirjalili, 2015).

    Modeled on the *transverse orientation* by which moths navigate:
    they hold a fixed angle to a distant light, which works for the moon
    but spirals them into a nearby flame.

    The population is kept in two arrays. **Moths** are the search
    agents; **flames** are the best positions found so far, refreshed
    each iteration by merging moths and flames and keeping the best.
    Each moth then spirals around *its own* flame:

    \\[
    M_i = D_i \\, e^{bt} \\cos(2\\pi t) + F_j,
    \\qquad D_i = \\lvert F_j - M_i \\rvert
    \\]

    Two design choices matter. Each moth has **its own** flame rather
    than a shared global best, so early on the population is pulled
    toward many different attractors at once. And the number of flames
    **shrinks from N to 1** across the run, so those attractors merge
    and the search collapses onto the single best position — the
    exploration/exploitation schedule is the flame count itself, not a
    step size.

    The spiral parameter \\(t\\) is drawn per coordinate from
    ``[a, 1]`` with ``a`` falling from -1 to -2, so a moth can land
    inside, beyond, or on either side of its flame.

    Args:
        population_size: Number of moths, and the initial flame count.
        spiral_constant: Shape constant ``b`` of the logarithmic spiral.
        a_start: Initial lower bound of the spiral parameter.
        a_end: Final lower bound of the spiral parameter.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, spiral_constant=1.0,
                 a_start=-1.0, a_end=-2.0, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if spiral_constant <= 0:
            raise ValueError("spiral_constant must be > 0")
        if a_start >= 0:
            raise ValueError("a_start must be < 0")
        if a_end >= a_start:
            raise ValueError("a_end must be < a_start")
        self.spiral_constant = float(spiral_constant)
        self.a_start = float(a_start)
        self.a_end = float(a_end)

    def init_population(self, task):
        moths = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in moths])
        order = np.argsort(fitness)
        # Flames start as the moths themselves, best first.
        return moths, fitness, moths[order].copy(), fitness[order].copy()

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _flame_count(self, progress):
        """Flames shrink from N to 1: the whole exploration schedule."""
        count = round(self.population_size
                      - progress * (self.population_size - 1))
        return int(np.clip(count, 1, self.population_size))

    def _refresh_flames(self, moths, fitness, flames, flame_fitness):
        """Merge moths and flames, keep the best N as the new flames."""
        pool = np.vstack([flames, moths])
        pool_fitness = np.concatenate([flame_fitness, fitness])
        keep = np.argsort(pool_fitness)[: self.population_size]
        return pool[keep], pool_fitness[keep]

    def run_iteration(self, task, state):
        moths, fitness, flames, flame_fitness = state

        flames, flame_fitness = self._refresh_flames(
            moths, fitness, flames, flame_fitness)

        progress = self._progress(task)
        n_flames = self._flame_count(progress)
        a = self.a_start + (self.a_end - self.a_start) * progress

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            # Moths past the flame count all share the last surviving one.
            flame = flames[min(i, n_flames - 1)]

            distance = np.abs(flame - moths[i])
            t = (a - 1.0) * self.rng.random(task.dimension) + 1.0
            spiral = (distance * np.exp(self.spiral_constant * t)
                      * np.cos(2.0 * np.pi * t))

            moths[i] = task.repair(spiral + flame)
            fitness[i] = task.eval(moths[i])

        return moths, fitness, flames, flame_fitness
