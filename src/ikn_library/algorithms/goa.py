"""Grasshopper Optimisation Algorithm for continuous search spaces.

Reference:
    S. Saremi, S. Mirjalili, and A. Lewis, "Grasshopper Optimisation
    Algorithm: theory and application," Advances in Engineering
    Software, 105, 30-47, 2017.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class GrasshopperOptimizationAlgorithm(Algorithm):
    """Grasshopper Optimisation Algorithm (Saremi et al., 2017).

    Modeled on grasshopper swarms, and built around one idea nothing
    else in this library has: an explicit **short-range repulsion**.

    Every pair of grasshoppers exerts a force on each other given by

    \\[
    s(r) = f\\,e^{-r/l} - e^{-r}
    \\]

    which is *negative* (repulsive) at very short range, *positive*
    (attractive) at medium range, and fades to nothing far away. The
    distance where it crosses zero is the swarm's **comfort zone** —
    grasshoppers neither approach nor retreat there. Elsewhere the whole
    library relies on attraction alone plus some source of randomness;
    here spacing is maintained by a force law.

    A grasshopper's new position is the sum of those social forces plus
    the **target**, the best solution found so far:

    \\[
    x_i \\leftarrow c \\sum_{j \\neq i} c\\,\\frac{u - l}{2}\\,
    s(\\hat{d}_{ij})\\,\\frac{x_j - x_i}{d_{ij}} \\;+\\; T
    \\]

    Note what that means: a grasshopper's own position enters only
    through the differences. The coefficient \\(c\\) falls to nearly
    zero over the run, so the swarm collapses onto the target — the
    exploration/exploitation balance is this one shrinking factor,
    appearing twice.

    Args:
        population_size: Number of grasshoppers.
        c_max: Initial value of the shrinking coefficient.
        c_min: Final value of the shrinking coefficient.
        intensity: Attraction intensity ``f`` in the force law. The
            paper uses 0.5; ``f`` and ``attraction_length`` jointly
            decide whether a comfort zone exists at all, so change
            them together — see the algorithm page.
        attraction_length: Attractive length scale ``l``, which sets
            where the comfort zone falls.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, c_max=1.0, c_min=0.00004,
                 intensity=0.6, attraction_length=1.5, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if c_max <= 0:
            raise ValueError("c_max must be > 0")
        if not 0 < c_min <= c_max:
            raise ValueError("c_min must be in (0, c_max]")
        if intensity <= 0:
            raise ValueError("intensity must be > 0")
        if attraction_length <= 0:
            raise ValueError("attraction_length must be > 0")
        self.c_max = float(c_max)
        self.c_min = float(c_min)
        self.intensity = float(intensity)
        self.attraction_length = float(attraction_length)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _social_force(self, distance):
        """Repulsive at short range, attractive further out.

        The zero crossing is the comfort zone: grasshoppers neither
        approach nor retreat at that separation.
        """
        return (self.intensity * np.exp(-distance / self.attraction_length)
                - np.exp(-distance))

    def run_iteration(self, task, state):
        grasshoppers, fitness = state
        target = grasshoppers[np.argmin(fitness)].copy()
        span = task.upper - task.lower

        c = self.c_max - (self.c_max - self.c_min) * self._progress(task)

        offsets = grasshoppers[None, :, :] - grasshoppers[:, None, :]
        distances = np.linalg.norm(offsets, axis=2)

        # The force law is only meaningful on a bounded range, so raw
        # distances are rescaled into [1, 4] as the authors prescribe.
        largest = distances.max()
        if largest > 0:
            scaled = 1.0 + 3.0 * distances / largest
        else:
            scaled = np.ones_like(distances)

        directions = offsets / (distances[:, :, None] + 1e-12)
        forces = self._social_force(scaled)
        np.fill_diagonal(forces, 0.0)          # a grasshopper ignores itself

        interaction = (c * (span / 2.0)
                       * (forces[:, :, None] * directions)).sum(axis=1)
        candidates = c * interaction + target

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            grasshoppers[i] = task.repair(candidates[i])
            fitness[i] = task.eval(grasshoppers[i])

        return grasshoppers, fitness
