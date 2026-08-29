"""Ant Colony Optimization for continuous domains (ACO-R).

Reference:
    K. Socha and M. Dorigo, "Ant colony optimization for continuous
    domains," European Journal of Operational Research, 185(3), 2008.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class AntColonyOptimization(Algorithm):
    """ACO-R: Ant Colony Optimization for continuous search spaces.

    Keeps an archive of the best solutions found so far. Each ant builds
    a new solution by picking a guide solution from the archive (better
    solutions are picked with higher probability) and sampling each
    coordinate from a Gaussian centered on the guide. The Gaussian width
    shrinks as the archive converges, balancing exploration and
    exploitation — the archive plays the role of the pheromone trail.

    Args:
        population_size: Number of ants (new solutions) per iteration.
        archive_size: Number of solutions kept in the archive (k).
        intensification: Locality of search (q). Small values focus on
            the best archive solutions; larger values spread the
            selection more evenly.
        evaporation: Speed of convergence (xi). Plays a role similar to
            pheromone evaporation: higher values mean slower convergence.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, archive_size=50,
                 intensification=0.1, evaporation=0.85, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if archive_size < 2:
            raise ValueError("archive_size must be >= 2")
        self.archive_size = int(archive_size)
        self.intensification = float(intensification)
        self.evaporation = float(evaporation)

    def init_population(self, task):
        archive = self.rng.uniform(
            task.lower, task.upper, (self.archive_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in archive])
        order = np.argsort(fitness)
        return archive[order], fitness[order]

    def _selection_weights(self):
        k, q = self.archive_size, self.intensification
        ranks = np.arange(k)
        weights = np.exp(-(ranks ** 2) / (2.0 * (q * k) ** 2)) / (q * k * np.sqrt(2.0 * np.pi))
        return weights / np.sum(weights)

    def run_iteration(self, task, state):
        archive, fitness = state
        probabilities = self._selection_weights()

        ants = np.empty((self.population_size, task.dimension))
        for a in range(self.population_size):
            guide = self.rng.choice(self.archive_size, p=probabilities)
            # Gaussian width per dimension: mean distance from the guide
            # to the rest of the archive, scaled by the evaporation rate.
            sigma = self.evaporation * np.sum(
                np.abs(archive - archive[guide]), axis=0
            ) / (self.archive_size - 1)
            ants[a] = task.repair(self.rng.normal(archive[guide], np.maximum(sigma, 1e-12)))
        ant_fitness = np.array([task.eval(x) for x in ants])

        # Merge ants into the archive and keep the best archive_size solutions.
        merged = np.vstack([archive, ants])
        merged_fitness = np.concatenate([fitness, ant_fitness])
        order = np.argsort(merged_fitness)[: self.archive_size]
        return merged[order], merged_fitness[order]
