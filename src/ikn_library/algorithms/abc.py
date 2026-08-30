"""Artificial Bee Colony for continuous search spaces.

Reference:
    D. Karaboga, "An idea based on honey bee swarm for numerical
    optimization," Technical Report TR06, Erciyes University, 2005;
    D. Karaboga and B. Basturk, "A powerful and efficient algorithm for
    numerical function optimization: artificial bee colony (ABC)
    algorithm," Journal of Global Optimization, 39(3), 459-471, 2007.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class ArtificialBeeColony(Algorithm):
    """Artificial Bee Colony (ABC), modeled on honey-bee foraging.

    Each food source is a candidate solution, and one iteration runs
    the colony's three roles:

    - **Employed bees** each probe their own source by stepping toward
      a random partner in one random dimension; a better neighbor
      replaces the source (greedy selection).
    - **Onlooker bees** pick sources to probe with probability
      proportional to their quality, so promising regions get more
      trials.
    - **Scout bees** abandon a source that has not improved for
      ``limit`` consecutive trials and replace it with a fresh random
      solution — the mechanism that keeps ABC exploring.

    Args:
        population_size: Number of food sources (employed bees; the
            same number of onlookers are dispatched each iteration).
        limit: Trials without improvement before a source is abandoned.
            ``None`` (default) uses the common heuristic
            ``population_size * dimension``.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, limit=None, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = limit

    def init_population(self, task):
        sources = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in sources])
        trials = np.zeros(self.population_size, dtype=int)
        return sources, fitness, trials

    def _neighbor(self, sources, index, task):
        partner = self.rng.integers(self.population_size)
        while partner == index and self.population_size > 1:
            partner = self.rng.integers(self.population_size)
        candidate = sources[index].copy()
        dimension = self.rng.integers(task.dimension)
        phi = self.rng.uniform(-1.0, 1.0)
        candidate[dimension] += phi * (candidate[dimension] - sources[partner, dimension])
        return task.repair(candidate)

    def _probabilities(self, fitness):
        # Karaboga's fitness transform: minimization values are mapped to
        # positive quality scores, then normalized into probabilities.
        quality = np.where(fitness >= 0, 1.0 / (1.0 + fitness), 1.0 + np.abs(fitness))
        total = quality.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(self.population_size, 1.0 / self.population_size)
        return quality / total

    def _probe(self, task, sources, fitness, trials, index):
        candidate = self._neighbor(sources, index, task)
        candidate_fitness = task.eval(candidate)
        if candidate_fitness < fitness[index]:
            sources[index] = candidate
            fitness[index] = candidate_fitness
            trials[index] = 0
        else:
            trials[index] += 1

    def run_iteration(self, task, state):
        sources, fitness, trials = state
        limit = self.limit or self.population_size * task.dimension

        # Employed bees: every source is probed once.
        for index in range(self.population_size):
            self._probe(task, sources, fitness, trials, index)

        # Onlooker bees: sources are probed proportionally to quality.
        probabilities = self._probabilities(fitness)
        for index in self.rng.choice(self.population_size, self.population_size,
                                     p=probabilities):
            self._probe(task, sources, fitness, trials, index)

        # Scout bees: abandon the most stagnant exhausted source.
        exhausted = np.flatnonzero(trials >= limit)
        if len(exhausted) > 0:
            index = exhausted[np.argmax(trials[exhausted])]
            sources[index] = self.rng.uniform(task.lower, task.upper)
            fitness[index] = task.eval(sources[index])
            trials[index] = 0

        return sources, fitness, trials
