"""Monarch Butterfly Optimization for continuous search spaces.

Reference:
    G.-G. Wang, S. Deb, and Z. Cui, "Monarch butterfly optimization,"
    Neural Computing and Applications, 31(7), 1995-2014, 2019
    (first published online 2015).
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm
from ikn_library.algorithms.levy import levy_flight


class MonarchButterflyOptimization(Algorithm):
    """Monarch Butterfly Optimization (Wang et al., 2015).

    Modeled on the monarch's migration between two lands. The population
    is split into two subpopulations that run **different operators**,
    and butterflies exchange coordinates across the divide:

    - **Migration** rebuilds each butterfly in Land 1 coordinate by
      coordinate, taking each one from a random butterfly in Land 1 or
      Land 2 depending on a per-coordinate draw.
    - **Adjusting** rebuilds each butterfly in Land 2 from either the
      current best or a random Land 2 member, sometimes adding a Lévy
      walk step.

    Both operators are **per-coordinate discrete recombination**, the
    same family as Harmony Search: a new solution is assembled from
    pieces of several existing ones rather than moved through space.
    Nothing here computes a direction or a velocity.

    Elitism carries the best few butterflies through unchanged.

    Args:
        population_size: Number of butterflies.
        partition: Fraction assigned to Land 1, and the per-coordinate
            probability of drawing from it. The paper uses 5/12.
            Retuned here; see the algorithm page.
        period: Migration period, scaling the migration draw.
        bar: Butterfly adjusting rate — below it, no Lévy step is
            added. The paper uses 5/12.
        max_step: Maximum Lévy walk step.
        n_elite: Butterflies carried through unchanged each iteration.
        budget_tied_step: Scale the Lévy step by the remaining budget
            instead of by ``1 / t**2``. See the algorithm page for why
            the published schedule is a problem.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=20, partition=0.5, period=1.2,
                 bar=0.85, max_step=3.0, n_elite=4,
                 budget_tied_step=True, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 < partition < 1.0:
            raise ValueError("partition must be in (0, 1)")
        if period <= 0:
            raise ValueError("period must be > 0")
        if not 0.0 <= bar <= 1.0:
            raise ValueError("bar must be in [0, 1]")
        if max_step <= 0:
            raise ValueError("max_step must be > 0")
        if not 0 <= n_elite < population_size:
            raise ValueError("n_elite must be in [0, population_size)")
        self.partition = float(partition)
        self.period = float(period)
        self.bar = float(bar)
        self.max_step = float(max_step)
        self.n_elite = int(n_elite)
        self.budget_tied_step = bool(budget_tied_step)

    def init_population(self, task):
        butterflies = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in butterflies])
        order = np.argsort(fitness)
        return butterflies[order], fitness[order]

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _land_sizes(self):
        n1 = int(np.ceil(self.partition * self.population_size))
        return max(min(n1, self.population_size - 1), 1)

    def _migrate(self, butterflies, land1, land2, dimension):
        """Assemble a Land 1 butterfly from both lands, coordinate-wise.

        Each coordinate is read from its own randomly chosen donor, so
        one offspring can draw on many butterflies at once.
        """
        coordinates = np.arange(dimension)
        from_land1 = self.rng.random(dimension) * self.period <= self.partition
        donors = np.where(from_land1,
                          self.rng.choice(land1, dimension),
                          self.rng.choice(land2, dimension))
        return butterflies[donors, coordinates]

    def _adjust(self, butterflies, best, land2, dimension, alpha):
        """Assemble a Land 2 butterfly from the best or a peer."""
        coordinates = np.arange(dimension)
        take_best = self.rng.random(dimension) <= self.partition
        peers = butterflies[self.rng.choice(land2, dimension), coordinates]
        candidate = np.where(take_best, best, peers)

        # Peer-derived coordinates may additionally get a Lévy walk step.
        walk = self.rng.random(dimension) > self.bar
        step = alpha * (levy_flight(self.rng, dimension) - 0.5)
        return np.where(~take_best & walk, candidate + step, candidate)

    def run_iteration(self, task, state):
        butterflies, fitness = state
        dimension = task.dimension
        elites = butterflies[: self.n_elite].copy()
        elite_fitness = fitness[: self.n_elite].copy()

        n1 = self._land_sizes()
        land1 = np.arange(n1)
        land2 = np.arange(n1, self.population_size)
        best = butterflies[0].copy()

        # The published weight is max_step / t**2, which is tied to the
        # absolute generation count rather than the budget.
        if self.budget_tied_step:
            alpha = self.max_step * max(1.0 - self._progress(task), 1e-6) ** 2
        else:
            alpha = self.max_step / max(task.iters, 1) ** 2

        for i in land1:
            if task.stopping_condition():
                break
            candidate = self._migrate(butterflies, land1, land2, dimension)
            butterflies[i] = task.repair(candidate)
            fitness[i] = task.eval(butterflies[i])

        for j in land2:
            if task.stopping_condition():
                break
            candidate = self._adjust(butterflies, best, land2, dimension,
                                     alpha)
            butterflies[j] = task.repair(candidate)
            fitness[j] = task.eval(butterflies[j])

        # Elitism: the best of the previous generation displace the worst.
        if self.n_elite:
            worst = np.argsort(fitness)[-self.n_elite:]
            butterflies[worst] = elites
            fitness[worst] = elite_fitness

        order = np.argsort(fitness)
        return butterflies[order], fitness[order]
