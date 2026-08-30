"""The Bees Algorithm for continuous search spaces.

Reference:
    D. T. Pham, A. Ghanbarzadeh, E. Koc, S. Otri, S. Rahim, and
    M. Zaidi, "The Bees Algorithm," Technical Note, Manufacturing
    Engineering Centre, Cardiff University, 2005;
    D. T. Pham and M. Castellani, "The Bees Algorithm: modelling
    foraging behaviour to solve continuous optimisation problems,"
    Proceedings of the Institution of Mechanical Engineers, Part C,
    223(12), 2919-2938, 2009.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class BeesAlgorithm(Algorithm):
    """The Bees Algorithm (Pham et al., 2005), a neighborhood search.

    Scout bees sample the space at random; the best **sites** they find
    are then searched more closely, with the very best (**elite**)
    sites receiving the most foragers. Each site keeps only its best
    forager, its neighborhood **shrinks** after every unsuccessful
    search so the site refines its own scale, and a site that stagnates
    for ``stagnation_limit`` searches is abandoned and replaced by a
    fresh random scout.

    Where :class:`~ikn_library.algorithms.ArtificialBeeColony` gives
    every food source the same treatment and lets probability decide
    where onlookers go, the Bees Algorithm allocates its effort
    explicitly: a fixed, larger number of recruits for elite sites and
    a smaller one for the remaining selected sites.

    Args:
        population_size: Number of scout bees ``n``.
        selected_sites: Sites kept for neighborhood search ``m``
            (must be < ``population_size``).
        elite_sites: How many of those are elite ``e`` (must be <=
            ``selected_sites``).
        elite_bees: Recruits per elite site ``nep``.
        selected_bees: Recruits per non-elite selected site ``nsp``.
        neighborhood: Initial neighborhood radius as a fraction of each
            dimension's bound range.
        shrink: Factor applied to a site's radius after an
            unsuccessful search, in (0, 1].
        stagnation_limit: Unsuccessful searches before a site is
            abandoned.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=25, selected_sites=12, elite_sites=4,
                 elite_bees=8, selected_bees=4, neighborhood=0.1, shrink=0.9,
                 stagnation_limit=15, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 1 <= selected_sites < population_size:
            raise ValueError("selected_sites must be in [1, population_size)")
        if not 1 <= elite_sites <= selected_sites:
            raise ValueError("elite_sites must be in [1, selected_sites]")
        if elite_bees < 1 or selected_bees < 1:
            raise ValueError("elite_bees and selected_bees must be >= 1")
        if neighborhood <= 0:
            raise ValueError("neighborhood must be > 0")
        if not 0.0 < shrink <= 1.0:
            raise ValueError("shrink must be in (0, 1]")
        if stagnation_limit < 1:
            raise ValueError("stagnation_limit must be >= 1")
        self.selected_sites = int(selected_sites)
        self.elite_sites = int(elite_sites)
        self.elite_bees = int(elite_bees)
        self.selected_bees = int(selected_bees)
        self.neighborhood = float(neighborhood)
        self.shrink = float(shrink)
        self.stagnation_limit = int(stagnation_limit)

    def init_population(self, task):
        sites = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in sites])
        order = np.argsort(fitness)
        radii = np.full(self.population_size, self.neighborhood)
        stagnation = np.zeros(self.population_size, dtype=int)
        return sites[order], fitness[order], radii, stagnation

    def _forage(self, task, centre, radius, n_recruits):
        """Send ``n_recruits`` bees into a site; return the best find."""
        span = radius * (task.upper - task.lower)
        best_x, best_fitness = None, np.inf
        for _ in range(n_recruits):
            if task.stopping_condition():
                break
            candidate = task.repair(self.rng.uniform(centre - span, centre + span))
            candidate_fitness = task.eval(candidate)
            if candidate_fitness < best_fitness:
                best_x, best_fitness = candidate, candidate_fitness
        return best_x, best_fitness

    def run_iteration(self, task, state):
        sites, fitness, radii, stagnation = state

        # Neighborhood search: elite sites get more recruits.
        for i in range(self.selected_sites):
            recruits = (self.elite_bees if i < self.elite_sites
                        else self.selected_bees)
            found, found_fitness = self._forage(task, sites[i], radii[i], recruits)
            if found is not None and found_fitness < fitness[i]:
                sites[i], fitness[i] = found, found_fitness
                stagnation[i] = 0
                # The radius is deliberately *not* reset here: letting it
                # keep shrinking across successes is what gives the site
                # an ever-finer search scale (progressive neighborhood
                # shrinking, Pham & Castellani 2009).
            else:
                radii[i] *= self.shrink           # failure: search finer
                stagnation[i] += 1
                if stagnation[i] >= self.stagnation_limit:
                    # Site abandonment: replace it with a fresh scout.
                    sites[i] = self.rng.uniform(task.lower, task.upper)
                    fitness[i] = task.eval(sites[i])
                    radii[i] = self.neighborhood
                    stagnation[i] = 0

        # Global search: the remaining scouts sample the space at random.
        for i in range(self.selected_sites, self.population_size):
            sites[i] = self.rng.uniform(task.lower, task.upper)
            fitness[i] = task.eval(sites[i])
            radii[i] = self.neighborhood
            stagnation[i] = 0

        order = np.argsort(fitness)
        return sites[order], fitness[order], radii[order], stagnation[order]
