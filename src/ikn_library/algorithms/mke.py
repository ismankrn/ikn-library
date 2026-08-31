"""Monkey King Evolution for continuous search spaces.

Reference:
    Z. Meng and J.-S. Pan, "Monkey King Evolution: a new memetic
    evolutionary algorithm and its application in vehicle fuel
    consumption optimization," Knowledge-Based Systems, 97, 144-157,
    2016.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class MonkeyKingEvolution(Algorithm):
    """Monkey King Evolution (Meng & Pan, 2016).

    Named for the Monkey King's trick of plucking out his hairs and
    turning them into copies of himself. The best individual — the
    **king** — spawns a group of clones that each probe a different
    direction, and the best clone found replaces him if it is better.

    That is the algorithm's one structural idea, and it is a genuine
    asymmetry: the king spends ``n_clones`` evaluations per iteration
    exploring around the current optimum, while every other individual
    spends exactly one. The budget concentrates where the search is
    already doing well, without any of the usual ranking or weighting
    machinery.

    Both moves use the same **difference vector**, masked so that only
    some coordinates change:

    \\[
    x' = x + \\text{FC}\\,(x_{r_1} - x_{r_2})
    \\]

    which is the operator Differential Evolution is built on. What
    MKE adds is the clone group, not a new way of moving.

    !!! note
        The published description of MKE leaves several details open,
        and implementations in circulation differ. This is a coherent
        reading of the core mechanism rather than a claim of exact
        fidelity; see the algorithm page.

    Args:
        population_size: Number of individuals.
        n_clones: Clones the king spawns each iteration. Every clone
            costs an evaluation.
        fluctuation: Scale ``FC`` of the difference vector.
        change_rate: Per-coordinate chance that a coordinate is
            altered. Low values help on separable problems; see the
            algorithm page for the caveat.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, n_clones=8, fluctuation=0.7,
                 change_rate=0.1, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        if n_clones < 1:
            raise ValueError("n_clones must be >= 1")
        if fluctuation <= 0:
            raise ValueError("fluctuation must be > 0")
        if not 0.0 < change_rate <= 1.0:
            raise ValueError("change_rate must be in (0, 1]")
        self.n_clones = int(n_clones)
        self.fluctuation = float(fluctuation)
        self.change_rate = float(change_rate)

    def _difference_move(self, population, index, task):
        """Perturb some coordinates along a random difference vector."""
        others = [k for k in range(self.population_size) if k != index]
        r1, r2 = self.rng.choice(others, 2, replace=False)
        mask = self.rng.random(task.dimension) < self.change_rate
        if not mask.any():                      # always change something
            mask[self.rng.integers(task.dimension)] = True
        step = self.fluctuation * (population[r1] - population[r2])
        return task.repair(np.where(mask, population[index] + step,
                                    population[index]))

    def _clone_the_king(self, task, population, fitness, king):
        """The king probes many directions; the best probe survives."""
        best_position, best_fitness = population[king].copy(), fitness[king]
        for _ in range(self.n_clones):
            if task.stopping_condition():
                break
            clone = self._difference_move(population, king, task)
            clone_fitness = task.eval(clone)
            if clone_fitness < best_fitness:
                best_position, best_fitness = clone, clone_fitness
        return best_position, best_fitness

    def run_iteration(self, task, state):
        population, fitness = state
        king = int(np.argmin(fitness))

        # The king gets many trials; everyone else gets one.
        position, king_fitness = self._clone_the_king(
            task, population, fitness, king)
        population[king], fitness[king] = position, king_fitness

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            if i == king:
                continue
            candidate = self._difference_move(population, i, task)
            candidate_fitness = task.eval(candidate)
            if candidate_fitness < fitness[i]:      # greedy: never worsen
                population[i], fitness[i] = candidate, candidate_fitness

        return population, fitness
