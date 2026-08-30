"""NSGA-II: elitist non-dominated sorting genetic algorithm.

Reference:
    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
    multiobjective genetic algorithm: NSGA-II," IEEE Transactions on
    Evolutionary Computation, 6(2), 182-197, 2002.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm
from ikn_library.multiobjective.pareto import crowding_distance, non_dominated_sort


class NSGA2(Algorithm):
    """NSGA-II for problems with several conflicting objectives.

    Instead of one best solution, NSGA-II returns a **Pareto front**: a
    set of solutions where none is better than another in every
    objective, so the user can pick the trade-off they want after the
    search rather than guessing weights before it.

    Three mechanisms replace the single-objective machinery:

    - **Non-dominated sorting** ranks the population into fronts: front
      0 is non-dominated, front 1 is dominated only by front 0, and so
      on. This replaces "sort by fitness".
    - **Crowding distance** breaks ties inside a front, favouring
      solutions in sparsely populated regions so the front stays spread
      out instead of clustering at one end.
    - **Elitist replacement** merges parents and offspring and refills
      the next generation front by front, so no Pareto solution is
      lost.

    Run it with a :class:`~ikn_library.multiobjective.MultiObjectiveTask`,
    whose ``result()`` returns ``(solutions, objectives)`` of the front.

    Args:
        population_size: Individuals per generation.
        crossover_rate: Probability that a selected pair recombines.
        mutation_rate: Per-gene mutation probability. Defaults to
            ``1 / dimension``.
        mutation_scale: Initial mutation width as a fraction of each
            dimension's bound range; decays with the budget.
        blend_alpha: BLX-alpha crossover interval widening.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, crossover_rate=0.9,
                 mutation_rate=None, mutation_scale=0.1, blend_alpha=0.5,
                 seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if mutation_rate is not None and not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if mutation_scale <= 0:
            raise ValueError("mutation_scale must be > 0")
        if blend_alpha < 0:
            raise ValueError("blend_alpha must be >= 0")
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = mutation_rate
        self.mutation_scale = float(mutation_scale)
        self.blend_alpha = float(blend_alpha)

    def init_population(self, task):
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        objectives = np.array([task.eval(x) for x in population])
        return population, objectives

    @staticmethod
    def _ranks_and_crowding(objectives):
        """Front index and crowding distance for every solution."""
        ranks = np.empty(len(objectives), dtype=int)
        crowding = np.empty(len(objectives))
        for rank, front in enumerate(non_dominated_sort(objectives)):
            ranks[front] = rank
            crowding[front] = crowding_distance(objectives[front])
        return ranks, crowding

    def _tournament(self, population, ranks, crowding):
        """Binary tournament: lower front wins; ties go to the lonelier one."""
        i, j = self.rng.integers(0, len(population), 2)
        if ranks[i] != ranks[j]:
            winner = i if ranks[i] < ranks[j] else j
        else:
            winner = i if crowding[i] > crowding[j] else j
        return population[winner]

    def _crossover(self, parent1, parent2):
        low = np.minimum(parent1, parent2)
        high = np.maximum(parent1, parent2)
        spread = self.blend_alpha * (high - low)
        return (self.rng.uniform(low - spread, high + spread),
                self.rng.uniform(low - spread, high + spread))

    def _mutate(self, child, task, mutation_rate, scale):
        mask = self.rng.random(task.dimension) < mutation_rate
        noise = self.rng.normal(0.0, scale * (task.upper - task.lower))
        return np.where(mask, child + noise, child)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return task.evals / task.max_evals
        if np.isfinite(task.max_iters):
            return task.iters / task.max_iters
        return 0.0

    def run_iteration(self, task, state):
        population, objectives = state
        mutation_rate = (1.0 / task.dimension if self.mutation_rate is None
                         else self.mutation_rate)
        scale = self.mutation_scale * max(1.0 - self._progress(task), 1e-3)
        ranks, crowding = self._ranks_and_crowding(objectives)

        # Create offspring by tournament selection, crossover, mutation.
        offspring = []
        while len(offspring) < self.population_size:
            parent1 = self._tournament(population, ranks, crowding)
            parent2 = self._tournament(population, ranks, crowding)
            if self.rng.random() < self.crossover_rate:
                children = self._crossover(parent1, parent2)
            else:
                children = (parent1.copy(), parent2.copy())
            for child in children:
                if len(offspring) >= self.population_size:
                    break
                offspring.append(
                    task.repair(self._mutate(child, task, mutation_rate, scale)))
        offspring = np.array(offspring)
        offspring_objectives = np.array([task.eval(x) for x in offspring])

        # Elitist replacement: refill front by front from parents+offspring.
        merged = np.vstack([population, offspring])
        merged_objectives = np.vstack([objectives, offspring_objectives])
        keep = []
        for front in non_dominated_sort(merged_objectives):
            if len(keep) + len(front) <= self.population_size:
                keep.extend(front)
                continue
            # The last front only partly fits: take the least crowded.
            distances = crowding_distance(merged_objectives[front])
            room = self.population_size - len(keep)
            keep.extend(front[np.argsort(-distances)[:room]])
            break
        keep = np.array(keep, dtype=int)
        return merged[keep], merged_objectives[keep]
