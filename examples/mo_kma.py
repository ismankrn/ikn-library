"""Example: turning a single-objective algorithm multi-objective.

MO-KMA — the Komodo Mlipir Algorithm with Pareto ranking instead of
fitness ranking. KMA splits its population into big males, one female,
and small males *by rank*, so replacing that ranking with NSGA-II's
(front first, crowding distance second) is enough to make the whole
algorithm multi-objective. No other operator changes.

The same recipe works for any rank-based algorithm in this library.
"""

import numpy as np

from ikn_library.algorithms import KomodoMlipirAlgorithm
from ikn_library.multiobjective import (
    MultiObjectiveProblem,
    MultiObjectiveTask,
    dominates,
    pareto_sort_indices,
)


class MOKomodoMlipir(KomodoMlipirAlgorithm):
    """Komodo Mlipir Algorithm ranked by Pareto dominance."""

    def init_population(self, task):
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        objectives = np.array([task.eval(x) for x in population])
        order = pareto_sort_indices(objectives)
        return population[order], objectives[order], []

    def _move_big_males(self, task, big_males, big_objectives):
        """HILE, with 'better' meaning 'dominates'."""
        n_big = len(big_males)
        moved = np.empty_like(big_males)
        for i in range(n_big):
            shift = np.zeros(task.dimension)
            for j in range(n_big):
                if i == j:
                    continue
                r1 = self.rng.random(task.dimension)
                attract = (dominates(big_objectives[j], big_objectives[i])
                           or self.rng.random() < 0.5)
                direction = (big_males[j] - big_males[i] if attract
                             else big_males[i] - big_males[j])
                shift += r1 * direction
            moved[i] = task.repair(big_males[i] + shift)
        moved_objectives = np.array([task.eval(x) for x in moved])

        # Keep the q best of parents + offspring by Pareto rank.
        merged = np.vstack([big_males, moved])
        merged_objectives = np.vstack([big_objectives, moved_objectives])
        keep = pareto_sort_indices(merged_objectives)[:n_big]
        return merged[keep], merged_objectives[keep]

    def run_iteration(self, task, state):
        population, objectives, _ = state
        n_big, n_small = self._group_sizes(len(population))

        big_males, big_objectives = population[:n_big], objectives[:n_big]
        female, female_objectives = population[n_big].copy(), objectives[n_big]
        small_males = population[n_big + 1:]

        big_males, big_objectives = self._move_big_males(
            task, big_males, big_objectives)

        # The "winner" is now the top-ranked big male, not the minimum.
        winner = big_males[0]
        if self.rng.random() < 0.5:
            r = self.rng.random(task.dimension)
            candidate = task.repair(r * winner + (1.0 - r) * female)
        else:
            step = ((2.0 * self.rng.random(task.dimension) - 1.0)
                    * self.parthenogenesis_radius * (task.upper - task.lower))
            candidate = task.repair(female + step)
        candidate_objectives = task.eval(candidate)
        if dominates(candidate_objectives, female_objectives):
            female, female_objectives = candidate, candidate_objectives

        if n_small > 0:
            moved = np.empty_like(small_males)
            for i in range(len(small_males)):
                shift = np.zeros(task.dimension)
                for big_male in big_males:
                    follow = self.rng.random(task.dimension) < self.mlipir_rate
                    r1 = self.rng.random(task.dimension)
                    shift += np.where(follow, r1 * (big_male - small_males[i]), 0.0)
                moved[i] = task.repair(small_males[i] + shift)
            small_males = moved
            small_objectives = np.array([task.eval(x) for x in small_males])
        else:
            small_objectives = np.empty((0, task.n_objectives))

        population = np.vstack([big_males, female[None, :], small_males])
        objectives = np.vstack([big_objectives, female_objectives[None, :],
                                small_objectives])
        order = pareto_sort_indices(objectives)
        return population[order], objectives[order], []


class ZDT1(MultiObjectiveProblem):
    """Benchmark whose true Pareto front is f2 = 1 - sqrt(f1)."""

    def __init__(self, dimension=10):
        super().__init__(dimension, n_objectives=2, lower=0.0, upper=1.0,
                         objective_names=["f1", "f2"])

    def _evaluate(self, x):
        g = 1.0 + 9.0 * np.mean(x[1:])
        return np.array([x[0], g * (1.0 - np.sqrt(x[0] / g))])


if __name__ == "__main__":
    from ikn_library.algorithms import NSGA2

    for name, algorithm in [("MO-KMA", MOKomodoMlipir(population_size=40, seed=42)),
                            ("NSGA-II", NSGA2(population_size=40, seed=42))]:
        task = MultiObjectiveTask(problem=ZDT1(10), max_evals=20000)
        _, objectives = algorithm.run(task)
        error = np.abs(objectives[:, 1] - (1.0 - np.sqrt(objectives[:, 0])))
        print(f"{name:<8}: {len(objectives):3d} solutions | "
              f"mean distance to the true front = {error.mean():.4f} | "
              f"f1 coverage [{objectives[:, 0].min():.2f}, "
              f"{objectives[:, 0].max():.2f}]")
