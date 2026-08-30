"""Example: a multi-objective Simulated Annealing (MOSA).

Single-solution algorithms convert differently from population-based
ones. SA has no population to rank, so ``pareto_sort_indices`` does not
apply; what changes is the **acceptance rule**. Pareto dominance splits
a candidate into three cases instead of two:

1. the candidate **dominates** the current solution — always accept;
2. the current solution **dominates** the candidate — accept with the
   Metropolis probability, using an aggregated worsening as delta;
3. **neither dominates** the other (a trade-off) — accept, because such
   a move travels *along* the front rather than away from it.

Case 3 has no single-objective counterpart, and it is what lets one
wandering solution map out a whole front. The Pareto archive itself is
already handled by ``MultiObjectiveTask``.

Reference: B. Suman and P. Kumar, "A survey of simulated annealing as a
tool for single and multiobjective optimization," Journal of the
Operational Research Society, 57(10), 1143-1160, 2006.
"""

import numpy as np

from ikn_library.algorithms import SimulatedAnnealing
from ikn_library.multiobjective import (
    MultiObjectiveProblem,
    MultiObjectiveTask,
    dominates,
)


class MOSimulatedAnnealing(SimulatedAnnealing):
    """Simulated Annealing with a Pareto-dominance acceptance rule."""

    def init_population(self, task):
        x = self.rng.uniform(task.lower, task.upper)
        return x, task.eval(x), self.initial_temperature

    @staticmethod
    def _worsening(candidate, current):
        """Aggregate how much worse the candidate is, for Metropolis.

        Only the objectives that got worse count, averaged and scaled by
        the current values so objectives on different scales contribute
        comparably.
        """
        scale = np.maximum(np.abs(current), 1e-12)
        return float(np.mean(np.maximum(candidate - current, 0.0) / scale))

    def run_iteration(self, task, state):
        x, current, temperature = state
        step_fraction = np.sqrt(max(temperature / self.initial_temperature, 1e-12))
        scale = self.step_size * (task.upper - task.lower) * step_fraction
        candidate = task.repair(x + self.rng.normal(0.0, scale))
        candidate_objectives = task.eval(candidate)

        if dominates(candidate_objectives, current):
            x, current = candidate, candidate_objectives          # case 1
        elif dominates(current, candidate_objectives):
            delta = self._worsening(candidate_objectives, current)
            if self.rng.random() < np.exp(-delta / temperature):   # case 2
                x, current = candidate, candidate_objectives
        else:
            x, current = candidate, candidate_objectives           # case 3

        return x, current, temperature * self.cooling


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

    algorithms = [
        ("MOSA", MOSimulatedAnnealing(cooling=0.9995, step_size=0.2, seed=42)),
        ("NSGA-II", NSGA2(population_size=40, seed=42)),
    ]
    for name, algorithm in algorithms:
        task = MultiObjectiveTask(problem=ZDT1(10), max_evals=20000)
        _, objectives = algorithm.run(task)
        error = np.abs(objectives[:, 1] - (1.0 - np.sqrt(objectives[:, 0])))
        print(f"{name:<8}: {len(objectives):3d} solutions | "
              f"mean distance to the true front = {error.mean():.4f} | "
              f"f1 coverage [{objectives[:, 0].min():.2f}, "
              f"{objectives[:, 0].max():.2f}]")
