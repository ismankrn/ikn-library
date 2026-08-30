"""Differential Evolution for continuous search spaces.

Reference:
    R. Storn and K. Price, "Differential evolution — a simple and
    efficient heuristic for global optimization over continuous
    spaces," Journal of Global Optimization, 11(4), 341-359, 1997.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm

STRATEGIES = ("rand/1", "best/1", "rand/2", "current-to-best/1")


class DifferentialEvolution(Algorithm):
    """Differential Evolution (Storn & Price, 1997).

    DE's central idea is beautifully simple: build a mutant by adding
    the **scaled difference between two population members** to a third
    one. Because the difference vectors come from the population
    itself, the step size adapts automatically — large while the
    population is spread out, tiny once it converges, with no schedule
    to tune.

    Each generation, every individual ``x_i`` (the *target*) produces a
    mutant, mixes it with itself by binomial crossover into a *trial*
    vector, and is replaced by that trial only if the trial is better.

    Args:
        population_size: Number of individuals ``NP``.
        differential_weight: Scale ``F`` of the difference vector,
            typically in [0.4, 1.0].
        crossover_rate: Probability ``CR`` that a gene comes from the
            mutant rather than the target, in [0, 1].
        strategy: Mutation strategy:

            - ``"best/1"`` (default) — ``x_best + F(x_r1 - x_r2)``;
              fastest convergence, and the strongest of the four on
              this library's benchmarks.
            - ``"rand/1"`` — ``x_r1 + F(x_r2 - x_r3)``; the classic,
              more exploratory choice, better suited to very long runs.
            - ``"rand/2"`` — ``x_r1 + F(x_r2 - x_r3) + F(x_r4 - x_r5)``;
              even more exploratory.
            - ``"current-to-best/1"`` —
              ``x_i + F(x_best - x_i) + F(x_r1 - x_r2)``; a compromise.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, differential_weight=0.6,
                 crossover_rate=0.5, strategy="best/1", seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if population_size < 5:
            raise ValueError("population_size must be >= 5 for DE strategies")
        if differential_weight <= 0:
            raise ValueError("differential_weight must be > 0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.differential_weight = float(differential_weight)
        self.crossover_rate = float(crossover_rate)
        self.strategy = strategy

    def _pick(self, exclude, count):
        """``count`` distinct indices, none of them ``exclude``."""
        choices = self.rng.permutation(self.population_size)
        return [i for i in choices if i != exclude][:count]

    def _mutant(self, population, index, best_index):
        f = self.differential_weight
        if self.strategy == "rand/1":
            r1, r2, r3 = self._pick(index, 3)
            return population[r1] + f * (population[r2] - population[r3])
        if self.strategy == "best/1":
            r1, r2 = self._pick(index, 2)
            return population[best_index] + f * (population[r1] - population[r2])
        if self.strategy == "rand/2":
            r1, r2, r3, r4, r5 = self._pick(index, 5)
            return (population[r1]
                    + f * (population[r2] - population[r3])
                    + f * (population[r4] - population[r5]))
        # current-to-best/1
        r1, r2 = self._pick(index, 2)
        return (population[index]
                + f * (population[best_index] - population[index])
                + f * (population[r1] - population[r2]))

    def _crossover(self, target, mutant, dimension):
        """Binomial crossover, with one gene always taken from the mutant."""
        take = self.rng.random(dimension) < self.crossover_rate
        take[self.rng.integers(dimension)] = True   # guarantees a real trial
        return np.where(take, mutant, target)

    def run_iteration(self, task, state):
        population, fitness = state
        best_index = int(np.argmin(fitness))

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            mutant = self._mutant(population, i, best_index)
            trial = task.repair(self._crossover(population[i], mutant,
                                                task.dimension))
            trial_fitness = task.eval(trial)
            # Greedy selection: the trial replaces its own target only.
            if trial_fitness <= fitness[i]:
                population[i], fitness[i] = trial, trial_fitness
                if trial_fitness < fitness[best_index]:
                    best_index = i

        return population, fitness
