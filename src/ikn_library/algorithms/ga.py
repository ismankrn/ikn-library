"""Real-coded Genetic Algorithm for continuous search spaces.

Reference:
    J. H. Holland, "Adaptation in Natural and Artificial Systems,"
    University of Michigan Press, 1975. Blend crossover follows
    L. J. Eshelman and J. D. Schaffer, "Real-coded genetic algorithms
    and interval-schemata," Foundations of Genetic Algorithms 2, 1993.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class GeneticAlgorithm(Algorithm):
    """Real-coded Genetic Algorithm (GA) with tournament selection,
    blend crossover (BLX-alpha), Gaussian mutation, and elitism.

    Each generation: the best ``elitism`` individuals survive unchanged;
    the rest of the population is refilled by repeatedly picking two
    parents via tournament selection, recombining them with blend
    crossover (children are sampled from an interval slightly wider
    than the one their parents span), and mutating each gene with a
    small probability by adding Gaussian noise. The mutation step
    shrinks linearly as the evaluation budget is consumed (non-uniform
    mutation), so early generations explore widely and late generations
    refine locally.

    Args:
        population_size: Number of individuals per generation.
        crossover_rate: Probability that a selected pair is recombined
            (otherwise the parents are copied).
        mutation_rate: Per-gene mutation probability. Defaults to
            ``1 / dimension``, so on average one gene mutates per child.
        mutation_scale: Initial mutation step as a fraction of each
            dimension's bound range; decays with search progress.
        tournament_size: Number of contestants per tournament; larger
            values increase selection pressure.
        blend_alpha: BLX-alpha expansion factor of the crossover
            interval beyond the parents' values.
        elitism: Number of best individuals carried over unchanged.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, crossover_rate=0.9,
                 mutation_rate=None, mutation_scale=0.1, tournament_size=2,
                 blend_alpha=0.5, elitism=1, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if mutation_rate is not None and not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if mutation_scale <= 0:
            raise ValueError("mutation_scale must be > 0")
        if tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if blend_alpha < 0:
            raise ValueError("blend_alpha must be >= 0")
        if not 0 <= elitism < population_size:
            raise ValueError("elitism must be in [0, population_size)")
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = mutation_rate
        self.mutation_scale = float(mutation_scale)
        self.tournament_size = int(tournament_size)
        self.blend_alpha = float(blend_alpha)
        self.elitism = int(elitism)

    def _tournament(self, population, fitness):
        contestants = self.rng.integers(0, self.population_size, self.tournament_size)
        winner = contestants[np.argmin(fitness[contestants])]
        return population[winner]

    def _blend_crossover(self, parent1, parent2):
        low = np.minimum(parent1, parent2)
        high = np.maximum(parent1, parent2)
        spread = self.blend_alpha * (high - low)
        return (self.rng.uniform(low - spread, high + spread),
                self.rng.uniform(low - spread, high + spread))

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return task.evals / task.max_evals
        if np.isfinite(task.max_iters):
            return task.iters / task.max_iters
        return 0.0

    def _mutate(self, child, task, mutation_rate, scale):
        mask = self.rng.random(task.dimension) < mutation_rate
        noise = self.rng.normal(0.0, scale * (task.upper - task.lower))
        return np.where(mask, child + noise, child)

    def run_iteration(self, task, state):
        population, fitness = state
        mutation_rate = (1.0 / task.dimension if self.mutation_rate is None
                         else self.mutation_rate)
        # Non-uniform mutation: the step decays linearly with progress.
        scale = self.mutation_scale * max(1.0 - self._progress(task), 1e-3)

        order = np.argsort(fitness)
        offspring = [population[i].copy() for i in order[:self.elitism]]
        offspring_fitness = [fitness[i] for i in order[:self.elitism]]

        while len(offspring) < self.population_size:
            parent1 = self._tournament(population, fitness)
            parent2 = self._tournament(population, fitness)
            if self.rng.random() < self.crossover_rate:
                children = self._blend_crossover(parent1, parent2)
            else:
                children = (parent1.copy(), parent2.copy())
            for child in children:
                if len(offspring) >= self.population_size:
                    break
                child = task.repair(self._mutate(child, task, mutation_rate, scale))
                offspring.append(child)
                offspring_fitness.append(task.eval(child))

        return np.array(offspring), np.array(offspring_fitness)
