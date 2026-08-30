"""Komodo Mlipir Algorithm (KMA) for continuous search spaces.

Reference:
    S. Suyanto, A. A. Ariyanto, and A. F. Ariyanto, "Komodo Mlipir
    Algorithm," Applied Soft Computing, 114, 108043, 2022.
    https://doi.org/10.1016/j.asoc.2021.108043
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class KomodoMlipirAlgorithm(Algorithm):
    """Komodo Mlipir Algorithm (Suyanto et al., 2022).

    KMA takes its two ideas from Komodo dragons and from *mlipir*, a
    Javanese word for walking along the side of the road to arrive
    safely. Every iteration the ranked population is split into three
    groups that play different roles:

    - **Big males** (the ``q`` best) perform *high-exploitation
      low-exploration* (HILE): each is attracted to higher-quality big
      males, while a big male may also be pushed away from a
      lower-quality one with probability 0.5.
    - **The female** (middle quality) either mates with the winning big
      male — an arithmetic crossover producing two offspring, of which
      the better is kept — or reproduces by **parthenogenesis**, a small
      random step; each with probability 0.5.
    - **Small males** (the rest) move *mlipir*: they follow the big
      males in only a random subset of dimensions, chosen with
      probability ``mlipir_rate``, which diversifies the population.

    The population size itself **self-adapts**: it shrinks while the
    best-so-far fitness keeps improving and grows again when the search
    stagnates.

    Args:
        population_size: Initial number of Komodo individuals.
        big_male_portion: Portion ``p`` of the population that becomes
            big males, in (0, 1).
        mlipir_rate: Probability ``d`` that a small male follows a big
            male in a given dimension, in (0, 1).
        max_big_males: Cap on the number of big males. The paper notes
            that "two or three big males will give an optimum
            interaction"; without the cap, a big male's step (the sum
            over all other big males) grows with the group and
            destroys convergence.
        adaptation_step: Individuals added or removed when the
            population adapts (``a`` in the paper).
        min_population: Lower limit of the adaptive population size.
        max_population: Upper limit of the adaptive population size.
        parthenogenesis_radius: Radius ``alpha`` of the female's asexual
            step, as a fraction of each dimension's bound range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=15, big_male_portion=0.5,
                 mlipir_rate=0.5, max_big_males=3, adaptation_step=5,
                 min_population=10, max_population=200,
                 parthenogenesis_radius=0.1, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 < big_male_portion < 1.0:
            raise ValueError("big_male_portion must be in (0, 1)")
        if max_big_males < 2:
            raise ValueError("max_big_males must be >= 2")
        if not 0.0 < mlipir_rate < 1.0:
            raise ValueError("mlipir_rate must be in (0, 1)")
        if adaptation_step < 1:
            raise ValueError("adaptation_step must be >= 1")
        if min_population < 5:
            raise ValueError("min_population must be >= 5")
        if max_population < min_population:
            raise ValueError("max_population must be >= min_population")
        if parthenogenesis_radius <= 0:
            raise ValueError("parthenogenesis_radius must be > 0")
        self.big_male_portion = float(big_male_portion)
        self.mlipir_rate = float(mlipir_rate)
        self.max_big_males = int(max_big_males)
        self.adaptation_step = int(adaptation_step)
        self.min_population = int(min_population)
        self.max_population = int(max_population)
        self.parthenogenesis_radius = float(parthenogenesis_radius)

    def init_population(self, task):
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in population])
        order = np.argsort(fitness)
        # history holds the last three best-so-far fitness values, used
        # by the population adaptation scheme.
        return population[order], fitness[order], [task.best_fitness]

    def _group_sizes(self, n_individuals):
        """Split ``n`` individuals into big males, one female, small males.

        The number of big males is capped at ``max_big_males``: because
        a big male's step is the *sum* of its interactions with all the
        others (Eq. 4), a large group produces enormous steps that wreck
        convergence. The paper states that "two or three big males will
        give an optimum interaction", which this cap enforces.
        """
        n_big = int(self.big_male_portion * (n_individuals - 1))
        n_big = max(2, min(n_big, self.max_big_males, n_individuals - 3))
        return n_big, n_individuals - n_big - 1

    def _move_big_males(self, task, big_males, big_fitness):
        """HILE: attraction to better males, random distraction from worse."""
        n_big = len(big_males)
        moved = np.empty_like(big_males)
        for i in range(n_big):
            shift = np.zeros(task.dimension)
            for j in range(n_big):
                if i == j:
                    continue
                r1 = self.rng.random(task.dimension)
                attract = big_fitness[j] < big_fitness[i] or self.rng.random() < 0.5
                direction = (big_males[j] - big_males[i] if attract
                             else big_males[i] - big_males[j])
                shift += r1 * direction
            moved[i] = task.repair(big_males[i] + shift)
        moved_fitness = np.array([task.eval(x) for x in moved])

        # Keep the q best positions among the old and the new ones.
        merged = np.vstack([big_males, moved])
        merged_fitness = np.concatenate([big_fitness, moved_fitness])
        keep = np.argsort(merged_fitness)[:n_big]
        return merged[keep], merged_fitness[keep]

    def _mate(self, task, winner, female):
        """Arithmetic crossover; the better of the two offspring wins."""
        r = self.rng.random(task.dimension)
        offspring = np.vstack([
            task.repair(r * winner + (1.0 - r) * female),
            task.repair(r * female + (1.0 - r) * winner),
        ])
        offspring_fitness = np.array([task.eval(x) for x in offspring])
        best = int(np.argmin(offspring_fitness))
        return offspring[best], offspring_fitness[best]

    def _parthenogenesis(self, task, female):
        """Asexual reproduction: a small symmetric random step."""
        r = self.rng.random(task.dimension)
        step = (2.0 * r - 1.0) * self.parthenogenesis_radius * (task.upper - task.lower)
        candidate = task.repair(female + step)
        return candidate, task.eval(candidate)

    def _move_small_males(self, task, small_males, big_males):
        """Mlipir: follow the big males in a random subset of dimensions."""
        moved = np.empty_like(small_males)
        for i in range(len(small_males)):
            shift = np.zeros(task.dimension)
            for big_male in big_males:
                follow = self.rng.random(task.dimension) < self.mlipir_rate
                r1 = self.rng.random(task.dimension)
                shift += np.where(follow, r1 * (big_male - small_males[i]), 0.0)
            moved[i] = task.repair(small_males[i] + shift)
        return moved, np.array([task.eval(x) for x in moved])

    def _adapt_population(self, task, population, fitness, history):
        """Shrink while improving, grow while stagnating (Eq. 10)."""
        history = [*history, task.best_fitness][-3:]
        if len(history) < 3:
            return population, fitness, history

        f1, f2, f3 = history[2], history[1], history[0]
        improved = not (np.isclose(f1, f2) or np.isclose(f2, f3))
        stagnated = np.isclose(f1, f2) and np.isclose(f2, f3)
        n_individuals = len(population)

        if improved and n_individuals - self.adaptation_step >= self.min_population:
            keep = np.argsort(fitness)[: n_individuals - self.adaptation_step]
            return population[keep], fitness[keep], history
        if stagnated and n_individuals + self.adaptation_step <= self.max_population:
            # New individuals are random moves of the best-so-far Komodo.
            best = population[np.argmin(fitness)]
            span = task.upper - task.lower
            newcomers = np.array([
                task.repair(best + (2.0 * self.rng.random(task.dimension) - 1.0)
                            * self.parthenogenesis_radius * span)
                for _ in range(self.adaptation_step)
            ])
            newcomer_fitness = np.array([task.eval(x) for x in newcomers])
            return (np.vstack([population, newcomers]),
                    np.concatenate([fitness, newcomer_fitness]), history)
        return population, fitness, history

    def run_iteration(self, task, state):
        population, fitness, history = state
        n_big, n_small = self._group_sizes(len(population))

        big_males, big_fitness = population[:n_big], fitness[:n_big]
        female, female_fitness = population[n_big].copy(), fitness[n_big]
        small_males = population[n_big + 1:]

        big_males, big_fitness = self._move_big_males(task, big_males, big_fitness)

        winner = big_males[int(np.argmin(big_fitness))]
        if self.rng.random() < 0.5:
            candidate, candidate_fitness = self._mate(task, winner, female)
        else:
            candidate, candidate_fitness = self._parthenogenesis(task, female)
        if candidate_fitness < female_fitness:
            female, female_fitness = candidate, candidate_fitness

        if n_small > 0:
            small_males, small_fitness = self._move_small_males(
                task, small_males, big_males)
        else:
            small_fitness = np.empty(0)

        population = np.vstack([big_males, female[None, :], small_males])
        fitness = np.concatenate([big_fitness, [female_fitness], small_fitness])

        population, fitness, history = self._adapt_population(
            task, population, fitness, history)
        order = np.argsort(fitness)
        return population[order], fitness[order], history
