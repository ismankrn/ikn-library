"""Fireworks Algorithm for continuous search spaces.

Reference:
    Y. Tan and Y. Zhu, "Fireworks algorithm for optimization," in
    Advances in Swarm Intelligence (ICSI 2010), Lecture Notes in
    Computer Science 6145, Springer, 355-364, 2010;
    S. Zheng, A. Janecek, and Y. Tan, "Enhanced fireworks algorithm," in
    IEEE Congress on Evolutionary Computation (CEC 2013), 2069-2077,
    2013.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm

EPSILON = 1e-38


class FireworksAlgorithm(Algorithm):
    """Fireworks Algorithm (Tan & Zhu, 2010).

    Each firework is a candidate solution that **explodes**, scattering
    sparks around itself. Two quantities are set by the firework's
    quality, and they pull in opposite directions:

    - **Number of sparks** grows with quality — good fireworks get more
      of the evaluation budget.
    - **Explosion amplitude** *shrinks* with quality — a good firework
      explodes tightly around itself (exploitation), while a poor one
      scatters sparks far and wide (exploration).

    That inverse coupling is what makes the algorithm distinctive: the
    same population simultaneously refines promising regions and
    searches unknown ones, with no global schedule deciding when to
    switch. A few **Gaussian sparks** add a second kind of mutation,
    and the next generation keeps the best solution plus a random
    sample of the rest.

    Args:
        population_size: Number of fireworks ``n``.
        n_sparks: Total sparks shared out among the fireworks each
            iteration.
        max_amplitude: Largest explosion amplitude, as a fraction of
            each dimension's bound range.
        n_gaussian_sparks: Gaussian mutation sparks per iteration.
        spark_bounds: ``(a, b)`` limiting each firework's share of the
            sparks to ``[a * n_sparks, b * n_sparks]``, so no single
            firework starves or monopolizes the budget.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=5, n_sparks=10, max_amplitude=0.5,
                 n_gaussian_sparks=5, spark_bounds=(0.04, 0.8), seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if n_sparks < 1:
            raise ValueError("n_sparks must be >= 1")
        if max_amplitude <= 0:
            raise ValueError("max_amplitude must be > 0")
        if n_gaussian_sparks < 0:
            raise ValueError("n_gaussian_sparks must be >= 0")
        low, high = spark_bounds
        if not 0.0 < low < high <= 1.0:
            raise ValueError("spark_bounds must satisfy 0 < a < b <= 1")
        self.n_sparks = int(n_sparks)
        self.max_amplitude = float(max_amplitude)
        self.n_gaussian_sparks = int(n_gaussian_sparks)
        self.spark_bounds = (float(low), float(high))

    def _spark_counts(self, fitness):
        """Sparks per firework: more for the good ones (Eq. 1)."""
        worst = fitness.max()
        shares = (worst - fitness + EPSILON) / (
            np.sum(worst - fitness) + EPSILON)
        counts = np.round(self.n_sparks * shares).astype(int)
        low, high = self.spark_bounds
        return np.clip(counts, max(1, int(low * self.n_sparks)),
                       int(high * self.n_sparks))

    def _amplitudes(self, fitness, span):
        """Explosion amplitude: *smaller* for the good ones (Eq. 2)."""
        best = fitness.min()
        shares = (fitness - best + EPSILON) / (
            np.sum(fitness - best) + EPSILON)
        return self.max_amplitude * shares[:, None] * span

    def run_iteration(self, task, state):
        fireworks, fitness = state
        span = task.upper - task.lower
        counts = self._spark_counts(fitness)
        amplitudes = self._amplitudes(fitness, span)

        candidates, candidate_fitness = list(fireworks), list(fitness)

        # Explosion sparks: displace a random subset of the dimensions.
        for i in range(self.population_size):
            for _ in range(counts[i]):
                if task.stopping_condition():
                    break
                spark = fireworks[i].copy()
                mask = self.rng.random(task.dimension) < 0.5
                if not mask.any():
                    mask[self.rng.integers(task.dimension)] = True
                spark[mask] += (amplitudes[i][mask]
                                * self.rng.uniform(-1.0, 1.0, int(mask.sum())))
                spark = task.repair(spark)
                candidates.append(spark)
                candidate_fitness.append(task.eval(spark))

        # Gaussian sparks: multiplicative mutation around a firework.
        for _ in range(self.n_gaussian_sparks):
            if task.stopping_condition():
                break
            source = self.rng.integers(self.population_size)
            spark = fireworks[source].copy()
            mask = self.rng.random(task.dimension) < 0.5
            if not mask.any():
                mask[self.rng.integers(task.dimension)] = True
            spark[mask] *= self.rng.normal(1.0, 1.0, int(mask.sum()))
            spark = task.repair(spark)
            candidates.append(spark)
            candidate_fitness.append(task.eval(spark))

        # Selection: keep the best, then a random sample of the rest —
        # the "elitism-random" scheme of the enhanced FWA, which avoids
        # the original's costly pairwise-distance roulette.
        candidates = np.array(candidates)
        candidate_fitness = np.array(candidate_fitness)
        best = int(np.argmin(candidate_fitness))
        others = np.delete(np.arange(len(candidates)), best)
        n_random = min(self.population_size - 1, len(others))
        chosen = np.concatenate([
            [best], self.rng.choice(others, n_random, replace=False)])
        return candidates[chosen], candidate_fitness[chosen]
