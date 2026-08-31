"""Flower Pollination Algorithm for continuous search spaces.

Reference:
    X.-S. Yang, "Flower pollination algorithm for global optimization,"
    in Unconventional Computation and Natural Computation, LNCS 7445,
    Springer, 240-249, 2012.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm
from ikn_library.algorithms.levy import levy_flight


class FlowerPollinationAlgorithm(Algorithm):
    """Flower Pollination Algorithm (Yang, 2012).

    Modeled on how flowering plants reproduce, and notable for being one
    of the simplest algorithms in this library: each flower flips a coin
    every iteration and applies **one of two rules**.

    - **Global pollination** (probability ``switch_probability``) —
      pollen is carried long distances by insects and birds, modelled as
      a Lévy flight aimed at the current best flower. Heavy-tailed steps
      mean mostly small moves with rare long jumps.
    - **Local pollination** (otherwise) — pollen moves between two
      neighbouring flowers, modelled as a scaled difference between two
      randomly chosen population members.

    A new flower replaces its parent only if it is better, so the
    population never worsens.

    The two rules are worth recognising: the global one is the same
    Lévy-toward-best move that drives Cuckoo Search, and the local one
    is a difference vector of the kind Differential Evolution is built
    on, without any crossover. FPA's contribution is not a new operator
    but the observation that switching between a long-range and a
    short-range rule, per individual and per iteration, is enough to
    search well.

    Args:
        population_size: Number of flowers.
        switch_probability: Chance of global rather than local
            pollination. Yang recommends 0.8; 0.5 measures better here,
            see the algorithm page.
        gamma: Scaling of the global pollination step.
        levy_exponent: Lévy distribution exponent ``beta`` in (0, 2];
            lower values give heavier tails and longer jumps.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=15, switch_probability=0.5,
                 gamma=0.5, levy_exponent=1.5, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 <= switch_probability <= 1.0:
            raise ValueError("switch_probability must be in [0, 1]")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if not 0.0 < levy_exponent <= 2.0:
            raise ValueError("levy_exponent must be in (0, 2]")
        self.switch_probability = float(switch_probability)
        self.gamma = float(gamma)
        self.levy_exponent = float(levy_exponent)

    def _levy_flight(self, size):
        """Lévy-distributed steps of exponent ``levy_exponent``."""
        return levy_flight(self.rng, size, self.levy_exponent)

    def run_iteration(self, task, state):
        flowers, fitness = state
        best = flowers[np.argmin(fitness)]

        for i in range(self.population_size):
            if task.stopping_condition():
                break

            if self.rng.random() < self.switch_probability:
                # Global pollination: a Lévy flight toward the best flower.
                step = self._levy_flight(task.dimension)
                candidate = flowers[i] + self.gamma * step * (best - flowers[i])
            else:
                # Local pollination: drift along a neighbour difference.
                j, k = self.rng.choice(self.population_size, 2, replace=False)
                epsilon = self.rng.random()
                candidate = flowers[i] + epsilon * (flowers[j] - flowers[k])

            candidate = task.repair(candidate)
            candidate_fitness = task.eval(candidate)
            if candidate_fitness < fitness[i]:        # greedy: never worsen
                flowers[i] = candidate
                fitness[i] = candidate_fitness

        return flowers, fitness
