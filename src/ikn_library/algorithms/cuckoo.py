"""Cuckoo Search via Lévy flights.

Reference:
    X.-S. Yang and S. Deb, "Cuckoo search via Lévy flights," in
    Proceedings of the World Congress on Nature and Biologically
    Inspired Computing (NaBIC 2009), IEEE, 210-214, 2009;
    X.-S. Yang and S. Deb, "Engineering optimisation by cuckoo search,"
    International Journal of Mathematical Modelling and Numerical
    Optimisation, 1(4), 330-343, 2010.
"""

import math

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class CuckooSearch(Algorithm):
    """Cuckoo Search (Yang & Deb, 2009), based on brood parasitism.

    Each nest holds one egg — a candidate solution. Every iteration a
    cuckoo lays a new egg by taking a **Lévy flight** from an existing
    nest, and drops it into a randomly chosen nest, which keeps it only
    if it is better. Afterwards a fraction ``discovery_rate`` of the
    worst nests are "discovered" by the host birds and abandoned, being
    rebuilt elsewhere.

    The Lévy flight is what distinguishes this algorithm: its
    heavy-tailed step distribution produces mostly small moves with the
    occasional very long jump, so the search refines locally while
    still being able to leap into unexplored regions — a balance the
    Gaussian steps used by most other algorithms cannot provide.

    Args:
        population_size: Number of nests.
        discovery_rate: Fraction ``pa`` of nests abandoned each
            iteration, in [0, 1).
        step_size: Scale ``alpha`` of the Lévy step, as a fraction of
            each dimension's bound range.
        levy_exponent: Lévy distribution exponent ``beta`` in (0, 2];
            smaller values produce longer jumps.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=15, discovery_rate=0.5,
                 step_size=0.01, levy_exponent=1.5, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 <= discovery_rate < 1.0:
            raise ValueError("discovery_rate must be in [0, 1)")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if not 0.0 < levy_exponent <= 2.0:
            raise ValueError("levy_exponent must be in (0, 2]")
        self.discovery_rate = float(discovery_rate)
        self.step_size = float(step_size)
        self.levy_exponent = float(levy_exponent)

    def _levy_flight(self, size):
        """Lévy-distributed steps via Mantegna's algorithm.

        ``u / |v|^(1/beta)`` with normally distributed ``u`` and ``v``
        reproduces a Lévy distribution of exponent ``beta``: many small
        steps, rare very large ones.
        """
        beta = self.levy_exponent
        sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
                 / (math.gamma((1.0 + beta) / 2.0) * beta
                    * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)
        u = self.rng.normal(0.0, sigma, size)
        v = self.rng.normal(0.0, 1.0, size)
        return u / np.abs(v) ** (1.0 / beta)

    def run_iteration(self, task, state):
        nests, fitness = state
        span = task.upper - task.lower

        # A cuckoo lays an egg after a Lévy flight, and drops it into a
        # random nest that keeps it only if it is better. The step is
        # scaled to the search range rather than to the distance from
        # the best nest: the latter collapses to zero once the nests
        # converge, which stalls the search entirely.
        for i in range(self.population_size):
            if task.stopping_condition():
                break
            step = self.step_size * span * self._levy_flight(task.dimension)
            candidate = task.repair(nests[i] + step)
            candidate_fitness = task.eval(candidate)
            target = self.rng.integers(self.population_size)
            if candidate_fitness < fitness[target]:
                nests[target], fitness[target] = candidate, candidate_fitness

        # Host birds discover a fraction of the worst nests, which are
        # rebuilt by a biased random walk between two other nests.
        n_abandoned = int(self.discovery_rate * self.population_size)
        if n_abandoned > 0:
            worst = np.argsort(fitness)[-n_abandoned:]
            for i in worst:
                if task.stopping_condition():
                    break
                j, k = self.rng.integers(self.population_size, size=2)
                candidate = task.repair(
                    nests[i] + self.rng.random(task.dimension) * (nests[j] - nests[k])
                )
                candidate_fitness = task.eval(candidate)
                if candidate_fitness < fitness[i]:
                    nests[i], fitness[i] = candidate, candidate_fitness

        return nests, fitness
