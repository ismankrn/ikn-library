"""Hybrid Self-Adaptive Bat Algorithm for continuous search spaces.

Reference:
    I. Fister Jr., S. Fong, J. Brest, and I. Fister, "A novel hybrid
    self-adaptive bat algorithm," The Scientific World Journal, 2014,
    709738, 2014.
"""

import numpy as np

from ikn_library.algorithms.hybrid_bat import HybridBatAlgorithm


class HybridSelfAdaptiveBatAlgorithm(HybridBatAlgorithm):
    """Hybrid Self-Adaptive Bat Algorithm (Fister Jr. et al., 2014).

    The Hybrid Bat Algorithm with its two control
    parameters made **self-adaptive**. Each bat carries its own loudness
    \\(A_i\\) and pulse rate \\(r_i\\), and instead of following a fixed
    schedule they are occasionally **re-drawn at random**:

    \\[
    A_i \\leftarrow A_{\\min} + \\rho (A_{\\max} - A_{\\min})
    \\ \\text{ if } \\text{rand} < \\tau_1,
    \\qquad
    r_i \\leftarrow \\text{rand}
    \\ \\text{ if } \\text{rand} < \\tau_2
    \\]

    This is the self-adaptation scheme of **jDE** (Brest et al., 2006)
    applied to the Bat Algorithm's parameters rather than to
    Differential Evolution's — Brest is a co-author of both.

    The mechanism is worth understanding structurally. In plain BA
    and in HBA, loudness only ever **decays**: each
    accepted move multiplies it by \\(\\alpha\\), so it slides toward
    zero and acceptance eventually becomes near-impossible. Re-drawing
    \\(A_i\\) removes that ratchet — a bat whose loudness has collapsed
    can be revived. The HBA page records that raising ``alpha`` from
    0.9 to 0.99 was worth eight orders of magnitude there; this
    algorithm addresses the same problem by design instead of by
    tuning.

    Args:
        population_size: Number of bats.
        tau_1: Probability of re-drawing a bat's loudness.
        tau_2: Probability of re-drawing a bat's pulse rate.
        min_loudness: Lower end of the loudness range.
        max_loudness: Upper end of the loudness range.
        differential_weight: DE scale factor ``F`` for the local step.
        crossover_rate: Binomial crossover probability ``CR``.
        min_frequency: Lower bound of the frequency range.
        max_frequency: Upper bound of the frequency range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, tau_1=0.1, tau_2=0.02,
                 min_loudness=0.99, max_loudness=1.0,
                 differential_weight=0.3, crossover_rate=0.9,
                 min_frequency=0.0, max_frequency=2.0, seed=None):
        super().__init__(population_size=population_size,
                         differential_weight=differential_weight,
                         crossover_rate=crossover_rate,
                         min_frequency=min_frequency,
                         max_frequency=max_frequency, seed=seed)
        if not 0.0 <= tau_1 <= 1.0:
            raise ValueError("tau_1 must be in [0, 1]")
        if not 0.0 <= tau_2 <= 1.0:
            raise ValueError("tau_2 must be in [0, 1]")
        if min_loudness <= 0:
            raise ValueError("min_loudness must be > 0")
        if max_loudness < min_loudness:
            raise ValueError("max_loudness must be >= min_loudness")
        self.tau_1 = float(tau_1)
        self.tau_2 = float(tau_2)
        self.min_loudness = float(min_loudness)
        self.max_loudness = float(max_loudness)

    def init_population(self, task):
        positions = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in positions])
        velocities = np.zeros_like(positions)
        loudness = self.rng.uniform(self.min_loudness, self.max_loudness,
                                    self.population_size)
        rates = self.rng.random(self.population_size)
        return positions, velocities, fitness, loudness, rates

    def _self_adapt(self, loudness, rates, index):
        """jDE-style re-draw: no decay, no ratchet.

        Both parameters are replaced outright with a fresh sample, so a
        bat that has become unable to accept anything can recover.
        """
        if self.rng.random() < self.tau_1:
            loudness[index] = self.rng.uniform(self.min_loudness,
                                               self.max_loudness)
        if self.rng.random() < self.tau_2:
            rates[index] = self.rng.random()

    def run_iteration(self, task, state):
        positions, velocities, fitness, loudness, rates = state
        best = task.best_x

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            self._self_adapt(loudness, rates, i)

            frequency = self.rng.uniform(self.min_frequency,
                                         self.max_frequency)
            velocities[i] += (best - positions[i]) * frequency
            candidate = positions[i] + velocities[i]

            # Each bat now has its own pulse rate rather than a shared one.
            if self.rng.random() > rates[i]:
                candidate = self._local_step(task, positions, i, best,
                                             walk_scale=None)

            candidate = task.repair(candidate)
            candidate_fitness = task.eval(candidate)
            if (candidate_fitness <= fitness[i]
                    and self.rng.random() < loudness[i]):
                positions[i] = candidate
                fitness[i] = candidate_fitness

        return positions, velocities, fitness, loudness, rates
