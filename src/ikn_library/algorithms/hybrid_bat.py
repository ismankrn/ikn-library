"""Hybrid Bat Algorithm for continuous search spaces.

Reference:
    I. Fister Jr., D. Fister, and X.-S. Yang, "A hybrid bat algorithm,"
    Elektrotehniski Vestnik / Electrotechnical Review, 80(1-2), 1-7,
    2013.
"""

import numpy as np

from ikn_library.algorithms.bat import BatAlgorithm


class HybridBatAlgorithm(BatAlgorithm):
    """Hybrid Bat Algorithm (Fister Jr., Fister & Yang, 2013).

    The Bat Algorithm with one operator swapped: its local
    random walk around the best solution is replaced by a **Differential
    Evolution move**, mutation plus binomial crossover.

    Everything else is unchanged — the frequency-tuned velocity, the
    growing pulse rate that decides when the local step fires, and the
    loudness-based acceptance. This is why the class subclasses
    ``BatAlgorithm`` and overrides a single method: the hybrid *is* the
    original with a different local search, and the code says so.

    The motivation is worth understanding. Plain BA's local step is a
    Gaussian walk around the current best, which explores only where the
    best already is and needs a hand-tuned step size. A DE move instead
    takes its scale from the **spread of the population**, so it
    contracts automatically as the search converges and needs no
    schedule at all.

    Args:
        population_size: Number of bats.
        differential_weight: DE scale factor ``F`` on the difference
            vector.
        crossover_rate: Binomial crossover probability ``CR``.
        loudness: Initial loudness ``A0`` (acceptance probability).
        pulse_rate: Final pulse emission rate ``r0``.
        alpha: Loudness decay factor per accepted improvement.
        gamma: Growth rate of the pulse rate over iterations.
        min_frequency: Lower bound of the frequency range.
        max_frequency: Upper bound of the frequency range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, differential_weight=0.3,
                 crossover_rate=0.9, loudness=1.0, pulse_rate=0.1,
                 alpha=0.99, gamma=0.9, min_frequency=0.0, max_frequency=2.0,
                 seed=None):
        # local_scale is inherited but unused: the DE move replaces the walk.
        super().__init__(population_size=population_size, loudness=loudness,
                         pulse_rate=pulse_rate, alpha=alpha, gamma=gamma,
                         min_frequency=min_frequency,
                         max_frequency=max_frequency, seed=seed)
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        if differential_weight <= 0:
            raise ValueError("differential_weight must be > 0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        self.differential_weight = float(differential_weight)
        self.crossover_rate = float(crossover_rate)

    def _local_step(self, task, positions, index, best, walk_scale):
        """DE/rand/1/bin in place of the Bat Algorithm's random walk.

        The step size comes from the population's own spread, so it
        shrinks as the bats converge without any decay schedule.
        """
        others = [k for k in range(self.population_size) if k != index]
        r1, r2, r3 = self.rng.choice(others, 3, replace=False)
        donor = (positions[r1] + self.differential_weight
                 * (positions[r2] - positions[r3]))

        crossover = self.rng.random(task.dimension) < self.crossover_rate
        # At least one coordinate always comes from the donor.
        crossover[self.rng.integers(task.dimension)] = True
        return np.where(crossover, donor, positions[index])
