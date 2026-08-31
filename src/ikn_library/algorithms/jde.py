"""Self-Adaptive Differential Evolution (jDE).

Reference:
    J. Brest, S. Greiner, B. Boskovic, M. Mernik, and V. Zumer,
    "Self-adapting control parameters in differential evolution: a
    comparative study on numerical benchmark problems," IEEE
    Transactions on Evolutionary Computation, 10(6), 646-657, 2006.
"""

import numpy as np

from ikn_library.algorithms.de import DifferentialEvolution


class SelfAdaptiveDifferentialEvolution(DifferentialEvolution):
    """Self-Adaptive Differential Evolution, jDE (Brest et al., 2006).

    Differential Evolution with its two hardest-to-set parameters
    removed from the user's hands. Each individual carries its **own**
    \\(F_i\\) and \\(CR_i\\), which are occasionally re-drawn at random
    and then **kept only if the trial they produced won**:

    \\[
    F_i \\leftarrow F_l + \\rho (F_u - F_l)
    \\ \\text{ if } \\text{rand} < \\tau_1,
    \\qquad
    CR_i \\leftarrow \\text{rand}
    \\ \\text{ if } \\text{rand} < \\tau_2
    \\]

    That last condition is the whole mechanism, and it is easy to miss.
    The parameters are not adapted by any rule or heuristic — they ride
    along with the solution they generated. Values that produce winning
    trials survive into the next generation with their offspring; values
    that produce losers are discarded along with them. Selection does
    the tuning, on the parameters and the solutions at once.

    This is the scheme the Hybrid Self-Adaptive Bat Algorithm
    borrows, applied here to the algorithm it was designed for.

    Args:
        population_size: Number of individuals ``NP``.
        min_weight: Lower end \\(F_l\\) of the range ``F`` is drawn from.
        max_weight: Upper end \\(F_u\\) of that range.
        tau_1: Probability of re-drawing an individual's ``F``.
        tau_2: Probability of re-drawing an individual's ``CR``.
        strategy: Mutation strategy, as in
            :class:`DifferentialEvolution`. jDE was published with
            ``"rand/1"``.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, min_weight=0.1, max_weight=0.9,
                 tau_1=0.1, tau_2=0.1, strategy="rand/1", seed=None):
        # F and CR are per-individual here; the inherited scalars are
        # only fallbacks and are never used by run_iteration.
        super().__init__(population_size=population_size,
                         differential_weight=max_weight,
                         crossover_rate=0.5, strategy=strategy, seed=seed)
        if min_weight <= 0:
            raise ValueError("min_weight must be > 0")
        if max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight")
        if not 0.0 <= tau_1 <= 1.0:
            raise ValueError("tau_1 must be in [0, 1]")
        if not 0.0 <= tau_2 <= 1.0:
            raise ValueError("tau_2 must be in [0, 1]")
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.tau_1 = float(tau_1)
        self.tau_2 = float(tau_2)

    def init_population(self, task):
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in population])
        weights = self.rng.uniform(self.min_weight, self.max_weight,
                                   self.population_size)
        rates = self.rng.random(self.population_size)
        return population, fitness, weights, rates

    def _propose_parameters(self, weight, rate):
        """Re-draw F and CR with probability tau_1 and tau_2.

        The proposals are used for this trial whether or not they are
        eventually kept; only a winning trial makes them permanent.
        """
        if self.rng.random() < self.tau_1:
            weight = self.min_weight + self.rng.random() * (
                self.max_weight - self.min_weight)
        if self.rng.random() < self.tau_2:
            rate = self.rng.random()
        return weight, rate

    def run_iteration(self, task, state):
        population, fitness, weights, rates = state
        best_index = int(np.argmin(fitness))

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            weight, rate = self._propose_parameters(weights[i], rates[i])

            mutant = self._mutant(population, i, best_index, f=weight)
            trial = task.repair(self._crossover(population[i], mutant,
                                                task.dimension, cr=rate))
            trial_fitness = task.eval(trial)

            if trial_fitness <= fitness[i]:
                population[i], fitness[i] = trial, trial_fitness
                # The parameters are inherited only by a winning trial.
                weights[i], rates[i] = weight, rate
                if trial_fitness < fitness[best_index]:
                    best_index = i

        return population, fitness, weights, rates
