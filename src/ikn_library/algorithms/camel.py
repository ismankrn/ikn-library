"""Camel Algorithm (CA) for continuous search spaces.

Reference:
    R. M. Ali, "Novel optimization algorithm inspired by camel traveler
    behavior," International Journal of Sciences: Basic and Applied
    Research, 2016. See also M. K. Ibrahim et al., "Camel Algorithm"
    follow-up studies.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class CamelAlgorithm(Algorithm):
    """Camel Algorithm (Ali, 2016), a caravan crossing the desert.

    Each camel is a candidate solution travelling toward the best
    oasis found so far. Three state variables shape its journey:

    - **Temperature** \\(T\\), drawn randomly each step, models the
      desert heat a camel happens to face.
    - **Endurance** \\(E\\) falls with the temperature and with the
      distance already travelled. A strong camel strides far, an
      exhausted one shuffles — so the step is scaled *by* \\(E\\), which
      also makes the caravan refine its search as the journey ends.
    - **Supply** \\(S\\) (water and food) is consumed at the burden
      rate; as it runs low the camel searches more desperately, which
      *lengthens* its steps and counteracts the fading endurance.

    A camel that finds a better position has reached an **oasis**: its
    endurance and supply are replenished. One that exhausts its
    endurance **dies** and is reborn at a random location — the
    algorithm's restart mechanism.

    Args:
        population_size: Number of camels in the caravan.
        min_temperature: Lower bound of the random temperature.
        max_temperature: Upper bound of the random temperature.
        burden_rate: Rate \\(\\omega\\) at which supply is consumed over
            the journey, in [0, 1].
        death_rate: Fraction of the initial endurance below which a
            camel dies and is reborn, in [0, 1).
        visibility: Scale of the step toward the best solution.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, min_temperature=-1.0,
                 max_temperature=1.0, burden_rate=0.9, death_rate=0.01,
                 visibility=1.0, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if min_temperature >= max_temperature:
            raise ValueError("min_temperature must be < max_temperature")
        if max_temperature <= 0:
            raise ValueError("max_temperature must be > 0")
        if not 0.0 <= burden_rate <= 1.0:
            raise ValueError("burden_rate must be in [0, 1]")
        if not 0.0 <= death_rate < 1.0:
            raise ValueError("death_rate must be in [0, 1)")
        if visibility <= 0:
            raise ValueError("visibility must be > 0")
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.burden_rate = float(burden_rate)
        self.death_rate = float(death_rate)
        self.visibility = float(visibility)

    def init_population(self, task):
        caravan = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in caravan])
        endurance = np.ones(self.population_size)   # E_0 = 1
        supply = np.ones(self.population_size)       # S_0 = 1
        return caravan, fitness, endurance, supply

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def run_iteration(self, task, state):
        caravan, fitness, endurance, supply = state
        travelled = self._progress(task)
        best = task.best_x

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            # Heat of the day, and how it wears the camel down.
            temperature = self.rng.uniform(self.min_temperature,
                                           self.max_temperature)
            endurance[i] = (1.0 - temperature / self.max_temperature) * \
                           (1.0 - travelled)
            supply[i] = 1.0 - self.burden_rate * travelled

            # Step toward the best oasis: endurance sets how far the
            # camel can stride, dwindling supplies push it further.
            delta = self.rng.uniform(-1.0, 1.0, task.dimension)
            step = (self.visibility * delta * endurance[i]
                    * np.exp(1.0 - supply[i]) * (best - caravan[i]))
            candidate = task.repair(caravan[i] + step)
            candidate_fitness = task.eval(candidate)

            if candidate_fitness < fitness[i]:
                # An oasis: move there and replenish.
                caravan[i], fitness[i] = candidate, candidate_fitness
                endurance[i], supply[i] = 1.0, 1.0
            elif endurance[i] < self.death_rate:
                # The camel dies of exhaustion and is reborn elsewhere.
                caravan[i] = self.rng.uniform(task.lower, task.upper)
                fitness[i] = task.eval(caravan[i])
                endurance[i], supply[i] = 1.0, 1.0

        return caravan, fitness, endurance, supply
