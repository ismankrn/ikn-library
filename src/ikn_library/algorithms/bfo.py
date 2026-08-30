"""Bacterial Foraging Optimization for continuous search spaces.

Reference:
    K. M. Passino, "Biomimicry of bacterial foraging for distributed
    optimization and control," IEEE Control Systems Magazine, 22(3),
    52-67, 2002.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class BacterialForagingOptimization(Algorithm):
    """Bacterial Foraging Optimization (Passino, 2002).

    Modeled on how *E. coli* hunts for nutrients, using three nested
    mechanisms that operate on very different timescales:

    1. **Chemotaxis** (every iteration) — a bacterium *tumbles* into a
       random direction, then *swims*: it keeps stepping that same way
       for as long as conditions improve, up to ``n_swim`` steps. This
       is a directional line search, and it is what separates BFO from
       algorithms that take one isolated random step per iteration.
    2. **Reproduction** (every ``reproduction_interval`` iterations) —
       bacteria are ranked by **health**, the fitness accumulated over
       their whole lifetime rather than their current position. The
       healthier half splits in two and the weaker half dies, so a
       bacterium that has been consistently good is rewarded even if it
       currently sits somewhere mediocre.
    3. **Elimination-dispersal** (every ``elimination_interval``
       iterations) — each bacterium is wiped out and re-placed at random
       with probability ``elimination_prob``, modelling a sudden change
       in the environment. This is the algorithm's escape from local
       optima.

    Args:
        population_size: Number of bacteria.
        step_size: Initial chemotaxis step, as a fraction of each
            dimension's bound range. Decays quadratically over the run,
            so late chemotaxis becomes a fine local search.
        n_swim: Maximum consecutive swim steps in one direction.
        reproduction_interval: Chemotaxis steps between reproductions.
        elimination_interval: Chemotaxis steps between
            elimination-dispersal events.
        elimination_prob: Probability that a given bacterium is
            dispersed at such an event.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=15, step_size=0.3, n_swim=4,
                 reproduction_interval=20, elimination_interval=100,
                 elimination_prob=0.25, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if n_swim < 1:
            raise ValueError("n_swim must be >= 1")
        if reproduction_interval < 1:
            raise ValueError("reproduction_interval must be >= 1")
        if elimination_interval < 1:
            raise ValueError("elimination_interval must be >= 1")
        if not 0.0 <= elimination_prob <= 1.0:
            raise ValueError("elimination_prob must be in [0, 1]")
        self.step_size = float(step_size)
        self.n_swim = int(n_swim)
        self.reproduction_interval = int(reproduction_interval)
        self.elimination_interval = int(elimination_interval)
        self.elimination_prob = float(elimination_prob)

    def init_population(self, task):
        bacteria = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in bacteria])
        health = fitness.copy()      # cumulative cost over a lifetime
        return bacteria, fitness, health

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _chemotaxis(self, task, bacterium, current, step):
        """Tumble into a random direction, then swim while it pays off."""
        direction = self.rng.normal(0.0, 1.0, task.dimension)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else direction

        for _ in range(self.n_swim):
            if task.stopping_condition():
                break
            candidate = task.repair(bacterium + step * direction)
            candidate_fitness = task.eval(candidate)
            if candidate_fitness >= current:
                break                 # conditions worsened: stop swimming
            bacterium, current = candidate, candidate_fitness
        return bacterium, current

    def _reproduce(self, bacteria, fitness, health):
        """The healthiest half splits; the weakest half dies."""
        order = np.argsort(health)           # lower cumulative cost = healthier
        survivors = order[: self.population_size // 2]
        bacteria = np.vstack([bacteria[survivors], bacteria[survivors]])
        fitness = np.concatenate([fitness[survivors], fitness[survivors]])
        # Trim or pad to keep the population size exact when it is odd.
        if len(bacteria) < self.population_size:
            extra = order[len(survivors)]
            bacteria = np.vstack([bacteria, bacteria[:1]])
            fitness = np.concatenate([fitness, [fitness[extra % len(fitness)]]])
        bacteria = bacteria[: self.population_size]
        fitness = fitness[: self.population_size]
        return bacteria, fitness, fitness.copy()   # health resets

    def run_iteration(self, task, state):
        bacteria, fitness, health = state
        step = self.step_size * max(1.0 - self._progress(task), 1e-4) ** 2
        step = step * (task.upper - task.lower)

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            bacteria[i], fitness[i] = self._chemotaxis(
                task, bacteria[i], fitness[i], step)
            health[i] += fitness[i]           # accumulate a lifetime's cost

        if task.iters > 0 and task.iters % self.reproduction_interval == 0:
            bacteria, fitness, health = self._reproduce(bacteria, fitness, health)

        if task.iters > 0 and task.iters % self.elimination_interval == 0:
            for i in range(self.population_size):
                if task.stopping_condition():
                    break
                if self.rng.random() < self.elimination_prob:
                    bacteria[i] = self.rng.uniform(task.lower, task.upper)
                    fitness[i] = task.eval(bacteria[i])
                    health[i] = fitness[i]

        return bacteria, fitness, health
