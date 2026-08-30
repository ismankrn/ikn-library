"""Cat Swarm Optimization for continuous search spaces.

Reference:
    S.-C. Chu, P.-W. Tsai, and J.-S. Pan, "Cat swarm optimization," in
    PRICAI 2006: Trends in Artificial Intelligence, Lecture Notes in
    Computer Science 4099, Springer, 854-858, 2006;
    S.-C. Chu and P.-W. Tsai, "Computational intelligence based on the
    behavior of cats," International Journal of Innovative Computing,
    Information and Control, 3(1), 163-173, 2007.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class CatSwarmOptimization(Algorithm):
    """Cat Swarm Optimization (Chu, Tsai & Pan, 2006).

    Cats spend most of their time resting but alert, watching their
    surroundings, and occasionally burst into pursuit of prey. The
    algorithm mirrors that with **two modes**, and each iteration every
    cat is assigned one at random:

    - **Seeking mode** (the resting majority): the cat makes ``smp``
      copies of itself, perturbs a few dimensions of each by up to
      ``srd``, and moves to the best copy. This is careful local
      exploitation.
    - **Tracing mode** (a small fraction ``mixture_ratio``): the cat
      accelerates toward the best solution found so far with a velocity
      update, covering ground quickly. This is exploration.

    The **mixture ratio** is deliberately small — as in nature, most
    cats are watching rather than chasing.

    Args:
        population_size: Number of cats.
        mixture_ratio: Fraction of cats in tracing mode, in (0, 1).
        smp: Seeking memory pool — copies a seeking cat considers.
        srd: Initial seeking range of the selected dimension, as a
            fraction of each dimension's bound range. It decays
            quadratically with the evaluation budget so late seeking
            becomes fine-grained.
        cdc: Counts of dimension to change, as a fraction of the
            dimensions, in (0, 1].
        spc: Self-position considering — when ``True`` the cat's current
            position counts as one of the ``smp`` candidates.
        velocity_factor: Acceleration constant ``c`` in tracing mode.
        max_velocity: Velocity limit as a fraction of the bound range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, mixture_ratio=0.2, smp=5, srd=0.2,
                 cdc=0.1, spc=True, velocity_factor=2.0, max_velocity=0.2,
                 seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 < mixture_ratio < 1.0:
            raise ValueError("mixture_ratio must be in (0, 1)")
        if smp < 1:
            raise ValueError("smp must be >= 1")
        if srd <= 0:
            raise ValueError("srd must be > 0")
        if not 0.0 < cdc <= 1.0:
            raise ValueError("cdc must be in (0, 1]")
        if velocity_factor <= 0:
            raise ValueError("velocity_factor must be > 0")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be > 0")
        self.mixture_ratio = float(mixture_ratio)
        self.smp = int(smp)
        self.srd = float(srd)
        self.cdc = float(cdc)
        self.spc = bool(spc)
        self.velocity_factor = float(velocity_factor)
        self.max_velocity = float(max_velocity)

    def init_population(self, task):
        cats = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in cats])
        limit = self.max_velocity * (task.upper - task.lower)
        velocities = self.rng.uniform(-limit, limit,
                                      (self.population_size, task.dimension))
        return cats, fitness, velocities

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return task.evals / task.max_evals
        if np.isfinite(task.max_iters):
            return task.iters / task.max_iters
        return 0.0

    def _seek(self, task, cat, cat_fitness):
        """Seeking mode: perturb copies, move to the best one."""
        span = task.upper - task.lower
        # The seeking range narrows as the budget is spent, so early
        # cats survey widely and late ones inspect their own whiskers.
        srd = self.srd * max(1.0 - self._progress(task), 1e-4) ** 2
        n_changed = max(1, round(self.cdc * task.dimension))
        n_copies = self.smp - 1 if self.spc else self.smp

        best_x, best_fitness = cat, cat_fitness
        for _ in range(n_copies):
            if task.stopping_condition():
                break
            candidate = cat.copy()
            dimensions = self.rng.choice(task.dimension, n_changed, replace=False)
            # Each chosen dimension moves up or down by up to srd.
            sign = self.rng.choice([-1.0, 1.0], n_changed)
            candidate[dimensions] += (sign * srd
                                      * self.rng.random(n_changed)
                                      * span[dimensions])
            candidate = task.repair(candidate)
            candidate_fitness = task.eval(candidate)
            if candidate_fitness < best_fitness:
                best_x, best_fitness = candidate, candidate_fitness
        return best_x, best_fitness

    def _trace(self, task, cat, velocity):
        """Tracing mode: accelerate toward the best solution."""
        limit = self.max_velocity * (task.upper - task.lower)
        velocity = velocity + (self.velocity_factor
                               * self.rng.random(task.dimension)
                               * (task.best_x - cat))
        velocity = np.clip(velocity, -limit, limit)
        return task.repair(cat + velocity), velocity

    def run_iteration(self, task, state):
        cats, fitness, velocities = state
        tracing = self.rng.random(self.population_size) < self.mixture_ratio

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            if tracing[i]:
                candidate, velocities[i] = self._trace(task, cats[i], velocities[i])
                candidate_fitness = task.eval(candidate)
                if candidate_fitness < fitness[i]:
                    cats[i], fitness[i] = candidate, candidate_fitness
            else:
                cats[i], fitness[i] = self._seek(task, cats[i], fitness[i])

        return cats, fitness, velocities
