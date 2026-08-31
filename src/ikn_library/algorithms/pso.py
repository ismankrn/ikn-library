"""Particle Swarm Optimization for continuous search spaces.

References:
    J. Kennedy and R. Eberhart, "Particle swarm optimization,"
    Proceedings of ICNN'95, 1942-1948, 1995.

    Y. Shi and R. Eberhart, "A modified particle swarm optimizer,"
    IEEE World Congress on Computational Intelligence, 69-73, 1998.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class ParticleSwarmOptimization(Algorithm):
    """Particle Swarm Optimization (Kennedy & Eberhart, 1995).

    The oldest and best known swarm algorithm, and the one worth
    knowing first: a great many later metaheuristics turn out to be
    variations on it, and it is the baseline any new method should be
    measured against.

    Each particle carries a **velocity** and is pulled by two
    attractors — the best position it has personally visited
    (*cognitive*) and the best any particle has found (*social*):

    \\[
    v \\leftarrow w v
    + c_1 r_1 (p_{\\text{best}} - x)
    + c_2 r_2 (g_{\\text{best}} - x)
    \\]

    Velocity is what distinguishes PSO from the many algorithms that
    reposition individuals directly: a particle carries momentum, so it
    overshoots its attractors and oscillates around them rather than
    settling immediately. That oscillation is the search.

    The **inertia weight** ``w`` falls across the run, damping the
    momentum so the swarm converges. Shi and Eberhart added it in 1998
    for exactly that reason, which makes PSO one of the few algorithms
    in this library whose step schedule was tied to the run's progress
    from early on.

    Note that ``r1`` and ``r2`` are drawn **per coordinate**, which
    makes the search axis-aligned; see the algorithm page.

    Args:
        population_size: Number of particles.
        w_start: Initial inertia weight. Shi & Eberhart use 0.9;
            0.7 measures better here.
        w_end: Final inertia weight.
        c1: Cognitive coefficient, pulling toward the personal best.
        c2: Social coefficient, pulling toward the global best.
        max_velocity: Speed limit, as a fraction of each bound range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, w_start=0.7, w_end=0.4,
                 c1=2.0, c2=2.0, max_velocity=0.2, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if w_start < 0 or w_end < 0:
            raise ValueError("inertia weights must be >= 0")
        if w_end > w_start:
            raise ValueError("w_end must be <= w_start")
        if c1 < 0 or c2 < 0:
            raise ValueError("c1 and c2 must be >= 0")
        if c1 == 0 and c2 == 0:
            raise ValueError("at least one of c1, c2 must be > 0")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be > 0")
        self.w_start = float(w_start)
        self.w_end = float(w_end)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_velocity = float(max_velocity)

    def init_population(self, task):
        particles = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in particles])
        # Start at rest, so the first move is driven purely by attraction.
        velocities = np.zeros_like(particles)
        return particles, fitness, velocities, particles.copy(), fitness.copy()

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _inertia(self, task):
        """Momentum is damped linearly so the swarm settles."""
        return self.w_start - (self.w_start - self.w_end) * self._progress(task)

    def run_iteration(self, task, state):
        particles, fitness, velocities, best_x, best_f = state
        dimension = task.dimension
        span = task.upper - task.lower
        limit = self.max_velocity * span

        w = self._inertia(task)
        global_best = best_x[np.argmin(best_f)]

        # r1 and r2 are per-coordinate, which is what makes PSO
        # rotationally variant; see the algorithm page.
        r1 = self.rng.random((self.population_size, dimension))
        r2 = self.rng.random((self.population_size, dimension))

        velocities = (w * velocities
                      + self.c1 * r1 * (best_x - particles)
                      + self.c2 * r2 * (global_best - particles))
        velocities = np.clip(velocities, -limit, limit)
        candidates = particles + velocities

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            particles[i] = task.repair(candidates[i])
            fitness[i] = task.eval(particles[i])
            if fitness[i] < best_f[i]:            # remember personal bests
                best_x[i], best_f[i] = particles[i].copy(), fitness[i]

        return particles, fitness, velocities, best_x, best_f
