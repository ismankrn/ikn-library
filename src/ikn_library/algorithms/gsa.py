"""Gravitational Search Algorithm for continuous search spaces.

Reference:
    E. Rashedi, H. Nezamabadi-pour, and S. Saryazdi, "GSA: a
    gravitational search algorithm," Information Sciences, 179(13),
    2232-2248, 2009.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm

EPSILON = 1e-12


class GravitationalSearchAlgorithm(Algorithm):
    """Gravitational Search Algorithm (Rashedi et al., 2009).

    Candidate solutions are masses in a search space governed by
    Newton's law of gravitation. Each agent's **mass grows with its
    fitness**, and that single fact produces the search dynamics:

    - Heavy agents (good solutions) exert strong attraction, pulling
      the swarm toward promising regions.
    - Heavy agents also have large **inertia**, so they move slowly and
      hold their ground — the best solutions are naturally conservative
      while poor, light agents move far and explore.

    Two schedules tighten the search over time: the gravitational
    constant \\(G\\) decays exponentially, and **Kbest** — the number of
    agents allowed to exert force — shrinks from the whole population
    down to one, so the swarm gradually listens only to its elite.

    Args:
        population_size: Number of agents.
        g0: Initial gravitational constant.
        alpha: Decay rate of the gravitational constant.
        final_kbest: Number of attracting agents at the end of the run;
            Kbest shrinks linearly from ``population_size`` to this.
        max_velocity: Velocity limit as a fraction of each dimension's
            bound range.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, g0=100.0, alpha=30.0,
                 final_kbest=1, max_velocity=0.5, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if g0 <= 0:
            raise ValueError("g0 must be > 0")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not 1 <= final_kbest <= population_size:
            raise ValueError("final_kbest must be in [1, population_size]")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be > 0")
        self.g0 = float(g0)
        self.alpha = float(alpha)
        self.final_kbest = int(final_kbest)
        self.max_velocity = float(max_velocity)

    def init_population(self, task):
        agents = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in agents])
        velocities = np.zeros_like(agents)
        return agents, fitness, velocities

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _gravitational_constant(self, progress):
        """G decays exponentially over the run (Eq. 1)."""
        return self.g0 * np.exp(-self.alpha * progress)

    def _masses(self, fitness):
        """Normalized masses: best agent gets 1, worst gets 0 (Eq. 2)."""
        best, worst = fitness.min(), fitness.max()
        if worst - best < EPSILON:
            return np.full(self.population_size, 1.0 / self.population_size)
        raw = (worst - fitness) / (worst - best)
        return raw / (raw.sum() + EPSILON)

    def _kbest(self, progress):
        """Number of attracting agents, shrinking to ``final_kbest``."""
        span = self.population_size - self.final_kbest
        return max(self.final_kbest,
                   int(round(self.population_size - progress * span)))

    def run_iteration(self, task, state):
        agents, fitness, velocities = state
        progress = self._progress(task)
        g = self._gravitational_constant(progress)
        masses = self._masses(fitness)
        kbest = self._kbest(progress)
        span = task.upper - task.lower

        # Only the kbest heaviest agents exert force.
        attractors = np.argsort(fitness)[:kbest]

        # Acceleration: the agent's own mass cancels out of F = ma, so
        # only the attractors' masses appear (Eq. 3-5). Distances are
        # scaled by the search range to keep g0 problem-independent.
        acceleration = np.zeros_like(agents)
        for i in range(self.population_size):
            offsets = agents[attractors] - agents[i]
            distances = np.linalg.norm(offsets / span, axis=1)
            pull = (masses[attractors] / (distances + EPSILON))[:, None]
            randomness = self.rng.random((len(attractors), 1))
            contribution = randomness * pull * offsets
            # An agent exerts no force on itself.
            self_index = np.flatnonzero(attractors == i)
            if len(self_index) > 0:
                contribution[self_index[0]] = 0.0
            acceleration[i] = g * contribution.sum(axis=0)

        # Velocity keeps a random fraction of its previous value (Eq. 6).
        limit = self.max_velocity * span
        velocities = np.clip(
            self.rng.random((self.population_size, 1)) * velocities + acceleration,
            -limit, limit)
        agents = np.array([task.repair(x) for x in agents + velocities])
        fitness = np.array([task.eval(x) if not task.stopping_condition()
                            else f for x, f in zip(agents, fitness)])
        return agents, fitness, velocities
