"""Firefly Algorithm for continuous search spaces.

Reference:
    X.-S. Yang, "Firefly algorithms for multimodal optimization," in
    Stochastic Algorithms: Foundations and Applications (SAGA 2009),
    Lecture Notes in Computer Science 5792, Springer, 169-178, 2009;
    X.-S. Yang, Nature-Inspired Metaheuristic Algorithms, Luniver
    Press, 2008.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class FireflyAlgorithm(Algorithm):
    """Firefly Algorithm (Yang, 2008), based on bioluminescent attraction.

    All fireflies are unisex, so any one is attracted to any brighter
    one — brightness being the objective value. Attraction weakens with
    distance, because light is absorbed by the medium:

    \\[
    \\beta(r) = \\beta_0 \\, e^{-\\gamma r^{2}}
    \\]

    That distance decay is the algorithm's defining feature. With a
    large absorption coefficient \\(\\gamma\\), fireflies only see close
    neighbours and the swarm splits into subgroups that explore
    different regions in parallel — which is why the method was
    proposed specifically for **multimodal** problems.

    Every firefly is drawn toward all the brighter ones each iteration.
    The textbook formulation applies those attractions one at a time in
    a nested loop; here they are combined into a single weighted pull
    (see :meth:`_attract`), which is equivalent in spirit and far
    faster. One iteration costs exactly ``population_size``
    evaluations.

    Args:
        population_size: Number of fireflies.
        alpha: Randomization strength, as a fraction of each dimension's
            bound range.
        alpha_decay: Geometric decay applied to ``alpha`` every
            iteration, in (0, 1]; 1.0 disables the decay.
        beta0: Attractiveness at zero distance.
        gamma: Light absorption coefficient, scaled internally by the
            search range so it behaves the same on any problem.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=25, alpha=1.0, alpha_decay=0.92,
                 beta0=1.0, gamma=1.0, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not 0.0 < alpha_decay <= 1.0:
            raise ValueError("alpha_decay must be in (0, 1]")
        if beta0 <= 0:
            raise ValueError("beta0 must be > 0")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.alpha = float(alpha)
        self.alpha_decay = float(alpha_decay)
        self.beta0 = float(beta0)
        self.gamma = float(gamma)

    def init_population(self, task):
        fireflies = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        brightness = np.array([task.eval(x) for x in fireflies])
        return fireflies, brightness, self.alpha

    def run_iteration(self, task, state):
        fireflies, brightness, alpha = state
        span = task.upper - task.lower
        # Scale the absorption to the search range so gamma means the
        # same thing whatever the problem's bounds are.
        gamma = self.gamma / np.mean(span) ** 2

        order = np.argsort(brightness)
        fireflies, brightness = fireflies[order], brightness[order]

        # The brightest firefly (index 0 after sorting) has nobody to be
        # attracted to; it only takes a random walk.
        for i in range(self.population_size):
            if task.stopping_condition():
                break
            moved = self._attract(fireflies, brightness, i, gamma)
            # A random walk on top of the attraction keeps the swarm
            # from collapsing onto the brightest firefly.
            moved = moved + alpha * span * (self.rng.random(task.dimension) - 0.5)
            candidate = task.repair(moved)
            candidate_fitness = task.eval(candidate)
            if candidate_fitness < brightness[i]:
                fireflies[i], brightness[i] = candidate, candidate_fitness

        return fireflies, brightness, alpha * self.alpha_decay

    def _attract(self, fireflies, brightness, i, gamma):
        """Pull firefly ``i`` toward every brighter one, vectorized.

        The population is kept sorted by brightness, so the brighter
        fireflies are exactly those before index ``i``. Their individual
        attractions are combined into a single weighted pull, which
        avoids the nested Python loop of the textbook formulation.
        """
        if i == 0:
            return fireflies[i].copy()
        brighter = fireflies[:i]
        distances_squared = np.sum((brighter - fireflies[i]) ** 2, axis=1)
        betas = self.beta0 * np.exp(-gamma * distances_squared)
        weight = betas.sum()
        if weight <= 0:
            return fireflies[i].copy()
        centroid = (betas[:, None] * brighter).sum(axis=0) / weight
        # Cap the pull at 1 so a firefly never overshoots past the
        # brighter ones it is chasing.
        return fireflies[i] + min(weight, 1.0) * (centroid - fireflies[i])
