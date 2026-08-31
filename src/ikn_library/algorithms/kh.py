"""Krill Herd Algorithm for continuous search spaces.

Reference:
    A. H. Gandomi and A. H. Alavi, "Krill herd: a new bio-inspired
    optimization algorithm," Communications in Nonlinear Science and
    Numerical Simulation, 17(12), 4831-4845, 2012.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class KrillHerd(Algorithm):
    """Krill Herd Algorithm (Gandomi & Alavi, 2012).

    Models a krill swarm as a **Lagrangian system**: each krill has a
    velocity built from three physical terms, and position follows from
    integrating it over a time step.

    \\[
    \\frac{dX_i}{dt} = N_i + F_i + D_i
    \\]

    - **Induced motion** \\(N_i\\) — neighbours pull or push, weighted by
      how much better or worse they are. Each krill senses only within a
      radius derived from the herd's own mean spacing, so the
      neighbourhood adapts on its own.
    - **Foraging motion** \\(F_i\\) — attraction to the **food centre**, an
      inverse-fitness-weighted centroid of the whole herd, plus a pull
      toward the krill's own personal best.
    - **Physical diffusion** \\(D_i\\) — random drift that decays over the
      run, the only stochastic term.

    Two features set it apart here. The first two terms carry
    **inertia**: each keeps a fraction of its previous value, so motion
    is smoothed across iterations rather than recomputed from scratch —
    only the Gravitational Search Algorithm does anything
    comparable. The second is the food
    centre itself, a *weighted* centroid; no other algorithm in this
    library builds an attractor that every member contributes to in
    proportion to its quality.

    All couplings use differences between positions and **normalized**
    fitness gaps, which makes the whole update translation-invariant.

    Args:
        population_size: Number of krill.
        n_max: Maximum induced speed.
        v_f: Foraging speed.
        d_max: Maximum diffusion speed, decaying to zero over the run.
        inertia: Fraction of the previous induced and foraging motion
            retained each iteration.
        c_t: Time-step factor; the position step is ``c_t`` times the
            summed bound ranges, decaying quadratically over the run.
            The decay is not in the original — see the algorithm page.
        crossover_rate: Chance of a binomial crossover toward the best
            krill after moving. Set to 0 to disable.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=25, n_max=0.01, v_f=0.02,
                 d_max=0.005, inertia=0.9, c_t=0.5, crossover_rate=0.2,
                 seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if n_max <= 0:
            raise ValueError("n_max must be > 0")
        if v_f <= 0:
            raise ValueError("v_f must be > 0")
        if d_max <= 0:
            raise ValueError("d_max must be > 0")
        if not 0.0 <= inertia <= 1.0:
            raise ValueError("inertia must be in [0, 1]")
        if c_t <= 0:
            raise ValueError("c_t must be > 0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        self.n_max = float(n_max)
        self.v_f = float(v_f)
        self.d_max = float(d_max)
        self.inertia = float(inertia)
        self.c_t = float(c_t)
        self.crossover_rate = float(crossover_rate)

    def init_population(self, task):
        krill = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in krill])
        induced = np.zeros_like(krill)
        foraging = np.zeros_like(krill)
        return krill, fitness, induced, foraging, krill.copy(), fitness.copy()

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    @staticmethod
    def _normalize(fitness):
        """Map fitness onto [0, 1], 0 for the best krill.

        Every coupling uses these normalized gaps rather than raw
        fitness, which is what keeps the algorithm scale-invariant.
        """
        best, worst = fitness.min(), fitness.max()
        spread = worst - best
        if spread < 1e-30:
            return np.zeros_like(fitness)
        return (fitness - best) / spread

    def _food_centre(self, krill, normalized):
        """Inverse-fitness-weighted centroid of the whole herd.

        The published weight is ``1 / K_i``, which is undefined when a
        fitness reaches zero — as it does on every benchmark here. Using
        the normalized gap with an offset keeps the weights bounded and
        scale-free; see the algorithm page.
        """
        weights = 1.0 / (normalized + 0.1)
        return (weights[:, None] * krill).sum(axis=0) / weights.sum()

    def _induced_motion(self, krill, normalized, previous):
        """Neighbours attract or repel, by how much better they are."""
        offsets = krill[None, :, :] - krill[:, None, :]     # X_j - X_i
        distances = np.linalg.norm(offsets, axis=2)
        directions = offsets / (distances[:, :, None] + 1e-12)

        # Each krill senses within a radius set by the herd's spacing.
        sensing = distances.sum(axis=1) / (5.0 * self.population_size)
        neighbours = distances < sensing[:, None]
        np.fill_diagonal(neighbours, False)

        # K_i - K_j > 0 means j is better, so the pull is toward j.
        gaps = normalized[:, None] - normalized[None, :]
        local = ((gaps * neighbours)[:, :, None] * directions).sum(axis=1)

        return self.n_max * local + self.inertia * previous, gaps, directions

    def run_iteration(self, task, state):
        krill, fitness, induced, foraging, best_x, best_f = state
        progress = self._progress(task)
        normalized = self._normalize(fitness)
        best = int(np.argmin(fitness))

        induced, gaps, directions = self._induced_motion(
            krill, normalized, induced)
        # Attraction to the best krill strengthens as the run proceeds.
        c_best = 2.0 * (self.rng.random() + progress)
        induced = induced + self.n_max * (
            c_best * gaps[:, best, None] * directions[:, best, :])

        # Foraging: toward the food centre, and toward each personal best.
        centre = self._food_centre(krill, normalized)
        to_centre = centre - krill
        to_centre = to_centre / (
            np.linalg.norm(to_centre, axis=1, keepdims=True) + 1e-12)
        c_food = 2.0 * (1.0 - progress)
        to_best = best_x - krill
        to_best = to_best / (
            np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-12)
        beta = c_food * normalized[:, None] * to_centre + to_best
        foraging = self.v_f * beta + self.inertia * foraging

        # Diffusion: pure noise, fading out as the run converges.
        diffusion = self.d_max * (1.0 - progress) * self.rng.uniform(
            -1.0, 1.0, krill.shape)

        # The published step is fixed, so the herd never settles; tying
        # it to the budget is a deviation. See the algorithm page.
        step = (self.c_t * float(np.sum(task.upper - task.lower))
                * max(1.0 - progress, 1e-6) ** 2)
        candidates = krill + step * (induced + foraging + diffusion)

        for i in range(self.population_size):
            if task.stopping_condition():
                break
            candidate = candidates[i]
            if self.crossover_rate > 0 and i != best:
                swap = self.rng.random(task.dimension) < self.crossover_rate
                candidate = np.where(swap, krill[best], candidate)

            krill[i] = task.repair(candidate)
            fitness[i] = task.eval(krill[i])
            if fitness[i] < best_f[i]:              # remember personal bests
                best_x[i], best_f[i] = krill[i].copy(), fitness[i]

        return krill, fitness, induced, foraging, best_x, best_f
