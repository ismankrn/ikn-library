"""Harmony Search for continuous search spaces.

Reference:
    Z. W. Geem, J. H. Kim, and G. V. Loganathan, "A new heuristic
    optimization algorithm: harmony search," Simulation, 76(2), 60-68,
    2001.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class HarmonySearch(Algorithm):
    """Harmony Search (Geem et al., 2001).

    Modeled on musicians improvising together, each choosing a pitch
    either from what the group has played before or from scratch. The
    algorithm keeps a **harmony memory** of the best solutions found and
    improvises **one new solution per iteration**, which replaces the
    worst harmony if it beats it.

    What makes it structurally different from everything else here is
    how that new solution is built. Each decision variable is drawn
    **independently, from a different randomly chosen harmony**:

    - with probability ``hmcr`` the value comes from the memory, and is
      then nudged with probability ``par`` (*pitch adjustment*);
    - otherwise it is drawn uniformly from the whole range.

    So one improvisation can combine coordinate 1 from harmony 7,
    coordinate 2 from harmony 2, coordinate 3 from a fresh random draw,
    and so on. Every other recombination in this library mixes exactly
    two parents; Harmony Search mixes across the entire memory at once.

    Note that the algorithm has been shown to be a special case of
    evolution strategies rather than a genuinely new method — see the
    algorithm page, which documents the analysis and what it means for
    using it.

    Args:
        population_size: Harmony memory size (HMS).
        hmcr: Harmony memory considering rate — chance a variable is
            taken from memory rather than drawn at random.
        par: Pitch adjusting rate — chance a memory-derived variable is
            then nudged.
        bandwidth: Pitch adjustment size, as a fraction of the bound
            range. Decays quadratically over the run, which the original
            does not do; see the algorithm page.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, hmcr=0.95, par=0.6,
                 bandwidth=0.1, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 <= hmcr <= 1.0:
            raise ValueError("hmcr must be in [0, 1]")
        if not 0.0 <= par <= 1.0:
            raise ValueError("par must be in [0, 1]")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        self.hmcr = float(hmcr)
        self.par = float(par)
        self.bandwidth = float(bandwidth)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _improvise(self, task, memory, width):
        """Build one harmony, each variable from its own source."""
        dim = task.dimension
        from_memory = self.rng.random(dim) < self.hmcr

        # Each variable is read from a *different* harmony in memory.
        rows = self.rng.integers(0, self.population_size, dim)
        remembered = memory[rows, np.arange(dim)]
        fresh = self.rng.uniform(task.lower, task.upper, dim)
        harmony = np.where(from_memory, remembered, fresh)

        # Pitch adjustment nudges only the remembered variables.
        adjust = from_memory & (self.rng.random(dim) < self.par)
        nudge = self.rng.uniform(-1.0, 1.0, dim) * width
        return np.where(adjust, harmony + nudge, harmony)

    def run_iteration(self, task, state):
        memory, fitness = state
        width = (self.bandwidth * (task.upper - task.lower)
                 * max(1.0 - self._progress(task), 1e-6) ** 2)

        harmony = task.repair(self._improvise(task, memory, width))
        harmony_fitness = task.eval(harmony)

        # The new harmony displaces the worst one, if it is better.
        worst = np.argmax(fitness)
        if harmony_fitness < fitness[worst]:
            memory[worst] = harmony
            fitness[worst] = harmony_fitness

        return memory, fitness
