"""Clonal Selection Algorithm (CLONALG) for continuous search spaces.

Reference:
    L. N. de Castro and F. J. Von Zuben, "Learning and optimization
    using the clonal selection principle," IEEE Transactions on
    Evolutionary Computation, 6(3), 239-251, 2002.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class ClonalSelectionAlgorithm(Algorithm):
    """Clonal Selection Algorithm (de Castro & Von Zuben, 2002).

    Modeled on the adaptive immune system: candidate solutions are
    **antibodies**, and their fitness is their *affinity* for the
    antigen (the problem). The immune response works by two coupled
    rules, which the algorithm implements directly:

    - **Cloning is proportional to affinity** — the better an antibody,
      the more copies of it are made, so the search budget concentrates
      on promising solutions.
    - **Hypermutation is inversely proportional to affinity** — good
      antibodies mutate gently (refining what already works), poor ones
      mutate wildly (searching elsewhere).

    Together these give a self-regulating balance similar in spirit to
    the Fireworks Algorithm, but reached through rank-based cloning and
    an *exponential* mutation law rather than explosion amplitudes. A
    handful of the worst antibodies are replaced by fresh random ones
    each generation, which maintains the receptor diversity a real
    immune system needs.

    Args:
        population_size: Number of antibodies ``N``.
        n_select: How many of the best antibodies are cloned each
            generation.
        clone_factor: Multiplier ``beta`` controlling how many clones
            the selected antibodies receive in total.
        n_replace: Worst antibodies replaced by random ones each
            generation (receptor editing).
        rho: Decay of the mutation rate with affinity; larger values
            make good antibodies mutate much less than poor ones.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=20, n_select=10, clone_factor=0.5,
                 n_replace=2, rho=3.0, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 1 <= n_select <= population_size:
            raise ValueError("n_select must be in [1, population_size]")
        if clone_factor <= 0:
            raise ValueError("clone_factor must be > 0")
        if not 0 <= n_replace < population_size:
            raise ValueError("n_replace must be in [0, population_size)")
        if rho <= 0:
            raise ValueError("rho must be > 0")
        self.n_select = int(n_select)
        self.clone_factor = float(clone_factor)
        self.n_replace = int(n_replace)
        self.rho = float(rho)

    def init_population(self, task):
        antibodies = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        affinity = np.array([task.eval(x) for x in antibodies])
        order = np.argsort(affinity)
        return antibodies[order], affinity[order]

    def _clone_counts(self):
        """Clones per selected antibody, by rank (Eq. 1).

        Rank ``i`` receives ``round(beta * N / i)`` clones, so the best
        antibody gets the most and the share falls off hyperbolically.
        """
        ranks = np.arange(1, self.n_select + 1)
        counts = np.round(self.clone_factor * self.population_size / ranks)
        return np.maximum(counts, 1).astype(int)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _mutation_rates(self, n, progress):
        """Mutation rate per antibody: low for good ones (Eq. 2).

        Antibodies arrive sorted, so normalized affinity is taken from
        the **rank** — 1 for the best, near 0 for the worst — and fed
        through ``exp(-rho * affinity)``. The whole profile is then
        scaled down quadratically as the budget is spent, which is a
        deviation from the original; see the docs for why it is needed.
        """
        normalized = (n - np.arange(n)) / n
        decay = max(1.0 - progress, 1e-6) ** 2
        return np.exp(-self.rho * normalized) * decay

    def run_iteration(self, task, state):
        antibodies, affinity = state
        span = task.upper - task.lower
        counts = self._clone_counts()
        rates = self._mutation_rates(len(antibodies), self._progress(task))

        candidates = list(antibodies)
        candidate_affinity = list(affinity)

        # Clone the best antibodies and hypermutate every clone.
        for i in range(self.n_select):
            for _ in range(counts[i]):
                if task.stopping_condition():
                    break
                clone = antibodies[i] + (rates[i] * span
                                         * self.rng.normal(0.0, 1.0, task.dimension))
                clone = task.repair(clone)
                candidates.append(clone)
                candidate_affinity.append(task.eval(clone))

        # Keep the best N antibodies among parents and clones.
        candidates = np.array(candidates)
        candidate_affinity = np.array(candidate_affinity)
        keep = np.argsort(candidate_affinity)[: self.population_size]
        antibodies, affinity = candidates[keep], candidate_affinity[keep]

        # Receptor editing: replace the worst antibodies at random.
        for i in range(self.population_size - self.n_replace,
                       self.population_size):
            if task.stopping_condition():
                break
            antibodies[i] = self.rng.uniform(task.lower, task.upper)
            affinity[i] = task.eval(antibodies[i])

        order = np.argsort(affinity)
        return antibodies[order], affinity[order]
