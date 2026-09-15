"""Random search: the baseline every other algorithm has to beat."""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class RandomSearch(Algorithm):
    """Uniform sampling of the search space, with no search at all.

    Every iteration draws ``population_size`` fresh points uniformly from
    the search box, evaluates them, and forgets them. There is no
    memory, no attraction, no step size and nothing to tune: the only
    thing that improves over a run is the best value the
    :class:`~ikn_library.Task` happens to have seen, and that improves
    for any objective whatsoever — even pure noise.

    That is exactly why it belongs in the library. A metaheuristic's
    score is only evidence of *searching* if it beats this at the same
    evaluation budget; the difference between the two is the part of the
    result the algorithm is responsible for. The teaching note on
    beating random search works through what happens when that
    comparison is skipped.

    Two properties are worth knowing before reading a comparison
    against it:

    - **It is unbiased by construction.** Uniform sampling has no
      preferred region, so a shifted or rotated optimum costs it
      nothing beyond how the transformation reshapes the distribution
      of objective values inside the box. Several algorithms in this
      library are not invariant in that way, and the gap between them
      and this baseline changes accordingly.
    - **On subset problems it samples the middle.** Solutions are drawn
      uniformly from ``[0, 1]``, so with the default threshold of 0.5
      each variable is selected with probability one half: its subsets
      cluster around half the features and it will essentially never
      propose a small one. Against a subset-size objective, that is a
      handicap rather than a neutral reference.

    Args:
        population_size: Points drawn per iteration. Only affects how
            evaluations are grouped into iterations; the sequence of
            points is the same stream either way.
        seed: Random seed for reproducibility (optional).

    Example:
        >>> task = Task(problem=Sphere(dimension=10), max_evals=20000)
        >>> best_x, best_fitness = RandomSearch(seed=42).run(task)
    """

    def run_iteration(self, task, state):
        """Draw and evaluate a fresh batch; keep nothing from the last one."""
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in population])
        return population, fitness
