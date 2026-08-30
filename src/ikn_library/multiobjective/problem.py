"""Base class for problems with several conflicting objectives."""

import numpy as np

from ikn_library.problems.problem import Problem


class MultiObjectiveProblem(Problem):
    """A problem whose evaluation returns several objectives at once.

    Subclass this and implement :meth:`_evaluate` to return a vector of
    ``n_objectives`` values, **all of them minimized**. A criterion that
    should be maximized (accuracy, F1, ...) is expressed as ``1 - value``
    or ``-value``.

    Args:
        dimension: Number of decision variables.
        n_objectives: How many objectives ``_evaluate`` returns.
        lower: Lower bound(s) of the search space.
        upper: Upper bound(s) of the search space.
        objective_names: Optional labels, used when reporting or
            plotting a Pareto front.

    Example:
        >>> class AccuracyAndSize(MultiObjectiveProblem):
        ...     def __init__(self, n_features):
        ...         super().__init__(n_features, n_objectives=2,
        ...                          lower=0.0, upper=1.0,
        ...                          objective_names=["error", "n_features"])
        ...     def _evaluate(self, x):
        ...         mask = x > 0.5
        ...         return np.array([1 - score(mask), mask.mean()])
    """

    def __init__(self, dimension, n_objectives, lower=-1.0, upper=1.0,
                 objective_names=None):
        super().__init__(dimension, lower, upper)
        if n_objectives < 2:
            raise ValueError("n_objectives must be >= 2; use Problem for one")
        self.n_objectives = int(n_objectives)
        if objective_names is None:
            objective_names = [f"objective_{i}" for i in range(self.n_objectives)]
        elif len(objective_names) != self.n_objectives:
            raise ValueError("objective_names must have n_objectives entries")
        self.objective_names = list(objective_names)

    def evaluate(self, x):
        """Evaluate ``x`` and return its objective vector."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"expected solution of shape ({self.dimension},), "
                             f"got {x.shape}")
        values = np.asarray(self._evaluate(x), dtype=float).ravel()
        if values.shape != (self.n_objectives,):
            raise ValueError(f"_evaluate must return {self.n_objectives} "
                             f"objectives, got {values.shape[0]}")
        return values
