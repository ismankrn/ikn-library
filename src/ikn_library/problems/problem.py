"""Base class for optimization problems."""

import numpy as np


class Problem:
    """An optimization problem defined on a box-constrained search space.

    Subclass this and implement :meth:`_evaluate` to define a custom
    problem (e.g. a machine-learning objective for parameter optimization
    or a wrapper objective for feature selection).

    Args:
        dimension: Number of decision variables.
        lower: Lower bound(s) of the search space. Scalar or array of
            shape ``(dimension,)``.
        upper: Upper bound(s) of the search space. Scalar or array of
            shape ``(dimension,)``.

    Example:
        >>> class MyProblem(Problem):
        ...     def __init__(self, dimension):
        ...         super().__init__(dimension, lower=-10.0, upper=10.0)
        ...     def _evaluate(self, x):
        ...         return float(np.sum(x ** 2))
    """

    def __init__(self, dimension, lower=-1.0, upper=1.0):
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = int(dimension)
        self.lower = np.full(self.dimension, lower, dtype=float) if np.isscalar(lower) else np.asarray(lower, dtype=float)
        self.upper = np.full(self.dimension, upper, dtype=float) if np.isscalar(upper) else np.asarray(upper, dtype=float)
        if self.lower.shape != (self.dimension,) or self.upper.shape != (self.dimension,):
            raise ValueError("lower/upper must be scalars or arrays of shape (dimension,)")
        if np.any(self.lower >= self.upper):
            raise ValueError("each lower bound must be strictly less than its upper bound")

    def evaluate(self, x):
        """Evaluate a solution vector and return its fitness as ``float``."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"expected solution of shape ({self.dimension},), got {x.shape}")
        return float(self._evaluate(x))

    def _evaluate(self, x):
        """Compute the objective value for solution ``x``. Must be overridden."""
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__
