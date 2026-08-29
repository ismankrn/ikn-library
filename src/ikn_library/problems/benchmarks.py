"""Standard benchmark functions for testing algorithms.

All benchmarks are minimization problems with a known global optimum of 0.
"""

import numpy as np

from ikn_library.problems.problem import Problem


class Sphere(Problem):
    """Sphere function: ``f(x) = sum(x_i^2)``. Optimum ``f(0) = 0``."""

    def __init__(self, dimension=10, lower=-5.12, upper=5.12):
        super().__init__(dimension, lower, upper)

    def _evaluate(self, x):
        return np.sum(x ** 2)


class Rastrigin(Problem):
    """Rastrigin function, highly multimodal. Optimum ``f(0) = 0``."""

    def __init__(self, dimension=10, lower=-5.12, upper=5.12):
        super().__init__(dimension, lower, upper)

    def _evaluate(self, x):
        return 10.0 * self.dimension + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))


class Ackley(Problem):
    """Ackley function, multimodal with a nearly flat outer region. Optimum ``f(0) = 0``."""

    def __init__(self, dimension=10, lower=-32.768, upper=32.768):
        super().__init__(dimension, lower, upper)

    def _evaluate(self, x):
        n = self.dimension
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
        return term1 + term2 + 20.0 + np.e
