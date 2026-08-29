"""Task: wraps a Problem with a stopping condition and bookkeeping."""

from enum import Enum

import numpy as np


class OptimizationType(Enum):
    MINIMIZATION = 1
    MAXIMIZATION = -1


class Task:
    """An optimization run: a problem plus a budget and progress tracking.

    The task counts evaluations, tracks the best solution found so far,
    and records the convergence history. Algorithms should call
    :meth:`eval` for every candidate solution and check :meth:`stopping_condition`.

    Args:
        problem: The :class:`~ikn_library.problems.Problem` to optimize.
        max_evals: Stop after this many fitness evaluations (optional).
        max_iters: Stop after this many iterations (optional). The
            algorithm must call :meth:`next_iter` once per iteration.
        optimization_type: Minimize (default) or maximize.

    At least one of ``max_evals`` / ``max_iters`` must be given.
    """

    def __init__(self, problem, max_evals=None, max_iters=None,
                 optimization_type=OptimizationType.MINIMIZATION):
        if max_evals is None and max_iters is None:
            raise ValueError("provide max_evals and/or max_iters")
        self.problem = problem
        self.max_evals = np.inf if max_evals is None else int(max_evals)
        self.max_iters = np.inf if max_iters is None else int(max_iters)
        self.optimization_type = optimization_type

        self.evals = 0
        self.iters = 0
        self.best_x = None
        self.best_fitness = np.inf
        self.convergence = []  # best internal fitness after each iteration

    @property
    def dimension(self):
        return self.problem.dimension

    @property
    def lower(self):
        return self.problem.lower

    @property
    def upper(self):
        return self.problem.upper

    def repair(self, x):
        """Clip a solution back into the search-space bounds."""
        return np.clip(x, self.lower, self.upper)

    def eval(self, x):
        """Evaluate ``x``, update counters and the best-so-far solution.

        Returns the fitness in *internal* form (maximization problems are
        negated so that algorithms can always minimize).
        """
        if self.stopping_condition():
            return np.inf
        fitness = self.problem.evaluate(x) * self.optimization_type.value
        self.evals += 1
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_x = np.array(x, dtype=float)
        return fitness

    def next_iter(self):
        """Advance the iteration counter and record convergence."""
        self.iters += 1
        self.convergence.append(self.best_fitness)

    def stopping_condition(self):
        """True when the evaluation or iteration budget is exhausted."""
        return self.evals >= self.max_evals or self.iters >= self.max_iters

    def result(self):
        """Return ``(best_x, best_fitness)`` in the problem's original sense."""
        return self.best_x, self.best_fitness * self.optimization_type.value

    def convergence_data(self):
        """Convergence history as ``(iterations, best_fitness_values)``."""
        values = np.array(self.convergence) * self.optimization_type.value
        return np.arange(1, len(values) + 1), values
