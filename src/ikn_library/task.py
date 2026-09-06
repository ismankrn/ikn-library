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
        patience: Stop early after this many consecutive iterations
            without an improvement to the best solution (optional).
            ``None`` (default) disables early stopping. A budget is
            still required: patience shortens a run, it does not bound
            one, since a search that keeps improving by a hair would
            otherwise never stop.
        min_delta: How much the best fitness must improve for an
            iteration to count as an improvement (default 0.0, meaning
            any improvement counts). Expressed in the problem's own
            units, always as a positive magnitude regardless of whether
            the task minimizes or maximizes.

    At least one of ``max_evals`` / ``max_iters`` must be given.

    Attributes:
        stalled_iters: Consecutive iterations without an improvement.
        stopped_early: ``True`` when patience ended the run.
    """

    def __init__(self, problem, max_evals=None, max_iters=None,
                 optimization_type=OptimizationType.MINIMIZATION,
                 patience=None, min_delta=0.0):
        if max_evals is None and max_iters is None:
            raise ValueError("provide max_evals and/or max_iters")
        if patience is not None and int(patience) < 1:
            raise ValueError("patience must be >= 1 (or None to disable it)")
        if float(min_delta) < 0.0:
            raise ValueError("min_delta must be >= 0")
        self.problem = problem
        self.max_evals = np.inf if max_evals is None else int(max_evals)
        self.max_iters = np.inf if max_iters is None else int(max_iters)
        self.optimization_type = optimization_type
        self.patience = None if patience is None else int(patience)
        self.min_delta = float(min_delta)

        self.evals = 0
        self.iters = 0
        self.best_x = None
        self.best_fitness = np.inf
        self.convergence = []  # best internal fitness after each iteration
        self.stalled_iters = 0
        self._last_improvement = np.inf  # best internal fitness when last improved

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
        """Advance the iteration counter and record convergence.

        Also updates the patience counter: the iteration counts as an
        improvement when the best fitness dropped by more than
        ``min_delta`` since the last one that did.
        """
        self.iters += 1
        self.convergence.append(self.best_fitness)
        if self.best_fitness < self._last_improvement - self.min_delta:
            self._last_improvement = self.best_fitness
            self.stalled_iters = 0
        else:
            self.stalled_iters += 1

    @property
    def budget_exhausted(self):
        """True when the evaluation or iteration budget has run out."""
        return self.evals >= self.max_evals or self.iters >= self.max_iters

    @property
    def stopped_early(self):
        """True when ``patience`` ended the run before its budget did."""
        return (self.patience is not None
                and self.stalled_iters >= self.patience
                and not self.budget_exhausted)

    def stopping_condition(self):
        """True when the budget is exhausted or patience has run out."""
        return self.budget_exhausted or (
            self.patience is not None and self.stalled_iters >= self.patience
        )

    def result(self):
        """Return ``(best_x, best_fitness)`` in the problem's original sense."""
        return self.best_x, self.best_fitness * self.optimization_type.value

    def convergence_data(self):
        """Convergence history as ``(iterations, best_fitness_values)``."""
        values = np.array(self.convergence) * self.optimization_type.value
        return np.arange(1, len(values) + 1), values

    def stall_lengths(self):
        """How long this run went without improving, plateau by plateau.

        Replays the convergence history under this task's ``min_delta``,
        so it reports exactly what ``patience`` would have counted —
        whether or not patience was set. Use it to calibrate a patience
        value from a run you have already paid for:

        - ``interior`` holds one entry per plateau that *ended* in an
          improvement. A patience value must be **larger than the
          longest of these**, or it would have cut the run short.
        - ``tail`` is the run of iterations after the last improvement.
          It never ended, so it is the budget a patience value could
          have saved.

        Returns:
            tuple: ``(interior, tail)`` — an ``int`` array and an ``int``.
        """
        interior = []
        stalled = 0
        reference = np.inf
        for value in self.convergence:
            if value < reference - self.min_delta:
                reference = value
                if stalled:
                    interior.append(stalled)
                stalled = 0
            else:
                stalled += 1
        return np.array(interior, dtype=int), stalled
