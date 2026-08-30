"""Task for multi-objective runs: keeps a Pareto archive, not one best."""

import numpy as np

from ikn_library.multiobjective.pareto import crowding_distance, pareto_front


class MultiObjectiveTask:
    """A multi-objective optimization run.

    Mirrors :class:`~ikn_library.task.Task` — same budget handling,
    ``eval`` / ``repair`` / ``next_iter`` / ``stopping_condition`` — but
    instead of tracking a single best solution it maintains an **archive
    of every non-dominated solution seen so far**. All objectives are
    minimized.

    Args:
        problem: A :class:`MultiObjectiveProblem`.
        max_evals: Stop after this many evaluations (optional).
        max_iters: Stop after this many iterations (optional).

    At least one of ``max_evals`` / ``max_iters`` must be given.
    """

    def __init__(self, problem, max_evals=None, max_iters=None,
                 archive_size=200):
        if max_evals is None and max_iters is None:
            raise ValueError("provide max_evals and/or max_iters")
        if archive_size < 2:
            raise ValueError("archive_size must be >= 2")
        self.problem = problem
        self.max_evals = np.inf if max_evals is None else int(max_evals)
        self.max_iters = np.inf if max_iters is None else int(max_iters)
        self.archive_size = int(archive_size)

        self.evals = 0
        self.iters = 0
        self._solutions = []
        self._objectives = []
        self.front_sizes = []      # archive size after each iteration

    @property
    def dimension(self):
        return self.problem.dimension

    @property
    def lower(self):
        return self.problem.lower

    @property
    def upper(self):
        return self.problem.upper

    @property
    def n_objectives(self):
        return self.problem.n_objectives

    def repair(self, x):
        """Clip a solution back into the search-space bounds."""
        return np.clip(x, self.lower, self.upper)

    def eval(self, x):
        """Evaluate ``x``, count it, and update the Pareto archive."""
        if self.stopping_condition():
            return np.full(self.n_objectives, np.inf)
        objectives = self.problem.evaluate(x)
        self.evals += 1
        self._solutions.append(np.array(x, dtype=float))
        self._objectives.append(objectives)
        # Prune only when the archive has grown well past its target, so
        # pruning cannot be triggered on every single evaluation.
        if len(self._solutions) >= 2 * self.archive_size:
            self._prune()
        return objectives

    def _prune(self):
        """Reduce the archive to its Pareto front, capped by crowding."""
        solutions, objectives = pareto_front(
            np.array(self._solutions), np.array(self._objectives))
        if len(solutions) > self.archive_size:
            # Keep the most spread-out solutions so the front stays
            # representative instead of clustering in one region.
            keep = np.argsort(-crowding_distance(objectives))[:self.archive_size]
            keep = np.sort(keep)
            solutions, objectives = solutions[keep], objectives[keep]
        self._solutions = list(solutions)
        self._objectives = list(objectives)

    def next_iter(self):
        """Advance the iteration counter and record the archive size."""
        self.iters += 1
        self.front_sizes.append(len(self._solutions))

    def stopping_condition(self):
        """True when the evaluation or iteration budget is exhausted."""
        return self.evals >= self.max_evals or self.iters >= self.max_iters

    def result(self):
        """The Pareto front found: ``(solutions, objectives)``.

        Solutions are sorted by the first objective, so the pair reads
        as a trade-off curve.
        """
        if not self._solutions:
            return np.empty((0, self.dimension)), np.empty((0, self.n_objectives))
        return pareto_front(np.array(self._solutions), np.array(self._objectives))
