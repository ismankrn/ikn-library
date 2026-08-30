"""Pareto dominance utilities shared by the multi-objective machinery.

All objectives are **minimized**, so a problem that maximizes accuracy
should return ``1 - accuracy`` (or the negated value).
"""

import numpy as np


def dominates(a, b):
    """True when objective vector ``a`` Pareto-dominates ``b``.

    ``a`` dominates ``b`` if it is no worse in every objective and
    strictly better in at least one.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives):
    """Sort solutions into Pareto fronts (the NSGA-II ranking step).

    Args:
        objectives: Array of shape ``(n_solutions, n_objectives)``.

    Returns:
        list[numpy.ndarray]: Front 0 holds the non-dominated solutions,
        front 1 those dominated only by front 0, and so on. Each entry
        is an array of row indices into ``objectives``.
    """
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    n_solutions = len(objectives)
    if n_solutions == 0:
        return []

    # Vectorized dominance matrix: dominance[i, j] is True when i
    # dominates j. Building it in numpy rather than in Python loops
    # keeps NSGA-II usable at realistic population sizes.
    no_worse = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    better = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    dominance = no_worse & better

    counts = dominance.sum(axis=0)          # how many dominate each solution
    fronts = []
    assigned = np.zeros(n_solutions, dtype=bool)
    current = np.flatnonzero(counts == 0)
    while len(current) > 0:
        fronts.append(current)
        assigned[current] = True
        # Remove the current front's dominance, then take the new zeros.
        counts = counts - dominance[current].sum(axis=0)
        current = np.flatnonzero((counts == 0) & ~assigned)
    return fronts


def crowding_distance(objectives):
    """Crowding distance of each solution within one front.

    Measures how isolated a solution is in objective space; NSGA-II uses
    it as the tie-breaker inside a front so the Pareto set stays spread
    out instead of clustering. Boundary solutions get infinite distance.

    Args:
        objectives: Array of shape ``(n_solutions, n_objectives)``.
    """
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    n_solutions, n_objectives = objectives.shape
    distance = np.zeros(n_solutions)
    if n_solutions <= 2:
        return np.full(n_solutions, np.inf)

    for m in range(n_objectives):
        order = np.argsort(objectives[:, m])
        values = objectives[order, m]
        distance[order[0]] = distance[order[-1]] = np.inf
        spread = values[-1] - values[0]
        if spread <= 0:
            continue
        distance[order[1:-1]] += (values[2:] - values[:-2]) / spread
    return distance


def pareto_front(solutions, objectives, unique=True):
    """Keep only the non-dominated solutions.

    Args:
        solutions: Array of shape ``(n_solutions, dimension)``.
        objectives: Array of shape ``(n_solutions, n_objectives)``.
        unique: When ``True`` (default), solutions that land on exactly
            the same objective values are collapsed to one. Discrete
            objectives — a feature count, a number of ensemble members —
            produce many such duplicates, and keeping them would bloat
            the front and distort crowding distances.

    Returns:
        tuple: ``(solutions, objectives)`` of the Pareto front, sorted
        by the first objective.
    """
    solutions = np.asarray(solutions, dtype=float)
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    if len(solutions) == 0:
        return solutions, objectives
    keep = non_dominated_sort(objectives)[0]
    if unique:
        _, first = np.unique(objectives[keep], axis=0, return_index=True)
        keep = keep[np.sort(first)]
    order = keep[np.argsort(objectives[keep, 0])]
    return solutions[order], objectives[order]
