"""Multi-objective optimization: Pareto fronts instead of one best solution."""

from ikn_library.multiobjective.feature_selection import (
    MultiObjectiveFeatureSelection,
)
from ikn_library.multiobjective.pareto import (
    crowding_distance,
    dominates,
    non_dominated_sort,
    pareto_front,
)
from ikn_library.multiobjective.problem import MultiObjectiveProblem
from ikn_library.multiobjective.task import MultiObjectiveTask

__all__ = [
    "MultiObjectiveFeatureSelection",
    "MultiObjectiveProblem",
    "MultiObjectiveTask",
    "crowding_distance",
    "dominates",
    "non_dominated_sort",
    "pareto_front",
]
