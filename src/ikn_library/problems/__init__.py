"""Optimization problems: base class, benchmarks, and feature selection."""

from ikn_library.problems.benchmarks import Ackley, Rastrigin, Sphere
from ikn_library.problems.feature_selection import FeatureSelectionProblem
from ikn_library.problems.problem import Problem

__all__ = ["Ackley", "FeatureSelectionProblem", "Problem", "Rastrigin", "Sphere"]
