"""ikn_library: research and data-science tools.

The first module provides nature-inspired metaheuristic algorithms for
continuous optimization, feature selection, and parameter optimization,
following a NiaPy-like workflow:

    >>> from ikn_library import Task
    >>> from ikn_library.problems import Sphere
    >>> from ikn_library.algorithms import AntColonyOptimization
    >>> task = Task(problem=Sphere(dimension=10), max_evals=10000)
    >>> algo = AntColonyOptimization(population_size=30, seed=42)
    >>> best_x, best_fitness = algo.run(task)
"""

from ikn_library.task import OptimizationType, Task

__version__ = "0.10.0"

__all__ = ["OptimizationType", "Task", "__version__"]
