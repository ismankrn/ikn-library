# Getting Started

## Defining a task

A `Task` combines a problem with a stopping condition:

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
```

You can bound the run by evaluations (`max_evals`), iterations
(`max_iters`), or both. After a run, the task holds useful information:

```python
task.evals               # evaluations used
task.iters               # iterations performed
task.result()            # (best_x, best_fitness)
task.convergence_data()  # (iterations, best_fitness_per_iteration) for plotting
```

## Maximization

By default a task minimizes. When higher is better (e.g. model accuracy),
pass the optimization type:

```python
from ikn_library import OptimizationType, Task

task = Task(
    problem=my_problem,
    max_evals=5000,
    optimization_type=OptimizationType.MAXIMIZATION,
)
```

## Custom problems

Subclass `Problem` and implement `_evaluate`. This is how you plug in any
objective — including parameter optimization of a machine-learning model:

```python
import numpy as np
from ikn_library.problems import Problem

class MyProblem(Problem):
    def __init__(self, dimension=10):
        super().__init__(dimension, lower=-10.0, upper=10.0)

    def _evaluate(self, x):
        return float(np.sum(np.abs(x)))
```

## Running an algorithm

All algorithms share the same interface:

```python
from ikn_library.algorithms import AntColonyOptimization

algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
best_x, best_fitness = algo.run(task)
```

Pass a `seed` for reproducible runs.

## Available algorithms

See the **[Algorithms](algorithms.md)** page for the full list with
descriptions of how each algorithm works, its key parameters, and
literature references.
