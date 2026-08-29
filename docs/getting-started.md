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

| Algorithm | Class | Domain |
|---|---|---|
| Ant Colony Optimization (ACO-R) | `AntColonyOptimization` | continuous |
| Binary Ant Colony Optimization | `BinaryAntColonyOptimization` | binary / subsets |

**ACO-R** (Socha & Dorigo, 2008) keeps an archive of the best solutions;
each ant samples a new solution from a Gaussian centered on an archive
member, with the Gaussian width shrinking as the archive converges.

**Binary ACO** maintains a pheromone value per (feature, bit) pair and
builds bit strings by sampling each bit proportionally to its pheromone,
with a hyper-cube pheromone update toward the best solution found.
