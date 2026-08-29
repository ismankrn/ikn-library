# ikn-library

A growing Python library of research and data-science tools. Its first
module provides nature-inspired metaheuristic algorithms for continuous
optimization, feature selection, and parameter optimization — focusing on
algorithms not yet available in [NiaPy](https://github.com/NiaOrg/NiaPy).
More components will be added over time.

## Installation

```bash
pip install ikn-library
```

With scikit-learn support for feature selection:

```bash
pip install "ikn-library[ml]"
```

## Quick example

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import AntColonyOptimization

task = Task(problem=Sphere(dimension=10), max_evals=10000)
algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
best_x, best_fitness = algo.run(task)

print("Best fitness:", best_fitness)
```

## How it works

The library follows a NiaPy-like workflow built from three pieces:

1. **`Problem`** — defines the search space and the objective function.
   Subclass it and implement `_evaluate` for custom problems.
2. **`Task`** — wraps a problem with a budget (`max_evals` / `max_iters`),
   counts evaluations, tracks the best solution, and records the
   convergence history.
3. **`Algorithm`** — a metaheuristic that consumes a task via
   `algorithm.run(task)` and returns `(best_x, best_fitness)`.

## Tutorials

- **[Getting Started](getting-started.md)** — tasks, custom problems,
  maximization, and the available algorithms.
- **[Feature Selection](feature-selection.md)** — wrapper-based feature
  selection with Binary ACO and a scikit-learn estimator.
- **[Parameter Optimization](parameter-optimization.md)** — tuning model
  hyperparameters (e.g. SVM `C` and `gamma`) with continuous ACO-R,
  including log-scale search spaces and convergence plotting.
- **[API Reference](api.md)** — full reference generated from the
  docstrings.
