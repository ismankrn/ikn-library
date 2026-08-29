# ikn-library

Nature-inspired metaheuristic algorithms for continuous optimization, feature
selection, and parameter optimization — focusing on algorithms not yet
available in [NiaPy](https://github.com/NiaOrg/NiaPy), starting with
**Ant Colony Optimization for continuous domains (ACO-R)**.

## Installation

```bash
pip install ikn-library
```

Or from source (development mode):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import AntColonyOptimization

task = Task(problem=Sphere(dimension=10), max_evals=10000)
algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
best_x, best_fitness = algo.run(task)

print("Best fitness:", best_fitness)
print("Best solution:", best_x)
```

## Custom problems

Subclass `Problem` and implement `_evaluate` — for example, a
cross-validation score for hyperparameter optimization:

```python
import numpy as np
from ikn_library.problems import Problem

class MyProblem(Problem):
    def __init__(self, dimension=10):
        super().__init__(dimension, lower=-10.0, upper=10.0)

    def _evaluate(self, x):
        return float(np.sum(np.abs(x)))
```

Use `OptimizationType.MAXIMIZATION` in the `Task` when higher is better
(e.g. accuracy).

## Feature selection

Wrapper-based feature selection with a scikit-learn estimator
(`pip install ikn-library[ml]`):

```python
from sklearn.datasets import load_breast_cancer

from ikn_library import Task
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

X, y = load_breast_cancer(return_X_y=True)
problem = FeatureSelectionProblem(X, y, cv=5, alpha=0.99)
task = Task(problem=problem, max_evals=1000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)

print("Selected features:", problem.selected_features(best_x))
```

The fitness balances the cross-validated score against the subset size:
`alpha * (1 - cv_score) + (1 - alpha) * n_selected / n_features`.

## Algorithms

| Algorithm | Class | Domain | Reference |
|---|---|---|---|
| Ant Colony Optimization (ACO-R) | `AntColonyOptimization` | continuous | Socha & Dorigo, EJOR 185(3), 2008 |
| Binary Ant Colony Optimization | `BinaryAntColonyOptimization` | binary / subsets | hyper-cube pheromone update |

More algorithms are planned.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
