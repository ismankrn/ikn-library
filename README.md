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

## Parameter optimization

Tune model hyperparameters by subclassing `Problem`: each dimension is
one hyperparameter, and `_evaluate` returns the cross-validated score
(searched in log scale where appropriate):

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task
from ikn_library.problems import Problem
from ikn_library.algorithms import AntColonyOptimization

class SVMTuning(Problem):
    def __init__(self, X, y):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y = X, y

    def _evaluate(self, x):
        model = SVC(kernel="rbf", C=10.0 ** x[0], gamma=10.0 ** x[1])
        return cross_val_score(model, self.X, self.y, cv=5).mean()

task = Task(problem=SVMTuning(X, y), max_evals=150,
            optimization_type=OptimizationType.MAXIMIZATION)
best_x, best_score = AntColonyOptimization(population_size=10, seed=42).run(task)
```

See the full tutorial:
[Parameter Optimization](https://ikn-library.readthedocs.io/en/latest/parameter-optimization/)
and the runnable script [examples/parameter_optimization.py](examples/parameter_optimization.py).

## Algorithms

| Algorithm | Class | Domain | Reference |
|---|---|---|---|
| Ant Colony Optimization (ACO-R) | `AntColonyOptimization` | continuous | Socha & Dorigo, EJOR 185(3), 2008 |
| Binary Ant Colony Optimization | `BinaryAntColonyOptimization` | binary / subsets | hyper-cube pheromone update |

More algorithms are planned.

## Microarray data

Load NCBI GEO microarray series into ML-ready tables — downloaded once,
cached locally, with missing-value handling built in:

```python
from ikn_library.microarray import load_geo, top_variance

data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
X = top_variance(data.X, 500)   # (202 samples, 500 most variable probes)
y = data.y("disease")           # UC vs Normal labels from sample metadata
```

The result plugs directly into `FeatureSelectionProblem` — see
[examples/microarray_pipeline.py](examples/microarray_pipeline.py) for the
full GEO-to-feature-selection pipeline.

## Documentation

Full documentation: [ikn-library.readthedocs.io](https://ikn-library.readthedocs.io)

## Development

```bash
pip install -e ".[dev]"
pytest
```
