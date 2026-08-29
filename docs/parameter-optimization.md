# Parameter Optimization

Metaheuristics shine at tuning model hyperparameters when the search
space is continuous and the objective (a cross-validated score) is
expensive and non-differentiable. This tutorial tunes an SVM with the
continuous ACO-R algorithm.

The recipe:

1. Subclass `Problem`; each dimension of the search space is one
   hyperparameter.
2. In `_evaluate`, decode the solution vector into hyperparameter values
   and return the cross-validated score.
3. Wrap it in a `Task` with `OptimizationType.MAXIMIZATION` (higher
   score is better) and run a continuous algorithm such as
   `AntColonyOptimization`.

Requires scikit-learn:

```bash
pip install "ikn-library[ml]"
```

## Defining the problem

An SVM with an RBF kernel has two key hyperparameters, `C` and `gamma`.
Both are scale parameters, so we search their **base-10 logarithms** —
`10^x` maps a uniform search dimension onto several orders of magnitude:

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from ikn_library.problems import Problem


class SVMTuning(Problem):
    """Search log10(C) in [-2, 3] and log10(gamma) in [-4, 1]."""

    def __init__(self, X, y, cv=5):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y, self.cv = X, y, cv

    def decode(self, x):
        return {"C": 10.0 ** x[0], "gamma": 10.0 ** x[1]}

    def _evaluate(self, x):
        params = self.decode(x)
        model = SVC(kernel="rbf", **params)
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()
```

The `decode` helper keeps the mapping between the search vector and the
actual hyperparameters in one place, so you can reuse it on the final
result.

## Running the optimization

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from ikn_library import OptimizationType, Task
from ikn_library.algorithms import AntColonyOptimization

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

problem = SVMTuning(X, y, cv=5)
task = Task(
    problem=problem,
    max_evals=150,
    optimization_type=OptimizationType.MAXIMIZATION,
)
algo = AntColonyOptimization(population_size=10, archive_size=15, seed=42)
best_x, best_score = algo.run(task)

print("Best parameters:", problem.decode(best_x))
print("Cross-validated accuracy:", best_score)
```

Each fitness evaluation runs a full cross-validation, so choose
`max_evals` according to your time budget. With `max_evals=150` this
typically beats the default `SVC()` accuracy on the breast-cancer
dataset.

!!! tip "Integer and categorical hyperparameters"
    The search space is continuous, but you can still tune integer
    parameters by rounding inside `_evaluate` (e.g.
    `n_estimators=int(round(x[0]))`) and categorical ones by mapping
    ranges of a dimension onto choices.

To visualize how the best score improves over the iterations, see the
teaching note [Plotting Convergence](convergence-plot.md).

A complete runnable script is available at
[`examples/parameter_optimization.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/parameter_optimization.py).
