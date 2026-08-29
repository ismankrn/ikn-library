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

## Integer and categorical hyperparameters

The search space is continuous, but `decode` can map a dimension onto
**categorical** choices by splitting its `[0, 1]` range into equal
segments — and onto **integers** by rounding. Here the SVM's `kernel`
becomes a third, categorical search dimension:

```python
class SVMTuningWithKernel(Problem):
    KERNELS = ("rbf", "poly", "sigmoid")

    def __init__(self, X, y, cv=5):
        super().__init__(dimension=3, lower=[-2.0, -4.0, 0.0], upper=[3.0, 1.0, 1.0])
        self.X, self.y, self.cv = X, y, cv

    def decode(self, x):
        kernel_index = min(int(x[2] * len(self.KERNELS)), len(self.KERNELS) - 1)
        return {
            "C": 10.0 ** x[0],
            "gamma": 10.0 ** x[1],
            "kernel": self.KERNELS[kernel_index],
        }

    def _evaluate(self, x):
        model = SVC(**self.decode(x))
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()
```

How the categorical mapping works:

- `x[2]` lives in `[0, 1]`; multiplying by the number of options and
  truncating with `int()` splits the range into three equal segments —
  `[0, 1/3)` → `"rbf"`, `[1/3, 2/3)` → `"poly"`, `[2/3, 1]` →
  `"sigmoid"` — so every category gets the same share of the search
  space.
- The `min(..., len(KERNELS) - 1)` guard handles the edge value
  `x[2] = 1.0`, which would otherwise index one past the end.
- For **integer** parameters, map and round instead, e.g.
  `n_neighbors = int(round(1 + x[0] * 29))` for a range of 1–30.

Running it with the same task setup as above yields:

```text
Best parameters: {'C': 7.5849, 'gamma': 0.0185, 'kernel': 'rbf'}
Cross-validated accuracy: 0.9825
```

The optimizer settled on the RBF kernel on its own — the choice of
kernel was part of the search, not an assumption.

!!! note "A caveat on categorical dimensions"
    Continuous algorithms assume nearby points have similar fitness.
    That holds within one category's segment but breaks at segment
    borders, where a tiny step flips the category. With a few categories
    this works well in practice; for many unordered categories a
    discrete algorithm is the better tool.

## Example: MLP architecture search

The same recipe extends to neural-network architecture: search the
**number of hidden layers** and the **nodes per layer** of an
`MLPClassifier`, with the **validation loss** as the fitness. Unlike the
SVM examples, this fitness is *minimized*, so the task needs no
`optimization_type` (minimization is the default):

```python
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class MLPTuning(Problem):
    """Search: number of hidden layers (1-3) and nodes per layer (8-128)."""

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__(dimension=2, lower=[0.0, 3.0], upper=[1.0, 7.0])
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def decode(self, x):
        n_layers = min(int(x[0] * 3), 2) + 1        # 1, 2, or 3 layers
        n_nodes = int(round(2.0 ** x[1]))           # 8..128 nodes (log2 scale)
        return {"hidden_layer_sizes": (n_nodes,) * n_layers}

    def _evaluate(self, x):
        model = MLPClassifier(**self.decode(x), max_iter=300, random_state=0)
        model.fit(self.X_train, self.y_train)
        return log_loss(self.y_val, model.predict_proba(self.X_val))


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

problem = MLPTuning(X_train, y_train, X_val, y_val)
task = Task(problem=problem, max_evals=60)   # minimization by default
algo = AntColonyOptimization(population_size=8, archive_size=12, seed=42)
best_x, best_loss = algo.run(task)

print("Best architecture:", problem.decode(best_x))
print("Validation log-loss:", best_loss)
```

Decoding details:

- **Number of layers** uses the categorical mapping from the previous
  section: `min(int(x[0] * 3), 2) + 1` gives 1, 2, or 3 layers with
  equal shares of the search space.
- **Nodes per layer** is searched on a **log2 scale**: `x[1]` in
  `[3, 7]` maps to `2^x` = 8..128 nodes, so doubling the width is one
  unit of search distance — the same reasoning as searching `C` and
  `gamma` in log10.
- Each fitness evaluation trains a full network, so the budget is small
  (`max_evals=60`); a fixed `random_state` keeps the fitness
  deterministic (see below).

Output:

```text
Best architecture: {'hidden_layer_sizes': (23,)}
Validation log-loss: 0.062
```

For comparison, the default `MLPClassifier` architecture `(100,)`
reaches a validation log-loss of 0.0721 on the same split — the search
found that a *smaller* single-layer network fits this dataset better.

!!! warning "Is validation loss a valid fitness? Yes — with care"
    Using the validation loss as the fitness is standard practice, and
    the *loss* is actually a better search signal than accuracy (it is
    smooth: it distinguishes a confident correct prediction from a
    barely-correct one). But four caveats keep it honest:

    1. **The validation set must be disjoint from the training set** —
       here the split does that; the loss on training data would just
       reward memorization.
    2. **Report final results on a third, untouched test set.** The
       search consumed the validation set, so the best validation loss
       is optimistically biased — the same three-split discipline as in
       [Ensemble Weights](ensemble.md#the-protocol-three-splits).
    3. **Fix the training seed** (`random_state=0` above). Network
       training is stochastic; without a fixed seed the same
       architecture returns different losses and the optimizer partly
       chases noise. (The remaining caveat: results are then tied to
       one initialization — averaging a few seeds is more robust but
       proportionally more expensive.)
    4. **"Final" loss deserves early stopping.** The loss at the last
       epoch can be worse than the model's best point if the network
       overfits late in training. Passing `early_stopping=True` to
       `MLPClassifier` (or restoring the best epoch in other
       frameworks) makes the fitness reflect the best achievable
       model rather than an arbitrary stopping point.

To visualize how the best score improves over the iterations, see the
teaching note [Plotting Convergence](convergence-plot.md).

A complete runnable script is available at
[`examples/parameter_optimization.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/parameter_optimization.py).
