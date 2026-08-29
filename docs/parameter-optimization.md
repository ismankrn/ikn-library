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
    """Search: number of hidden layers (1-3), each with its own width (16-128)."""

    MAX_LAYERS = 3
    MIN_NODES, MAX_NODES = 16, 128

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__(dimension=1 + self.MAX_LAYERS, lower=0.0, upper=1.0)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def decode(self, x):
        n_layers = min(int(x[0] * self.MAX_LAYERS), self.MAX_LAYERS - 1) + 1
        span = self.MAX_NODES - self.MIN_NODES
        sizes = tuple(
            self.MIN_NODES + min(int(x[i + 1] * (span + 1)), span)
            for i in range(n_layers)
        )
        return {"hidden_layer_sizes": sizes}

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

- The search space has **4 dimensions**: `x[0]` chooses the number of
  layers, and `x[1..3]` each control the width of one layer — so a
  two-layer network can be wide-then-narrow, narrow-then-wide, or
  anything in between.
- **Number of layers** uses the categorical mapping from the previous
  section: `min(int(x[0] * MAX_LAYERS), MAX_LAYERS - 1) + 1` gives 1,
  2, or 3 layers with equal shares of the search space.
- **Width of each layer** maps `x[i+1]` in `[0, 1]` onto the integer
  range 16..128 with the same fair-partition-plus-edge-guard pattern:
  `MIN_NODES + min(int(x * (span + 1)), span)`, where
  `span = MAX_NODES - MIN_NODES`.
- When `decode` selects fewer than `MAX_LAYERS` layers, the leftover
  width dimensions are simply **ignored** — inactive dimensions are a
  normal feature of variable-length architecture search and do no harm
  beyond mildly enlarging the space.
- Each fitness evaluation trains a full network, so the budget is small
  (`max_evals=60`); a fixed `random_state` keeps the fitness
  deterministic (see below).

Output:

```text
Best architecture: {'hidden_layer_sizes': (75,)}
Validation log-loss: 0.065
```

For comparison, the default `MLPClassifier` architecture `(100,)`
reaches a validation log-loss of 0.0721 on the same split — even with
per-layer widths available, the search settled on a single hidden
layer for this dataset.

!!! note "scikit-learn vs Keras: where the validation loss lives"
    In **scikit-learn**, `model.fit()` returns the estimator itself —
    there is no training history object, so the final validation loss
    is computed explicitly, as above:
    `log_loss(y_val, model.predict_proba(X_val))`. In **Keras**, the
    same fitness would come from the history that `fit` returns:

    ```python
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val), epochs=100)
    return history.history["val_loss"][-1]
    ```

    Both express the same idea — the model's loss on held-out data at
    the end of training. Writing `history.history["val_loss"]` against
    a scikit-learn estimator raises an `AttributeError`, because its
    `fit` returns the estimator, not a history.

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
