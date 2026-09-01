# Hyperparameter Optimization

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

### The same search with Keras / TensorFlow

With Keras, the fitness comes straight from the training history:
`fit` accepts `validation_data=` and returns a `History` whose
`history["val_loss"][-1]` is the final validation loss. (This is a
Keras idiom — scikit-learn's `fit` returns the estimator itself, so
there the loss is computed explicitly with `log_loss`, as above.)
The search space and `decode` are unchanged; only `_evaluate` differs:

```python
from tensorflow import keras


class KerasMLPTuning(Problem):
    """Same search space as MLPTuning, with a Keras model and
    history-based fitness."""

    MAX_LAYERS = 3
    MIN_NODES, MAX_NODES = 16, 128

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__(dimension=1 + self.MAX_LAYERS, lower=0.0, upper=1.0)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def decode(self, x):
        n_layers = min(int(x[0] * self.MAX_LAYERS), self.MAX_LAYERS - 1) + 1
        span = self.MAX_NODES - self.MIN_NODES
        return tuple(
            self.MIN_NODES + min(int(x[i + 1] * (span + 1)), span)
            for i in range(n_layers)
        )

    def _evaluate(self, x):
        keras.utils.set_random_seed(0)   # deterministic fitness
        model = keras.Sequential(
            [keras.layers.Input(shape=(self.X_train.shape[1],))]
            + [keras.layers.Dense(n, activation="relu") for n in self.decode(x)]
            + [keras.layers.Dense(1, activation="sigmoid")]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        history = model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=50, batch_size=32, verbose=0)
        return history.history["val_loss"][-1]


problem = KerasMLPTuning(X_train, y_train, X_val, y_val)
task = Task(problem=problem, max_evals=30)
algo = AntColonyOptimization(population_size=6, archive_size=10, seed=42)
best_x, best_loss = algo.run(task)

print("Best architecture:", problem.decode(best_x))
print("Final val_loss  :", best_loss)
```

Output:

```text
Best architecture: (61,)
Final val_loss  : 0.0652
```

!!! note "TensorFlow is not a dependency"
    `ikn-library` does not require TensorFlow — the optimizer only ever
    sees a `Problem` with an `_evaluate` method, so any framework works
    inside it. Install TensorFlow yourself (`pip install tensorflow`)
    to run this variant. Training here is slower per evaluation than
    the scikit-learn version, hence the smaller budget; with Keras you
    can also pass an `EarlyStopping` callback and return
    `min(history.history["val_loss"])` for the caveat-4 variant
    discussed above.

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

### Keeping the best model as you search

By default a search hands back the best *vector*, and you rebuild the
model from it — which means training the winning architecture a second
time. That is wasteful: the search already trained exactly that model
while evaluating it, then threw it away.

The fix is four lines inside `_evaluate`. Track the best loss seen so
far, and whenever an evaluation beats it, write that model to disk
before returning:

```python
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class KerasMLPTuningWithCheckpoint(Problem):
    """Same search as KerasMLPTuning, but it keeps the best model on disk."""

    MAX_LAYERS = 3
    MIN_NODES, MAX_NODES = 16, 128

    def __init__(self, X_train, y_train, X_val, y_val, path="best_model.h5"):
        super().__init__(dimension=1 + self.MAX_LAYERS, lower=0.0, upper=1.0)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.path = path
        self.best_loss = np.inf          # nothing beaten yet
        self.best_architecture = None
        self.n_trained = self.n_saved = 0

    def decode(self, x):
        n_layers = min(int(x[0] * self.MAX_LAYERS), self.MAX_LAYERS - 1) + 1
        span = self.MAX_NODES - self.MIN_NODES
        return tuple(
            self.MIN_NODES + min(int(x[i + 1] * (span + 1)), span)
            for i in range(n_layers)
        )

    def _evaluate(self, x):
        keras.utils.set_random_seed(0)
        architecture = self.decode(x)
        model = keras.Sequential(
            [keras.layers.Input(shape=(self.X_train.shape[1],))]
            + [keras.layers.Dense(n, activation="relu") for n in architecture]
            + [keras.layers.Dense(1, activation="sigmoid")]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        history = model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=50, batch_size=32, verbose=0)
        val_loss = history.history["val_loss"][-1]
        self.n_trained += 1

        if val_loss < self.best_loss:      # a new best: keep this model
            self.best_loss = val_loss
            self.best_architecture = architecture
            model.save(self.path)          # <- saved as HDF5 (.h5)
            self.n_saved += 1
        return val_loss
```

The search itself is unchanged. Note the **three-way split**: the test
set is carved out first and never touched during the search, exactly as
caveat 2 above requires, and the scaler is fitted on the training part
only:

```python
X, y = load_breast_cancer(return_X_y=True)
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))
print("train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)

problem = KerasMLPTuningWithCheckpoint(X_train, y_train, X_val, y_val)
task = Task(problem=problem, max_evals=30)
algo = AntColonyOptimization(population_size=6, archive_size=10, seed=42)
best_x, best_loss = algo.run(task)

print("Best architecture :", problem.best_architecture)
print("Best val_loss     :", round(best_loss, 4))
print("Models trained    :", problem.n_trained)
print("Models saved      :", problem.n_saved)
print("Saved file        :", problem.path,
      f"({Path(problem.path).stat().st_size} bytes)")
```

Output:

```text
train: (341, 30)  val: (114, 30)  test: (114, 30)
Best architecture : (24, 61)
Best val_loss     : 0.0462
Models trained    : 30
Models saved      : 3
Saved file        : best_model.h5 (59440 bytes)
```

Look at the last two counters: **30 models were trained, but only 3
were written to disk.** The file is overwritten only when the loss
actually improves, so at the end it holds the single best network the
search ever saw — already trained, with its weights exactly as they
were when that score was measured.

### Loading the model and predicting

Nothing is retrained here. The file is read and used directly:

```python
model = keras.models.load_model("best_model.h5")
model.summary()
```

Output:

```text
Model: "sequential_12"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_33 (Dense)                │ (None, 24)             │           744 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_34 (Dense)                │ (None, 61)             │         1,525 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_35 (Dense)                │ (None, 1)              │            62 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,333 (9.12 KB)
 Trainable params: 2,331 (9.11 KB)
 Non-trainable params: 0 (0.00 B)
```

The architecture came back intact — two hidden layers of 24 and 61
units, matching `best_architecture` above. Now predict on the test set
the search never saw:

```python
from sklearn.metrics import accuracy_score, classification_report

proba = model.predict(X_test, verbose=0).ravel()
y_pred = (proba >= 0.5).astype(int)
print("test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred,
                            target_names=load_breast_cancer().target_names))
```

Output:

```text
test accuracy: 0.9561

              precision    recall  f1-score   support

   malignant       0.91      0.98      0.94        42
      benign       0.99      0.94      0.96        72

    accuracy                           0.96       114
   macro avg       0.95      0.96      0.95       114
weighted avg       0.96      0.96      0.96       114
```

Because the output layer is a sigmoid, `predict` returns
probabilities; the threshold is yours to choose:

```python
print("first 5 probabilities:", np.round(proba[:5], 4))
print("predicted            :", y_pred[:5])
print("actual               :", y_test[:5])
```

Output:

```text
first 5 probabilities: [0.000e+00 1.000e+00 2.000e-04 1.777e-01 0.000e+00]
predicted            : [0 1 0 0 0]
actual               : [0 1 0 1 0]
```

The fourth sample is worth a look: at a probability of 0.178 the model
leans malignant and is wrong. Three of these five predictions are made
with near-total confidence and one is genuinely uncertain — which is
the argument for reading probabilities rather than only labels, and for
choosing a threshold that suits the cost of each kind of error.

!!! note "Test accuracy is lower than the validation loss suggests"
    A validation loss of 0.0462 looks excellent, but the test accuracy
    is 0.9561 — noticeably below what the earlier two-way-split
    examples on this page report. That is caveat 2 in action: the
    search consumed the validation set, so its score is optimistically
    biased, and the training set here is smaller (341 rows) because a
    third split was carved out. The test number is the honest one.

!!! warning "`.h5` is a legacy format in Keras 3"
    `model.save("best_model.h5")` still works, but Keras 3 prints a
    warning recommending its native format instead:

    ```text
    WARNING:absl:You are saving your model as an HDF5 file via
    `model.save()`. This file format is considered legacy. We recommend
    using instead the native Keras format, e.g.
    `model.save('my_model.keras')`.
    ```

    Use `.h5` when you need it — older tooling and non-Keras readers
    still expect HDF5. Otherwise change one character in the filename:
    `best_model.keras` saves and loads with the same two calls, with no
    warning. These results were produced with **Keras 3.15** on
    **TensorFlow 2.21**.

!!! tip "The saved model was trained on the training split only"
    It is the network exactly as evaluated — which is what makes the
    score meaningful. Retraining it on train + validation afterwards
    usually helps a little, but then the model you ship is no longer
    the one you measured. Decide which you want before you report a
    number.

To visualize how the best score improves over the iterations, see the
teaching note [Plotting Convergence](convergence-plot.md).

A complete runnable script is available at
[`examples/parameter_optimization.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/parameter_optimization.py).
