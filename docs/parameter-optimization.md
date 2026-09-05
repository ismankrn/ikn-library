# Hyperparameter Optimization

Metaheuristics shine at tuning model hyperparameters when the search
space is continuous and the objective (a cross-validated score) is
expensive and non-differentiable. This tutorial tunes an SVM with the
continuous ACO-R algorithm.

The recipe:

1. **Carve out a test set before anything else.** The search will
   consume whatever data it is allowed to score against.
2. Subclass `Problem`; each dimension of the search space is one
   hyperparameter.
3. In `_evaluate`, decode the solution vector into hyperparameter values
   and return the cross-validated score.
4. Wrap it in a `Task` with `OptimizationType.MAXIMIZATION` (higher
   score is better) and run a continuous algorithm such as
   `AntColonyOptimization`.
5. Report the **test** score, not the best score the search found.

Requires scikit-learn:

```bash
pip install "ikn-library[ml]"
```

## Splitting the data before the search

Two decisions here shape every number on this page, so they come first:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

X, y = load_breast_cancer(return_X_y=True)

# The test set is carved out BEFORE the search starts — the search never sees it
X_search, X_test, y_search, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# One fold object, shuffled with a fixed seed: independent of row order,
# and identical for every candidate the search evaluates
CV = StratifiedKFold(5, shuffle=True, random_state=42)

print("search:", X_search.shape, " test:", X_test.shape)
```

Output:

```text
search: (455, 30)  test: (114, 30)
```

Use an explicit fold object with `shuffle=True` and a fixed
`random_state` rather than a bare `cv=5`, which builds folds in file
order — on a dataset sorted by class that alone can wreck the scores.
And understand what the fold reuse costs: the CV score the search
maximizes gets **consumed** exactly the way a validation split does.
Hundreds of evaluations against the same five folds mean the best of
them is optimistically biased, which is why the number reported at the
end always comes from the test set.

## Defining the problem

An SVM with an RBF kernel has two key hyperparameters, `C` and `gamma`.
Both are scale parameters, so we search their **base-10 logarithms** —
`10^x` maps a uniform search dimension onto several orders of magnitude:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ikn_library.problems import Problem


class SVMTuning(Problem):
    """Search log10(C) in [-2, 3] and log10(gamma) in [-4, 1]."""

    def __init__(self, X, y, cv):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y, self.cv = X, y, cv

    def decode(self, x):
        return {"C": 10.0 ** x[0], "gamma": 10.0 ** x[1]}

    def _evaluate(self, x):
        # Scaler INSIDE the pipeline: refitted per fold, never sees the fold
        # it is validated on
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", **self.decode(x)))
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()
```

Two things are doing quiet work here:

- The `decode` helper keeps the mapping between the search vector and
  the actual hyperparameters in one place, so you can reuse it on the
  final result.
- `make_pipeline(StandardScaler(), ...)` puts the scaler *inside* the
  cross-validated estimator. Scaling the whole array once, before the
  split, would let every fold's mean and variance leak into the fold
  used to score it — and would leak the test set into all of them.

## Running the optimization

```python
from ikn_library import OptimizationType, Task
from ikn_library.algorithms import AntColonyOptimization

problem = SVMTuning(X_search, y_search, cv=CV)
task = Task(
    problem=problem,
    max_evals=150,
    optimization_type=OptimizationType.MAXIMIZATION,
)
algo = AntColonyOptimization(population_size=10, archive_size=15, seed=42)
best_x, best_score = algo.run(task)

params = problem.decode(best_x)
final = make_pipeline(StandardScaler(),
                      SVC(kernel="rbf", **params)).fit(X_search, y_search)

print(f"Best parameters : C={params['C']:.4f}, gamma={params['gamma']:.4f}")
print(f"Best CV score   : {best_score:.4f}  (search maximum, optimistically biased)")
print(f"Test accuracy   : {final.score(X_test, y_test):.4f}  (the number to report)")
```

Output:

```text
Best parameters : C=6.7972, gamma=0.0031
Best CV score   : 0.9780  (search maximum, optimistically biased)
Test accuracy   : 0.9825  (the number to report)
```

The best CV score is the maximum of 150 evaluations against one fixed
set of folds, so it is optimistically biased — the number to report is
the test accuracy. Each fitness evaluation runs a full
cross-validation, so choose `max_evals` according to your time budget.

## Did the tuning actually help?

Comparing the search's best CV score against the default model's CV
score is not a symmetric comparison: the first is the maximum of 150
candidates, the second is a single unselected number. The fair
comparison refits both on the same data and scores both on the test set:

```python
default = make_pipeline(StandardScaler(), SVC())
default_cv = cross_val_score(default, X_search, y_search, cv=CV).mean()
default.fit(X_search, y_search)

print(f"CV   — default {default_cv:.4f}   tuned {best_score:.4f}")
print(f"Test — default {default.score(X_test, y_test):.4f}   tuned "
      f"{final.score(X_test, y_test):.4f}")
print("Test predictions that agree:",
      f"{(default.predict(X_test) == final.predict(X_test)).sum()}/{len(y_test)}")
```

Output:

```text
CV   — default 0.9692   tuned 0.9780
Test — default 0.9825   tuned 0.9825
Test predictions that agree: 114/114
```

This is the **winner's curse**, and it is worth sitting with. On the
folds, tuning bought 0.9 accuracy points. On the test set it bought
nothing at all: the two models make the *same prediction on all 114
rows*. Searching hard for the maximum of a noisy score finds
configurations whose noise happens to point up, and that part of the
advantage does not survive contact with new data. `SVC()`'s defaults
(`C=1`, `gamma="scale"`) were already in a good region for this dataset.

Tuning is not therefore useless — it is how you *find out* that the
default was fine, and on a dataset where the default is badly placed the
gain is real. But the claim has to be made on the test set, and a gain
of a fold-noise's width is not a gain.

## Integer and categorical hyperparameters

The search space is continuous, but `decode` can map a dimension onto
**categorical** choices by splitting its `[0, 1]` range into equal
segments — and onto **integers** by rounding. Here the SVM's `kernel`
becomes a third, categorical search dimension:

```python
class SVMTuningWithKernel(Problem):
    KERNELS = ("rbf", "poly", "sigmoid")

    def __init__(self, X, y, cv):
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
        model = make_pipeline(StandardScaler(), SVC(**self.decode(x)))
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

Running it with the same task setup as above:

```python
problem_k = SVMTuningWithKernel(X_search, y_search, cv=CV)
task_k = Task(problem=problem_k, max_evals=150,
              optimization_type=OptimizationType.MAXIMIZATION)
best_xk, best_scorek = AntColonyOptimization(
    population_size=10, archive_size=15, seed=42).run(task_k)

params_k = problem_k.decode(best_xk)
final_k = make_pipeline(StandardScaler(), SVC(**params_k)).fit(X_search, y_search)

print(f"Best parameters : C={params_k['C']:.4f}, gamma={params_k['gamma']:.4f}, "
      f"kernel={params_k['kernel']}")
print(f"Best CV score   : {best_scorek:.4f}")
print(f"Test accuracy   : {final_k.score(X_test, y_test):.4f}")
```

Output:

```text
Best parameters : C=31.7194, gamma=0.0036, kernel=sigmoid
Best CV score   : 0.9780
Test accuracy   : 0.9825
```

The kernel was part of the search rather than an assumption — and the
result is a third configuration landing on exactly the same numbers as
the other two: CV 0.9780, test 0.9825. Three different corners of the
space, one plateau. When several configurations tie like this, the tie
is the finding; picking the "winner" among them is picking noise.

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


# A held-out validation split, taken from the search data — the test set
# carved out at the top of the page stays untouched
X_train, X_val, y_train, y_val = train_test_split(
    X_search, y_search, test_size=0.25, stratify=y_search, random_state=42)

# The estimator here is not a pipeline, so the scaler is fitted on the
# training rows only and merely applied to the others
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))
print("train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)

problem = MLPTuning(X_train, y_train, X_val, y_val)
task = Task(problem=problem, max_evals=60)   # minimization by default
algo = AntColonyOptimization(population_size=8, archive_size=12, seed=42)
best_x, best_loss = algo.run(task)

print("Best architecture   :", problem.decode(best_x))
print(f"Validation log-loss : {best_loss:.4f}")
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
train: (341, 30)  val: (114, 30)  test: (114, 30)
Best architecture   : {'hidden_layer_sizes': (93,)}
Validation log-loss : 0.0528
```

The default `MLPClassifier` architecture `(100,)` reaches 0.0549 on the
same split, so 60 evaluations bought 0.0021 of log-loss — the search
walked most of the way back to the default and confirmed it. That is a
useful result to be able to state, and the same warning as in the SVM
section applies to reading it as an improvement.

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
print(f"Final val_loss   : {best_loss:.4f}")
```

Output:

```text
Best architecture: (24, 61)
Final val_loss   : 0.0462
```

!!! note "TensorFlow is not a dependency"
    `ikn-library` does not require TensorFlow — the optimizer only ever
    sees a `Problem` with an `_evaluate` method, so any framework works
    inside it. Install TensorFlow yourself (`pip install tensorflow`)
    to run this variant. Training here is slower per evaluation than
    the scikit-learn version, hence the smaller budget. This `_evaluate`
    still returns the *last* epoch's loss; the next section fixes that
    along with everything else.

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
       is optimistically biased — exactly as the SVM example at the top
       of this page demonstrated, and the same three-split discipline
       as in [Ensemble Weights](ensemble.md#the-protocol-three-splits).
    3. **Fix the training seed** (`random_state=0` above). Network
       training is stochastic; without a fixed seed the same
       architecture returns different losses and the optimizer partly
       chases noise. (The remaining caveat: results are then tied to
       one initialization — averaging a few seeds is more robust but
       proportionally more expensive.)
    4. **"Final" loss deserves early stopping.** The loss at the last
       epoch can be worse than the model's best point if the network
       overfits late in training. Pass `early_stopping=True` to
       `MLPClassifier`, or an `EarlyStopping` callback with
       `restore_best_weights=True` in Keras, so the fitness reflects
       the best achievable model rather than an arbitrary stopping
       point.

### Keeping the best model as you search

By default a search hands back the best *vector*, and you rebuild the
model from it — which means training the winning architecture a second
time. That is wasteful: the search already trained exactly that model
while evaluating it, then threw it away.

The fix is a few lines inside `_evaluate`. Track the best loss seen so
far, and whenever an evaluation beats it, write that model to disk
before returning. This version also applies caveat 4 — the two changes
belong together, as the note after the code explains:

```python
from pathlib import Path

import numpy as np


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
        self.best_epoch = None           # needed to refit later
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
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True)
        history = model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=50, batch_size=32, verbose=0,
                            callbacks=[stop])
        val_loss = min(history.history["val_loss"])   # not [-1]
        self.n_trained += 1

        if val_loss < self.best_loss:      # a new best: keep this model
            self.best_loss = val_loss
            self.best_architecture = architecture
            self.best_epoch = int(np.argmin(history.history["val_loss"])) + 1
            model.save(self.path)          # <- saved as HDF5 (.h5)
            self.n_saved += 1
        return val_loss
```

!!! danger "`restore_best_weights=True` is not optional here"
    The fitness is now `min(history.history["val_loss"])` — the loss at
    the network's *best* epoch. Without `restore_best_weights=True`,
    the weights still in memory when `model.save()` runs are the *last*
    epoch's. The score and the artefact would then describe two
    different networks: you would report a number the saved file cannot
    reproduce. Whenever the fitness is a `min` over epochs, the
    checkpoint has to be the epoch that produced it.

The search itself is unchanged, and the splits are already in place from
the previous section: the test set was carved out at the top of the
page, and the scaler was fitted on the training rows only.

```python
problem = KerasMLPTuningWithCheckpoint(X_train, y_train, X_val, y_val)
task = Task(problem=problem, max_evals=30)
algo = AntColonyOptimization(population_size=6, archive_size=10, seed=42)
best_x, best_loss = algo.run(task)

print("Best architecture :", problem.best_architecture)
print(f"Best val_loss     : {best_loss:.4f}")
print("Best epoch        :", problem.best_epoch)
print("Models trained    :", problem.n_trained)
print("Models saved      :", problem.n_saved)
print("Saved file        :", problem.path,
      f"({Path(problem.path).stat().st_size} bytes)")
```

Output:

```text
Best architecture : (23, 67, 16)
Best val_loss     : 0.0484
Best epoch        : 35
Models trained    : 30
Models saved      : 5
Saved file        : best_model.h5 (78464 bytes)
```

Look at the two counters: **30 models were trained, but only 5 were
written to disk.** The file is overwritten only when the loss actually
improves, so at the end it holds the single best network the search ever
saw — already trained, with its weights exactly as they were when that
score was measured.

This search picked a different architecture from the one above,
`(23, 67, 16)` rather than `(24, 61)`, and its 0.0484 is not comparable
with the previous 0.0462: changing the fitness from "last epoch" to
"best epoch, with early stopping" changes the landscape the optimizer
is climbing. They are two different searches, not two attempts at one.

### Loading the model and predicting

Nothing is retrained here. The file is read and used directly:

```python
model = keras.models.load_model("best_model.h5")
model.summary()
```

Output:

```text
Model: "sequential_50"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_137 (Dense)               │ (None, 23)             │           713 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_138 (Dense)               │ (None, 67)             │         1,608 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_139 (Dense)               │ (None, 16)             │         1,088 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_140 (Dense)               │ (None, 1)              │            17 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,428 (13.39 KB)
 Trainable params: 3,426 (13.38 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)
```

The architecture came back intact — three hidden layers of 23, 67 and
16 units, matching `best_architecture` above. Now predict on the test
set the search never saw:

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
first 5 probabilities: [0.000e+00 1.000e+00 5.000e-04 2.516e-01 0.000e+00]
predicted            : [0 1 0 0 0]
actual               : [0 1 0 1 0]
```

The fourth sample is worth a look: at a probability of 0.2516 the model
leans malignant and is wrong. Three of these five predictions are made
with near-total confidence and one is genuinely uncertain — which is
the argument for reading probabilities rather than only labels, and for
choosing a threshold that suits the cost of each kind of error.

!!! note "Test accuracy is lower than the validation loss suggests"
    A validation loss of 0.0484 looks excellent, but the test accuracy
    is 0.9561 — below what the SVM examples on this page report. That
    is caveat 2 in action: the search consumed the validation set, so
    its score is optimistically biased, and the training set here is
    smaller (341 rows) because a third split was carved out. The test
    number is the honest one.

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

### Ship the saved model, or refit from the recipe?

A search leaves you with two different things: a trained artefact, and a
recipe (the architecture, plus the epoch count that worked). They are
not interchangeable, and which one you should use depends on the claim
you are about to make.

!!! tip "Which to use"
    **Use the saved model** — load it, no retraining — when the search
    budget was small (tens of evaluations), there is a single pipeline,
    and the goal is to ship a measured artefact. Its test score is an
    honest measurement *of that artefact*.

    **Refit from the recipe** — best configuration, fresh
    initialization, the epoch count recorded during the search, trained
    on train + validation combined — when either of these holds:

    - **The search budget was large.** The more candidates scored
      against the same validation set, the larger the share of "luck"
      frozen into the saved weights.
    - **The claim is comparative or methodological** — "algorithm X
      finds good architectures", "scheme A beats scheme B". A claim
      about a *method* has to be tested on a refitted recipe, ideally
      averaged over several seeds, not on the single draw that happened
      to win the validation lottery.

```python
# Refit from the recipe: keep the architecture, discard the weights
best_epoch = problem.best_epoch                 # recorded during the search
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

keras.utils.set_random_seed(42)                 # a fresh draw, not the search's
refit = keras.Sequential(
    [keras.layers.Input(shape=(X_full.shape[1],))]
    + [keras.layers.Dense(n, activation="relu")
       for n in problem.best_architecture]
    + [keras.layers.Dense(1, activation="sigmoid")]
)
refit.compile(optimizer="adam", loss="binary_crossentropy")
refit.fit(X_full, y_full, epochs=best_epoch, batch_size=32, verbose=0)

refit_pred = (refit.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
print("Epochs from the search :", best_epoch)
print("Training rows          :", X_full.shape[0], "(train + val)")
print(f"Saved model, test acc  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Refit model, test acc  : {accuracy_score(y_test, refit_pred):.4f}")
```

Output:

```text
Epochs from the search : 35
Training rows          : 455 (train + val)
Saved model, test acc  : 0.9561
Refit model, test acc  : 0.9649
```

The refit wins here — one extra correct prediction out of 114, which is
well within noise, but it also trains on 33% more rows, which is a real
advantage rather than a lucky one. Note what the refit cannot do:
there is no validation set left to early-stop on, which is exactly why
`best_epoch` had to be recorded during the search.

To visualize how the best score improves over the iterations, see the
teaching note [Plotting Convergence](convergence-plot.md).

A complete runnable script is available at
[`examples/parameter_optimization.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/parameter_optimization.py).
