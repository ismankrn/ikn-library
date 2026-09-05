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

## Example: MLP architecture search with Keras

The same recipe extends to neural-network architecture: search the
**number of hidden layers** and the **nodes per layer**, with the
**validation loss** as the fitness. Unlike the SVM examples this fitness
is *minimized*, so the task needs no `optimization_type` (minimization
is the default).

The data needs one more split first. The SVM examples cross-validated
inside `X_search`; a network that is trained once per evaluation is
better served by a single fixed validation set, carved out of
`X_search` so the test set stays untouched:

```python
# A held-out validation split, taken from the search data — the test set
# carved out at the top of the page stays untouched
X_train, X_val, y_train, y_val = train_test_split(
    X_search, y_search, test_size=0.25, stratify=y_search, random_state=42)

# The estimator here is not a pipeline, so the scaler is fitted on the
# training rows only and merely applied to the others
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))
print("train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)
```

Output:

```text
train: (341, 30)  val: (114, 30)  test: (114, 30)
```

With Keras the fitness comes straight from the training history: `fit`
accepts `validation_data=` and returns a `History`, so
`min(history.history["val_loss"])` is the loss at the network's best
epoch. (That is a Keras idiom — scikit-learn's `fit` returns the
estimator itself, so with an `MLPClassifier` the loss would be computed
explicitly with `log_loss`.)

A search trains hundreds of networks and keeps the one that scored best
on the validation set. It is tempting to save that network to disk — it
is already trained, after all — but the weights that won are the weights
that happened to suit *this* validation set out of everything tried.
Saving them ships the lottery ticket.

So the fitness records the **recipe** instead: the architecture, and the
epoch at which it peaked. The weights are allowed to die with the
function scope. `decode` maps the four search dimensions onto a layer
count and three widths with the fair-partition pattern from the
[previous section](#integer-and-categorical-hyperparameters); widths
beyond the chosen layer count are simply ignored.

```python
import numpy as np
from tensorflow import keras


def build(architecture, n_features):
    """The only model factory — used during the search AND for the refit,
    so the retrained network is identical to the one that was searched."""
    model = keras.Sequential(
        [keras.layers.Input(shape=(n_features,))]
        + [keras.layers.Dense(n, activation="relu") for n in architecture]
        + [keras.layers.Dense(1, activation="sigmoid")]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


class KerasMLPTuningRecipe(Problem):
    """Search: 1-3 hidden layers, each 16-128 units wide. What the search
    keeps is the winning recipe, not the winning weights."""

    MAX_LAYERS = 3
    MIN_NODES, MAX_NODES = 16, 128

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__(dimension=1 + self.MAX_LAYERS, lower=0.0, upper=1.0)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        # What is tracked is the RECIPE — there is no model.save() in this class
        self.best = {"val_loss": np.inf, "architecture": None, "epoch": None}
        self.n_trained = 0

    def decode(self, x):
        n_layers = min(int(x[0] * self.MAX_LAYERS), self.MAX_LAYERS - 1) + 1
        span = self.MAX_NODES - self.MIN_NODES
        return tuple(
            self.MIN_NODES + min(int(x[i + 1] * (span + 1)), span)
            for i in range(n_layers)
        )

    def _evaluate(self, x):
        architecture = self.decode(x)
        keras.utils.set_random_seed(0)          # deterministic across candidates
        model = build(architecture, self.X_train.shape[1])
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True)
        history = model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=50, batch_size=32, verbose=0, callbacks=[stop])
        val_loss = float(min(history.history["val_loss"]))   # not [-1]
        self.n_trained += 1

        if val_loss < self.best["val_loss"]:
            self.best = {
                "val_loss": val_loss,
                "architecture": architecture,
                "epoch": int(np.argmin(history.history["val_loss"])) + 1,
            }
        return val_loss      # the weights die with this function's scope


problem = KerasMLPTuningRecipe(X_train, y_train, X_val, y_val)
task = Task(problem=problem, max_evals=30)
algo = AntColonyOptimization(population_size=6, archive_size=10, seed=42)
best_x, best_loss = algo.run(task)

print("Models trained    :", problem.n_trained)
print("Models kept       : 0")
print("Best architecture :", problem.best["architecture"])
print(f"Best val_loss     : {problem.best['val_loss']:.4f}")
print("Epoch budget      :", problem.best["epoch"])
```

Output:

```text
Models trained    : 30
Models kept       : 0
Best architecture : (23, 67, 16)
Best val_loss     : 0.0484
Epoch budget      : 35
```

Note what is **absent**: no `model.save()` inside `_evaluate`. Thirty
networks are trained and thirty are discarded; what comes home is an
architecture and an epoch count.

!!! note "One factory, two callers"
    `build()` is used by `_evaluate` *and* by the refit below. That is
    deliberate: if the refit built its network from a second, parallel
    piece of code, nothing would guarantee that the thing you ship is
    the thing you searched. One factory makes the guarantee structural.

!!! note "Why `restore_best_weights=True` is still here"
    The fitness is `min(...)` over epochs, while the weights left in
    memory after `fit` are the *last* epoch's. That mismatch is a real
    bug when the model is saved — the score and the artefact would
    describe different networks. Discarding the weights removes the
    failure mode entirely; the callback stays because it costs nothing
    and keeps the in-memory model consistent with the number reported.

!!! note "TensorFlow is not a dependency"
    `ikn-library` does not require TensorFlow — the optimizer only ever
    sees a `Problem` with an `_evaluate` method, so any framework works
    inside it. Install TensorFlow yourself (`pip install tensorflow`)
    to run this variant. Each evaluation trains a network from scratch,
    which is why the budget is 30 rather than the SVM examples' 150.

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
    3. **Fix the training seed** (`set_random_seed(0)` above). Network
       training is stochastic; without a fixed seed the same
       architecture returns different losses and the optimizer partly
       chases noise. (The remaining caveat: results are then tied to
       one initialization — averaging a few seeds is more robust but
       proportionally more expensive.)
    4. **Never score the last epoch.** The loss at the last epoch can be
       worse than the model's best point if the network overfits late in
       training, which would make the fitness an arbitrary stopping
       point rather than what the architecture can do. The code above
       does this correctly — `EarlyStopping` plus `min(...)` over the
       history; with scikit-learn's `MLPClassifier`, pass
       `early_stopping=True`.

### Refitting on train + validation

Every decision is frozen at this point: the architecture and the epoch
budget were both chosen *from the validation set*. The validation set
has finished its job as a judge — so it can join the training data.

```python
best_architecture = problem.best["architecture"]   # the recipe
epoch_budget = problem.best["epoch"]               # inherited from the search

X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])
print("training rows     :", X_full.shape[0], "(train + val)")

FINAL_SEEDS = (42, 142, 242)
final_models = []
for seed in FINAL_SEEDS:
    keras.utils.set_random_seed(seed)              # a FRESH initialization
    model = build(best_architecture, X_full.shape[1])
    model.fit(X_full, y_full,
              epochs=epoch_budget,                 # fixed: nothing left to monitor
              batch_size=32,                       # (no validation_data,
              verbose=0, shuffle=True)             #  no early stopping)
    model.save(f"final_mlp_seed{seed}.keras")
    final_models.append(model)

print("saved             :", [f"final_mlp_seed{s}.keras" for s in FINAL_SEEDS])
```

Output:

```text
training rows     : 455 (train + val)
saved             : ['final_mlp_seed42.keras', 'final_mlp_seed142.keras', 'final_mlp_seed242.keras']
```

Saving is legitimate *here*: no selection of any kind touched these
weights. They are the product of a frozen recipe applied to fixed data.
`model.save("name.keras")` writes Keras 3's native format; `.h5` still
works and is what older tooling expects, but it prints a legacy warning.
These results were produced with **Keras 3.15** on **TensorFlow 2.21**.

!!! question "Why a fixed epoch count, with no early stopping?"
    Because there is no honest data left to monitor — the validation set
    is inside `X_full` now. The best epoch found during the search is a
    reasonable estimate of how long this architecture needs, and since
    the refit sees 33% more rows, that budget errs on the short side,
    which is the safe direction.

    If that trade feels uncomfortable, the compromise is to refit on
    `X_train` only, with early stopping on `X_val` as before: fresh
    weights, one fresh initialization per seed, at the cost of the extra
    data.

### Reporting: open the test set once

Three models, three seeds, one look at the test set:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

rows = []
for seed, model in zip(FINAL_SEEDS, final_models):
    proba = model.predict(X_test, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    rows.append({"seed": seed,
                 "test_acc": accuracy_score(y_test, pred),
                 "test_f1": f1_score(y_test, pred)})

report = pd.DataFrame(rows)
print(report.round(4))
print(f"Test accuracy: {report.test_acc.mean():.4f} +/- {report.test_acc.std():.4f}")
print(f"Test F1      : {report.test_f1.mean():.4f} +/- {report.test_f1.std():.4f}")

# Optional: average the three seeds' probabilities into one ensemble
proba_ens = np.mean([m.predict(X_test, verbose=0).ravel()
                     for m in final_models], axis=0)
print("Ensemble acc :", round(accuracy_score(y_test, (proba_ens >= 0.5).astype(int)), 4))
```

Output:

```text
   seed  test_acc  test_f1
0    42    0.9649   0.9718
1   142    0.9737   0.9790
2   242    0.9474   0.9571
Test accuracy: 0.9620 +/- 0.0134
Test F1      : 0.9693 +/- 0.0112
Ensemble acc : 0.9561
```

The three seeds span 0.9474 to 0.9737 — 2.6 accuracy points, on an
identical recipe and identical data, from nothing but the weight
initialization. Reporting whichever single seed you happened to run
first would have been a coin flip across that range, which is the whole
argument for the loop. `pandas`' `std` is the sample standard deviation,
so the honest headline is **0.9620 ± 0.0134**.

The probability-averaging ensemble scores 0.9561 here — *below* the mean
of its own members. Averaging seeds usually helps a little and is worth
trying, but on 114 test rows these differences are one or two
predictions; do not read a winner into them.

!!! note "The test number is lower than the validation loss suggests"
    A validation loss of 0.0484 looks excellent, but the refit models
    average 0.9620 accuracy on the test set. That is caveat 2 in action:
    the search consumed the validation set, so its score is
    optimistically biased. The test number is the honest one — and it is
    honest precisely because these weights were never selected on
    anything.

!!! warning "Baselines must be refit the same way"
    If the point of the exercise is "the search found something better
    than a sensible default", the default has to go through the same
    three-seed refit on `X_full`, with an epoch budget from its own
    development run, and land in the same table. Comparing a refit
    search result against a single old baseline run compares two
    protocols, not two architectures.

!!! note "If the fitness used cross-validation instead of a validation split"
    Nothing changes except where the two numbers come from: the epoch
    budget becomes the median `best_epoch` across the winning
    configuration's folds, and `X_full` becomes the whole development
    set. The refit, the seeds and the single look at the test set are
    identical.

To visualize how the best score improves over the iterations, see the
teaching note [Plotting Convergence](convergence-plot.md).

A complete runnable script is available at
[`examples/parameter_optimization.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/parameter_optimization.py).
