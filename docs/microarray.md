# Microarray Data

The `ikn_library.microarray` module turns NCBI GEO microarray series
into ML-ready tables: one function call downloads (with caching), parses
the series-matrix file, and returns expression values plus sample
metadata as pandas DataFrames.

## Loading a GEO series

```python
from ikn_library.microarray import load_geo

data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")

data.X          # DataFrame (n_samples, n_probes) — ready for scikit-learn
data.metadata   # DataFrame of sample characteristics (disease, location, ...)
data.y("disease")   # one metadata column as labels, aligned with X
```

- Passing an **accession** (`"GSE11223"`) downloads the series-matrix
  file once from the NCBI FTP server and caches it under
  `~/.ikn_library/geo/` (change with `cache_dir=`). Subsequent calls
  read from the cache — experiments run offline and fast.
- Passing a **local path** to a `*_series_matrix.txt[.gz]` file skips
  the network entirely. Use this for multi-platform series, which ship
  one matrix file per platform.

## Handling missing values

Microarray tables routinely contain missing measurements. Two options,
applied in this order:

- `dropna_threshold=0.1` — drop probes with more than 10% missing values.
- `impute="mean"` (or `"median"`) — fill the remaining gaps per probe;
  entirely-missing probes are dropped.

## Normalization

Four standard normalizations for expression tables, all taking and
returning a samples x probes DataFrame:

```python
from ikn_library.microarray import (
    log2_transform, quantile_normalize, zscore, median_center,
)

X = log2_transform(X)       # raw linear intensities -> log2(x + 1)
X = quantile_normalize(X)   # identical distribution for every sample
X = median_center(X)        # remove per-sample intensity shifts
X = zscore(X)               # zero mean / unit variance per probe
```

Which to use when:

- **`log2_transform`** — only for raw, linear-scale intensities. Many
  GEO series matrices are already log-transformed (values roughly in
  [-15, 15], often negative — like GSE11223's log ratios); the function
  raises an error on negative input as a guard.
- **`quantile_normalize`** — the de-facto microarray standard (Bolstad
  et al., 2003) for making arrays comparable before analysis. Requires
  complete data, so apply `impute=` in `load_geo` first. Tied values
  (including imputed ones) share their average-rank value.
- **`median_center`** — a lighter alternative that only removes global
  per-sample shifts.
- **`zscore`** — per-probe standardization; the usual last step before
  distance-based models such as KNN or SVM.

A sensible pipeline for classification:
`load_geo(..., dropna_threshold=0.1, impute="mean")` ->
`quantile_normalize` -> `top_variance` -> `zscore`.

## Dimensionality: variance filtering

Expression sets are extremely wide (GSE11223: 202 samples x ~39,000
probes after cleaning). A standard first step is an unsupervised
variance filter:

```python
from ikn_library.microarray import top_variance

X = top_variance(data.X, 500)   # keep the 500 most variable probes
```

## End-to-end: GEO to feature selection

The module plugs directly into the metaheuristic feature-selection
workflow. Two things differ from the [Feature Selection](feature-selection.md)
page, and both are forced by the shape of expression data:

- **A linear model, not KNN.** With 161 training samples and 200 probes,
  distance stops being informative: 5-NN scores 0.46-0.51 on the
  held-out split here — *below* the 0.63 majority-class rate — while a
  scaled logistic regression reaches 0.83. `n` far smaller than `p` is
  the normal situation for microarray data, so pick the model
  accordingly.
- **Three folds, not five.** 161 training rows split across two
  unbalanced classes leave thin folds; three keeps each one usable.

Everything else is the same discipline as elsewhere in these docs: the
test set is carved out before the search, the scaler sits inside the
pipeline so it is refitted per fold, and the folds are shuffled with a
fixed seed.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.microarray import load_geo, top_variance
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization


def model():
    """A fresh scaler + logistic regression: the scaler is refitted per fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
X = top_variance(data.X, 200)
y = data.y("disease")           # UC vs Normal
print("Labels:", {k: int(v) for k, v in y.value_counts().items()})

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y.values)
print("train:", X_train.shape, " test:", X_test.shape)

problem = FeatureSelectionProblem(
    X_train, y_train, estimator=model(),
    cv=StratifiedKFold(3, shuffle=True, random_state=42))
task = Task(problem=problem, max_evals=2000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)

selected = problem.selected_features(best_x)
selected_probes = X.columns[selected]

before = model().fit(X_train, y_train).score(X_test, y_test)
after = model().fit(X_train[:, selected], y_train).score(X_test[:, selected], y_test)
majority = max(np.unique(y_test, return_counts=True)[1]) / len(y_test)

print(f"majority class in test   : {majority:.4f}")
print(f"all 200 probes           : test accuracy = {before:.4f}")
print(f"{len(selected)} ACO-selected probes : test accuracy = {after:.4f}")
```

Output:

```text
Labels: {'UC': 129, 'Normal': 73}
train: (161, 200)  test: (41, 200)
majority class in test   : 0.6341
all 200 probes           : test accuracy = 0.8293
98 ACO-selected probes : test accuracy = 0.8537
```

Half the probes do the work of all 200. Read that as *no worse with half
the inputs* rather than as an improvement: 0.8537 against 0.8293 on 41
test samples is a single prediction, well inside the noise of one split.
The saving in assays is the solid part of the result, and it is the part
that matters when each probe costs money to measure.

!!! warning "Budget matters more here than on tabular data"
    The same search with `max_evals=200` — the budget a small tabular
    problem gets away with — selected 103 probes that scored **0.7317**,
    clearly worse than using everything. A 200-dimensional binary space
    needs a real budget before its answer means anything; 2000
    evaluations take about 35 seconds here.

!!! note "This example skips normalization"
    It runs straight from `load_geo` to `top_variance` to keep the
    snippet short. Quantile-normalizing first changes the final test
    score by one sample (0.8049 against 0.8293), so nothing here rests
    on that choice — but on data pooled from several batches, it will.

A complete runnable script is available at
[`examples/microarray_pipeline.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/microarray_pipeline.py).
