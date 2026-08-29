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
workflow:

```python
from ikn_library import Task
from ikn_library.microarray import load_geo, top_variance
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
X = top_variance(data.X, 200)
y = data.y("disease")           # UC vs Normal

problem = FeatureSelectionProblem(X.values, y.values, cv=3)
task = Task(problem=problem, max_evals=200)
algo = BinaryAntColonyOptimization(population_size=10, seed=42)
best_x, best_fitness = algo.run(task)

selected_probes = X.columns[problem.selected_features(best_x)]
```

On GSE11223 this pipeline raises the cross-validated KNN accuracy well
above the all-probes baseline while using a fraction of the probes.

A complete runnable script is available at
[`examples/microarray_pipeline.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/microarray_pipeline.py).
