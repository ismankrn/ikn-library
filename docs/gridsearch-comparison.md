# Comparing Tuning Results from `cv_results_`

A common mistake after hyperparameter tuning is to *re-run* the model
with default parameters just to compare it against the tuned result.
With `GridSearchCV` that re-run is unnecessary: the search has already
evaluated every parameter combination in the grid — including the
default one, if you put it there — and stored every score in
`cv_results_`.

This page shows how to make that comparison honestly, and turn it into
a bar chart.

## The one requirement

!!! warning "Include the default values in the grid"
    The comparison only works if the estimator's **default parameter
    combination is inside the searched grid**. If the defaults are not
    among the evaluated combinations, `cv_results_` has no score for
    them and you would have to evaluate them separately — losing the
    guarantee that both scores came from the exact same CV folds.

For an RBF-kernel `SVC`, the defaults are `C=1.0` and `gamma="scale"`,
so both values go into the grid:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

param_grid = {
    "C": [0.1, 1.0, 10.0, 100.0],        # 1.0     = sklearn default
    "gamma": ["scale", 0.01, 0.1, 1.0],  # "scale" = sklearn default
}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
grid.fit(X, y)
```

## What `cv_results_` looks like

`cv_results_` is a dict of arrays with one entry per combination; as a
DataFrame each row holds the combination (`params`, plus one
`param_<name>` column per parameter) and its cross-validated
performance (`mean_test_score`, `std_test_score`, `rank_test_score`).
With a 4x4 grid there are 16 rows — one per combination:

```python
results = pd.DataFrame(grid.cv_results_)
cols = ["param_C", "param_gamma", "mean_test_score",
        "std_test_score", "rank_test_score"]
print(results[cols].sort_values("rank_test_score").head(6).to_string(index=False))
```

Output:

```text
 param_C param_gamma  mean_test_score  std_test_score  rank_test_score
    10.0        0.01         0.978932        0.006990                1
    10.0       scale         0.977177        0.008921                2
     1.0       scale         0.973638        0.014679                3
   100.0        0.01         0.968374        0.008919                4
     1.0        0.01         0.966636        0.010182                5
     1.0         0.1         0.959587        0.008910                6
```

How to read it:

- **`mean_test_score`** — the score averaged over the 5 CV folds; this
  is the number we compare.
- **`std_test_score`** — the spread across folds; larger values mean
  the score is less certain.
- **`rank_test_score`** — 1 is the winner (`grid.best_params_`). Note
  where the default combination (`C=1.0, gamma="scale"`) sits: **rank 3
  of 16** — decent, but not the best. That is the whole comparison,
  already visible in the table.

## Reading the comparison from `cv_results_`

No model is retrained below — everything is a lookup:

```python
default_params = {"C": 1.0, "gamma": "scale"}
default_row = results.loc[
    results["params"].apply(lambda p: p == default_params)
].iloc[0]
best_row = results.iloc[grid.best_index_]

print("default:", default_params, "->", default_row["mean_test_score"])
print("best   :", grid.best_params_, "->", best_row["mean_test_score"])
```

Output:

```text
default: {'C': 1.0, 'gamma': 'scale'} -> 0.9736376339077782
best   : {'C': 10.0, 'gamma': 0.01} -> 0.9789318428815401
```

Two details worth pointing out to students:

- `grid.best_score_` is by definition the maximum `mean_test_score` in
  `cv_results_` — the "best" row is not special, it is just the row
  that happened to win.
- Because every combination was scored on the **same CV folds**, the
  two numbers are directly comparable. That is exactly what a manual
  re-run with a fresh split would *not* guarantee.

## The bar chart

```python
import matplotlib.pyplot as plt

labels = ["Default\n(C=1.0, gamma='scale')", f"Best\n{grid.best_params_}"]
scores = [default_row["mean_test_score"], best_row["mean_test_score"]]
errors = [default_row["std_test_score"], best_row["std_test_score"]]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, scores, yerr=errors, capsize=6,
              color=["#9aa5b1", "#2a9d8f"])
ax.set_ylabel("Mean CV accuracy (mean_test_score)")
ax.set_ylim(min(scores) - 0.03, 1.0)
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_title("Default vs tuned parameters (from cv_results_)")
fig.tight_layout()
plt.show()
```

The result:

![Bar chart: default vs tuned parameters](img/gridsearch_comparison.png)

On the breast-cancer dataset this yields **0.9736** for the defaults
versus **0.9789** for the best combination (`C=10.0, gamma=0.01`) — a
real but modest gain, which is itself a useful lesson:
tuning helps, and the error bars (`std_test_score`) show how much of
the difference could be fold-to-fold noise.

!!! note "About the truncated y-axis"
    The chart starts its y-axis near the scores to make the difference
    visible. When presenting results, always mention this — a truncated
    axis exaggerates differences, which is fine for inspection but can
    mislead in a report.

## Beyond two bars

Since `cv_results_` holds *every* combination, the same technique
extends to richer views without any retraining: plot all 16
combinations sorted by `mean_test_score`, or pivot `C` against `gamma`
into a heatmap. And when comparing grid search against other tuning
strategies — such as the metaheuristic approach in
[Parameter Optimization](parameter-optimization.md) — the same rule
applies: compare scores obtained under the same cross-validation setup.
