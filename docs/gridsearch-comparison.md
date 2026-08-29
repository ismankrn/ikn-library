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

## Reading the comparison from `cv_results_`

`cv_results_` is a dict of arrays with one entry per combination; as a
DataFrame each row holds the combination (`params`) and its
cross-validated performance (`mean_test_score`, `std_test_score`).
No model is retrained below — everything is a lookup:

```python
results = pd.DataFrame(grid.cv_results_)

default_params = {"C": 1.0, "gamma": "scale"}
default_row = results.loc[
    results["params"].apply(lambda p: p == default_params)
].iloc[0]
best_row = results.iloc[grid.best_index_]

print("default:", default_params, "->", default_row["mean_test_score"])
print("best   :", grid.best_params_, "->", best_row["mean_test_score"])
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

On the breast-cancer dataset this yields roughly **0.9736** for the
defaults versus **0.9789** for the best combination (`C=10.0,
gamma=0.01`) — a real but modest gain, which is itself a useful lesson:
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
