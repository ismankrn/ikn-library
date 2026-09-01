# Hyperparameter Tuning

Tuning is not finished when `GridSearchCV` reports a best score. This
page walks the whole workflow on one example: **search** a parameter
grid, **compare** the tuned result against the defaults honestly, then
**use** the resulting model — evaluate it on held-out data, save it to
disk, load it back, and predict.

## Setting up the search

Two decisions here matter more than the grid itself.

**Hold out a test set before tuning.** The cross-validated score
inside `GridSearchCV` is used to *choose* parameters, so it is no
longer an unbiased estimate of how the chosen model performs. Keeping a
test set aside gives you a number that the search never saw.

**Put preprocessing inside a `Pipeline`.** Scaling the full dataset
before splitting leaks test statistics into training. A pipeline scales
within each CV fold, and — as the last section shows — it also means the
saved model carries its own preprocessing.

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("svc", SVC(kernel="rbf"))])

param_grid = {
    "svc__C": [0.1, 1.0, 10.0, 100.0],        # 1.0     = sklearn default
    "svc__gamma": ["scale", 0.01, 0.1, 1.0],  # "scale" = sklearn default
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("train:", X_train.shape, " test:", X_test.shape)
```

Output:

```text
train: (455, 30)  test: (114, 30)
```

The `svc__` prefix is how a pipeline addresses a step's parameters:
`<step name>__<parameter>`.

## Comparing against the defaults

A common mistake is to *re-run* the model with default parameters just
to compare against the tuned result. That re-run is unnecessary — and
worse, a fresh split would make the two numbers incomparable. The
search has already evaluated every combination in the grid on the same
CV folds, and stored every score in `cv_results_`.

!!! warning "Include the default values in the grid"
    The comparison only works if the estimator's **default parameter
    combination is inside the searched grid**. If the defaults are not
    among the evaluated combinations, `cv_results_` has no score for
    them and you would have to evaluate them separately — losing the
    guarantee that both scores came from the exact same folds.

That is why `1.0` and `"scale"` appear in the grid above.

### What `cv_results_` looks like

`cv_results_` is a dict of arrays with one entry per combination; as a
DataFrame each row holds the combination (`params`, plus one
`param_<name>` column per parameter) and its cross-validated
performance. With a 4x4 grid there are 16 rows:

```python
results = pd.DataFrame(grid.cv_results_)
cols = ["param_svc__C", "param_svc__gamma", "mean_test_score",
        "std_test_score", "rank_test_score"]
print(results[cols].sort_values("rank_test_score").head(6).to_string(index=False))
```

Output:

```text
 param_svc__C param_svc__gamma  mean_test_score  std_test_score  rank_test_score
         10.0             0.01         0.980220        0.016150                1
          1.0            scale         0.971429        0.017855                2
         10.0            scale         0.971429        0.017855                2
          1.0             0.01         0.969231        0.023466                4
        100.0             0.01         0.969231        0.016150                4
          1.0              0.1         0.960440        0.016447                6
```

How to read it:

- **`mean_test_score`** — the score averaged over the 5 CV folds; this
  is the number we compare.
- **`std_test_score`** — the spread across folds; larger values mean
  the score is less certain.
- **`rank_test_score`** — 1 is the winner (`grid.best_params_`). Note
  where the default combination (`C=1.0, gamma="scale"`) sits: **rank 2
  of 16**, tied with another combination. Already decent, but not the
  best — and that is the whole comparison, visible in the table.

### Reading the comparison

No model is retrained below — everything is a lookup:

```python
default_params = {"svc__C": 1.0, "svc__gamma": "scale"}
default_row = results.loc[
    results["params"].apply(lambda p: p == default_params)
].iloc[0]
best_row = results.iloc[grid.best_index_]

print("default:", default_params, "->", default_row["mean_test_score"])
print("best   :", grid.best_params_, "->", best_row["mean_test_score"])
```

Output:

```text
default: {'svc__C': 1.0, 'svc__gamma': 'scale'} -> 0.9714285714285715
best   : {'svc__C': 10.0, 'svc__gamma': 0.01} -> 0.9802197802197803
```

Two details worth pointing out to students:

- `grid.best_score_` is by definition the maximum `mean_test_score` in
  `cv_results_` — the "best" row is not special, it is just the row
  that happened to win.
- Because every combination was scored on the **same CV folds**, the
  two numbers are directly comparable. That is exactly what a manual
  re-run with a fresh split would *not* guarantee.

### The bar chart

```python
import matplotlib.pyplot as plt

labels = ["Default\n(C=1.0, gamma='scale')",
          f"Best\n(C={grid.best_params_['svc__C']}, "
          f"gamma={grid.best_params_['svc__gamma']})"]
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

Tuning moved the cross-validated accuracy from **0.9714** to
**0.9802** — a real but modest gain, which is itself a useful lesson.
The error bars (`std_test_score`) are about 0.016, larger than the gap
between the bars, so some of that difference could be fold-to-fold
noise rather than a genuinely better model.

!!! note "About the truncated y-axis"
    The chart starts its y-axis near the scores to make the difference
    visible. When presenting results, always mention this — a truncated
    axis exaggerates differences, which is fine for inspection but can
    mislead in a report.

## Using the tuned model

### `best_estimator_` is already trained

With the default `refit=True`, `GridSearchCV` refits the winning
configuration on the **whole training set** once the search finishes.
So there is no need to rebuild anything by hand — assign it to a
variable and it is ready to predict:

```python
model = grid.best_estimator_
print(model)
```

Output:

```text
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC(C=10.0, gamma=0.01))])
```

Note what came back: the entire **pipeline**, not just the `SVC`. It
carries its own `StandardScaler`, already fitted on the training data,
so it takes raw features and handles the scaling itself.

### Evaluating on the held-out test set

This is the number to report, because the search never saw these rows:

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred,
                            target_names=load_breast_cancer().target_names))
```

Output:

```text
accuracy: 0.9825

              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

The test accuracy (0.9825) is slightly *higher* than the
cross-validated score (0.9802). That is normal variation on 114 test
samples, not evidence that the model improved — with a test set this
small, a single sample changes accuracy by almost a percentage point.

### Saving the model with `pickle`

Training is the expensive part; you do not want to repeat it every time
the model is used. Pickle serialises the fitted object to a file:

```python
import pickle
from pathlib import Path

out = Path("svc_breast_cancer.pkl")
with open(out, "wb") as f:
    pickle.dump(model, f)

print("saved:", out, f"({out.stat().st_size} bytes)")
```

Output:

```text
saved: svc_breast_cancer.pkl (15850 bytes)
```

Because `model` is the whole pipeline, that one file contains the
scaler's learned mean and variance *and* the trained SVC. Saving only
`model.named_steps["svc"]` would be a common and painful mistake — the
loaded classifier would then expect scaled input with no way to
reproduce the scaling.

### Loading it back and predicting

```python
with open("svc_breast_cancer.pkl", "rb") as f:
    loaded = pickle.load(f)

print("loaded:", type(loaded).__name__)
y_pred_loaded = loaded.predict(X_test)
print("identical predictions:", (y_pred_loaded == y_pred).all())
print("accuracy from loaded model:", round(accuracy_score(y_test, y_pred_loaded), 4))
```

Output:

```text
loaded: Pipeline
identical predictions: True
accuracy from loaded model: 0.9825
```

The loaded object behaves exactly like the original — same class, same
predictions. It can be used on genuinely new data the same way:

```python
new_samples = X_test[:5]          # stands in for unseen data
print("predicted:", loaded.predict(new_samples))
print("actual   :", y_test[:5])
```

Output:

```text
predicted: [0 1 0 1 0]
actual   : [0 1 0 1 0]
```

!!! warning "Two things to know about pickle"
    **Only unpickle files you trust.** Loading a pickle can execute
    arbitrary code, so a `.pkl` from an untrusted source is as dangerous
    as running an unknown script.

    **Pickles are not portable across versions.** A model saved under
    one scikit-learn version may fail to load, or load with subtly
    different behaviour, under another. Record the versions alongside
    the file — these results were produced with **scikit-learn 1.9.0**
    on **Python 3.13** — and pin them when the model matters.

    For long-lived models, `joblib.dump` / `joblib.load` is the
    scikit-learn recommendation: same interface, more efficient with the
    large NumPy arrays that fitted estimators contain.

## Beyond two bars

Since `cv_results_` holds *every* combination, the same technique
extends to richer views without any retraining: plot all 16
combinations sorted by `mean_test_score`, or pivot `C` against `gamma`
into a heatmap. And when comparing grid search against other tuning
strategies — such as the metaheuristic approach in
[Hyperparameter Optimization](parameter-optimization.md) — the same rule
applies: compare scores obtained under the same cross-validation setup.
