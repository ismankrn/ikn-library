# Feature Selection

`FeatureSelectionProblem` implements wrapper-based feature selection: an
optimizer searches for the feature subset that maximizes a cross-validated
model score while keeping the subset small.

Requires scikit-learn:

```bash
pip install "ikn-library[ml]"
```

## Split first, then select

Feature selection is part of *training*, not a preprocessing step applied
to the whole dataset. The optimizer below evaluates a thousand candidate
subsets and keeps whichever scored best. If it is allowed to see the test
rows while doing that, the winning subset has been chosen partly *for*
those rows, and the final score stops measuring what it claims to.

So the split comes first, and the search only ever sees `X_train`:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("train:", X_train.shape, " test:", X_test.shape)

# One fold object, shuffled with a fixed seed — reused by all 1000 evaluations
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def knn():
    """A fresh scaler + KNN pipeline: the scaler is refitted per fold."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


problem = FeatureSelectionProblem(
    X_train, y_train,          # <- training rows only
    estimator=knn(),
    cv=CV,
    scoring="accuracy",
    alpha=0.99,
)
task = Task(problem=problem, max_evals=1000)
algo = BinaryAntColonyOptimization(population_size=20, evaporation=0.1, seed=42)
best_x, best_fitness = algo.run(task)

selected = problem.selected_features(best_x)
print("Selected features:", selected)
```

Output:

```text
train: (455, 30)  test: (114, 30)
Selected features: [ 6  7  9 11 12 13 16 20 21 23 25 26 28]
```

Thirteen of the thirty columns survive. Two details in that snippet are
hygiene rather than decoration:

- **The scaler lives inside the pipeline.** KNN measures distances, and
  the breast-cancer columns differ by orders of magnitude — `mean area`
  runs into the hundreds while `mean smoothness` sits near 0.1, so
  without scaling a handful of columns decide every neighbour. Putting
  `StandardScaler` in a pipeline means it is refitted on each training
  fold and never sees the fold it is scored on.
- **The folds are explicit and shuffled.** A bare `cv=5` splits in file
  order, which on a dataset sorted by class is a trap; and the same
  thousand evaluations reuse whatever folds you give them, so it is
  worth fixing them deliberately with a seed rather than by accident.

## Before vs after: scoring on the held-out test set

Now fit the same pipeline twice on the training set — once on all 30
columns, once on the 13 selected ones — and score both on the test set:

```python
from sklearn.metrics import accuracy_score

model_all = knn().fit(X_train, y_train)
model_selected = knn().fit(X_train[:, selected], y_train)

acc_all = accuracy_score(y_test, model_all.predict(X_test))
acc_selected = accuracy_score(y_test, model_selected.predict(X_test[:, selected]))

n = len(y_test)
print(f"Before (all {X.shape[1]} features)      : test accuracy = {acc_all:.4f}"
      f"  ({round(acc_all * n)}/{n} correct)")
print(f"After  ({len(selected)} selected features) : test accuracy = {acc_selected:.4f}"
      f"  ({round(acc_selected * n)}/{n} correct)")
```

Output:

```text
Before (all 30 features)      : test accuracy = 0.9561  (109/114 correct)
After  (13 selected features) : test accuracy = 0.9649  (110/114 correct)
```

The comparison as a bar chart, with each bar labelled by its score:

```python
import matplotlib.pyplot as plt

labels = [f"Before\n(all {X.shape[1]} features)",
          f"After\n({len(selected)} selected features)"]
accuracies = [acc_all, acc_selected]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, accuracies, color=["#9aa5b1", "#2a9d8f"])
ax.set_ylabel("Accuracy on the held-out test set")
ax.set_ylim(min(accuracies) - 0.05, 1.0)
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_title("KNN test accuracy before vs after feature selection")
fig.tight_layout()
plt.show()
```

The result:

![Bar chart: test accuracy before vs after feature selection](img/feature_selection_comparison.png)

The comparison itself is fair in the two ways that matter — only the
feature set differs (same pipeline, same training rows, same test rows),
and fewer features can genuinely help a distance-based model, because
every irrelevant column adds its noise to the distance between
neighbours. But look at what the bars are actually worth: **one extra
correct prediction out of 114.** Hold that thought; the last section
puts a number on it.

## What the optimizer saw vs what the test set says

The score the search actually maximized was a 5-fold cross-validation on
the training rows. It is worth printing next to the test score:

```python
from sklearn.model_selection import cross_val_score

cv_all = cross_val_score(knn(), X_train, y_train, cv=CV)
cv_selected = cross_val_score(knn(), X_train[:, selected], y_train, cv=CV)

print(f"train CV, all {X.shape[1]} features      : "
      f"{cv_all.mean():.4f} (std {cv_all.std():.4f})")
print(f"train CV, {len(selected)} selected features : "
      f"{cv_selected.mean():.4f} (std {cv_selected.std():.4f})")
```

Output:

```text
train CV, all 30 features      : 0.9626 (std 0.0112)
train CV, 13 selected features : 0.9802 (std 0.0146)
```

On the folds, selection gains 1.8 accuracy points. On the test set it
gains 0.9. That gap is selection bias: the subset was picked because it
scored best on *those five folds* out of a thousand candidates, so part
of its 0.9802 belongs to the folds rather than to the features.
Reporting 0.9802 as the outcome of feature selection would overstate it;
0.9649 is the number that has not been optimized against.

This is the same three-split discipline described in
[Ensemble Weights](ensemble.md#the-protocol-three-splits) — here the
cross-validation inside the training set plays the role of the
validation split.

## Does the gain survive a different split?

A single train/test split gives one draw of a noisy number, and a gain
of one prediction in 114 is well inside that noise. The way to find out
whether it is real is to repeat the *entire* procedure — split, select,
score — several times and look at the spread. Wrap it in a function and
loop:

```python
import numpy as np


def select_and_score(seed):
    """Split, select and score once, end to end, with everything seeded."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    problem = FeatureSelectionProblem(
        X_tr, y_tr, estimator=knn(),
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="accuracy", alpha=0.99,
    )
    best_x, _ = BinaryAntColonyOptimization(
        population_size=20, evaporation=0.1, seed=seed
    ).run(Task(problem=problem, max_evals=1000))

    sel = problem.selected_features(best_x)
    before = knn().fit(X_tr, y_tr).score(X_te, y_te)
    after = knn().fit(X_tr[:, sel], y_tr).score(X_te[:, sel], y_te)
    return len(sel), before, after


rows = [select_and_score(seed) for seed in range(5)]

print("seed  n_selected   before    after     gain")
for seed, (k, before, after) in enumerate(rows):
    print(f"{seed:>4}  {k:>10}   {before:.4f}   {after:.4f}   {after - before:+.4f}")

gains = np.array([after - before for _, before, after in rows])
print(f"\nmean gain over 5 splits: {gains.mean():+.4f} +/- {gains.std():.4f}")
```

Output:

```text
seed  n_selected   before    after     gain
   0          13   0.9561   0.9561   +0.0000
   1          14   0.9737   0.9649   -0.0088
   2          11   0.9825   0.9561   -0.0263
   3           8   0.9737   0.9649   -0.0088
   4          15   0.9825   0.9912   +0.0088

mean gain over 5 splits: -0.0070 +/- 0.0116
```

Read that honestly: **on this dataset, wrapper feature selection does
not improve KNN.** One split out of five improves, one ties, three get
worse, and the mean gain is slightly negative with a standard deviation
larger than itself. The `+0.0088` from the worked example above is the
best of six draws, not a result. Five runs take about 40 seconds — a
cheap price for not publishing a claim the data does not support.

Nothing was done wrong in the sections above; that is the point. Every
individual step was sound, and a single split still produced a bar chart
that looks like a win. Only the repetition tells you it is not.

What the search *did* find is a model that matches 30 features using 13,
which is a real result of a different kind: a smaller, cheaper, more
interpretable model at no measurable cost in accuracy. Where wrapper
selection earns its keep on accuracy is data with many irrelevant
columns — see [Microarray Data](microarray.md), where the feature count
runs into the thousands and most genes are noise.

!!! tip "Reporting this properly"
    Report the mean and spread over several repetitions, not the best
    single split — and state how many repetitions you ran. If you need
    a stricter estimate, wrap the whole procedure in an outer
    cross-validation loop (nested CV) rather than a handful of random
    splits.

## The fitness function

The fitness (minimized) balances score quality against subset size:

```
alpha * (1 - cv_score) + (1 - alpha) * n_selected / n_features
```

- `alpha` close to 1 prioritizes the model score.
- Lower `alpha` presses harder for smaller subsets. At `alpha=0.99`, as
  used above, dropping half the features is worth only 0.005 of
  fitness — less than one point of accuracy, which is why the winning
  subsets keep 8-15 features rather than 3.
- An empty subset receives the worst possible fitness (1.0).

The `cv_score` here is computed on whatever data was handed to
`FeatureSelectionProblem` — which is why the example passes `X_train`,
not `X`.

This weighted formulation is standard in the wrapper feature-selection
literature — see E. Emary, H. M. Zawbaa, and A. E. Hassanien, "Binary
grey wolf optimization approaches for feature selection,"
*Neurocomputing*, 172, 371–381, 2016; the same form is used in J. Too's
[Wrapper-Feature-Selection-Toolbox](https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox)
and in NiaPy's feature-selection tutorial.

## Notes

- Solutions are vectors in `[0, 1]`; entries above `threshold` (default
  0.5) mark selected features. This means **continuous algorithms** like
  `AntColonyOptimization` can also optimize the problem, not only binary
  ones.
- Any scikit-learn estimator and scoring name works (`"f1"`,
  `"roc_auc"`, regressors with `"r2"`, ...) — including a `Pipeline`,
  which is how preprocessing stays inside the cross-validation.
- `cv` accepts either a number of folds or a splitter object such as
  `StratifiedKFold`; it is passed straight to `cross_val_score`.
- Use `problem.selected_features(best_x)` to get the selected column
  indices and `problem.feature_mask(best_x)` for a boolean mask.
- Index the test set with the same indices, `X_test[:, selected]`, before
  predicting — the fitted model expects the columns in that order.
