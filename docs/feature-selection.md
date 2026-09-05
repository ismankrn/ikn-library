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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ikn_library import Task
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("train:", X_train.shape, " test:", X_test.shape)

problem = FeatureSelectionProblem(
    X_train, y_train,          # <- training rows only
    estimator=KNeighborsClassifier(n_neighbors=5),
    cv=5,
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
Selected features: [ 1  2 21 22]
```

Four of the thirty columns survive — mean texture, mean perimeter, worst
texture, worst perimeter. The 114 test rows have not been touched by
anything yet.

## Before vs after: scoring on the held-out test set

Now fit the same model twice on the training set — once on all 30
columns, once on the 4 selected ones — and score both on the test set:

```python
from sklearn.metrics import accuracy_score

model_all = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
model_selected = KNeighborsClassifier(n_neighbors=5).fit(X_train[:, selected], y_train)

acc_all = accuracy_score(y_test, model_all.predict(X_test))
acc_selected = accuracy_score(y_test, model_selected.predict(X_test[:, selected]))

n = len(y_test)
print(f"Before (all {X.shape[1]} features)     : test accuracy = {acc_all:.4f}"
      f"  ({round(acc_all * n)}/{n} correct)")
print(f"After  ({len(selected)} selected features) : test accuracy = {acc_selected:.4f}"
      f"  ({round(acc_selected * n)}/{n} correct)")
```

Output:

```text
Before (all 30 features)     : test accuracy = 0.9123  (104/114 correct)
After  (4 selected features) : test accuracy = 0.9298  (106/114 correct)
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

The subset wins on both axes — higher accuracy on 4 of the 30 columns —
and the comparison is fair in the two ways that matter:

- **Only the feature set differs**: same estimator, same training rows,
  same test rows.
- **Fewer features can genuinely help**: distance-based models like KNN
  are hurt by irrelevant dimensions, because every extra column adds its
  noise to the distance between neighbours. Removing them often improves
  the score rather than merely matching it.

!!! note "How large is this gain, really?"
    Two extra correct predictions out of 114 is well inside the noise of
    a single train/test split. Repeat the whole procedure — split,
    select, score — with several `random_state` values before claiming
    one subset beats another. Holding out a test set buys an *unbiased*
    number, not a precise one.

## What the optimizer saw vs what the test set says

The score the search actually maximized was a 5-fold cross-validation on
the training rows. It is worth printing next to the test score:

```python
from sklearn.model_selection import cross_val_score

cv_all = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5)
cv_selected = cross_val_score(
    KNeighborsClassifier(n_neighbors=5), X_train[:, selected], y_train, cv=5
)

print(f"train CV, all {X.shape[1]} features      : "
      f"{cv_all.mean():.4f} (std {cv_all.std():.4f})")
print(f"train CV, {len(selected)} selected features : "
      f"{cv_selected.mean():.4f} (std {cv_selected.std():.4f})")
```

Output:

```text
train CV, all 30 features      : 0.9363 (std 0.0189)
train CV, 4 selected features : 0.9560 (std 0.0184)
```

Both CV numbers sit about two and a half points above the corresponding
test numbers, and the selected subset falls slightly further (0.9560 →
0.9298) than the full set does (0.9363 → 0.9123). That is what selection
bias looks like: the subset was picked because it scored best on *those
folds* out of a thousand candidates, so part of its 0.9560 belongs to the
folds rather than to the features. Reporting 0.9560 as the outcome of
feature selection would overstate it; 0.9298 is the number that has not
been optimized against.

This is the same three-split discipline described in
[Ensemble Weights](ensemble.md#the-protocol-three-splits) — here the
cross-validation inside the training set plays the role of the
validation split.

## The fitness function

The fitness (minimized) balances score quality against subset size:

```
alpha * (1 - cv_score) + (1 - alpha) * n_selected / n_features
```

- `alpha` close to 1 prioritizes the model score.
- Lower `alpha` presses harder for smaller subsets.
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
  `"roc_auc"`, regressors with `"r2"`, ...).
- Use `problem.selected_features(best_x)` to get the selected column
  indices and `problem.feature_mask(best_x)` for a boolean mask.
- Index the test set with the same indices, `X_test[:, selected]`, before
  predicting — the fitted model expects the columns in that order.
