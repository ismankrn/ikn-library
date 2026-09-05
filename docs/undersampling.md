# Undersampling

When one class vastly outnumbers the other (say A: 100, B: 1000), models
drown in the majority class. *Random* undersampling fixes the ratio by
discarding majority samples blindly — but **which** samples are
discarded matters: noisy and redundant ones deserve to go, informative
ones near the decision boundary deserve to stay.

`ikn_library.sampling` turns that choice into an optimization problem
(*evolutionary undersampling*, Garcia & Herrera, 2009): a bit string
over the majority-class training samples marks which ones to keep, the
minority class is always kept in full, and the fitness is the
performance of a model trained on the reduced set.

## The constraint: repair to an exact ratio

Every candidate is **repaired** so that exactly
``target = round(target_ratio * n_minority)`` majority samples stay
selected — excess bits are switched off (or missing ones on) at random,
deterministically per candidate. The evaluation budget is therefore
only ever spent on subsets with the desired class ratio;
``target_ratio=1.0`` balances the classes exactly, ``1.5`` keeps a
150:100 ratio, and so on.

## The protocol: undersample the training data only

!!! warning "Never undersample the evaluation data"
    The fitness model is trained on the reduced training set but scored
    on an untouched — still imbalanced — **validation set**. Scoring on
    an artificially balanced set would make every subset look better
    than it is in the real class distribution. As always, report final
    numbers on a third, untouched test set.

Accuracy is misleading under imbalance (predicting "all majority" on a
100:1000 dataset is already 91% accurate), so the default metric is
**F1 with the minority class as positive**; `"balanced_accuracy"`,
`"accuracy"`, and custom callables are also available.

## Example

The snippet below runs end to end. It starts where every other page in
these docs starts — by carving out the splits — because undersampling
is a *training-set* operation and the protocol only holds if the
validation and test rows are separated first:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.sampling import UndersamplingProblem

# An imbalanced dataset whose majority class carries label noise: a quarter
# of the minority samples are mislabeled as majority. This is the regime
# where *which* samples you discard matters most.
X, y = make_classification(n_samples=3000, n_features=10, n_informative=4,
                           weights=[0.85, 0.15], flip_y=0.0, random_state=0)
rng = np.random.default_rng(0)
minority_idx = np.flatnonzero(y == 1)
mislabeled = rng.choice(minority_idx, int(0.25 * len(minority_idx)), replace=False)
y = y.copy()
y[mislabeled] = 0

# Three splits, before anything else: the search is scored on validation,
# and the test rows are not touched until the last block on this page
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

print(f"Train class counts: minority={np.sum(y_train == 1)}, "
      f"majority={np.sum(y_train == 0)}")


def knn():
    """A fresh scaler + KNN pipeline: the scaler is refitted per subset."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


problem = UndersamplingProblem(
    X_train, y_train, X_val, y_val,
    estimator=knn(),
    target_ratio=1.0,        # balance the classes exactly
    metric="f1",
)
task = Task(problem=problem, max_evals=3000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)

X_reduced, y_reduced = problem.resampled_data(best_x)   # train your final model
kept_rows = problem.selected_indices(best_x)            # row indices into X_train
```

Output:

```text
Train class counts: minority=169, majority=1331
```

The estimator is a `Pipeline` rather than a bare `KNeighborsClassifier`
for the usual reason — KNN measures distances, so unscaled columns with
larger ranges decide every neighbour — and for one specific to this
problem: each candidate trains on a *different* subset of the majority
class, so each one has its own means and variances. Putting the scaler
inside the estimator means it is refitted on whatever subset is being
evaluated, instead of carrying statistics from a set that candidate
never saw.

### Does it beat discarding at random?

The question the whole method exists to answer, settled on the test
split that nothing above has touched:

```python
def test_f1(X_fit, y_fit):
    return f1_score(y_test, knn().fit(X_fit, y_fit).predict(X_test))


# five random subsets with the same class ratio, for comparison
random_scores = [
    test_f1(*problem.resampled_data(
        np.random.default_rng(7 + i).random(problem.dimension)))
    for i in range(5)
]

print(f"No undersampling (imbalanced) : test F1 = {test_f1(X_train, y_train):.4f}")
print(f"Random undersampling (mean/5) : test F1 = {np.mean(random_scores):.4f}")
print(f"Optimized undersampling       : test F1 = {test_f1(X_reduced, y_reduced):.4f}")
print(f"Reduced training set: {len(kept_rows)} samples (from {len(X_train)})")
```

Output:

```text
No undersampling (imbalanced) : test F1 = 0.5116
Random undersampling (mean/5) : test F1 = 0.5581
Optimized undersampling       : test F1 = 0.5902
Reduced training set: 338 samples (from 1500)
```

Both comparisons matter. Undersampling at all is worth 4.7 F1 points
over the imbalanced training set; *choosing* which majority samples to
discard is worth another 3.2 on top of discarding them at random — and
the training set ends up at 338 samples, under a quarter of its original
size. The random baseline is averaged over five draws rather than taken
once, because a single random subset would be one sample of a noisy
quantity, and beating one lucky or unlucky draw proves nothing.

The same code is available as a script at
[`examples/undersampling.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/undersampling.py).

!!! note "Gains vary by dataset"
    Undersampling of any kind helps most when the majority class is
    noisy or heavily redundant. On clean, well-separated data the
    ordering can flip — sometimes not undersampling at all is best.
    That is exactly why the protocol reports on a held-out test set:
    verify the gain, don't assume it.

## Notes

- The optimizer sees a standard binary problem, so
  `BinaryAntColonyOptimization` works out of the box — and continuous
  algorithms can optimize it too (entries above `threshold` count as
  "keep").
- Any estimator with `fit`/`predict` works — including a `Pipeline`,
  which is how preprocessing stays inside each candidate's fit. The
  estimator is cloned per evaluation, so one instance can be passed and
  reused safely. scikit-learn is only required for the default KNN.
- Binary classification only; the minority/majority classes are
  detected automatically from `y_train`.

## Reference

S. Garcia and F. Herrera, "Evolutionary undersampling for
classification with imbalanced datasets: proposals and taxonomy,"
*Evolutionary Computation*, 17(3), 275-306, 2009.
[doi:10.1162/evco.2009.17.3.275](https://doi.org/10.1162/evco.2009.17.3.275).
