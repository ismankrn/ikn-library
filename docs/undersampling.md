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

```python
from sklearn.neighbors import KNeighborsClassifier

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.sampling import UndersamplingProblem

problem = UndersamplingProblem(
    X_train, y_train, X_val, y_val,
    estimator=KNeighborsClassifier(n_neighbors=5),
    target_ratio=1.0,        # balance the classes exactly
    metric="f1",
)
task = Task(problem=problem, max_evals=3000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)

X_reduced, y_reduced = problem.resampled_data(best_x)   # train your final model
kept_rows = problem.selected_indices(best_x)            # row indices into X_train
```

On a synthetic dataset (3,000 samples, 15% minority) whose majority
class is polluted with mislabeled samples — the regime where the
*choice* of discarded samples matters most — the full pipeline
(`examples/undersampling.py`) gives:

```text
Train class counts: minority=169, majority=1331
No undersampling (imbalanced) : test F1 = 0.5390
Random undersampling (mean/5) : test F1 = 0.6012
Optimized undersampling       : test F1 = 0.6122
Reduced training set: 338 samples (from 1500)
```

The optimizer beats random undersampling by learning to discard the
mislabeled majority samples, and the training set shrinks to a quarter
of its size.

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
- Any estimator with `fit`/`predict` works; scikit-learn is only
  required for the default KNN.
- Binary classification only; the minority/majority classes are
  detected automatically from `y_train`.

## Reference

S. Garcia and F. Herrera, "Evolutionary undersampling for
classification with imbalanced datasets: proposals and taxonomy,"
*Evolutionary Computation*, 17(3), 275-306, 2009.
[doi:10.1162/evco.2009.17.3.275](https://doi.org/10.1162/evco.2009.17.3.275).
