# Ensemble Weight Optimization

A Random Forest decides by majority voting: every tree gets an equal
say. The `ikn_library.ensemble` module replaces that with **weighted
voting** — each member gets a weight, and a metaheuristic searches for
the weight vector that maximizes a classification metric.

The scheme:

1. Extract the probability matrix ``P`` of shape
   ``(n_samples, n_members)`` — member ``j``'s class-1 probability for
   each sample.
2. Combine: ``score = P @ w`` with normalized weights ``w`` (so the
   score stays in [0, 1]).
3. Threshold: ``score > 0.5`` predicts class 1.
4. Optimize ``w`` with a continuous metaheuristic. Uniform weights
   recover plain soft majority voting, so that baseline is always
   inside the search space.

No changes to the algorithms are needed — this is just a new `Problem`.

## The protocol: three splits

!!! warning "Optimize weights on data the ensemble never saw"
    If the weights are tuned on the ensemble's own training data, trees
    that memorized that data get large weights — overfitting. Use three
    splits: **train** the forest, **validate** to optimize the weights,
    and report on an untouched **test** set.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ikn_library import Task
from ikn_library.algorithms import AntColonyOptimization
from ikn_library.ensemble import EnsembleWeightProblem, tree_proba_matrix

X, y = make_classification(n_samples=600, n_features=20, n_informative=5,
                           flip_y=0.1, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

forest = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=42)
forest.fit(X_train, y_train)

P_val = tree_proba_matrix(forest, X_val)
P_test = tree_proba_matrix(forest, X_test)

problem = EnsembleWeightProblem(P_val, y_val)   # metric="accuracy" default
task = Task(problem=problem, max_evals=4000)
algo = AntColonyOptimization(population_size=20, archive_size=30, seed=42)
best_x, best_fitness = algo.run(task)

weighted_pred = problem.predict(best_x, P_test)   # labels on the test set
```

Output of the full script (see below) on this synthetic task:

```text
Validation: uniform = 0.7333, optimized = 0.8600
Majority voting (RF default) : test accuracy = 0.7133
Uniform soft voting          : test accuracy = 0.7133
ACO-optimized weights        : test accuracy = 0.8333
```

The optimized weights beat majority voting by 12 points on the test
set — a gain that survived the transfer from validation to test.

## When does re-weighting help?

The forest above is deliberately weak (30 stumps of depth 2). That is
the regime where weighting matters: members are inaccurate but
*diverse*, so promoting the right ones changes decisions. A forest of
large, well-tuned trees is already near its ceiling — on such tasks the
optimizer may improve the validation score without improving the test
score (the gain is noise). Always check that the validation gain
transfers to the test set.

## Ensemble pruning with Binary ACO

Run the **same problem** with
[`BinaryAntColonyOptimization`][ikn_library.algorithms.BinaryAntColonyOptimization]
and the weights become 0/1 — dropping or keeping members. That is
ensemble pruning: a smaller, faster ensemble selected by the same
machinery.

```python
from ikn_library.algorithms import BinaryAntColonyOptimization

algo = BinaryAntColonyOptimization(population_size=15, seed=42)
best_bits, _ = algo.run(Task(problem=problem, max_evals=2000))
kept = int(best_bits.sum())   # members kept out of 30
```

## Multi-feature ensembles (GA-WE style)

The weighted-ensemble scheme is not limited to the trees of one forest.
A powerful variant from the bioinformatics literature — the **GA-WE**
method of Li et al. (2016), which this module's design follows — trains
**one classifier per feature representation** of the same data and lets
the metaheuristic weigh the representations. In their piRNA-prediction
study, 23 Random Forests (one per sequence feature: k-mer spectrum,
mismatch profiles, pseudo-nucleotide compositions, PSSM, ...) were
combined with weights optimized by a genetic algorithm against the
**validation AUC**, outperforming every individual representation.

When every member reads the **same** feature matrix (a heterogeneous
SVM + KNN + RF ensemble, say), pass the list of fitted classifiers to
`tree_proba_matrix` as usual. When each member has its **own** feature
representation — the GA-WE setting — build the probability matrix
explicitly, one column per representation-specific classifier (each
fitted on the train split, predicting on the validation split):

```python
import numpy as np

P_val = np.column_stack([
    clf_kmers.predict_proba(X_val_kmers)[:, 1],
    clf_pssm.predict_proba(X_val_pssm)[:, 1],
    clf_composition.predict_proba(X_val_composition)[:, 1],
])
problem = EnsembleWeightProblem(P_val, y_val, metric="auc")
```

Following the paper, use `metric="auc"`: it is **threshold-free**
(computed on the combined scores before the 0.5 cut-off), a smoother
search signal than accuracy, and the safer choice on imbalanced data.
To reproduce the paper's setup faithfully, optimize the weights with
[`GeneticAlgorithm`][ikn_library.algorithms.GeneticAlgorithm] — the
algorithm family GA-WE itself uses.

## Notes

- `tree_proba_matrix` accepts any fitted scikit-learn ensemble exposing
  `estimators_`, or a plain list of fitted classifiers — heterogeneous
  ensembles (SVM + KNN + RF) work the same way.
- `metric` accepts `"accuracy"`, `"f1"`, `"auc"`, or any callable
  `f(y_true, y_pred)` where higher is better; the fitness minimized is
  `1 - metric`. The `"auc"` metric is computed on the continuous
  combined scores and ignores `threshold`; callables receive
  thresholded 0/1 predictions.
- Currently binary classification (labels 0/1); the positive class is
  each member's `classes_[1]`.

## Reference

The weighted-ensemble scheme and its train/validation/test protocol
follow: D. Li, L. Luo, W. Zhang, F. Liu, and F. Luo, "A genetic
algorithm-based weighted ensemble method for predicting
transposon-derived piRNAs," *BMC Bioinformatics*, 17:329, 2016.
[doi:10.1186/s12859-016-1206-3](https://doi.org/10.1186/s12859-016-1206-3).

A complete runnable script is available at
[`examples/ensemble_weight_optimization.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/ensemble_weight_optimization.py).
