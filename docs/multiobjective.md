# Multi-Objective Optimization

Most real problems pull in two directions at once. Feature selection
wants high accuracy *and* few features; a drug candidate should be
potent *and* non-toxic; a diagnostic test needs sensitivity *and*
specificity. `ikn_library.multiobjective` optimizes such goals
**without collapsing them into one number**, returning the whole
trade-off curve — the **Pareto front** — in a single run.

## The problem with weighted sums

[`FeatureSelectionProblem`](feature-selection.md) folds two goals into
one fitness:

```
alpha * (1 - accuracy) + (1 - alpha) * n_selected / n_features
```

That forces you to pick `alpha` *before* the search, even though no
principled value exists — and each choice yields a different answer, so
you end up re-running the whole optimization to explore the trade-off.
On the breast-cancer training split, `alpha` = 0.99, 0.95, 0.90 and
0.80 returned subsets of 13, 9, 4 and 4 features in four separate runs —
and the two four-feature answers do not even agree on which four.

## Pareto dominance

Solution **A dominates B** when A is no worse in every objective and
strictly better in at least one. The **Pareto front** is the set of
solutions no other solution dominates.

- 5 features / 95.1% vs 4 features / 94.7% → *neither dominates*; one
  is more accurate, the other smaller. Both are legitimate answers.
- 4 features / 95.2% would dominate 5 features / 95.1% — fewer
  features *and* more accurate, so the latter becomes irrelevant.

The front is exactly the menu a domain expert needs: pick the point
that fits the clinical or budgetary context, after seeing what each
subset size actually buys.

## NSGA-II

[`NSGA2`][ikn_library.algorithms.NSGA2] (Deb et al., 2002) finds that
whole front in one run using three mechanisms that replace
single-objective ranking:

1. **Non-dominated sorting** ranks the population into fronts: front 0
   is non-dominated, front 1 is dominated only by front 0, and so on.
2. **Crowding distance** breaks ties *within* a front, favouring
   solutions in sparsely populated regions so the front spreads out
   instead of clustering at one end.
3. **Elitist replacement** merges parents and offspring and refills the
   next generation front by front, so no Pareto solution is ever lost.

## Usage

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library.multiobjective import (
    MultiObjectiveFeatureSelection, MultiObjectiveTask,
)
from ikn_library.algorithms import NSGA2

# The scaler goes inside the estimator so it is refitted per fold, and the
# folds are shuffled with a fixed seed rather than following row order
estimator = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
problem = MultiObjectiveFeatureSelection(
    X, y, estimator=estimator, cv=StratifiedKFold(5, shuffle=True, random_state=42))
task = MultiObjectiveTask(problem=problem, max_evals=4000)
solutions, objectives = NSGA2(population_size=40, seed=42).run(task)

for solution, (error, fraction) in zip(solutions, objectives):
    features = problem.selected_features(solution)
    print(len(features), 1 - error, list(features))
```

Note what `run()` returns here: **two arrays**, not one best solution.
`objectives[i]` holds the objective values of `solutions[i]`, sorted by
the first objective.

## Example: the whole trade-off curve at once

One NSGA-II run on a synthetic dataset (400 samples, 30 features, 12 of
them informative) maps the whole curve:

!!! warning "Run the selection on training data only"
    Feature selection is part of model building, so it must not see the
    test set. Split first, optimize on the training split, and keep the
    test split untouched for the final number — otherwise the reported
    accuracy is optimistic. Everything below is built on `X_train`.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library.algorithms import NSGA2
from ikn_library.multiobjective import (
    MultiObjectiveFeatureSelection, MultiObjectiveTask,
)


def knn():
    """A fresh scaler + KNN pipeline: the scaler is refitted per fold."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


CV = StratifiedKFold(5, shuffle=True, random_state=42)

X, y = make_classification(n_samples=400, n_features=30, n_informative=12,
                           n_redundant=5, flip_y=0.03, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

problem = MultiObjectiveFeatureSelection(X_train, y_train, estimator=knn(), cv=CV)
solutions, objectives = NSGA2(population_size=40, seed=42).run(
    MultiObjectiveTask(problem=problem, max_evals=3000))

n_features = np.array([len(problem.selected_features(s)) for s in solutions])
cv_accuracy = 1.0 - objectives[:, 0]
order = np.argsort(n_features)
solutions, n_features, cv_accuracy = (solutions[order], n_features[order],
                                      cv_accuracy[order])

print("n_features | CV accuracy")
print("-----------|------------")
for k, accuracy in zip(n_features, cv_accuracy):
    print(f"{k:>7}    |   {accuracy:.4f}")
```

Output:

```text
n_features | CV accuracy
-----------|------------
      3    |   0.6536
      4    |   0.7321
      5    |   0.7643
      6    |   0.8250
      7    |   0.8393
      8    |   0.8607
      9    |   0.8786
     10    |   0.8821
```

As a picture:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(n_features, cv_accuracy, "o-", color="#2a9d8f",
        label="Pareto front (one NSGA-II run)")
for k, accuracy in zip(n_features, cv_accuracy):
    ax.annotate(f"{accuracy:.3f}", (k, accuracy), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)
ax.set_xlabel("Number of features selected")
ax.set_ylabel("Cross-validated accuracy")
ax.set_title("Accuracy vs subset size: the whole trade-off curve")
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()
```

![Pareto front: accuracy versus number of features](img/pareto_front.png)

The curve makes the decision obvious in a way no single `alpha` could:
accuracy climbs steeply to about 6 features, then bends — the four
features after that buy 5.7 points together, and the last one buys 0.35.
A clinician paying per assay can see exactly where to stop.

!!! tip "A smaller subset can be just as good"
    On the breast-cancer training split, NSGA-II returned a **6-feature**
    subset scoring 0.9780 and a 12-feature subset scoring 0.9802. The
    weighted-sum version needed **13** features to reach that same
    0.9802 at `alpha=0.99` — so the front contains an answer that is
    half the size for 0.2 points, and the scalarized run never surfaced
    it.

## Using the selected features to train a model

A Pareto front is a menu, not an answer. Two steps turn the front built
in the previous section into a working model.

**Step 1 — pick a point on the front.** Common criteria:

```python
most_accurate = int(np.argmax(cv_accuracy))     # best score, size ignored
smallest = int(np.argmin(n_features))           # fewest features
# or: the smallest subset meeting a required accuracy
budget = np.flatnonzero(cv_accuracy >= 0.85)
cheapest_good_enough = budget[np.argmin(n_features[budget])]
```

A fourth criterion often mentioned is the **knee point** — the front
position furthest from the straight line joining its two ends, i.e.
where the curve stops paying off:

```python
def knee_index(sizes, scores):
    """The point furthest from the line joining the front's two ends."""
    points = np.column_stack([sizes / sizes.max(), scores])   # normalize axes
    start, end = points[0], points[-1]
    line = (end - start) / np.linalg.norm(end - start)
    offsets = points - start
    projections = np.outer(offsets @ line, line)
    return int(np.argmax(np.linalg.norm(offsets - projections, axis=1)))
```

(Read the caveat below before trusting it.)

**Step 2 — extract the features and fit the final model.** The solution
vector is a mask, so `selected_features` gives the column indices:

```python
from sklearn.metrics import accuracy_score

features = problem.selected_features(solutions[most_accurate])

model = knn().fit(X_train[:, features], y_train)
accuracy = accuracy_score(y_test, model.predict(X_test[:, features]))
```

Note that **the same column indices must be applied to any future
data** — store `features` alongside the model.

### What it gives you

Comparing the three candidates against using every feature — the front
is sorted by subset size, so `smallest` is simply index 0:

```python
candidates = {
    "most accurate": int(np.argmax(cv_accuracy)),
    "knee point": knee_index(n_features, cv_accuracy),
    "smallest": 0,
}

baseline = accuracy_score(y_test, knn().fit(X_train, y_train).predict(X_test))
print(f"all {X.shape[1]} features            : test accuracy = {baseline:.4f}")

for label, index in candidates.items():
    features = problem.selected_features(solutions[index])
    model = knn().fit(X_train[:, features], y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test[:, features]))
    print(f"{label:<14} ({len(features):>2} features): "
          f"test accuracy = {accuracy:.4f}")
```

Output:

```text
all 30 features            : test accuracy = 0.7583
most accurate  (10 features): test accuracy = 0.8083
knee point     ( 6 features): test accuracy = 0.7500
smallest       ( 3 features): test accuracy = 0.6917
```

The complete runnable script is
[`examples/multiobjective_feature_selection.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/multiobjective_feature_selection.py).

The 10-feature subset **beats using all 30** — five points of test
accuracy from a third of the inputs, which is the payoff feature
selection promises when most of the columns are noise.

!!! warning "The knee point is a heuristic, not a rule"
    A popular shortcut is to take the *knee* — the point of maximum
    curvature, where accuracy stops climbing steeply. Here it picked
    6 features and scored **0.7500 on the test set**: below the
    10-feature subset, and below using all 30.

    The knee is a property of the *shape* of the training-CV curve, and
    that shape does not have to reflect generalization. Treat the front
    as a shortlist: check the candidates you care about on held-out
    data before committing.

## Writing your own multi-objective problem

Subclass `MultiObjectiveProblem` and return a vector; **every objective
is minimized**, so maximize accuracy by returning `1 - accuracy`:

```python
import numpy as np
from ikn_library.multiobjective import MultiObjectiveProblem

class PotencyAndToxicity(MultiObjectiveProblem):
    def __init__(self, dimension):
        super().__init__(dimension, n_objectives=2, lower=0.0, upper=1.0,
                         objective_names=["1 - potency", "toxicity"])

    def _evaluate(self, x):
        return np.array([1.0 - potency(x), toxicity(x)])
```

## Making another algorithm multi-objective

Non-dominated sorting and crowding distance are not GA-specific — they
are a *ranking rule*, and any algorithm that ranks its population can
adopt it. That is how MO-ABC, MO-PSO and friends are built in the
literature, and the same recipe works here.

The recipe applies to **population-based** algorithms — the ones that
rank a set of candidates each iteration (KMA, ABC, Bees, CSO, FSS, ...).
A single-solution method such as Simulated Annealing has no population
to rank and can only occupy one point at a time, so it cannot map out a
front; use a population-based algorithm when you need one.

!!! warning "A single-objective algorithm cannot just be pointed at a MultiObjectiveTask"
    It will fail immediately:

    ```python
    task = MultiObjectiveTask(problem=TwoObjectiveProblem(), max_evals=500)
    KomodoMlipirAlgorithm(seed=0).run(task)
    # AttributeError: 'MultiObjectiveTask' object has no attribute 'best_fitness'
    ```

    Single-objective algorithms rely on things that do not exist when
    there are several objectives: a single `task.best_fitness`, a single
    `task.best_x`, and comparisons like `if new < old`. With a vector of
    objectives, `<` is ambiguous — that is precisely the problem Pareto
    dominance solves.

### What has to change

Exactly three things, and they are all mechanical:

| Single-objective | Multi-objective replacement |
|---|---|
| `np.argsort(fitness)` | `pareto_sort_indices(objectives)` |
| `if new_fitness < old_fitness` | `if dominates(new_objectives, old_objectives)` |
| `task.best_x` (the single best) | the top-ranked solution, `population[0]` after sorting |

`pareto_sort_indices` is the drop-in replacement for `argsort`: it
orders solutions by Pareto front first and, within a front, by
decreasing crowding distance.

```python
from ikn_library.multiobjective import pareto_sort_indices, dominates

order = pareto_sort_indices(objectives)   # best-to-worst, like argsort
population, objectives = population[order], objectives[order]
```

### Worked example: MO-KMA

The [Komodo Mlipir Algorithm](algorithm-details/kma.md) splits its
population into big males, one female, and small males **by rank** — so
swapping the ranking rule is enough to make the whole algorithm
multi-objective. Its movement operators are untouched:

```python
class MOKomodoMlipir(KomodoMlipirAlgorithm):
    def init_population(self, task):
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension))
        objectives = np.array([task.eval(x) for x in population])
        order = pareto_sort_indices(objectives)          # <- the only change
        return population[order], objectives[order], []

    def run_iteration(self, task, state):
        population, objectives, _ = state
        n_big, n_small = self._group_sizes(len(population))
        # ... the same three groups, but "better" now means "dominates"
        # and the winner is population[0] rather than argmin(fitness)
        order = pareto_sort_indices(objectives)
        return population[order], objectives[order], []
```

The full, runnable class is in
[`examples/mo_kma.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/mo_kma.py).
Running it against NSGA-II on ZDT1:

```text
MO-KMA  : 292 solutions | mean distance to the true front = 0.1424 | f1 coverage [0.00, 1.00]
NSGA-II : 259 solutions | mean distance to the true front = 0.0012 | f1 coverage [0.00, 1.00]
```

MO-KMA works — it produces a genuine, fully spread Pareto front — but
its front sits further from the true one than NSGA-II's. That is the
honest starting point, not a finished result: NSGA-II's operators were
designed around Pareto ranking, while KMA's were not, so a new
multi-objective variant normally needs its operators and parameters
re-tuned for the multi-objective setting. **That gap is exactly where
the research contribution lies.**

## Pareto utilities

The building blocks are public, so you can analyse any set of results:

```python
from ikn_library.multiobjective import (
    dominates, non_dominated_sort, crowding_distance,
    pareto_sort_indices, pareto_front,
)

dominates([1, 1], [2, 2])            # True
non_dominated_sort(objectives)       # list of fronts (index arrays)
crowding_distance(objectives)        # spread measure within one front
pareto_sort_indices(objectives)      # best-to-worst order (argsort replacement)
pareto_front(solutions, objectives)  # keep the non-dominated ones
```

`pareto_front` deduplicates by default: discrete objectives (a feature
count, a number of ensemble members) produce many solutions that land
on identical objective values, and keeping them all would bloat the
front and distort crowding distances.

## Reference

K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
multiobjective genetic algorithm: NSGA-II," *IEEE Transactions on
Evolutionary Computation*, 6(2), 182-197, 2002.
[doi:10.1109/4235.996017](https://doi.org/10.1109/4235.996017).
