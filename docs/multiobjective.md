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
On the breast-cancer dataset, four values of `alpha` gave four
different subsets (5, 3, 4, and 2 features) at four separate runs.

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
from ikn_library.multiobjective import (
    MultiObjectiveFeatureSelection, MultiObjectiveTask,
)
from ikn_library.algorithms import NSGA2

problem = MultiObjectiveFeatureSelection(X, y, cv=5)
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

On a synthetic dataset (400 samples, 30 features, 12 informative), one
NSGA-II run produces:

![Pareto front: accuracy versus number of features](img/pareto_front.png)

```text
n_features | CV accuracy
-----------|------------
     2     |   0.5850
     3     |   0.7000
     4     |   0.7575
     5     |   0.8000
     6     |   0.8300
     7     |   0.8525
     8     |   0.8675
     9     |   0.8825
    12     |   0.8850
```

The curve makes the decision obvious in a way no single `alpha` could:
accuracy climbs steeply up to about 9 features, then flattens — the
last three features buy 0.25 percentage points. A clinician paying per
assay would stop at 8 or 9.

!!! tip "A smaller subset can be just as good"
    On the breast-cancer data, NSGA-II found a **2-feature** subset with
    95.08% accuracy — matching the *5-feature* subset that the
    weighted-sum version returned with `alpha=0.99`. Optimizing the
    objectives separately exposed a solution the scalarized version
    never surfaced.

## Using the selected features to train a model

A Pareto front is a menu, not an answer. Three steps turn it into a
working model.

!!! warning "Run the selection on training data only"
    Feature selection is part of model building, so it must not see the
    test set. Split first, optimize on the training split, and keep the
    test split untouched for the final number — otherwise the reported
    accuracy is optimistic.

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 1. Build the front from the TRAINING data
problem = MultiObjectiveFeatureSelection(X_train, y_train, cv=5)
solutions, objectives = NSGA2(population_size=40, seed=42).run(
    MultiObjectiveTask(problem=problem, max_evals=3000))
```

**Step 2 — pick a point on the front.** Common criteria:

```python
import numpy as np

n_features = np.array([len(problem.selected_features(s)) for s in solutions])
cv_accuracy = 1 - objectives[:, 0]

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

**Step 3 — extract the features and fit the final model.** The solution
vector is a mask, so `selected_features` gives the column indices:

```python
features = problem.selected_features(solutions[most_accurate])

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train[:, features], y_train)
accuracy = accuracy_score(y_test, model.predict(X_test[:, features]))
```

Note that **the same column indices must be applied to any future
data** — store `features` alongside the model.

### What it gives you

Comparing the three candidates against using every feature — the front
is sorted by subset size, so `smallest` is simply index 0:

```python
model = KNeighborsClassifier(n_neighbors=5)
candidates = {
    "most accurate": int(np.argmax(cv_accuracy)),
    "knee point": knee_index(n_features, cv_accuracy),
    "smallest": 0,
}

baseline = accuracy_score(y_test, model.fit(X_train, y_train).predict(X_test))
print(f"all {X.shape[1]} features            : test accuracy = {baseline:.4f}")

for label, index in candidates.items():
    features = problem.selected_features(solutions[index])
    model.fit(X_train[:, features], y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test[:, features]))
    print(f"{label:<14} ({len(features):>2} features): "
          f"test accuracy = {accuracy:.4f}")
```

Output:

```text
all 30 features            : test accuracy = 0.8333
most accurate  (11 features): test accuracy = 0.8583
knee point     ( 5 features): test accuracy = 0.6667
smallest       ( 2 features): test accuracy = 0.6333
```

The complete runnable script is
[`examples/multiobjective_feature_selection.py`](https://github.com/ismankrn/ikn-library/blob/main/examples/multiobjective_feature_selection.py).

The 11-feature subset **beats using all 30** — higher accuracy from a
third of the inputs, which is the payoff feature selection promises.

!!! warning "The knee point is a heuristic, not a rule"
    A popular shortcut is to take the *knee* — the point of maximum
    curvature, where accuracy stops climbing steeply. Here that picked
    5 features and scored **0.6667 on the test set**, far below both the
    11-feature subset and using everything.

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
