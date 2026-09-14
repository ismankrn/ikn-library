# Beating Random Search

[Hyperparameter Optimization](parameter-optimization.md) shows how to
drive a metaheuristic over a hyperparameter space, and ends by reporting
the test score rather than the search's best. This page covers what that
test score has to be compared against before the search can be said to
have contributed anything.

The natural comparison — tuned model versus default model — measures
tuning, not the algorithm. A metaheuristic is one way of spending an
evaluation budget; drawing the same number of configurations uniformly
at random is another, and it costs nothing to implement. Random search
is therefore the reference point for a *search* claim, in the same way
that `tuning alone` is the reference point for a feature-selection
claim in [Feature Selection Protocol](proving-feature-selection.md).

Four measurements make the comparison interpretable.

## 1. A best-of-N score improves with budget on its own

Before comparing algorithms, it helps to see how much of a search score
is produced by the budget alone. This problem ignores its input
entirely — every candidate draws an independent standard normal, so
there is nothing whatsoever to optimize:

```python
import numpy as np

from ikn_library import Task
from ikn_library.algorithms import (AntColonyOptimization, GreyWolfOptimizer,
                                    ParticleSwarmOptimization)
from ikn_library.problems import Problem


class PureNoise(Problem):
    """Fitness ignores x: every candidate draws N(0, 1)."""

    def __init__(self, seed=0):
        super().__init__(dimension=5, lower=-5.0, upper=5.0)
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, x):
        return float(self.rng.normal())


def random_search(n, seed):
    return float(np.random.default_rng(seed).normal(size=n).min())


ALG = {"ACO-R": AntColonyOptimization,
       "PSO": ParticleSwarmOptimization,
       "GWO": GreyWolfOptimizer}

print(f"{'budget':>7} " + " ".join(f"{k:>8}" for k in ALG) + f"{'random':>9}")
for n in (50, 200, 1000, 5000):
    row = []
    for i, (name, algorithm) in enumerate(ALG.items()):
        best = []
        for s in range(10):
            task = Task(problem=PureNoise(seed=1000 * i + s), max_evals=n)
            algorithm(population_size=20, seed=s).run(task)
            best.append(task.best_fitness)
        row.append(np.mean(best))
    rnd = np.mean([random_search(n, 9000 + s) for s in range(10)])
    print(f"{n:>7} " + " ".join(f"{v:>8.3f}" for v in row) + f"{rnd:>9.3f}")
```

Output:

```text
 budget    ACO-R      PSO      GWO   random
     50   -2.399   -2.077   -1.829   -2.389
    200   -2.681   -2.611   -2.506   -2.796
   1000   -3.313   -3.077   -3.249   -3.201
   5000   -3.570   -3.558   -3.547   -3.623
```

Every column improves monotonically, and every convergence curve drawn
from these runs would slope downward convincingly. The improvement is
the expected minimum of N draws, which grows with N whatever the
objective is doing — here, nothing.

Random search is level with all three algorithms at every budget, which
is the expected result when there is no structure to exploit. That is
precisely what makes it a useful reference: on a real objective, the
gap between a metaheuristic and random search at equal budget is the
part of the score that came from searching rather than from sampling.

!!! note "Report the budget with the score"
    Two searches are only comparable at equal evaluation counts. A
    population of 30 for 50 iterations is 1500 evaluations; a population
    of 10 for 50 iterations is 500. Comparing their best scores compares
    budgets as much as algorithms.

## 2. Random search at the same budget

The same comparison on a real objective — an RBF SVM tuned over
`log10(C)` and `log10(gamma)`, exactly the problem from
[Hyperparameter Optimization](parameter-optimization.md):

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task

X, y = load_breast_cancer(return_X_y=True)
X_search, X_test, y_search, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def build(x):
    return make_pipeline(StandardScaler(), SVC(C=10 ** x[0], gamma=10 ** x[1]))


class SVMTuning(Problem):
    def __init__(self):
        super().__init__(dimension=2, lower=np.array([-3.0, -4.0]),
                         upper=np.array([3.0, 1.0]))

    def _evaluate(self, x):
        return cross_val_score(build(x), X_search, y_search, cv=CV, n_jobs=-1).mean()


def random_tuning(n, seed):
    """Same box, same objective, same budget — uniform draws."""
    rng = np.random.default_rng(seed)
    problem = SVMTuning()
    best_x, best = None, -np.inf
    for _ in range(n):
        x = rng.uniform(problem.lower, problem.upper)
        score = problem._evaluate(x)
        if score > best:
            best, best_x = score, x
    return best, best_x


N = 150
print(f"{'method':>14} {'CV best':>18} {'test':>18}")
for name in ("ACO-R", "random"):
    cvs, tests = [], []
    for s in range(5):
        if name == "ACO-R":
            task = Task(problem=SVMTuning(), max_evals=N,
                        optimization_type=OptimizationType.MAXIMIZATION)
            best_x, best = AntColonyOptimization(population_size=20, seed=s).run(task)
        else:
            best, best_x = random_tuning(N, 100 + s)
        model = build(best_x).fit(X_search, y_search)
        cvs.append(best)
        tests.append(model.score(X_test, y_test))
    print(f"{name:>14} {np.mean(cvs):>10.4f} +/- {np.std(cvs):<6.4f}"
          f"{np.mean(tests):>10.4f} +/- {np.std(tests):<6.4f}")

default = make_pipeline(StandardScaler(), SVC())
default_cv = cross_val_score(default, X_search, y_search, cv=CV, n_jobs=-1).mean()
default.fit(X_search, y_search)
print(f"{'defaults':>14} {default_cv:>10.4f} {'':<10}"
      f"{default.score(X_test, y_test):>10.4f}")
```

Output:

```text
budget = 150 evaluations, 5 seeds

        method            CV best               test
         ACO-R     0.9771 +/- 0.0011    0.9825 +/- 0.0000
        random     0.9758 +/- 0.0000    0.9807 +/- 0.0035
      defaults     0.9692               0.9825
```

ACO-R does beat random search, by **+0.0013** on the objective it
optimized and **+0.0018** on the test set. Both gaps are smaller than
one test-set observation (1/114 = 0.0088), so the honest summary is that
the two methods are indistinguishable here.

The third row is the one worth carrying into a discussion section:
`SVC()`'s defaults reach **0.9825** on the test set, matching ACO-R
exactly, with no search at all. The search space was already centred on
a good region, so 150 evaluations bought CV score that did not convert
into test accuracy.

None of that makes the search pointless — it is how the fact that the
defaults were well placed got established. It does mean the claim to
report is "tuning confirmed the default region", not "the metaheuristic
improved accuracy".

## 3. The objective's own noise floor

When the model is stochastic, the same hyperparameters return different
scores on different training seeds. Any difference smaller than that
spread is not attributable to the hyperparameters:

```python
from sklearn.neural_network import MLPClassifier


def mlp(width, lr, seed):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(width,), learning_rate_init=lr,
                      max_iter=300, early_stopping=True, random_state=seed))


scores = [cross_val_score(mlp(64, 1e-3, s), X_search, y_search,
                          cv=CV, n_jobs=-1).mean() for s in range(10)]
print(f"one configuration (64, 1e-3), 10 training seeds")
print(f"  mean   : {np.mean(scores):.4f}")
print(f"  spread : SD {np.std(scores):.4f} | range {np.ptp(scores):.4f}")
```

Output:

```text
one configuration (64, 1e-3), 10 training seeds
  mean   : 0.9211
  spread : SD 0.0143 | range 0.0462
```

One configuration spans **0.0462** across seeds. A search reporting a
0.02 improvement over a baseline, with one training seed each, is
reporting something this spread could have produced on its own.

Averaging over seeds changes the comparison it supports:

```python
A, B = (64, 1e-3), (16, 1e-2)


def evaluate(cfg, seeds):
    return np.mean([cross_val_score(mlp(*cfg, s), X_search, y_search,
                                    cv=CV, n_jobs=-1).mean() for s in seeds])


print(f"{'':<22}{'A=(64, 1e-3)':>14}{'B=(16, 1e-2)':>14}{'A - B':>10}")
for label, seeds in (("single seed", [0]), ("mean of 10 seeds", range(10))):
    a, b = evaluate(A, seeds), evaluate(B, seeds)
    print(f"{label:<22}{a:>14.4f}{b:>14.4f}{a - b:>+10.4f}")
```

Output:

```text
                        A=(64, 1e-3)  B=(16, 1e-2)     A - B
single seed                   0.9209        0.9451   -0.0242
mean of 10 seeds              0.9211        0.9558   -0.0347
```

The seed-averaged gap is 0.0347 and the single-seed estimate misses it
by 0.010 — and 0.0242 sits inside the 0.0462 range from a single
configuration, so one seed could not have established the ordering even
though it happened to get the sign right.

Two ways to handle this, both fine as long as the choice is stated:
fix the training seed so the objective is deterministic and the search
ranks architectures rather than initializations; or average a few seeds
per evaluation, which costs proportionally more but produces differences
that survive re-running.

!!! tip "Measure the floor before reading the table"
    Re-evaluating one configuration a handful of times takes a fraction
    of a search budget and tells you the smallest difference worth
    interpreting. Quote it next to the results.

## 4. Can the search reach the baseline?

A comparison against a baseline configuration assumes the search could
have found it. When the decode cannot express it, the comparison has no
interpretation in either direction.

A common shortcut is to draw one layer width and repeat it for every
layer, which quietly removes every narrowing architecture from the
space:

```python
UNITS = [16, 32, 64, 128, 256]


def decode_uniform(x):
    """One width, repeated for every layer."""
    n_layers = int(np.clip(np.floor(x[0] * 3) + 1, 1, 3))
    width = UNITS[int(np.clip(np.floor(x[1] * len(UNITS)), 0, len(UNITS) - 1))]
    return tuple([width] * n_layers)


def decode_per_layer(x):
    """One width per layer."""
    n_layers = int(np.clip(np.floor(x[0] * 3) + 1, 1, 3))
    return tuple(UNITS[int(np.clip(np.floor(x[1 + i] * len(UNITS)), 0,
                                   len(UNITS) - 1))]
                 for i in range(n_layers))


BASELINES = [(64,), (128, 64), (256, 128, 64)]
rng = np.random.default_rng(0)

print(f"{'decode':>14} {'distinct architectures':>24}   baselines reachable")
for name, fn, dim in (("uniform width", decode_uniform, 2),
                      ("per layer", decode_per_layer, 4)):
    seen = {fn(rng.random(dim)) for _ in range(200000)}
    reachable = [b for b in BASELINES if b in seen]
    print(f"{name:>14} {len(seen):>24}   {len(reachable)}/{len(BASELINES)}  "
          f"{[str(b) for b in reachable]}")
```

Output:

```text
        decode   distinct architectures   baselines reachable
 uniform width                       15   1/3  ['(64,)']
     per layer                       155   3/3  ['(64,)', '(128, 64)', '(256, 128, 64)']
```

The uniform-width decode spans 15 architectures and can express one of
the three baselines. Two of the comparisons it would appear in are
therefore between a search and a configuration the search was never able
to consider.

The check is one assertion, and it belongs next to the decode:

```python
for baseline in BASELINES:
    assert baseline in seen, f"{baseline} is outside the search space"
```

## Best practice: the four measurements

1. **Run random search at the same evaluation budget**, and report the
   gap against it as the contribution of the algorithm. Report the
   budget itself, in evaluations, not in iterations.
2. **Keep the default configuration in the table.** It is the cheapest
   arm and frequently a strong one; a search that matches it has
   established something worth stating.
3. **Measure the objective's noise floor** by re-evaluating one
   configuration several times, and interpret no difference smaller
   than that spread.
4. **Assert that every baseline is reachable by the decode** before
   comparing against it.
5. **Read the test set once**, after the configuration is frozen — see
   [Hyperparameter Optimization](parameter-optimization.md#reporting-open-the-test-set-once).
6. **Repeat over seeds** and report mean and spread; one run of a
   stochastic search on a stochastic objective is one draw from two
   distributions at once.

## Checklist

- [ ] Random search ran at the same evaluation budget.
- [ ] The budget is reported in evaluations.
- [ ] The default configuration appears in the results table.
- [ ] The objective's seed-to-seed spread was measured and quoted.
- [ ] Reported differences exceed that spread.
- [ ] Every baseline configuration is reachable by the decode.
- [ ] The test set was read once, after all choices were frozen.
- [ ] Results are averaged over several seeds.

## Reference

- [Hyperparameter Optimization](parameter-optimization.md) — defining the
  problem, encoding integer and categorical dimensions, refitting, and
  reporting.
- [Hyperparameter Tuning](gridsearch-comparison.md) — grid search as the
  exhaustive counterpart to the budgets used here.
- [Feature Selection Protocol](proving-feature-selection.md) — the
  same attribution question for feature subsets.
- [Setting the Annealing Temperature](annealing-temperature.md) — if the
  search is `SimulatedAnnealing`, calibrate it before benchmarking it.
- [Plotting Convergence](convergence-plot.md) — reading the curve that
  section 1 shows can slope downward on pure noise.
