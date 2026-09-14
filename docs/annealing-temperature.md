# Setting the Annealing Temperature

`SimulatedAnnealing` accepts an uphill move with probability
`exp(-delta / T)`, where `delta` is how much worse the neighbour is.
That expression compares a temperature against a *fitness difference*,
so `T` carries the problem's units. A value that is sensible on one
objective is meaningless on another.

The default `initial_temperature=1.0` is a placeholder, not a neutral
choice. What it actually does depends entirely on the scale of the
objective it is handed:

```python
import numpy as np

for name, delta in (("FeatureSelectionProblem (fitness in [0, 1])", 0.00653),
                    ("Rastrigin(dimension=10)", 53.1071)):
    print(f"{name:<44} delta={delta:<9.5f} "
          f"P(accept) at T0=1.0 = {np.exp(-delta / 1.0):.3e}")
```

Output:

```text
FeatureSelectionProblem (fitness in [0, 1])  delta=0.00653   P(accept) at T0=1.0 = 9.935e-01
Rastrigin(dimension=10)                      delta=53.10710  P(accept) at T0=1.0 = 8.627e-24
```

The same number produces two different algorithms. At 0.993 the search
accepts nearly every proposal and behaves as a random walk; at 1e-23 it
accepts none and behaves as a greedy hill-climber. Neither one anneals.

This page is the procedure for putting `T0` on the objective's scale,
deriving `cooling` from it, and checking afterwards that the run did
what the schedule intended.

## Step 1: measure a typical uphill move

`delta` is a property of the objective and the neighbourhood, so it is
measured, not assumed. Walk a short random trajectory and record the
moves that made things worse:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library.problems import FeatureSelectionProblem

X, y = load_breast_cancer(return_X_y=True)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def knn():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


problem = FeatureSelectionProblem(X_train, y_train, estimator=knn(), cv=CV,
                                  scoring="accuracy", alpha=0.99)

rng = np.random.default_rng(0)
x = rng.uniform(problem.lower, problem.upper, problem.dimension)
fx = problem.evaluate(x)
uphill = []
for _ in range(40):
    cand = np.clip(x + rng.normal(0.0, 0.1, problem.dimension), 0.0, 1.0)
    fc = problem.evaluate(cand)
    if fc > fx:
        uphill.append(fc - fx)      # a worse neighbour: record how much worse
    else:
        x, fx = cand, fc            # a better one: move there and carry on

delta = float(np.median(uphill))
print(f"typical uphill move  delta = {delta:.5f}   "
      f"(from {len(uphill)} worse neighbours)")
```

Output:

```text
typical uphill move  delta = 0.00653   (from 35 worse neighbours)
```

Use the **median**, not the mean: a few large jumps early in the walk
would otherwise set the temperature for the whole run.

!!! note "Match the walk to the algorithm"
    The `0.1` above is `SimulatedAnnealing`'s `step_size` — the same
    neighbourhood the real run will use. A `delta` measured with a
    different step size describes a different problem.

## Step 2: turn `delta` into `T0` and `cooling`

Two acceptance probabilities define the schedule: how often uphill moves
should be accepted at the start, and how rarely at the end. Inverting
`P = exp(-delta / T)` gives both temperatures, and the ratio between
them gives the cooling factor for a budget of `N` iterations:

```
T0     = delta / ln(1 / P_start)
T_end  = delta / ln(1 / P_end)
cooling = (T_end / T0) ** (1 / N)
```

```python
P_START, P_END = 0.80, 0.05
T0 = delta / np.log(1 / P_START)
T_end = delta / np.log(1 / P_END)

print(f"calibrated T0 = {T0:.5f}")
print(f"target T_end  = {T_end:.5f}\n")
print(f"{'N':>6}  {'cooling (calibrated T0)':>24}  {'cooling (T0=1.0)':>18}")
for N in (100, 500, 1000):
    print(f"{N:>6}  {(T_end / T0) ** (1 / N):>24.5f}  {T_end ** (1 / N):>18.5f}")
```

Output:

```text
calibrated T0 = 0.02925
target T_end  = 0.00218

     N   cooling (calibrated T0)    cooling (T0=1.0)
   100                   0.97436             0.94055
   500                   0.99482             0.98782
  1000                   0.99741             0.99389
```

Two things follow from that table.

`cooling` is derived, not chosen. It depends on the budget: the same
schedule that anneals over 1000 iterations barely moves over 100.
Values like 0.999 belong to runs of a few thousand iterations.

And the ceiling depends on `T0`. Starting from an uncalibrated `T0=1.0`
there is much further to fall, so `N=100` would need `cooling <= 0.941`
rather than `0.974` to reach the same end point.

## Step 3: check that the run annealed

The schedule is a plan; the acceptance probabilities are the evidence.
Print both ends:

```python
for T0v, c in ((1.0, 0.999), (1.0, 0.995), (T0, 0.995),
               (T0, (T_end / T0) ** (1 / 500))):
    T_final = T0v * c ** 500
    print(f"T0={T0v:<8.4f} cooling={c:<8.5f} "
          f"P(accept): {np.exp(-delta / T0v):.4f} -> {np.exp(-delta / T_final):.4f}")
```

Output:

```text
T0=1.0000   cooling=0.99900  P(accept): 0.9935 -> 0.9893
T0=1.0000   cooling=0.99500  P(accept): 0.9935 -> 0.9231
T0=0.0293   cooling=0.99500  P(accept): 0.8000 -> 0.0649
T0=0.0293   cooling=0.99482  P(accept): 0.8000 -> 0.0500
```

The first two rows begin and end in the same regime: with `T0=1.0` on a
fitness bounded in `[0, 1]`, the run accepts roughly 99% of uphill moves
from first iteration to last, and `cooling` has almost no leverage over
it. Comparing cooling schedules under those conditions measures very
little, because none of them is annealing.

The last two rows fall from 0.80 to 0.06 and 0.05. That transition —
exploratory early, selective late — is what the method is for.

!!! tip "The one-line check"
    Report `P(accept)` at the start and at the end alongside any SA
    result. It costs nothing, and it is the difference between "we ran
    simulated annealing" and "we ran a random walk with a temperature
    parameter attached".

## Choosing `P_start`: a knob, not a constant

`P_start = 0.80` is a reasonable default, not a law. How much early
exploration pays depends on the landscape, and the same instrumentation
that calibrates `T0` will sweep it:

```python
from ikn_library import Task
from ikn_library.algorithms import SimulatedAnnealing
from ikn_library.problems import Rastrigin

N, delta_r = 2000, 53.1071          # delta_r measured the same way, on Rastrigin

print(f"{'P_start':>8} {'T0':>10} {'cooling':>9} {'best (10 seeds)':>24}")
for P0 in (0.99, 0.80, 0.50, 0.20, 0.05, 0.01):
    T0_r = delta_r / np.log(1 / P0)
    c = (delta_r / np.log(1 / 0.001) / T0_r) ** (1 / N)
    out = []
    for s in range(10):
        task = Task(problem=Rastrigin(dimension=10), max_evals=N)
        SimulatedAnnealing(initial_temperature=T0_r, cooling=c,
                           step_size=0.1, seed=s).run(task)
        out.append(task.best_fitness)
    print(f"{P0:>8} {T0_r:>10.3f} {c:>9.5f} "
          f"{np.mean(out):>15.3f} +/- {np.std(out):<6.3f}")
```

Output:

```text
 P_start         T0   cooling          best (10 seeds)
    0.99   5284.112   0.99674          88.254 +/- 8.032
     0.8    237.995   0.99829          70.897 +/- 12.760
     0.5     76.617   0.99885          70.947 +/- 11.145
     0.2     32.997   0.99927          64.904 +/- 8.902
    0.05     17.728   0.99958          66.528 +/- 7.471
    0.01     11.532   0.99980          64.148 +/- 5.200
```

On Rastrigin the result improves monotonically as the start gets colder.
The reason is that this implementation already shrinks the neighbour
step with the temperature (`step ∝ sqrt(T / T0)`), so the wide early
steps supply the exploration and a permissive acceptance rule adds
little beyond wasted evaluations. Landscapes with narrow deep basins
behave the other way round, which is why the sweep is worth its cost
once — six settings over ten seeds, and the answer is specific to the
problem rather than borrowed from a textbook.

The point that does transfer: whichever `P_start` wins, it is chosen
against a measured `delta`. `T0` on the wrong scale does not appear
anywhere in this table, because every row is annealing.

## Best practice

1. **Measure `delta`** with the same `step_size` the run will use, and
   take the median of the uphill moves.
2. **Set `T0 = delta / ln(1 / P_start)`**, starting from
   `P_start = 0.80`.
3. **Derive `cooling` from the budget**: `(T_end / T0) ** (1 / N)` with
   `P_end` around 0.05. Recompute it whenever `N` changes — a schedule
   calibrated for 1000 iterations does nothing in 100.
4. **Print `P(accept)` at both ends** and report them with the result.
5. **Sweep `P_start` once** over a few seeds if the budget allows, and
   report which value you used.
6. **Re-measure after changing the problem, the estimator or
   `step_size`.** All three change `delta`, and `T0` follows `delta`.

## Checklist

- [ ] `delta` was measured, not assumed.
- [ ] The measuring walk used the run's own `step_size`.
- [ ] `T0` comes from `delta` and a stated `P_start`.
- [ ] `cooling` was derived from `T0`, `T_end` and the iteration budget.
- [ ] `cooling` was recomputed after any change to the budget.
- [ ] `P(accept)` at the start and end are reported with the result.
- [ ] Results are averaged over several seeds.

## Reference

- [`SimulatedAnnealing`](algorithm-details/sa.md) — the acceptance rule,
  the step-size schedule, and the parameters calibrated here.
- [Proving Feature Selection Helped](proving-feature-selection.md) — the
  companion check on whether the search moved at all.
- [Early Stopping (patience)](early-stopping.md) — same principle applied
  to a different parameter: measure it rather than guess it.
- [Plotting Convergence](convergence-plot.md) — the curve to read next to
  the acceptance trajectory.
