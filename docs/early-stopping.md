# Early Stopping with `patience`

A convergence curve that runs flat for its last third is wasted budget,
and `Task(..., patience=N)` is the tool for reclaiming it: the run stops
once **N consecutive iterations** pass without improving the best
solution. It works with every algorithm in the library, because the stop
is decided by the task rather than by the algorithm.

The catch is that a plateau is not convergence. Metaheuristics are
*designed* to sit still and then escape — that is what Lévy flights,
scout bees and nomad lions are for — so a patience value chosen by
instinct will eventually stop a run one iteration before its best
solution. This page is the procedure for choosing one that will not.

!!! warning "Never for a number you intend to report"
    Use `patience` for cheap sweeps, classroom demos and CI smoke tests.
    Do not use it for thesis results, algorithm comparisons, or anything
    that goes in the benchmark tables — it systematically favours
    algorithms that converge greedily over algorithms that explore well,
    which is precisely the comparison such tables exist to make.

## The two numbers that decide everything

`task.stall_lengths()` replays a run's convergence history and returns
the two quantities the decision rests on:

- **interior plateaus** — stretches that *ended* in an improvement. Your
  patience value must be larger than the longest of these, or it would
  have cut that run short.
- **the tail** — iterations after the last improvement. It never ended,
  so it is the budget patience could have saved.

If the tail is consistently much longer than the longest interior
plateau, patience pays. If they are the same size, it does not. Both
answers are useful; only measurement tells them apart.

## Case 1: patience pays

A continuous problem where you need six decimal places and the optimizer
keeps polishing far past them. `min_delta=1e-6` says "an improvement
smaller than this does not interest me", which is what turns an endless
trickle of 1e-15 improvements into a plateau patience can see:

```python
from ikn_library import Task
from ikn_library.algorithms import ParticleSwarmOptimization
from ikn_library.problems import Sphere

# Step 1 — one full run, no patience, to see the structure
task = Task(problem=Sphere(dimension=10), max_evals=20000, min_delta=1e-6)
ParticleSwarmOptimization(population_size=30, seed=0).run(task)
interior, tail = task.stall_lengths()

print("iterations              :", task.iters)
print("longest interior plateau:", int(interior.max()))
print("tail (recoverable)      :", tail)
```

Output:

```text
iterations              : 667
longest interior plateau: 15
tail (recoverable)      : 419
```

A tail of 419 iterations against a longest interior plateau of 15: the
run found everything it was going to find in its first third and then
polished digits nobody asked for. Patience should be just above 15 —
call it 20 — and the result confirms the reasoning:

```python
for patience in (None, 10, 20):
    task = Task(problem=Sphere(dimension=10), max_evals=20000,
                patience=patience, min_delta=1e-6)
    _, best = ParticleSwarmOptimization(population_size=30, seed=0).run(task)
    print(f"patience={str(patience):<5} iters={task.iters:>3} "
          f"evals={task.evals:>5} best={best:.2e} early={task.stopped_early}")
```

Output:

```text
patience=None  iters=667 evals=20000 best=9.21e-30 early=False
patience=10    iters= 75 evals= 2250 best=8.15e-02 early=True
patience=20    iters=268 evals= 8040 best=7.46e-08 early=True
```

`patience=20` **saves 60% of the budget** and still delivers 7.5e-08,
comfortably inside the 1e-6 that was asked for. `patience=10` is below
the measured plateau length of 15, and it fails exactly as the
measurement predicts: it stops after 75 iterations at 8e-02, five orders
of magnitude short.

That is the whole method in one comparison. The plateau measurement
separated the working value from the broken one *before* either was run.

## Case 2: patience does not pay

Now a discrete problem — wrapper feature selection, where each
evaluation costs a 5-fold cross-validation and saving iterations is
worth real time. Measure across several seeds, because one run's
plateau structure is itself noisy:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.problems import FeatureSelectionProblem

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def knn():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


rows = []
for seed in range(5):
    problem = FeatureSelectionProblem(X_train, y_train, estimator=knn(), cv=CV)
    task = Task(problem=problem, max_evals=1000)
    BinaryAntColonyOptimization(population_size=20, evaporation=0.1,
                                seed=seed).run(task)
    interior, tail = task.stall_lengths()
    rows.append((seed, task.iters, int(interior.max()) if len(interior) else 0, tail))

print("seed  iters  longest interior  tail")
for seed, iters, longest, tail in rows:
    print(f"{seed:>4}  {iters:>5}  {longest:>16}  {tail:>4}")

safe = max(row[2] for row in rows) + 1
saved = [max(0, tail - safe) for _, _, _, tail in rows]
print(f"\nsafe patience  : {safe}  (longest interior plateau + 1)")
print(f"average saving : {sum(saved) / len(saved):.1f} of "
      f"{rows[0][1]} iterations ({100 * sum(saved) / len(saved) / rows[0][1]:.0f}%)")
```

Output:

```text
seed  iters  longest interior  tail
   0     50                18    18
   1     50                10     8
   2     50                 2    44
   3     50                 8    19
   4     50                22     8

safe patience  : 23  (longest interior plateau + 1)
average saving : 4.2 of 50 iterations (8%)
```

Here the interior plateaus are as long as the tails. One run sat still
for 22 iterations and then improved; another stopped improving after
iteration 6. A patience value safe for the worst of them is 23, which is
longer than the wasted tail in four runs out of five — 8% of the budget
saved on average, in exchange for a real risk of losing a solution.

**On this problem, do not use patience.** That is a finding, not a
failed experiment, and it took one measurement to reach.

## Best practice: how to pick the number

1. **Run once without patience.** You need a complete run to see a
   plateau you would have survived; a truncated run cannot show you one.
   This is the same run you would draw a
   [convergence curve](convergence-plot.md) from anyway.
2. **Measure with `stall_lengths()` over at least three seeds**, five if
   you can afford them. Take the largest interior plateau across all of
   them, not the average — you are protecting against the worst run, not
   the typical one.
3. **Compare it against the tails.** If the tails are not consistently
   longer, stop here and do not use patience.
4. **Set `patience` = longest interior plateau + 50% margin**, and set
   `min_delta` to the precision you actually need. On continuous
   problems `min_delta` is not optional: with the default of 0.0, an
   improvement of 1e-15 resets the counter and patience never fires.
5. **Re-measure whenever the configuration changes.** Problem,
   algorithm, population size and budget each change the plateau
   structure, and the budget changes it more than you would expect:
   re-running the feature-selection measurement above with
   `max_evals=3000` — 150 iterations instead of 50 — moves the longest
   interior plateau from 22 to **98**, from 44% of the run to 65% of it.
   A value calibrated at one budget does not transfer to another.

### If you cannot afford the calibration run

Then either do not use patience, or make it deliberately generous —
half the iteration budget is defensible, since it can only ever trim a
tail longer than half the run. It saves less than a calibrated value,
but it cannot quietly cost you the answer, which is the failure mode
worth avoiding.

## Checklist

- [ ] I ran at least one full run with no patience.
- [ ] I measured `stall_lengths()` on three or more seeds.
- [ ] My patience exceeds the **largest** interior plateau I observed.
- [ ] I set `min_delta` if the problem is continuous.
- [ ] I re-measured after changing problem, algorithm, population or budget.
- [ ] This run is **not** producing a number I will report.

## Reference

- [`Task`](api.md) — `patience`, `min_delta`, `stalled_iters`,
  `stopped_early`, `stall_lengths()`.
- [Plotting Convergence](convergence-plot.md) — the curve these
  plateaus come from, and how to read one.
