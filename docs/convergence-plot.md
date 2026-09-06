# Plotting Convergence

A *convergence curve* shows the best objective value found so far after
each iteration of a metaheuristic. It is the primary diagnostic for an
optimization run: it tells you whether the algorithm is still improving
or has stalled — and therefore whether your evaluation budget was spent
well.

## Getting the data

Every `Task` records the best score after each iteration automatically;
no extra bookkeeping is needed. After a run (here, the SVM tuning task
from the [Hyperparameter Optimization](parameter-optimization.md) tutorial):

```python
iters, scores = task.convergence_data()

import matplotlib.pyplot as plt
plt.plot(iters, scores)
plt.xlabel("Iteration")
plt.ylabel("Best CV accuracy")
plt.show()
```

`convergence_data()` returns two arrays: the iteration numbers and the
best score at the end of each iteration, expressed in the problem's
original sense (so for a maximization task the curve goes up).

## Example output

Running the SVM tuning task (`max_evals=150`, ACO-R with
`population_size=10`, `seed=42`) produces:

![Convergence curve of ACO-R on SVM tuning](img/convergence_plot.png)

## How to read the curve

- **It never goes down.** The curve tracks the *best-so-far* score, so
  it is monotone by construction — a drop would indicate a bug, not bad
  luck.
- **Steps, not slopes.** Improvements arrive as jumps whenever some ant
  finds a better solution; flat stretches mean iterations passed
  without a new best.
- **Read the plateaus, and be careful what you conclude.** The example
  gains 0.9736 -> 0.9780 in two steps, at iterations 4 and 12 of 15.
  The seven flat iterations in between look like a finished search, but
  they are not: the last improvement arrives in the final fifth of the
  budget, so cutting the budget on the strength of that plateau would
  have cost the best solution. A plateau is only evidence of
  convergence when it runs to the end of a budget that was generous to
  begin with — and if the score is still disappointing, the fix is
  usually more exploration (a larger population or archive), not more
  iterations.
- **Compare runs fairly.** When comparing algorithms or settings, plot
  their curves against *evaluations* used, not wall-clock time, and use
  the same seed policy — otherwise the comparison mixes convergence
  behavior with implementation speed.

## Stopping early when the curve flattens

Reading a plateau off a finished plot is hindsight. `Task` can apply the
same judgement *during* the run: pass `patience=N` and the search stops
once **N consecutive iterations** pass without improving the best
solution. It works with every algorithm in the library, because the stop
is decided by the task rather than by the algorithm.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task
from ikn_library.algorithms import AntColonyOptimization
from ikn_library.problems import Problem

X, y = load_breast_cancer(return_X_y=True)
X_search, X_test, y_search, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)


class SVMTuning(Problem):
    """The tuning problem from the Hyperparameter Optimization page."""

    def __init__(self, X, y, cv):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y, self.cv = X, y, cv

    def decode(self, x):
        return {"C": 10.0 ** x[0], "gamma": 10.0 ** x[1]}

    def _evaluate(self, x):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", **self.decode(x)))
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()


for patience in (None, 3, 5, 8):
    task = Task(problem=SVMTuning(X_search, y_search, cv=CV), max_evals=150,
                optimization_type=OptimizationType.MAXIMIZATION,
                patience=patience)
    _, best = AntColonyOptimization(population_size=10, archive_size=15,
                                    seed=42).run(task)
    print(f"patience={str(patience):<4} iters={task.iters:>2} "
          f"evals={task.evals:>3} best CV={best:.4f} "
          f"stopped_early={task.stopped_early}")
```

Output:

```text
patience=None iters=15 evals=150 best CV=0.9780 stopped_early=False
patience=3    iters= 7 evals= 75 best CV=0.9758 stopped_early=True
patience=5    iters= 9 evals= 95 best CV=0.9758 stopped_early=True
patience=8    iters=15 evals=150 best CV=0.9780 stopped_early=False
```

This is the same run the curve above plots, so the table can be read
against it directly — and it makes the earlier warning concrete.
`patience=3` and `patience=5` both stop inside the seven-iteration
plateau, halving the budget and **missing the improvement that arrives
at iteration 12**. `patience=8` is the smallest value that survives this
run, and the only way to know that was to run without patience first.

!!! warning "Patience is a budget saver, not a convergence detector"
    A plateau means no candidate has improved recently. It does not mean
    none will. Use `patience` when evaluations are expensive and a
    slightly worse answer is acceptable — a tuning sweep across many
    datasets, a classroom demo, a smoke test — and not when the run
    produces a number you intend to report.

    A budget is still required alongside it. `patience` shortens a run;
    it cannot bound one, because a search that keeps improving by a hair
    every iteration would never stop.

### `min_delta`: how big an improvement counts

By default any strict improvement resets the counter, and on continuous
problems that is almost always what happens — improvements of 1e-15
arrive forever, so patience never triggers. `min_delta` sets the size an
improvement must reach to count, measured against the last *improving*
iteration rather than the previous one, so slow steady progress still
accumulates and resets the counter:

```python
task = Task(problem=..., max_evals=20000, patience=10, min_delta=1e-6)
```

Two attributes report what happened: `task.stalled_iters` is the current
run of iterations without improvement, and `task.stopped_early` is
`True` only when patience — not the budget — ended the run.

!!! tip "Where does the starting point come from?"
    The curve starts high already at iteration 1 because the initial
    population is evaluated before the first iteration — with 10+
    random candidates, the best of them is usually far better than a
    single random guess.
