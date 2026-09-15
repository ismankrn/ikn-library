# Random Search

Random search is not a metaheuristic and makes no attempt to be one. It
draws points uniformly from the search box, scores them, and forgets
them. Nothing is carried between iterations: no population, no
attraction, no step size, no parameter to tune.

It is in this library for one reason. **A metaheuristic's score is only
evidence of searching if it beats uniform sampling at the same
evaluation budget.** Every run of every algorithm produces a
best-so-far curve that slopes downward, because the minimum of N draws
improves with N whatever the objective is doing — even for pure noise.
Random search is the line that separates that arithmetic from an actual
search. The teaching note
[Beating Random Search](../beating-random-search.md) works through what
goes wrong when the comparison is skipped.

## Equations

There is one:

$$x_i \sim \mathcal{U}(\text{lower}, \text{upper}), \quad i = 1 \dots n$$

drawn independently for every coordinate of every candidate, in every
iteration. The best solution is whatever the task has recorded, not
something the algorithm maintains.

## Pseudocode

```text
repeat until the budget is exhausted:
    draw population_size points uniformly from the search box
    evaluate them
    discard them
return the best point the task recorded
```

The implementation is the same four lines:

```python
def run_iteration(self, task, state):
    population = self.rng.uniform(
        task.lower, task.upper, (self.population_size, task.dimension)
    )
    fitness = np.array([task.eval(x) for x in population])
    return population, fitness
```

## Parameters

| Parameter | Meaning | Default | Notes |
|---|---|---|---|
| `population_size` | points drawn per iteration | 25 | Only groups evaluations into iterations; the stream of points is unchanged |
| `seed` | reproducibility | `None` | |

There is nothing else, and that is the point: any advantage another
algorithm shows over this one cannot be attributed to tuning.

## Behavior

### It is the flattest profile in the library

Uniform sampling has no preferred region, so transforming the problem
costs it almost nothing (10 dimensions, 20,000 evaluations, mean over 3
seeds; rotation is a fixed orthogonal matrix, shift is +1.0 in every
coordinate):

| Function | plain | rotated | shifted | both | max/min |
|---|---|---|---|---|---|
| Sphere | 11.38 | 11.38 | 10.86 | 10.86 | **1.0x** |
| Rastrigin | 64.46 | 68.67 | 70.96 | 65.03 | **1.1x** |
| Ackley | 16.68 | 16.65 | 16.53 | 16.20 | **1.0x** |

Sphere is radially symmetric, so rotating it cannot change anything —
the identical pair of columns is a check that the harness is doing what
it claims. The remaining variation is seed noise.

### What searching is worth, and how much of it is an artefact

Against a fixed reference, the benchmark suite's flattery becomes a
number instead of an argument. On Sphere, under the same protocol:

| | plain | shifted | advantage over random, plain | advantage, shifted |
|---|---|---|---|---|
| Random Search | 11.38 | 10.86 | — | — |
| [GWO](gwo.md) | 4e-88 | 3e-06 | 10^89 | 10^6.5 |
| [DE](de.md) | 2e-41 | 0 | 10^42 | exact |
| [PSO](pso.md) | 4e-29 | 2e-28 | 10^30 | 10^29 |

Grey Wolf's advantage over random sampling falls by **82 orders of
magnitude** when the optimum is moved off the origin; DE's and PSO's do
not move. The origin bias documented on the [GWO page](gwo.md) is
visible here as a ratio against a baseline that cannot itself be
biased.

### On real problems it is often not far behind

Tuning an RBF SVM over `log10(C)` and `log10(gamma)` — the problem from
[Hyperparameter Optimization](../parameter-optimization.md), same split
and same folds, 150 evaluations:

| | best CV | test | parameters |
|---|---|---|---|
| Random Search | 0.9780 | 0.9825 | C=6.2522, gamma=0.0033 |
| ACO-R | 0.9780 | 0.9825 | C=6.7972, gamma=0.0031 |
| default `SVC()` | 0.9692 | 0.9825 | — |

In two dimensions with a smooth objective, 150 random draws find the
same place a guided search finds. Reporting ACO-R's 0.9780 without this
row would credit the algorithm for what the budget did.

The picture changes in a larger, discrete space. Wrapper feature
selection on the same data, 1000 evaluations:

| | fitness | features | test accuracy |
|---|---|---|---|
| Random Search | 0.0318 | 17 | 0.9474 |
| [Binary ACO](binary-aco.md) | **0.0239** | **13** | **0.9649** |
| all 30 features | — | 30 | 0.9561 |

Here guidance earns its keep. Which of the two situations you are in is
a property of your problem, and running this algorithm is how you find
out.

!!! warning "On subset problems it samples the middle"
    Solutions are drawn uniformly from `[0, 1]`, so with the default
    threshold of 0.5 each variable is kept with probability one half.
    Its subsets cluster around half the features — 17 of 30 above — and
    it will essentially never propose a small one. Against an objective
    that rewards small subsets it is therefore a *handicapped*
    reference, not a neutral one. Compare subset sizes as well as
    scores before concluding that a wrapper beat it.

## Usage

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import RandomSearch

task = Task(problem=Sphere(dimension=10), max_evals=20000)
best_x, best_fitness = RandomSearch(population_size=20, seed=42).run(task)
```

It accepts the same `Problem` objects as everything else, so the same
call works for `FeatureSelectionProblem`, a hyperparameter search, or
an ensemble-weight problem — swap the algorithm and keep the rest of
the script.

## References

- J. Bergstra and Y. Bengio, "Random search for hyper-parameter
  optimization," *Journal of Machine Learning Research*, 13, 281-305,
  2012 — the paper that established random search as the baseline for
  hyperparameter tuning, and showed it beating grid search on the same
  budget.
- Z. B. Zabinsky, "Random search algorithms," in *Wiley Encyclopedia of
  Operations Research and Management Science*, 2010 — convergence
  properties of pure random search.
