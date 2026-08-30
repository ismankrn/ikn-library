# Fish School Search (FSS)

**FSS** (Bastos Filho et al., 2008) models a school of fish foraging.
Its distinguishing idea is that every fish carries a **weight** that
grows when it finds food — a running memory of success that no other
algorithm in this library maintains.

That weight does two jobs: it biases the school's centre of mass toward
the fish that have been doing well, and its *total* decides whether the
school **contracts** around a promising area or **expands** to look
elsewhere. Exploration and exploitation are therefore switched by the
school's own recent success rather than by a preset schedule.

## The four operators

Each iteration runs them in sequence:

| # | Operator | What it does |
|---|---|---|
| 1 | Individual movement | each fish tries a small random step, keeping it only if the food improves |
| 2 | Feeding | weights change in proportion to each fish's improvement |
| 3 | Collective instinctive | the whole school drifts along the improvement-weighted average step |
| 4 | Collective volitive | the school contracts or expands around its barycentre |

## Equations

**1. Individual movement.** With the current step \(s^{(t)}\):

\[
x_i^{\text{new}} = x_i + s^{(t)}\,(\text{upper} - \text{lower}) \odot
\mathcal{U}(-1, 1)^m
\]

accepted only if \(f(x_i^{\text{new}}) < f(x_i)\). The step decays
linearly from ``step_individual`` to ``step_individual_final`` over the
run.

**2. Feeding.** Weights grow with the improvement \(\Delta f_i\),
normalized by the best improvement in the school, and are clipped to
\([1, W_{\text{scale}}]\):

\[
W_i \leftarrow W_i + \frac{\Delta f_i}{\max_j \Delta f_j}
\]

**3. Collective instinctive movement.** The school drifts by the
improvement-weighted mean of the successful displacements, so fish that
found nothing still benefit from those that did:

\[
m = \frac{\sum_i \Delta x_i \, \Delta f_i}{\sum_i \Delta f_i},
\qquad x_i \leftarrow x_i + m
\]

**4. Collective volitive movement.** With the weight-biased barycentre

\[
B = \frac{\sum_i x_i W_i}{\sum_i W_i}
\]

the school moves toward it when the total weight **grew** this
iteration (the food is here — exploit) and away when it **fell**
(explore):

\[
x_i \leftarrow x_i \pm s_{\text{vol}}^{(t)} \, r \,
\frac{x_i - B}{\lVert x_i - B \rVert},
\qquad r \sim \mathcal{U}(0,1)
\]

## Pseudocode

```text
input: fish n, step s0 -> s_final, volitive factor, weight scale W
x <- n random solutions, evaluated;  W_i <- W/2

repeat until the budget is exhausted:
    s <- step decayed linearly with the spent budget
    for each fish i:                                  # 1. individual
        cand <- repair(x[i] + s * (upper - lower) * U(-1,1))
        if cand is better: record the improvement and move there

    W <- clip(W + improvement / max(improvement), 1, W_scale)   # 2. feeding

    if any fish improved:                             # 3. instinctive
        m <- improvement-weighted mean displacement
        x <- x + m

    B <- weight-biased barycentre                     # 4. volitive
    direction <- -1 if the school gained weight else +1
    x <- x + direction * s_vol * U(0,1) * (x - B) / ||x - B||

    re-evaluate the school (steps 3 and 4 moved every fish)

return best solution found
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 100 | Number of fish; a large school matters here (see below) |
| `step_individual` | 0.05 | Initial individual step (fraction of the bound range) |
| `step_individual_final` | 1e-6 | Final individual step |
| `step_volitive_factor` | 2.0 | Volitive step as a multiple of the individual step |
| `weight_scale` | 100.0 | Upper weight limit; fish start at half |
| `seed` | `None` | Reproducibility |

Note that one iteration costs about **two evaluations per fish** (the
individual move plus the refresh after the collective moves), so a
large school means comparatively few iterations.

!!! note "Tuning at a small budget can mislead"
    While tuning this algorithm, a 5,000-evaluation screen favoured a
    50-fish school (Rastrigin ≈ 3.3, versus 9.6 for 100 fish). At the
    full 20,000-evaluation budget the ranking **reversed**: 100 fish
    reached ≈ 2.7 while 50 fish fell to ≈ 9.6.

    A larger school explores longer before the volitive operator pulls
    it together, which only pays off when there are enough iterations
    left to exploit afterwards. Screen cheaply if you must, but confirm
    the finalists at the budget you will actually use.

## Behavior

FSS is a **mid-field all-rounder** with a good multimodal showing:
Sphere ≈ 7e-05, Ackley ≈ 0.06, Rastrigin ≈ 2.7 — third-best on
Rastrigin behind ABC and CSO. It does not reach the extreme precision
of Firefly or DE on smooth functions, because the collective operators
keep moving every fish each iteration and the school never fully
settles.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import FishSchoolSearch

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = FishSchoolSearch(population_size=100, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- C. J. A. Bastos Filho, F. B. de Lima Neto, A. J. C. C. Lins,
  A. I. S. Nascimento, and M. P. Lima, "A novel search algorithm based
  on fish school behavior," in *IEEE International Conference on
  Systems, Man and Cybernetics (SMC 2008)*, 2646-2651, 2008.
  [doi:10.1109/ICSMC.2008.4811695](https://doi.org/10.1109/ICSMC.2008.4811695).
- C. J. A. Bastos Filho, F. B. de Lima Neto, M. F. C. Sousa,
  M. R. Pontes, and S. S. Madeiro, "On the influence of the swimming
  operators in the fish school search algorithm," in *IEEE SMC 2009*,
  5012-5017, 2009.
  [doi:10.1109/ICSMC.2009.5346737](https://doi.org/10.1109/ICSMC.2009.5346737).
