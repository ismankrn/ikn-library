# Monarch Butterfly Optimization (MBO)

**MBO** (Wang et al., 2015) splits the population into two lands that
run **different operators**, with butterflies exchanging coordinates
across the divide.

Both operators are **per-coordinate discrete recombination**: a new
butterfly is assembled from pieces of several existing ones. Nothing in
MBO computes a direction or a velocity — even its Lévy step is applied
to each coordinate independently. It is the purest coordinate-wise
search in this library, which turns out to determine everything about
where it works and where it does not.

## Equations

**1. Migration** (Land 1). Each coordinate \(k\) is copied from a
randomly chosen donor, in Land 1 or Land 2 depending on a per-coordinate
draw:

\[
x_{i,k} =
\begin{cases}
x_{r_1,k}, \quad r_1 \in \text{Land 1}
  & \text{if } \text{rand} \cdot \text{peri} \le p \\[4pt]
x_{r_2,k}, \quad r_2 \in \text{Land 2} & \text{otherwise}
\end{cases}
\]

**2. Adjusting** (Land 2). Each coordinate comes from the current best
or from a Land 2 peer:

\[
x_{j,k} =
\begin{cases}
x_{\text{best},k} & \text{if } \text{rand} \le p \\[4pt]
x_{r_3,k} & \text{otherwise, plus a walk if } \text{rand} > \text{BAR}
\end{cases}
\]

**3. The Lévy walk**, the only operator producing values not already in
the population:

\[
x_{j,k} \leftarrow x_{j,k} + \alpha\bigl(L_k - 0.5\bigr)
\]

**4. Step weight.** The paper sets \(\alpha = S_{\max} / t^2\), tied to
the absolute generation count. This implementation ties it to the
remaining budget instead — see below:

\[
\alpha = S_{\max}\left(1 - \frac{\text{evals}}{\text{max\_evals}}\right)^{2}
\]

## Pseudocode

```text
input: butterflies n, partition p, period, BAR, max_step, elites
x <- n random solutions, evaluated and sorted

repeat until the budget is exhausted:
    keep a copy of the best `elites` butterflies
    Land 1 <- the best ceil(p*n);  Land 2 <- the rest
    alpha <- max_step * (1 - evals/max_evals)^2                   (eq. 4)

    for each butterfly in Land 1:
        rebuild it coordinate by coordinate from both lands       (eq. 1)
    for each butterfly in Land 2:
        rebuild it from the best or a Land 2 peer                 (eq. 2)
        peer coordinates may get a Levy step                      (eq. 3)

    restore the elites over the worst;  re-sort

return best solution found
```

## Parameters

| Parameter | Default | Paper | Effect |
|---|---|---|---|
| `population_size` | 20 | 50 | Number of butterflies |
| `partition` | 0.5 | 5/12 | Land 1 share and per-coordinate draw |
| `period` | 1.2 | 1.2 | Scales the migration draw |
| `bar` | 0.85 | 5/12 | Below it, no Lévy step is added |
| `max_step` | 3.0 | 1.0 | Maximum Lévy walk step |
| `n_elite` | 4 | 2 | Butterflies carried through unchanged |
| `budget_tied_step` | `True` | — | Use the budget instead of \(1/t^2\) |
| `seed` | `None` | — | Reproducibility |

!!! danger "MBO collapses on non-separable problems — by a factor of 200,000"
    MBO's Rastrigin score of 3e-04 puts it near the top of the
    comparison table. Rotating the function so its coordinates couple
    destroys it (20,000 evaluations, 3 seeds):

    | Variant | MBO | [HS](hs.md) | [KH](kh.md) |
    |---|---|---|---|
    | plain | **3e-04** | 0.018 | 2.99 |
    | shifted | **1e-04** | 0.075 | 2.65 |
    | rotated | **39.6** | 26.3 | 2.65 |
    | rotated + shifted | **57.5** | 30.4 | 2.99 |

    This is the most extreme separability dependence measured in this
    library — roughly **200,000×** from plain to rotated, against about
    1,500× for Harmony Search.

    The cause is visible in the equations rather than inferred: every
    operator assigns coordinates *independently*. Equations 1 and 2 copy
    coordinate \(k\) from some donor's coordinate \(k\); equation 3 adds
    an independent Lévy value to each. There is no term anywhere that
    relates two coordinates of the same solution, so MBO cannot
    represent, let alone search along, a diagonal valley.

    Note the shifted column: MBO has **no origin bias** at all, because
    it copies values rather than scaling absolute coordinates. It fails
    on exactly one of the two transformations, and completely.

    **Use MBO only where you have reason to believe the variables are
    close to separable.** Where they are, it is genuinely strong.

!!! warning "The published step schedule fails at this budget"
    MBO defines \(\alpha = S_{\max}/t^2\) against the **absolute
    generation number**. With 20,000 evaluations and 20 butterflies
    there are ~1,000 generations, so \(\alpha\) falls below \(10^{-4}\)
    within the first 100 — and since the Lévy walk is the only source of
    values not already in the population, the search then has nothing
    left but reshuffling.

    | Step schedule | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | \(S_{\max}/t^2\), as published | 10.6 | 43.3 | 15.4 |
    | **tied to the budget** | **2e-07** | **4e-04** | **9e-04** |

    Set `budget_tied_step=False` to reproduce the published behaviour.
    Note this is a subtler failure than the fixed steps the
    [CRO](cro.md) and [KH](kh.md) pages describe: MBO *does* decay, but
    against a quantity that has nothing to do with how much budget
    remains.

!!! note "Tuning notes"
    - **The population's coordinate diversity collapses early.** Tracing
      a run with 50 butterflies, the number of distinct values per
      coordinate falls from 50 to about 17 within 500 evaluations and
      never recovers, because both operators only copy. Strong elitism
      (`n_elite=4` of 20) and a large Lévy step are the compensations.
    - **`max_step` wants to be large**, the opposite of most algorithms
      here: 1.0 → 3.0 is worth an order of magnitude, because the walk
      is the only refinement mechanism.
    - **`bar` should be high** (0.85 against the paper's 5/12), so the
      walk is applied selectively rather than to most coordinates.

## Behavior

On separable problems MBO is strong: Sphere ≈ 4e-07, Rastrigin ≈ 0.40
(median 7e-04), Ackley ≈ 9e-04. Off them it is unusable.

That makes it a clean teaching example of a point the
[index page](index.md#benchmark-comparison) makes in general: a
benchmark table row is a statement about a *pairing*. MBO's row looks
excellent and is honestly measured, and it still tells you almost
nothing about how the algorithm will behave on a problem whose
variables interact.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import MonarchButterflyOptimization

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = MonarchButterflyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- G.-G. Wang, S. Deb, and Z. Cui, "Monarch butterfly optimization,"
  *Neural Computing and Applications*, 31(7), 1995-2014, 2019 (first
  published online 2015).
  [doi:10.1007/s00521-015-1923-y](https://doi.org/10.1007/s00521-015-1923-y).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010 — on why rotated
  variants belong in any benchmark suite.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
