# Forest Optimization Algorithm (FOA)

**FOA** (Ghaemi & Feizi-Derakhshi, 2014) models how a forest seeds
itself. Each tree is a solution carrying one extra attribute nothing
else in this library has: an **age**. Age is not decoration — it decides
who may reproduce, and it gives every tree exactly one year in which it
is allowed to drop seeds.

The algorithm's other distinctive idea is what happens to the trees it
throws away. Cut trees are not deleted; they go to a **candidate
population**, and that discarded pool is the *only* raw material for
long-range exploration. FOA is the only algorithm here that recycles
solutions it has already rejected.

## Equations

**1. Local seeding.** Only trees of age 0 seed. Each of the `lsc` seeds
copies its parent and nudges a **single** randomly chosen coordinate:

\[
x'_d = x_d + r \, \delta^{(t)}_d,
\qquad r \sim \mathcal{U}(-1, 1),
\quad d \sim \mathcal{U}\{1,\dots,D\}
\]

**2. Step decay.** The nudge shrinks with the spent budget:

\[
\delta^{(t)} = \mathrm{dx} \cdot (u - l)
\left(\max\left(1 - \tfrac{\text{evals}}{\text{max\_evals}},\ 10^{-6}\right)\right)^{2}
\]

**3. Ageing.** Every tree already standing gains a year; the new seeds
enter at age 0. A tree therefore seeds locally **once**, then ages out
of the operator.

**4. Population limiting.** Trees with \(\text{age} > \text{life\_time}\)
are cut, then the worst trees above the area limit are cut too. Both
groups join the candidate population \(C\).

**5. Global seeding.** A share of \(C\) is replanted, each replacing
`gsc` coordinates with fresh uniform values:

\[
|C_{\text{transfer}}| = \operatorname{round}(\text{transfer\_rate} \cdot |C|),
\qquad
x_d \sim \mathcal{U}(l_d, u_d) \ \text{ for } \text{gsc} \text{ chosen } d
\]

**6. Elitism.** The best tree's age is reset to 0, so the incumbent
optimum keeps seeding and can never age out.

## Pseudocode

```text
input: area limit N, life_time, lsc, gsc, transfer_rate, dx
trees <- N random solutions, evaluated, all age 0

repeat until the budget is exhausted:
    delta <- dx * (u - l) * (1 - evals/max_evals)^2              (eq. 2)
    for each tree of age 0:                       # local seeding
        drop lsc seeds, each nudging one coordinate by delta     (eq. 1)
    age of every standing tree += 1;  new seeds enter at age 0   (eq. 3)

    C <- trees older than life_time                # population limiting
    C += worst trees above the area limit                        (eq. 4)
    remove both groups from the forest

    for round(transfer_rate * |C|) trees drawn from C:   # global seeding
        replace gsc coordinates with uniform random values       (eq. 5)
        replant at age 0

    age[best tree] <- 0                            # elitism      (eq. 6)

return best solution found
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 15 | Area limit — maximum trees, not a fixed headcount |
| `life_time` | 6 | Age at which a tree is cut |
| `lsc` | 2 | Seeds dropped per age-0 tree |
| `gsc` | 1 | Coordinates randomised when replanting a candidate |
| `transfer_rate` | 0.1 | Share of the candidate pool replanted |
| `dx` | 0.2 | Local seeding step (fraction of the bound range) |
| `seed` | `None` | Reproducibility |

!!! danger "Read this before trusting the benchmark numbers"
    FOA scores Rastrigin ≈ 7e-06, which on the table below makes it look
    like the second-strongest algorithm in the library. **Most of that
    score is an artefact of the benchmark, not a property of the
    algorithm.**

    Local seeding changes **one coordinate at a time** (eq. 1). Sphere,
    Rastrigin, and Ackley are all *separable* — each coordinate can be
    optimized independently — so a one-coordinate-at-a-time search is
    close to the ideal strategy for them. Rotate the same function so
    the coordinates become coupled, and the advantage evaporates:

    | Configuration | Rastrigin | Rastrigin, rotated |
    |---|---|---|
    | `lsc=1` (one seed per coordinate) | **3e-07** | 42.5 |
    | `lsc=2` (default) | 9e-06 | **27.2** |

    The setting that is **30× better** on the published benchmark is
    **1.6× worse** once the problem is rotated. Tuning FOA on this
    benchmark suite therefore actively selects for a weakness.

    The defaults here deliberately **decline the better benchmark
    score** in favour of the configuration that survives rotation. For
    comparison on the rotated function, [CRO](cro.md) reaches 7.6 and
    [FWA](fwa.md) 17.7 — both well ahead of FOA, and the ordering on
    the plain benchmark reverses it.

    All three benchmarks in this library are separable. When you
    evaluate any algorithm for real work, include a non-separable
    problem such as Rosenbrock or a rotated variant.

!!! note "Tuning notes"
    - **`dx` is the parameter that matters most**, and its effect is
      almost entirely the separability effect above: raising it from
      0.1 to 0.3 takes Rastrigin from 3.1 to 1e-04 by making the
      per-coordinate jumps large enough to hop between the function's
      regularly spaced local optima.
    - **`transfer_rate` above ~0.2 is harmful** (Sphere 0.59, Ackley
      3.9 at 0.3): too much of the forest becomes freshly randomised
      trees and the search cannot settle. Setting it to 0 is also worse
      (Rastrigin 8.6), which confirms the candidate pool is doing real
      work rather than being decorative.
    - **`gsc` should stay small.** Randomising 3 coordinates instead of
      1 costs an order of magnitude on Rastrigin (9.95 vs 3.08 at the
      original defaults) — a replanted tree that keeps most of its
      inherited coordinates is far more useful than a near-random one.

## Behavior

On the library's benchmarks FOA reaches Sphere ≈ 3e-08,
Rastrigin ≈ 7e-06, Ackley ≈ 1e-03. Read alongside the warning above,
the honest summary is: **excellent on separable problems, unremarkable
otherwise.**

Tracing a run at area limit 15, the forest holds 15–18 trees and ages
top out at exactly `life_time`, so population limiting is continuously
active rather than a rare event. Its real function is to keep feeding
the candidate pool: with `transfer_rate=0` the algorithm loses roughly
a factor of three on Rastrigin, so recycling discarded trees genuinely
drives the exploration.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import ForestOptimizationAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = ForestOptimizationAlgorithm(population_size=15, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- M. Ghaemi and M.-R. Feizi-Derakhshi, "Forest optimization algorithm,"
  *Expert Systems with Applications*, 41(15), 6676-6687, 2014.
  [doi:10.1016/j.eswa.2014.05.009](https://doi.org/10.1016/j.eswa.2014.05.009).
- M. Ghaemi and M.-R. Feizi-Derakhshi, "Feature selection using forest
  optimization algorithm," *Pattern Recognition*, 60, 121-129, 2016.
  [doi:10.1016/j.patcog.2016.05.012](https://doi.org/10.1016/j.patcog.2016.05.012).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010 — on why rotated
  and non-separable functions belong in any benchmark suite.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
