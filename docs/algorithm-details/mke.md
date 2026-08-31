# Monkey King Evolution (MKE)

**MKE** (Meng & Pan, 2016) is named for the Monkey King's trick of
plucking out his hairs and turning them into copies of himself. The best
individual — the **king** — spawns a group of clones that each probe a
different direction, and the best clone replaces him if it beats him.

That asymmetry is the algorithm's one structural idea, and it is a real
one: the king spends `n_clones` evaluations per iteration exploring
around the current optimum while every other individual spends exactly
one. The budget concentrates where the search is already doing well,
with none of the ranking or weighting machinery other algorithms use for
the same purpose.

!!! note "On fidelity"
    The published description of MKE leaves several operator details
    open and implementations in circulation differ. What is documented
    here is a coherent reading of the core mechanism — the clone group
    plus a masked difference move — not a claim of exact
    correspondence with the paper. The measurements below describe this
    implementation.

## Equations

**1. Masked difference move.** Both the king and the rest use the same
operator, with only some coordinates altered:

\[
x'_k =
\begin{cases}
x_k + \text{FC}\,(x_{r_1,k} - x_{r_2,k}) & \text{with probability } C \\[4pt]
x_k & \text{otherwise}
\end{cases}
\]

This is the move [Differential Evolution](de.md) is built on. At least
one coordinate always changes.

**2. The clone group.** The king generates `n_clones` independent
trials and keeps the best:

\[
x_{\text{king}} \leftarrow
\arg\min_{c \in \{x_{\text{king}}\} \cup \text{clones}} f(c)
\]

**3. Greedy selection.** Everyone else accepts a trial only if it
improves, so no individual ever worsens.

## Pseudocode

```text
input: individuals n, clones MK, fluctuation FC, change rate C
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    king <- argmin(fitness)

    repeat MK times:                              # the clone group
        clone <- masked difference move from the king            (eq. 1)
        keep it if it beats the best clone so far                (eq. 2)
    the best clone replaces the king if better

    for every other individual i:
        trial <- masked difference move from x[i]                (eq. 1)
        x[i] <- trial if it improves                             (eq. 3)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Number of individuals |
| `n_clones` | MK | 8 | Trials the king gets each iteration |
| `fluctuation` | FC | 0.7 | Scale of the difference vector |
| `change_rate` | \(C\) | 0.1 | Per-coordinate chance of being altered |
| `seed` | — | `None` | Reproducibility |

!!! danger "A lattice resonance, not a result"
    Tuning `fluctuation` on Rastrigin produces a startling and
    **reproducible** pattern (5 seeds, 20,000 evaluations, `C=0.1`):

    | FC | Rastrigin | spread across seeds |
    |---|---|---|
    | 0.5 | **5e-05** | [8e-06, 1e-04] |
    | 0.6 | 2.79 | [1.00, 4.98] |
    | 0.7 | 2.99 | [1.99, 3.98] |
    | 0.9 | **8e-05** | [4e-05, 2e-04] |

    The ranges are tight and disjoint, so this is not seed noise: FC of
    0.5 and 0.9 essentially solve Rastrigin while 0.6 and 0.7 stall
    around 3. With `population_size=15` the effect is stronger still,
    reaching **6e-07**.

    It is an artefact. Rastrigin's local optima sit on an axis-aligned
    lattice of spacing 1, and once the population settles onto that
    lattice the difference \(x_{r_1} - x_{r_2}\) is close to an integer
    vector. Certain FC values then map lattice points onto other lattice
    points, letting the search hop between local optima for free.
    Transforming the problem confirms it:

    | Configuration | plain | **shifted** | **rotated** | rot + shift |
    |---|---|---|---|---|
    | FC=0.5, N=15 | 6e-07 | 2e-06 | **31.3** | 39.0 |
    | FC=0.7, N=30 | 2.99 | 2.80 | **26.0** | 24.6 |

    Note which column breaks it. **Shifting leaves the result intact**
    — a translation moves the lattice but preserves its spacing — while
    **rotating destroys it**, because the lattice is no longer
    axis-aligned. That pattern distinguishes a lattice resonance from
    generic separability dependence or origin bias, and it is why the
    four-variant check is worth running rather than just one
    transformation.

    **The defaults take FC=0.7 deliberately**, giving up seven orders of
    magnitude on the published benchmark for the configuration that is
    better on *both* rotated variants. A default of 0.5 would put a
    spectacular number in the comparison table that means almost
    nothing.

!!! note "Tuning notes"
    - **A low `change_rate` is worth an order of magnitude**: 0.6 → 0.1
      moves Rastrigin from 23.5 to 2.66 and Ackley from 0.42 to 0.004.
      This is the same finding the [CSO](cso.md) (`cdc=0.1`) and
      [DE](de.md) (`CR=0.5`) pages record — changing few coordinates per
      trial suits separable multimodal landscapes — and it carries the
      same caveat about what it costs when coordinates interact.
    - **The clone group earns its place.** With `n_clones=1` the
      algorithm is ordinary DE/rand/1 and reaches Rastrigin 4.18;
      8 clones give 2.66 and 30 give 1.66, at some cost to Sphere and
      Ackley. Concentrating budget on the incumbent genuinely helps.
    - **No origin bias.** Every term is a difference of positions, so
      the search is translation-invariant; there is a test asserting a
      shifted problem shifts the whole trajectory.

## Behavior

MKE reaches Sphere ≈ 4e-08, Rastrigin ≈ 2.99, Ackley ≈ 4e-03 — solid
mid-field, with Rastrigin in the better half of the library.

Its interest is less the numbers than the two things it demonstrates.
The clone group is a clean, cheap way to bias a budget toward the
incumbent without a ranking scheme. And the FC pattern above is the
sharpest example in this library of a benchmark result that is real,
reproducible, tightly distributed across seeds — and still tells you
nothing about the algorithm.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import MonkeyKingEvolution

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = MonkeyKingEvolution(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- Z. Meng and J.-S. Pan, "Monkey King Evolution: a new memetic
  evolutionary algorithm and its application in vehicle fuel
  consumption optimization," *Knowledge-Based Systems*, 97, 144-157,
  2016.
  [doi:10.1016/j.knosys.2016.01.009](https://doi.org/10.1016/j.knosys.2016.01.009).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010 — on why rotated
  and shifted variants belong in any benchmark suite.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
