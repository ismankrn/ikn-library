# Grasshopper Optimisation Algorithm (GOA)

**GOA** (Saremi, Mirjalili & Lewis, 2017) is built around one idea
nothing else in this library has: an explicit **short-range repulsion**.

Every pair of grasshoppers exerts a force on the other that is
*repulsive* when they are close, *attractive* at medium range, and
negligible far away. The separation where it changes sign is the swarm's
**comfort zone** — grasshoppers neither approach nor retreat there.
Everywhere else in this library, spacing is maintained by randomness on
top of pure attraction; here it comes from a force law.

## Equations

**1. Social force.** With \(f\) the attraction intensity and \(l\) the
attractive length scale:

\[
s(r) = f\,e^{-r/l} - e^{-r}
\]

The two exponentials decay at different rates, so \(s\) is negative for
small \(r\) and positive further out. Distances are rescaled into
\([1, 4]\) before the law is applied, as the authors prescribe.

**2. Position update.** The social forces plus the **target** — the best
solution found so far:

\[
x_i \leftarrow c \sum_{j \neq i} c\,\frac{u - l}{2}\,
s(\hat{d}_{ij})\,\frac{x_j - x_i}{d_{ij}} \;+\; T
\]

A grasshopper's own position enters only through the differences
\(x_j - x_i\); it is otherwise *replaced*, not moved.

**3. The shrinking coefficient**, appearing twice in equation 2 — once
damping the whole social term, once shrinking the comfort zone:

\[
c = c_{\max} - (c_{\max} - c_{\min})\frac{\text{evals}}{\text{max\_evals}}
\]

As \(c \to 0\) the social term vanishes and every grasshopper lands on
\(T\).

## Pseudocode

```text
input: grasshoppers n, c_max, c_min, intensity f, length scale l
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    T <- the best solution so far
    c <- c_max - (c_max - c_min) * evals/max_evals                (eq. 3)

    d      <- pairwise distances, rescaled into [1, 4]
    forces <- s(d) = f*exp(-d/l) - exp(-d)                        (eq. 1)
    x <- c * sum_j [ c * (u-l)/2 * forces * unit(x_j - x_i) ] + T (eq. 2)
    evaluate the new positions

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Paper | Effect |
|---|---|---|---|---|
| `population_size` | \(n\) | 30 | 30 | Number of grasshoppers |
| `c_max` | \(c_{\max}\) | 1.0 | 1.0 | Initial shrinking coefficient |
| `c_min` | \(c_{\min}\) | 4e-05 | 4e-05 | Final shrinking coefficient |
| `intensity` | \(f\) | 0.6 | 0.5 | Attraction intensity |
| `attraction_length` | \(l\) | 1.5 | 1.5 | Attractive length scale |
| `seed` | — | `None` | — | Reproducibility |

!!! danger "`intensity` and `attraction_length` decide whether the comfort zone exists at all"
    These two parameters do not tune the repulsion's strength — they
    determine whether it happens. Distances are rescaled into
    \([1, 4]\), so if \(s(r)\) does not cross zero inside that interval,
    one of the two forces never fires:

    | \(f\) | \(l\) | zero crossing | Sphere | Rastrigin | Ackley |
    |---|---|---|---|---|---|
    | 0.5 | 1.5 | 2.08 | 4e-08 | 29.1 | 1e-03 |
    | **0.6** | **1.5** | **1.53** | **2e-08** | **16.5** | **9e-04** |
    | 0.8 | 2.5 | *none* — all attraction | 8e-08 | 18.3 | 6e-03 |
    | 0.5 | 1.2 | *none* — all attraction | 2e-09 | 30.2 | 4e-04 |

    Tuning this naively is a trap, and one worth reporting: an earlier
    pass here settled on \(f = 0.8, l = 2.5\) purely on the numbers,
    which silently **switched the repulsion off entirely** — turning GOA
    into an attraction-only method while still calling it GOA. Checking
    what the parameters *meant*, rather than only what they scored,
    found \(f = 0.6\) instead: it keeps a genuine comfort zone at
    \(r = 1.53\) **and** beats the degenerate setting on all three
    benchmarks.

    So the repulsion earns its place. The paper's \(f = 0.5\) is not
    wrong, merely mistuned — its comfort zone at 2.08 is too wide.
    There is a test asserting the defaults keep the crossing inside
    \([1, 4]\), and another asserting that \(f = 0.8, l = 2.5\) does not.

!!! warning "GOA is deterministic after initialisation"
    The published update rule contains **no random term at all**. Two
    runs from the same starting population produce identical
    trajectories regardless of seed — there is a test asserting exactly
    that. GOA's only randomness is where its grasshoppers start.

    Combined with equation 3, that has a consequence. Tracing a run, the
    mean distance from a grasshopper to the target falls from 2.5 to
    0.002: the swarm collapses onto whatever the best solution is, and
    with a deterministic update there is nothing left to escape with.
    **GOA gets one shot at finding the right basin, from its initial
    sample.**

    That prediction is testable, and it holds: raising the population
    from 30 to 100 improves Rastrigin from 19.6 to 13.6, because more
    initial samples mean a better chance of starting in the right place.
    Past that it reverses (400 grasshoppers give 39.1) as the fixed
    budget leaves too few iterations to converge.

!!! success "No origin bias — the odd one out in its family"
    The library now holds five algorithms from the same group, and GOA
    is the only one that survives a shift intact:

    | Variant | GOA | [WOA](woa.md) | [GWO](gwo.md) | [SCA](sca.md) |
    |---|---|---|---|---|
    | plain | 16.5 | **0** | **0** | 2e-11 |
    | rotated | 17.7 | 6.01 | 11.5 | 44.6 |
    | shifted | 22.5 | 13.9 | 15.1 | 42.5 |
    | rot + shift | **13.3** | 20.7 | 25.8 | 44.5 |

    Equation 2 is the reason: the social term uses only differences and
    the target is added unmodified, so the whole update is
    translation-equivariant. A test asserts it directly. GOA's scores
    vary by less than a factor of 1.7 across all four variants — the
    third-flattest profile in this library, after
    [Krill Herd](kh.md) and the [Hybrid Bat Algorithm](hybrid-bat.md).

## Behavior

GOA reaches Sphere ≈ 2e-08, Rastrigin ≈ 16.5, Ackley ≈ 9e-04. In
absolute terms that puts it in the bottom third on the multimodal
function, and the reason is structural rather than a tuning failure:
a deterministic update plus a collapsing swarm cannot recover from a
bad start.

What makes it worth reading is the contrast with its siblings. Four
algorithms from the same group score spectacularly on the plain
benchmarks and lose most of it under transformation; GOA scores
modestly and keeps what it scores. On a problem that is not centred on
the origin, its row is worth more than theirs.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import GrasshopperOptimizationAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = GrasshopperOptimizationAlgorithm(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S. Saremi, S. Mirjalili, and A. Lewis, "Grasshopper Optimisation
  Algorithm: theory and application," *Advances in Engineering
  Software*, 105, 30-47, 2017.
  [doi:10.1016/j.advengsoft.2017.01.004](https://doi.org/10.1016/j.advengsoft.2017.01.004).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
