# Moth-Flame Optimization (MFO)

**MFO** (Mirjalili, 2015) is built on *transverse orientation*: a moth
navigates by holding a fixed angle to a distant light, which works for
the moon but spirals it into a nearby flame.

The population is kept in two arrays. **Moths** are the search agents;
**flames** are the best positions found so far, refreshed each iteration
by merging both sets and keeping the best. Each moth then spirals around
**its own** flame — moth *i* around flame *i* — rather than around a
single shared best.

The exploration schedule is unusual: it is not a step size but the
**flame count**, which shrinks from *N* to 1 across the run. Early on
the population is pulled toward many different attractors; by the end
every moth targets the same one.

## Equations

**1. Logarithmic spiral.** Each moth moves relative to its flame:

\[
M_i = D_i \, e^{bt} \cos(2\pi t) + F_j,
\qquad
D_i = \lvert F_j - M_i \rvert
\]

The cosine makes the moth circle the flame; \(e^{bt}\) sets how tightly.

**2. Spiral parameter.** \(t\) is drawn per coordinate from \([a, 1]\),
so the moth can land inside, beyond, or on either side of the flame:

\[
t = (a - 1) r + 1, \qquad r \sim \mathcal{U}(0,1)
\]

**3. Convergence schedule.** \(a\) falls from -1 to -2, and the flame
count falls from \(N\) to 1:

\[
a = a_{\text{start}} + (a_{\text{end}} - a_{\text{start}})
\frac{\text{evals}}{\text{max\_evals}}
\]
\[
n_{\text{flames}} = \operatorname{round}\!\left(N - \frac{\text{evals}}{\text{max\_evals}}(N - 1)\right)
\]

**4. Flame assignment.** Moth \(i\) uses flame \(i\) while flames last;
once the count has shrunk, the surplus moths all share the last one:

\[
j = \min(i,\ n_{\text{flames}} - 1)
\]

## Pseudocode

```text
input: moths n, spiral constant b, a_start, a_end
x <- n random solutions, evaluated;  flames <- x sorted

repeat until the budget is exhausted:
    flames <- best n of {flames} U {moths}, sorted
    a <- a_start + (a_end - a_start) * evals/max_evals             (eq. 3)
    n_flames <- round(n - (evals/max_evals) * (n - 1))             (eq. 3)

    for each moth i:
        F <- flames[min(i, n_flames - 1)]                          (eq. 4)
        D <- |F - x[i]|
        t <- (a - 1) * U(0,1) + 1        # per coordinate          (eq. 2)
        x[i] <- repair(D * exp(b*t) * cos(2*pi*t) + F)             (eq. 1)
        evaluate x[i]

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(N\) | 50 | Moths, and the starting flame count |
| `spiral_constant` | \(b\) | 1.0 | Shape of the logarithmic spiral |
| `a_start` | — | -1.0 | Initial lower bound of \(t\) |
| `a_end` | — | -2.0 | Final lower bound of \(t\) |
| `seed` | — | `None` | Reproducibility |

!!! warning "The metaphor has been challenged in the literature"
    Moth-Flame Optimization is one of six algorithms examined by
    Camacho-Villalón, Dorigo and Stützle (2023), who argue that its
    components correspond to established techniques and that the moth
    imagery obscures rather than explains what the algorithm does. The
    same paper covers the [Grey Wolf](gwo.md), [Firefly](firefly.md),
    and [Bat](bat.md) algorithms, all of which are in this library.

    That is a claim about **novelty**, not about correctness: MFO works,
    and the measurements below are what they are. It does mean you
    should compare it against a conventional baseline rather than
    treating it as a distinct paradigm — the same conclusion the
    [Harmony Search page](hs.md) reaches from Weyland's analysis.

!!! note "Robustness: no origin bias, moderate rotation sensitivity"
    The spiral is built on \(\lvert F - M \rvert\), a **difference**, so
    the update is translation-equivariant and shifting the optimum costs
    nothing (20,000 evaluations):

    | Variant | Rastrigin | | Sphere |
    |---|---|---|---|
    | plain | 13.5 | | 2e-17 |
    | shifted | 13.9 | | 1e-16 |
    | rotated | 24.1 | | — |
    | rotated + shifted | 33.5 | | — |

    There is a test asserting the translation invariance directly.
    Rotation costs a factor of about 1.8 — worse than [KH](kh.md) (1.0)
    or [HHO](hho.md), far better than [MBO](mbo.md) (200,000) or
    [HS](hs.md) (1,500). The per-coordinate draw of \(t\) is what
    introduces the mild axis alignment; the move toward the flame itself
    is not coordinate-wise.

!!! note "Tuning notes"
    - **Population size dominates, and short budgets mislead.** At 5,000
      evaluations `population_size=15` gives the best Sphere (7e-10) and
      the worst Rastrigin (39.3); at 20,000 the picture is different
      again and 50 is clearly the best compromise. The flame count is
      tied to \(N\), so changing it changes the whole schedule.
    - **Rastrigin is unreliable.** Across 5 seeds at the default
      settings the result ranges over `[7.96, 25.9]`, a factor of three.
      The 3-seed figure in the comparison table (9.29) is at the
      optimistic end of that; the 5-seed mean is 13.5.
    - **`spiral_constant` and the `a` schedule matter little** — varying
      \(b\) between 0.5 and 2 moves results by less than the seed noise.

## Behavior

MFO reaches Sphere ≈ 3e-16 and Ackley ≈ 1e-07, both respectable, and
Rastrigin ≈ 13.5, which puts it in the bottom third of the library on
the multimodal function.

The shrinking flame count is a genuinely neat idea — it makes the
exploration/exploitation transition structural rather than a tuned
constant, and it degrades gracefully because a moth always has *some*
flame to work with. The weakness is the same thing viewed differently:
once the count reaches one, MFO is a single-attractor spiral search with
no diversity mechanism left, which is where the multimodal result is
lost.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import MothFlameOptimization

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = MothFlameOptimization(population_size=50, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S. Mirjalili, "Moth-flame optimization algorithm: a novel
  nature-inspired heuristic paradigm," *Knowledge-Based Systems*, 89,
  228-249, 2015.
  [doi:10.1016/j.knosys.2015.07.006](https://doi.org/10.1016/j.knosys.2015.07.006).
- C. L. Camacho-Villalón, M. Dorigo, and T. Stützle, "Exposing the grey
  wolf, moth-flame, whale, firefly, bat, and antlion algorithms: six
  misleading optimization techniques inspired by bestial metaphors,"
  *International Transactions in Operational Research*, 30(6),
  2945-2971, 2023.
  [doi:10.1111/itor.13176](https://doi.org/10.1111/itor.13176).
