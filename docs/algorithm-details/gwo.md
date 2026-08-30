# Grey Wolf Optimizer (GWO)

**GWO** (Mirjalili et al., 2014) ranks the pack by fitness and names the
three best wolves **alpha**, **beta** and **delta**. Every other wolf
moves to the average of three estimates of where the prey is — one from
each leader.

Following three leaders instead of one is the whole idea, and it is a
neat piece of design. A single global attractor collapses a swarm onto a
point; three attractors that disagree keep the pack spread across the
region they bracket, so GWO gets diversity maintenance without any
explicit diversity mechanism.

It also belongs to the small group of algorithms here whose step
schedule was **tied to the run's progress in the original paper** — most
of its contemporaries needed that added, as the [SA](sa.md),
[CLONALG](clonalg.md), and [BFO](bfo.md) pages record.

## Equations

**1. Control coefficient.** One scalar governs the whole
explore/exploit balance, falling linearly across the run:

\[
a(t) = a_{\text{start}} - (a_{\text{start}} - a_{\text{end}})
\frac{\text{evals}}{\text{max\_evals}}
\]

**2. Random coefficients.** Drawn fresh per wolf, per leader, per
dimension:

\[
A = 2a\,r_1 - a,
\qquad
C = 2 r_2,
\qquad r_1, r_2 \sim \mathcal{U}(0,1)
\]

\(A\) is the switch: while \(|A| > 1\) a wolf is driven **away** from its
leader (search), and once \(|A| < 1\) it is drawn **toward** it (attack).
Because \(a\) falls from 2 to 0, the pack shifts from exploring to
attacking automatically.

**3. Encircling.** Each leader \(L \in \{\alpha, \beta, \delta\}\)
proposes a position:

\[
D_L = \bigl| C \odot X_L - X \bigr|,
\qquad
X_L' = X_L - A \odot D_L
\]

**4. Hunting.** The wolf takes the mean of the three proposals:

\[
X^{t+1} = \frac{X_\alpha' + X_\beta' + X_\delta'}{3}
\]

There is no greedy acceptance — a wolf moves whether or not the new
position is better.

## Pseudocode

```text
input: wolves n, a_start, a_end
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    alpha, beta, delta <- the three best wolves
    a <- a_start - (a_start - a_end) * evals / max_evals          (eq. 1)

    for each wolf i:
        for each leader L in (alpha, beta, delta):
            A <- 2a*rand() - a ;  C <- 2*rand()                   (eq. 2)
            X_L' <- L - A * |C*L - x[i]|                          (eq. 3)
        x[i] <- mean(X_alpha', X_beta', X_delta')                 (eq. 4)
        evaluate x[i]

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Number of wolves |
| `a_start` | \(a_0\) | 2.0 | Initial control coefficient (paper's value) |
| `a_end` | — | 0.0 | Final control coefficient |
| `seed` | — | `None` | Reproducibility |

!!! danger "GWO is biased toward the origin, and this benchmark suite hides it"
    On the table below GWO reaches Sphere ≈ 2e-88, **Rastrigin exactly
    0 on every seed**, and Ackley at machine precision — tied for the
    best entries in the library. Move the optimum away from the origin
    and those results collapse (3 seeds, 20,000 evaluations):

    | Function | Optimum at origin | Optimum shifted |
    |---|---|---|
    | Sphere | 4e-88 | 8e-06 |
    | Rastrigin | **0** | 15.6 |
    | Ackley | 4e-16 | 0.019 |

    Sphere loses **82 orders of magnitude** from nothing but a
    translation of the problem. Since Sphere, Rastrigin, and Ackley all
    place their optimum at exactly \(x = 0\), the standard suite cannot
    see this at all.

    It is not simply that shifted problems are harder. Running the same
    shift against other algorithms isolates it:

    | Algorithm | Sphere at 0 | Sphere shifted | Ackley at 0 | Ackley shifted |
    |---|---|---|---|---|
    | **GWO** | 4e-88 | 8e-06 | 4e-16 | 0.019 |
    | [FWA](fwa.md) | 6e-89 | 0.13 | 3e-15 | 3.76 |
    | [DE](de.md) | 2e-41 | **0** | 5e-15 | **4e-15** |
    | [FPA](fpa.md) | 3e-30 | 5e-30 | 2e-13 | 4e-14 |

    DE and FPA are essentially **unaffected**; GWO and FWA both
    collapse. The bias is a property of those two algorithms, not of the
    shifted problems.

    The mechanism is visible in equation 3. The update
    \(X_L - A \odot |C \odot X_L - X|\) is not translation-invariant:
    the \(C \odot X_L\) term scales the leader's *absolute coordinates*,
    so the size of the correction depends on how far the leader sits
    from zero, not only on the distances between wolves. A well-behaved
    search operator should depend only on differences. This is a
    documented criticism of GWO in the literature (see Niu et al.).

    **What to do about it:** GWO remains a reasonable general optimizer
    — its shifted Sphere of 8e-06 is mid-field, not broken — but treat
    its headline benchmark numbers as unreliable, and never select GWO
    over another algorithm on the basis of a suite whose optima sit at
    the origin.

!!! note "Tuning notes"
    - **The paper's defaults are kept**, because tuning on the standard
      suite selects for the bias. `population_size=15` looks best there
      (Sphere 1e-32, Rastrigin 0 at only 5,000 evaluations), but on
      shifted problems it is the worst configuration tried
      (Rastrigin+ 34.0, Ackley+ 4.90) against 15.6 and 0.019 for the
      default 30. This is the same trap the [FOA page](foa.md)
      documents.
    - **`a_start` is best left at 2.** Lowering it to 1.0 costs five
      orders of magnitude on shifted Sphere; raising it to 3 is roughly
      neutral.
    - **Larger packs help the multimodal case only.** 120 wolves give
      the best shifted Rastrigin (11.0 vs 15.6) but lose on both other
      functions.

## Behavior

Taken at face value GWO is among the strongest entries here. Taken
honestly, it is a **solid mid-field optimizer with an unusually
flattering benchmark profile**. Its genuine strengths are real enough:
the three-leader average is an elegant diversity mechanism, the pack
demonstrably contracts over the run (there is a test asserting it), and
the budget-tied `a` schedule was correct from the start.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import GreyWolfOptimizer

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = GreyWolfOptimizer(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey wolf optimizer,"
  *Advances in Engineering Software*, 69, 46-61, 2014.
  [doi:10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007).
- Y. Niu, X. Yan, Y. Wang, and Y. Niu, "The defect of the Grey Wolf
  optimization algorithm and its verification method," *Knowledge-Based
  Systems*, 171, 37-43, 2019 — on the origin bias documented above.
  [doi:10.1016/j.knosys.2019.01.018](https://doi.org/10.1016/j.knosys.2019.01.018).
- C. L. Camacho-Villalón, M. Dorigo, and T. Stützle, "Exposing the
  grey wolf, moth-flame, whale, firefly, bat, and antlion algorithms:
  six misleading optimization techniques inspired by bestial
  metaphors," *International Transactions in Operational Research*,
  30(6), 2945-2971, 2023.
  [doi:10.1111/itor.13176](https://doi.org/10.1111/itor.13176).
