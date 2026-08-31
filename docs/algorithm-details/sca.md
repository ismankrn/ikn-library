# Sine Cosine Algorithm (SCA)

**SCA** (Mirjalili, 2016) is among the simplest algorithms here and one
of the few built on a mathematical function rather than an animal. Every
solution moves toward the best one found so far, by a displacement
scaled by a **sine or a cosine** chosen at random.

The trigonometric factor is the whole idea: because \(\sin\) and
\(\cos\) both range over \([-1, 1]\), a solution can move toward the
destination, away from it, or past it, and the balance between those is
set by geometry rather than by a tuned probability.

It also produces the most instructive failure in this library.

## Equations

**1. Position update**, with a coin flip choosing the function:

\[
x \leftarrow x + r_1 \sin(r_2)\,\lvert r_3 P - x \rvert
\quad\text{or}\quad
x + r_1 \cos(r_2)\,\lvert r_3 P - x \rvert
\]

where \(P\) is the best solution found so far and

\[
r_2 \sim \mathcal{U}(0, 2\pi),
\qquad
r_3 \sim \mathcal{U}(0, 2)
\]

are drawn per coordinate.

**2. Amplitude decay.** \(r_1\) falls linearly to zero, so excursions
shrink and the population converges:

\[
r_1 = a\left(1 - \frac{\text{evals}}{\text{max\_evals}}\right)
\]

Like [GWO](gwo.md) and [HHO](hho.md), this schedule was tied to the
run's progress in the original paper.

## Pseudocode

```text
input: solutions n, amplitude a
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    P  <- the best solution so far
    r1 <- a * (1 - evals/max_evals)                               (eq. 2)
    for each solution and each coordinate:
        r2 <- U(0, 2*pi) ;  r3 <- U(0, 2)
        swing <- sin(r2) or cos(r2), by a coin flip
        x <- x + r1 * swing * |r3*P - x|                          (eq. 1)
    evaluate the new positions

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Number of solutions |
| `amplitude` | \(a\) | 2.0 | Initial movement amplitude (paper's value) |
| `seed` | — | `None` | Reproducibility |

!!! danger "SCA converges to the origin, not to the optimum"
    SCA's benchmark row looks superb — Sphere ≈ 5e-26, Rastrigin
    ≈ 2e-11, Ackley ≈ 3e-10. Almost all of it comes from where the
    benchmarks put their optimum.

    Look at equation 1. The displacement is proportional to
    \(\lvert r_3 P - x \rvert\). When the destination \(P\) and the
    solution \(x\) are both at the **origin**, that term is identically
    zero **for every draw of \(r_3\)** — so the origin is an exact fixed
    point of the dynamics, reached regardless of the objective function.
    There is a test asserting this.

    Away from the origin nothing of the sort happens. At \(x = P\) the
    displacement is \(\lvert P \rvert \lvert r_3 - 1 \rvert\), which is
    zero only if \(P\) is. The residual step never vanishes, so the
    population cannot settle — and the error scales with how far the
    optimum sits from zero:

    | Optimum shifted by | Sphere result |
    |---|---|
    | 0.0 | **9e-26** |
    | **0.1** | **0.021** |
    | 0.5 | 0.419 |
    | 1.5 | 3.60 |
    | 3.0 | 4.81 |

    Moving the optimum by **0.1** — a fiftieth of the search range —
    costs twenty-four orders of magnitude.

    Tracing a run confirms the mechanism directly. On the standard
    Sphere the population mean reaches \(3.6 \times 10^{-21}\) from the
    origin. On the same function shifted by 1.5, after the full budget
    it is still 2.86 from the optimum: it converges to neither point.

!!! warning "It fails on both transformations, and no setting fixes it"
    | Variant | Rastrigin |
    |---|---|
    | plain | **2e-11** |
    | rotated | 44.6 |
    | shifted | 42.5 |
    | rotated + shifted | 44.5 |

    A factor of \(2 \times 10^{12}\), and the worst four-variant profile
    measured in this library — SCA needs the optimum to be at the origin
    **and** the problem to be axis-aligned. Compare
    [MBO](mbo.md), which fails only on rotation, and [GWO](gwo.md),
    which is biased but survives a shift far better.

    Retuning does not help, because the problem is structural rather
    than a bad parameter choice. On shifted problems the paper defaults,
    `amplitude=0.5`, and `population_size=50` land within noise of each
    other (shifted Sphere 2.99 / 2.24 / 1.60). For scale, **random
    search** over the same 20,000 evaluations gets 11.8 on shifted
    Sphere and 71.1 on shifted Rastrigin, so SCA off the origin is
    beating random sampling by a factor of two to six, and nothing more.

    The published defaults are therefore kept: no alternative is
    meaningfully better where it matters.

## Behavior

Use SCA as a teaching example rather than an optimizer.

It is the sharpest demonstration in this library of a benchmark result
that is entirely real — reproducible, tight across seeds, honestly
measured — and entirely uninformative about the algorithm. The whole
row is an artefact of the standard suite placing every optimum at
\(x = 0\), and the mechanism is short enough to read off the update rule
in one line.

That makes it a useful companion to the [index page](index.md)'s
discussion of benchmark variants: SCA is what the four-variant check is
*for*.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import SineCosineAlgorithm

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = SineCosineAlgorithm(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S. Mirjalili, "SCA: a Sine Cosine Algorithm for solving optimization
  problems," *Knowledge-Based Systems*, 96, 120-133, 2016.
  [doi:10.1016/j.knosys.2015.12.022](https://doi.org/10.1016/j.knosys.2015.12.022).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010 — on why shifted
  and rotated variants belong in any benchmark suite.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
- K. Sörensen, "Metaheuristics — the metaphor exposed," *International
  Transactions in Operational Research*, 22(1), 3-18, 2015.
  [doi:10.1111/itor.12001](https://doi.org/10.1111/itor.12001).
