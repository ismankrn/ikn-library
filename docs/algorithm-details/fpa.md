# Flower Pollination Algorithm (FPA)

**FPA** (Yang, 2012) is the simplest algorithm in this library. Every
flower flips a coin each iteration and applies **one of two rules** —
that is the entire method. No archive, no ageing, no roles, no adaptive
schedule.

Its two rules are worth recognising rather than taking as novel: global
pollination is the Lévy-flight-toward-best move that also drives
[Cuckoo Search](cuckoo.md), and local pollination is a difference vector
of the kind [Differential Evolution](de.md) is built on, minus the
crossover. FPA's real claim is a structural one — that *switching*
between a long-range and a short-range rule, independently per flower
per iteration, is enough.

The measurements below suggest that claim holds, but for a reason the
standard benchmarks cannot show.

## Equations

**1. Global pollination** (probability \(p\)) — pollen carried far by
animals, aimed at the current best flower \(g^*\):

\[
x_i^{t+1} = x_i^t + \gamma \, L(\lambda) \odot \bigl(g^* - x_i^t\bigr)
\]

\(L(\lambda)\) is a Lévy step drawn by Mantegna's method, giving mostly
small moves with rare very long jumps:

\[
L = \frac{u}{|v|^{1/\beta}},
\qquad
u \sim \mathcal{N}(0, \sigma^2),\quad v \sim \mathcal{N}(0, 1)
\]

**2. Local pollination** (probability \(1-p\)) — pollen moving between
neighbours, as a scaled difference of two random flowers:

\[
x_i^{t+1} = x_i^t + \varepsilon \bigl(x_j^t - x_k^t\bigr),
\qquad \varepsilon \sim \mathcal{U}(0,1),\quad j \neq k
\]

**3. Greedy selection.** The new flower is kept only if it is better, so
no flower ever worsens.

## Pseudocode

```text
input: flowers n, switch probability p, gamma, Levy exponent beta
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    g* <- the best flower
    for each flower i:
        if rand() < p:
            x' <- x[i] + gamma * Levy(beta) * (g* - x[i])        (eq. 1)
        else:
            pick j != k at random
            x' <- x[i] + U(0,1) * (x[j] - x[k])                  (eq. 2)
        x' <- repair(x')
        if f(x') < f(x[i]):  x[i] <- x'                          (eq. 3)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 15 | Number of flowers |
| `switch_probability` | \(p\) | 0.5 | Chance of global rather than local pollination |
| `gamma` | \(\gamma\) | 0.5 | Scaling of the global pollination step |
| `levy_exponent` | \(\beta\) | 1.5 | Lévy exponent; lower = heavier tails |
| `seed` | — | `None` | Reproducibility |

!!! warning "The benchmark suite says to delete half the algorithm. It is wrong."
    Ablating the switch probability on the three standard benchmarks
    produces an unambiguous — and misleading — verdict (5 seeds, 20,000
    evaluations):

    | \(p\) | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | 0.0 — local only | 0.220 | 16.07 | 5.797 |
    | 0.5 — **default** | 1e-29 | 6.99 | 2e-13 |
    | 0.8 — Yang's recommendation | 1e-30 | 8.56 | 1e-14 |
    | 1.0 — **global only** | **8e-31** | **5.59** | **7e-15** |

    Read alone, this says local pollination is dead weight: turning it
    off entirely wins on all three functions. Since Sphere, Rastrigin,
    and Ackley are all **separable**, and global pollination pulls each
    flower along its own independent line toward \(g^*\), that result
    is exactly what a separable suite should be expected to reward.

    Adding two non-separable problems reverses it:

    | \(p\) | Rosenbrock | Rastrigin, rotated |
    |---|---|---|
    | 0.5 — **default** | **0.015** | **12.5** |
    | 0.8 | 1.61 | 17.8 |
    | 1.0 — global only | 5.58 | 31.0 |

    Pure global pollination is **360× worse** on Rosenbrock and **2.5×
    worse** on rotated Rastrigin. The reason is visible in equation 1:
    the global rule moves each flower along the single direction
    \(g^* - x_i\), and never combines information *between* two
    non-best flowers. Only the local rule does that, and coupled
    coordinates are precisely where it is needed.

    This is the same trap the [FOA page](foa.md) documents, arriving
    from the opposite direction: there the benchmark flattered a
    coordinate-wise operator, here it hides the value of a
    population-mixing one. `switch_probability=0.5` is the default
    because it is the best compromise **once non-separable problems are
    included** — it gives up an order of magnitude on Sphere to gain
    two on Rosenbrock.

!!! note "Tuning notes"
    - **`p=0.5`, not Yang's 0.8.** The paper's recommendation is close
      but measurably worse here on every non-separable problem tested
      (Rosenbrock 1.61 vs 0.015).
    - **`gamma` matters more than \(\beta\).** Yang's \(\gamma = 0.01\)
      is far too small at this budget (Sphere 8e-04, Ackley 4.3);
      raising it to 0.5 is worth sixteen orders of magnitude on Sphere.
    - **Small populations win.** 15 flowers beat 25 by nine orders of
      magnitude on Sphere and 60 by far more. With a greedy accept and
      a strong pull toward \(g^*\), extra flowers mostly consume budget.

!!! info "A known structural quirk"
    Because the global step is proportional to \(g^* - x_i\), it goes to
    **exactly zero** for any flower that reaches the best position — and
    for the best flower itself, always. On a fully converged population
    global pollination stops moving anything at all (there is a test
    asserting this). Local pollination is what keeps such a population
    alive, which is a second, independent reason not to set \(p = 1\).

## Behavior

FPA is a **strong all-rounder and the best performer measured here on
Rosenbrock**: Sphere ≈ 1e-29, Ackley ≈ 2e-13, Rastrigin ≈ 7.0, and
Rosenbrock ≈ 0.015 — ahead of [DE](de.md) (1.58), [FWA](fwa.md) (7.0),
and [CRO](cro.md) (28.5) on that non-separable valley.

Rastrigin ≈ 7 is only mid-field, so this is not a universal winner. But
the combination of near-machine-precision on smooth functions with the
best Rosenbrock result in the library is unusual, and it comes from an
algorithm that fits in about fifteen lines.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import FlowerPollinationAlgorithm

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = FlowerPollinationAlgorithm(population_size=15, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- X.-S. Yang, "Flower pollination algorithm for global optimization,"
  in *Unconventional Computation and Natural Computation*, LNCS 7445,
  Springer, 240-249, 2012.
  [doi:10.1007/978-3-642-32894-7_27](https://doi.org/10.1007/978-3-642-32894-7_27).
- X.-S. Yang, M. Karamanoglu, and X. He, "Flower pollination algorithm:
  a novel approach for multiobjective optimization," *Engineering
  Optimization*, 46(9), 1222-1237, 2014.
  [doi:10.1080/0305215X.2013.832237](https://doi.org/10.1080/0305215X.2013.832237).
- R. N. Mantegna, "Fast, accurate algorithm for numerical simulation of
  Lévy stable stochastic processes," *Physical Review E*, 49(5),
  4677-4683, 1994.
  [doi:10.1103/PhysRevE.49.4677](https://doi.org/10.1103/PhysRevE.49.4677).
