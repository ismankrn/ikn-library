# Fireworks Algorithm (FWA)

**FWA** (Tan & Zhu, 2010) treats each candidate solution as a firework
that **explodes**, scattering sparks around itself. Two quantities are
set by the firework's quality — and, crucially, they move in *opposite*
directions:

| Quality of the firework | Number of sparks | Explosion amplitude |
|---|---|---|
| good (low fitness) | **many** | **small** — tight, exploitative |
| poor (high fitness) | few | **large** — wide, exploratory |

That inverse coupling is what makes FWA distinctive. Most algorithms
switch between exploration and exploitation over *time*, on a schedule.
FWA does both *simultaneously*, distributed across the population: good
fireworks polish their neighbourhood while bad ones scout distant
regions, and the balance re-adjusts itself every iteration.

## Equations

**1. Number of sparks.** With \(y_{\max}\) the worst fitness in the
population:

\[
s_i = \hat{m} \cdot
\frac{y_{\max} - f(x_i) + \varepsilon}
     {\sum_{j}\bigl(y_{\max} - f(x_j)\bigr) + \varepsilon}
\]

clipped to \([a\hat{m},\, b\hat{m}]\) so that no firework starves or
monopolizes the budget.

**2. Explosion amplitude.** With \(y_{\min}\) the best fitness — note
that the numerator is inverted relative to equation 1:

\[
A_i = \hat{A} \cdot
\frac{f(x_i) - y_{\min} + \varepsilon}
     {\sum_{j}\bigl(f(x_j) - y_{\min}\bigr) + \varepsilon}
\]

**3. Explosion sparks.** For a random half of the dimensions:

\[
x_{i,d}^{\text{spark}} = x_{i,d} + A_i \cdot \mathcal{U}(-1, 1)
\]

**4. Gaussian sparks.** A second, multiplicative mutation applied to a
few randomly chosen fireworks:

\[
x_{i,d}^{\text{spark}} = x_{i,d} \cdot g,
\qquad g \sim \mathcal{N}(1, 1)
\]

Because it *scales* rather than shifts, this operator can cross orders
of magnitude in one step — which is why it matters so much (see below).

**5. Selection.** The best solution among fireworks and all sparks is
always kept; the remaining slots are filled at random. This is the
"elitism-random" scheme of the enhanced FWA, which avoids the original
formulation's costly pairwise-distance roulette.

## Pseudocode

```text
input: fireworks n, total sparks m, max amplitude A, Gaussian sparks g
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    s <- sparks per firework, more for the better ones          (eq. 1)
    A <- amplitude per firework, smaller for the better ones    (eq. 2)

    candidates <- fireworks
    for each firework i, s[i] times:
        candidates += explosion spark around x[i] with amplitude A[i]
                                                                (eq. 3)
    for g times:
        candidates += Gaussian spark around a random firework   (eq. 4)

    x <- the best candidate, plus n-1 random others             (eq. 5)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 5 | Number of fireworks; deliberately small |
| `n_sparks` | \(\hat{m}\) | 10 | Sparks shared out each iteration |
| `max_amplitude` | \(\hat{A}\) | 0.5 | Largest amplitude (fraction of the bound range) |
| `n_gaussian_sparks` | \(g\) | 5 | Gaussian mutation sparks per iteration |
| `spark_bounds` | \((a, b)\) | (0.04, 0.8) | Limits on each firework's share of the sparks |
| `seed` | — | `None` | Reproducibility |

!!! note "Two things the benchmarking revealed"
    - **A small spark budget wins.** Cutting `n_sparks` from 50 to 10
      improved Sphere from ≈ 1e-58 to ≈ 6e-89. Fewer sparks per
      iteration means *many more iterations* for the same evaluation
      budget, and FWA's selection step is where progress is locked in.
    - **Gaussian sparks are essential, not decoration.** Setting
      `n_gaussian_sparks=0` collapses the algorithm entirely: Sphere
      goes from ≈ 1e-88 to ≈ 1.2, Rastrigin from 0 to 30.8. Explosion
      sparks are *additive* and cannot easily reach a different order of
      magnitude; the multiplicative Gaussian spark can, and it supplies
      the fine-scale refinement the additive operator lacks. There is a
      test pinning this behaviour.

## Behavior

FWA is the **strongest algorithm in this library by a wide margin**:

| Function | FWA | Runner-up |
|---|---|---|
| Sphere | **3e-88** | 7e-57 (Firefly) |
| Ackley | **2e-15** | 2e-15 (Firefly) |
| Rastrigin | **0** (exact) | 2e-08 (ABC) |

Rastrigin reaching **exactly zero in all seven test seeds** is the
headline: FWA locates the global optimum of a 10-dimensional,
massively multimodal function precisely, not approximately. The
combination of self-balancing amplitudes and multiplicative mutation
appears to suit these benchmarks unusually well.

Treat the ranking with the usual caution, though: three benchmark
functions are not a proof of general superiority, and the no-free-lunch
theorem still applies. Benchmark on a problem resembling yours.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import FireworksAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = FireworksAlgorithm(population_size=5, n_sparks=10, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- Y. Tan and Y. Zhu, "Fireworks algorithm for optimization," in
  *Advances in Swarm Intelligence (ICSI 2010)*, Lecture Notes in
  Computer Science 6145, Springer, 355-364, 2010.
  [doi:10.1007/978-3-642-13495-1_44](https://doi.org/10.1007/978-3-642-13495-1_44).
- S. Zheng, A. Janecek, and Y. Tan, "Enhanced fireworks algorithm," in
  *IEEE Congress on Evolutionary Computation (CEC 2013)*, 2069-2077,
  2013 (the elitism-random selection used in equation 5).
  [doi:10.1109/CEC.2013.6557813](https://doi.org/10.1109/CEC.2013.6557813).
- Y. Tan, *Fireworks Algorithm: A Novel Swarm Intelligence Optimization
  Method*, Springer, 2015.
