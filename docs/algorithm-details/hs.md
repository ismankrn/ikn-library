# Harmony Search (HS)

**HS** (Geem et al., 2001) keeps a **harmony memory** of the best
solutions found and improvises **one new solution per iteration**, which
replaces the worst harmony if it beats it.

Its one structurally distinctive feature is how that solution is built.
Each decision variable is drawn **independently, from a different
randomly chosen harmony**. A single improvisation might take coordinate
1 from harmony 7, coordinate 2 from harmony 2, and coordinate 3 from a
fresh random draw. Every other recombination operator in this library
mixes exactly two parents; Harmony Search mixes across the entire memory
at once.

## Equations

**1. Improvisation.** For each variable \(j\) independently:

\[
x_j =
\begin{cases}
x_j^{(r)}, \quad r \sim \mathcal{U}\{1,\dots,\text{HMS}\}
  & \text{with probability } \text{HMCR} \\[4pt]
\mathcal{U}(l_j, u_j) & \text{otherwise}
\end{cases}
\]

Note that \(r\) is redrawn **per variable**, not once per harmony.

**2. Pitch adjustment.** A variable taken from memory is then nudged
with probability PAR:

\[
x_j \leftarrow x_j + \mathcal{U}(-1, 1)\, \text{bw}^{(t)}_j
\]

**3. Bandwidth decay.** The nudge shrinks with the spent budget:

\[
\text{bw}^{(t)} = \text{bw}_0 (u - l)
\left(\max\left(1 - \tfrac{\text{evals}}{\text{max\_evals}},\ 10^{-6}\right)\right)^{2}
\]

**4. Memory update.** The new harmony replaces the worst one if better,
so memory quality never degrades.

## Pseudocode

```text
input: memory size HMS, HMCR, PAR, bandwidth bw0
HM <- HMS random solutions, evaluated

repeat until the budget is exhausted:
    bw <- bw0 * (u - l) * (1 - evals/max_evals)^2                (eq. 3)
    for each variable j:
        if rand() < HMCR:
            x[j] <- HM[random row][j]        # a different row each time (eq. 1)
            if rand() < PAR:
                x[j] <- x[j] + U(-1,1) * bw[j]                   (eq. 2)
        else:
            x[j] <- U(lower[j], upper[j])
    evaluate x
    if f(x) < worst fitness in HM:  replace the worst harmony     (eq. 4)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | HMS | 30 | Harmony memory size |
| `hmcr` | HMCR | 0.95 | Chance a variable comes from memory |
| `par` | PAR | 0.6 | Chance a remembered variable is nudged |
| `bandwidth` | bw | 0.1 | Nudge size (fraction of the bound range) |
| `seed` | — | `None` | Reproducibility |

!!! danger "Harmony Search is not a new algorithm"
    In 2010 Weyland showed that Harmony Search is a **special case of
    Evolution Strategies** — specifically, that it is equivalent to a
    \((\mu + 1)\) ES with discrete recombination, an approach published
    decades earlier. The musical metaphor renames existing components:
    the harmony memory is the parent population, improvisation is
    discrete recombination, pitch adjustment is mutation, and the
    replace-the-worst rule is \((\mu + 1)\) selection.

    This matters here for two reasons. It is a well-known case study in
    how a metaphor can obscure that a method is already known — the
    reason later authors (Sörensen; Camacho-Villalón et al.) have argued
    for describing metaheuristics in standard optimization terms rather
    than through novel imagery. And practically, it means you should
    **compare HS against an ES or a GA baseline**, not treat it as an
    independent alternative.

    None of this makes the implementation wrong or the algorithm
    useless; it performs respectably below. It does mean the *novelty*
    claim, not the method, is what fails.

!!! warning "Strong on separable problems, weak once coordinates couple"
    HS builds each variable independently (eq. 1), which is close to
    ideal for *separable* functions — and all three benchmarks here are
    separable. Rotating Rastrigin so its coordinates couple exposes the
    limit (3 seeds, 20,000 evaluations):

    | Memory size | Rastrigin | Rastrigin, rotated | Rosenbrock |
    |---|---|---|---|
    | **30 (default)** | 0.018 | **26.3** | **5.53** |
    | 60 | **1e-04** | 28.2 | 14.6 |
    | 100 | 5e-04 | 26.8 | 7.38 |

    `HMS=60` is **140× better** on the published benchmark and clearly
    worse on both non-separable problems. The default takes the weaker
    headline number, for the same reason the [FOA page](foa.md) gives.

    HS shows **no origin bias**, unlike [GWO](gwo.md) — shifting the
    optimum leaves Sphere essentially unchanged (5e-08 against 6e-08),
    because values are copied from memory rather than scaled by absolute
    coordinates.

!!! note "Tuning notes"
    - **`hmcr` below ~0.9 is fatal.** At 0.7 a third of every solution
      is random noise: Sphere 0.25, Ackley 5.45. The published 0.9 works;
      0.95 is better.
    - **`par` should be higher than the classic 0.3.** Raising it to 0.6
      moves Rastrigin from 2.28 to 1.0 at the screening budget, since
      pitch adjustment is the only operator that produces values not
      already in memory.
    - **The bandwidth decay is an addition.** The 2001 formulation uses
      a fixed `bw`; tying it to the budget is the same fix the
      [SA](sa.md), [CLONALG](clonalg.md), and [CRO](cro.md) pages
      document.

## Behavior

HS reaches Sphere ≈ 6e-08, Rastrigin ≈ 0.21, Ackley ≈ 2e-03 — a solid
mid-field result, with Rastrigin among the better scores in the library.
Read with the two caveats above: much of the Rastrigin strength is
separability, and the method is an evolution strategy under another
name.

Its genuinely interesting property is the whole-memory recombination.
Mixing coordinates from many solutions at once is a real alternative to
two-parent crossover, and it is worth studying on that basis rather than
for the metaphor.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import HarmonySearch

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = HarmonySearch(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- Z. W. Geem, J. H. Kim, and G. V. Loganathan, "A new heuristic
  optimization algorithm: harmony search," *Simulation*, 76(2), 60-68,
  2001.
  [doi:10.1177/003754970107600201](https://doi.org/10.1177/003754970107600201).
- D. Weyland, "A rigorous analysis of the harmony search algorithm: how
  the research community can be misled by a 'novel' methodology,"
  *International Journal of Applied Metaheuristic Computing*, 1(2),
  50-60, 2010.
  [doi:10.4018/jamc.2010040104](https://doi.org/10.4018/jamc.2010040104).
- K. Sörensen, "Metaheuristics — the metaphor exposed," *International
  Transactions in Operational Research*, 22(1), 3-18, 2015.
  [doi:10.1111/itor.12001](https://doi.org/10.1111/itor.12001).
- M. Mahdavi, M. Fesanghary, and E. Damangir, "An improved harmony
  search algorithm for solving optimization problems," *Applied
  Mathematics and Computation*, 188(2), 1567-1579, 2007 — the origin of
  the dynamic PAR and bandwidth used here.
  [doi:10.1016/j.amc.2006.11.033](https://doi.org/10.1016/j.amc.2006.11.033).
