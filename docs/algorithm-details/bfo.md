# Bacterial Foraging Optimization (BFO)

**BFO** (Passino, 2002) models how *E. coli* hunts for nutrients. Its
distinguishing feature among the algorithms here is that it runs three
mechanisms on **three different timescales**, from every iteration down
to once every hundred.

| Mechanism | Timescale | Biological analogue |
|---|---|---|
| Chemotaxis | every iteration | tumbling and swimming toward nutrients |
| Reproduction | every ~20 iterations | healthy bacteria split, weak ones die |
| Elimination-dispersal | every ~100 iterations | the environment changes suddenly |

## Equations

**1. Chemotaxis: tumble then swim.** A bacterium picks a random unit
direction \(\phi\) (the *tumble*) and then keeps stepping that same way
while conditions improve (the *swim*), up to `n_swim` steps:

\[
x_i \leftarrow x_i + C^{(t)} \, \phi,
\qquad
\phi = \frac{\Delta}{\lVert \Delta \rVert},
\quad \Delta \sim \mathcal{N}(0, I)
\]

The swim stops as soon as the fitness fails to improve. This makes
chemotaxis a small **directional line search**, not an isolated random
step — the feature that separates BFO from most of its contemporaries.

**2. Step decay.** The step shrinks quadratically with the spent
budget, so late chemotaxis becomes a fine local search:

\[
C^{(t)} = C_0 \left(\max\left(1 - \frac{\text{evals}}{\text{max\_evals}},\ 10^{-4}\right)\right)^{2}
\]

**3. Health and reproduction.** A bacterium's **health** is the fitness
accumulated over its whole lifetime, not its current value:

\[
H_i = \sum_{t} f\bigl(x_i^{(t)}\bigr)
\]

Every `reproduction_interval` iterations, the healthiest half splits in
two and the weakest half dies. Note what this rewards: *consistency*.
A bacterium that has been good throughout survives even if it currently
sits somewhere mediocre, while one that stumbled into a good spot at
the last moment does not.

**4. Elimination-dispersal.** Every `elimination_interval` iterations,
each bacterium is destroyed and re-placed uniformly at random with
probability \(P_{ed}\) — the algorithm's escape mechanism.

## Pseudocode

```text
input: bacteria n, step C0, swim length Ns, intervals, dispersal prob
x <- n random solutions, evaluated;  H <- fitness

repeat until the budget is exhausted:
    C <- C0 * (1 - evals / max_evals)^2                          (eq. 2)
    for each bacterium i:                        # chemotaxis
        phi <- a random unit direction
        repeat up to Ns times:                                   (eq. 1)
            step along phi; stop as soon as it does not improve
        H[i] <- H[i] + f(x[i])                                   (eq. 3)

    every reproduction_interval iterations:       # reproduction
        keep the healthiest half, duplicated;  reset H           (eq. 3)

    every elimination_interval iterations:        # dispersal
        each bacterium is re-placed at random with prob. P_ed    (eq. 4)

return best solution found
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 15 | Number of bacteria |
| `step_size` | 0.3 | Initial chemotaxis step (fraction of the bound range) |
| `n_swim` | 4 | Maximum consecutive swim steps in one direction |
| `reproduction_interval` | 20 | Iterations between reproductions |
| `elimination_interval` | 100 | Iterations between dispersal events |
| `elimination_prob` | 0.25 | Chance a bacterium is dispersed at such an event |
| `seed` | `None` | Reproducibility |

!!! note "Tuning notes"
    - **The step decay is not in the original.** With a fixed step, BFO
      stalls at Sphere ≈ 1e-04; the quadratic decay brings it to
      ≈ 4e-06 with a large initial step, and to ≈ 3e-09 with a small
      one. The 2002 formulation predates the convention of tying step
      sizes to the evaluation budget.
    - **Large steps help the multimodal case, small ones help
      precision.** `step_size=0.3` gives Rastrigin ≈ 8.6 but Sphere
      ≈ 4e-06; `step_size=0.1` gives Sphere ≈ 2e-07 but Rastrigin ≈ 30.
      The default favours the harder multimodal case.

## Behavior

BFO is the **weakest algorithm in this library** on these benchmarks:
Sphere ≈ 4e-06, Ackley ≈ 0.02, Rastrigin ≈ 19 — several orders of
magnitude behind DE, Firefly, or FWA on the smooth functions.

That is worth stating plainly rather than hiding. BFO dates from 2002
and is widely reported in the literature as being outperformed by
later methods; its reproduction step in particular **halves the
population's diversity** every time it fires, duplicating survivors
instead of generating new material. It remains valuable as a study in
multi-timescale design and as a baseline, and its swim operator — a
directional line search — is an idea worth borrowing.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import BacterialForagingOptimization

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = BacterialForagingOptimization(population_size=15, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- K. M. Passino, "Biomimicry of bacterial foraging for distributed
  optimization and control," *IEEE Control Systems Magazine*, 22(3),
  52-67, 2002.
  [doi:10.1109/MCS.2002.1004010](https://doi.org/10.1109/MCS.2002.1004010).
- S. Das, A. Biswas, S. Dasgupta, and A. Abraham, "Bacterial foraging
  optimization algorithm: theoretical foundations, analysis, and
  applications," in *Foundations of Computational Intelligence
  Volume 3*, Springer, 23-55, 2009.
  [doi:10.1007/978-3-642-01085-9_2](https://doi.org/10.1007/978-3-642-01085-9_2).
