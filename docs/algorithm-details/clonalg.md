# Clonal Selection Algorithm (CLONALG)

**CLONALG** (de Castro & Von Zuben, 2002) comes from *artificial immune
systems*, a family with a different ancestry than the swarm and
evolutionary methods that make up the rest of this library. Candidate
solutions are **antibodies**, and their fitness is their *affinity* for
the antigen — the problem being solved.

Its defining idea is a pair of rules that pull in opposite directions
and are both tied to the same quantity, affinity:

| Rule | Direction | Effect on the search |
|---|---|---|
| Cloning ∝ affinity | good antibodies make **more** copies | budget concentrates on what works |
| Hypermutation ∝ 1/affinity | good antibodies mutate **less** | copies refine rather than wander |

A good antibody therefore gets many clones, each a small perturbation
(local refinement), while a poor one gets a single clone flung far away
(exploration). Exploration and exploitation are not balanced by a
schedule or a tuning constant here — they are two readings of the same
biological rule.

## Equations

**1. Cloning by rank (Eq. 1).** After sorting by affinity, the antibody
at rank \(i\) receives

\[
N_c(i) = \max\left(\left\lfloor \beta \cdot \frac{N}{i} \right\rceil,\ 1\right),
\qquad i = 1, \dots, n
\]

clones. The share falls off hyperbolically, so the best antibody gets
several times the clones of the tenth-best.

**2. Hypermutation (Eq. 2).** Normalized affinity \(D \in [0, 1]\) is
taken from the rank (\(D = 1\) for the best, near 0 for the worst), and
the mutation rate falls off exponentially with it:

\[
\alpha(i) = e^{-\rho D_i} \cdot
\left(\max\left(1 - \frac{\text{evals}}{\text{max\_evals}},\ 10^{-6}\right)\right)^{2}
\]

The second factor is a **deviation from the original** — see the tuning
note below. Each clone is then perturbed in every dimension:

\[
x' = x + \alpha(i) \, (u - l) \odot \mathcal{N}(0, I)
\]

**3. Selection.** Parents and clones compete together, and the best
\(N\) survive. Because parents are in that pool, the elite can never
regress.

**4. Receptor editing.** The worst \(d\) antibodies are then discarded
and replaced by uniformly random ones — the immune system's
*metadynamics*, and CLONALG's only source of genuinely new material.

## Pseudocode

```text
input: antibodies N, selected n, clone factor beta, replaced d, decay rho
x <- N random solutions, evaluated, sorted by affinity

repeat until the budget is exhausted:
    Nc <- clones per rank:  round(beta * N / i)                 (eq. 1)
    a  <- mutation rate per rank:  exp(-rho * D) * decay        (eq. 2)

    for each selected antibody i = 1..n:
        repeat Nc[i] times:
            clone <- repair(x[i] + a[i] * (u - l) * N(0, I))
            add clone to the candidate pool, evaluated

    x <- the best N of {parents} U {clones}                     (eq. 3)
    replace the worst d antibodies with random ones             (eq. 4)
    sort x by affinity

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(N\) | 20 | Number of antibodies |
| `n_select` | \(n\) | 10 | How many best antibodies are cloned |
| `clone_factor` | \(\beta\) | 0.5 | Scales the total number of clones |
| `n_replace` | \(d\) | 2 | Worst antibodies replaced each generation |
| `rho` | \(\rho\) | 3.0 | How sharply mutation falls off with affinity |
| `seed` | — | `None` | Reproducibility |

!!! warning "The budget decay is not in the original — and CLONALG needs it"
    In the 2002 formulation the mutation rate depends only on
    **relative** affinity within the current population. That looks
    self-regulating, but it never actually converges: however tightly
    the population clusters, the best antibody still mutates by
    \(e^{-\rho}\) of the *full search range*, because the normalization
    rescales to \([0, 1]\) again every generation. The result is a hard
    precision floor.

    | Mutation rate | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | As published (no decay) | 5e-03 | 54.6 | 1.25 |
    | **× quadratic budget decay** | **3e-08** | **10.4** | **1e-03** |

    Five orders of magnitude on Sphere, from one extra factor. This is
    the same lesson the [SA](sa.md), [CSO](cso.md), and [BFO](bfo.md)
    pages record: pre-2005 algorithms were generally specified without
    tying step sizes to the evaluation budget, and adding that tie is
    usually the single largest improvement available.

!!! note "`rho` trades precision against multimodal performance"
    `rho` is the one parameter that matters, and it does not have a
    best value — it picks a point on a trade-off (5 seeds, 20,000
    evaluations):

    | `rho` | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | 1 | 1e-06 | **6.3** | 6e-03 |
    | 2 | 5e-08 | 9.4 | 3e-03 |
    | **3** | **3e-08** | 10.4 | **1e-03** |
    | 4 | 2e-09 | 44.1 | 3e-04 |

    Low `rho` means even the best antibodies keep mutating hard, which
    is what escapes Rastrigin's local optima but costs final precision.
    The default sits just before the cliff between 3 and 4. Lower it to
    1–2 for rugged multimodal problems; raise it to 4 when you want
    precision on a smooth one.

    The decay exponent behaves the same way: raising it from 2 to 3
    takes Sphere to 3e-11 but pushes Rastrigin to 18.

## Behavior

CLONALG lands in the **middle of the field**: Sphere ≈ 3e-08,
Ackley ≈ 1e-03, Rastrigin ≈ 10.4. It is well clear of
[BFO](bfo.md) on all three, but far behind [DE](de.md),
[Firefly](firefly.md), or [FWA](fwa.md) on the smooth functions, where
those reach 1e-20 and below.

The reason is visible in the equations: every clone is a **Gaussian
perturbation of a single parent**. CLONALG has no crossover, no
difference vector, no attraction between antibodies — nothing that
combines information from two solutions. It is a portfolio of
independent local searches with a shared budget, and its whole
intelligence lies in how that budget is split. That makes it an
unusually clean teaching example of affinity-proportional search, and
it explains why immune-inspired methods are most often used in
*hybrids*, supplying the diversity mechanism to an algorithm that
supplies the recombination.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import ClonalSelectionAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = ClonalSelectionAlgorithm(population_size=20, rho=3.0, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- L. N. de Castro and F. J. Von Zuben, "Learning and optimization using
  the clonal selection principle," *IEEE Transactions on Evolutionary
  Computation*, 6(3), 239-251, 2002.
  [doi:10.1109/TEVC.2002.1011539](https://doi.org/10.1109/TEVC.2002.1011539).
- L. N. de Castro and J. Timmis, *Artificial Immune Systems: A New
  Computational Intelligence Approach*, Springer, 2002.
- E. Ulutas and S. Kulturel-Konak, "A review of clonal selection
  algorithm and its applications," *Artificial Intelligence Review*,
  36(2), 117-138, 2011.
  [doi:10.1007/s10462-011-9206-1](https://doi.org/10.1007/s10462-011-9206-1).
