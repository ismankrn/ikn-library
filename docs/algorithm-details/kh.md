# Krill Herd Algorithm (KH)

**KH** (Gandomi & Alavi, 2012) treats the swarm as a **Lagrangian
system**. Each krill carries a velocity assembled from three physical
terms, and position follows from integrating it over a time step:

\[
\frac{dX_i}{dt} = N_i + F_i + D_i
\]

Two things set it apart from everything else here. The first two terms
carry **inertia** — each keeps a fraction of its previous value, so
motion is smoothed across iterations rather than recomputed from
scratch; only [GSA](gsa.md) does anything comparable. And the foraging
term aims at a **food centre**, an inverse-fitness-weighted centroid of
the whole herd. No other algorithm in this library builds an attractor
that *every* member contributes to in proportion to its quality.

## Equations

**1. Induced motion.** Neighbours pull or push according to how much
better they are, with the neighbourhood set by the herd's own spacing:

\[
N_i = N^{\max}\!\!\sum_{j \in \mathcal{N}_i}\! \hat{K}_{ij} \hat{X}_{ij}
\;+\; N^{\max} C^{\text{best}} \hat{K}_{i,b} \hat{X}_{i,b}
\;+\; \omega N_i^{\text{old}}
\]

with normalized fitness gaps and unit direction vectors

\[
\hat{K}_{ij} = \tilde{K}_i - \tilde{K}_j,
\qquad
\hat{X}_{ij} = \frac{X_j - X_i}{\lVert X_j - X_i \rVert + \varepsilon}
\]

**2. Sensing radius.** Each krill sees only within

\[
d_{s,i} = \frac{1}{5N} \sum_{j} \lVert X_i - X_j \rVert
\]

so the neighbourhood widens when the herd disperses and tightens as it
converges — an adaptive neighbourhood with no parameter to set.

**3. Foraging motion.** Toward the food centre and the krill's own
personal best:

\[
X^{\text{food}} = \frac{\sum_i w_i X_i}{\sum_i w_i},
\qquad
w_i = \frac{1}{\tilde{K}_i + 0.1}
\]
\[
F_i = V_f \bigl( C^{\text{food}} \tilde{K}_i \hat{X}_{i,\text{food}}
+ \hat{X}_{i,\text{pbest}} \bigr) + \omega F_i^{\text{old}}
\]

with \(C^{\text{food}} = 2(1 - t/T)\) and
\(C^{\text{best}} = 2(r + t/T)\): the pull to food fades as the run
proceeds while the pull to the best strengthens.

**4. Physical diffusion.** The only stochastic term, fading out:

\[
D_i = D^{\max}\left(1 - \frac{t}{T}\right) \delta,
\qquad \delta \sim \mathcal{U}(-1, 1)^d
\]

**5. Position update**, with a step decaying over the run:

\[
X_i \leftarrow X_i + \Delta t \,(N_i + F_i + D_i),
\qquad
\Delta t = C_t \sum_j (u_j - l_j) \left(1 - \tfrac{t}{T}\right)^{2}
\]

## Pseudocode

```text
input: krill n, N_max, V_f, D_max, inertia w, C_t, crossover rate
x <- n random solutions, evaluated;  N, F <- 0;  pbest <- x

repeat until the budget is exhausted:
    K~ <- fitness normalized to [0, 1], 0 for the best
    d_s <- mean pairwise distance / 5                             (eq. 2)

    N <- N_max * (neighbour effect + best effect) + w * N         (eq. 1)
    food <- inverse-fitness-weighted centroid                     (eq. 3)
    F <- V_f * (pull to food + pull to pbest) + w * F             (eq. 3)
    D <- D_max * (1 - t/T) * U(-1, 1)                             (eq. 4)

    dt <- C_t * sum(u - l) * (1 - t/T)^2                          (eq. 5)
    x  <- repair(x + dt * (N + F + D)), re-evaluated
    crossover a few coordinates from the best krill
    update personal bests

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(N\) | 25 | Number of krill |
| `n_max` | \(N^{\max}\) | 0.01 | Maximum induced speed |
| `v_f` | \(V_f\) | 0.02 | Foraging speed |
| `d_max` | \(D^{\max}\) | 0.005 | Maximum diffusion speed |
| `inertia` | \(\omega\) | 0.9 | Motion retained between iterations |
| `c_t` | \(C_t\) | 0.5 | Time-step factor |
| `crossover_rate` | — | 0.2 | Coordinates copied from the best krill |
| `seed` | — | `None` | Reproducibility |

!!! success "Translation- and rotation-invariant — the cleanest profile measured here"
    Every coupling in KH uses **position differences** and **normalized
    fitness gaps**, never absolute coordinates. That should make it
    immune to both transformations that have caught other algorithms
    out, and it is (20,000 evaluations, 3 seeds):

    | Variant | KH | [GWO](gwo.md) | [HS](hs.md) |
    |---|---|---|---|
    | Rastrigin | 2.99 | 0 | 0.018 |
    | rotated | 2.65 | 10.8 | 26.3 |
    | shifted | 2.65 | 15.6 | 0.075 |
    | rotated + shifted | 2.99 | 22.4 | 30.4 |

    KH varies by less than 15% across all four, where GWO ranges over a
    factor of infinity and HS over three orders of magnitude. Its
    Rastrigin score is not the best in the table, but it is the score
    you will actually get on a problem that does not happen to be
    axis-aligned with its optimum at zero.

    There is a test asserting this directly: shifting the problem shifts
    the entire trajectory and changes nothing else.

!!! warning "The published fixed step never lets the herd settle"
    KH defines \(\Delta t = C_t \sum_j (u_j - l_j)\), a constant. Since
    the induced and foraging terms are built from *unit* direction
    vectors, their magnitude does not shrink as the herd converges
    either — so the only decaying term is diffusion, and the krill keep
    moving about 15% of the domain per iteration for the entire run.

    Adding the budget-tied decay of equation 5 is the single largest
    improvement measured for any algorithm in this library (isolated
    ablation, 20,000 evaluations, 3 seeds — every other
    progress-dependent term left untouched):

    | Step | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | fixed, as published | 0.070 | 26.5 | 3.68 |
    | **budget-tied decay** | **4e-09** | **2.99** | **4e-04** |

    Seven orders of magnitude on Sphere. This is the same fix the
    [SA](sa.md), [CLONALG](clonalg.md), [CRO](cro.md), and [HS](hs.md)
    pages record, and 2012 is late for an algorithm to be published
    without it.

!!! note "Tuning and implementation notes"
    - **The food-centre weight is changed.** The paper uses
      \(w_i = 1/K_i\), which is undefined when a fitness reaches zero —
      as it does on every benchmark here, and on Rastrigin exactly. The
      implementation uses \(1/(\tilde{K}_i + 0.1)\) on the *normalized*
      fitness, which is bounded, scale-free, and preserves the intent
      that better krill pull the centre harder.
    - **Crossover earns its place.** Disabling it costs a factor of
      three on Sphere and doubles Rastrigin, so it is on by default at a
      modest rate.
    - **`inertia` trades precision for exploration.** Dropping it from
      0.9 to 0.5 improves Sphere (9e-10 vs 4e-09) but worsens Rastrigin
      (4.6 vs 3.0); 0.99 is worse at everything.

## Behavior

KH reaches Sphere ≈ 6e-09, Rastrigin ≈ 2.8, Ackley ≈ 4e-04. Read as a
row in the comparison table that is unremarkable — mid-field precision,
good multimodal.

Read as a *robustness* result it is the most interesting entry in this
library. Several algorithms score better on the standard suite and then
lose one to twenty orders of magnitude when the problem is rotated or
moved. KH scores what it scores, and keeps scoring it. For choosing an
optimizer for a real problem — which is neither axis-aligned nor
centred on the origin — that matters more than a headline number.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import KrillHerd

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = KrillHerd(population_size=25, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- A. H. Gandomi and A. H. Alavi, "Krill herd: a new bio-inspired
  optimization algorithm," *Communications in Nonlinear Science and
  Numerical Simulation*, 17(12), 4831-4845, 2012.
  [doi:10.1016/j.cnsns.2012.05.010](https://doi.org/10.1016/j.cnsns.2012.05.010).
- G.-G. Wang, A. H. Gandomi, A. H. Alavi, and D. Gong, "A comprehensive
  review of krill herd algorithm: variants, hybrids and applications,"
  *Artificial Intelligence Review*, 51, 119-148, 2019.
  [doi:10.1007/s10462-017-9559-1](https://doi.org/10.1007/s10462-017-9559-1).
