# Particle Swarm Optimization (PSO)

**PSO** (Kennedy & Eberhart, 1995) is the oldest and best known swarm
algorithm, and the one to learn first. Each particle carries a
**velocity** and is pulled by two attractors: the best position it has
personally visited, and the best any particle has found.

Velocity is what separates PSO from the many algorithms that reposition
individuals directly. A particle carries momentum, so it **overshoots**
its attractors and oscillates around them rather than settling on them.
That oscillation is the search: it explores the region between and
beyond the two bests without any explicit exploration operator.

For this library PSO plays a second role. Several pages here — on
[Harmony Search](hs.md), [Moth-Flame](mfo.md), [Grey Wolf](gwo.md) —
record published arguments that a later algorithm is a re-description of
something already known. PSO is usually one of the things it is a
re-description *of*, which makes it the natural baseline: **if a new
metaheuristic cannot beat PSO on your problem, its novelty is not worth
much to you.**

## Equations

**1. Velocity update.** Momentum plus two attractions:

\[
v \leftarrow w v
+ c_1 r_1 \odot (p_{\text{best}} - x)
+ c_2 r_2 \odot (g_{\text{best}} - x),
\qquad r_1, r_2 \sim \mathcal{U}(0,1)^d
\]

**2. Velocity clamp.** Speed is limited to a fraction of each bound
range, without which the swarm diverges:

\[
v \leftarrow \operatorname{clip}\bigl(v,\ -v_{\max}(u - l),\ v_{\max}(u - l)\bigr)
\]

**3. Position update.**

\[
x \leftarrow x + v
\]

**4. Inertia schedule.** \(w\) falls linearly across the run, damping
the momentum so the swarm converges:

\[
w = w_{\text{start}} - (w_{\text{start}} - w_{\text{end}})
\frac{\text{evals}}{\text{max\_evals}}
\]

Shi and Eberhart added this in 1998 for exactly that purpose, which
makes PSO one of the few algorithms in this library whose step schedule
was tied to the run's progress from the start — the fix the
[SA](sa.md), [CRO](cro.md) and [KH](kh.md) pages have to retrofit.

## Pseudocode

```text
input: particles n, w_start, w_end, c1, c2, v_max
x <- n random solutions, evaluated;  v <- 0;  pbest <- x

repeat until the budget is exhausted:
    w <- w_start - (w_start - w_end) * evals/max_evals            (eq. 4)
    g <- best of all personal bests

    v <- w*v + c1*r1*(pbest - x) + c2*r2*(g - x)                  (eq. 1)
    v <- clip(v, +-v_max*(u - l))                                 (eq. 2)
    x <- repair(x + v), re-evaluated                              (eq. 3)
    update each particle's personal best

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Number of particles |
| `w_start` | — | 0.7 | Initial inertia (Shi & Eberhart use 0.9) |
| `w_end` | — | 0.4 | Final inertia |
| `c1` | \(c_1\) | 2.0 | Cognitive pull, toward the personal best |
| `c2` | \(c_2\) | 2.0 | Social pull, toward the global best |
| `max_velocity` | \(v_{\max}\) | 0.2 | Speed limit as a fraction of the range |
| `seed` | — | `None` | Reproducibility |

!!! warning "PSO is translation-invariant but not rotation-invariant"
    Both pulls in equation 1 are **differences** of positions, so
    shifting the problem shifts the whole trajectory and nothing else —
    there is a test asserting this. But \(r_1\) and \(r_2\) are drawn
    **per coordinate**, so the velocity update stretches the attraction
    differently along each axis. The search is therefore axis-aligned,
    and rotating the problem costs real performance (20,000
    evaluations, 5 seeds):

    | Variant | Rastrigin | | Sphere |
    |---|---|---|---|
    | plain | 3.58 | | 1e-28 |
    | shifted | 5.37 | | 4e-28 |
    | rotated | **15.9** | | — |
    | rotated + shifted | 12.1 | | — |

    A factor of about 4.4 under rotation. This is a long-documented
    property of PSO rather than an artefact of this implementation —
    Wilke, Kok and Groenwold analysed it in 2007 and it is the reason
    rotation-invariant PSO variants exist.

    In the library's terms, PSO sits between the extremes: far more
    robust than [MBO](mbo.md) (200,000×) or [HS](hs.md) (1,500×), less
    so than [KH](kh.md) (1.0×) or [HHO](hho.md).

!!! note "Tuning notes"
    - **`w_start=0.7` beats the textbook 0.9** here (Rastrigin 3.58
      against 4.38, Sphere 1e-28 against 3e-19). The classic value was
      chosen for smaller budgets; at 20,000 evaluations less initial
      momentum converges better.
    - **`c1 = c2` is a good default, but not the only sensible one.**
      Lowering both to 1.5 buys fourteen orders of magnitude on Sphere
      (2e-42) and costs Rastrigin (5.31 against 4.38) — a clean
      illustration of the exploration/exploitation trade in a single
      parameter.
    - **The velocity clamp matters more than it looks.** At
      `max_velocity=0.5` Rastrigin improves to 2.70 while Sphere loses
      three orders of magnitude; at 0.1 Rastrigin degrades to 8.29.

## Behavior

PSO reaches Sphere ≈ 1e-28, Ackley ≈ 1e-13, Rastrigin ≈ 3.58 — strong
across the board and, importantly, strong *without* any of the
mechanisms later algorithms add: no archive, no roles, no ageing, no
branching, no Lévy flights. Two attractors and momentum.

That is the point worth taking from this page. Of the twenty-nine
algorithms measured here, only a handful clearly beat PSO on all three
benchmarks, and several of the ones that beat it on a single row do so
through the separability or origin artefacts documented elsewhere.
Before adopting a newer method, check that it beats this one.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import ParticleSwarmOptimization

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = ParticleSwarmOptimization(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- J. Kennedy and R. Eberhart, "Particle swarm optimization,"
  *Proceedings of ICNN'95 — International Conference on Neural
  Networks*, 1942-1948, 1995.
  [doi:10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).
- Y. Shi and R. Eberhart, "A modified particle swarm optimizer," *IEEE
  World Congress on Computational Intelligence*, 69-73, 1998 — the
  inertia weight.
  [doi:10.1109/ICEC.1998.699146](https://doi.org/10.1109/ICEC.1998.699146).
- M. Clerc and J. Kennedy, "The particle swarm — explosion, stability,
  and convergence in a multidimensional complex space," *IEEE
  Transactions on Evolutionary Computation*, 6(1), 58-73, 2002 — the
  constriction-factor alternative to the inertia weight.
  [doi:10.1109/4235.985692](https://doi.org/10.1109/4235.985692).
- D. N. Wilke, S. Kok, and A. A. Groenwold, "Comparison of linear and
  classical velocity update rules in particle swarm optimization: notes
  on scale and frame invariance," *International Journal for Numerical
  Methods in Engineering*, 70(8), 985-1008, 2007 — on the rotational
  variance measured above.
  [doi:10.1002/nme.1914](https://doi.org/10.1002/nme.1914).
- K. Sörensen, "Metaheuristics — the metaphor exposed," *International
  Transactions in Operational Research*, 22(1), 3-18, 2015 — on why a
  well-understood baseline matters.
  [doi:10.1111/itor.12001](https://doi.org/10.1111/itor.12001).
