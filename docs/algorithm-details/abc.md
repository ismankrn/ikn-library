# Artificial Bee Colony (ABC)

**ABC** (Karaboga, 2005) models how a honey-bee colony forages. Each
**food source** is a candidate solution, and the colony splits into
three roles that run in sequence every iteration: employed bees exploit
known sources, onlooker bees concentrate on the promising ones, and a
scout bee abandons a source that has stopped paying off. That last
mechanism is why ABC escapes local optima so well — on this library's
benchmarks it is by far the strongest on the highly multimodal
Rastrigin function.

## Equations

**1. Neighbor generation.** A bee at source \(i\) picks a random
partner \(j \neq i\) and a random dimension \(d\), then steps along the
difference between them:

\[
v_{d} = x_{i,d} + \phi\,\bigl(x_{i,d} - x_{j,d}\bigr),
\qquad \phi \sim \mathcal{U}(-1, 1)
\]

All other coordinates are copied unchanged. As the population
converges, \(|x_{i,d} - x_{j,d}|\) shrinks, so the step size adapts by
itself.

**2. Greedy selection.** The neighbor replaces the source only if it is
better; otherwise the source's trial counter grows:

\[
x_i \leftarrow
\begin{cases}
v, \; t_i \leftarrow 0 & \text{if } f(v) < f(x_i)\\
x_i, \; t_i \leftarrow t_i + 1 & \text{otherwise}
\end{cases}
\]

**3. Onlooker probability.** Minimization fitness values are first
mapped to positive quality scores,

\[
\mathrm{fit}_i =
\begin{cases}
\dfrac{1}{1 + f_i} & f_i \ge 0\\[2ex]
1 + |f_i| & f_i < 0
\end{cases}
\qquad
p_i = \frac{\mathrm{fit}_i}{\sum_{j} \mathrm{fit}_j}
\]

so that better (lower) objective values receive higher probability.

**4. Scout.** If \(t_i \ge \text{limit}\), the source is abandoned and
replaced by a fresh uniform random solution:

\[
x_i \sim \mathcal{U}(\text{lower},\, \text{upper}), \qquad t_i \leftarrow 0
\]

Only one scout is sent per iteration — Karaboga's original
formulation — so the most stagnant exhausted source is chosen.

## Pseudocode

```text
input: sources n, abandonment limit L (default n * dimension)
x     <- n uniform random solutions, evaluated
t[i]  <- 0                                  # trial counters

repeat until the budget is exhausted:
    # employed bees
    for i = 1 .. n:
        v <- neighbor of x[i] via a random partner and dimension  (eq. 1)
        greedy selection between x[i] and v, update t[i]          (eq. 2)

    # onlooker bees
    p <- selection probabilities from fitness                     (eq. 3)
    for n draws of index i using p:
        v <- neighbor of x[i]                                     (eq. 1)
        greedy selection between x[i] and v, update t[i]          (eq. 2)

    # scout bee
    if max(t) >= L:
        replace the most stagnant source with a random solution   (eq. 4)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Food sources (and onlookers per iteration) |
| `limit` | \(L\) | \(n \times \text{dim}\) | Trials before abandonment; smaller = more restarts |
| `seed` | — | `None` | Reproducibility |

Note that one ABC iteration costs about \(2n + 1\) evaluations
(employed + onlooker + occasional scout), so with a fixed `max_evals`
budget a larger colony means fewer iterations.

## Behavior

ABC is the **all-rounder** of this library and the only algorithm here
that essentially *solves* Rastrigin (~2e-08 where others stall at 3-31).
The reason is equation 4: a source trapped in a local basin is
eventually thrown away entirely, something gradient-like refinements
never do. It reached these results with **default parameters** — no
tuning was needed, unlike SA, GA, and BA.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import ArtificialBeeColony

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = ArtificialBeeColony(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- D. Karaboga, "An idea based on honey bee swarm for numerical
  optimization," Technical Report TR06, Erciyes University, 2005.
- D. Karaboga and B. Basturk, "A powerful and efficient algorithm for
  numerical function optimization: artificial bee colony (ABC)
  algorithm," *Journal of Global Optimization*, 39(3), 459-471, 2007.
  [doi:10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x).
- D. Karaboga and B. Akay, "A comparative study of artificial bee
  colony algorithm," *Applied Mathematics and Computation*, 214(1),
  108-132, 2009.
  [doi:10.1016/j.amc.2009.03.090](https://doi.org/10.1016/j.amc.2009.03.090).
