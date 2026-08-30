# Differential Evolution (DE)

**DE** (Storn & Price, 1997) is one of the most widely used
metaheuristics, and its central idea is remarkably simple: build a
mutant by adding the **scaled difference between two population
members** to a third one.

That single choice gives DE a property most algorithms need extra
machinery for — a **self-adaptive step size**. Difference vectors come
from the population itself, so while the population is spread out the
steps are large, and once it converges they shrink automatically. No
cooling schedule, no decaying radius, nothing to tune.

## Equations

**1. Mutation.** For each target \(x_i\), a mutant \(v_i\) is built
from randomly chosen distinct members (all \(\neq i\)). Four strategies
are available:

\[
\begin{aligned}
\text{best/1:} \quad & v_i = x^{*} + F\,(x_{r_1} - x_{r_2})\\
\text{rand/1:} \quad & v_i = x_{r_1} + F\,(x_{r_2} - x_{r_3})\\
\text{rand/2:} \quad & v_i = x_{r_1} + F\,(x_{r_2} - x_{r_3}) + F\,(x_{r_4} - x_{r_5})\\
\text{current-to-best/1:} \quad & v_i = x_i + F\,(x^{*} - x_i) + F\,(x_{r_1} - x_{r_2})
\end{aligned}
\]

where \(F\) is the differential weight and \(x^{*}\) the current best.

**2. Binomial crossover.** The trial vector mixes mutant and target
gene by gene, with one gene \(j_{\text{rand}}\) always taken from the
mutant so the trial can never be an exact copy of the target:

\[
u_{i,j} =
\begin{cases}
v_{i,j} & \text{if } \mathcal{U}(0,1) < CR \text{ or } j = j_{\text{rand}}\\
x_{i,j} & \text{otherwise}
\end{cases}
\]

**3. Greedy selection.** The trial replaces **its own target** — and
only its own — if it is at least as good:

\[
x_i \leftarrow
\begin{cases}
u_i & \text{if } f(u_i) \le f(x_i)\\
x_i & \text{otherwise}
\end{cases}
\]

This one-to-one replacement is why DE never loses diversity abruptly:
each individual competes only with its own offspring.

## Pseudocode

```text
input: population NP, differential weight F, crossover rate CR, strategy
x <- NP random solutions, evaluated

repeat until the budget is exhausted:
    best <- index of the best individual
    for each target i:
        v <- mutant built from random distinct members            (eq. 1)
        u <- binomial crossover of v and x[i]                     (eq. 2)
        u <- repair(u)
        if f(u) <= f(x[i]):                                       (eq. 3)
            x[i] <- u

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | NP | 50 | Individuals; larger = more diverse, slower |
| `differential_weight` | \(F\) | 0.6 | Difference scale; larger = more exploration |
| `crossover_rate` | CR | 0.5 | Chance a gene comes from the mutant |
| `strategy` | — | `"best/1"` | Mutation scheme (see equation 1) |
| `seed` | — | `None` | Reproducibility |

!!! note "Why these defaults differ from the textbook"
    The classic recommendation is `strategy="rand/1"`, `F=0.8`,
    `CR=0.9`. On this library's benchmark budget (20,000 evaluations)
    that combination underperformed badly — Rastrigin ≈ 46 — because
    `rand/1` is exploratory and needs far longer runs to pay off.

    Two changes fixed it. Switching to **`best/1`** brought Rastrigin to
    9.6 and Sphere to 3e-22. Lowering **`CR` to 0.5** then improved
    Rastrigin to 1.7: a low crossover rate changes only a few
    coordinates per trial, and separable multimodal functions reward
    moving one coordinate at a time. (Cat Swarm Optimization showed the
    same effect through its `cdc` parameter.)

    All four strategies remain available — `rand/1` is still the better
    choice for very long runs or highly deceptive landscapes.

## Behavior

DE is the **strongest all-round algorithm in this library**:

| Function | DE | Best of the others |
|---|---|---|
| Sphere | **2e-41** | 9e-25 (ACO-R) |
| Ackley | **5e-15** | 1e-12 (ACO-R) |
| Rastrigin | 2.0 | 2e-08 (ABC) |

It takes the top spot on both smooth landscapes by many orders of
magnitude, and comes second on Rastrigin. This matches DE's reputation
in the literature — it has won or placed highly in numerous
optimization competitions, and remains a standard baseline that new
metaheuristics are expected to beat.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import DifferentialEvolution

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = DifferentialEvolution(population_size=50, strategy="best/1", seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- R. Storn and K. Price, "Differential evolution — a simple and
  efficient heuristic for global optimization over continuous spaces,"
  *Journal of Global Optimization*, 11(4), 341-359, 1997.
  [doi:10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328).
- K. Price, R. Storn, and J. Lampinen, *Differential Evolution: A
  Practical Approach to Global Optimization*, Springer, 2005.
- S. Das and P. N. Suganthan, "Differential evolution: a survey of the
  state-of-the-art," *IEEE Transactions on Evolutionary Computation*,
  15(1), 4-31, 2011.
  [doi:10.1109/TEVC.2010.2059031](https://doi.org/10.1109/TEVC.2010.2059031).
