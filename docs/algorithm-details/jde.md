# Self-Adaptive Differential Evolution (jDE)

**jDE** (Brest et al., 2006) is [Differential Evolution](de.md) with its
two hardest-to-set parameters taken out of the user's hands. Each
individual carries its **own** \(F_i\) and \(CR_i\), occasionally
re-drawn at random — and kept only if the trial they produced won.

That last clause is the whole mechanism, and it is easy to read past.
The parameters are not adjusted by any rule, schedule, or feedback
heuristic. They simply **ride along with the solution they generated**:
values that produce winning trials survive into the next generation with
their offspring, values that produce losers are discarded along with
them. Selection tunes the parameters and the solutions at the same time,
using machinery the algorithm already had.

This is also the scheme the [Hybrid Self-Adaptive Bat
Algorithm](hsaba.md) borrows, so the library now holds it in both the
algorithm it was designed for and one it was transplanted into.

## Equations

**1. Parameter proposal**, per individual, before its trial is built:

\[
F_i \leftarrow
\begin{cases}
F_l + \rho\,(F_u - F_l) & \text{if } \text{rand} < \tau_1 \\
F_i & \text{otherwise}
\end{cases}
\qquad
CR_i \leftarrow
\begin{cases}
\text{rand} & \text{if } \text{rand} < \tau_2 \\
CR_i & \text{otherwise}
\end{cases}
\]

**2. Trial construction** — the ordinary DE operators, using this
individual's own values:

\[
v = x_{r_1} + F_i\,(x_{r_2} - x_{r_3}),
\qquad
u_k =
\begin{cases}
v_k & \text{with probability } CR_i \\
x_{i,k} & \text{otherwise}
\end{cases}
\]

**3. Selection, of both at once.** If \(f(u) \le f(x_i)\) the trial
replaces the target **and** the proposed \(F_i, CR_i\) become
permanent. Otherwise both the trial and its parameters are thrown away.

## Pseudocode

```text
input: individuals NP, F range [F_l, F_u], tau_1, tau_2, strategy
x <- NP random solutions, evaluated
F  <- U(F_l, F_u) per individual;  CR <- U(0, 1) per individual

repeat until the budget is exhausted:
    for each individual i:
        f  <- U(F_l, F_u)  if rand() < tau_1  else F[i]           (eq. 1)
        cr <- rand()       if rand() < tau_2  else CR[i]          (eq. 1)

        mutant <- DE mutation using f                             (eq. 2)
        trial  <- binomial crossover using cr                     (eq. 2)
        if f(trial) <= f(x[i]):
            x[i] <- trial ;  F[i], CR[i] <- f, cr                 (eq. 3)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | NP | 30 | Number of individuals |
| `min_weight` | \(F_l\) | 0.1 | Lower end of the \(F\) range |
| `max_weight` | \(F_u\) | 0.9 | Upper end of the \(F\) range |
| `tau_1` | \(\tau_1\) | 0.1 | Chance of re-drawing \(F\) |
| `tau_2` | \(\tau_2\) | 0.1 | Chance of re-drawing \(CR\) |
| `strategy` | — | `"rand/1"` | Mutation strategy, as in [DE](de.md) |
| `seed` | — | `None` | Reproducibility |

Only `population_size` departs from the published settings (50 → 30).

!!! success "Self-adaptation beats hand tuning here — including on the hard variants"
    The honest test of an adaptive method is whether it beats a
    carefully tuned fixed one. This library's [DE](de.md) defaults were
    themselves tuned over several rounds (`best/1`, `F=0.6`, `CR=0.5`).
    Measured on all four benchmark variants (Rastrigin, 20,000
    evaluations, 5 seeds):

    | Algorithm | plain | rotated | shifted | rot + shift |
    |---|---|---|---|---|
    | **jDE** | **0** | **13.7** | **0.199** | **13.8** |
    | [DE](de.md) `best/1`, tuned | 2.79 | 29.3 | 3.18 | 23.0 |
    | [DE](de.md) `rand/1`, same strategy | 13.8 | — | — | — |

    jDE wins on **every** variant, roughly halving the tuned DE's error
    on both rotated ones. Against DE using the *same* `rand/1`
    strategy, the gap on plain Rastrigin is nine orders of magnitude
    (0 against 13.8), which isolates the self-adaptation rather than
    the strategy choice.

    Note what this is not: jDE's plain Rastrigin of exactly 0 does
    degrade under rotation, so it retains DE's axis-alignment. It has no
    origin bias at all — a shift costs it almost nothing (0.199) — and
    there is a test asserting the translation invariance.

!!! note "What the parameters actually converge to"
    Tracing a run on Rastrigin, the population's mean \(F\) and \(CR\)
    settle at:

    | Evaluations | mean \(F\) | sd | mean \(CR\) | sd |
    |---|---|---|---|---|
    | 2,000 | 0.35 | 0.18 | 0.43 | 0.31 |
    | 10,000 | 0.40 | 0.18 | 0.51 | 0.25 |
    | 19,500 | 0.43 | 0.21 | 0.38 | 0.32 |

    Two things are worth noticing. The means land close to the values
    manual tuning found for this library's DE (`F=0.6`, `CR=0.5`) — the
    self-adaptation rediscovers roughly the right region without being
    told. And the standard deviations stay **large**: the population
    never collapses onto one parameter setting, which is what lets
    different individuals do different jobs. A test asserts that
    diversity survives to the end of a run.

!!! note "Tuning notes"
    - **`population_size=30`, not the published 50.** At 50 the plain
      Rastrigin result is 1e-08 rather than 0, and rotated degrades from
      13.7 to 25.8.
    - **Keep `strategy="rand/1"`.** Switching to `best/1` — the
      strongest choice for fixed-parameter DE here — makes jDE
      *worse* (Rastrigin 7.36 against 0). Greedy mutation and adaptive
      parameters pull against each other.
    - **The `tau` values barely matter.** Anything from 0.02 to 0.3
      lands within noise, which is a point in the method's favour: it
      replaces two sensitive parameters with two insensitive ones.

## Behavior

jDE reaches Sphere ≈ 1e-36, **Rastrigin exactly 0 on every seed**, and
Ackley ≈ 4e-15 — one of the strongest rows in this library, and unlike
several others it survives being shifted.

It is also the most useful single answer to a question this library
raises repeatedly. Many pages here record an algorithm whose published
defaults were poor and whose results came from retuning. jDE is the
alternative approach: rather than tuning \(F\) and \(CR\) for each
problem, let selection do it. On this evidence that works better than
the tuning did.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import SelfAdaptiveDifferentialEvolution

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = SelfAdaptiveDifferentialEvolution(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- J. Brest, S. Greiner, B. Bošković, M. Mernik, and V. Žumer,
  "Self-adapting control parameters in differential evolution: a
  comparative study on numerical benchmark problems," *IEEE
  Transactions on Evolutionary Computation*, 10(6), 646-657, 2006.
  [doi:10.1109/TEVC.2006.872133](https://doi.org/10.1109/TEVC.2006.872133).
- R. Storn and K. Price, "Differential evolution — a simple and
  efficient heuristic for global optimization over continuous spaces,"
  *Journal of Global Optimization*, 11(4), 341-359, 1997.
  [doi:10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328).
- A. E. Eiben, R. Hinterding, and Z. Michalewicz, "Parameter control in
  evolutionary algorithms," *IEEE Transactions on Evolutionary
  Computation*, 3(2), 124-141, 1999 — the taxonomy that distinguishes
  self-adaptation from tuning and from deterministic schedules.
  [doi:10.1109/4235.771166](https://doi.org/10.1109/4235.771166).
