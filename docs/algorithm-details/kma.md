# Komodo Mlipir Algorithm (KMA)

**KMA** (Suyanto, Ariyanto & Ariyanto, 2022) draws on two Indonesian
sources: the foraging and reproduction of **Komodo dragons** of East
Nusa Tenggara, and *mlipir*, a Javanese term for walking along the side
of the road to reach a destination safely.

Its distinguishing idea is that the population is **not homogeneous**.
Every iteration the ranked individuals are split into three groups with
different jobs, and the population size itself adapts to how well the
search is going.

## The three groups

| Group | Who | Strategy |
|---|---|---|
| **Big males** (\(q\) best) | high quality | HILE — high exploitation, low exploration |
| **Female** (1, middle) | middle quality | mating **or** parthenogenesis, 50/50 |
| **Small males** (\(s\) rest) | low quality | *mlipir* — follow big males in some dimensions |

## Equations

**1. Group split.** With portion \(p\) and population \(n\):

\[
q = \lfloor p\,(n-1) \rfloor, \qquad s = n - q - 1
\]

**2. Big males — HILE.** Big male \(i\) is *attracted* to a better
male, and to a worse one it is attracted or *distracted* with
probability 0.5 each:

\[
w_{ij} =
\begin{cases}
r_1 (k_j - k_i) & \text{if } f(k_j) < f(k_i) \text{ or } r_2 < 0.5\\
r_1 (k_i - k_j) & \text{otherwise}
\end{cases}
\qquad
k_i' = k_i + \sum_{j \neq i}^{q} w_{ij}
\]

Because at least one big male always moves toward a better one,
exploitation is guaranteed to outweigh exploration. The \(q\) best
positions among old and new survive.

**3. Female — mating.** With probability 0.5 the female mates the
winning big male, an arithmetic crossover per dimension \(l\):

\[
k_{il}' = r_l\,k_{il} + (1 - r_l)\,k_{jl},
\qquad
k_{jl}' = r_l\,k_{jl} + (1 - r_l)\,k_{il}
\]

Two offspring are produced and the better one replaces the female if it
improves on her.

**4. Female — parthenogenesis.** Otherwise she reproduces asexually
with a small symmetric step of radius \(\alpha\):

\[
k_{ij}' = k_{ij} + (2r - 1)\,\alpha\,\lvert ub_j - lb_j \rvert
\]

**5. Small males — mlipir.** A small male follows each big male in only
a random subset of dimensions, selected with probability \(d\) (the
*mlipir rate*):

\[
w_{ij} =
\begin{cases}
r_1 (k_{jl} - k_{il}) & \text{if } r_2 < d\\
0 & \text{otherwise}
\end{cases}
\qquad
k_i' = k_i + \sum_{j=1}^{q} w_{ij}
\]

Following only *part* of a big male's coordinates is what keeps the
low-quality group diverse instead of collapsing onto the leaders.

**6. Self-adapting population.** With \(\delta f_1, \delta f_2\) the
relative fitness changes of the last two generations:

\[
n' =
\begin{cases}
n - a & \text{if } \delta f_1 > 0 \text{ and } \delta f_2 > 0 \quad (\text{improving})\\
n + a & \text{if } \delta f_1 = 0 \text{ and } \delta f_2 = 0 \quad (\text{stagnating})
\end{cases}
\]

While the search improves the population shrinks (cheaper iterations);
when it stalls, new individuals — random moves of the best-so-far
Komodo — are added to restore diversity.

## Pseudocode

```text
input: population n, portion p, mlipir rate d, step a, radius alpha
population <- n random solutions, evaluated and ranked

repeat until the budget is exhausted:
    split into q big males, 1 female, s small males               (eq. 1)

    for each big male i:
        k'[i] <- k[i] + sum over the other big males of w_ij      (eq. 2)
    keep the q best positions among old and new

    if Uniform(0,1) < 0.5:
        offspring <- mate(winner big male, female)                (eq. 3)
    else:
        offspring <- parthenogenesis(female)                      (eq. 4)
    female <- offspring if it is better

    for each small male i:
        k'[i] <- k[i] + mlipir steps toward the big males         (eq. 5)
    keep all new small male positions

    adapt the population size                                     (eq. 6)
    re-rank the population

return the best Komodo found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 15 | Initial number of Komodo individuals |
| `big_male_portion` | \(p\) | 0.5 | Share of the population that becomes big males |
| `mlipir_rate` | \(d\) | 0.5 | Chance a small male follows a big male per dimension |
| `max_big_males` | — | 3 | Cap on the big-male group (see the note below) |
| `adaptation_step` | \(a\) | 5 | Individuals added/removed when adapting |
| `min_population`, `max_population` | — | 10, 200 | Limits of the adaptive size |
| `parthenogenesis_radius` | \(\alpha\) | 0.1 | Female's asexual step, as a fraction of the range |
| `seed` | — | `None` | Reproducibility |

!!! warning "Why the big-male group is capped"
    A big male's step is the **sum** of its interactions with every
    other big male (equation 2), so the step length grows with the size
    of that group. With the paper's phase-2 setting of \(n = 200\) and
    \(p = 0.5\), each of the ~99 big males takes a step built from 98
    terms — in our benchmarks the population scattered and the search
    stalled (Sphere ≈ 8).

    The paper itself states that "two or three big males will give an
    optimum interaction", so this implementation enforces that as
    `max_big_males=3`. With the cap, the same benchmark improves from
    ≈ 8 to ≈ 1e-21. Set `max_big_males` higher to reproduce the
    uncapped formulation.

## Behavior

KMA is a **strong all-rounder** in this library: near-best precision on
smooth landscapes (Sphere ≈ 1e-21, close to ACO-R) *and* solid
multimodal performance (Rastrigin ≈ 17, better than every algorithm
here except ABC and GA). The mix of three strategies is what buys that
range — exploitative big males refine, the female injects either
recombination or a random jump, and the mlipir movement keeps partial
diversity alive.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import KomodoMlipirAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = KomodoMlipirAlgorithm(population_size=15, seed=42)
best_x, best_fitness = algo.run(task)
```

## Reference

S. Suyanto, A. A. Ariyanto, and A. F. Ariyanto, "Komodo Mlipir
Algorithm," *Applied Soft Computing*, 114, 108043, 2022.
[doi:10.1016/j.asoc.2021.108043](https://doi.org/10.1016/j.asoc.2021.108043).
The authors' reference implementation (MATLAB) is linked from the
paper.
