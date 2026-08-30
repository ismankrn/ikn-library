# Ant Colony Optimization for Continuous Domains (ACO-R)

Classical Ant Colony Optimization solves *discrete* problems: ants walk
a graph and deposit pheromone on edges. **ACO-R** (Socha & Dorigo,
2008) carries the idea to continuous search spaces by replacing the
discrete pheromone table with a **solution archive**: the \(k\) best
solutions found so far. Each ant builds a new solution by picking one
archive member as a guide and sampling around it — the archive *is* the
pheromone.

## Equations

**1. Rank-based selection weight.** Archive solutions are sorted by
fitness; solution of rank \(i\) (starting at 0) gets weight

\[
w_i = \frac{1}{qk\sqrt{2\pi}}\,
      \exp\!\left(-\frac{i^{2}}{2q^{2}k^{2}}\right)
\]

where \(k\) is the archive size and \(q\) the `intensification`
parameter. Small \(q\) concentrates the choice on the very best
solutions; large \(q\) spreads it out. An ant picks guide \(l\) with
probability \(p_l = w_l / \sum_j w_j\).

**2. Gaussian width.** For the chosen guide \(l\), the standard
deviation in dimension \(d\) is the mean distance from the guide to the
rest of the archive, scaled by the `evaporation` rate \(\xi\):

\[
\sigma_d^{l} = \xi \sum_{j=1}^{k}
               \frac{\left|x_d^{j} - x_d^{l}\right|}{k - 1}
\]

As the archive converges, the members crowd together, \(\sigma\)
shrinks, and the search automatically turns from exploration into
refinement — no explicit cooling schedule needed.

**3. Sampling.** Each coordinate of the new solution is drawn
independently:

\[
x_d^{\text{new}} \sim \mathcal{N}\!\left(x_d^{l},\, \sigma_d^{l}\right)
\]

then clipped back into the bounds.

## Pseudocode

```text
input: archive size k, ants m, intensification q, evaporation xi
archive <- k uniform random solutions, evaluated and sorted

repeat until the budget is exhausted:
    w[i]  <- Gaussian rank weight of archive member i        (eq. 1)
    p     <- w / sum(w)

    for a = 1 .. m:
        l          <- sample a guide index using p
        sigma      <- xi * mean(|archive - archive[l]|)      (eq. 2)
        ant[a]     <- repair(Normal(archive[l], sigma))      (eq. 3)
        f_ant[a]   <- evaluate(ant[a])

    merged <- archive + ants, sorted by fitness
    archive <- best k of merged

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(m\) | 30 | Ants (new solutions) per iteration |
| `archive_size` | \(k\) | 50 | Solutions remembered; larger = more diversity |
| `intensification` | \(q\) | 0.1 | Locality of guide selection; smaller = greedier |
| `evaporation` | \(\xi\) | 0.85 | Width scaling; smaller = faster convergence |
| `seed` | — | `None` | Reproducibility |

## Behavior

ACO-R is the **precision specialist** of this library: on the smooth
Sphere function it reaches ~1e-25, orders of magnitude beyond the
others, because the shrinking archive gives it an ever-finer search
scale. On the highly multimodal Rastrigin it is weaker (~31) — the
archive can converge on one local basin.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import AntColonyOptimization

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- K. Socha and M. Dorigo, "Ant colony optimization for continuous
  domains," *European Journal of Operational Research*, 185(3),
  1155-1173, 2008.
  [doi:10.1016/j.ejor.2006.06.046](https://doi.org/10.1016/j.ejor.2006.06.046).
- M. Dorigo, V. Maniezzo, and A. Colorni, "Ant system: optimization by
  a colony of cooperating agents," *IEEE Transactions on Systems, Man,
  and Cybernetics, Part B*, 26(1), 29-41, 1996.
  [doi:10.1109/3477.484436](https://doi.org/10.1109/3477.484436).
- M. Dorigo and T. Stützle, *Ant Colony Optimization*, MIT Press, 2004.
