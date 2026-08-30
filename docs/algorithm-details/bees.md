# The Bees Algorithm

The **Bees Algorithm** (Pham et al., 2005) imitates how a colony scouts
for flower patches: scouts wander at random, the promising **sites**
they report get more foragers, and the very best (**elite**) sites get
the most. Around each site the colony searches a shrinking
neighborhood, and a patch that stops yielding is abandoned.

It shares its inspiration with the
[Artificial Bee Colony](abc.md) but organizes the search differently:
here effort allocation is **explicit** — a fixed number of recruits per
site class — while ABC lets a probability distribution decide where
onlookers go. The Bees Algorithm also keeps a **per-site search
radius**, which ABC does not have.

## Equations

**1. Site selection.** After sorting the \(n\) scouts by fitness, the
best \(m\) become sites for local search, of which the top \(e\) are
elite:

\[
\text{sites} = \{x_{(1)}, \dots, x_{(m)}\}, \qquad
\text{elite} = \{x_{(1)}, \dots, x_{(e)}\}, \qquad e \le m < n
\]

**2. Neighborhood search (recruitment).** Site \(i\) receives
\(n_{ep}\) recruits if elite, else \(n_{sp}\). Each recruit samples
uniformly inside a box of radius \(\eta_i\) around the site:

\[
x^{\text{rec}} \sim \mathcal{U}\!\left(x_i - \eta_i \cdot R,\;
                                       x_i + \eta_i \cdot R\right),
\qquad R = \text{upper} - \text{lower}
\]

Only the **best** recruit of a site is kept, and only if it beats the
site itself — a greedy, per-site competition.

**3. Progressive neighborhood shrinking.** A site that fails to improve
narrows its search radius:

\[
\eta_i \leftarrow
\begin{cases}
\eta_i & \text{if the site improved}\\
\alpha\,\eta_i & \text{otherwise}
\end{cases}
\qquad 0 < \alpha \le 1
\]

The radius is deliberately **not** reset after a success, so a
productive site keeps refining at an ever-finer scale. This detail
matters enormously: resetting it on every success leaves the algorithm
stuck around 3e-02 on the Sphere function, while letting it shrink
reaches ~5e-14.

**4. Site abandonment.** After `stagnation_limit` consecutive failures
the site is discarded and replaced by a fresh random scout, with its
radius restored to \(\eta_0\) — the mechanism that prevents the colony
from over-committing to an exhausted basin.

**5. Global search.** The remaining \(n - m\) scouts are re-sampled
uniformly across the whole space every iteration, which is what keeps
new regions entering the competition.

## Pseudocode

```text
input: scouts n, sites m, elite sites e, recruits nep / nsp,
       initial radius eta0, shrink alpha, stagnation limit L
x    <- n uniform random solutions, evaluated and sorted
eta  <- eta0 for every site
s[i] <- 0                                     # stagnation counters

repeat until the budget is exhausted:
    for i = 1 .. m:                           # neighborhood search
        recruits <- nep if i <= e else nsp                        (eq. 1)
        best_find <- best of `recruits` samples around x[i]       (eq. 2)
        if best_find is better than x[i]:
            x[i] <- best_find;  s[i] <- 0     # radius left as-is (eq. 3)
        else:
            eta[i] <- alpha * eta[i];  s[i] <- s[i] + 1           (eq. 3)
            if s[i] >= L:                                         (eq. 4)
                x[i] <- a fresh random solution
                eta[i] <- eta0;  s[i] <- 0

    for i = m+1 .. n:                         # global search
        x[i] <- a fresh random solution, evaluated                (eq. 5)

    sort sites by fitness

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 25 | Scout bees |
| `selected_sites` | \(m\) | 12 | Sites given a local search |
| `elite_sites` | \(e\) | 4 | Of those, how many count as elite |
| `elite_bees` | \(n_{ep}\) | 8 | Recruits per elite site |
| `selected_bees` | \(n_{sp}\) | 4 | Recruits per non-elite site |
| `neighborhood` | \(\eta_0\) | 0.1 | Initial radius (fraction of the bound range) |
| `shrink` | \(\alpha\) | 0.9 | Radius factor after a failed search |
| `stagnation_limit` | \(L\) | 15 | Failures before a site is abandoned |
| `seed` | — | `None` | Reproducibility |

One iteration costs roughly
\(e \cdot n_{ep} + (m - e) \cdot n_{sp} + (n - m)\) evaluations, so the
site counts directly control how the budget is spent between
exploitation and exploration.

## Behavior

The Bees Algorithm is a **balanced performer** on the benchmarks
(Sphere ~5e-14, Ackley ~5e-06, Rastrigin ~22.9): the shrinking radii
give it fine local refinement, while the constant stream of fresh
scouts keeps some exploration alive. It is neither as precise as ACO-R
on smooth functions nor as strong as ABC on Rastrigin, but it is
competitive on all three — and its explicit recruitment makes it easy
to reason about where the evaluation budget goes.

```python
from ikn_library import Task
from ikn_library.problems import Ackley
from ikn_library.algorithms import BeesAlgorithm

task = Task(problem=Ackley(dimension=10), max_evals=20000)
algo = BeesAlgorithm(population_size=25, selected_sites=12, elite_sites=4, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- D. T. Pham, A. Ghanbarzadeh, E. Koç, S. Otri, S. Rahim, and M. Zaidi,
  "The Bees Algorithm," Technical Note, Manufacturing Engineering
  Centre, Cardiff University, UK, 2005.
- D. T. Pham and M. Castellani, "The Bees Algorithm: modelling foraging
  behaviour to solve continuous optimisation problems," *Proceedings of
  the Institution of Mechanical Engineers, Part C: Journal of Mechanical
  Engineering Science*, 223(12), 2919-2938, 2009.
  [doi:10.1243/09544062JMES1494](https://doi.org/10.1243/09544062JMES1494).
- D. T. Pham and M. Castellani, "A comparative study of the Bees
  Algorithm as a tool for function optimisation," *Cogent Engineering*,
  2(1), 1091540, 2015.
  [doi:10.1080/23311916.2015.1091540](https://doi.org/10.1080/23311916.2015.1091540).
