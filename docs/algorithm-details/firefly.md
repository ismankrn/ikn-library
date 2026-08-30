# Firefly Algorithm (FA)

**FA** (Yang, 2008) is built on bioluminescent attraction. All
fireflies are treated as unisex, so any firefly is drawn toward any
**brighter** one — brightness being the objective value. The twist is
that light is absorbed by the medium, so attraction **fades with
distance**.

That decay is the algorithm's defining feature. With a strong
absorption coefficient a firefly only notices its close neighbours, and
the swarm naturally splits into subgroups that explore different
regions in parallel — which is why Yang proposed the method
specifically for **multimodal** problems.

## Equations

**1. Attractiveness.** Between fireflies at distance \(r\):

\[
\beta(r) = \beta_0 \, e^{-\gamma r^{2}},
\qquad r_{ij} = \lVert x_i - x_j \rVert
\]

\(\beta_0\) is the attractiveness at zero distance and \(\gamma\) the
light absorption coefficient. In this implementation \(\gamma\) is
divided by the squared mean bound range, so the same value behaves
consistently whatever the problem's scale.

**2. Movement.** Firefly \(i\) moves toward every brighter firefly
\(j\), then takes a random walk:

\[
x_i \leftarrow x_i
+ \beta(r_{ij})\,(x_j - x_i)
+ \alpha^{(t)} \, (\text{upper} - \text{lower}) \odot
\left(\mathcal{U}(0,1)^m - \tfrac{1}{2}\right)
\]

The random term is essential: without it the swarm collapses onto the
brightest individual within a few iterations.

**3. Decaying randomization.** The random walk shrinks geometrically,
turning exploration into refinement:

\[
\alpha^{(t+1)} = \theta \, \alpha^{(t)}, \qquad 0 < \theta \le 1
\]

## Pseudocode

```text
input: fireflies n, randomization alpha, decay theta, beta0, gamma
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    sort x by brightness (best first)
    for each firefly i:
        pull  <- combined attraction toward all brighter fireflies (eq. 1, 2)
        cand  <- repair(pull + alpha * (upper - lower) * (U(0,1) - 0.5))
        if cand is brighter than x[i]:
            x[i] <- cand
    alpha <- theta * alpha                                        (eq. 3)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 25 | Number of fireflies |
| `alpha` | \(\alpha\) | 1.0 | Initial randomization (fraction of the bound range) |
| `alpha_decay` | \(\theta\) | 0.92 | Geometric decay of the randomization |
| `beta0` | \(\beta_0\) | 1.0 | Attractiveness at zero distance |
| `gamma` | \(\gamma\) | 1.0 | Light absorption; larger = shorter sight |
| `seed` | — | `None` | Reproducibility |

!!! note "Implementation notes"
    - **The attractions are combined, not applied one by one.** The
      textbook version updates a firefly's position inside a nested
      loop over every brighter firefly, which makes each iteration
      \(O(n^2)\) in pure Python. Here the individual pulls are merged
      into a single weighted move toward the brighter fireflies'
      centroid — equivalent in spirit, and fast enough that a full
      20,000-evaluation benchmark runs in about a second instead of
      minutes.
    - **`alpha_decay` matters enormously.** At 0.99 the swarm never
      settles (Sphere ≈ 0.17); at 0.92 it reaches ≈ 7e-57. The decay
      is what converts the random walk from an exploration device into
      a refinement one.

## Behavior

The Firefly Algorithm holds the library's **best result on the smooth
benchmarks**:

| Function | FA | Runner-up |
|---|---|---|
| Sphere | **7e-57** | 2e-41 (DE) |
| Ackley | **2e-15** | 5e-15 (DE) |
| Rastrigin | 6.0 | 2e-08 (ABC) |

The combination of distance-weighted attraction and a shrinking random
walk gives it extraordinary precision once the swarm has located the
right basin. On Rastrigin it is mid-field: the swarm tends to agree on
one basin before the randomization has decayed enough to explore
others.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import FireflyAlgorithm

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = FireflyAlgorithm(population_size=25, alpha=1.0, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- X.-S. Yang, *Nature-Inspired Metaheuristic Algorithms*, Luniver
  Press, 2008.
- X.-S. Yang, "Firefly algorithms for multimodal optimization," in
  *Stochastic Algorithms: Foundations and Applications (SAGA 2009)*,
  Lecture Notes in Computer Science 5792, Springer, 169-178, 2009.
  [doi:10.1007/978-3-642-04944-6_14](https://doi.org/10.1007/978-3-642-04944-6_14).
- I. Fister, I. Fister Jr., X.-S. Yang, and J. Brest, "A comprehensive
  review of firefly algorithms," *Swarm and Evolutionary Computation*,
  13, 34-46, 2013.
  [doi:10.1016/j.swevo.2013.06.001](https://doi.org/10.1016/j.swevo.2013.06.001).
