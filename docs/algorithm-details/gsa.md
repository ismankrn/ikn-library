# Gravitational Search Algorithm (GSA)

**GSA** (Rashedi et al., 2009) is the only algorithm in this library
built on a law of physics rather than on animal behaviour. Candidate
solutions are **masses** in a search space governed by Newton's law of
gravitation, and every agent's mass grows with its fitness.

That single fact produces the whole search dynamic, and it does so with
an elegance worth pointing out: mass appears **twice** in Newtonian
mechanics, as *gravitational* mass (how strongly you attract others)
and as *inertial* mass (how hard you are to move). So a good solution
both pulls the swarm toward itself **and** resists being moved — the
best agents are naturally conservative, while light, poor agents are
flung around and explore.

## Equations

**1. Gravitational constant.** \(G\) decays exponentially over the run,
weakening all attraction and letting the swarm settle:

\[
G(t) = G_0 \, e^{-\alpha\, t / T}
\]

**2. Mass.** Fitness is normalized so the best agent has mass 1 and the
worst has 0, then scaled to sum to one:

\[
m_i = \frac{f_{\text{worst}} - f(x_i)}{f_{\text{worst}} - f_{\text{best}}},
\qquad
M_i = \frac{m_i}{\sum_j m_j}
\]

**3. Force.** The force agent \(j\) exerts on agent \(i\) follows the
inverse-distance law (GSA uses \(R\) rather than \(R^2\), which the
authors found works better in practice):

\[
F_{ij}^{d} = G(t)\,\frac{M_i M_j}{R_{ij} + \varepsilon}\,
\bigl(x_j^{d} - x_i^{d}\bigr)
\]

**4. Acceleration.** By \(a = F/M\), the agent's own mass **cancels**:

\[
a_i^{d} = \sum_{j \in \text{Kbest},\, j \neq i}
r_j \, G(t) \, \frac{M_j}{R_{ij} + \varepsilon}\,
\bigl(x_j^{d} - x_i^{d}\bigr)
\]

with \(r_j \sim \mathcal{U}(0,1)\) adding stochasticity. Only the
**Kbest** heaviest agents exert force, and Kbest shrinks linearly from
the whole population to `final_kbest`, so the swarm gradually listens
only to its elite.

**5. Motion.** Velocity keeps a random fraction of its previous value:

\[
v_i^{d} \leftarrow r_i \, v_i^{d} + a_i^{d},
\qquad
x_i^{d} \leftarrow x_i^{d} + v_i^{d}
\]

## Pseudocode

```text
input: agents n, G0, decay alpha, final Kbest, velocity limit
x <- n random solutions, evaluated;  v <- 0

repeat until the budget is exhausted:
    G     <- G0 * exp(-alpha * progress)                        (eq. 1)
    M     <- normalized masses from fitness                     (eq. 2)
    Kbest <- population_size shrinking toward final_kbest

    for each agent i:
        a[i] <- sum over the Kbest heaviest j != i of
                r * G * M[j] / (dist(i,j) + eps) * (x[j] - x[i])
                                                                (eq. 4)
    v <- clip(r * v + a, ±v_max)                                (eq. 5)
    x <- repair(x + v), re-evaluated

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 50 | Number of agents |
| `g0` | \(G_0\) | 100.0 | Initial gravitational constant |
| `alpha` | \(\alpha\) | 30.0 | Decay rate of \(G\); larger = faster settling |
| `final_kbest` | — | 1 | Attracting agents at the end of the run |
| `max_velocity` | — | 0.5 | Velocity limit (fraction of the bound range) |
| `seed` | — | `None` | Reproducibility |

Distances are divided by the search range internally, so `g0` means the
same thing whatever bounds a problem uses.

!!! note "`alpha` is the parameter that matters — and small budgets mislead"
    Screening at 5,000 evaluations picked `alpha=10`, with the paper's
    `alpha=20` looking poor (Sphere ≈ 0.09). At the full 20,000-evaluation
    budget the ranking **inverted completely**:

    | `alpha` | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | 10 | 9e-08 | 8.96 | 3e-03 |
    | 20 (paper) | 8e-16 | 5.17 | 3e-07 |
    | **30** | **7e-24** | **3.78** | **2e-11** |
    | 60 | 0.123 | 6.39 | 0.583 |

    A fast-decaying \(G\) only pays off when there are enough iterations
    left to exploit the settled swarm — which a short run does not have.
    This is the same trap the [Fish School Search](fss.md) page
    documents: screen cheaply if you must, but confirm at the budget you
    will actually use.

## Behavior

GSA is a **strong all-rounder**: Sphere ≈ 7e-24, Ackley ≈ 2e-11,
Rastrigin ≈ 3.8 — mid-to-upper field on every benchmark without topping
any. The shrinking Kbest is what keeps it competitive on the multimodal
function: early on the whole population attracts, preserving diversity,
and only later does the swarm collapse onto its elite.

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import GravitationalSearchAlgorithm

task = Task(problem=Sphere(dimension=10), max_evals=20000)
algo = GravitationalSearchAlgorithm(population_size=50, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- E. Rashedi, H. Nezamabadi-pour, and S. Saryazdi, "GSA: a
  gravitational search algorithm," *Information Sciences*, 179(13),
  2232-2248, 2009.
  [doi:10.1016/j.ins.2009.03.004](https://doi.org/10.1016/j.ins.2009.03.004).
- E. Rashedi, E. Rashedi, and H. Nezamabadi-pour, "A comprehensive
  survey on gravitational search algorithm," *Swarm and Evolutionary
  Computation*, 41, 141-158, 2018.
  [doi:10.1016/j.swevo.2018.02.018](https://doi.org/10.1016/j.swevo.2018.02.018).
