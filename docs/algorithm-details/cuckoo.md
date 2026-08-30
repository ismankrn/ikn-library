# Cuckoo Search (CS)

**Cuckoo Search** (Yang & Deb, 2009) is built on brood parasitism: some
cuckoo species lay their eggs in other birds' nests, and the host
either raises the foreign chick or discovers it and abandons the nest.
Each nest here holds one candidate solution.

What sets this algorithm apart from every other one in this library is
its step distribution: instead of Gaussian noise it uses a **Lévy
flight**, a heavy-tailed random walk. Most steps are tiny, but a few
are enormous — so the search refines locally while retaining the
ability to leap into completely unexplored regions.

## Equations

**1. Lévy step (Mantegna's algorithm).** Drawing directly from a Lévy
distribution is awkward, so it is synthesized from two normal
variables:

\[
s = \frac{u}{\lvert v \rvert^{1/\beta}},
\qquad u \sim \mathcal{N}(0, \sigma_u^{2}),
\qquad v \sim \mathcal{N}(0, 1)
\]

\[
\sigma_u = \left[
\frac{\Gamma(1+\beta)\,\sin\!\left(\pi\beta/2\right)}
     {\Gamma\!\left(\frac{1+\beta}{2}\right)\,\beta\,2^{(\beta-1)/2}}
\right]^{1/\beta}
\]

The exponent \(\beta \in (0, 2]\) controls the tail: smaller \(\beta\)
means longer jumps (\(\beta = 2\) degenerates to a Gaussian).

**2. Laying an egg.** A cuckoo flies from nest \(i\) and lays a new
egg, then drops it into a **randomly chosen** nest \(t\), which keeps
it only if it is better:

\[
x^{\text{new}} = x_i + \alpha \, (\text{upper} - \text{lower}) \odot s,
\qquad
x_t \leftarrow x^{\text{new}} \ \text{ if } f(x^{\text{new}}) < f(x_t)
\]

Dropping the egg into a *random* nest rather than the parent's own is
what spreads good solutions through the population.

**3. Discovery and abandonment.** A fraction \(p_a\) of the worst nests
are found out by the hosts and rebuilt by a biased random walk between
two other nests:

\[
x_i^{\text{new}} = x_i + r \odot \bigl(x_j - x_k\bigr),
\qquad r \sim \mathcal{U}(0,1)^m
\]

with \(j, k\) chosen at random. This is the algorithm's diversification
mechanism, analogous to mutation in a GA.

## Pseudocode

```text
input: nests n, discovery rate pa, step scale alpha, Levy exponent beta
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    for each nest i:                        # a cuckoo lays an egg
        s    <- Levy(beta) step vector                          (eq. 1)
        cand <- repair(x[i] + alpha * (upper - lower) * s)      (eq. 2)
        t    <- a random nest index
        if cand is better than x[t]:
            x[t] <- cand

    for the pa * n worst nests:              # hosts discover the eggs
        j, k  <- two random nest indices
        cand  <- repair(x[i] + Uniform(0,1) * (x[j] - x[k]))    (eq. 3)
        if cand is better than x[i]:
            x[i] <- cand

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 15 | Number of nests |
| `discovery_rate` | \(p_a\) | 0.5 | Fraction of worst nests rebuilt each iteration |
| `step_size` | \(\alpha\) | 0.01 | Lévy step scale, as a fraction of the bound range |
| `levy_exponent` | \(\beta\) | 1.5 | Tail heaviness; smaller = longer jumps |
| `seed` | — | `None` | Reproducibility |

!!! note "Two deviations from the common formulation"
    - **The step is scaled to the search range, not to \(x_i - x^{*}\).**
      Many write-ups multiply the Lévy step by the distance to the best
      nest. That distance collapses to zero once the nests converge, and
      in benchmarking it stalled the search completely (Sphere ≈ 16
      versus ≈ 4e-10 with the form used here).
    - **`discovery_rate` defaults to 0.5, not the paper's 0.25.** With
      the higher rate, Sphere improved from ≈ 2e-05 to ≈ 2e-07 and
      Rastrigin from 12.7 to 7.1 — this implementation leans more on
      abandonment for its exploration.

## Behavior

Cuckoo Search is a **balanced performer** with a distinctly good
multimodal profile: Sphere ≈ 4e-10, Ackley ≈ 3e-04, Rastrigin ≈ 6.8 —
fourth-best on Rastrigin behind ABC, GA, and the Camel Algorithm, while
remaining precise on smooth landscapes. The Lévy tail is doing exactly
what it promises: enough long jumps to escape local optima, enough
small steps to polish a solution.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import CuckooSearch

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = CuckooSearch(population_size=15, discovery_rate=0.5, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- X.-S. Yang and S. Deb, "Cuckoo search via Lévy flights," in
  *Proceedings of the World Congress on Nature and Biologically
  Inspired Computing (NaBIC 2009)*, IEEE, 210-214, 2009.
  [doi:10.1109/NABIC.2009.5393690](https://doi.org/10.1109/NABIC.2009.5393690).
- X.-S. Yang and S. Deb, "Engineering optimisation by cuckoo search,"
  *International Journal of Mathematical Modelling and Numerical
  Optimisation*, 1(4), 330-343, 2010.
  [doi:10.1504/IJMMNO.2010.035430](https://doi.org/10.1504/IJMMNO.2010.035430).
- R. N. Mantegna, "Fast, accurate algorithm for numerical simulation of
  Lévy stable stochastic processes," *Physical Review E*, 49(5),
  4677-4683, 1994 (the sampling method of equation 1).
  [doi:10.1103/PhysRevE.49.4677](https://doi.org/10.1103/PhysRevE.49.4677).
