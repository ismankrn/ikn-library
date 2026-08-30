# Coral Reefs Optimization (CRO)

**CRO** (Salcedo-Sanz et al., 2014) models a coral reef colonising a
rocky bed. It is the only algorithm in this library whose population
lives on an explicit **substrate**: a fixed number of squares, each
either holding one coral (a solution) or lying empty.

Every other algorithm here carries a constant population. CRO does not
— the reef's occupancy rises and falls, and that single design choice
changes what selection *means*. Elsewhere, a new solution is compared
against the population and ranked. Here it must **find a square and
hold it**, which makes survival partly a matter of where a larva
happens to land.

## Equations

**1. Broadcast spawning.** A fraction \(F_b\) of the living corals
release gametes; they are paired at random and blended, with the weight
drawn outside \([0,1]\) so offspring may fall beyond either parent:

\[
x_{\text{larva}} = x_a + w \odot (x_b - x_a),
\qquad w \sim \mathcal{U}(-0.25,\, 1.25)^d
\]

This is the algorithm's only recombination, and it **self-scales**: as
the reef converges, \(\lVert x_b - x_a \rVert\) shrinks and the larvae
land closer to their parents automatically.

**2. Brooding.** The remaining \(1 - F_b\) corals self-fertilise,
perturbing the parent by a step that decays with the spent budget:

\[
x_{\text{larva}} = x_i + \sigma^{(t)} (u - l) \odot \mathcal{N}(0, I),
\qquad
\sigma^{(t)} = \sigma_0 \left(\max\left(1 - \tfrac{\text{evals}}{\text{max\_evals}},\ 10^{-6}\right)\right)^{2}
\]

**3. Settlement.** Each larva picks a random square \(s\) and gets
\(\kappa\) attempts to take hold:

\[
\text{settle if } \quad \neg\,\text{occupied}(s)
\quad\text{or}\quad f(x_{\text{larva}}) < f(x_s)
\]

A larva that never wins in \(\kappa\) attempts is **lost**, however good
it was. This is the mechanism that distinguishes CRO: selection is
local and stochastic rather than a global ranking.

**4. Depredation.** With probability \(P_d\), each of the worst \(F_d\)
corals is eaten and its square freed. Without this the reef saturates
and admits only strict improvements.

## Pseudocode

```text
input: reef capacity N, occupation rho0, Fb, Fa, Fd, Pd, attempts k
reef <- N squares; occupy rho0*N of them with random solutions

repeat until the budget is exhausted:
    split the living corals into Fb broadcasters and the rest
    larvae <- pairwise blend crossover of the broadcasters      (eq. 1)
    larvae += mutated copies of the brooders                    (eq. 2)
    larvae += clones of the fittest Fa corals        # budding, off by default

    shuffle the larvae
    for each larva:
        try up to k random squares; take an empty one,
        or displace an occupant it beats; otherwise the larva dies   (eq. 3)

    each of the worst Fd corals is eaten with probability Pd    (eq. 4)

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(N\) | 80 | Reef **capacity** (squares), not live corals |
| `initial_occupation` | \(\rho_0\) | 0.8 | Fraction of squares occupied at the start |
| `broadcast_fraction` | \(F_b\) | 0.5 | Share reproducing by crossover vs brooding |
| `asexual_fraction` | \(F_a\) | 0.0 | Fittest corals that clone themselves |
| `depredation_fraction` | \(F_d\) | 0.1 | Worst corals eligible to be eaten |
| `depredation_prob` | \(P_d\) | 0.1 | Chance an eligible coral is eaten |
| `settlement_attempts` | \(\kappa\) | 3 | Tries a larva gets to find a square |
| `mutation_scale` | \(\sigma_0\) | 0.03 | Brooding step (fraction of the bound range) |
| `seed` | — | `None` | Reproducibility |

!!! warning "Budding is off by default — it collapses the search"
    Asexual reproduction is one of the algorithm's five named
    mechanisms, and switching it on makes CRO **markedly worse** (5
    seeds, 20,000 evaluations):

    | `asexual_fraction` | Sphere | Rastrigin | Ackley |
    |---|---|---|---|
    | **0.0 (default)** | 2e-07 | **3.6** | **4e-03** |
    | 0.02 | 1e-07 | 10.0 | 0.234 |
    | 0.05 | 1e-07 | 14.7 | 3e-03 |
    | 0.1 | 1e-07 | 20.3 | 0.464 |

    The reason is a direct interaction with settlement. Budding clones
    the *current best* corals, and those clones are by construction good
    enough to displace most incumbents — so they win their squares and
    the reef fills with **duplicates of one solution**. Crossover then
    blends near-identical parents, \(x_b - x_a \to 0\), and equation 1
    stops generating anything new. The mechanism meant to exploit good
    solutions destroys the diversity that crossover needs.

    The parameter is kept so this collapse can be reproduced: set
    `asexual_fraction=0.1` and watch Rastrigin degrade by a factor of
    five. It is a compact illustration of why elitism and recombination
    have to be balanced against each other.

!!! note "Tuning notes"
    - **The published defaults are much weaker.** \(F_b = 0.9\),
      \(\rho_0 = 0.4\), \(N = 40\), \(\sigma_0 = 0.1\) with budding on
      gives Sphere 2e-06, Rastrigin 19.9, Ackley 1.16. The defaults here
      reach Rastrigin 3.0 and Ackley 4e-03.
    - **`broadcast_fraction` should not be 1.0.** Pure crossover has no
      way to introduce new material and converges prematurely (Sphere
      3e-04, Ackley 4.8). An even split with brooding works best.
    - **The reef saturates anyway.** Tracing occupancy at capacity 80:
      it starts at 75 and sits at 79–80 for the rest of the run.
      Depredation frees squares that are refilled within an iteration,
      so the empty-space dynamic matters mostly at the very start —
      which is why `initial_occupation` has little effect on the result.

## Behavior

CRO is **strong on the multimodal benchmark and mid-field on the smooth
ones**: Rastrigin ≈ 3.0 (fifth best in the library, with a tight 2.0–4.0
spread across seeds), Sphere ≈ 3e-07, Ackley ≈ 4e-03.

That profile follows from the settlement rule. Because a larva can be
lost simply by landing badly, and because depredation keeps removing the
worst corals, the reef never converges as aggressively as a
ranking-based algorithm — good for escaping Rastrigin's local optima,
costly when the last few digits of precision are what matter. Compare
[DE](de.md) or [FWA](fwa.md), which reach 1e-40 and below on Sphere but
achieve it by committing hard to the best region found.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import CoralReefsOptimization

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = CoralReefsOptimization(population_size=80, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-López, and
  J. A. Portilla-Figueras, "The coral reefs optimization algorithm: a
  novel metaheuristic for efficiently solving optimization problems,"
  *The Scientific World Journal*, 2014, 739768.
  [doi:10.1155/2014/739768](https://doi.org/10.1155/2014/739768).
- S. Salcedo-Sanz, "A review on the coral reefs optimization algorithm:
  new development lines and current applications," *Progress in
  Artificial Intelligence*, 6(1), 1-15, 2017.
  [doi:10.1007/s13748-016-0104-2](https://doi.org/10.1007/s13748-016-0104-2).
