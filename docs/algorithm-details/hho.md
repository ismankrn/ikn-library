# Harris Hawks Optimization (HHO)

**HHO** (Heidari et al., 2019) models the cooperative "pounce" by which
Harris's hawks hunt, and it is the most **branched** algorithm in this
library: six distinct moves, selected by a two-level test on how much
energy the prey has left.

While the prey is strong the hawks scatter; as its energy drains they
close in, and *how* they close in depends both on the remaining energy
and on whether the prey manages to bolt.

| Prey bolts? | Energy | Move |
|---|---|---|
| — | \(\lvert E \rvert \ge 1\) | explore |
| no | \(\lvert E \rvert \ge 0.5\) | soft besiege |
| no | \(\lvert E \rvert < 0.5\) | hard besiege |
| yes | \(\lvert E \rvert \ge 0.5\) | soft besiege + rapid dives |
| yes | \(\lvert E \rvert < 0.5\) | hard besiege + rapid dives |

The two **dive** moves are the distinctive part. They build *two*
candidates — a direct approach and a Lévy-flight zigzag — evaluate both,
and keep whichever improves on the hawk, or neither. No other algorithm
here spends extra evaluations comparing alternatives before committing.

## Equations

**1. Escaping energy.** One quantity drives every branch:

\[
E = 2 E_0 \left(1 - \frac{\text{evals}}{\text{max\_evals}}\right),
\qquad E_0 \sim \mathcal{U}(-1, 1)
\]

Because the envelope falls to zero, \(\lvert E \rvert \ge 1\) becomes
impossible partway through — exploration switches itself off. Like
[GWO](gwo.md), this schedule was budget-tied in the original paper.

**2. Exploration** \((\lvert E \rvert \ge 1)\), one of two at random:

\[
X \leftarrow X_{\text{rand}} - r_1 \lvert X_{\text{rand}} - 2 r_2 X \rvert
\qquad\text{or}\qquad
X \leftarrow (X_{\text{prey}} - X_{\text{mean}}) - r_3\bigl(l + r_4 (u - l)\bigr)
\]

**3. Soft and hard besiege**, with jump strength \(J = 2(1 - r_5)\):

\[
X \leftarrow (X_{\text{prey}} - X) - E \lvert J X_{\text{prey}} - X \rvert
\qquad\text{(soft)}
\]
\[
X \leftarrow X_{\text{prey}} - E \lvert X_{\text{prey}} - X \rvert
\qquad\text{(hard)}
\]

**4. Rapid dives.** The anchor is the hawk itself when soft, the flock
mean when hard:

\[
Y = X_{\text{prey}} - E \lvert J X_{\text{prey}} - A \rvert,
\qquad
Z = Y + S \odot \text{LF}(D)
\]

and the hawk moves to \(Y\) or \(Z\) only if that beats its current
position.

## Pseudocode

```text
input: hawks n, energy_start, Levy exponent beta, Levy scale
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    prey <- best hawk ;  mean <- flock centroid
    envelope <- energy_start * (1 - evals / max_evals)           (eq. 1)

    for each hawk i:
        E <- envelope * U(-1, 1)
        if |E| >= 1:
            x[i] <- one of the two exploration moves              (eq. 2)
        else if not bolts:
            x[i] <- soft or hard besiege by |E| vs 0.5            (eq. 3)
        else:
            Y <- direct approach ;  Z <- Y + Levy zigzag          (eq. 4)
            x[i] <- best of {Y, Z} if it beats x[i], else unchanged

return best solution found
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 30 | Number of hawks |
| `energy_start` | 2.0 | Initial escaping-energy envelope (paper's value) |
| `levy_exponent` | 1.5 | Lévy exponent for the dive zigzag |
| `levy_scale` | 0.002 | Lévy step scale; the paper uses 0.01 |
| `seed` | `None` | Reproducibility |

!!! success "The strongest algorithm measured here on hard problem variants"
    HHO reaches Sphere ≈ 1e-88, **Rastrigin exactly 0 on every seed**,
    and Ackley at machine precision. Those headline numbers would
    normally warrant the scepticism the [GWO](gwo.md) and
    [FOA](foa.md) pages describe — so they were checked against a
    Rastrigin that is both **rotated and shifted**, which removes both
    the separability and the origin advantages at once:

    | Algorithm | Rastrigin | rotated | shifted | rotated + shifted |
    |---|---|---|---|---|
    | **HHO** | **0** | **0** | **0.017** | **0.006** |
    | [GWO](gwo.md) | 0 | 10.8 | 15.6 | 22.4 |
    | [DE](de.md) | 1.99 | 28.9 | 2.65 | 19.1 |
    | [FPA](fpa.md) | 7.34 | 10.6 | 7.30 | 17.6 |
    | [HS](hs.md) | 0.018 | 26.3 | 0.075 | 30.4 |

    HHO is roughly **three thousand times** better than the next
    algorithm on the hardest variant. Unlike the others, its advantage
    survives every transformation, which is strong evidence that the
    six-branch structure is doing real work rather than exploiting the
    benchmark.

!!! warning "It does still carry an origin bias — just a survivable one"
    On Sphere, moving the optimum off zero costs HHO **84 orders of
    magnitude** (8e-89 → 4e-05), the same structural problem
    [GWO](gwo.md) has, and for the same reason: terms like
    \(\lvert J X_{\text{prey}} - X \rvert\) scale the prey's *absolute*
    coordinates, so the operator is not translation-invariant.

    The difference is severity. GWO's Rastrigin goes from 0 to 15.6 on
    a shift; HHO's goes from 0 to 0.017. The bias is present but the
    algorithm is strong enough elsewhere to absorb it. Still, treat the
    1e-88 figure as a benchmark artefact rather than a capability.

!!! note "Tuning notes"
    - **`levy_scale` is lowered from the paper's 0.01 to 0.002.** It
      makes no difference on the standard suite but consistently helps
      on the hard variants (rotated+shifted 0.0023 vs 0.0058, shifted
      Sphere 7e-06 vs 3e-05, shifted Ackley 0.007 vs 0.015, 5 seeds).
    - **`energy_start` must not go below 2.** At 1.5 the rotated+shifted
      result collapses from 0.006 to 7.3, because exploration switches
      off too early. Raising it to 3 is roughly neutral.
    - **Population size barely matters** — 15, 30, 50 and 80 all land
      within a factor of three on every problem tested, which is
      unusual and makes the algorithm easy to deploy.

## Behavior

HHO is the best all-round performer in this library on the measurements
taken here, and the only one whose ranking is unchanged by rotating and
shifting the benchmark.

The likely reason is redundancy. Six moves with different geometry mean
that no single structural weakness — coordinate-wise bias, dependence on
one attractor, premature convergence — can dominate, because some other
branch keeps firing. The cost is complexity: HHO has by far the most
branching of anything here, and the dive moves consume two evaluations
each, so it does less per evaluation than a simpler method.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import HarrisHawksOptimization

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = HarrisHawksOptimization(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- A. A. Heidari, S. Mirjalili, H. Faris, I. Aljarah, M. Mafarja, and
  H. Chen, "Harris hawks optimization: algorithm and applications,"
  *Future Generation Computer Systems*, 97, 849-872, 2019.
  [doi:10.1016/j.future.2019.02.028](https://doi.org/10.1016/j.future.2019.02.028).
- R. N. Mantegna, "Fast, accurate algorithm for numerical simulation of
  Lévy stable stochastic processes," *Physical Review E*, 49(5),
  4677-4683, 1994.
  [doi:10.1103/PhysRevE.49.4677](https://doi.org/10.1103/PhysRevE.49.4677).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010 — on why rotated
  and shifted variants belong in any benchmark suite.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
