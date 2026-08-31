# Hybrid Bat Algorithm (HBA)

**HBA** (Fister Jr., Fister & Yang, 2013) is the [Bat
Algorithm](bat.md) with exactly one operator swapped: its local random
walk around the best solution is replaced by a **Differential Evolution
move**, mutation plus binomial crossover.

Everything else is untouched — the frequency-tuned velocity, the growing
pulse rate that decides when the local step fires, and the
loudness-based acceptance. The implementation says so directly: it
subclasses `BatAlgorithm` and overrides a single method, and there is a
test asserting that `run_iteration` is inherited unchanged.

The reason for the swap is worth understanding. Plain BA's local step is
a Gaussian walk of a hand-tuned size around the current best, so it
searches only where the best already is. A DE move instead takes its
scale from the **spread of the population**, so it contracts on its own
as the search converges and needs no schedule at all — a test asserts
the hybrid's local step ignores the walk-scale argument entirely.

## Equations

**1. Bat motion** (unchanged from [BA](bat.md)):

\[
f_i \sim \mathcal{U}(f_{\min}, f_{\max}),
\qquad
v_i \leftarrow v_i + (x_* - x_i) f_i,
\qquad
x' = x_i + v_i
\]

**2. Local step — the substitution.** When the pulse rate fires, plain
BA takes a Gaussian walk; HBA builds a DE/rand/1/bin trial instead:

\[
d = x_{r_1} + F\,(x_{r_2} - x_{r_3}),
\qquad
x'_k =
\begin{cases}
d_k & \text{with probability } CR \\
x_{i,k} & \text{otherwise}
\end{cases}
\]

with \(r_1, r_2, r_3\) distinct and at least one coordinate always taken
from the donor.

**3. Acceptance** (unchanged): a trial is kept only if it improves *and*
a loudness draw succeeds, after which that bat's loudness decays by
\(\alpha\).

## Pseudocode

```text
input: bats n, F, CR, loudness A0, pulse rate r0, alpha, gamma
x <- n random solutions, evaluated;  v <- 0;  A <- A0

repeat until the budget is exhausted:
    rate <- r0 * (1 - exp(-gamma * iters))
    for each bat i:
        f <- U(f_min, f_max)
        v[i] <- v[i] + (best - x[i]) * f
        trial <- x[i] + v[i]                                     (eq. 1)
        if rand() > rate:
            trial <- DE/rand/1/bin from the population           (eq. 2)
        if f(trial) <= f(x[i]) and rand() < A[i]:                (eq. 3)
            x[i] <- trial ;  A[i] <- alpha * A[i]

return best solution found
```

## Parameters

| Parameter | Default | Paper | Effect |
|---|---|---|---|
| `population_size` | 50 | 30 | Number of bats |
| `differential_weight` | 0.3 | 0.5 | DE scale factor \(F\) |
| `crossover_rate` | 0.9 | 0.9 | Binomial crossover \(CR\) |
| `loudness` | 1.0 | 1.0 | Initial acceptance probability |
| `pulse_rate` | 0.1 | 0.5 | How often the DE step fires |
| `alpha` | 0.99 | 0.9 | Loudness decay per accepted move |
| `gamma` | 0.9 | 0.9 | Growth of the pulse rate |
| `seed` | `None` | — | Reproducibility |

!!! success "The one hybrid here that earns its keep — after retuning"
    The interesting question for any hybrid is whether it beats **both**
    of its parts. Measured on all four benchmark variants (Rastrigin,
    20,000 evaluations, 5 seeds):

    | Algorithm | plain | rotated | shifted | rot + shift |
    |---|---|---|---|---|
    | **HBA (tuned)** | 3.86 | **10.5** | 3.75 | **10.3** |
    | HBA (as published) | 30.9 | 36.9 | 28.5 | 30.3 |
    | plain [Bat](bat.md) | 41.6 | 36.6 | 21.3 | 27.1 |
    | plain [DE](de.md) | **2.79** | 29.3 | **3.18** | 23.0 |

    Two readings, and both matter.

    First, the swap works: tuned HBA improves on plain Bat by a factor
    of eleven on plain Rastrigin, and it beats **plain DE on both
    rotated variants** — 10.5 against 29.3, and 10.3 against 23.0. Its
    profile is also unusually flat (3.8 to 10.5 across all four), second
    only to [KH](kh.md) in this library. DE still wins on the two
    axis-aligned variants, so the hybrid trades peak performance for
    robustness.

    Second, and less comfortable: **the published parameters barely
    help**. As published, HBA scores 30.9 against plain Bat's 41.6 — an
    improvement, but not the kind that justifies a new algorithm. The
    gain reported above needs `alpha` raised from 0.9 to 0.99, `F`
    lowered to 0.3, `pulse_rate` cut to 0.1, and a larger population.

!!! note "Tuning notes"
    - **`alpha` is the parameter that matters most.** At 0.9 the
      loudness decays so fast that acceptance becomes near-impossible
      early and the search freezes; raising it to 0.99 is worth eight
      orders of magnitude on Sphere (1.6e-03 → 3.3e-11). This is
      inherited from plain BA and likely affects it too.
    - **`pulse_rate=0.1` means the DE step fires about 90% of the
      time.** The hybrid works best when it is mostly the DE half — a
      finding worth sitting with, given that plain DE then beats it on
      axis-aligned problems.
    - **Keep `CR` at the paper's 0.9.** Lowering it to 0.2 produces a
      spectacular plain-Rastrigin score (2e-10) that collapses to 30.1
      once rotated — the separability trap the [FOA](foa.md),
      [HS](hs.md) and [MKE](mke.md) pages document. `CR=0.9` gives the
      flat profile above. This is one of the few cases in this library
      where the published value survives the four-variant check and the
      tuned one does not.

## Behavior

On the standard suite HBA reaches Sphere ≈ 5e-04, Rastrigin ≈ 3.86,
Ackley ≈ 1.68 — an unremarkable row, and Ackley in particular is weak.

Read across the four variants it is a different story: near-flat
performance and the best rotated-Rastrigin result of any algorithm here
except [KH](kh.md). If your problem's variables interact, this row is
worth more than several that look better in the table.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import HybridBatAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = HybridBatAlgorithm(population_size=50, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- I. Fister Jr., D. Fister, and X.-S. Yang, "A hybrid bat algorithm,"
  *Elektrotehniški Vestnik / Electrotechnical Review*, 80(1-2), 1-7,
  2013. [arXiv:1303.6310](https://arxiv.org/abs/1303.6310).
- X.-S. Yang, "A new metaheuristic bat-inspired algorithm," in *Nature
  Inspired Cooperative Strategies for Optimization (NICSO 2010)*,
  Studies in Computational Intelligence 284, Springer, 65-74, 2010.
  [doi:10.1007/978-3-642-12538-6_6](https://doi.org/10.1007/978-3-642-12538-6_6).
- R. Storn and K. Price, "Differential evolution — a simple and
  efficient heuristic for global optimization over continuous spaces,"
  *Journal of Global Optimization*, 11(4), 341-359, 1997.
  [doi:10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328).
- C. L. Camacho-Villalón, M. Dorigo, and T. Stützle, "Exposing the grey
  wolf, moth-flame, whale, firefly, bat, and antlion algorithms: six
  misleading optimization techniques inspired by bestial metaphors,"
  *International Transactions in Operational Research*, 30(6),
  2945-2971, 2023 — the critique covers the Bat Algorithm this hybrid
  builds on.
  [doi:10.1111/itor.13176](https://doi.org/10.1111/itor.13176).
