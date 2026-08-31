# Hybrid Self-Adaptive Bat Algorithm (HSABA)

**HSABA** (Fister Jr., Fong, Brest & Fister, 2014) takes the [Hybrid Bat
Algorithm](hybrid-bat.md) and makes its two control parameters
**self-adaptive**. Each bat carries its own loudness \(A_i\) and pulse
rate \(r_i\), and rather than following a schedule they are occasionally
re-drawn at random.

This is the self-adaptation scheme of **jDE** (Brest et al., 2006)
applied to the Bat Algorithm's parameters instead of Differential
Evolution's — Brest is a co-author of both papers.

The mechanism is worth understanding structurally rather than as a
metaphor. In plain [BA](bat.md) and in [HBA](hybrid-bat.md), loudness
only ever **decays**: every accepted move multiplies it by \(\alpha\),
so it slides toward zero and acceptance becomes near-impossible. That is
a one-way ratchet. Re-drawing \(A_i\) removes it — a bat whose loudness
has collapsed can be revived. The HBA page records that raising
`alpha` from 0.9 to 0.99 was worth eight orders of magnitude; HSABA
addresses the same problem by design rather than by tuning.

## Equations

**1. Self-adaptation**, per bat, before it moves:

\[
A_i \leftarrow
\begin{cases}
A_{\min} + \rho\,(A_{\max} - A_{\min}) & \text{if } \text{rand} < \tau_1 \\
A_i & \text{otherwise}
\end{cases}
\qquad
r_i \leftarrow
\begin{cases}
\text{rand} & \text{if } \text{rand} < \tau_2 \\
r_i & \text{otherwise}
\end{cases}
\]

**2. Bat motion** (unchanged from [BA](bat.md)):

\[
f_i \sim \mathcal{U}(f_{\min}, f_{\max}),
\qquad
v_i \leftarrow v_i + (x_* - x_i) f_i,
\qquad
x' = x_i + v_i
\]

**3. Local step** — the DE/rand/1/bin move inherited from
[HBA](hybrid-bat.md), fired when \(\text{rand} > r_i\), now using each
bat's **own** pulse rate rather than a shared schedule.

**4. Acceptance.** A trial is kept if it improves and a draw against
that bat's own \(A_i\) succeeds. There is no decay step.

## Pseudocode

```text
input: bats n, tau_1, tau_2, loudness range [A_min, A_max], F, CR
x <- n random solutions, evaluated;  v <- 0
A <- U(A_min, A_max) per bat;  r <- U(0, 1) per bat

repeat until the budget is exhausted:
    for each bat i:
        if rand() < tau_1:  A[i] <- U(A_min, A_max)               (eq. 1)
        if rand() < tau_2:  r[i] <- rand()                        (eq. 1)

        f <- U(f_min, f_max)
        v[i] <- v[i] + (best - x[i]) * f
        trial <- x[i] + v[i]                                      (eq. 2)
        if rand() > r[i]:
            trial <- DE/rand/1/bin from the population            (eq. 3)
        if f(trial) <= f(x[i]) and rand() < A[i]:                 (eq. 4)
            x[i] <- trial

return best solution found
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 30 | Number of bats |
| `tau_1` | 0.1 | Chance of re-drawing a bat's loudness |
| `tau_2` | 0.02 | Chance of re-drawing a bat's pulse rate |
| `min_loudness` | 0.99 | Lower end of the loudness range |
| `max_loudness` | 1.0 | Upper end of the loudness range |
| `differential_weight` | 0.3 | DE scale factor \(F\) |
| `crossover_rate` | 0.9 | Binomial crossover \(CR\) |
| `min_frequency` / `max_frequency` | 0.0 / 2.0 | Frequency range |
| `seed` | `None` | Reproducibility |

!!! warning "The self-adaptation does not clearly beat the hybrid it extends"
    The natural question for a paper that adds machinery to an existing
    algorithm is whether the addition pays. Measured on all four
    benchmark variants (Rastrigin, 20,000 evaluations, 5 seeds):

    | Algorithm | plain | rotated | shifted | rot + shift |
    |---|---|---|---|---|
    | plain [Bat](bat.md) | 41.6 | 36.6 | 21.3 | 27.1 |
    | [Hybrid Bat](hybrid-bat.md) | **3.86** | 10.5 | **3.75** | **10.3** |
    | **HSABA (tuned)** | 4.63 | **9.75** | 7.97 | 14.9 |
    | HSABA (published settings) | 13.7 | 19.7 | 13.3 | 20.0 |

    Both hybrids improve enormously on plain Bat. Between them, HSABA is
    marginally better on the rotated variant and clearly worse on the
    other three — so on this evidence the self-adaptive layer **is not
    an improvement over HBA**, only a different trade.

    That is worth stating plainly in a teaching library. Adding an
    adaptive mechanism is a natural-looking move and it did not pay off
    here; the honest comparison is against the algorithm you extended,
    not only against the original.

    A caveat in the other direction: HSABA needs **no `alpha`
    tuning**, because it has no decay to tune. HBA's headline result
    depends on `alpha=0.99`, a value the original paper does not use.
    Judged by how much hand-tuning each needs to reach its result, the
    self-adaptive version is the less fragile of the two.

!!! note "Tuning notes"
    - **The loudness range wants to be high and narrow.** \([0.99, 1]\)
      beats \([0.9, 1]\) by two orders of magnitude on Sphere. Since
      \(A_i\) is the acceptance probability, a low draw simply throws
      away good trials.
    - **`tau_2 = 0.02`, not 0.1.** Re-drawing the pulse rate rarely lets
      a bat keep a working exploration/exploitation balance for a while
      instead of resampling it away.
    - **Keep `CR` at 0.9.** At 0.5 the rotated result degrades from 18.5
      to 35.0 — the same separability trap the
      [HBA](hybrid-bat.md) and [MKE](mke.md) pages record.

## Behavior

HSABA reaches Sphere ≈ 1e-06, Rastrigin ≈ 4.63, Ackley ≈ 1.13, with the
best rotated-Rastrigin score in the bat family.

Its real value here is comparative. The library now holds three points
on one lineage — [Bat](bat.md), [Hybrid Bat](hybrid-bat.md), and this —
which makes it possible to ask what each addition actually bought.
Replacing the random walk with a DE move bought a factor of eleven.
Adding self-adaptation on top bought robustness to hand-tuning, and
roughly nothing on the benchmarks.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import HybridSelfAdaptiveBatAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = HybridSelfAdaptiveBatAlgorithm(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- I. Fister Jr., S. Fong, J. Brest, and I. Fister, "A novel hybrid
  self-adaptive bat algorithm," *The Scientific World Journal*, 2014,
  709738, 2014.
  [doi:10.1155/2014/709738](https://doi.org/10.1155/2014/709738).
- J. Brest, S. Greiner, B. Bošković, M. Mernik, and V. Žumer,
  "Self-adapting control parameters in differential evolution: a
  comparative study on numerical benchmark problems," *IEEE
  Transactions on Evolutionary Computation*, 10(6), 646-657, 2006 —
  the jDE scheme adapted here.
  [doi:10.1109/TEVC.2006.872133](https://doi.org/10.1109/TEVC.2006.872133).
- I. Fister Jr., D. Fister, and X.-S. Yang, "A hybrid bat algorithm,"
  *Elektrotehniški Vestnik*, 80(1-2), 1-7, 2013.
  [arXiv:1303.6310](https://arxiv.org/abs/1303.6310).
- C. L. Camacho-Villalón, M. Dorigo, and T. Stützle, "Exposing the grey
  wolf, moth-flame, whale, firefly, bat, and antlion algorithms: six
  misleading optimization techniques inspired by bestial metaphors,"
  *International Transactions in Operational Research*, 30(6),
  2945-2971, 2023 — the critique covers the Bat Algorithm underlying
  this lineage.
  [doi:10.1111/itor.13176](https://doi.org/10.1111/itor.13176).
