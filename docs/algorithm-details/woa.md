# Whale Optimization Algorithm (WOA)

**WOA** (Mirjalili & Lewis, 2016) models the *bubble-net* feeding of
humpback whales. Each whale flips a coin and either **swims toward** the
best solution or **spirals around** it; a second control decides whether
that target is the best whale or a random one.

| Coin \(p\) | \(\lvert A \rvert\) | Move |
|---|---|---|
| < 0.5 | \(\ge 1\) | search — encircle a **random** whale |
| < 0.5 | \(< 1\) | encircle the **best** whale |
| \(\ge 0.5\) | — | spiral toward the best whale |

The spiral is the same logarithmic curve [Moth-Flame](mfo.md) uses. What
differs is that here it alternates with a straight-line approach on
every individual, rather than being the only move — and, as the
measurements below show, those two halves behave very differently when
the problem is transformed.

## Equations

**1. Encircling and search.** Both use the same rule, differing only in
the target \(X_t\) (the best whale, or a random one):

\[
X \leftarrow X_t - A \odot \bigl\lvert C \odot X_t - X \bigr\rvert,
\qquad
A = 2 a r_1 - a,
\quad
C = 2 r_2
\]

**2. Bubble-net spiral**, around the best whale:

\[
X \leftarrow \lvert X_* - X \rvert \, e^{b\ell} \cos(2\pi \ell) + X_*,
\qquad \ell \sim \mathcal{U}(-1, 1)
\]

Note the asymmetry: equation 1 multiplies the target's **absolute
coordinates** by \(C\); equation 2 uses a plain difference. There are
tests asserting each property.

**3. Control schedule.** \(a\) falls linearly to zero, which caps
\(\lvert A \rvert\):

\[
a = a_{\text{start}}\left(1 - \frac{\text{evals}}{\text{max\_evals}}\right)
\]

Since \(\lvert A \rvert \le a\), once \(a < 1\) the search branch
becomes impossible and exploration switches itself off — a test asserts
this.

## Pseudocode

```text
input: whales n, a_start, spiral constant b
x <- n random solutions, evaluated

repeat until the budget is exhausted:
    best <- the best whale
    a    <- a_start * (1 - evals/max_evals)                       (eq. 3)

    for each whale i:
        if rand() < 0.5:
            target <- a random whale if |A| >= 1 else best
            x[i] <- target - A * |C*target - x[i]|                (eq. 1)
        else:
            x[i] <- |best - x[i]| * exp(b*l) * cos(2*pi*l) + best (eq. 2)
        evaluate x[i]

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Paper | Effect |
|---|---|---|---|---|
| `population_size` | \(n\) | 30 | 30 | Number of whales |
| `a_start` | \(a\) | 2.0 | 2.0 | Initial control coefficient |
| `spiral_constant` | \(b\) | 2.0 | 1.0 | Shape of the logarithmic spiral |
| `seed` | — | `None` | — | Reproducibility |

!!! warning "Half the algorithm is translation-invariant and half is not"
    WOA scores Sphere ≈ 5e-62, **Rastrigin exactly 0 on every seed**,
    and Ackley at machine precision — numbers that, by now, this library
    treats as a prompt to check rather than a result.

    The check finds a **partial** origin bias, and the equations predict
    it exactly. Equation 1 scales the target's absolute coordinates
    through \(C\), so it breaks under translation; equation 2 uses a
    plain difference and does not. Since a whale takes one or the other
    on a coin flip, roughly half its moves survive a shift:

    | Function | at origin | shifted |
    |---|---|---|
    | Sphere | 5e-62 | 8e-05 |
    | Rastrigin | 0 | 13.9 |
    | Ackley | 4e-16 | 0.070 |

    Sphere loses 57 orders of magnitude, but compare where that leaves
    it: shifted Rastrigin 13.9 against [GWO](gwo.md)'s 15.1 and
    [SCA](sca.md)'s 42.5. SCA, whose *only* mechanism carries the
    \(C\)-style factor, is destroyed by a shift; WOA is merely hurt by
    it. Having a translation-invariant half is what makes the
    difference.

!!! note "The Mirjalili family, side by side"
    The library now holds four algorithms by the same author, and the
    four-variant check makes their family resemblance legible
    (Rastrigin, 20,000 evaluations, 5 seeds):

    | Algorithm | plain | rotated | shifted | rot + shift |
    |---|---|---|---|---|
    | **Whale (WOA)** | **0** | **6.01** | **13.9** | **20.7** |
    | [Grey Wolf](gwo.md) | **0** | 11.5 | 15.1 | 25.8 |
    | [Moth-Flame](mfo.md) | 13.5 | 24.1 | 13.9 | 33.5 |
    | [Sine Cosine](sca.md) | 2e-11 | 44.6 | 42.5 | 44.5 |

    All four look excellent or near-excellent on the plain benchmark and
    degrade under transformation; they differ mostly in how far. WOA is
    the best of the four on every variant, and its rotated result of
    6.01 is genuinely strong — third best in the whole library, behind
    only [HHO](hho.md) and [KH](kh.md).

    Note also that WOA's profile is the **mirror image** of
    [HS](hs.md)'s: WOA tolerates rotation and suffers from shifts, HS
    tolerates shifts and collapses under rotation. Neither is visible on
    the standard suite.

!!! note "Tuning notes"
    - **`spiral_constant=2.0`, not the paper's 1.0.** It is neutral on
      the standard suite and clearly better on the hard variants
      (shifted 8.0 against 13.9, rotated+shifted 16.9 against 20.7).
      A tighter spiral covers less ground around the incumbent.
    - **`a_start` should stay at 2.** Lowering it to 1 removes the
      search branch entirely from the start, and raising it to 3 is
      worse on every variant.
    - **Population size trades precision against robustness**, as
      usual: 15 whales give Sphere 4e-94 and the best rotated result
      (5.05), 30 give the best rotated+shifted (16.9).

## Behavior

WOA is the strongest of the four Mirjalili algorithms here and a solid
mid-to-upper performer overall once its origin bias is accounted for.

Its interest for this library is structural. It is the clearest example
of an algorithm whose robustness comes from **having two mechanisms that
fail differently** — the spiral half keeps working when the encircling
half breaks. That is a design principle worth more than either half
alone, and it is the same reason [HHO](hho.md)'s six branches make it
the most transformation-robust entry here.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import WhaleOptimizationAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = WhaleOptimizationAlgorithm(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S. Mirjalili and A. Lewis, "The Whale Optimization Algorithm,"
  *Advances in Engineering Software*, 95, 51-67, 2016.
  [doi:10.1016/j.advengsoft.2016.01.008](https://doi.org/10.1016/j.advengsoft.2016.01.008).
- C. L. Camacho-Villalón, M. Dorigo, and T. Stützle, "Exposing the grey
  wolf, moth-flame, whale, firefly, bat, and antlion algorithms: six
  misleading optimization techniques inspired by bestial metaphors,"
  *International Transactions in Operational Research*, 30(6),
  2945-2971, 2023 — WOA is one of the six named.
  [doi:10.1111/itor.13176](https://doi.org/10.1111/itor.13176).
- N. Hansen, A. Auger, R. Ros, S. Finck, and P. Pošík, "Comparing
  results of 31 algorithms from the black-box optimization benchmarking
  BBOB-2009," *GECCO 2010 Companion*, 1689-1696, 2010.
  [doi:10.1145/1830761.1830790](https://doi.org/10.1145/1830761.1830790).
