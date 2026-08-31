# Lion Optimization Algorithm (LOA)

**LOA** (Yazdani & Jolai, 2016) is the most elaborate algorithm in this
library — seven operators across two population types. Lions are split
into **prides** and a group of **nomads**, each lion is male or female,
and the group and sex a lion belongs to determine which operator moves
it.

That makes LOA the only algorithm here with a **heterogeneous
population**. Everywhere else, every individual runs the same update
(possibly with a random branch); here a pride female and a nomad male
genuinely do different things for the whole run.

The idea holding it together is the **territory**: the set of best
positions a pride's members have ever visited. It is a memory that
outlives whoever is standing on those positions, and both the
safe-place move and male roaming navigate by it.

## Equations

**1. Hunting.** Half the pride's females hunt. They are split into three
groups; the group with the best summed fitness becomes the **centre**,
the others are **wings**. A prey sits at the hunters' mean position:

\[
\text{PREY} = \frac{1}{|H|}\sum_{i \in H} X_i
\]

Centre hunters close in directly, wings attack from the far side:

\[
X_i' \sim \mathcal{U}\bigl(\min(X_i, \text{PREY}),\ \max(X_i, \text{PREY})\bigr)
\]
\[
X_i' \sim \mathcal{U}\bigl(\min(2\,\text{PREY} - X_i, \text{PREY}),\
\max(2\,\text{PREY} - X_i, \text{PREY})\bigr)
\]

A hunter that improves makes the prey bolt, by the fraction it gained:

\[
\text{PREY} \leftarrow \text{PREY} + r \cdot \text{PI} \cdot
(\text{PREY} - X_i),
\qquad
\text{PI} = \frac{|f_{\text{old}} - f_{\text{new}}|}{|f_{\text{old}}|}
\]

**2. Moving to a safe place.** The other females pick a territory
position by tournament and approach it with a sideways kick:

\[
X_i \leftarrow X_i + 2 d\, r_1 \vec{D} + \tan(\theta)\, d \vec{R},
\qquad \theta \sim \mathcal{U}(-\tfrac{\pi}{6}, \tfrac{\pi}{6})
\]

with \(d\) the distance to the target, \(\vec{D}\) the unit direction
toward it, and \(\vec{R}\) a random unit vector perpendicular to
\(\vec{D}\).

**3. Roaming.** Pride males wander toward a sample of territory
positions. Nomads instead jump anywhere in the search space, with a
probability that rises the worse they are.

**4. Mating.** An offspring blends a female with a pride male, then
mutates:

\[
X_{\text{cub}} = \beta X_{\text{female}} + (1 - \beta) X_{\text{male}},
\qquad \beta \sim \mathcal{N}(0.5,\ 0.1)
\]

**5. Defence and migration.** The weakest pride male is exiled if a
nomad male beats him. Then a fraction of each pride's females leave, and
the best nomad females fill the places they opened.

## Pseudocode

```text
input: lions n, prides P, nomad/sex/roaming/mating/migration ratios
x <- n random solutions; split into P prides plus nomads; assign sexes
territory <- best position each lion has visited

repeat until the budget is exhausted:
    for each pride:
        half the females hunt around a shared prey                (eq. 1)
        the other females move toward a territory position        (eq. 2)
        males roam through the territory                          (eq. 3)
        some females mate with a pride male; cubs replace
            their mother only if better                           (eq. 4)
    nomads jump at random, more often when doing badly            (eq. 3)
    weak males exiled, strong nomad males promoted                (eq. 5)
    some females migrate out; best nomad females fill the gaps    (eq. 5)

return best solution found
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 50 | Total lions |
| `n_prides` | 4 | Prides the residents are split into |
| `nomad_ratio` | 0.2 | Fraction that are nomads |
| `sex_ratio` | 0.8 | Fraction of pride lions that are female |
| `roaming_ratio` | 0.2 | Share of the territory a male roams toward |
| `mating_ratio` | 0.3 | Females that mate each iteration |
| `mutation_prob` | 0.1 | Per-coordinate mutation chance for cubs |
| `migration_ratio` | 0.4 | Pride females that migrate out |
| `seed` | `None` | Reproducibility |

!!! warning "Ackley convergence is bimodal — the mean misleads"
    LOA either solves Ackley to about 1e-06 or gets stuck near 1.2.
    Across 7 seeds at the default settings:

    ```
    7e-07, 9e-07, 0.011, 1.2, 1.2, 5e-06, 1.6e-04
    ```

    Mean 0.33, **median 1.6e-04** — the mean is set almost entirely by
    the two failures. The comparison table reports means for
    consistency, so LOA's Ackley entry of 0.39 describes neither a
    typical run nor a bad one.

    This matters when tuning. Screening `migration_ratio=0.6` over 3
    seeds gave Ackley 5e-06 and looked like a large win; at 7 seeds it
    was *worse* than the default (3 stuck runs against 2). Any
    parameter choice made on a handful of seeds here is selecting seed
    luck, not a setting.

!!! note "Tuning notes: small budgets invert the ranking"
    Screening at 5,000 evaluations picks a very different configuration
    from the one that wins at 20,000:

    | Configuration | 5k Sphere | 5k Rastrigin | **20k Sphere** | **20k Rastrigin** |
    |---|---|---|---|---|
    | Paper defaults | 5e-03 | 17.5 | **2e-13** | **2.4** |
    | `N=25, nomad=.05, mating=.1` | **2e-04** | **13.6** | 2e-16 | 4.8 |

    The tuned-at-5k configuration looks 20× better on Sphere and
    clearly better on Rastrigin, and is *worse* on Rastrigin at the real
    budget. LOA has seven operators and needs iterations for the social
    structure to do anything; a short run only measures its startup
    cost. The published defaults are kept for this reason — the same
    trap the [FSS](fss.md) and [GSA](gsa.md) pages document.

!!! info "An implementation note worth reading"
    Migration must refill the places it empties. An early version here
    exiled 40% of each pride's females per iteration but admitted only a
    few nomads back, so the nomad group grew from 17 lions to 28 out of
    50 — and since nomads only search at random, more than half the
    evaluation budget was being spent on noise while the prides did the
    real work. Fixing the refill moved Sphere from 6e-03 to 3e-13 at
    20,000 evaluations. There is a regression test asserting the nomad
    group stays bounded.

## Behavior

LOA reaches Sphere ≈ 2e-13, Rastrigin ≈ 2.1, Ackley ≈ 0.46 (median
0.011). The Rastrigin score is among the best in the library.

Robustness is middling. Across the four benchmark variants
([why these](index.md#benchmark-comparison)):

| Variant | Rastrigin |
|---|---|
| plain | 2.08 |
| rotated | 7.96 |
| shifted | 4.59 |
| rotated + shifted | 12.9 |

A factor of six from end to end — better than [GWO](gwo.md) or
[HS](hs.md), well behind [KH](kh.md) or [HHO](hho.md).

The honest summary is that LOA works, and that it is very hard to say
*which* of its seven operators makes it work. That is the real cost of
this much machinery: with a heterogeneous population and seven
interacting operators, an ablation is a research project rather than a
paragraph, and a poor result gives you little idea what to change.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import LionOptimizationAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = LionOptimizationAlgorithm(population_size=50, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- M. Yazdani and F. Jolai, "Lion Optimization Algorithm (LOA): a
  nature-inspired metaheuristic algorithm," *Journal of Computational
  Design and Engineering*, 3(1), 24-36, 2016.
  [doi:10.1016/j.jcde.2015.06.003](https://doi.org/10.1016/j.jcde.2015.06.003).
- K. Sörensen, "Metaheuristics — the metaphor exposed," *International
  Transactions in Operational Research*, 22(1), 3-18, 2015 — on
  judging algorithms of this kind by their components rather than their
  imagery.
  [doi:10.1111/itor.12001](https://doi.org/10.1111/itor.12001).
