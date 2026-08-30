# Cat Swarm Optimization (CSO)

**CSO** (Chu, Tsai & Pan, 2006) starts from an observation about cats:
they spend most of their time resting — but alert, watching their
surroundings — and only occasionally burst into a fast chase. The
algorithm gives every cat **two modes** and assigns one at random each
iteration.

| Mode | Who | What it does | Role |
|---|---|---|---|
| **Seeking** | the resting majority | copies itself, tweaks a few dimensions of each copy, moves to the best copy | exploitation |
| **Tracing** | a fraction `mixture_ratio` | accelerates toward the best solution with a velocity update | exploration |

This split is what makes CSO unusual: most algorithms apply one
movement rule to every individual, while CSO runs two qualitatively
different searches side by side.

## Equations

**1. Mode assignment.** Each iteration, cat \(i\) traces with
probability \(\text{MR}\) and seeks otherwise:

\[
\text{mode}_i =
\begin{cases}
\text{tracing} & \text{if } \mathcal{U}(0,1) < \text{MR}\\
\text{seeking} & \text{otherwise}
\end{cases}
\]

**2. Seeking mode.** The cat makes \(\text{SMP}\) copies (one of them
being itself when `spc=True`). In each copy, \(\text{CDC} \times m\)
randomly chosen dimensions are shifted by up to \(\text{SRD}\):

\[
x_d^{\text{copy}} = x_d \pm r \cdot \text{SRD}^{(t)} \cdot
(\text{upper}_d - \text{lower}_d),
\qquad r \sim \mathcal{U}(0,1)
\]

The cat then moves to the best copy, but only if it beats its current
position — seeking is strictly greedy.

**3. Decaying seeking range.** The range narrows quadratically as the
budget is spent:

\[
\text{SRD}^{(t)} = \text{SRD}_0
\left(\max\left(1 - \frac{\text{evals}}{\text{max\_evals}},\ 10^{-4}\right)\right)^{2}
\]

**4. Tracing mode.** A velocity update toward the best solution
\(x^{*}\), clipped to a limit:

\[
v_i \leftarrow \mathrm{clip}\Bigl(v_i + c \cdot r \odot (x^{*} - x_i),\ 
-v_{\max},\ v_{\max}\Bigr),
\qquad
x_i \leftarrow x_i + v_i
\]

## Pseudocode

```text
input: cats n, mixture ratio MR, SMP, SRD, CDC, SPC, c, v_max
x <- n random solutions, evaluated;  v <- random velocities

repeat until the budget is exhausted:
    for each cat i:
        if Uniform(0,1) < MR:                       # tracing   (eq. 1)
            v[i] <- clip(v[i] + c * r * (best - x[i]), ±v_max)  (eq. 4)
            cand <- repair(x[i] + v[i])
            if cand is better: x[i] <- cand
        else:                                       # seeking
            srd  <- SRD * (1 - evals / max_evals)^2            (eq. 3)
            make SMP copies, each with CDC*m dimensions
                shifted by up to srd                           (eq. 2)
            x[i] <- the best copy, if it beats x[i]

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Number of cats |
| `mixture_ratio` | MR | 0.2 | Fraction in tracing mode |
| `smp` | SMP | 5 | Copies a seeking cat considers |
| `srd` | SRD | 0.2 | Initial seeking range (fraction of the bound range) |
| `cdc` | CDC | 0.1 | Fraction of dimensions changed per copy |
| `spc` | SPC | `True` | Count the current position as one candidate |
| `velocity_factor` | \(c\) | 2.0 | Tracing acceleration |
| `max_velocity` | \(v_{\max}\) | 0.2 | Velocity limit (fraction of the range) |
| `seed` | — | `None` | Reproducibility |

!!! note "Two tuned departures from the original"
    - **A decaying seeking range (equation 3).** With SRD fixed at 20%
      of the search range, seeking never becomes fine-grained and the
      search stalls (Sphere ≈ 1.3). Adding the quadratic decay brought
      that to ≈ 1e-05.
    - **`cdc` defaults to 0.1, not 0.8.** Changing only a tenth of the
      dimensions per copy — closer to coordinate-wise search — improved
      Rastrigin from 6.1 to 2.3. Separable multimodal functions reward
      moving one coordinate at a time.

## Behavior

CSO is the library's **second-best algorithm on Rastrigin** (≈ 2.3,
behind only ABC), with respectable Sphere (≈ 1e-05) and Ackley
(≈ 0.03). The greedy, coordinate-wise seeking is what does it: on a
separable multimodal landscape, refining one dimension at a time walks
straight down the ridges that trap algorithms moving in all dimensions
at once.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import CatSwarmOptimization

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = CatSwarmOptimization(population_size=30, mixture_ratio=0.2, seed=42)
best_x, best_fitness = algo.run(task)
```

## References

- S.-C. Chu, P.-W. Tsai, and J.-S. Pan, "Cat swarm optimization," in
  *PRICAI 2006: Trends in Artificial Intelligence*, Lecture Notes in
  Computer Science 4099, Springer, 854-858, 2006.
  [doi:10.1007/978-3-540-36668-3_94](https://doi.org/10.1007/978-3-540-36668-3_94).
- S.-C. Chu and P.-W. Tsai, "Computational intelligence based on the
  behavior of cats," *International Journal of Innovative Computing,
  Information and Control*, 3(1), 163-173, 2007.
