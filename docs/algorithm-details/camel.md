# Camel Algorithm (CA)

The **Camel Algorithm** (Ali, 2016) models a caravan crossing the
desert in search of an oasis. Each camel is a candidate solution, and
its journey is governed by three state variables that make the search
adapt over time without any explicit cooling schedule.

| Variable | Meaning | Effect on the step |
|---|---|---|
| Temperature \(T\) | the heat a camel meets, drawn at random | high heat drains endurance |
| Endurance \(E\) | strength left, falls with heat and distance | a strong camel strides far; an exhausted one shuffles |
| Supply \(S\) | water and food, consumed at the burden rate | dwindling supplies make the camel search more desperately |

Two events punctuate the journey: reaching an **oasis** (a better
position) replenishes endurance and supply, while running out of
endurance means the camel **dies** and is reborn at a random location.

## Equations

**1. Temperature.** Each camel faces a random heat every step:

\[
T_i = \mathcal{U}(T_{\min},\, T_{\max})
\]

**2. Endurance.** Falls with the heat and with the fraction of the
journey already travelled, \(\theta = \text{evals}/\text{max\_evals}\):

\[
E_i = \left(1 - \frac{T_i}{T_{\max}}\right)\,\bigl(1 - \theta\bigr)
\]

**3. Supply.** Consumed steadily at the burden rate \(\omega\):

\[
S_i = 1 - \omega\,\theta
\]

**4. Movement toward the oasis.** The camel steps toward the best
solution \(x^{*}\), scaled by its endurance and by how depleted its
supplies are:

\[
x_i' = x_i + v\,\delta \cdot E_i \cdot e^{\,1 - S_i} \cdot \bigl(x^{*} - x_i\bigr),
\qquad \delta \sim \mathcal{U}(-1, 1)^m
\]

The two factors pull in opposite directions: \(E_i\) shrinks as the
journey advances (refinement), while \(e^{1-S_i}\) grows as supplies
run out (renewed exploration). Because \(\delta\) is drawn per
dimension and may be negative, a camel can also overshoot or move away
from the oasis.

**5. Oasis and death.**

\[
\begin{cases}
x_i \leftarrow x_i', \; E_i, S_i \leftarrow 1
  & \text{if } f(x_i') < f(x_i) \quad (\text{oasis found})\\[1ex]
x_i \sim \mathcal{U}(\text{lower}, \text{upper}), \; E_i, S_i \leftarrow 1
  & \text{if } E_i < \text{death rate} \quad (\text{camel dies})
\end{cases}
\]

## Pseudocode

```text
input: camels n, temperature range, burden rate w, death rate, visibility v
x <- n random solutions, evaluated;  E, S <- 1

repeat until the budget is exhausted:
    theta <- evals / max_evals                 # fraction travelled
    for each camel i:
        T    <- Uniform(T_min, T_max)                          (eq. 1)
        E[i] <- (1 - T / T_max) * (1 - theta)                  (eq. 2)
        S[i] <- 1 - w * theta                                  (eq. 3)
        cand <- x[i] + v * delta * E[i] * exp(1 - S[i]) * (best - x[i])
                                                               (eq. 4)
        if cand is better than x[i]:
            x[i] <- cand;  E[i], S[i] <- 1     # oasis          (eq. 5)
        else if E[i] < death rate:
            x[i] <- a fresh random solution;  E[i], S[i] <- 1   # death

return best solution found
```

## Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `population_size` | \(n\) | 30 | Camels in the caravan |
| `min_temperature`, `max_temperature` | \(T_{\min}, T_{\max}\) | -1.0, 1.0 | Range of the random heat |
| `burden_rate` | \(\omega\) | 0.9 | How fast supplies are consumed; higher = more late exploration |
| `death_rate` | — | 0.01 | Endurance below which a camel restarts; higher = more restarts |
| `visibility` | \(v\) | 1.0 | Overall step scale |
| `seed` | — | `None` | Reproducibility |

!!! note "Two implementation choices"
    - **The step is scaled by \(E\), not by \(1 - E\).** Some
      write-ups of the algorithm use the latter, which makes an
      exhausted camel take *longer* strides — the opposite of the
      metaphor, and in benchmarking it prevented any late refinement
      (Sphere stalled at ~6e-03 versus ~1e-09 with the form used here).
    - **`death_rate` defaults to 0.01, not 0.1.** Frequent restarts
      kept throwing away good camels; the lower value improved Sphere
      from ~3e-04 to ~1e-09 and Rastrigin from ~11.5 to ~4.4.

## Behavior

The Camel Algorithm is this library's **third-strongest on multimodal
landscapes** (Rastrigin ≈ 4.4, behind only ABC and GA) while remaining
respectable elsewhere (Sphere ≈ 1e-09, Ackley ≈ 4e-04). The death
mechanism is what drives that: camels trapped in poor regions are
periodically wiped out and restarted, so the caravan keeps sampling
new basins instead of all converging on the first oasis it finds.

```python
from ikn_library import Task
from ikn_library.problems import Rastrigin
from ikn_library.algorithms import CamelAlgorithm

task = Task(problem=Rastrigin(dimension=10), max_evals=20000)
algo = CamelAlgorithm(population_size=30, seed=42)
best_x, best_fitness = algo.run(task)
```

## Reference

R. M. Ali, "Novel optimization algorithm inspired by camel traveler
behavior," *International Journal of Sciences: Basic and Applied
Research*, 2016. The algorithm has since been applied and refined in
several follow-up studies; the formulation implemented here follows the
original description with the two adjustments documented above.
