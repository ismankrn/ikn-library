# Algorithms

All algorithms share the same interface: construct with their
parameters, then call `run(task)` to get `(best_x, best_fitness)`.

```python
from ikn_library.algorithms import (
    AntColonyOptimization,
    BinaryAntColonyOptimization,
    SimulatedAnnealing,
)
```

| Algorithm | Class | Domain | Reference |
|---|---|---|---|
| Ant Colony Optimization (ACO-R) | `AntColonyOptimization` | continuous | Socha & Dorigo, EJOR 185(3), 2008 |
| Binary Ant Colony Optimization | `BinaryAntColonyOptimization` | binary / subsets | hyper-cube pheromone update |
| Simulated Annealing | `SimulatedAnnealing` | continuous | Kirkpatrick et al., Science 220, 1983 |

## Ant Colony Optimization (ACO-R)

`AntColonyOptimization` — for **continuous** problems (parameter
optimization, ensemble weights). Keeps an archive of the best solutions
found so far; each ant samples a new solution from a Gaussian centered
on an archive member, with the Gaussian width shrinking as the archive
converges — the archive plays the role of the pheromone trail.

Key parameters: `population_size` (ants per iteration), `archive_size`
(k), `intensification` (q, locality of search), `evaporation` (xi,
convergence speed), `seed`.

## Binary Ant Colony Optimization

`BinaryAntColonyOptimization` — for **binary / subset** problems
(feature selection, ensemble pruning). Maintains a pheromone value per
(variable, bit) pair and builds bit strings by sampling each bit
proportionally to its pheromone, with a hyper-cube pheromone update
toward the best solution found and `[tau_min, tau_max]` limits to
preserve exploration.

Key parameters: `population_size`, `evaporation` (rho), `alpha`
(pheromone exponent), `tau_min`, `tau_max`, `seed`.

## Simulated Annealing

`SimulatedAnnealing` — a **single-solution** method for continuous
problems, and a useful baseline against the population-based
algorithms. Each iteration proposes one Gaussian neighbor, accepting
worse moves with probability `exp(-delta / T)` while the temperature —
and with it the step size — cools geometrically.

Key parameters: `initial_temperature`, `cooling`, `step_size`, `seed`.

---

More algorithms are planned. For usage tutorials see the
[documentation](https://ikn-library.readthedocs.io); to add a new
algorithm, subclass `Algorithm` and implement `init_population` and
`run_iteration` — the shared `run` loop, budget handling, and
convergence tracking come from the base class and `Task`.
