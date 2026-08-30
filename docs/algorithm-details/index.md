# Detail of Algorithm

These pages document each metaheuristic in this library in depth: the
idea it is built on, a flowchart of its control flow, the equations it
implements, pseudocode matching the actual source, its parameters, and
the literature it comes from.

| Algorithm | Domain | Search style | Detail page |
|---|---|---|---|
| Ant Colony Optimization (ACO-R) | continuous | population + solution archive | [ACO-R](aco.md) |
| Binary Ant Colony Optimization | binary / subsets | population + pheromone per bit | [Binary ACO](binary-aco.md) |
| Artificial Bee Colony | continuous | population + abandonment | [ABC](abc.md) |
| Bat Algorithm | continuous | population + echolocation | [Bat](bat.md) |
| Genetic Algorithm | continuous | population + recombination | [GA](ga.md) |
| Simulated Annealing | continuous | single solution + cooling | [SA](sa.md) |

## Shared structure

Every algorithm subclasses `Algorithm` and implements two methods, so
the pseudocode on each page slots into one shared loop:

```python
state = algorithm.init_population(task)      # build the initial solutions
while not task.stopping_condition():         # budget: max_evals / max_iters
    state = algorithm.run_iteration(task, state)
    task.next_iter()
return task.result()                         # (best_x, best_fitness)
```

`task.eval(x)` scores a solution, counts the evaluation, and tracks the
best solution ever seen; `task.repair(x)` clips a solution back inside
the bounds. Because the budget check happens in the shared loop, every
algorithm respects `max_evals` exactly.

## Benchmark comparison

All six on the standard benchmarks (10 dimensions, 20,000 evaluations,
mean over 3 seeds — lower is better):

| Algorithm | Sphere | Ackley | Rastrigin |
|---|---|---|---|
| ACO-R | **9e-25** | 1e-12 | 31.0 |
| Artificial Bee Colony | 6e-15 | **5e-06** | **2e-08** |
| Simulated Annealing | 8e-13 | 8e-06 | 26.9 |
| Bat Algorithm | 4e-05 | 0.045 | 31.5 |
| Genetic Algorithm | 1.7e-04 | 0.124 | 3.0 |

Two lessons for students in this table. First, **no algorithm wins
everywhere** (the "no free lunch" theorem in miniature): ACO-R is
unmatched at polishing a smooth unimodal landscape, while ABC and GA
dominate the highly multimodal Rastrigin. Second, the ranking depends
on the *landscape*, not on how fashionable the metaphor is — always
benchmark on a problem resembling yours.
