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
| Bacterial Foraging Optimization | continuous | population + three timescales | [BFO](bfo.md) |
| Bat Algorithm | continuous | population + echolocation | [Bat](bat.md) |
| Bees Algorithm | continuous | population + site recruitment | [Bees](bees.md) |
| Camel Algorithm | continuous | population + endurance/supply + restarts | [Camel](camel.md) |
| Cat Swarm Optimization | continuous | population + two behaviour modes | [CSO](cso.md) |
| Clonal Selection Algorithm | continuous | population + affinity-proportional cloning | [CLONALG](clonalg.md) |
| Cuckoo Search | continuous | population + Levy flights | [Cuckoo](cuckoo.md) |
| Differential Evolution | continuous | population + difference vectors | [DE](de.md) |
| Firefly Algorithm | continuous | population + distance-faded attraction | [Firefly](firefly.md) |
| Fireworks Algorithm | continuous | population + quality-coupled explosions | [FWA](fwa.md) |
| Fish School Search | continuous | population + weights + school contraction | [FSS](fss.md) |
| Genetic Algorithm | continuous | population + recombination | [GA](ga.md) |
| Gravitational Search Algorithm | continuous | population + mass-based attraction | [GSA](gsa.md) |
| Komodo Mlipir Algorithm | continuous | three role groups + adaptive population | [KMA](kma.md) |
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

All eighteen on the standard benchmarks (10 dimensions, 20,000 evaluations,
mean over 3 seeds — lower is better):

| Algorithm | Sphere | Ackley | Rastrigin |
|---|---|---|---|
| Fireworks Algorithm | **3e-88** | **2e-15** | **0** |
| Firefly Algorithm | 7e-57 | 2e-15 | 6.0 |
| Differential Evolution | 2e-41 | 5e-15 | 2.0 |
| Gravitational Search | 7e-24 | 2e-11 | 3.8 |
| ACO-R | 9e-25 | 1e-12 | 31.0 |
| Artificial Bee Colony | 6e-15 | 5e-06 | 2e-08 |
| Cat Swarm Optimization | 1e-05 | 0.031 | 2.3 |
| Fish School Search | 7e-05 | 0.057 | 2.7 |
| Simulated Annealing | 8e-13 | 8e-06 | 26.9 |
| Komodo Mlipir Algorithm | 1e-21 | 1e-06 | 16.9 |
| Bees Algorithm | 5e-14 | 5e-06 | 22.9 |
| Bat Algorithm | 4e-05 | 0.045 | 31.5 |
| Genetic Algorithm | 1.7e-04 | 0.124 | 3.0 |
| Bacterial Foraging | 4e-06 | 0.017 | 18.9 |
| Camel Algorithm | 1e-09 | 4e-04 | 4.4 |
| Cuckoo Search | 4e-10 | 3e-04 | 6.8 |
| Clonal Selection | 4e-08 | 1e-03 | 8.6 |

Two lessons for students in this table. First, **no algorithm wins
everywhere** (the "no free lunch" theorem in miniature): the Firefly
Algorithm and DE dominate the smooth landscapes, while ABC is
untouchable on
the highly multimodal Rastrigin, with CSO, DE, and GA close behind.
Second, the ranking depends
on the *landscape*, not on how fashionable the metaphor is — always
benchmark on a problem resembling yours.
