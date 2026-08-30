# Detail of Algorithm

These pages document each metaheuristic in this library in depth: the
idea it is built on, the equations it implements, pseudocode matching
the actual source, its parameters, and the literature it comes from.

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
| Coral Reefs Optimization | continuous | reef substrate + settlement | [CRO](cro.md) |
| Cuckoo Search | continuous | population + Levy flights | [Cuckoo](cuckoo.md) |
| Differential Evolution | continuous | population + difference vectors | [DE](de.md) |
| Firefly Algorithm | continuous | population + distance-faded attraction | [Firefly](firefly.md) |
| Fireworks Algorithm | continuous | population + quality-coupled explosions | [FWA](fwa.md) |
| Fish School Search | continuous | population + weights + school contraction | [FSS](fss.md) |
| Flower Pollination Algorithm | continuous | two rules switched per flower | [FPA](fpa.md) |
| Forest Optimization Algorithm | continuous | age-gated seeding + recycled discards | [FOA](foa.md) |
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

All twenty-one on the standard benchmarks (10 dimensions, 20,000 evaluations,
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
| Coral Reefs Optimization | 3e-07 | 3e-03 | 3.0 |
| Forest Optimization* | 2e-08 | 1e-03 | 9e-06 |
| Flower Pollination | 3e-30 | 2e-13 | 7.3 |

\* Forest Optimization's Rastrigin score is **largely an artefact of
this benchmark suite** — see the caveat below and the
[FOA page](foa.md).

Three lessons for students in this table. First, **no algorithm wins
everywhere** (the "no free lunch" theorem in miniature): the Firefly
Algorithm and DE dominate the smooth landscapes, while ABC is
untouchable on
the highly multimodal Rastrigin, with CSO, DE, and GA close behind.
Second, the ranking depends
on the *landscape*, not on how fashionable the metaphor is — always
benchmark on a problem resembling yours.

Third, and most easily missed: **all three of these benchmarks are
separable**, meaning each coordinate can be optimized independently.
Any algorithm that happens to search one coordinate at a time is
flattered by them. Forest Optimization is exactly that case — it looks
like the second-best entry in the table, but rotating Rastrigin so the
coordinates become coupled sends it from 9e-06 to 27, well behind
Coral Reefs (7.6) and Fireworks (17.7) on the same rotated function.
A benchmark table is a measurement of the *pairing* of algorithm and
problem, never of the algorithm alone; before trusting any row here,
add a non-separable problem such as Rosenbrock or a rotated variant.
