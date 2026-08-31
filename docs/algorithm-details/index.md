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
| Grey Wolf Optimizer | continuous | three-leader hierarchy | [GWO](gwo.md) |
| Harmony Search | continuous | memory + whole-memory recombination | [HS](hs.md) |
| Harris Hawks Optimization | continuous | six moves gated by prey energy | [HHO](hho.md) |
| Komodo Mlipir Algorithm | continuous | three role groups + adaptive population | [KMA](kma.md) |
| Krill Herd Algorithm | continuous | three motions with inertia + food centre | [KH](kh.md) |
| Lion Optimization Algorithm | continuous | prides and nomads + seven operators | [LOA](loa.md) |
| Monarch Butterfly Optimization | continuous | two lands + per-coordinate recombination | [MBO](mbo.md) |
| Monkey King Evolution | continuous | clone group around the incumbent | [MKE](mke.md) |
| Moth-Flame Optimization | continuous | logarithmic spiral + shrinking flame count | [MFO](mfo.md) |
| Particle Swarm Optimization | continuous | velocity + two attractors | [PSO](pso.md) |
| Simulated Annealing | continuous | single solution + cooling | [SA](sa.md) |
| Sine Cosine Algorithm | continuous | trigonometric swing toward the best | [SCA](sca.md) |

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

All thirty-one on the standard benchmarks (10 dimensions, 20,000 evaluations,
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
| Grey Wolf Optimizer** | 4e-88 | 4e-16 | **0** |
| Harmony Search* | 5e-08 | 3e-03 | 0.018 |
| Harris Hawks Optimization | 2e-88 | 4e-16 | **0** |
| Krill Herd | 4e-09 | 4e-04 | 3.0 |
| Lion Optimization*** | 2e-13 | 0.39 | 2.4 |
| Monarch Butterfly* | 2e-07 | 9e-04 | 3e-04 |
| Monkey King Evolution | 5e-08 | 4e-03 | 2.7 |
| Moth-Flame Optimization | 2e-17 | 2e-07 | 9.3 |
| Particle Swarm Optimization | 1e-28 | 2e-13 | 3.0 |
| Sine Cosine Algorithm**** | 9e-26 | 5e-10 | 4e-11 |

\* Forest Optimization, Harmony Search, and Monarch Butterfly all
build solutions one coordinate at a time, so their Rastrigin scores are
**largely artefacts of this benchmark suite** — see the caveat below,
the [FOA page](foa.md), the [HS page](hs.md), and the
[MBO page](mbo.md). Monarch Butterfly is the extreme case: rotating
Rastrigin moves it from 3e-04 to 39.6.

\*\* Grey Wolf's whole row is inflated by an **origin bias**: all three
optima sit at \(x = 0\), and shifting them costs GWO 82 orders of
magnitude on Sphere. The Fireworks Algorithm shares this bias. See the
[GWO page](gwo.md).

\*\*\*\* Sine Cosine is the extreme case of the same problem: the origin
is an **exact fixed point** of its update rule, reached whatever the
objective. Shifting the optimum by just 0.1 costs twenty-four orders of
magnitude, and off the origin it beats random search only by a factor of
two to six. Its entire row is an artefact — see the [SCA page](sca.md).

\*\*\* Lion Optimization's Ackley result is **bimodal** — most runs
reach ~1e-06, a minority stall near 1.2. Its mean of 0.39 describes no
actual run; the median is 0.011. See the [LOA page](loa.md).

Three lessons for students in this table. First, **no algorithm wins
everywhere** (the "no free lunch" theorem in miniature): the Firefly
Algorithm and DE dominate the smooth landscapes, while ABC is
untouchable on
the highly multimodal Rastrigin, with CSO, DE, and GA close behind.
Second, the ranking depends
on the *landscape*, not on how fashionable the metaphor is — always
benchmark on a problem resembling yours.

Third, and most easily missed: these three benchmarks share **two
properties that flatter particular algorithms**, and both are invisible
unless you go looking.

They are all **separable** — each coordinate can be optimized
independently — which flatters anything that searches one coordinate at
a time. Forest Optimization is exactly that case: it looks like a
near-winner above, but rotating Rastrigin so the coordinates couple
sends it from 9e-06 to 27.

They also all place their optimum at exactly \(x = 0\), which flatters
algorithms whose update rule is not translation-invariant. Grey Wolf
loses 82 orders of magnitude on Sphere from nothing but moving the
optimum.

**The two transformations test different things, and a rotation alone
does not test both.** Rotating a function turns it about the origin, so
an optimum sitting at zero stays at zero — a rotated benchmark still
tells you nothing about origin bias. To separate the effects you need
all four variants:

| Variant | Tests for |
|---|---|
| plain | — |
| rotated | reliance on separability |
| shifted | origin bias |
| rotated **and** shifted | both at once |

Run that way, the picture changes. Harmony Search is untroubled by a
shift (0.075) but collapses under rotation (26.3); Grey Wolf degrades
under either; Harris Hawks and Krill Herd hold up under both, which is
why their rows above can be taken at face value while others cannot.

Krill Herd is the clearest illustration of why that is worth checking.
Its Rastrigin score of 3.0 is beaten by eight algorithms in the table,
yet it varies by under 15% across all four variants, because every
coupling in it uses position *differences* and normalized fitness gaps
rather than absolute coordinates. On a real problem — neither
axis-aligned nor centred on zero — it will deliver roughly what the
table says, which several higher-scoring rows will not.

A benchmark table measures the *pairing* of algorithm and problem, never
the algorithm alone.
