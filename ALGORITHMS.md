# Algorithms

All algorithms share the same interface: construct with their
parameters, then call `run(task)` to get `(best_x, best_fitness)`.

```python
from ikn_library.algorithms import (
    AntColonyOptimization,
    ArtificialBeeColony,
    BatAlgorithm,
    BeesAlgorithm,
    BinaryAntColonyOptimization,
    CamelAlgorithm,
    CatSwarmOptimization,
    CuckooSearch,
    DifferentialEvolution,
    FireflyAlgorithm,
    GeneticAlgorithm,
    KomodoMlipirAlgorithm,
    NSGA2,
    SimulatedAnnealing,
)
```

| Algorithm | Class | Domain | Reference |
|---|---|---|---|
| Ant Colony Optimization (ACO-R) | `AntColonyOptimization` | continuous | Socha & Dorigo, EJOR 185(3), 2008 |
| Artificial Bee Colony | `ArtificialBeeColony` | continuous | Karaboga, TR06, 2005; Karaboga & Basturk, JOGO 39(3), 2007 |
| Bat Algorithm | `BatAlgorithm` | continuous | Yang, NICSO 2010 |
| Bees Algorithm | `BeesAlgorithm` | continuous | Pham et al., 2005; Pham & Castellani, 2009 |
| Binary Ant Colony Optimization | `BinaryAntColonyOptimization` | binary / subsets | hyper-cube pheromone update |
| Camel Algorithm | `CamelAlgorithm` | continuous | Ali, IJSBAR, 2016 |
| Cat Swarm Optimization | `CatSwarmOptimization` | continuous | Chu, Tsai & Pan, PRICAI 2006 |
| Cuckoo Search | `CuckooSearch` | continuous | Yang & Deb, NaBIC 2009 |
| Differential Evolution | `DifferentialEvolution` | continuous | Storn & Price, JOGO 11(4), 1997 |
| Firefly Algorithm | `FireflyAlgorithm` | continuous | Yang, SAGA 2009 |
| Genetic Algorithm (real-coded) | `GeneticAlgorithm` | continuous | Holland, 1975; BLX-alpha: Eshelman & Schaffer, 1993 |
| Komodo Mlipir Algorithm | `KomodoMlipirAlgorithm` | continuous | Suyanto et al., Applied Soft Computing 114, 2022 |
| NSGA-II | `NSGA2` | continuous, **multi-objective** | Deb et al., IEEE TEVC 6(2), 2002 |
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

## Artificial Bee Colony

`ArtificialBeeColony` — for **continuous** problems, modeled on honey-bee
foraging. Each iteration runs three roles: **employed bees** probe their
own food source by stepping toward a random partner in one random
dimension (greedy selection keeps the better one); **onlooker bees**
re-probe sources with probability proportional to quality; and a
**scout bee** abandons any source that has not improved for `limit`
trials, replacing it with a fresh random solution. That abandonment
mechanism gives ABC unusually strong escape from local optima —
in this library's benchmarks it is by far the best on the highly
multimodal Rastrigin function.

Key parameters: `population_size` (food sources), `limit` (trials
before abandonment; defaults to `population_size * dimension`), `seed`.

## Bat Algorithm

`BatAlgorithm` — for **continuous** problems, inspired by the
echolocation of microbats (Yang, 2010). Each bat flies toward the best
solution with a randomly tuned frequency; a growing **pulse rate**
increasingly triggers local random walks around the best solution,
while **loudness** decays on every accepted improvement, making
acceptance more selective as the search converges. The local-walk step
is scaled to the bound range and decays with the evaluation budget.

Key parameters: `population_size`, `loudness` (A0), `pulse_rate` (r0),
`alpha` (loudness decay), `gamma` (pulse-rate growth),
`min_frequency` / `max_frequency`, `local_scale`, `seed`.

## Bees Algorithm

`BeesAlgorithm` — for **continuous** problems, based on the foraging of
honey bees but organized differently from ABC. Scout bees sample the
space at random; the best **sites** are then searched more closely,
with **elite** sites receiving more recruits than the other selected
ones. Each site's neighborhood shrinks after every unsuccessful search
(progressive neighborhood shrinking), and a site that stagnates is
abandoned for a fresh scout. Effort allocation is thus *explicit* —
fixed recruit counts per site class — where ABC leaves it to
probability.

Key parameters: `population_size` (scouts), `selected_sites`,
`elite_sites`, `elite_bees`, `selected_bees`, `neighborhood`,
`shrink`, `stagnation_limit`, `seed`.

## Binary Ant Colony Optimization

`BinaryAntColonyOptimization` — for **binary / subset** problems
(feature selection, ensemble pruning). Maintains a pheromone value per
(variable, bit) pair and builds bit strings by sampling each bit
proportionally to its pheromone, with a hyper-cube pheromone update
toward the best solution found and `[tau_min, tau_max]` limits to
preserve exploration.

Key parameters: `population_size`, `evaporation` (rho), `alpha`
(pheromone exponent), `tau_min`, `tau_max`, `seed`.

## Camel Algorithm

`CamelAlgorithm` — for **continuous** problems, modelling a caravan
crossing the desert (Ali, 2016). Each camel steps toward the best oasis
found so far, its stride set by **endurance** (which fades with random
desert heat and with the distance travelled) and stretched by
**dwindling supplies**. Reaching a better position replenishes the
camel; exhausting its endurance kills it and it is reborn at random.
Those restarts make it this library's third-strongest algorithm on
multimodal landscapes.

Key parameters: `population_size`, `min_temperature` / `max_temperature`,
`burden_rate`, `death_rate`, `visibility`, `seed`.

## Cat Swarm Optimization

`CatSwarmOptimization` — for **continuous** problems, based on the two
behaviours of cats (Chu, Tsai & Pan, 2006). Each iteration every cat is
assigned a mode: the resting majority **seeks** — making copies of
itself, tweaking a few dimensions of each and moving to the best one —
while a small fraction **traces**, accelerating toward the best
solution with a velocity update. Running two qualitatively different
searches side by side makes it the library's second-strongest algorithm
on the multimodal Rastrigin function.

Key parameters: `population_size`, `mixture_ratio` (MR), `smp`, `srd`,
`cdc`, `spc`, `velocity_factor`, `max_velocity`, `seed`.

## Cuckoo Search

`CuckooSearch` — for **continuous** problems, based on brood parasitism
(Yang & Deb, 2009). Each nest holds one solution; a cuckoo lays a new
egg after a **Lévy flight** — a heavy-tailed random walk of mostly tiny
steps with rare enormous jumps — and drops it into a random nest, which
keeps it only if it is better. A fraction of the worst nests is then
discovered by the hosts and rebuilt. The Lévy tail is the distinguishing
feature: it escapes local optima without giving up local refinement.

Key parameters: `population_size` (nests), `discovery_rate` (pa),
`step_size` (alpha), `levy_exponent` (beta), `seed`.

## Differential Evolution

`DifferentialEvolution` — for **continuous** problems, and the
strongest all-round algorithm in this library (Storn & Price, 1997).
Its mutant is built by adding the **scaled difference between two
population members** to a third, which makes the step size adapt to the
population spread automatically — no schedule needed. Binomial
crossover mixes the mutant with its target, and greedy selection keeps
the better of the two. Four mutation strategies are available
(`best/1`, `rand/1`, `rand/2`, `current-to-best/1`).

Key parameters: `population_size` (NP), `differential_weight` (F),
`crossover_rate` (CR), `strategy`, `seed`.

## Firefly Algorithm

`FireflyAlgorithm` — for **continuous** problems, based on
bioluminescent attraction (Yang, 2008). Every firefly is drawn toward
brighter ones, but the pull **fades exponentially with distance**, so
each firefly mainly notices its neighbours and the swarm can explore
several regions at once. A decaying random walk on top of the
attraction turns exploration into refinement. It holds this library's
best results on the smooth Sphere and Ackley benchmarks.

Key parameters: `population_size`, `alpha` (randomization),
`alpha_decay`, `beta0`, `gamma` (light absorption), `seed`.

## Genetic Algorithm

`GeneticAlgorithm` — real-coded GA for **continuous** problems.
Evolves a population by tournament selection, blend crossover
(BLX-alpha: children are sampled from an interval slightly wider than
the one their parents span), and Gaussian mutation whose step shrinks
linearly as the evaluation budget is consumed (non-uniform mutation);
elitism carries the best individuals over unchanged. Notably strong on
highly multimodal landscapes, and the algorithm family used by the
GA-WE weighted-ensemble method (Li et al., 2016) that the ensemble
module follows.

Key parameters: `population_size`, `crossover_rate`, `mutation_rate`
(default `1/dimension`), `mutation_scale`, `tournament_size`,
`blend_alpha`, `elitism`, `seed`.

## Komodo Mlipir Algorithm

`KomodoMlipirAlgorithm` — for **continuous** problems, inspired by
Komodo dragons and the Javanese *mlipir* gait (Suyanto et al., 2022).
The ranked population is split into three groups with different jobs:
**big males** exploit by attracting each other (with a 0.5 chance of
distraction), the single **female** either mates the winning big male
or reproduces by parthenogenesis, and **small males** move *mlipir* —
following the big males in only a random subset of dimensions. The
population size **self-adapts**, shrinking while the search improves
and growing when it stagnates. A strong all-rounder: near-ACO-R
precision on smooth functions with much better multimodal behavior.

Key parameters: `population_size`, `big_male_portion` (p),
`mlipir_rate` (d), `max_big_males`, `adaptation_step`,
`min_population` / `max_population`, `parthenogenesis_radius`, `seed`.

## NSGA-II

`NSGA2` — the elitist non-dominated sorting genetic algorithm for
problems with **several conflicting objectives** (Deb et al., 2002).
Unlike every other algorithm here it returns a **Pareto front** rather
than one solution: non-dominated sorting ranks the population into
layers, crowding distance keeps the front spread out, and elitist
replacement ensures no Pareto solution is lost. Use it with
`MultiObjectiveTask` and a `MultiObjectiveProblem` — see the
[multi-objective guide](https://ikn-library.readthedocs.io/en/latest/multiobjective/).

Key parameters: `population_size`, `crossover_rate`, `mutation_rate`,
`mutation_scale`, `blend_alpha`, `seed`.

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
