# Algorithms

All algorithms share the same interface: construct with their
parameters, then call `run(task)` to get `(best_x, best_fitness)`.

```python
from ikn_library.algorithms import (
    AntColonyOptimization,
    ArtificialBeeColony,
    BacterialForagingOptimization,
    BatAlgorithm,
    BeesAlgorithm,
    BinaryAntColonyOptimization,
    CamelAlgorithm,
    CatSwarmOptimization,
    ClonalSelectionAlgorithm,
    CoralReefsOptimization,
    CuckooSearch,
    DifferentialEvolution,
    FireflyAlgorithm,
    FireworksAlgorithm,
    FishSchoolSearch,
    FlowerPollinationAlgorithm,
    ForestOptimizationAlgorithm,
    GeneticAlgorithm,
    GravitationalSearchAlgorithm,
    GreyWolfOptimizer,
    HarmonySearch,
    HarrisHawksOptimization,
    KomodoMlipirAlgorithm,
    KrillHerd,
    LionOptimizationAlgorithm,
    NSGA2,
    SimulatedAnnealing,
)
```

| Algorithm | Class | Domain | Reference |
|---|---|---|---|
| Ant Colony Optimization (ACO-R) | `AntColonyOptimization` | continuous | Socha & Dorigo, EJOR 185(3), 2008 |
| Artificial Bee Colony | `ArtificialBeeColony` | continuous | Karaboga, TR06, 2005; Karaboga & Basturk, JOGO 39(3), 2007 |
| Bacterial Foraging Optimization | `BacterialForagingOptimization` | continuous | Passino, IEEE CSM 22(3), 2002 |
| Bat Algorithm | `BatAlgorithm` | continuous | Yang, NICSO 2010 |
| Bees Algorithm | `BeesAlgorithm` | continuous | Pham et al., 2005; Pham & Castellani, 2009 |
| Binary Ant Colony Optimization | `BinaryAntColonyOptimization` | binary / subsets | hyper-cube pheromone update |
| Camel Algorithm | `CamelAlgorithm` | continuous | Ali, IJSBAR, 2016 |
| Cat Swarm Optimization | `CatSwarmOptimization` | continuous | Chu, Tsai & Pan, PRICAI 2006 |
| Clonal Selection Algorithm | `ClonalSelectionAlgorithm` | continuous | de Castro & Von Zuben, IEEE TEC 6(3), 2002 |
| Coral Reefs Optimization | `CoralReefsOptimization` | continuous | Salcedo-Sanz et al., Sci. World J. 2014 |
| Cuckoo Search | `CuckooSearch` | continuous | Yang & Deb, NaBIC 2009 |
| Differential Evolution | `DifferentialEvolution` | continuous | Storn & Price, JOGO 11(4), 1997 |
| Firefly Algorithm | `FireflyAlgorithm` | continuous | Yang, SAGA 2009 |
| Fireworks Algorithm | `FireworksAlgorithm` | continuous | Tan & Zhu, ICSI 2010 |
| Fish School Search | `FishSchoolSearch` | continuous | Bastos Filho et al., IEEE SMC 2008 |
| Genetic Algorithm (real-coded) | `GeneticAlgorithm` | continuous | Holland, 1975; BLX-alpha: Eshelman & Schaffer, 1993 |
| Gravitational Search Algorithm | `GravitationalSearchAlgorithm` | continuous | Rashedi et al., Inf. Sci. 179(13), 2009 |
| Grey Wolf Optimizer | `GreyWolfOptimizer` | continuous | Mirjalili et al., Adv. Eng. Software 69, 2014 |
| Harmony Search | `HarmonySearch` | continuous | Geem, Kim & Loganathan, Simulation 76(2), 2001 |
| Harris Hawks Optimization | `HarrisHawksOptimization` | continuous | Heidari et al., FGCS 97, 2019 |
| Krill Herd Algorithm | `KrillHerd` | continuous | Gandomi & Alavi, CNSNS 17(12), 2012 |
| Lion Optimization Algorithm | `LionOptimizationAlgorithm` | continuous | Yazdani & Jolai, JCDE 3(1), 2016 |
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

## Bacterial Foraging Optimization

`BacterialForagingOptimization` — for **continuous** problems, modelling
how *E. coli* forages (Passino, 2002). Three mechanisms run on three
timescales: **chemotaxis** every iteration (tumble into a random
direction, then swim along it while conditions improve — a small
directional line search), **reproduction** every ~20 iterations (the
healthiest half, judged by fitness accumulated over a lifetime, splits
while the rest die), and **elimination-dispersal** every ~100 iterations
(bacteria are randomly re-placed). It is the weakest performer here on
the standard benchmarks, but valuable as a study in multi-timescale
design.

Key parameters: `population_size`, `step_size`, `n_swim`,
`reproduction_interval`, `elimination_interval`, `elimination_prob`,
`seed`.

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

## Clonal Selection Algorithm

`ClonalSelectionAlgorithm` — for **continuous** problems, drawn from
the adaptive immune system (de Castro & Von Zuben, 2002). Solutions are
antibodies ranked by *affinity*, and two rules follow from that single
quantity: better antibodies are **cloned more** (concentrating the
budget on what works) and their clones **mutate less** (refining rather
than wandering). A poor antibody thus gets one clone flung far away
while a good one gets many small perturbations, so exploration and
exploitation are balanced by biology rather than by a schedule. The
worst few antibodies are replaced at random each generation, which is
the algorithm's only source of new material — it has no crossover or
recombination of any kind.

Key parameters: `population_size`, `n_select`, `clone_factor`,
`n_replace`, `rho`, `seed`.

## Coral Reefs Optimization

`CoralReefsOptimization` — for **continuous** problems, modelling a reef
colonising a rocky bed (Salcedo-Sanz et al., 2014). It is the only
algorithm here whose population lives on an explicit **substrate**: a
fixed number of squares, each holding one coral or lying empty, so the
number of live solutions changes over the run. Corals reproduce by
crossover (broadcast spawning) or mutation (brooding), and the resulting
larvae must then **compete for space** — a larva takes an empty square
freely but displaces an occupant only by beating it, and is lost
entirely if it never lands well. Depredation eats the worst corals to
keep the reef from saturating. Selection is therefore local and
stochastic rather than a global ranking, which makes it one of the
library's stronger performers on the multimodal Rastrigin function.

Key parameters: `population_size` (reef capacity), `initial_occupation`,
`broadcast_fraction`, `asexual_fraction`, `depredation_fraction`,
`depredation_prob`, `settlement_attempts`, `mutation_scale`, `seed`.

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

## Fireworks Algorithm

`FireworksAlgorithm` — for **continuous** problems, and the strongest
algorithm in this library on the standard benchmarks (Tan & Zhu, 2010).
Each firework explodes into sparks, with two quantities set by its
quality in *opposite* directions: good fireworks get **more sparks** but
a **smaller amplitude** (exploitation), poor ones get few sparks spread
**wide** (exploration). Exploration and exploitation therefore happen
simultaneously across the population rather than in phases. A few
multiplicative Gaussian sparks supply the fine-scale refinement.

Key parameters: `population_size` (fireworks), `n_sparks`,
`max_amplitude`, `n_gaussian_sparks`, `spark_bounds`, `seed`.

## Fish School Search

`FishSchoolSearch` — for **continuous** problems, modelling a foraging
fish school (Bastos Filho et al., 2008). Each fish carries a **weight**
that grows when it finds food, and four operators run each iteration:
an individual random step, feeding, a collective drift along the
improvement-weighted average step, and a volitive move that
**contracts** the school when it gained weight or **expands** it when
it did not. Exploration and exploitation are thus switched by the
school's own recent success rather than a preset schedule.

Key parameters: `population_size`, `step_individual` (and its final
value), `step_volitive_factor`, `weight_scale`, `seed`.

## Flower Pollination Algorithm

`FlowerPollinationAlgorithm` — for **continuous** problems, and the
simplest algorithm in the library (Yang, 2012). Each flower flips a coin
each iteration: with probability `switch_probability` it does **global
pollination**, a Lévy flight aimed at the current best flower; otherwise
**local pollination**, drifting along the difference between two random
flowers. A new flower is kept only if it is better. The two operators
are borrowed rather than new — the first is Cuckoo Search's move, the
second is a Differential Evolution difference vector without crossover —
but switching between them per flower per iteration turns out to be
enough. It reaches near-machine precision on smooth functions and gives
the best Rosenbrock result measured in this library.

Key parameters: `population_size`, `switch_probability`, `gamma`,
`levy_exponent`, `seed`.

## Forest Optimization Algorithm

`ForestOptimizationAlgorithm` — for **continuous** problems, modelling
how a forest seeds itself (Ghaemi & Feizi-Derakhshi, 2014). Each tree
carries an **age** that gates reproduction: only age-0 trees drop local
seeds, and every standing tree grows a year older each iteration, so a
tree gets exactly one chance to seed before ageing out. Trees that
exceed the life time, or that fall below the area limit in fitness, are
cut into a **candidate population** — and that discarded pool is the
only source of long-range exploration, making this the one algorithm
here that recycles solutions it has already rejected.

Note that local seeding changes **one coordinate at a time**, which
suits *separable* problems especially well; its strong Rastrigin score
is largely an artefact of that, and the detail page documents the
rotated-function evidence.

Key parameters: `population_size` (area limit), `life_time`, `lsc`,
`gsc`, `transfer_rate`, `dx`, `seed`.

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

## Gravitational Search Algorithm

`GravitationalSearchAlgorithm` — for **continuous** problems, and the
only algorithm here based on physics rather than biology (Rashedi
et al., 2009). Solutions are masses obeying Newton's law of
gravitation, with mass growing with fitness. Because mass acts twice —
as gravitational mass (attracting others) and as inertia (resisting
movement) — good solutions both pull the swarm in and hold their
ground, while light poor agents are flung about and explore. The
gravitational constant decays over the run and the set of attracting
agents (**Kbest**) shrinks to the elite.

Key parameters: `population_size`, `g0`, `alpha` (decay rate),
`final_kbest`, `max_velocity`, `seed`.

## Grey Wolf Optimizer

`GreyWolfOptimizer` — for **continuous** problems, based on the social
hierarchy of grey wolves (Mirjalili et al., 2014). The three best
solutions are named alpha, beta and delta, and every other wolf moves to
the **average of three proposals**, one from each leader. Following
three disagreeing attractors rather than one global best keeps the pack
spread over the region they bracket, giving diversity maintenance with
no explicit mechanism for it. A single coefficient falling from 2 to 0
switches the pack from searching to attacking, and unusually for its
era that schedule was tied to the run's progress in the original paper.

Note that GWO is **biased toward the origin**: its exceptional scores on
the standard benchmarks depend on their optima sitting at zero, and
shifting them costs 82 orders of magnitude on Sphere. The detail page
documents the evidence and the mechanism.

Key parameters: `population_size`, `a_start`, `a_end`, `seed`.

## Harmony Search

`HarmonySearch` — for **continuous** problems, modelled on musicians
improvising (Geem et al., 2001). A **harmony memory** holds the best
solutions, and one new solution is improvised per iteration, replacing
the worst if it is better. Its distinctive trait is that each decision
variable is drawn **independently from a different randomly chosen
harmony** — every other recombination here mixes exactly two parents,
while Harmony Search mixes across the whole memory at once.

Two caveats are documented on the detail page: the algorithm was shown
by Weyland (2010) to be a special case of evolution strategies rather
than a new method, and its per-coordinate construction flatters
*separable* benchmarks, so its Rastrigin score does not transfer to
rotated problems.

Key parameters: `population_size` (HMS), `hmcr`, `par`, `bandwidth`,
`seed`.

## Harris Hawks Optimization

`HarrisHawksOptimization` — for **continuous** problems, modelling the
cooperative pounce of Harris's hawks (Heidari et al., 2019). It is the
most branched algorithm here: six moves selected by a two-level test on
the prey's remaining *escaping energy*. While the prey is strong the
hawks scatter; as its energy drains they close in with one of four
besiege moves, chosen by how much energy is left and whether the prey
bolts. The two **dive** moves build two candidates — a direct approach
and a Lévy-flight zigzag — evaluate both, and keep whichever improves,
or neither.

It is the strongest all-round performer measured in this library, and
the only algorithm whose ranking survives rotating **and** shifting the
benchmark. It does still carry a mild origin bias, documented on the
detail page.

Key parameters: `population_size`, `energy_start`, `levy_exponent`,
`levy_scale`, `seed`.

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


## Krill Herd Algorithm

`KrillHerd` — for **continuous** problems, modelling a krill swarm as a
Lagrangian system (Gandomi & Alavi, 2012). Each krill's velocity is
built from three terms: **induced motion** from neighbours within a
radius derived from the herd's own spacing, **foraging** toward an
inverse-fitness-weighted *food centre* plus its personal best, and
**physical diffusion** that fades out. The first two carry inertia, so
motion is smoothed across iterations rather than recomputed.

Its notable property is robustness: every coupling uses position
differences and normalized fitness gaps, so the search is invariant to
translating **and** rotating the problem. Its scores vary by under 15%
across all four benchmark variants, where several higher-scoring
algorithms lose orders of magnitude.

Key parameters: `population_size`, `n_max`, `v_f`, `d_max`, `inertia`,
`c_t`, `crossover_rate`, `seed`.

## Lion Optimization Algorithm

`LionOptimizationAlgorithm` — for **continuous** problems, and the most
elaborate algorithm here (Yazdani & Jolai, 2016). Lions are split into
**prides** and **nomads**, each is male or female, and group and sex
decide which of seven operators moves it — making this the only
algorithm in the library with a genuinely **heterogeneous population**.
Pride females hunt around a shared prey or move toward the pride's
**territory** (the best positions its members have ever visited); males
roam that territory; females mate with males to produce blended cubs;
weak males are exiled to the nomads and strong nomads take their place.

Two caveats are documented on the detail page: its Ackley result is
bimodal, so the mean describes no actual run, and screening it on a
short budget picks a configuration that loses at the full one.

Key parameters: `population_size`, `n_prides`, `nomad_ratio`,
`sex_ratio`, `roaming_ratio`, `mating_ratio`, `mutation_prob`,
`migration_ratio`, `seed`.
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
