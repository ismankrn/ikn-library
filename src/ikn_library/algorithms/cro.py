"""Coral Reefs Optimization for continuous search spaces.

Reference:
    S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-Lopez, and
    J. A. Portilla-Figueras, "The coral reefs optimization algorithm:
    a novel metaheuristic for efficiently solving optimization
    problems," The Scientific World Journal, 2014, 739768.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class CoralReefsOptimization(Algorithm):
    """Coral Reefs Optimization (Salcedo-Sanz et al., 2014).

    Models a coral reef colonising a rocky bed. Unlike every other
    algorithm in this library, the population lives on an explicit
    **substrate**: a fixed number of squares, each either holding one
    coral (a solution) or lying empty. Occupancy changes over the run,
    so the effective population size is not constant.

    That substrate is what makes the algorithm distinctive, because it
    turns reproduction into a **competition for space**:

    1. **Broadcast spawning** — a fraction ``broadcast_fraction`` of
       corals release gametes into the water; they are paired at random
       and crossed to make larvae (external sexual reproduction).
    2. **Brooding** — the rest release self-fertilised larvae, made by
       mutating the parent (internal sexual reproduction).
    3. **Larval settlement** — each larva picks a random square and gets
       ``settlement_attempts`` tries to take hold. An empty square is
       free real estate; an occupied one is taken over only if the larva
       is fitter than the incumbent. Larvae that never win are lost.
    4. **Budding** — the fittest ``asexual_fraction`` of corals clone
       themselves and their copies compete for settlement too.
    5. **Depredation** — with probability ``depredation_prob``, the
       worst ``depredation_fraction`` of corals are eaten, freeing their
       squares.

    Selection therefore happens at *settlement*, not by ranking the
    population: a good larva can still be lost if it keeps landing on
    better-occupied squares. Depredation is what stops the reef from
    saturating, since a full reef admits only strict improvements.

    Args:
        population_size: Number of squares in the reef (its capacity,
            not the number of live corals).
        initial_occupation: Fraction of squares occupied at the start.
            Free space early on lets weak-but-novel larvae survive.
        broadcast_fraction: Fraction of corals that spawn by crossover
            rather than by brooding.
        asexual_fraction: Fraction of the fittest corals that bud.
            Defaults to 0 (budding off): cloning the best corals floods
            the reef with duplicates, which collapses crossover. See the
            algorithm page for the measurements.
        depredation_fraction: Fraction of the worst corals eligible to
            be eaten each generation.
        depredation_prob: Probability that an eligible coral is eaten.
        settlement_attempts: Tries a larva gets to find a square.
        mutation_scale: Brooding mutation size, as a fraction of the
            bound range. Decays quadratically over the run.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=80, initial_occupation=0.8,
                 broadcast_fraction=0.5, asexual_fraction=0.0,
                 depredation_fraction=0.1, depredation_prob=0.1,
                 settlement_attempts=3, mutation_scale=0.03, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if not 0.0 < initial_occupation <= 1.0:
            raise ValueError("initial_occupation must be in (0, 1]")
        if not 0.0 <= broadcast_fraction <= 1.0:
            raise ValueError("broadcast_fraction must be in [0, 1]")
        if not 0.0 <= asexual_fraction <= 1.0:
            raise ValueError("asexual_fraction must be in [0, 1]")
        if not 0.0 <= depredation_fraction <= 1.0:
            raise ValueError("depredation_fraction must be in [0, 1]")
        if not 0.0 <= depredation_prob <= 1.0:
            raise ValueError("depredation_prob must be in [0, 1]")
        if settlement_attempts < 1:
            raise ValueError("settlement_attempts must be >= 1")
        if mutation_scale <= 0:
            raise ValueError("mutation_scale must be > 0")
        self.initial_occupation = float(initial_occupation)
        self.broadcast_fraction = float(broadcast_fraction)
        self.asexual_fraction = float(asexual_fraction)
        self.depredation_fraction = float(depredation_fraction)
        self.depredation_prob = float(depredation_prob)
        self.settlement_attempts = int(settlement_attempts)
        self.mutation_scale = float(mutation_scale)

    def init_population(self, task):
        reef = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.full(self.population_size, np.inf)
        occupied = np.zeros(self.population_size, dtype=bool)

        n_alive = max(round(self.initial_occupation * self.population_size), 1)
        for i in self.rng.choice(self.population_size, n_alive, replace=False):
            occupied[i] = True
            fitness[i] = task.eval(reef[i])
        return reef, fitness, occupied

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _spawn(self, task, reef, alive, scale):
        """Broadcast spawning and brooding produce the larvae."""
        shuffled = self.rng.permutation(alive)
        n_broadcast = int(len(shuffled) * self.broadcast_fraction)
        n_broadcast -= n_broadcast % 2          # spawners pair up
        broadcasters = shuffled[:n_broadcast]
        brooders = shuffled[n_broadcast:]
        span = task.upper - task.lower
        larvae = []

        # External sexual reproduction: blend two parents.
        for a, b in zip(broadcasters[0::2], broadcasters[1::2]):
            weight = self.rng.uniform(-0.25, 1.25, task.dimension)
            larvae.append(reef[a] + weight * (reef[b] - reef[a]))

        # Internal sexual reproduction: perturb one parent.
        for i in brooders:
            larvae.append(reef[i]
                          + scale * span * self.rng.normal(0.0, 1.0, task.dimension))
        return larvae

    def _settle(self, task, reef, fitness, occupied, larvae):
        """Each larva competes for a square, and may not find one."""
        for larva in larvae:
            if task.stopping_condition():
                break
            larva = task.repair(larva)
            larva_fitness = task.eval(larva)
            for _ in range(self.settlement_attempts):
                square = self.rng.integers(self.population_size)
                if not occupied[square] or larva_fitness < fitness[square]:
                    reef[square] = larva
                    fitness[square] = larva_fitness
                    occupied[square] = True
                    break                       # settled; the larva is done
        return reef, fitness, occupied

    def _depredate(self, fitness, occupied):
        """The worst corals are eaten, freeing their squares."""
        alive = np.flatnonzero(occupied)
        if len(alive) <= 1:
            return occupied
        n_eaten = round(self.depredation_fraction * len(alive))
        if n_eaten < 1:
            return occupied
        worst = alive[np.argsort(fitness[alive])[-n_eaten:]]
        for i in worst:
            # Never eat the last coral, and never the reef's best.
            if occupied.sum() <= 1:
                break
            if self.rng.random() < self.depredation_prob:
                occupied[i] = False
                fitness[i] = np.inf
        return occupied

    def run_iteration(self, task, state):
        reef, fitness, occupied = state
        alive = np.flatnonzero(occupied)
        if len(alive) == 0:                     # the reef died out; reseed
            return self.init_population(task)

        scale = self.mutation_scale * max(1.0 - self._progress(task), 1e-6) ** 2
        larvae = self._spawn(task, reef, alive, scale)

        # Budding: the fittest corals clone themselves.
        n_bud = round(self.asexual_fraction * len(alive))
        if n_bud >= 1:
            best = alive[np.argsort(fitness[alive])[:n_bud]]
            larvae.extend(reef[i].copy() for i in best)

        self.rng.shuffle(larvae)
        reef, fitness, occupied = self._settle(
            task, reef, fitness, occupied, larvae)
        occupied = self._depredate(fitness, occupied)
        return reef, fitness, occupied
