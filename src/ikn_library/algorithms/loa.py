"""Lion Optimization Algorithm for continuous search spaces.

Reference:
    M. Yazdani and F. Jolai, "Lion Optimization Algorithm (LOA): a
    nature-inspired metaheuristic algorithm," Journal of Computational
    Design and Engineering, 3(1), 24-36, 2016.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class LionOptimizationAlgorithm(Algorithm):
    """Lion Optimization Algorithm (Yazdani & Jolai, 2016).

    The most elaborate algorithm in this library. Lions are split into
    **prides** and a group of **nomads**, each lion is male or female,
    and the two groups run different operators — so unlike every other
    algorithm here, the population is *heterogeneous* and a lion's role
    determines how it moves.

    Each iteration runs the pride social cycle:

    1. **Hunting** — some pride females hunt together. They are split
       into a centre group and two wings, and a *prey* is placed at
       their mean position. The centre closes in directly while the
       wings encircle from the opposite side; a hunter that improves
       makes the prey bolt further away.
    2. **Moving to a safe place** — the remaining females move toward a
       position in the pride's **territory**, the set of best positions
       its members have ever visited.
    3. **Roaming** — pride males wander through the territory, and
       nomads roam the whole search space at random.
    4. **Mating** — females breed with pride males; offspring are a
       weighted blend of both parents, then mutated.
    5. **Defence and migration** — weak males are driven out to become
       nomads, strong nomad males take over prides, and some females
       migrate between prides.

    The territory is what ties this together: it is a *memory* of good
    positions that outlives the lions standing on them, and both the
    safe-place move and roaming draw from it.

    Args:
        population_size: Total number of lions.
        n_prides: Number of prides the residents are split into.
        nomad_ratio: Fraction of lions that are nomads.
        sex_ratio: Fraction of pride lions that are female.
        roaming_ratio: Fraction of the territory a male roams toward.
        mating_ratio: Fraction of females that mate each iteration.
        mutation_prob: Per-coordinate mutation chance for offspring.
        migration_ratio: Fraction of pride females that migrate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=50, n_prides=4, nomad_ratio=0.2,
                 sex_ratio=0.8, roaming_ratio=0.2, mating_ratio=0.3,
                 mutation_prob=0.1, migration_ratio=0.4, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if n_prides < 1:
            raise ValueError("n_prides must be >= 1")
        if not 0.0 < nomad_ratio < 1.0:
            raise ValueError("nomad_ratio must be in (0, 1)")
        if not 0.0 < sex_ratio < 1.0:
            raise ValueError("sex_ratio must be in (0, 1)")
        if not 0.0 < roaming_ratio <= 1.0:
            raise ValueError("roaming_ratio must be in (0, 1]")
        if not 0.0 <= mating_ratio <= 1.0:
            raise ValueError("mating_ratio must be in [0, 1]")
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1]")
        if not 0.0 <= migration_ratio <= 1.0:
            raise ValueError("migration_ratio must be in [0, 1]")
        self.n_prides = int(n_prides)
        self.nomad_ratio = float(nomad_ratio)
        self.sex_ratio = float(sex_ratio)
        self.roaming_ratio = float(roaming_ratio)
        self.mating_ratio = float(mating_ratio)
        self.mutation_prob = float(mutation_prob)
        self.migration_ratio = float(migration_ratio)

    # ---------------------------------------------------------------- setup

    def init_population(self, task):
        lions = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in lions])

        n_nomads = max(round(self.nomad_ratio * self.population_size), 1)
        order = self.rng.permutation(self.population_size)
        pride = np.empty(self.population_size, dtype=int)
        pride[order[:n_nomads]] = -1                      # -1 marks a nomad
        pride[order[n_nomads:]] = self.rng.integers(
            0, self.n_prides, self.population_size - n_nomads)

        female = self.rng.random(self.population_size) < self.sex_ratio
        # Nomads invert the ratio: mostly male, per the paper.
        female[pride == -1] = (
            self.rng.random(int((pride == -1).sum())) < 1.0 - self.sex_ratio)

        # Territory: the best position each lion has ever occupied.
        return lions, fitness, pride, female, lions.copy(), fitness.copy()

    def _remember(self, i, lions, fitness, best_x, best_f):
        if fitness[i] < best_f[i]:
            best_x[i], best_f[i] = lions[i].copy(), fitness[i]

    # -------------------------------------------------------------- hunting

    def _hunt(self, task, members, lions, fitness, best_x, best_f):
        """Females encircle a prey placed at their mean position."""
        if len(members) < 3:
            return
        groups = self.rng.integers(0, 3, len(members))
        # The group with the best total fitness takes the centre.
        totals = [fitness[members[groups == g]].sum() if (groups == g).any()
                  else np.inf for g in range(3)]
        centre_group = int(np.argmin(totals))

        prey = lions[members].mean(axis=0)
        for k, i in enumerate(members):
            if task.stopping_condition():
                return
            if groups[k] == centre_group:
                low, high = np.minimum(lions[i], prey), np.maximum(lions[i], prey)
            else:                                   # wings attack the far side
                opposite = 2.0 * prey - lions[i]
                low, high = np.minimum(opposite, prey), np.maximum(opposite, prey)

            previous = fitness[i]
            lions[i] = task.repair(self.rng.uniform(low, high))
            fitness[i] = task.eval(lions[i])
            self._remember(i, lions, fitness, best_x, best_f)

            if fitness[i] < previous:               # a hit: the prey bolts
                gain = abs(previous - fitness[i]) / (abs(previous) + 1e-12)
                prey = prey + self.rng.random() * min(gain, 1.0) * (
                    prey - lions[i])

    # ---------------------------------------------------- safe place / roam

    def _tournament_pick(self, candidates, best_f):
        """Pick a territory position, favouring good ones."""
        a, b = self.rng.choice(candidates, 2, replace=True)
        return a if best_f[a] <= best_f[b] else b

    def _move_to_safe_place(self, task, members, territory, lions, fitness,
                            best_x, best_f):
        """Drift toward a remembered good position, with a sideways kick."""
        for i in members:
            if task.stopping_condition():
                return
            target = best_x[self._tournament_pick(territory, best_f)]
            offset = target - lions[i]
            distance = np.linalg.norm(offset)
            if distance == 0:
                continue
            direction = offset / distance

            # A random direction perpendicular to the approach.
            sideways = self.rng.normal(size=task.dimension)
            sideways -= sideways.dot(direction) * direction
            norm = np.linalg.norm(sideways)
            sideways = sideways / norm if norm > 0 else sideways

            angle = self.rng.uniform(-np.pi / 6.0, np.pi / 6.0)
            lions[i] = task.repair(
                lions[i] + 2.0 * distance * self.rng.random() * direction
                + np.tan(angle) * distance * sideways)
            fitness[i] = task.eval(lions[i])
            self._remember(i, lions, fitness, best_x, best_f)

    def _roam_pride(self, task, males, territory, lions, fitness,
                    best_x, best_f):
        """Males wander the territory, one remembered position at a time."""
        n_targets = max(round(self.roaming_ratio * len(territory)), 1)
        for i in males:
            for target in self.rng.choice(territory, n_targets, replace=True):
                if task.stopping_condition():
                    return
                offset = best_x[target] - lions[i]
                distance = np.linalg.norm(offset)
                if distance == 0:
                    continue
                angle = self.rng.uniform(-np.pi / 6.0, np.pi / 6.0)
                lions[i] = task.repair(
                    lions[i] + self.rng.uniform(0.0, 2.0 * distance)
                    * offset / distance
                    + np.tan(angle) * self.rng.normal(size=task.dimension)
                    * distance * 0.1)
                fitness[i] = task.eval(lions[i])
                self._remember(i, lions, fitness, best_x, best_f)

    def _roam_nomads(self, task, nomads, lions, fitness, best_x, best_f):
        """Nomads jump anywhere, more often when they are doing badly."""
        if len(nomads) == 0:
            return
        worst, best = fitness[nomads].max(), fitness[nomads].min()
        spread = worst - best
        for i in nomads:
            if task.stopping_condition():
                return
            # Probability rises with how poor this nomad is.
            chance = 0.1 + 0.9 * ((fitness[i] - best) / spread
                                  if spread > 1e-30 else 0.0)
            if self.rng.random() < chance:
                lions[i] = self.rng.uniform(task.lower, task.upper)
            else:
                lions[i] = task.repair(
                    lions[i] + 0.1 * (task.upper - task.lower)
                    * self.rng.normal(size=task.dimension))
            fitness[i] = task.eval(lions[i])
            self._remember(i, lions, fitness, best_x, best_f)

    # --------------------------------------------------------------- mating

    def _mate(self, task, females, males, lions, fitness, best_x, best_f):
        """Offspring blend one female with the pride's males, then mutate."""
        if len(females) == 0 or len(males) == 0:
            return
        n_mating = max(round(self.mating_ratio * len(females)), 1)
        for i in self.rng.choice(females, min(n_mating, len(females)),
                                 replace=False):
            if task.stopping_condition():
                return
            beta = self.rng.normal(0.5, 0.1)
            partner = lions[self.rng.choice(males)]
            offspring = beta * lions[i] + (1.0 - beta) * partner

            mutate = self.rng.random(task.dimension) < self.mutation_prob
            offspring = np.where(
                mutate, self.rng.uniform(task.lower, task.upper), offspring)
            offspring = task.repair(offspring)
            offspring_fitness = task.eval(offspring)

            # The cub replaces its mother only by being better.
            if offspring_fitness < fitness[i]:
                lions[i], fitness[i] = offspring, offspring_fitness
                self._remember(i, lions, fitness, best_x, best_f)

    # ------------------------------------------------ defence and migration

    def _defend_and_migrate(self, fitness, pride, female):
        """Weak pride males are exiled; strong nomad males take over."""
        for p in range(self.n_prides):
            males = np.flatnonzero((pride == p) & ~female)
            nomad_males = np.flatnonzero((pride == -1) & ~female)
            if len(males) == 0 or len(nomad_males) == 0:
                continue
            weakest = males[np.argmax(fitness[males])]
            strongest = nomad_males[np.argmin(fitness[nomad_males])]
            if fitness[strongest] < fitness[weakest]:
                pride[weakest], pride[strongest] = -1, p

        # Some females leave their pride, and the vacancies they open are
        # then refilled from the nomads. Both halves matter: exiling
        # without refilling lets the nomad group grow without bound, and
        # nomads only search at random.
        vacancies = []
        for p in range(self.n_prides):
            females = np.flatnonzero((pride == p) & female)
            n_move = round(self.migration_ratio * len(females))
            if n_move < 1:
                continue
            leaving = females[np.argsort(fitness[females])[-n_move:]]
            pride[leaving] = -1
            vacancies.extend([p] * n_move)

        # The best nomad females take the open places.
        nomad_females = np.flatnonzero((pride == -1) & female)
        ranked = nomad_females[np.argsort(fitness[nomad_females])]
        for p, i in zip(vacancies, ranked):
            pride[i] = p
        return pride

    # ------------------------------------------------------------ main loop

    def run_iteration(self, task, state):
        lions, fitness, pride, female, best_x, best_f = state

        for p in range(self.n_prides):
            if task.stopping_condition():
                break
            members = np.flatnonzero(pride == p)
            if len(members) == 0:
                continue
            females = members[female[members]]
            males = members[~female[members]]

            # Half the females hunt, the rest head for the territory.
            shuffled = self.rng.permutation(females)
            hunters, others = shuffled[: len(shuffled) // 2], \
                shuffled[len(shuffled) // 2:]

            self._hunt(task, hunters, lions, fitness, best_x, best_f)
            self._move_to_safe_place(task, others, members, lions, fitness,
                                     best_x, best_f)
            self._roam_pride(task, males, members, lions, fitness,
                             best_x, best_f)
            self._mate(task, females, males, lions, fitness, best_x, best_f)

        self._roam_nomads(task, np.flatnonzero(pride == -1), lions, fitness,
                          best_x, best_f)
        pride = self._defend_and_migrate(fitness, pride, female)

        return lions, fitness, pride, female, best_x, best_f
