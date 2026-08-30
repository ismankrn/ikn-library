"""Forest Optimization Algorithm for continuous search spaces.

Reference:
    M. Ghaemi and M.-R. Feizi-Derakhshi, "Forest optimization
    algorithm," Expert Systems with Applications, 41(15), 6676-6687,
    2014.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class ForestOptimizationAlgorithm(Algorithm):
    """Forest Optimization Algorithm (Ghaemi & Feizi-Derakhshi, 2014).

    Models how a forest seeds itself. Each tree is a solution carrying
    one extra attribute the rest of this library has no equivalent for:
    an **age**. Age decides who is allowed to reproduce, and it makes
    the algorithm's two seeding operators mutually exclusive.

    1. **Local seeding** — only trees of age 0 drop seeds, and each seed
       is the parent with a *single* coordinate nudged. This is a fine
       local search around the newest arrivals.
    2. **Ageing** — every tree that already existed grows one year older.
       A tree therefore gets exactly one chance to seed locally before
       it ages out of the operator.
    3. **Population limiting** — trees older than ``life_time``, and
       then the worst trees above ``population_size``, are cut from the
       forest. Crucially they are not destroyed: they go to the
       **candidate population**.
    4. **Global seeding** — a ``transfer_rate`` share of that candidate
       pool is replanted, each with ``gsc`` coordinates replaced by
       fresh random values. Discarded solutions are the raw material
       for long-range exploration, which is unique among the algorithms
       here.
    5. **Elitism** — the best tree's age is reset to 0, so the current
       optimum is always re-seeded locally and never ages out.

    The forest's size is not constant: local seeding inflates it and
    population limiting cuts it back, so ``population_size`` acts as an
    upper bound rather than a fixed headcount.

    Args:
        population_size: Maximum number of trees in the forest (the
            "area limit"), and the number planted at the start.
        life_time: Age at which a tree is cut and sent to the candidate
            population.
        lsc: Local seeding changes — seeds dropped per age-0 tree, each
            altering one randomly chosen coordinate.
        gsc: Global seeding changes — coordinates replaced by random
            values when a candidate tree is replanted.
        transfer_rate: Fraction of the candidate population selected for
            global seeding.
        dx: Local seeding step, as a fraction of the bound range. Decays
            quadratically over the run.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=15, life_time=6, lsc=2, gsc=1,
                 transfer_rate=0.1, dx=0.2, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if life_time < 1:
            raise ValueError("life_time must be >= 1")
        if lsc < 1:
            raise ValueError("lsc must be >= 1")
        if gsc < 1:
            raise ValueError("gsc must be >= 1")
        if not 0.0 <= transfer_rate <= 1.0:
            raise ValueError("transfer_rate must be in [0, 1]")
        if dx <= 0:
            raise ValueError("dx must be > 0")
        self.life_time = int(life_time)
        self.lsc = int(lsc)
        self.gsc = int(gsc)
        self.transfer_rate = float(transfer_rate)
        self.dx = float(dx)

    def init_population(self, task):
        trees = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in trees])
        age = np.zeros(self.population_size, dtype=int)
        return trees, fitness, age

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _local_seeding(self, task, trees, age, step):
        """Age-0 trees each drop ``lsc`` seeds, one coordinate apart."""
        seeds, seed_fitness = [], []
        for i in np.flatnonzero(age == 0):
            for _ in range(self.lsc):
                if task.stopping_condition():
                    return seeds, seed_fitness
                child = trees[i].copy()
                d = self.rng.integers(task.dimension)
                child[d] += self.rng.uniform(-1.0, 1.0) * step[d]
                child = task.repair(child)
                seeds.append(child)
                seed_fitness.append(task.eval(child))
        return seeds, seed_fitness

    def _limit_population(self, trees, fitness, age):
        """Cut old and surplus trees; both feed the candidate pool."""
        too_old = age > self.life_time
        candidates = list(trees[too_old])
        keep = np.flatnonzero(~too_old)

        if len(keep) > self.population_size:
            ranked = keep[np.argsort(fitness[keep])]
            candidates.extend(trees[ranked[self.population_size:]])
            keep = ranked[: self.population_size]

        return trees[keep], fitness[keep], age[keep], candidates

    def _global_seeding(self, task, candidates):
        """Replant part of the discarded pool, far from where it was."""
        seeds, seed_fitness = [], []
        n_transfer = round(self.transfer_rate * len(candidates))
        if n_transfer < 1:
            return seeds, seed_fitness

        chosen = self.rng.choice(len(candidates), n_transfer, replace=False)
        n_changes = min(self.gsc, task.dimension)
        for i in chosen:
            if task.stopping_condition():
                break
            tree = np.asarray(candidates[i]).copy()
            dims = self.rng.choice(task.dimension, n_changes, replace=False)
            tree[dims] = self.rng.uniform(task.lower, task.upper,
                                          task.dimension)[dims]
            seeds.append(tree)
            seed_fitness.append(task.eval(tree))
        return seeds, seed_fitness

    def run_iteration(self, task, state):
        trees, fitness, age = state
        step = self.dx * max(1.0 - self._progress(task), 1e-6) ** 2
        step = step * (task.upper - task.lower)

        seeds, seed_fitness = self._local_seeding(task, trees, age, step)
        age = age + 1                          # the standing forest ages
        if seeds:
            trees = np.vstack([trees, np.array(seeds)])
            fitness = np.concatenate([fitness, seed_fitness])
            age = np.concatenate([age, np.zeros(len(seeds), dtype=int)])

        trees, fitness, age, candidates = self._limit_population(
            trees, fitness, age)

        if candidates:
            seeds, seed_fitness = self._global_seeding(task, candidates)
            if seeds:
                trees = np.vstack([trees, np.array(seeds)])
                fitness = np.concatenate([fitness, seed_fitness])
                age = np.concatenate([age, np.zeros(len(seeds), dtype=int)])

        age[np.argmin(fitness)] = 0            # elitism: the best re-seeds
        return trees, fitness, age
