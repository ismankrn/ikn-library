"""Fish School Search for continuous search spaces.

Reference:
    C. J. A. Bastos Filho, F. B. de Lima Neto, A. J. C. C. Lins,
    A. I. S. Nascimento, and M. P. Lima, "A novel search algorithm based
    on fish school behavior," in IEEE International Conference on
    Systems, Man and Cybernetics (SMC 2008), 2646-2651, 2008.
"""

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class FishSchoolSearch(Algorithm):
    """Fish School Search (Bastos Filho et al., 2008).

    A school of fish searches for food, and each fish carries a
    **weight** that grows when it finds some. That weight is the
    algorithm's memory of success, and it drives everything else: the
    school's centre of mass is weight-biased, and the *total* weight
    decides whether the school contracts around good regions or expands
    to look elsewhere.

    Four operators run in sequence each iteration:

    1. **Individual movement** — every fish tries a small random step,
       keeping it only if the food improves.
    2. **Feeding** — weights change in proportion to how much each fish
       improved, relative to the best improvement in the school.
    3. **Collective instinctive movement** — the whole school drifts by
       the improvement-weighted average of the successful steps, so
       fish that found nothing still follow those that did.
    4. **Collective volitive movement** — if the school gained weight
       overall it *contracts* toward its barycentre (exploitation);
       if it lost weight it *expands* away from it (exploration).

    Args:
        population_size: Number of fish.
        step_individual: Initial individual step, as a fraction of each
            dimension's bound range. Decays linearly to
            ``step_individual_final``.
        step_individual_final: Final individual step fraction.
        step_volitive_factor: Volitive step as a multiple of the current
            individual step.
        weight_scale: Upper limit of a fish's weight; fish start at half
            of it.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=100, step_individual=0.05,
                 step_individual_final=1e-6, step_volitive_factor=2.0,
                 weight_scale=100.0, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if step_individual <= 0:
            raise ValueError("step_individual must be > 0")
        if not 0 < step_individual_final <= step_individual:
            raise ValueError(
                "step_individual_final must be in (0, step_individual]")
        if step_volitive_factor <= 0:
            raise ValueError("step_volitive_factor must be > 0")
        if weight_scale <= 1:
            raise ValueError("weight_scale must be > 1")
        self.step_individual = float(step_individual)
        self.step_individual_final = float(step_individual_final)
        self.step_volitive_factor = float(step_volitive_factor)
        self.weight_scale = float(weight_scale)

    def init_population(self, task):
        school = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in school])
        weights = np.full(self.population_size, self.weight_scale / 2.0)
        return school, fitness, weights

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _current_step(self, task):
        """Individual step, decaying linearly over the run."""
        progress = self._progress(task)
        return (self.step_individual
                - progress * (self.step_individual - self.step_individual_final))

    def run_iteration(self, task, state):
        school, fitness, weights = state
        span = task.upper - task.lower
        step = self._current_step(task) * span

        # 1. Individual movement: a greedy random step per fish.
        improvement = np.zeros(self.population_size)
        displacement = np.zeros_like(school)
        for i in range(self.population_size):
            if task.stopping_condition():
                break
            candidate = task.repair(
                school[i] + step * self.rng.uniform(-1.0, 1.0, task.dimension))
            candidate_fitness = task.eval(candidate)
            if candidate_fitness < fitness[i]:
                improvement[i] = fitness[i] - candidate_fitness
                displacement[i] = candidate - school[i]
                school[i], fitness[i] = candidate, candidate_fitness

        # 2. Feeding: weight grows with the relative improvement.
        best_improvement = improvement.max()
        if best_improvement > 0:
            weights = np.clip(weights + improvement / best_improvement,
                              1.0, self.weight_scale)

        # 3. Collective instinctive movement: everyone follows the
        #    improvement-weighted average of the successful steps.
        total_improvement = improvement.sum()
        if total_improvement > 0:
            drift = (displacement * improvement[:, None]).sum(axis=0) / total_improvement
            school = np.array([task.repair(x + drift) for x in school])

        # 4. Collective volitive movement: contract if the school fed
        #    well this round, expand if it did not.
        total_weight = weights.sum()
        previous_weight = state[2].sum()
        barycentre = (school * weights[:, None]).sum(axis=0) / total_weight
        direction = -1.0 if total_weight > previous_weight else 1.0
        volitive_step = self.step_volitive_factor * step
        offsets = school - barycentre
        distances = np.linalg.norm(offsets, axis=1, keepdims=True)
        distances[distances == 0] = 1.0
        school = school + (direction * volitive_step
                           * self.rng.random((self.population_size, 1))
                           * offsets / distances)
        school = np.array([task.repair(x) for x in school])

        # Positions changed in steps 3 and 4, so refresh the fitness.
        for i in range(self.population_size):
            if task.stopping_condition():
                break
            fitness[i] = task.eval(school[i])

        return school, fitness, weights
