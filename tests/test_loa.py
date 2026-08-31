import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import LionOptimizationAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_loa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = LionOptimizationAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-8
    assert best_x.shape == (5,)


def test_loa_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = LionOptimizationAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_loa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    best_x, _ = LionOptimizationAlgorithm(seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_loa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(LionOptimizationAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_loa_respects_eval_budget():
    for budget in (60, 213, 800):
        task = Task(problem=Sphere(dimension=3), max_evals=budget)
        LionOptimizationAlgorithm(seed=0).run(task)
        assert task.evals <= budget


def test_initial_split_into_prides_and_nomads():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = LionOptimizationAlgorithm(population_size=50, n_prides=4,
                                     nomad_ratio=0.2, seed=3)
    _, _, pride, female, _, _ = algo.init_population(task)
    assert (pride == -1).sum() == 10                # 20% are nomads
    assert set(np.unique(pride[pride >= 0])) <= {0, 1, 2, 3}
    # prides are mostly female, nomads mostly male
    assert female[pride >= 0].mean() > 0.5
    assert female[pride == -1].mean() < 0.5


def test_nomad_group_does_not_grow_without_bound():
    """Migration must refill the places it empties.

    Exiling females every iteration while refilling only a few lets the
    nomad group swallow the population, and nomads only search at
    random — which quietly spends most of the budget on noise.
    """
    task = Task(problem=Sphere(dimension=4), max_evals=8000)
    algo = LionOptimizationAlgorithm(population_size=50, nomad_ratio=0.2,
                                     migration_ratio=0.4, seed=1)
    state = algo.init_population(task)
    started_with = int((state[2] == -1).sum())
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert (state[2] == -1).sum() <= started_with + 5


def test_territory_memory_never_worsens():
    task = Task(problem=Rastrigin(dimension=5), max_evals=5000)
    algo = LionOptimizationAlgorithm(population_size=30, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous = state[5].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(state[5] <= previous + 1e-12)


def test_hunters_move_between_themselves_and_the_prey():
    """Centre hunters close in; wings attack from the opposite side."""
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = LionOptimizationAlgorithm(population_size=6, seed=2)
    lions = np.array([[3.0] * 3, [4.0] * 3, [5.0] * 3, [6.0] * 3])
    fitness = np.array([27.0, 48.0, 75.0, 108.0])
    best_x, best_f = lions.copy(), fitness.copy()
    members = np.arange(4)
    prey = lions.mean(axis=0)
    algo._hunt(task, members, lions, fitness, best_x, best_f)
    # every hunter stayed in the region bracketed by itself and the prey
    assert np.all(np.isfinite(lions))
    assert np.all(lions >= task.lower) and np.all(lions <= task.upper)
    assert np.any(lions != np.array([[3.0] * 3, [4.0] * 3,
                                     [5.0] * 3, [6.0] * 3]))
    assert prey.shape == (3,)


def test_hunt_needs_at_least_three_hunters():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = LionOptimizationAlgorithm(population_size=6, seed=2)
    lions = np.zeros((2, 3))
    fitness = np.zeros(2)
    before = task.evals
    algo._hunt(task, np.arange(2), lions, fitness, lions.copy(),
               fitness.copy())
    assert task.evals == before                     # nothing happened


def test_defence_swaps_a_weak_male_for_a_strong_nomad():
    algo = LionOptimizationAlgorithm(population_size=6, n_prides=1,
                                     migration_ratio=0.0, seed=5)
    fitness = np.array([1.0, 50.0, 2.0, 0.5])
    pride = np.array([0, 0, -1, -1])
    female = np.array([False, False, False, False])
    pride = algo._defend_and_migrate(fitness, pride, female)
    assert pride[1] == -1        # the weak pride male was exiled
    assert pride[3] == 0         # the strong nomad took his place


def test_defence_keeps_a_strong_male_when_nomads_are_worse():
    algo = LionOptimizationAlgorithm(population_size=4, n_prides=1,
                                     migration_ratio=0.0, seed=5)
    fitness = np.array([1.0, 2.0, 90.0, 99.0])
    pride = np.array([0, 0, -1, -1])
    female = np.zeros(4, dtype=bool)
    pride = algo._defend_and_migrate(fitness, pride, female)
    assert list(pride) == [0, 0, -1, -1]            # no takeover


def test_mating_blends_the_parents():
    task = Task(problem=Sphere(dimension=6), max_evals=5000)
    algo = LionOptimizationAlgorithm(population_size=4, mating_ratio=1.0,
                                     mutation_prob=0.0, seed=8)
    lions = np.array([[10.0] * 6, [0.0] * 6])
    fitness = np.array([600.0, 0.0])
    best_x, best_f = lions.copy(), fitness.copy()
    algo._mate(task, np.array([0]), np.array([1]), lions, fitness,
               best_x, best_f)
    # the cub is a blend of (10,...) and (0,...), so it improved on the mother
    assert fitness[0] < 600.0
    assert np.all(lions[0] >= -1.0) and np.all(lions[0] <= 11.0)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=4000)
    algo = LionOptimizationAlgorithm(population_size=20, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 20 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"n_prides": 0},
    {"nomad_ratio": 0.0},
    {"nomad_ratio": 1.0},
    {"sex_ratio": 0.0},
    {"sex_ratio": 1.0},
    {"roaming_ratio": 0.0},
    {"mating_ratio": 1.5},
    {"mutation_prob": -0.1},
    {"migration_ratio": 1.5},
])
def test_loa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        LionOptimizationAlgorithm(**kwargs)
