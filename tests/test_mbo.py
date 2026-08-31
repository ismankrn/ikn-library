import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import MonarchButterflyOptimization
from ikn_library.problems import Rastrigin, Sphere


def test_mbo_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = MonarchButterflyOptimization(seed=42).run(task)
    assert best_fitness < 1e-4
    assert best_x.shape == (5,)


def test_mbo_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = MonarchButterflyOptimization(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_mbo_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = MonarchButterflyOptimization(max_step=50.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_mbo_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(MonarchButterflyOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_mbo_respects_eval_budget():
    for budget in (37, 150, 400):
        task = Task(problem=Sphere(dimension=3), max_evals=budget)
        MonarchButterflyOptimization(seed=0).run(task)
        assert task.evals <= budget


def test_population_stays_sorted_by_fitness():
    task = Task(problem=Rastrigin(dimension=4), max_evals=3000)
    algo = MonarchButterflyOptimization(population_size=12, seed=5)
    state = algo.init_population(task)
    assert np.all(np.diff(state[1]) >= 0)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(np.diff(state[1]) >= 0)


def test_the_two_lands_partition_the_population():
    algo = MonarchButterflyOptimization(population_size=20, partition=0.5)
    n1 = algo._land_sizes()
    assert n1 == 10
    # both lands always keep at least one butterfly
    for partition in (0.01, 0.99):
        algo = MonarchButterflyOptimization(population_size=20,
                                            partition=partition)
        n1 = algo._land_sizes()
        assert 1 <= n1 <= 19


def test_migration_copies_existing_coordinate_values():
    """Migration recombines; it never invents a new value."""
    algo = MonarchButterflyOptimization(population_size=6, seed=3)
    # each butterfly is a constant row, so a value reveals its donor
    butterflies = np.repeat(np.arange(6.0).reshape(6, 1), 8, axis=1)
    for _ in range(10):
        child = algo._migrate(butterflies, np.arange(3), np.arange(3, 6), 8)
        assert np.all(np.isin(child, np.arange(6.0)))


def test_migration_draws_each_coordinate_independently():
    """One offspring can mix donors, unlike two-parent crossover."""
    algo = MonarchButterflyOptimization(population_size=6, seed=4)
    butterflies = np.repeat(np.arange(6.0).reshape(6, 1), 20, axis=1)
    mixed = False
    for _ in range(20):
        child = algo._migrate(butterflies, np.arange(3), np.arange(3, 6), 20)
        if len(np.unique(child)) > 1:
            mixed = True
            break
    assert mixed


def test_adjusting_can_take_coordinates_from_the_best():
    algo = MonarchButterflyOptimization(population_size=6, partition=0.999,
                                        seed=2)
    butterflies = np.full((6, 10), 5.0)
    best = np.zeros(10)
    # a high partition sends nearly every coordinate to the best
    child = algo._adjust(butterflies, best, np.arange(3, 6), 10, alpha=1.0)
    assert np.count_nonzero(child == 0.0) >= 8


def test_levy_step_only_touches_peer_coordinates():
    """With bar=1 no walk is ever added, so the result is a pure copy."""
    algo = MonarchButterflyOptimization(population_size=6, partition=0.0001,
                                        bar=1.0, seed=6)
    butterflies = np.repeat(np.arange(6.0).reshape(6, 1), 10, axis=1)
    child = algo._adjust(butterflies, np.zeros(10), np.arange(3, 6), 10,
                         alpha=99.0)
    assert np.all(np.isin(child, np.arange(6.0)))


def test_elitism_preserves_the_best_butterflies():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = MonarchButterflyOptimization(population_size=12, n_elite=3, seed=8)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous_best = state[1][0]
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert state[1][0] <= previous_best + 1e-12


def test_budget_tied_step_shrinks_while_the_published_one_does_not_track_budget():
    """The published alpha = max_step / t**2 ignores how much is left."""
    algo = MonarchButterflyOptimization(max_step=1.0, budget_tied_step=True)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    assert algo._progress(task) == 0.0
    for _ in range(900):
        task.eval(np.zeros(3))
    # tied to the budget: 90% spent leaves 1% of the step
    assert (max(1.0 - algo._progress(task), 1e-6) ** 2) == pytest.approx(0.01)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = MonarchButterflyOptimization(population_size=11, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 11


@pytest.mark.parametrize("kwargs", [
    {"partition": 0.0},
    {"partition": 1.0},
    {"period": 0.0},
    {"bar": 1.5},
    {"max_step": 0.0},
    {"n_elite": -1},
    {"n_elite": 20},
])
def test_mbo_invalid_params(kwargs):
    with pytest.raises(ValueError):
        MonarchButterflyOptimization(population_size=20, **kwargs)
