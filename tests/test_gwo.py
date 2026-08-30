import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import GreyWolfOptimizer
from ikn_library.problems import Rastrigin, Sphere


def test_gwo_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = GreyWolfOptimizer(seed=42).run(task)
    assert best_fitness < 1e-20
    assert best_x.shape == (5,)


def test_gwo_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = GreyWolfOptimizer(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_gwo_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = GreyWolfOptimizer(a_start=5.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_gwo_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(GreyWolfOptimizer(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_gwo_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    GreyWolfOptimizer(seed=0).run(task)
    assert task.evals <= 400


def test_control_coefficient_falls_linearly():
    """`a` drives the explore/attack switch and is tied to the budget."""
    algo = GreyWolfOptimizer(a_start=2.0, a_end=0.0)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    assert algo._progress(task) == 0.0
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo._progress(task) == pytest.approx(0.5)


def _late_run_task(dimension=2):
    """A task at 99.9% progress with budget to spare, so `a` is ~0."""
    task = Task(problem=Sphere(dimension=dimension), max_iters=1000)
    for _ in range(999):
        task.next_iter()
    return task


def test_pack_follows_three_leaders_not_one():
    """Late in the run, wolves land on the mean of alpha/beta/delta."""
    task = _late_run_task()
    algo = GreyWolfOptimizer(population_size=6, seed=3)
    wolves = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
                       [8.0, 8.0], [9.0, 9.0], [7.0, 7.0]])
    fitness = np.array([1.0, 2.0, 3.0, 8.0, 9.0, 7.0])
    new_wolves, _ = algo.run_iteration(task, (wolves.copy(), fitness.copy()))
    # a ~ 0 means A ~ 0, so every wolf lands on the leaders' mean
    expected = wolves[:3].mean(axis=0)
    for wolf in new_wolves:
        np.testing.assert_allclose(wolf, expected, atol=0.05)


def test_leaders_are_the_three_best_wolves():
    task = _late_run_task()
    algo = GreyWolfOptimizer(population_size=5, seed=1)
    wolves = np.array([[5.0, 5.0], [0.0, 0.0], [4.0, 4.0],
                       [1.0, 1.0], [2.0, 2.0]])
    fitness = np.array([50.0, 0.0, 32.0, 2.0, 8.0])
    new_wolves, _ = algo.run_iteration(task, (wolves.copy(), fitness.copy()))
    # the three best are indices 1, 3, 4 -> mean of (0,0), (1,1), (2,2)
    np.testing.assert_allclose(new_wolves[0], [1.0, 1.0], atol=0.05)


def test_early_iterations_explore_more_than_late_ones():
    """|A| > 1 pushes wolves away from leaders; |A| < 1 pulls them in."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = GreyWolfOptimizer(population_size=20, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    spreads = []
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        spreads.append(np.mean(np.std(state[0], axis=0)))
    assert spreads[0] > spreads[-1]          # the pack closes in over time


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = GreyWolfOptimizer(population_size=11, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 11


@pytest.mark.parametrize("kwargs", [
    {"a_start": 0.0},
    {"a_start": -1.0},
    {"a_end": -0.1},
    {"a_start": 1.0, "a_end": 1.0},
    {"a_start": 1.0, "a_end": 2.0},
])
def test_gwo_invalid_params(kwargs):
    with pytest.raises(ValueError):
        GreyWolfOptimizer(**kwargs)
