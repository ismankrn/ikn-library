import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import BatAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_bat_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = BatAlgorithm(population_size=30, seed=42).run(task)
    assert best_fitness < 1e-2
    assert best_x.shape == (5,)


def test_bat_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = BatAlgorithm(population_size=30, seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_bat_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = BatAlgorithm(population_size=10, local_scale=1.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_bat_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(BatAlgorithm(population_size=15, seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_bat_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    BatAlgorithm(population_size=10, seed=0).run(task)
    assert task.evals <= 500


def test_bat_loudness_decays_on_acceptance():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = BatAlgorithm(population_size=10, alpha=0.9, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(20):
        state = algo.run_iteration(task, state)
        task.next_iter()
    loudness = state[3]
    assert loudness.min() < algo.loudness  # some bat accepted an improvement
    assert (loudness <= algo.loudness).all()


@pytest.mark.parametrize("kwargs", [
    {"loudness": 0.0},
    {"pulse_rate": 1.5},
    {"alpha": 1.0},
    {"gamma": 0.0},
    {"min_frequency": 2.0, "max_frequency": 1.0},
    {"local_scale": 0.0},
])
def test_bat_invalid_params(kwargs):
    with pytest.raises(ValueError):
        BatAlgorithm(**kwargs)
