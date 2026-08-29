import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import AntColonyOptimization
from ikn_library.problems import Rastrigin, Sphere


def test_aco_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
    best_x, best_fitness = algo.run(task)
    assert best_fitness < 1e-3
    assert best_x.shape == (5,)


def test_aco_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    algo = AntColonyOptimization(population_size=30, archive_size=50, seed=1)
    _, best_fitness = algo.run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_aco_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    algo = AntColonyOptimization(population_size=10, archive_size=20, seed=7)
    best_x, _ = algo.run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_aco_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        algo = AntColonyOptimization(population_size=20, seed=123)
        results.append(algo.run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_aco_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    AntColonyOptimization(population_size=10, seed=0).run(task)
    assert task.evals <= 500


def test_invalid_archive_size():
    with pytest.raises(ValueError):
        AntColonyOptimization(archive_size=1)
