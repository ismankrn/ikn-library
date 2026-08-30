import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import GeneticAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_ga_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = GeneticAlgorithm(population_size=50, seed=42).run(task)
    assert best_fitness < 1e-2
    assert best_x.shape == (5,)


def test_ga_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = GeneticAlgorithm(population_size=50, seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_ga_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = GeneticAlgorithm(population_size=20, mutation_scale=1.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_ga_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(GeneticAlgorithm(population_size=20, seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_ga_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    GeneticAlgorithm(population_size=20, seed=0).run(task)
    assert task.evals <= 500


def test_ga_elitism_never_loses_the_best():
    task = Task(problem=Rastrigin(dimension=4), max_evals=3000)
    algo = GeneticAlgorithm(population_size=20, elitism=2, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    best_so_far = state[1].min()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert state[1].min() <= best_so_far + 1e-12
        best_so_far = min(best_so_far, state[1].min())


def test_ga_works_without_elitism_and_crossover():
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    algo = GeneticAlgorithm(population_size=10, elitism=0, crossover_rate=0.0, seed=3)
    _, best_fitness = algo.run(task)
    assert np.isfinite(best_fitness)


@pytest.mark.parametrize("kwargs", [
    {"crossover_rate": 1.5},
    {"mutation_rate": -0.1},
    {"mutation_scale": 0.0},
    {"tournament_size": 0},
    {"blend_alpha": -0.5},
    {"elitism": 50, "population_size": 50},
])
def test_ga_invalid_params(kwargs):
    with pytest.raises(ValueError):
        GeneticAlgorithm(**kwargs)
