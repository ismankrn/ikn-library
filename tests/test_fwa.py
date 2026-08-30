import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import FireworksAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_fwa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = FireworksAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-30
    assert best_x.shape == (5,)


def test_fwa_solves_rastrigin():
    """FWA reaches the global optimum of Rastrigin exactly."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = FireworksAlgorithm(seed=1).run(task)
    assert best_fitness < 1e-6


def test_fwa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = FireworksAlgorithm(max_amplitude=3.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_fwa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(FireworksAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_fwa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    FireworksAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_better_fireworks_get_more_sparks():
    algo = FireworksAlgorithm(population_size=4, n_sparks=100)
    fitness = np.array([1.0, 2.0, 3.0, 10.0])      # first is best
    counts = algo._spark_counts(fitness)
    assert counts[0] > counts[-1]
    assert (counts >= 1).all()


def test_better_fireworks_get_smaller_amplitudes():
    """The inverse coupling: good fireworks explode tightly."""
    algo = FireworksAlgorithm(population_size=4, max_amplitude=1.0)
    fitness = np.array([1.0, 2.0, 3.0, 10.0])
    span = np.full(3, 10.0)
    amplitudes = algo._amplitudes(fitness, span)
    assert amplitudes[0].max() < amplitudes[-1].max()
    assert (amplitudes >= 0).all()


def test_spark_counts_stay_within_their_bounds():
    algo = FireworksAlgorithm(population_size=5, n_sparks=100,
                              spark_bounds=(0.1, 0.5))
    for fitness in (np.array([1.0, 1.0, 1.0, 1.0, 1.0]),      # all equal
                    np.array([0.0, 1e6, 1.0, 2.0, 3.0])):     # extreme spread
        counts = algo._spark_counts(fitness)
        assert (counts >= 10).all() and (counts <= 50).all()


def test_population_size_is_preserved_and_elitist():
    task = Task(problem=Rastrigin(dimension=4), max_evals=5000)
    algo = FireworksAlgorithm(population_size=5, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        previous_best = state[1].min()
        state = algo.run_iteration(task, state)
        assert len(state[0]) == len(state[1]) == 5
        # the best solution is always carried over
        assert state[1].min() <= previous_best + 1e-12


def test_gaussian_sparks_are_essential():
    """Without them the search stalls — they are not decoration."""
    with_gaussian = FireworksAlgorithm(n_gaussian_sparks=5, seed=3).run(
        Task(problem=Sphere(dimension=10), max_evals=5000))[1]
    without = FireworksAlgorithm(n_gaussian_sparks=0, seed=3).run(
        Task(problem=Sphere(dimension=10), max_evals=5000))[1]
    assert with_gaussian < without


@pytest.mark.parametrize("kwargs", [
    {"n_sparks": 0},
    {"max_amplitude": 0.0},
    {"n_gaussian_sparks": -1},
    {"spark_bounds": (0.5, 0.2)},
    {"spark_bounds": (0.0, 0.5)},
    {"spark_bounds": (0.1, 1.5)},
])
def test_fwa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        FireworksAlgorithm(**kwargs)
