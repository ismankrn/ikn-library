import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import GravitationalSearchAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_gsa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = GravitationalSearchAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-10
    assert best_x.shape == (5,)


def test_gsa_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = GravitationalSearchAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_gsa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = GravitationalSearchAlgorithm(g0=1e5, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_gsa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(GravitationalSearchAlgorithm(
            population_size=20, seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_gsa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    GravitationalSearchAlgorithm(population_size=20, seed=0).run(task)
    assert task.evals <= 400


def test_better_agents_are_heavier():
    algo = GravitationalSearchAlgorithm(population_size=4)
    masses = algo._masses(np.array([1.0, 2.0, 3.0, 10.0]))
    assert masses[0] > masses[1] > masses[2] > masses[3]
    assert masses[-1] == pytest.approx(0.0)      # the worst agent has no mass
    assert masses.sum() == pytest.approx(1.0)


def test_masses_handle_a_uniform_population():
    """When every agent is equally good, mass is shared evenly."""
    algo = GravitationalSearchAlgorithm(population_size=5)
    masses = algo._masses(np.full(5, 3.0))
    np.testing.assert_allclose(masses, 0.2)


def test_gravitational_constant_decays():
    algo = GravitationalSearchAlgorithm(g0=100.0, alpha=30.0)
    start = algo._gravitational_constant(0.0)
    middle = algo._gravitational_constant(0.5)
    end = algo._gravitational_constant(1.0)
    assert start == pytest.approx(100.0)
    assert start > middle > end
    assert end < start * 1e-10                   # alpha=30 decays sharply


def test_kbest_shrinks_to_its_final_value():
    algo = GravitationalSearchAlgorithm(population_size=50, final_kbest=1)
    assert algo._kbest(0.0) == 50
    assert algo._kbest(1.0) == 1
    assert algo._kbest(0.0) > algo._kbest(0.5) > algo._kbest(1.0)


def test_velocity_stays_within_its_limit():
    task = Task(problem=Rastrigin(dimension=4), max_evals=4000)
    algo = GravitationalSearchAlgorithm(population_size=20, max_velocity=0.1,
                                        seed=6)
    state = algo.init_population(task)
    task.next_iter()
    limit = 0.1 * (task.upper - task.lower)
    for _ in range(8):
        state = algo.run_iteration(task, state)
        assert np.all(np.abs(state[2]) <= limit + 1e-9)


def test_fitness_matches_the_agent_positions():
    task = Task(problem=Sphere(dimension=4), max_evals=4000)
    algo = GravitationalSearchAlgorithm(population_size=15, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(6):
        state = algo.run_iteration(task, state)
        recomputed = np.array([task.problem.evaluate(x) for x in state[0]])
        np.testing.assert_allclose(state[1], recomputed)


@pytest.mark.parametrize("kwargs", [
    {"g0": 0.0},
    {"alpha": 0.0},
    {"final_kbest": 0},
    {"final_kbest": 100, "population_size": 50},
    {"max_velocity": 0.0},
])
def test_gsa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        GravitationalSearchAlgorithm(**kwargs)
