import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import SimulatedAnnealing
from ikn_library.problems import Rastrigin, Sphere


def test_sa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    algo = SimulatedAnnealing(seed=42)
    best_x, best_fitness = algo.run(task)
    assert best_fitness < 1e-6
    assert best_x.shape == (5,)


def test_sa_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = SimulatedAnnealing(cooling=0.999, seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_sa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    best_x, _ = SimulatedAnnealing(step_size=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_sa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=2000)
        results.append(SimulatedAnnealing(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_sa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    SimulatedAnnealing(seed=0).run(task)
    assert task.evals <= 500


def test_sa_accepts_uphill_moves_at_high_temperature():
    # With a huge temperature, exp(-delta/T) ~ 1, so nearly every
    # proposal is accepted, including worse ones.
    task = Task(problem=Sphere(dimension=2), max_evals=200)
    algo = SimulatedAnnealing(initial_temperature=1e9, cooling=0.999,
                              step_size=0.5, seed=3)
    state = algo.init_population(task)
    accepted_worse = 0
    for _ in range(150):
        new_state = algo.run_iteration(task, state)
        if new_state[1] > state[1]:
            accepted_worse += 1
        state = new_state
    assert accepted_worse > 0


@pytest.mark.parametrize("kwargs", [
    {"initial_temperature": 0.0},
    {"cooling": 1.0},
    {"cooling": 0.0},
    {"step_size": 0.0},
])
def test_sa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        SimulatedAnnealing(**kwargs)
