import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import CamelAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_camel_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = CamelAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-6
    assert best_x.shape == (5,)


def test_camel_handles_rastrigin_well():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = CamelAlgorithm(seed=1).run(task)
    assert best_fitness < 5.0        # strong on multimodal landscapes


def test_camel_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = CamelAlgorithm(visibility=5.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_camel_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(CamelAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_camel_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    CamelAlgorithm(seed=0).run(task)
    assert task.evals <= 500


def test_state_variables_stay_in_range():
    task = Task(problem=Rastrigin(dimension=4), max_evals=5000)
    algo = CamelAlgorithm(population_size=10, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        _, _, endurance, supply = state
        assert (endurance >= 0).all() and (endurance <= 2.0).all()
        assert (supply > 0).all() and (supply <= 1.0).all()


def test_finding_an_oasis_replenishes_the_camel():
    """A camel that improves resets its endurance and supply to 1."""
    task = Task(problem=Sphere(dimension=3), max_evals=4000)
    algo = CamelAlgorithm(population_size=8, seed=3)
    state = algo.init_population(task)
    task.next_iter()
    replenished = 0
    for _ in range(15):
        before = state[1].copy()
        state = algo.run_iteration(task, state)
        improved = state[1] < before
        # every improved camel must show full endurance and supply
        if improved.any():
            replenished += 1
            assert np.allclose(state[2][improved], 1.0)
            assert np.allclose(state[3][improved], 1.0)
    assert replenished > 0


def test_supply_decreases_as_the_journey_progresses():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = CamelAlgorithm(population_size=5, burden_rate=0.9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    assert np.allclose(state[3], 1.0)          # full supply at the start
    for _ in range(20):
        state = algo.run_iteration(task, state)
        task.next_iter()
    # some camel has consumed supply without finding an oasis
    assert (state[3] < 1.0).any()


@pytest.mark.parametrize("kwargs", [
    {"min_temperature": 2.0, "max_temperature": 1.0},
    {"min_temperature": -2.0, "max_temperature": -1.0},
    {"burden_rate": 1.5},
    {"burden_rate": -0.1},
    {"death_rate": 1.0},
    {"death_rate": -0.1},
    {"visibility": 0.0},
])
def test_camel_invalid_params(kwargs):
    with pytest.raises(ValueError):
        CamelAlgorithm(**kwargs)
