import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import FishSchoolSearch
from ikn_library.problems import Rastrigin, Sphere


def test_fss_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = FishSchoolSearch(seed=42).run(task)
    assert best_fitness < 1e-2
    assert best_x.shape == (5,)


def test_fss_handles_rastrigin_well():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = FishSchoolSearch(seed=1).run(task)
    assert best_fitness < 5.0


def test_fss_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    best_x, _ = FishSchoolSearch(step_individual=0.9, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_fss_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(FishSchoolSearch(population_size=20, seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_fss_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    FishSchoolSearch(population_size=20, seed=0).run(task)
    assert task.evals <= 500


def test_weights_stay_within_their_limits():
    task = Task(problem=Rastrigin(dimension=4), max_evals=6000)
    algo = FishSchoolSearch(population_size=20, weight_scale=50.0, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    assert np.allclose(state[2], 25.0)          # fish start at half scale
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert (state[2] >= 1.0).all() and (state[2] <= 50.0).all()


def test_feeding_increases_the_weight_of_successful_fish():
    task = Task(problem=Sphere(dimension=4), max_evals=4000)
    algo = FishSchoolSearch(population_size=20, seed=3)
    state = algo.init_population(task)
    task.next_iter()
    initial = state[2].copy()
    state = algo.run_iteration(task, state)
    # early on, some fish always find food, so the school gains weight
    assert state[2].sum() > initial.sum()


def test_individual_step_decays_over_the_run():
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    algo = FishSchoolSearch(population_size=10, step_individual=0.5,
                            step_individual_final=0.001, seed=2)
    start = algo._current_step(task)
    while not task.stopping_condition():         # burn the budget
        task.eval(np.zeros(3))
    end = algo._current_step(task)
    assert start == pytest.approx(0.5)
    assert end == pytest.approx(0.001, abs=1e-6)
    assert end < start


def test_fitness_matches_the_school_positions():
    """Steps 3 and 4 move every fish, so fitness must be refreshed."""
    task = Task(problem=Rastrigin(dimension=4), max_evals=5000)
    algo = FishSchoolSearch(population_size=15, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(5):
        state = algo.run_iteration(task, state)
        recomputed = np.array([task.problem.evaluate(x) for x in state[0]])
        np.testing.assert_allclose(state[1], recomputed)


@pytest.mark.parametrize("kwargs", [
    {"step_individual": 0.0},
    {"step_individual_final": 0.0},
    {"step_individual": 0.01, "step_individual_final": 0.5},
    {"step_volitive_factor": 0.0},
    {"weight_scale": 1.0},
])
def test_fss_invalid_params(kwargs):
    with pytest.raises(ValueError):
        FishSchoolSearch(**kwargs)
