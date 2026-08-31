import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import WhaleOptimizationAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_woa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = WhaleOptimizationAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-30
    assert best_x.shape == (5,)


def test_woa_solves_rastrigin():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = WhaleOptimizationAlgorithm(seed=1).run(task)
    assert best_fitness < 1e-8


def test_woa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = WhaleOptimizationAlgorithm(spiral_constant=5.0,
                                           seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_woa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(WhaleOptimizationAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_woa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    WhaleOptimizationAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_control_coefficient_falls_to_zero():
    algo = WhaleOptimizationAlgorithm(a_start=2.0)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    assert algo.a_start * (1.0 - algo._progress(task)) == pytest.approx(2.0)
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo.a_start * (1.0 - algo._progress(task)) == pytest.approx(1.0)
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo.a_start * (1.0 - algo._progress(task)) == pytest.approx(0.0)


def test_exploration_switches_itself_off_once_a_drops_below_one():
    """|A| = |2a*r - a| <= a, so a < 1 makes the search branch impossible."""
    algo = WhaleOptimizationAlgorithm(a_start=2.0, seed=3)
    for a in (0.9, 0.5, 0.1):
        magnitudes = [abs(2.0 * a * algo.rng.random() - a)
                      for _ in range(500)]
        assert max(magnitudes) < 1.0


def test_a_whale_on_the_best_position_stays_when_spiralling():
    """The spiral collapses to the target when the distance is zero."""
    algo = WhaleOptimizationAlgorithm(population_size=4, seed=1)
    best = np.full(5, 2.0)
    moved = algo._spiral(best, best.copy(), 5)
    np.testing.assert_allclose(moved, best, atol=1e-12)


def test_spiral_uses_a_plain_difference():
    """|X* - X| carries no C factor, unlike the encircling move.

    That makes the spiral half of the algorithm translation-equivariant;
    the algorithm page measures what the other half costs.
    """
    best = np.zeros(6)
    whale = np.full(6, 1.0)
    offset = 7.0
    a = WhaleOptimizationAlgorithm(population_size=4, seed=9)._spiral(
        best, whale, 6)
    b = WhaleOptimizationAlgorithm(population_size=4, seed=9)._spiral(
        best + offset, whale + offset, 6)
    np.testing.assert_allclose(b, a + offset, rtol=1e-9, atol=1e-9)


def test_encircling_is_not_translation_equivariant():
    """The C factor scales the target's absolute coordinates."""
    whale = np.full(6, 1.0)
    target = np.zeros(6)
    offset = 7.0
    a = WhaleOptimizationAlgorithm(seed=4)._encircle(target, whale, 1.0, 6)
    b = WhaleOptimizationAlgorithm(seed=4)._encircle(
        target + offset, whale + offset, 1.0, 6)
    assert not np.allclose(b, a + offset)


def test_the_swarm_contracts_over_a_run():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = WhaleOptimizationAlgorithm(population_size=20, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    spread_before = np.mean(np.std(state[0], axis=0))
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
    assert np.mean(np.std(state[0], axis=0)) < spread_before


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = WhaleOptimizationAlgorithm(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 9


@pytest.mark.parametrize("kwargs", [
    {"a_start": 0.0},
    {"a_start": -1.0},
    {"spiral_constant": 0.0},
    {"spiral_constant": -1.0},
])
def test_woa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        WhaleOptimizationAlgorithm(**kwargs)
