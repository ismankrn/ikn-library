import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import FlowerPollinationAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_fpa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = FlowerPollinationAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-10
    assert best_x.shape == (5,)


def test_fpa_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = FlowerPollinationAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_fpa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = FlowerPollinationAlgorithm(gamma=5.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_fpa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(FlowerPollinationAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_fpa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    FlowerPollinationAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_selection_is_greedy_so_no_flower_ever_worsens():
    task = Task(problem=Rastrigin(dimension=5), max_evals=5000)
    algo = FlowerPollinationAlgorithm(population_size=10, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        previous = state[1].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(state[1] <= previous + 1e-12)


def test_levy_flight_has_heavy_tails():
    """Mostly small steps, rare very large ones."""
    algo = FlowerPollinationAlgorithm(levy_exponent=1.5, seed=11)
    steps = np.abs(algo._levy_flight(20000))
    # a heavy-tailed sample has a maximum far above its median
    assert steps.max() > 100 * np.median(steps)


def test_heavier_tails_give_longer_jumps():
    heavy = FlowerPollinationAlgorithm(levy_exponent=1.1, seed=3)
    light = FlowerPollinationAlgorithm(levy_exponent=1.9, seed=3)
    assert (np.abs(heavy._levy_flight(20000)).max()
            > np.abs(light._levy_flight(20000)).max())


def test_local_pollination_cannot_move_an_identical_population():
    """The difference vector x_j - x_k vanishes when flowers coincide."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = FlowerPollinationAlgorithm(population_size=5,
                                      switch_probability=0.0, seed=2)
    flowers = np.full((5, 4), 2.0)
    fitness = np.full(5, task.problem.evaluate(flowers[0]))
    new_flowers, _ = algo.run_iteration(task, (flowers, fitness))
    np.testing.assert_array_equal(new_flowers, flowers)


def test_global_pollination_stalls_on_a_converged_population():
    """The (g* - x) factor is zero once every flower sits on the best.

    This is a real structural property of the published formulation,
    not an implementation bug; the algorithm page discusses it.
    """
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = FlowerPollinationAlgorithm(population_size=5,
                                      switch_probability=1.0, seed=2)
    flowers = np.full((5, 4), 2.0)
    fitness = np.full(5, task.problem.evaluate(flowers[0]))
    new_flowers, _ = algo.run_iteration(task, (flowers, fitness))
    np.testing.assert_array_equal(new_flowers, flowers)


def test_global_pollination_moves_flowers_toward_the_best():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = FlowerPollinationAlgorithm(population_size=6,
                                      switch_probability=1.0, seed=8)
    state = algo.init_population(task)
    best = state[0][np.argmin(state[1])].copy()
    spread_before = np.mean(np.linalg.norm(state[0] - best, axis=1))
    task.next_iter()
    for _ in range(20):
        state = algo.run_iteration(task, state)
        task.next_iter()
    best_after = state[0][np.argmin(state[1])]
    spread_after = np.mean(np.linalg.norm(state[0] - best_after, axis=1))
    assert spread_after < spread_before


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=4000)
    algo = FlowerPollinationAlgorithm(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 9


@pytest.mark.parametrize("kwargs", [
    {"switch_probability": -0.1},
    {"switch_probability": 1.5},
    {"gamma": 0.0},
    {"gamma": -1.0},
    {"levy_exponent": 0.0},
    {"levy_exponent": 2.5},
])
def test_fpa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        FlowerPollinationAlgorithm(**kwargs)
