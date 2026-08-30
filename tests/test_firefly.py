import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import FireflyAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_firefly_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = FireflyAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-20
    assert best_x.shape == (5,)


def test_firefly_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = FireflyAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_firefly_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = FireflyAlgorithm(alpha=3.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_firefly_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(FireflyAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_firefly_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    FireflyAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_brightest_firefly_is_not_attracted():
    """Index 0 after sorting has nobody brighter, so it does not move."""
    algo = FireflyAlgorithm(seed=0)
    fireflies = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    brightness = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        algo._attract(fireflies, brightness, 0, gamma=1.0), fireflies[0])


def test_attraction_moves_toward_brighter_fireflies():
    algo = FireflyAlgorithm(seed=0)
    fireflies = np.array([[0.0, 0.0], [10.0, 10.0]])
    brightness = np.array([0.0, 1.0])
    moved = algo._attract(fireflies, brightness, 1, gamma=1e-6)  # weak decay
    # the dim firefly must end up closer to the bright one at the origin
    assert np.linalg.norm(moved) < np.linalg.norm(fireflies[1])


def test_attraction_fades_with_distance():
    """Large gamma means light is absorbed: distant fireflies barely pull."""
    algo = FireflyAlgorithm(seed=0)
    brightness = np.array([0.0, 1.0])
    near = np.array([[0.0, 0.0], [0.5, 0.5]])
    far = np.array([[0.0, 0.0], [50.0, 50.0]])
    near_pull = np.linalg.norm(near[1] - algo._attract(near, brightness, 1, 1.0))
    far_pull = np.linalg.norm(far[1] - algo._attract(far, brightness, 1, 1.0))
    assert near_pull > far_pull


def test_randomization_decays_over_iterations():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = FireflyAlgorithm(alpha=1.0, alpha_decay=0.9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    alphas = [state[2]]
    for _ in range(5):
        state = algo.run_iteration(task, state)
        alphas.append(state[2])
    assert alphas == sorted(alphas, reverse=True)      # strictly decreasing
    assert alphas[-1] < alphas[0] * 0.6


def test_brightness_matches_the_firefly_positions():
    """The recorded brightness must always be the true objective value."""
    task = Task(problem=Rastrigin(dimension=4), max_evals=3000)
    algo = FireflyAlgorithm(population_size=15, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(8):
        state = algo.run_iteration(task, state)
        fireflies, brightness = state[0], state[1]
        assert len(fireflies) == len(brightness) == 15
        recomputed = np.array([task.problem.evaluate(x) for x in fireflies])
        np.testing.assert_allclose(brightness, recomputed)


def test_a_firefly_only_moves_when_it_improves():
    """Greedy acceptance: a worse candidate is discarded."""
    task = Task(problem=Sphere(dimension=4), max_evals=4000)
    algo = FireflyAlgorithm(population_size=12, seed=8)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(8):
        previous = np.sort(state[1])
        state = algo.run_iteration(task, state)
        # sorted fitness can only improve position by position
        assert (np.sort(state[1]) <= previous + 1e-12).all()


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0.0},
    {"alpha_decay": 0.0},
    {"alpha_decay": 1.5},
    {"beta0": 0.0},
    {"gamma": 0.0},
])
def test_firefly_invalid_params(kwargs):
    with pytest.raises(ValueError):
        FireflyAlgorithm(**kwargs)
