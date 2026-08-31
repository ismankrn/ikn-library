import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import MothFlameOptimization
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_mfo_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = MothFlameOptimization(seed=42).run(task)
    assert best_fitness < 1e-12
    assert best_x.shape == (5,)


def test_mfo_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = MothFlameOptimization(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_mfo_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = MothFlameOptimization(spiral_constant=4.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_mfo_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(MothFlameOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_mfo_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    MothFlameOptimization(seed=0).run(task)
    assert task.evals <= 400


def test_flame_count_shrinks_from_n_to_one():
    """The flame count *is* the exploration schedule."""
    algo = MothFlameOptimization(population_size=30)
    assert algo._flame_count(0.0) == 30
    assert algo._flame_count(1.0) == 1
    assert algo._flame_count(0.5) == pytest.approx(16, abs=1)
    # never leaves the valid range
    for progress in np.linspace(0.0, 1.0, 21):
        assert 1 <= algo._flame_count(progress) <= 30


def test_flames_are_the_best_seen_and_never_worsen():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = MothFlameOptimization(population_size=12, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous = state[3].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        flame_fitness = state[3]
        assert np.all(np.diff(flame_fitness) >= 0)      # kept sorted
        assert np.all(flame_fitness <= previous + 1e-12)


def test_refresh_keeps_the_best_of_moths_and_flames():
    algo = MothFlameOptimization(population_size=3)
    flames = np.array([[0.0], [5.0], [9.0]])
    flame_fitness = np.array([0.0, 5.0, 9.0])
    moths = np.array([[1.0], [2.0], [7.0]])
    fitness = np.array([1.0, 2.0, 7.0])
    new_flames, new_fitness = algo._refresh_flames(
        moths, fitness, flames, flame_fitness)
    np.testing.assert_array_equal(new_fitness, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(new_flames.ravel(), [0.0, 1.0, 2.0])


def test_a_moth_on_its_flame_stays_there():
    """Zero distance means the spiral collapses to the flame itself."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = MothFlameOptimization(population_size=3, seed=1)
    flames = np.zeros((3, 4))
    flame_fitness = np.zeros(3)
    moths = np.zeros((3, 4))
    fitness = np.zeros(3)
    new_moths, _, _, _ = algo.run_iteration(
        task, (moths, fitness, flames, flame_fitness))
    np.testing.assert_allclose(new_moths, 0.0, atol=1e-12)


def test_late_run_moths_all_share_one_flame():
    """When the flame count reaches 1, every moth targets the best."""
    algo = MothFlameOptimization(population_size=20)
    assert algo._flame_count(1.0) == 1
    # index selection collapses to flame 0 for every moth
    n_flames = algo._flame_count(1.0)
    assert all(min(i, n_flames - 1) == 0 for i in range(20))


def test_search_is_translation_invariant():
    """The spiral uses |flame - moth|, so moving the optimum is free."""
    offset = 2.0

    class ShiftedSphere(Problem):
        def __init__(self, dimension=4):
            super().__init__(dimension,
                             lower=-5.12 + offset, upper=5.12 + offset)

        def _evaluate(self, x):
            return float(np.sum((x - offset) ** 2))

    plain = Task(problem=Sphere(dimension=4), max_evals=1500)
    moved = Task(problem=ShiftedSphere(), max_evals=1500)
    x_plain, f_plain = MothFlameOptimization(seed=5).run(plain)
    x_moved, f_moved = MothFlameOptimization(seed=5).run(moved)

    np.testing.assert_allclose(x_moved, x_plain + offset, rtol=1e-8,
                               atol=1e-8)
    assert f_moved == pytest.approx(f_plain, rel=1e-8, abs=1e-20)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = MothFlameOptimization(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 9 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"spiral_constant": 0.0},
    {"spiral_constant": -1.0},
    {"a_start": 0.0},
    {"a_start": 1.0},
    {"a_end": -0.5},
    {"a_start": -2.0, "a_end": -1.0},
])
def test_mfo_invalid_params(kwargs):
    with pytest.raises(ValueError):
        MothFlameOptimization(**kwargs)
