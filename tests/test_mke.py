import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import MonkeyKingEvolution
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_mke_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = MonkeyKingEvolution(seed=42).run(task)
    assert best_fitness < 1e-6
    assert best_x.shape == (5,)


def test_mke_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = MonkeyKingEvolution(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_mke_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = MonkeyKingEvolution(fluctuation=5.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_mke_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(MonkeyKingEvolution(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_mke_respects_eval_budget():
    for budget in (25, 137, 600):
        task = Task(problem=Sphere(dimension=3), max_evals=budget)
        MonkeyKingEvolution(seed=0).run(task)
        assert task.evals <= budget


def test_the_king_gets_many_trials_and_everyone_else_one():
    """The asymmetry is the algorithm's whole idea."""
    task = Task(problem=Sphere(dimension=4), max_evals=100000)
    algo = MonkeyKingEvolution(population_size=10, n_clones=6, seed=3)
    state = algo.init_population(task)
    before = task.evals
    task.next_iter()
    algo.run_iteration(task, state)
    # 6 clones for the king, 1 each for the other 9 individuals
    assert task.evals - before == 6 + 9


def test_clone_count_scales_the_kings_share_of_the_budget():
    for n_clones, expected in ((1, 1 + 9), (12, 12 + 9)):
        task = Task(problem=Sphere(dimension=4), max_evals=100000)
        algo = MonkeyKingEvolution(population_size=10, n_clones=n_clones,
                                   seed=3)
        state = algo.init_population(task)
        before = task.evals
        task.next_iter()
        algo.run_iteration(task, state)
        assert task.evals - before == expected


def test_the_king_never_worsens():
    """Clones replace the king only by beating him."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=5000)
    algo = MonkeyKingEvolution(population_size=10, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous_best = state[1].min()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert state[1].min() <= previous_best + 1e-12


def test_no_individual_ever_worsens():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = MonkeyKingEvolution(population_size=8, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous = state[1].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(state[1] <= previous + 1e-12)


def test_change_rate_limits_how_many_coordinates_move():
    """A low change_rate is what makes MKE a coordinate-wise search."""
    task = Task(problem=Sphere(dimension=100), max_evals=50000)
    algo = MonkeyKingEvolution(population_size=6, change_rate=0.1, seed=5)
    population = algo.rng.uniform(-5, 5, (6, 100))
    changed = []
    for _ in range(20):
        moved = algo._difference_move(population, 0, task)
        changed.append(np.count_nonzero(moved != population[0]))
    # around 10% of 100 coordinates, comfortably below half
    assert 2 < np.mean(changed) < 25


def test_at_least_one_coordinate_always_changes():
    task = Task(problem=Sphere(dimension=5), max_evals=50000)
    algo = MonkeyKingEvolution(population_size=6, change_rate=0.001, seed=2)
    population = algo.rng.uniform(-5, 5, (6, 5))
    for _ in range(30):
        moved = algo._difference_move(population, 0, task)
        assert np.count_nonzero(moved != population[0]) >= 1


def test_search_is_translation_invariant():
    """Everything is a difference vector, so moving the optimum is free."""
    offset = 2.0

    class ShiftedSphere(Problem):
        def __init__(self, dimension=4):
            super().__init__(dimension,
                             lower=-5.12 + offset, upper=5.12 + offset)

        def _evaluate(self, x):
            return float(np.sum((x - offset) ** 2))

    plain = Task(problem=Sphere(dimension=4), max_evals=1500)
    moved = Task(problem=ShiftedSphere(), max_evals=1500)
    x_plain, f_plain = MonkeyKingEvolution(seed=5).run(plain)
    x_moved, f_moved = MonkeyKingEvolution(seed=5).run(moved)

    np.testing.assert_allclose(x_moved, x_plain + offset, rtol=1e-8,
                               atol=1e-8)
    assert f_moved == pytest.approx(f_plain, rel=1e-8, abs=1e-20)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = MonkeyKingEvolution(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 9


@pytest.mark.parametrize("kwargs", [
    {"population_size": 3},
    {"n_clones": 0},
    {"fluctuation": 0.0},
    {"fluctuation": -1.0},
    {"change_rate": 0.0},
    {"change_rate": 1.5},
])
def test_mke_invalid_params(kwargs):
    with pytest.raises(ValueError):
        MonkeyKingEvolution(**kwargs)
