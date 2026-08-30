import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import BacterialForagingOptimization
from ikn_library.problems import Rastrigin, Sphere


def test_bfo_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = BacterialForagingOptimization(seed=42).run(task)
    assert best_fitness < 1e-3
    assert best_x.shape == (5,)


def test_bfo_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = BacterialForagingOptimization(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_bfo_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = BacterialForagingOptimization(step_size=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_bfo_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(BacterialForagingOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_bfo_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    BacterialForagingOptimization(seed=0).run(task)
    assert task.evals <= 400


def test_swimming_stops_when_conditions_worsen():
    """A bacterium swims only while the fitness keeps improving."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = BacterialForagingOptimization(n_swim=10, seed=3)
    # start at the optimum: every direction is worse, so no step is taken
    start = np.zeros(4)
    moved, fitness = algo._chemotaxis(task, start, task.problem.evaluate(start),
                                      np.full(4, 0.5))
    np.testing.assert_array_equal(moved, start)
    assert fitness == pytest.approx(0.0)


def test_swimming_moves_when_conditions_improve():
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = BacterialForagingOptimization(n_swim=10, seed=3)
    start = np.full(4, 3.0)
    moved, fitness = algo._chemotaxis(task, start, task.problem.evaluate(start),
                                      np.full(4, 0.2))
    assert fitness <= task.problem.evaluate(start)


def test_chemotaxis_never_worsens_a_bacterium():
    task = Task(problem=Rastrigin(dimension=5), max_evals=5000)
    algo = BacterialForagingOptimization(population_size=10, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(5):                    # stay below reproduction_interval
        previous = state[1].copy()
        state = algo.run_iteration(task, state)
        assert (state[1] <= previous + 1e-12).all()


def test_reproduction_duplicates_the_healthiest_half():
    algo = BacterialForagingOptimization(population_size=4, seed=0)
    bacteria = np.array([[0.0], [1.0], [2.0], [3.0]])
    fitness = np.array([10.0, 20.0, 30.0, 40.0])
    health = np.array([1.0, 2.0, 3.0, 4.0])       # first two are healthiest
    new_bacteria, new_fitness, new_health = algo._reproduce(
        bacteria, fitness, health)
    assert len(new_bacteria) == 4
    # only the two healthiest positions survive, each twice
    assert sorted(new_bacteria.ravel().tolist()) == [0.0, 0.0, 1.0, 1.0]
    np.testing.assert_array_equal(new_health, new_fitness)   # health resets


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=6000)
    algo = BacterialForagingOptimization(population_size=11, seed=2,
                                         reproduction_interval=2,
                                         elimination_interval=3)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == len(state[2]) == 11


@pytest.mark.parametrize("kwargs", [
    {"step_size": 0.0},
    {"n_swim": 0},
    {"reproduction_interval": 0},
    {"elimination_interval": 0},
    {"elimination_prob": 1.5},
    {"elimination_prob": -0.1},
])
def test_bfo_invalid_params(kwargs):
    with pytest.raises(ValueError):
        BacterialForagingOptimization(**kwargs)
