import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import CatSwarmOptimization
from ikn_library.problems import Rastrigin, Sphere


def test_cso_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = CatSwarmOptimization(seed=42).run(task)
    assert best_fitness < 1e-3
    assert best_x.shape == (5,)


def test_cso_handles_rastrigin_well():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = CatSwarmOptimization(seed=1).run(task)
    assert best_fitness < 5.0


def test_cso_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = CatSwarmOptimization(srd=2.0, max_velocity=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_cso_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(CatSwarmOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_cso_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    CatSwarmOptimization(seed=0).run(task)
    assert task.evals <= 400


def test_seeking_never_worsens_a_cat():
    """Seeking mode is greedy: a cat only moves to a better copy."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=6000)
    algo = CatSwarmOptimization(population_size=10, mixture_ratio=0.01, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        previous = state[1].copy()
        state = algo.run_iteration(task, state)
        # with almost no tracing, every cat either improves or stays put
        assert (state[1] <= previous + 1e-12).all()


def test_seek_changes_only_cdc_dimensions():
    task = Task(problem=Sphere(dimension=20), max_evals=3000)
    algo = CatSwarmOptimization(cdc=0.1, smp=2, spc=True, seed=0)
    cat = np.zeros(20)
    moved, _ = algo._seek(task, cat, task.problem.evaluate(cat))
    changed = int(np.sum(~np.isclose(moved, cat)))
    assert changed <= 2          # round(0.1 * 20) = 2 dimensions at most


def test_seeking_range_narrows_over_time():
    """The seeking range decays with the budget, so late steps are small."""
    task = Task(problem=Sphere(dimension=10), max_evals=1000)
    algo = CatSwarmOptimization(srd=0.5, cdc=1.0, smp=2, seed=1)
    cat = np.zeros(10)
    early, _ = algo._seek(task, cat, np.inf)   # accept any copy
    early_step = np.abs(early - cat).max()
    while not task.stopping_condition():       # burn the budget
        task.eval(cat)
    late, _ = algo._seek(task, cat, np.inf)
    assert np.abs(late - cat).max() <= early_step


def test_velocity_stays_within_its_limit():
    task = Task(problem=Sphere(dimension=4), max_evals=4000)
    algo = CatSwarmOptimization(population_size=10, mixture_ratio=0.9,
                                max_velocity=0.1, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    limit = 0.1 * (task.upper - task.lower)
    for _ in range(10):
        state = algo.run_iteration(task, state)
        assert np.all(np.abs(state[2]) <= limit + 1e-9)


@pytest.mark.parametrize("kwargs", [
    {"mixture_ratio": 0.0},
    {"mixture_ratio": 1.0},
    {"smp": 0},
    {"srd": 0.0},
    {"cdc": 0.0},
    {"cdc": 1.5},
    {"velocity_factor": 0.0},
    {"max_velocity": 0.0},
])
def test_cso_invalid_params(kwargs):
    with pytest.raises(ValueError):
        CatSwarmOptimization(**kwargs)
