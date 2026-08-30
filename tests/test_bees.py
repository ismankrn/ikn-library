import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import BeesAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_bees_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = BeesAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-6
    assert best_x.shape == (5,)


def test_bees_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = BeesAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_bees_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = BeesAlgorithm(neighborhood=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_bees_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(BeesAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_bees_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    BeesAlgorithm(seed=0).run(task)
    assert task.evals <= 500


def test_sites_stay_sorted_by_fitness():
    task = Task(problem=Rastrigin(dimension=4), max_evals=5000)
    algo = BeesAlgorithm(seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        state = algo.run_iteration(task, state)
        fitness = state[1]
        assert (np.diff(fitness) >= 0).all()      # ascending = best first


def test_radius_shrinks_when_a_site_fails():
    """A site that cannot improve searches progressively finer."""
    task = Task(problem=Sphere(dimension=3), max_evals=6000)
    algo = BeesAlgorithm(shrink=0.5, stagnation_limit=10**6, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    initial = state[2].copy()
    for _ in range(12):
        state = algo.run_iteration(task, state)
    assert (state[2] <= initial + 1e-12).all()    # never grows
    assert (state[2] < initial).any()             # some site shrank


def test_stagnant_site_is_abandoned():
    task = Task(problem=Sphere(dimension=3), max_evals=6000)
    algo = BeesAlgorithm(stagnation_limit=1, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        state = algo.run_iteration(task, state)
        # abandonment resets both the counter and the radius
        assert (state[3] <= 1).all()
    assert np.isclose(state[2].max(), algo.neighborhood)


def test_elite_sites_get_more_recruits(monkeypatch):
    task = Task(problem=Sphere(dimension=3), max_evals=10000)
    algo = BeesAlgorithm(elite_sites=2, elite_bees=8, selected_bees=3, seed=0)
    recruit_counts = []
    original = algo._forage

    def spy(task_, centre, radius, n_recruits):
        recruit_counts.append(n_recruits)
        return original(task_, centre, radius, n_recruits)

    monkeypatch.setattr(algo, "_forage", spy)
    state = algo.init_population(task)
    task.next_iter()
    algo.run_iteration(task, state)
    assert recruit_counts[:2] == [8, 8]           # elite sites
    assert recruit_counts[2] == 3                 # non-elite selected site


@pytest.mark.parametrize("kwargs", [
    {"selected_sites": 0},
    {"selected_sites": 25, "population_size": 25},
    {"elite_sites": 0},
    {"elite_sites": 99},
    {"elite_bees": 0},
    {"neighborhood": 0.0},
    {"shrink": 0.0},
    {"shrink": 1.5},
    {"stagnation_limit": 0},
])
def test_bees_invalid_params(kwargs):
    with pytest.raises(ValueError):
        BeesAlgorithm(**kwargs)
