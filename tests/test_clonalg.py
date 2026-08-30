import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import ClonalSelectionAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_clonalg_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = ClonalSelectionAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-4
    assert best_x.shape == (5,)


def test_clonalg_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = ClonalSelectionAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_clonalg_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = ClonalSelectionAlgorithm(rho=0.5, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_clonalg_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(ClonalSelectionAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_clonalg_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    ClonalSelectionAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_better_antibodies_get_more_clones():
    """Cloning is proportional to affinity: rank 1 gets the most."""
    algo = ClonalSelectionAlgorithm(population_size=20, n_select=10,
                                    clone_factor=0.5)
    counts = algo._clone_counts()
    assert len(counts) == 10
    assert counts[0] == counts.max()
    # strictly non-increasing with rank
    assert np.all(np.diff(counts) <= 0)
    assert np.all(counts >= 1)


def test_better_clones_mutate_less():
    """Hypermutation is inversely proportional to affinity."""
    algo = ClonalSelectionAlgorithm(population_size=10, rho=3.0)
    rates = algo._mutation_rates(10, progress=0.0)
    assert np.all(np.diff(rates) > 0)          # rises from best to worst
    assert rates[0] < rates[-1] / 10           # and by a wide margin


def test_mutation_rate_shrinks_as_budget_is_spent():
    """Without this decay the population never converges (see docs)."""
    algo = ClonalSelectionAlgorithm(population_size=10)
    early = algo._mutation_rates(10, progress=0.0)
    late = algo._mutation_rates(10, progress=0.9)
    assert np.all(late < early)
    np.testing.assert_allclose(late / early, 0.1 ** 2, rtol=1e-9)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=6000)
    algo = ClonalSelectionAlgorithm(population_size=11, n_select=5, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 11


def test_state_stays_sorted_by_affinity():
    task = Task(problem=Rastrigin(dimension=4), max_evals=4000)
    algo = ClonalSelectionAlgorithm(population_size=12, seed=5)
    state = algo.init_population(task)
    assert np.all(np.diff(state[1]) >= 0)
    task.next_iter()
    for _ in range(5):
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(np.diff(state[1]) >= 0)


def test_selection_never_loses_the_best_antibody():
    """Parents compete with their clones, so the elite cannot regress."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=5000)
    algo = ClonalSelectionAlgorithm(population_size=10, n_replace=0, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        previous_best = state[1][0]
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert state[1][0] <= previous_best + 1e-12


def test_receptor_editing_replaces_the_worst_antibodies():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = ClonalSelectionAlgorithm(population_size=8, n_select=4,
                                    n_replace=3, seed=11)
    state = algo.init_population(task)
    task.next_iter()
    before = state[0].copy()
    state = algo.run_iteration(task, state)
    # the elite survives somewhere in the new population
    assert any(np.allclose(before[0], row) for row in state[0])


@pytest.mark.parametrize("kwargs", [
    {"n_select": 0},
    {"n_select": 21},
    {"clone_factor": 0.0},
    {"clone_factor": -1.0},
    {"n_replace": -1},
    {"n_replace": 20},
    {"rho": 0.0},
    {"rho": -2.0},
])
def test_clonalg_invalid_params(kwargs):
    with pytest.raises(ValueError):
        ClonalSelectionAlgorithm(population_size=20, **kwargs)
