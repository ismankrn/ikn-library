import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import CuckooSearch
from ikn_library.problems import Rastrigin, Sphere


def test_cuckoo_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = CuckooSearch(seed=42).run(task)
    assert best_fitness < 1e-8
    assert best_x.shape == (5,)


def test_cuckoo_handles_rastrigin_well():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = CuckooSearch(seed=1).run(task)
    assert best_fitness < 8.0


def test_cuckoo_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = CuckooSearch(step_size=1.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_cuckoo_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(CuckooSearch(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_cuckoo_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    CuckooSearch(seed=0).run(task)
    assert task.evals <= 400


def test_levy_flight_is_heavy_tailed():
    """Lévy steps are mostly small but produce rare huge jumps —
    unlike a Gaussian, whose tail decays far faster."""
    algo = CuckooSearch(levy_exponent=1.5, seed=0)
    steps = np.abs(algo._levy_flight(20000))
    gaussian = np.abs(np.random.default_rng(0).normal(0, 1, 20000))
    assert np.median(steps) < np.median(gaussian)      # typically smaller
    assert steps.max() > 20 * gaussian.max()           # but far longer jumps


def test_levy_exponent_controls_jump_length():
    small_beta = np.abs(CuckooSearch(levy_exponent=1.1, seed=0)._levy_flight(20000))
    large_beta = np.abs(CuckooSearch(levy_exponent=1.9, seed=0)._levy_flight(20000))
    assert small_beta.max() > large_beta.max()         # smaller beta jumps further


def test_no_abandonment_when_discovery_rate_is_zero():
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    algo = CuckooSearch(population_size=10, discovery_rate=0.0, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    evals_before = task.evals
    algo.run_iteration(task, state)
    # only the egg-laying phase runs: one evaluation per nest
    assert task.evals - evals_before == 10


def test_fitness_stays_consistent_with_nests():
    task = Task(problem=Rastrigin(dimension=4), max_evals=3000)
    algo = CuckooSearch(population_size=10, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        state = algo.run_iteration(task, state)
        nests, fitness = state
        recomputed = np.array([task.problem.evaluate(x) for x in nests])
        np.testing.assert_allclose(fitness, recomputed)


@pytest.mark.parametrize("kwargs", [
    {"discovery_rate": 1.0},
    {"discovery_rate": -0.1},
    {"step_size": 0.0},
    {"levy_exponent": 0.0},
    {"levy_exponent": 2.5},
])
def test_cuckoo_invalid_params(kwargs):
    with pytest.raises(ValueError):
        CuckooSearch(**kwargs)
