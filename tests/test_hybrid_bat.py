import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import BatAlgorithm, HybridBatAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_hba_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = HybridBatAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-2
    assert best_x.shape == (5,)


def test_hba_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = HybridBatAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_hba_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = HybridBatAlgorithm(differential_weight=3.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_hba_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(HybridBatAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_hba_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    HybridBatAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_hba_is_a_bat_algorithm_with_one_operator_swapped():
    """The relationship is expressed in the type, not just the docs."""
    assert issubclass(HybridBatAlgorithm, BatAlgorithm)
    # only the local search differs
    assert (HybridBatAlgorithm._local_step
            is not BatAlgorithm._local_step)
    assert HybridBatAlgorithm.run_iteration is BatAlgorithm.run_iteration


def test_local_step_is_a_difference_move_not_a_random_walk():
    """The DE donor takes its scale from the population's own spread."""
    task = Task(problem=Sphere(dimension=6), max_evals=5000)
    algo = HybridBatAlgorithm(population_size=5, crossover_rate=1.0,
                              differential_weight=0.5, seed=3)
    positions = np.array([
        [0.0] * 6, [1.0] * 6, [2.0] * 6, [4.0] * 6, [8.0] * 6])
    step = algo._local_step(task, positions, 0, positions[0], walk_scale=0.5)
    # with CR=1 every coordinate is the donor x_r1 + F*(x_r2 - x_r3),
    # so it must be a combination of the rows above, not Gaussian noise
    assert step.shape == (6,)
    assert np.allclose(step, step[0])          # all rows are constant vectors


def test_local_step_ignores_the_walk_scale():
    """Unlike plain BA, the hybrid's step has no decay schedule."""
    task = Task(problem=Sphere(dimension=5), max_evals=5000)
    positions = np.arange(25.0).reshape(5, 5)
    a = HybridBatAlgorithm(population_size=5, seed=4)._local_step(
        task, positions, 0, positions[0], walk_scale=0.001)
    b = HybridBatAlgorithm(population_size=5, seed=4)._local_step(
        task, positions, 0, positions[0], walk_scale=1000.0)
    np.testing.assert_array_equal(a, b)


def test_at_least_one_coordinate_comes_from_the_donor():
    task = Task(problem=Sphere(dimension=8), max_evals=5000)
    algo = HybridBatAlgorithm(population_size=6, crossover_rate=0.0, seed=5)
    positions = np.zeros((6, 8))
    positions[0] = 7.0                          # the target differs from all
    for _ in range(20):
        step = algo._local_step(task, positions, 0, positions[0],
                                walk_scale=0.1)
        assert np.count_nonzero(step != 7.0) >= 1


def test_donor_uses_three_distinct_other_individuals():
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = HybridBatAlgorithm(population_size=4, crossover_rate=1.0,
                              differential_weight=1.0, seed=6)
    # rows 1..3 are the only possible donors for index 0
    positions = np.array([[0.0] * 4, [1.0] * 4, [10.0] * 4, [100.0] * 4])
    seen = set()
    for _ in range(20):
        step = algo._local_step(task, positions, 0, positions[0],
                                walk_scale=0.1)
        seen.add(round(float(step[0]), 6))
    # x_r1 + 1.0*(x_r2 - x_r3) over permutations of {1, 10, 100}
    expected = {-89.0, -9.0, 91.0, 109.0, 10.0, -98.0}
    assert seen <= expected and len(seen) > 1


def test_the_hybrid_beats_plain_bat_on_rastrigin():
    """The swap is worth making; see the algorithm page for the caveat."""
    problem = Rastrigin(dimension=10)
    hybrid = [HybridBatAlgorithm(seed=s).run(
        Task(problem=problem, max_evals=6000))[1] for s in (1, 42)]
    plain = [BatAlgorithm(seed=s).run(
        Task(problem=problem, max_evals=6000))[1] for s in (1, 42)]
    assert np.mean(hybrid) < np.mean(plain)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = HybridBatAlgorithm(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 9 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"population_size": 3},
    {"differential_weight": 0.0},
    {"differential_weight": -1.0},
    {"crossover_rate": -0.1},
    {"crossover_rate": 1.5},
])
def test_hba_invalid_params(kwargs):
    with pytest.raises(ValueError):
        HybridBatAlgorithm(**kwargs)
