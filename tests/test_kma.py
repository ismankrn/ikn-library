import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import KomodoMlipirAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_kma_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = KomodoMlipirAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-10
    assert best_x.shape == (5,)


def test_kma_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = KomodoMlipirAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_kma_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    best_x, _ = KomodoMlipirAlgorithm(parthenogenesis_radius=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_kma_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(KomodoMlipirAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_kma_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    KomodoMlipirAlgorithm(seed=0).run(task)
    assert task.evals <= 500


def test_group_sizes_always_valid():
    """Big males, one female, and small males must all fit."""
    algo = KomodoMlipirAlgorithm(max_big_males=3)
    for n in range(5, 60):
        n_big, n_small = algo._group_sizes(n)
        assert n_big >= 2
        assert n_big <= algo.max_big_males
        assert n_small >= 2
        assert n_big + 1 + n_small == n


def test_big_male_cap_is_respected():
    algo = KomodoMlipirAlgorithm(max_big_males=5, big_male_portion=0.9)
    n_big, _ = algo._group_sizes(100)
    assert n_big == 5


def test_population_stays_within_adaptive_limits():
    task = Task(problem=Rastrigin(dimension=4), max_evals=8000)
    algo = KomodoMlipirAlgorithm(population_size=12, min_population=8,
                                 max_population=25, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    sizes = set()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        sizes.add(len(state[0]))
    assert min(sizes) >= 8 and max(sizes) <= 25
    assert len(sizes) > 1          # the population really does adapt


def test_population_stays_sorted():
    task = Task(problem=Sphere(dimension=4), max_evals=4000)
    algo = KomodoMlipirAlgorithm(seed=3)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(15):
        state = algo.run_iteration(task, state)
        assert (np.diff(state[1]) >= 0).all()


def test_mating_produces_a_convex_combination():
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    algo = KomodoMlipirAlgorithm(seed=0)
    winner = np.array([1.0, 1.0, 1.0])
    female = np.array([-1.0, -1.0, -1.0])
    child, _ = algo._mate(task, winner, female)
    assert np.all(child >= -1.0) and np.all(child <= 1.0)


def test_parthenogenesis_step_is_bounded_by_its_radius():
    task = Task(problem=Sphere(dimension=4), max_evals=1000)
    algo = KomodoMlipirAlgorithm(parthenogenesis_radius=0.1, seed=0)
    female = np.zeros(4)
    child, _ = algo._parthenogenesis(task, female)
    limit = 0.1 * (task.upper - task.lower)
    assert np.all(np.abs(child - female) <= limit + 1e-12)


@pytest.mark.parametrize("kwargs", [
    {"big_male_portion": 0.0},
    {"big_male_portion": 1.0},
    {"mlipir_rate": 0.0},
    {"mlipir_rate": 1.0},
    {"max_big_males": 1},
    {"adaptation_step": 0},
    {"min_population": 2},
    {"min_population": 50, "max_population": 20},
    {"parthenogenesis_radius": 0.0},
])
def test_kma_invalid_params(kwargs):
    with pytest.raises(ValueError):
        KomodoMlipirAlgorithm(**kwargs)
