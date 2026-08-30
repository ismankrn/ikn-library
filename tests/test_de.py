import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import DifferentialEvolution
from ikn_library.algorithms.de import STRATEGIES
from ikn_library.problems import Rastrigin, Sphere


def test_de_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = DifferentialEvolution(seed=42).run(task)
    assert best_fitness < 1e-20
    assert best_x.shape == (5,)


def test_de_handles_rastrigin_well():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = DifferentialEvolution(seed=1).run(task)
    assert best_fitness < 3.0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_optimizes(strategy):
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    _, best_fitness = DifferentialEvolution(strategy=strategy, seed=3).run(task)
    assert best_fitness < 1e-3


def test_de_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = DifferentialEvolution(differential_weight=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_de_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(DifferentialEvolution(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_de_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    DifferentialEvolution(seed=0).run(task)
    assert task.evals <= 400


def test_population_never_worsens():
    """Greedy selection: a target is replaced only by a better trial."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=6000)
    algo = DifferentialEvolution(population_size=20, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        previous = state[1].copy()
        state = algo.run_iteration(task, state)
        assert (state[1] <= previous + 1e-12).all()


def test_pick_returns_distinct_indices_excluding_target():
    algo = DifferentialEvolution(population_size=10, seed=0)
    for _ in range(20):
        picked = algo._pick(exclude=3, count=5)
        assert len(picked) == len(set(picked)) == 5
        assert 3 not in picked


def test_crossover_always_takes_at_least_one_gene_from_the_mutant():
    """Even with CR = 0 the trial must differ from its target."""
    algo = DifferentialEvolution(crossover_rate=0.0, seed=0)
    target, mutant = np.zeros(8), np.ones(8)
    for _ in range(20):
        trial = algo._crossover(target, mutant, 8)
        assert np.count_nonzero(trial) == 1


def test_crossover_rate_controls_the_mix():
    algo = DifferentialEvolution(crossover_rate=1.0, seed=0)
    trial = algo._crossover(np.zeros(8), np.ones(8), 8)
    np.testing.assert_array_equal(trial, np.ones(8))   # everything from mutant


def test_difference_vectors_shrink_with_the_population():
    """DE's step size is self-adaptive: it follows the population spread."""
    task = Task(problem=Sphere(dimension=4), max_evals=8000)
    algo = DifferentialEvolution(population_size=20, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    initial_spread = np.mean(np.std(state[0], axis=0))
    for _ in range(30):
        state = algo.run_iteration(task, state)
    assert np.mean(np.std(state[0], axis=0)) < initial_spread / 10


@pytest.mark.parametrize("kwargs", [
    {"population_size": 4},
    {"differential_weight": 0.0},
    {"crossover_rate": 1.5},
    {"crossover_rate": -0.1},
    {"strategy": "best/3"},
])
def test_de_invalid_params(kwargs):
    with pytest.raises(ValueError):
        DifferentialEvolution(**kwargs)
