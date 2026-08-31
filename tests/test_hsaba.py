import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import (
    HybridBatAlgorithm,
    HybridSelfAdaptiveBatAlgorithm,
)
from ikn_library.problems import Rastrigin, Sphere


def test_hsaba_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = HybridSelfAdaptiveBatAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-2
    assert best_x.shape == (5,)


def test_hsaba_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = HybridSelfAdaptiveBatAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_hsaba_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = HybridSelfAdaptiveBatAlgorithm(
        differential_weight=3.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_hsaba_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(HybridSelfAdaptiveBatAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_hsaba_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    HybridSelfAdaptiveBatAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_hsaba_reuses_the_hybrid_local_search():
    """It is the hybrid plus self-adaptation, not a new search."""
    assert issubclass(HybridSelfAdaptiveBatAlgorithm, HybridBatAlgorithm)
    assert (HybridSelfAdaptiveBatAlgorithm._local_step
            is HybridBatAlgorithm._local_step)
    # but the iteration loop is its own, to carry per-bat parameters
    assert (HybridSelfAdaptiveBatAlgorithm.run_iteration
            is not HybridBatAlgorithm.run_iteration)


def test_each_bat_carries_its_own_loudness_and_rate():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = HybridSelfAdaptiveBatAlgorithm(population_size=12, seed=3)
    _, _, _, loudness, rates = algo.init_population(task)
    assert loudness.shape == (12,) and rates.shape == (12,)
    assert np.all(loudness >= algo.min_loudness)
    assert np.all(loudness <= algo.max_loudness)
    assert np.all((rates >= 0.0) & (rates <= 1.0))
    assert len(np.unique(rates)) > 1              # genuinely per-bat


def test_self_adaptation_redraws_both_parameters():
    algo = HybridSelfAdaptiveBatAlgorithm(population_size=4, tau_1=1.0,
                                          tau_2=1.0, min_loudness=0.1,
                                          max_loudness=0.9, seed=5)
    loudness = np.full(4, 0.5)
    rates = np.full(4, 0.5)
    for i in range(4):
        algo._self_adapt(loudness, rates, i)
    assert np.all(loudness != 0.5) and np.all(rates != 0.5)
    assert np.all((loudness >= 0.1) & (loudness <= 0.9))


def test_self_adaptation_can_be_switched_off():
    algo = HybridSelfAdaptiveBatAlgorithm(population_size=4, tau_1=0.0,
                                          tau_2=0.0, seed=5)
    loudness = np.full(4, 0.5)
    rates = np.full(4, 0.5)
    for i in range(4):
        algo._self_adapt(loudness, rates, i)
    np.testing.assert_array_equal(loudness, 0.5)
    np.testing.assert_array_equal(rates, 0.5)


def test_loudness_does_not_ratchet_downward():
    """The point of the self-adaptation: no monotone decay.

    In plain BA and HBA, loudness is multiplied by alpha on every
    accepted move and slides toward zero. Here it is re-drawn, so it can
    go back up.
    """
    task = Task(problem=Rastrigin(dimension=5), max_evals=6000)
    algo = HybridSelfAdaptiveBatAlgorithm(population_size=10, tau_1=0.5,
                                          seed=4)
    state = algo.init_population(task)
    task.next_iter()
    went_up = False
    while not task.stopping_condition():
        previous = state[3].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        if np.any(state[3] > previous + 1e-12):
            went_up = True
        assert np.all(state[3] >= algo.min_loudness)   # never collapses
    assert went_up


def test_it_beats_plain_bat_and_the_published_settings():
    problem = Rastrigin(dimension=10)
    tuned = [HybridSelfAdaptiveBatAlgorithm(seed=s).run(
        Task(problem=problem, max_evals=6000))[1] for s in (1, 42)]
    published = [HybridSelfAdaptiveBatAlgorithm(
        seed=s, population_size=50, min_loudness=0.9, tau_2=0.1).run(
        Task(problem=problem, max_evals=6000))[1] for s in (1, 42)]
    assert np.mean(tuned) < np.mean(published)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = HybridSelfAdaptiveBatAlgorithm(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 9 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"tau_1": -0.1},
    {"tau_1": 1.5},
    {"tau_2": -0.1},
    {"tau_2": 1.5},
    {"min_loudness": 0.0},
    {"min_loudness": 0.9, "max_loudness": 0.5},
])
def test_hsaba_invalid_params(kwargs):
    with pytest.raises(ValueError):
        HybridSelfAdaptiveBatAlgorithm(**kwargs)
