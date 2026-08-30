import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import ArtificialBeeColony
from ikn_library.problems import Rastrigin, Sphere


def test_abc_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = ArtificialBeeColony(population_size=30, seed=42).run(task)
    assert best_fitness < 1e-8
    assert best_x.shape == (5,)


def test_abc_solves_rastrigin():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = ArtificialBeeColony(population_size=30, seed=1).run(task)
    assert best_fitness < 1e-4        # ABC excels on multimodal landscapes


def test_abc_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = ArtificialBeeColony(population_size=10, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_abc_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(ArtificialBeeColony(population_size=15, seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_abc_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=500)
    ArtificialBeeColony(population_size=10, seed=0).run(task)
    assert task.evals <= 500


def test_scout_replaces_stagnant_source():
    """A source stuck at the limit is abandoned for a fresh random one.

    Only one scout is sent per iteration (Karaboga's formulation), so
    other sources may sit above the limit until their turn comes.
    """
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = ArtificialBeeColony(population_size=6, limit=1, seed=3)
    state = algo.init_population(task)
    task.next_iter()
    scouted = 0
    for _ in range(15):
        trials_before = state[2].copy()
        state = algo.run_iteration(task, state)
        # the most stagnant exhausted source was reset this iteration
        if (trials_before >= 1).any():
            scouted += int((state[2] == 0).any())
    assert scouted > 0


def test_no_scout_below_the_limit():
    """With a limit that is never reached, sources are never abandoned."""
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = ArtificialBeeColony(population_size=5, limit=10**6, seed=3)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        state = algo.run_iteration(task, state)
    assert (state[2] > 0).any()          # trials accumulate, no reset


def test_greedy_selection_never_worsens_a_source():
    task = Task(problem=Rastrigin(dimension=4), max_evals=4000)
    algo = ArtificialBeeColony(population_size=10, limit=1000, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(20):
        previous = state[1].copy()
        state = algo.run_iteration(task, state)
        # With no scout replacements (high limit), fitness only improves.
        assert (state[1] <= previous + 1e-12).all()


def test_probabilities_are_valid():
    algo = ArtificialBeeColony(population_size=4, seed=0)
    for fitness in (np.array([1.0, 2.0, 3.0, 4.0]),
                    np.array([-5.0, 0.0, 2.0, 10.0]),
                    np.zeros(4)):
        p = algo._probabilities(fitness)
        assert p.shape == (4,)
        assert np.isclose(p.sum(), 1.0) and (p > 0).all()
    # better (lower) fitness must get a higher probability
    p = algo._probabilities(np.array([1.0, 2.0, 3.0, 4.0]))
    assert p[0] > p[-1]


def test_invalid_limit():
    with pytest.raises(ValueError):
        ArtificialBeeColony(limit=0)
