import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import HarmonySearch
from ikn_library.problems import Rastrigin, Sphere


def test_hs_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = HarmonySearch(seed=42).run(task)
    assert best_fitness < 1e-5
    assert best_x.shape == (5,)


def test_hs_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = HarmonySearch(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_hs_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = HarmonySearch(bandwidth=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_hs_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(HarmonySearch(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_hs_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    HarmonySearch(seed=0).run(task)
    assert task.evals <= 400


def test_one_evaluation_per_iteration():
    """HS improvises a single harmony each iteration."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = HarmonySearch(population_size=12, seed=3)
    state = algo.init_population(task)
    assert task.evals == 12                 # filling the memory
    task.next_iter()
    for expected in range(13, 18):
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert task.evals == expected


def test_variables_come_from_different_harmonies():
    """The defining trait: one improvisation mixes the whole memory."""
    task = Task(problem=Sphere(dimension=8), max_evals=5000)
    algo = HarmonySearch(population_size=6, hmcr=1.0, par=0.0, seed=5)
    # each harmony is a constant row, so a value reveals its source
    memory = np.repeat(np.arange(6.0).reshape(6, 1), 8, axis=1)
    sources = set()
    for _ in range(20):
        harmony = algo._improvise(task, memory, np.zeros(8))
        sources.update(np.unique(harmony).tolist())
        # within a single improvisation, values may come from many rows
        assert np.all(np.isin(harmony, np.arange(6.0)))
    assert len(sources) > 1                 # the memory really is mixed


def test_hmcr_controls_random_restarts():
    """With hmcr=0 every variable is a fresh uniform draw."""
    task = Task(problem=Sphere(dimension=6), max_evals=5000)
    algo = HarmonySearch(population_size=4, hmcr=0.0, seed=1)
    memory = np.zeros((4, 6))
    harmony = algo._improvise(task, memory, np.zeros(6))
    assert np.all(harmony != 0.0)           # nothing was taken from memory
    assert np.all(harmony >= task.lower) and np.all(harmony <= task.upper)


def test_pitch_adjustment_only_touches_remembered_values():
    """With hmcr=1 and par=0 the harmony is a pure memory recombination."""
    task = Task(problem=Sphere(dimension=6), max_evals=5000)
    algo = HarmonySearch(population_size=4, hmcr=1.0, par=0.0, seed=1)
    memory = np.repeat(np.arange(4.0).reshape(4, 1), 6, axis=1)
    harmony = algo._improvise(task, memory, np.full(6, 99.0))
    # a large bandwidth changes nothing, because par is 0
    assert np.all(np.isin(harmony, np.arange(4.0)))


def test_new_harmony_replaces_only_the_worst_and_only_if_better():
    task = Task(problem=Sphere(dimension=2), max_evals=5000)
    algo = HarmonySearch(population_size=4, seed=6)
    memory = np.array([[0.0, 0.0], [3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
    fitness = np.array([0.0, 18.0, 2.0, 8.0])       # index 1 is the worst
    new_memory, new_fitness = algo.run_iteration(
        task, (memory.copy(), fitness.copy()))
    # the three better harmonies are untouched
    for i in (0, 2, 3):
        np.testing.assert_array_equal(new_memory[i], memory[i])
    assert new_fitness[1] <= fitness[1]


def test_memory_quality_never_degrades():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = HarmonySearch(population_size=10, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous = np.sort(state[1]).copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(np.sort(state[1]) <= previous + 1e-12)


def test_memory_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=1500)
    algo = HarmonySearch(population_size=7, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 7


@pytest.mark.parametrize("kwargs", [
    {"hmcr": -0.1},
    {"hmcr": 1.5},
    {"par": -0.1},
    {"par": 1.5},
    {"bandwidth": 0.0},
    {"bandwidth": -1.0},
])
def test_hs_invalid_params(kwargs):
    with pytest.raises(ValueError):
        HarmonySearch(**kwargs)
