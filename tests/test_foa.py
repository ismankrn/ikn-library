import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import ForestOptimizationAlgorithm
from ikn_library.problems import Rastrigin, Sphere


def test_foa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = ForestOptimizationAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-4
    assert best_x.shape == (5,)


def test_foa_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = ForestOptimizationAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_foa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = ForestOptimizationAlgorithm(dx=2.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_foa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(ForestOptimizationAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_foa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    ForestOptimizationAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_only_age_zero_trees_seed_locally():
    """Age gates reproduction: older trees drop nothing."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = ForestOptimizationAlgorithm(population_size=5, lsc=3, seed=1)
    trees = np.zeros((5, 4))
    age = np.array([0, 1, 0, 2, 6])          # two trees are age 0
    seeds, _ = algo._local_seeding(task, trees, age, np.full(4, 0.1))
    assert len(seeds) == 2 * 3               # only those two seeded, lsc each


def test_local_seed_changes_exactly_one_coordinate():
    task = Task(problem=Sphere(dimension=6), max_evals=5000)
    algo = ForestOptimizationAlgorithm(population_size=1, lsc=5, seed=2)
    parent = np.full((1, 6), 1.0)
    seeds, _ = algo._local_seeding(task, parent, np.zeros(1, dtype=int),
                                   np.full(6, 0.5))
    for child in seeds:
        assert np.count_nonzero(child != parent[0]) == 1


def test_standing_trees_age_by_one_each_iteration():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = ForestOptimizationAlgorithm(population_size=6, life_time=50,
                                       transfer_rate=0.0, seed=3)
    trees, fitness, age = algo.init_population(task)
    task.next_iter()
    trees, fitness, age = algo.run_iteration(task, (trees, fitness, age))
    # the elite was reset to 0; every other original tree is now age 1
    assert sorted(set(age.tolist())) == [0, 1]


def test_old_trees_are_cut_into_the_candidate_pool():
    algo = ForestOptimizationAlgorithm(population_size=10, life_time=3)
    trees = np.arange(5, dtype=float).reshape(5, 1)
    fitness = np.zeros(5)
    age = np.array([0, 1, 4, 9, 2])          # indices 2 and 3 are too old
    kept, _, kept_age, candidates = algo._limit_population(trees, fitness, age)
    assert len(kept) == 3 and len(candidates) == 2
    assert np.all(kept_age <= 3)
    assert sorted(c.item() for c in candidates) == [2.0, 3.0]


def test_area_limit_cuts_the_worst_trees():
    algo = ForestOptimizationAlgorithm(population_size=2, life_time=50)
    trees = np.arange(5, dtype=float).reshape(5, 1)
    fitness = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    age = np.zeros(5, dtype=int)
    kept, kept_fitness, _, candidates = algo._limit_population(
        trees, fitness, age)
    assert len(kept) == 2
    np.testing.assert_array_equal(kept_fitness, [1.0, 2.0])   # the two best
    assert len(candidates) == 3                                # the rest recycled


def test_global_seeding_replants_from_the_candidate_pool():
    """Discarded trees are the raw material for long-range search."""
    task = Task(problem=Sphere(dimension=8), max_evals=5000)
    algo = ForestOptimizationAlgorithm(gsc=2, transfer_rate=1.0, seed=4)
    candidates = [np.zeros(8), np.zeros(8), np.zeros(8)]
    seeds, seed_fitness = algo._global_seeding(task, candidates)
    assert len(seeds) == 3 and len(seed_fitness) == 3
    for tree in seeds:
        # exactly gsc coordinates were replaced by random values
        assert np.count_nonzero(tree != 0.0) <= 2
        assert np.any(tree != 0.0)


def test_global_seeding_is_skipped_when_the_pool_is_tiny():
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = ForestOptimizationAlgorithm(transfer_rate=0.1, seed=4)
    seeds, _ = algo._global_seeding(task, [np.zeros(4)])   # round(0.1) == 0
    assert seeds == []
    assert task.evals == 0


def test_elitism_keeps_the_best_tree_seeding():
    task = Task(problem=Rastrigin(dimension=4), max_evals=5000)
    algo = ForestOptimizationAlgorithm(population_size=8, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(5):
        state = algo.run_iteration(task, state)
        task.next_iter()
        _, fitness, age = state
        assert age[np.argmin(fitness)] == 0


def test_forest_size_stays_within_the_area_limit():
    task = Task(problem=Sphere(dimension=3), max_evals=6000)
    algo = ForestOptimizationAlgorithm(population_size=12, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        trees, fitness, age = state
        assert len(trees) == len(fitness) == len(age)
        # global seeding may add a few above the limit, but not unboundedly
        assert 1 <= len(trees) <= 12 + algo.population_size


@pytest.mark.parametrize("kwargs", [
    {"life_time": 0},
    {"lsc": 0},
    {"gsc": 0},
    {"transfer_rate": -0.1},
    {"transfer_rate": 1.5},
    {"dx": 0.0},
])
def test_foa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        ForestOptimizationAlgorithm(**kwargs)
