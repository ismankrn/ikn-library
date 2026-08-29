import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.problems import Problem


class SubsetMatch(Problem):
    """Minimize the Hamming distance to a known target bit mask."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)
        super().__init__(dimension=len(self.target), lower=0.0, upper=1.0)

    def _evaluate(self, x):
        return float(np.sum((x > 0.5) != (self.target > 0.5)))


def test_binary_aco_finds_target_subset():
    target = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    task = Task(problem=SubsetMatch(target), max_evals=6000)
    algo = BinaryAntColonyOptimization(population_size=20, seed=42)
    best_x, best_fitness = algo.run(task)
    assert best_fitness == 0.0
    np.testing.assert_array_equal(best_x, target)


def test_binary_aco_emits_only_bits():
    target = np.ones(6)
    task = Task(problem=SubsetMatch(target), max_evals=500)
    best_x, _ = BinaryAntColonyOptimization(population_size=10, seed=3).run(task)
    assert set(np.unique(best_x)) <= {0.0, 1.0}


def test_binary_aco_never_selects_empty_subset():
    class CountOnes(Problem):
        def __init__(self):
            super().__init__(dimension=8, lower=0.0, upper=1.0)

        def _evaluate(self, x):
            assert np.sum(x) >= 1, "empty subset was evaluated"
            return float(np.sum(x))

    task = Task(problem=CountOnes(), max_evals=2000)
    best_x, _ = BinaryAntColonyOptimization(population_size=15, seed=5).run(task)
    assert np.sum(best_x) >= 1


def test_binary_aco_is_reproducible_with_seed():
    target = np.array([1, 0, 1, 0, 1, 0])
    results = []
    for _ in range(2):
        task = Task(problem=SubsetMatch(target), max_evals=1000)
        algo = BinaryAntColonyOptimization(population_size=10, seed=99)
        results.append(algo.run(task))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_binary_aco_respects_eval_budget():
    task = Task(problem=SubsetMatch(np.ones(5)), max_evals=300)
    BinaryAntColonyOptimization(population_size=10, seed=0).run(task)
    assert task.evals <= 300


@pytest.mark.parametrize("kwargs", [
    {"evaporation": 0.0},
    {"evaporation": 1.0},
    {"tau_min": 0.5, "tau_max": 0.4},
])
def test_binary_aco_invalid_params(kwargs):
    with pytest.raises(ValueError):
        BinaryAntColonyOptimization(**kwargs)
