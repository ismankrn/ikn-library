import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.problems import FeatureSelectionProblem


@pytest.fixture(scope="module")
def dataset():
    return make_classification(
        n_samples=120, n_features=10, n_informative=3, n_redundant=0,
        n_repeated=0, shuffle=False, random_state=42,
    )


def test_selected_features_and_mask(dataset):
    X, y = dataset
    problem = FeatureSelectionProblem(X, y)
    x = np.array([0.9, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    np.testing.assert_array_equal(problem.selected_features(x), [0, 2, 9])
    assert problem.feature_mask(x).sum() == 3


def test_empty_subset_gets_worst_fitness(dataset):
    X, y = dataset
    problem = FeatureSelectionProblem(X, y)
    assert problem.evaluate(np.zeros(10)) == 1.0


def test_fitness_in_unit_interval(dataset):
    X, y = dataset
    problem = FeatureSelectionProblem(X, y, cv=3)
    fitness = problem.evaluate(np.ones(10))
    assert 0.0 <= fitness <= 1.0


def test_binary_aco_feature_selection_improves(dataset):
    X, y = dataset
    problem = FeatureSelectionProblem(X, y, cv=3)
    all_features_fitness = problem.evaluate(np.ones(10))

    task = Task(problem=problem, max_evals=300)
    algo = BinaryAntColonyOptimization(population_size=10, seed=42)
    best_x, best_fitness = algo.run(task)

    assert best_fitness <= all_features_fitness
    assert len(problem.selected_features(best_x)) >= 1


def test_input_validation(dataset):
    X, y = dataset
    with pytest.raises(ValueError):
        FeatureSelectionProblem(X, y[:-5])
    with pytest.raises(ValueError):
        FeatureSelectionProblem(X, y, alpha=1.5)
    with pytest.raises(ValueError):
        FeatureSelectionProblem(X[0], y)


@pytest.mark.parametrize("threshold", [1.0, 1.5, -0.1])
def test_threshold_outside_unit_interval_is_rejected(dataset, threshold):
    """threshold=1.0 would make every candidate empty, silently."""
    X, y = dataset
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\)"):
        FeatureSelectionProblem(X, y, threshold=threshold)


def test_threshold_zero_is_allowed_for_binary_solutions(dataset):
    X, y = dataset
    problem = FeatureSelectionProblem(X, y, threshold=0.0)
    bits = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    np.testing.assert_array_equal(problem.selected_features(bits), [0, 2, 9])


def test_threshold_changes_the_selected_subset(dataset):
    X, y = dataset
    x = np.linspace(0.0, 1.0, X.shape[1])
    sizes = [len(FeatureSelectionProblem(X, y, threshold=t).selected_features(x))
             for t in (0.25, 0.5, 0.75)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1]
