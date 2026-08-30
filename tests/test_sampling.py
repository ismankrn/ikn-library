import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.sampling import UndersamplingProblem


class NearestCentroid:
    """Tiny numpy-only classifier for tests."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None], axis=2)
        return self.classes_[np.argmin(distances, axis=1)]


@pytest.fixture
def imbalanced():
    """Overlapping classes; half of the majority is a noisy cluster
    sitting on top of the minority, which ruins the majority centroid."""
    rng = np.random.default_rng(0)
    X_min = rng.normal([1.0, 1.0], 1.0, (30, 2))
    X_maj_clean = rng.normal([-1.0, -1.0], 1.0, (100, 2))
    X_maj_noise = rng.normal([1.0, 1.0], 0.5, (100, 2))
    X_train = np.vstack([X_min, X_maj_clean, X_maj_noise])
    y_train = np.array([1] * 30 + [0] * 200)

    X_val = np.vstack([rng.normal([1.0, 1.0], 1.0, (20, 2)),
                       rng.normal([-1.0, -1.0], 1.0, (80, 2))])
    y_val = np.array([1] * 20 + [0] * 80)
    return X_train, y_train, X_val, y_val


def make_problem(data, **kwargs):
    X_train, y_train, X_val, y_val = data
    kwargs.setdefault("estimator", NearestCentroid())
    return UndersamplingProblem(X_train, y_train, X_val, y_val, **kwargs)


def test_dimension_and_target(imbalanced):
    problem = make_problem(imbalanced)
    assert problem.dimension == 200         # one bit per majority sample
    assert problem.target == 30             # ratio 1.0 -> n_minority
    assert make_problem(imbalanced, target_ratio=1.5).target == 45


def test_repair_enforces_exact_count(imbalanced):
    problem = make_problem(imbalanced)
    for x in (np.zeros(200), np.ones(200),
              np.random.default_rng(1).random(200)):
        assert problem.majority_mask(x).sum() == problem.target


def test_repair_is_deterministic(imbalanced):
    problem = make_problem(imbalanced)
    x = np.random.default_rng(2).random(200)
    np.testing.assert_array_equal(problem.majority_mask(x), problem.majority_mask(x))
    assert problem.evaluate(x) == problem.evaluate(x)


def test_minority_class_always_fully_kept(imbalanced):
    problem = make_problem(imbalanced)
    _, y_res = problem.resampled_data(np.zeros(200))
    assert (y_res == 1).sum() == 30
    assert (y_res == 0).sum() == problem.target


def test_optimized_beats_random_undersampling(imbalanced):
    problem = make_problem(imbalanced)
    rng = np.random.default_rng(3)
    random_fitness = np.mean([problem.evaluate(rng.random(200)) for _ in range(20)])

    task = Task(problem=problem, max_evals=3000)
    algo = BinaryAntColonyOptimization(population_size=20, seed=42)
    best_x, best_fitness = algo.run(task)

    assert best_fitness < random_fitness
    # The optimizer should discard most of the noisy majority cluster
    # (indices 100..199 of the majority block overlap the minority).
    noisy_kept = problem.majority_mask(best_x)[100:].sum()
    assert noisy_kept < problem.target / 2


def test_selected_indices_are_valid(imbalanced):
    problem = make_problem(imbalanced)
    indices = problem.selected_indices(np.ones(200))
    assert len(indices) == 30 + problem.target
    assert len(np.unique(indices)) == len(indices)
    assert indices.min() >= 0 and indices.max() < 230


def test_works_with_sklearn_estimator(imbalanced):
    pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsClassifier
    problem = make_problem(imbalanced, estimator=KNeighborsClassifier(n_neighbors=3))
    fitness = problem.evaluate(np.ones(200))
    assert 0.0 <= fitness <= 1.0


def test_input_validation(imbalanced):
    X_train, y_train, X_val, y_val = imbalanced
    est = NearestCentroid()
    with pytest.raises(ValueError):
        UndersamplingProblem(X_train, y_train[:-1], X_val, y_val, estimator=est)
    with pytest.raises(ValueError):
        UndersamplingProblem(X_train, y_train, X_val, y_val[:-1], estimator=est)
    with pytest.raises(ValueError):
        UndersamplingProblem(X_train, np.zeros(230), X_val, y_val, estimator=est)
    with pytest.raises(ValueError):
        UndersamplingProblem(X_train, y_train, X_val, y_val, estimator=est,
                             target_ratio=0.0)
    with pytest.raises(ValueError):
        UndersamplingProblem(X_train, y_train, X_val, y_val, estimator=est,
                             metric="rmse")
