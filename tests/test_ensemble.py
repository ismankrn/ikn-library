import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import AntColonyOptimization, BinaryAntColonyOptimization
from ikn_library.ensemble import EnsembleWeightProblem, tree_proba_matrix


@pytest.fixture
def known_optimum():
    """Member 0 is a near-perfect predictor; members 1-4 are noise."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    perfect = np.where(y == 1, 0.9, 0.1)
    noise = rng.uniform(0.0, 1.0, (120, 4))
    P = np.column_stack([perfect, noise])
    return P, y


def test_uniform_weights_are_soft_majority_voting(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y)
    uniform = np.full(5, 0.2)
    np.testing.assert_allclose(problem.scores(uniform), P.mean(axis=1))


def test_zero_weights_fall_back_to_uniform(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y)
    np.testing.assert_allclose(problem.weights(np.zeros(5)), 0.2)


def test_weights_normalize_to_one(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y)
    w = problem.weights(np.array([0.3, 0.1, 0.0, 0.4, 0.2]))
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_aco_beats_uniform_voting_and_finds_good_member(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y)
    uniform_fitness = problem.evaluate(np.full(5, 0.2))

    task = Task(problem=problem, max_evals=2000)
    algo = AntColonyOptimization(population_size=20, archive_size=30, seed=42)
    best_x, best_fitness = algo.run(task)

    assert best_fitness < uniform_fitness
    assert best_fitness == pytest.approx(0.0)  # member 0 alone is perfect
    weights = problem.weights(best_x)
    assert weights[0] == max(weights)


def test_binary_aco_prunes_to_good_member(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y)
    task = Task(problem=problem, max_evals=1500)
    algo = BinaryAntColonyOptimization(population_size=15, seed=7)
    best_x, best_fitness = algo.run(task)
    assert best_x[0] == 1.0  # the perfect member is kept
    assert best_fitness == pytest.approx(0.0)


def test_predict_on_new_data(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y)
    P_new = np.array([[0.9, 0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5, 0.5]])
    pred = problem.predict(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), P_new)
    np.testing.assert_array_equal(pred, [1, 0])


def test_threshold_changes_predictions(known_optimum):
    P, y = known_optimum
    x = np.full(5, 0.2)
    low = EnsembleWeightProblem(P, y, threshold=0.1).predict(x).sum()
    high = EnsembleWeightProblem(P, y, threshold=0.9).predict(x).sum()
    assert low > high


def test_custom_metric_callable(known_optimum):
    P, y = known_optimum
    calls = []

    def metric(y_true, y_pred):
        calls.append(1)
        return float(np.mean(y_true == y_pred))

    problem = EnsembleWeightProblem(P, y, metric=metric)
    problem.evaluate(np.full(5, 0.2))
    assert calls


def test_f1_metric(known_optimum):
    P, y = known_optimum
    problem = EnsembleWeightProblem(P, y, metric="f1")
    fitness = problem.evaluate(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert fitness == pytest.approx(0.0)  # perfect member -> F1 = 1


def test_input_validation(known_optimum):
    P, y = known_optimum
    with pytest.raises(ValueError):
        EnsembleWeightProblem(P, y[:-1])
    with pytest.raises(ValueError):
        EnsembleWeightProblem(P[0], y)
    with pytest.raises(ValueError):
        EnsembleWeightProblem(P, np.where(y == 1, "UC", "N"))
    with pytest.raises(ValueError):
        EnsembleWeightProblem(P, y, threshold=1.5)
    with pytest.raises(ValueError):
        EnsembleWeightProblem(P, y, metric="rmse")


class _FakeMember:
    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])


class _FakeForest:
    def __init__(self, members):
        self.estimators_ = members


def test_tree_proba_matrix_from_ensemble_and_list():
    members = [_FakeMember([0.2, 0.8]), _FakeMember([0.6, 0.4])]
    expected = np.array([[0.2, 0.6], [0.8, 0.4]])
    X = np.zeros((2, 3))
    np.testing.assert_allclose(tree_proba_matrix(_FakeForest(members), X), expected)
    np.testing.assert_allclose(tree_proba_matrix(members, X), expected)
    with pytest.raises(ValueError):
        tree_proba_matrix([], X)
