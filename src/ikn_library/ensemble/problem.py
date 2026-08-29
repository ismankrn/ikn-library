"""Weighted-voting ensembles as an optimization problem."""

import numpy as np

from ikn_library.problems.problem import Problem


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def _f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator > 0 else 0.0

_METRICS = {"accuracy": _accuracy, "f1": _f1}


class EnsembleWeightProblem(Problem):
    """Optimize the voting weights of a fitted ensemble.

    Plain majority voting gives every ensemble member the same say. Here
    each member ``j`` gets a weight ``w_j``; a sample's combined score is
    the weighted average of the members' class-1 probabilities, and the
    predicted label is 1 when that score exceeds ``threshold``. The
    weight vector is what a metaheuristic optimizes.

    Weights are normalized internally (``w / sum(w)``), so the combined
    score stays in [0, 1] and the threshold keeps its meaning; uniform
    weights recover soft majority voting. A continuous algorithm
    (:class:`~ikn_library.algorithms.AntColonyOptimization`) searches
    real-valued weights; a binary one
    (:class:`~ikn_library.algorithms.BinaryAntColonyOptimization`) turns
    the same problem into ensemble pruning (0/1 = drop/keep a member).

    To avoid overfitting the weights, build ``P`` and ``y`` from data the
    ensemble was **not** trained on (a validation split), and report
    final performance on a third, untouched test set.

    Args:
        P: Probability matrix ``(n_samples, n_members)`` — see
            :func:`~ikn_library.ensemble.tree_proba_matrix`.
        y: Binary labels (0/1) of the same samples.
        metric: ``"accuracy"``, ``"f1"``, or a callable
            ``f(y_true, y_pred) -> float`` where higher is better.
        threshold: Cut-off on the combined score for predicting class 1.

    The fitness (minimized) is ``1 - metric``.
    """

    def __init__(self, P, y, metric="accuracy", threshold=0.5):
        self.P = np.asarray(P, dtype=float)
        self.y = np.asarray(y)
        if self.P.ndim != 2:
            raise ValueError("P must be 2-dimensional (n_samples, n_members)")
        if len(self.P) != len(self.y):
            raise ValueError("P and y must have the same number of samples")
        if not set(np.unique(self.y)) <= {0, 1}:
            raise ValueError("y must contain binary labels 0/1")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if callable(metric):
            self._metric = metric
        elif metric in _METRICS:
            self._metric = _METRICS[metric]
        else:
            raise ValueError(f'metric must be callable or one of {sorted(_METRICS)}')
        self.threshold = float(threshold)
        super().__init__(dimension=self.P.shape[1], lower=0.0, upper=1.0)

    def weights(self, x):
        """Normalized weight vector (non-negative, summing to 1).

        An all-zero input falls back to uniform weights — plain soft
        majority voting.
        """
        w = np.clip(np.asarray(x, dtype=float), 0.0, None)
        total = w.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.full(self.dimension, 1.0 / self.dimension)
        return w / total

    def scores(self, x, P=None):
        """Combined class-1 scores in [0, 1] for a weight vector."""
        P = self.P if P is None else np.asarray(P, dtype=float)
        return P @ self.weights(x)

    def predict(self, x, P=None):
        """Predicted labels for a weight vector (optionally on new ``P``)."""
        return (self.scores(x, P) > self.threshold).astype(int)

    def _evaluate(self, x):
        return 1.0 - self._metric(self.y, self.predict(x))
