"""Metaheuristic undersampling as an optimization problem.

Reference:
    S. Garcia and F. Herrera, "Evolutionary undersampling for
    classification with imbalanced datasets: proposals and taxonomy,"
    Evolutionary Computation, 17(3), 275-306, 2009.
"""

import copy
import zlib

import numpy as np

from ikn_library.problems.problem import Problem


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def _recall(y_true, y_pred, cls):
    mask = y_true == cls
    return float(np.mean(y_pred[mask] == cls)) if mask.any() else 0.0


class UndersamplingProblem(Problem):
    """Select which majority-class samples to keep in an imbalanced dataset.

    Random undersampling balances a dataset by discarding majority-class
    samples blindly. Here the choice becomes an optimization problem
    (evolutionary undersampling, Garcia & Herrera, 2009): a bit string
    over the majority-class training samples marks which ones to keep,
    the minority class is always kept in full, and the fitness is the
    performance of a model trained on the reduced training set.

    **Constraint (repair):** every candidate is repaired so that exactly
    ``target = round(target_ratio * n_minority)`` majority samples are
    kept — excess bits are switched off (or missing ones switched on) at
    random, deterministically per candidate, so the evaluation budget is
    only spent on subsets with the desired class ratio.

    **Evaluation protocol:** undersampling must only touch the training
    data. The model is trained on ``X_train`` reduced by the candidate
    subset and scored on the untouched — still imbalanced —
    ``X_val`` / ``y_val``. Report final results on a third test set.

    Args:
        X_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Binary training labels.
        X_val: Validation features (never undersampled).
        y_val: Binary validation labels.
        estimator: Model with ``fit``/``predict``. Defaults to
            ``KNeighborsClassifier(n_neighbors=5)`` (requires
            scikit-learn); any object with the two methods works.
        target_ratio: Kept-majority to minority ratio; ``1.0`` balances
            the classes exactly.
        metric: ``"f1"`` (default; positive class = minority),
            ``"balanced_accuracy"``, ``"accuracy"``, or a callable
            ``f(y_true, y_pred) -> float`` where higher is better.
        threshold: Cut-off above which a solution entry counts as
            "keep", so continuous algorithms can optimize this problem
            too.

    The fitness (minimized) is ``1 - metric``.
    """

    def __init__(self, X_train, y_train, X_val, y_val, estimator=None,
                 target_ratio=1.0, metric="f1", threshold=0.5):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train must have the same length")
        if len(self.X_val) != len(self.y_val):
            raise ValueError("X_val and y_val must have the same length")
        classes, counts = np.unique(self.y_train, return_counts=True)
        if len(classes) != 2:
            raise ValueError("undersampling requires exactly 2 classes in y_train")
        if target_ratio <= 0:
            raise ValueError("target_ratio must be > 0")

        self.minority_class = classes[np.argmin(counts)]
        self.majority_class = classes[np.argmax(counts)]
        self.majority_indices = np.flatnonzero(self.y_train == self.majority_class)
        self.minority_indices = np.flatnonzero(self.y_train == self.minority_class)
        n_minority = len(self.minority_indices)
        n_majority = len(self.majority_indices)
        self.target = max(1, min(round(target_ratio * n_minority), n_majority))

        if callable(metric):
            self._metric = metric
        elif metric == "f1":
            self._metric = self._f1_minority
        elif metric == "balanced_accuracy":
            self._metric = self._balanced_accuracy
        elif metric == "accuracy":
            self._metric = _accuracy
        else:
            raise ValueError('metric must be callable or one of '
                             "['accuracy', 'balanced_accuracy', 'f1']")

        if estimator is None:
            try:
                from sklearn.neighbors import KNeighborsClassifier
            except ImportError as exc:
                raise ImportError(
                    "the default estimator requires scikit-learn "
                    "(pip install ikn-library[ml]); or pass any object "
                    "with fit/predict methods"
                ) from exc
            estimator = KNeighborsClassifier(n_neighbors=5)
        self.estimator = estimator
        self.threshold = float(threshold)
        super().__init__(dimension=n_majority, lower=0.0, upper=1.0)

    def _f1_minority(self, y_true, y_pred):
        pos = self.minority_class
        tp = np.sum((y_pred == pos) & (y_true == pos))
        fp = np.sum((y_pred == pos) & (y_true != pos))
        fn = np.sum((y_pred != pos) & (y_true == pos))
        denominator = 2 * tp + fp + fn
        return float(2 * tp / denominator) if denominator > 0 else 0.0

    def _balanced_accuracy(self, y_true, y_pred):
        return 0.5 * (_recall(y_true, y_pred, self.minority_class)
                      + _recall(y_true, y_pred, self.majority_class))

    def majority_mask(self, x):
        """Repaired keep-mask over the majority samples (exactly ``target`` ones).

        The repair flips randomly chosen bits, but deterministically per
        solution vector — the same ``x`` always yields the same subset.
        """
        x = np.asarray(x, dtype=float)
        mask = x > self.threshold
        excess = int(mask.sum()) - self.target
        if excess != 0:
            rng = np.random.default_rng(zlib.crc32(x.tobytes()))
            if excess > 0:
                turn_off = rng.choice(np.flatnonzero(mask), excess, replace=False)
                mask[turn_off] = False
            else:
                turn_on = rng.choice(np.flatnonzero(~mask), -excess, replace=False)
                mask[turn_on] = True
        return mask

    def selected_indices(self, x):
        """Row indices into ``X_train`` of the reduced training set
        (all minority samples plus the kept majority samples), sorted."""
        kept_majority = self.majority_indices[self.majority_mask(x)]
        return np.sort(np.concatenate([self.minority_indices, kept_majority]))

    def resampled_data(self, x):
        """The reduced training set ``(X_res, y_res)`` for a solution."""
        indices = self.selected_indices(x)
        return self.X_train[indices], self.y_train[indices]

    def _clone_estimator(self):
        try:
            from sklearn.base import clone
            return clone(self.estimator)
        except (ImportError, TypeError):
            return copy.deepcopy(self.estimator)

    def _evaluate(self, x):
        X_res, y_res = self.resampled_data(x)
        model = self._clone_estimator()
        model.fit(X_res, y_res)
        return 1.0 - self._metric(self.y_val, np.asarray(model.predict(self.X_val)))
