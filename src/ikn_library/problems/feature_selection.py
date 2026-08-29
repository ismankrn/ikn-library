"""Wrapper-based feature selection as an optimization problem."""

import numpy as np

from ikn_library.problems.problem import Problem


class FeatureSelectionProblem(Problem):
    """Wrapper feature selection: pick the feature subset that maximizes
    a cross-validated model score while keeping the subset small.

    Solutions are vectors in ``[0, 1]``; entries above ``threshold`` mark
    selected features, so both binary algorithms (which emit 0/1 bits)
    and continuous algorithms can optimize this problem.

    The fitness (minimized) is::

        alpha * (1 - cv_score) + (1 - alpha) * n_selected / n_features

    Requires scikit-learn (``pip install ikn-library[ml]``).

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target vector of shape ``(n_samples,)``.
        estimator: A scikit-learn estimator. Defaults to
            ``KNeighborsClassifier(n_neighbors=5)``, a common choice in
            wrapper feature-selection studies.
        cv: Number of cross-validation folds.
        scoring: scikit-learn scoring name (e.g. ``"accuracy"``, ``"f1"``).
        alpha: Trade-off between score quality and subset size, in [0, 1].
            Values near 1 prioritize the model score.
        threshold: Cut-off above which a variable counts as selected.
    """

    def __init__(self, X, y, estimator=None, cv=5, scoring="accuracy",
                 alpha=0.99, threshold=0.5):
        try:
            from sklearn.model_selection import cross_val_score  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FeatureSelectionProblem requires scikit-learn; "
                "install it with: pip install ikn-library[ml]"
            ) from exc

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if self.X.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features)")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        super().__init__(dimension=self.X.shape[1], lower=0.0, upper=1.0)

        if estimator is None:
            from sklearn.neighbors import KNeighborsClassifier
            estimator = KNeighborsClassifier(n_neighbors=5)
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.alpha = float(alpha)
        self.threshold = float(threshold)

    def feature_mask(self, x):
        """Boolean mask of selected features for a solution vector."""
        return np.asarray(x, dtype=float) > self.threshold

    def selected_features(self, x):
        """Indices of the features selected by a solution vector."""
        return np.flatnonzero(self.feature_mask(x))

    def _evaluate(self, x):
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        mask = self.feature_mask(x)
        n_selected = int(np.sum(mask))
        if n_selected == 0:
            return 1.0  # worst possible fitness: nothing selected
        score = cross_val_score(
            clone(self.estimator), self.X[:, mask], self.y,
            cv=self.cv, scoring=self.scoring,
        ).mean()
        return (self.alpha * (1.0 - score)
                + (1.0 - self.alpha) * n_selected / self.dimension)
