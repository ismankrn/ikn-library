"""Feature selection as a genuine two-objective problem."""

import numpy as np

from ikn_library.multiobjective.problem import MultiObjectiveProblem


class MultiObjectiveFeatureSelection(MultiObjectiveProblem):
    """Wrapper feature selection with accuracy and subset size kept apart.

    :class:`~ikn_library.problems.FeatureSelectionProblem` folds the two
    goals into one number with a weight ``alpha``, forcing the user to
    guess a trade-off before the search. Here they stay separate:

    - objective 0: ``1 - cv_score`` (model error), minimized
    - objective 1: ``n_selected / n_features`` (subset size), minimized

    Optimizing this with :class:`~ikn_library.algorithms.NSGA2` returns
    the whole trade-off curve in a single run, so the choice of "how
    many features" is made *after* seeing what each subset size buys.

    Requires scikit-learn (``pip install ikn-library[ml]``).

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Target vector.
        estimator: A scikit-learn estimator. Defaults to
            ``KNeighborsClassifier(n_neighbors=5)``.
        cv: Number of cross-validation folds.
        scoring: scikit-learn scoring name.
        threshold: Entries above this count as selected, in ``[0, 1)``
            (default 0.5). Only meaningful for continuous algorithms;
            binary ones emit 0/1.
    """

    def __init__(self, X, y, estimator=None, cv=5, scoring="accuracy",
                 threshold=0.5):
        try:
            from sklearn.model_selection import cross_val_score  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MultiObjectiveFeatureSelection requires scikit-learn; "
                "install it with: pip install ikn-library[ml]"
            ) from exc

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if self.X.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features)")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")
        if not 0.0 <= threshold < 1.0:
            raise ValueError(
                "threshold must be in [0, 1): solutions are bounded to "
                "[0, 1] and an entry counts as selected only when it is "
                "strictly greater, so threshold=1.0 selects nothing"
            )

        super().__init__(dimension=self.X.shape[1], n_objectives=2,
                         lower=0.0, upper=1.0,
                         objective_names=["error", "feature_fraction"])

        if estimator is None:
            from sklearn.neighbors import KNeighborsClassifier
            estimator = KNeighborsClassifier(n_neighbors=5)
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.threshold = float(threshold)

    def feature_mask(self, x):
        """Boolean mask of the selected features."""
        return np.asarray(x, dtype=float) > self.threshold

    def selected_features(self, x):
        """Indices of the selected features."""
        return np.flatnonzero(self.feature_mask(x))

    def _evaluate(self, x):
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        mask = self.feature_mask(x)
        n_selected = int(np.sum(mask))
        if n_selected == 0:
            # Worst error, but honestly the smallest possible subset.
            return np.array([1.0, 0.0])
        score = cross_val_score(
            clone(self.estimator), self.X[:, mask], self.y,
            cv=self.cv, scoring=self.scoring,
        ).mean()
        return np.array([1.0 - score, n_selected / self.dimension])
