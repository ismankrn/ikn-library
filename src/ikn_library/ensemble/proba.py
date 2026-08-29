"""Extract per-member probability matrices from fitted ensembles."""

import numpy as np


def tree_proba_matrix(ensemble, X):
    """Build the probability matrix ``P`` of shape ``(n_samples, n_members)``.

    ``P[i, j]`` is member ``j``'s predicted probability that sample ``i``
    belongs to the positive class (the second entry of the member's
    ``classes_``, per scikit-learn convention for binary problems).

    Args:
        ensemble: A fitted scikit-learn ensemble exposing ``estimators_``
            (e.g. ``RandomForestClassifier``, ``BaggingClassifier``), or
            any iterable of fitted classifiers with ``predict_proba`` —
            so heterogeneous ensembles (SVM + KNN + RF, ...) work too.
        X: Feature matrix to predict on.

    Returns:
        numpy.ndarray: ``P`` with values in [0, 1].
    """
    members = getattr(ensemble, "estimators_", ensemble)
    members = list(members)
    if not members:
        raise ValueError("no fitted ensemble members found")
    return np.column_stack(
        [np.asarray(member.predict_proba(X))[:, 1] for member in members]
    )
