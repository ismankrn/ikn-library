"""Preprocessing helpers for high-dimensional expression tables."""


def top_variance(X, k):
    """Keep the ``k`` probes (columns) with the highest variance.

    A standard unsupervised filter to shrink tens of thousands of probes
    down to a workable number before wrapper-based feature selection.

    Args:
        X: ``pandas.DataFrame`` of shape ``(n_samples, n_probes)``.
        k: Number of probes to keep.

    Returns:
        pandas.DataFrame: ``X`` restricted to the ``k`` most variable probes.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    variances = X.var(axis=0, skipna=True)
    keep = variances.nlargest(min(k, X.shape[1])).index
    return X.loc[:, keep]
