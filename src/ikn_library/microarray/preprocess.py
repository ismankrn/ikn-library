"""Preprocessing helpers for high-dimensional expression tables.

All functions take and return a ``pandas.DataFrame`` of shape
``(n_samples, n_probes)`` — the orientation produced by
:func:`~ikn_library.microarray.load_geo`.
"""

import numpy as np
import pandas as pd


def log2_transform(X, offset=1.0):
    """Apply ``log2(x + offset)`` to raw (linear-scale) intensities.

    Only for data still on a linear scale; GEO series matrices are often
    already log-transformed (values roughly in [-15, 15], possibly
    negative) and should not be transformed again.

    Args:
        X: Expression table (samples x probes).
        offset: Added before the log to keep values positive.

    Raises:
        ValueError: If any ``x + offset`` is not strictly positive.
    """
    shifted = X + offset
    if (shifted <= 0).any().any():
        raise ValueError(
            "log2_transform requires x + offset > 0 for all values; "
            "negative values usually mean the data is already log-scale"
        )
    return np.log2(shifted)


def quantile_normalize(X):
    """Quantile normalization: give every sample an identical distribution.

    The de-facto standard for making microarray samples comparable
    (Bolstad et al., 2003). Each sample's values are replaced by the
    mean of all samples' values at the same rank; ties receive the
    interpolated value of their average rank.

    Args:
        X: Expression table (samples x probes) without missing values —
            apply ``impute=`` in :func:`~ikn_library.microarray.load_geo`
            first.

    Raises:
        ValueError: If ``X`` contains missing values.
    """
    if X.isna().any().any():
        raise ValueError("quantile_normalize requires complete data; impute first")
    values = X.to_numpy(dtype=float)
    mean_distribution = np.sort(values, axis=1).mean(axis=0)
    ranks = X.rank(axis=1, method="average").to_numpy()
    normalized = np.interp(ranks, np.arange(1, values.shape[1] + 1), mean_distribution)
    return pd.DataFrame(normalized, index=X.index, columns=X.columns)


def zscore(X):
    """Standardize each probe to zero mean and unit variance.

    The usual feature scaling before distance-based models (KNN, SVM).
    Probes with zero variance become all-zero columns.

    Args:
        X: Expression table (samples x probes).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (X - mean) / std


def median_center(X):
    """Subtract each sample's median from its values.

    A light per-sample normalization that removes global intensity
    shifts between arrays, common for log-ratio data.

    Args:
        X: Expression table (samples x probes).
    """
    return X.sub(X.median(axis=1), axis=0)


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
