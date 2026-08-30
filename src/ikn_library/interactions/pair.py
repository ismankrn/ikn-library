"""Combine two per-entity feature tables into pair features."""

import numpy as np
import pandas as pd

#: Symmetric combinations give the same features for (A, B) and (B, A) —
#: the right choice when the relation itself is symmetric (drug-drug).
SYMMETRIC_METHODS = ("sum", "product", "absdiff", "mean", "max")
METHODS = (*SYMMETRIC_METHODS, "concat")


def pair_features(X1, X2, method="concat"):
    """Combine the features of both partners of a pair.

    Args:
        X1: Features of the first entity, ``(n_pairs, n_features)``.
        X2: Features of the second entity. For the symmetric methods it
            must have the same shape and columns as ``X1``.
        method: How to combine them:

            - ``"concat"`` (default) — side-by-side, giving
              ``2 * n_features`` columns. Asymmetric: appropriate for
              **drug-target** pairs, where the two sides are different
              kinds of entity, or when the relation has a direction.
            - ``"sum"``, ``"product"``, ``"absdiff"``, ``"mean"``,
              ``"max"`` — element-wise and **symmetric**: the result is
              identical for ``(A, B)`` and ``(B, A)``, which is what an
              undirected **drug-drug** interaction requires.

    Returns:
        ``pandas.DataFrame`` with descriptive column names.

    Example:
        >>> X = pair_features(featurize(smiles1), featurize(smiles2),
        ...                   method="sum")     # drug-drug, symmetric
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    X1 = pd.DataFrame(X1).reset_index(drop=True)
    X2 = pd.DataFrame(X2).reset_index(drop=True)
    if len(X1) != len(X2):
        raise ValueError("X1 and X2 must have the same number of rows")

    if method == "concat":
        left = X1.add_suffix("_1")
        right = X2.add_suffix("_2")
        return pd.concat([left, right], axis=1)

    if X1.shape[1] != X2.shape[1]:
        raise ValueError(
            f'method={method!r} is element-wise and needs matching feature '
            f"counts, got {X1.shape[1]} and {X2.shape[1]}"
        )
    a, b = X1.to_numpy(dtype=float), X2.to_numpy(dtype=float)
    combined = {
        "sum": lambda: a + b,
        "product": lambda: a * b,
        "absdiff": lambda: np.abs(a - b),
        "mean": lambda: (a + b) / 2.0,
        "max": lambda: np.maximum(a, b),
    }[method]()
    columns = [f"{method}_{c}" for c in X1.columns]
    return pd.DataFrame(combined, columns=columns)
