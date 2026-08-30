"""Leakage-aware train/test splits for paired (drug, target) data."""

import numpy as np

MODES = ("random", "drug", "target", "both")


def _split_entities(entities, test_size, rng):
    unique = np.unique(entities)
    n_test = round(test_size * len(unique))
    n_test = max(1, min(n_test, len(unique) - 1))
    held_out = rng.choice(unique, n_test, replace=False)
    return np.isin(entities, held_out)


def cold_split(drug_ids, target_ids, test_size=0.2, mode="drug", seed=None):
    """Split pairs into train/test indices without entity leakage.

    In paired data each drug (and each target) appears in many rows, so
    a random split lets the *same* drug show up in both train and test
    — the model can memorize per-drug tendencies instead of learning
    the interaction, which inflates scores. A **cold** split holds out
    whole entities instead of individual pairs.

    Args:
        drug_ids: Drug identifier per pair.
        target_ids: Target identifier per pair.
        test_size: Fraction of *entities* (or of pairs, when
            ``mode="random"``) to hold out.
        mode: Which question the split answers:

            - ``"drug"`` (default) — cold-drug: test drugs never appear
              in training ("what does this **new drug** bind?"),
            - ``"target"`` — cold-target: test proteins never appear in
              training ("what binds this **new protein**?"),
            - ``"both"`` — neither the drug nor the target of a test
              pair appears in training (hardest, smallest test set),
            - ``"random"`` — ordinary per-pair split, for comparison.
        seed: Random seed.

    Returns:
        tuple: ``(train_index, test_index)`` arrays of row positions.

    Note:
        With ``mode="both"`` the test set holds only the pairs whose
        drug *and* target are both held out, so it is much smaller than
        ``test_size`` would suggest; training drops every pair touching
        a held-out entity.

    Example:
        >>> train, test = cold_split(drug_ids, target_ids, mode="drug", seed=0)
        >>> X_train, X_test = X[train], X[test]
    """
    drug_ids = np.asarray(drug_ids)
    target_ids = np.asarray(target_ids)
    if len(drug_ids) != len(target_ids):
        raise ValueError("drug_ids and target_ids must have the same length")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n_pairs = len(drug_ids)

    if mode == "random":
        order = rng.permutation(n_pairs)
        n_test = max(1, min(round(test_size * n_pairs), n_pairs - 1))
        return np.sort(order[n_test:]), np.sort(order[:n_test])

    cold_drug = _split_entities(drug_ids, test_size, rng) if mode in ("drug", "both") else None
    cold_target = _split_entities(target_ids, test_size, rng) if mode in ("target", "both") else None

    if mode == "drug":
        test_mask, dropped = cold_drug, cold_drug
    elif mode == "target":
        test_mask, dropped = cold_target, cold_target
    else:  # both: test pairs are cold in *both* entities; training keeps
        # only pairs whose drug and target are both seen.
        test_mask = cold_drug & cold_target
        dropped = cold_drug | cold_target

    return np.flatnonzero(~dropped), np.flatnonzero(test_mask)
