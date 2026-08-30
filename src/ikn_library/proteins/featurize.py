"""Sequence-based protein descriptors (pure numpy, no dependencies).

The CTD descriptors follow I. Dubchak, I. Muchnik, S. R. Holbrook, and
S.-H. Kim, "Prediction of protein folding class using global description
of amino acid sequence," PNAS, 92(19), 8700-8704, 1995.
"""

import warnings

import numpy as np
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Three-group amino-acid classifications for the CTD descriptors
#: (Dubchak et al., 1995; as used by PROFEAT / propy).
CTD_PROPERTIES = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}

METHODS = ("aac", "dpc", "ctd")


def _clean(sequence):
    return "".join(c for c in str(sequence).upper() if c in AMINO_ACIDS)


def _aac(sequence):
    counts = np.array([sequence.count(a) for a in AMINO_ACIDS], dtype=float)
    return counts / len(sequence)


def _dpc(sequence):
    index = {a: i for i, a in enumerate(AMINO_ACIDS)}
    counts = np.zeros((20, 20))
    for a, b in zip(sequence, sequence[1:]):
        counts[index[a], index[b]] += 1
    total = max(len(sequence) - 1, 1)
    return counts.ravel() / total


def _ctd(sequence):
    n = len(sequence)
    features = []
    for groups in CTD_PROPERTIES.values():
        encoded = np.zeros(n, dtype=int)
        for g, letters in enumerate(groups, start=1):
            for letter in letters:
                encoded[np.frombuffer(sequence.encode(), dtype=np.uint8)
                        == ord(letter)] = g
        # Composition: fraction of residues in each group.
        features.extend((encoded == g).mean() for g in (1, 2, 3))
        # Transition: fraction of adjacent pairs switching between groups.
        pairs = list(zip(encoded, encoded[1:]))
        total = max(n - 1, 1)
        for g1, g2 in ((1, 2), (1, 3), (2, 3)):
            features.append(sum(1 for a, b in pairs
                                if {a, b} == {g1, g2}) / total)
        # Distribution: relative position (%) of the first, 25%, 50%,
        # 75%, and last residue of each group.
        for g in (1, 2, 3):
            positions = np.flatnonzero(encoded == g) + 1
            if len(positions) == 0:
                features.extend([0.0] * 5)
                continue
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                rank = max(int(np.ceil(fraction * len(positions))), 1)
                features.append(positions[rank - 1] / n * 100.0)
    return np.array(features)


def _ctd_columns():
    columns = []
    for prop in CTD_PROPERTIES:
        columns.extend(f"{prop}_C{g}" for g in (1, 2, 3))
        columns.extend(f"{prop}_T{g1}{g2}" for g1, g2 in ((1, 2), (1, 3), (2, 3)))
        for g in (1, 2, 3):
            columns.extend(f"{prop}_D{g}_{p}" for p in (0, 25, 50, 75, 100))
    return columns


def featurize_protein(sequences, y=None, method="aac", on_invalid="drop"):
    """Turn protein sequences into a numeric feature table.

    Non-standard characters (anything outside the 20 amino-acid
    letters) are removed from each sequence before computing features.

    Args:
        sequences: Iterable of amino-acid sequences.
        y: Optional label array of the same length; rows dropped for
            invalid sequences are dropped from ``y`` too.
        method: One of:

            - ``"aac"`` — amino acid composition (20 features): the
              fraction of each amino acid in the sequence,
            - ``"dpc"`` — dipeptide composition (400 features): the
              fraction of each ordered amino-acid pair,
            - ``"ctd"`` — Composition / Transition / Distribution
              descriptors over 7 physicochemical properties
              (147 features; Dubchak et al., 1995).
        on_invalid: ``"drop"`` (default) removes sequences that contain
            no standard amino acids, with a warning; ``"raise"`` raises.

    Returns:
        ``pandas.DataFrame`` of shape ``(n_valid_sequences, n_features)``
        with named columns — or the tuple ``(X, y)`` when ``y`` is given.
    """
    sequences = list(sequences)
    if y is not None and len(y) != len(sequences):
        raise ValueError("sequences and y must have the same length")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    cleaned = [_clean(s) for s in sequences]
    valid = np.array([len(s) > 0 for s in cleaned])
    if not valid.all():
        n_bad = int((~valid).sum())
        if on_invalid == "raise":
            raise ValueError(f"{n_bad} sequences contain no standard amino acids")
        if on_invalid != "drop":
            raise ValueError('on_invalid must be "drop" or "raise"')
        warnings.warn(f"dropped {n_bad} invalid sequences", stacklevel=2)
        cleaned = [s for s in cleaned if s]

    if method == "aac":
        rows = [_aac(s) for s in cleaned]
        columns = [f"aac_{a}" for a in AMINO_ACIDS]
    elif method == "dpc":
        rows = [_dpc(s) for s in cleaned]
        columns = [f"dpc_{a}{b}" for a in AMINO_ACIDS for b in AMINO_ACIDS]
    else:
        rows = [_ctd(s) for s in cleaned]
        columns = _ctd_columns()

    X = pd.DataFrame(rows, columns=columns)
    if y is None:
        return X
    return X, np.asarray(y)[valid]
