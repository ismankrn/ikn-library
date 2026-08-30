"""Drug-target binding-affinity benchmarks: Davis and KIBA.

References:
    M. I. Davis et al., "Comprehensive analysis of kinase inhibitor
    selectivity," Nature Biotechnology, 29(11), 1046-1051, 2011.
    J. Tang et al., "Making sense of large-scale kinase inhibitor
    bioactivity data sets: a comparative and integrative analysis"
    (KIBA), Journal of Chemical Information and Modeling, 54(3), 2014.
    Files as harmonized by the Therapeutics Data Commons (Huang et al.,
    NeurIPS Datasets and Benchmarks, 2021).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ikn_library.molecules.base import fetch

DAVIS_URL = "https://dataverse.harvard.edu/api/access/datafile/5219748"
KIBA_URL = "https://dataverse.harvard.edu/api/access/datafile/5255037"

_CACHE = Path.home() / ".ikn_library" / "interactions"
_COLUMNS = {"ID1": "drug_id", "X1": "smiles", "ID2": "target_id",
            "X2": "sequence", "Y": "affinity"}


class DTIDataset:
    """Drug-target pairs with SMILES, protein sequences, and affinities.

    Attributes:
        frame: ``DataFrame`` with columns ``drug_id``, ``smiles``,
            ``target_id``, ``sequence``, ``affinity`` — one row per
            measured drug-target pair.
        name: Dataset name (``"davis"`` or ``"kiba"``).
    """

    def __init__(self, frame, name=""):
        missing = set(_COLUMNS.values()) - set(frame.columns)
        if missing:
            raise ValueError(f"not a DTI table: missing columns {sorted(missing)}")
        self.frame = frame
        self.name = name

    @property
    def n_drugs(self):
        return self.frame["drug_id"].nunique()

    @property
    def n_targets(self):
        return self.frame["target_id"].nunique()

    def arrays(self):
        """The pairs as three aligned arrays ``(smiles, sequences, y)``."""
        return (self.frame["smiles"].to_numpy(),
                self.frame["sequence"].to_numpy(),
                self.frame["affinity"].to_numpy(dtype=float))

    def __repr__(self):
        return (f"<DTIDataset {self.name!r}: {len(self.frame)} pairs, "
                f"{self.n_drugs} drugs x {self.n_targets} targets>")


def _load(url, filename, name, source, cache_dir):
    path = fetch(url, filename, source, cache_dir or _CACHE)
    frame = pd.read_csv(path, sep="\t").rename(columns=_COLUMNS)
    return DTIDataset(frame, name=name)


def load_davis(source=None, cache_dir=None, log_transform=True):
    """Load the Davis kinase-affinity dataset into a :class:`DTIDataset`.

    25,772 measured pairs of 68 kinase inhibitors x 442 kinases.

    Args:
        source: Path of a local ``davis.tab`` file; downloaded once
            (~20 MB) and cached otherwise.
        cache_dir: Cache location (default ``~/.ikn_library/interactions``).
        log_transform: When ``True`` (default), affinities are converted
            from Kd in nM to ``pKd = -log10(Kd * 1e-9)`` — the standard
            convention (DeepDTA and follow-ups); higher then means
            stronger binding.
    """
    data = _load(DAVIS_URL, "davis.tab", "davis", source, cache_dir)
    if log_transform:
        data.frame["affinity"] = -np.log10(
            data.frame["affinity"].astype(float) * 1e-9)
    return data


def load_kiba(source=None, cache_dir=None):
    """Load the KIBA bioactivity dataset into a :class:`DTIDataset`.

    117,657 pairs of 2,068 drugs x 229 targets, scored with the KIBA
    score (an integration of Ki, Kd, and IC50 measurements); used as-is.

    Args:
        source: Path of a local ``kiba.tab`` file; downloaded once
            (~92 MB) and cached otherwise.
        cache_dir: Cache location (default ``~/.ikn_library/interactions``).
    """
    return _load(KIBA_URL, "kiba.tab", "kiba", source, cache_dir)
