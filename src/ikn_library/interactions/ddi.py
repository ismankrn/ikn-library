"""Drug-drug interaction benchmark: DrugBank DDI (DeepDDI).

References:
    J. Y. Ryu, H. U. Kim, and S. Y. Lee, "Deep learning improves
    prediction of drug-drug and drug-food interactions," PNAS, 115(18),
    E4304-E4311, 2018.
    File as harmonized by the Therapeutics Data Commons (Huang et al.,
    NeurIPS Datasets and Benchmarks, 2021); drug data from DrugBank
    (Wishart et al., Nucleic Acids Research, 46(D1), 2018).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ikn_library.molecules.base import fetch

DRUGBANK_DDI_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/4139573?format=original"
)
_CACHE = Path.home() / ".ikn_library" / "interactions"
_COLUMNS = {"ID1": "drug1_id", "ID2": "drug2_id", "Y": "interaction_type",
            "Map": "description", "X1": "smiles1", "X2": "smiles2"}


class DDIDataset:
    """Drug pairs labeled with their interaction type.

    191,808 pairs over 1,706 drugs, each labeled with one of **86
    interaction types** (e.g. "the metabolism of Drug2 can be decreased
    when combined with Drug1"), with the SMILES of both drugs included.

    Attributes:
        frame: ``DataFrame`` with columns ``drug1_id``, ``drug2_id``,
            ``smiles1``, ``smiles2``, ``interaction_type`` (1-86), and
            ``description`` (the human-readable interaction template).
    """

    def __init__(self, frame):
        missing = set(_COLUMNS.values()) - set(frame.columns)
        if missing:
            raise ValueError(f"not a DDI table: missing columns {sorted(missing)}")
        self.frame = frame

    @property
    def n_drugs(self):
        return pd.unique(self.frame[["drug1_id", "drug2_id"]].values.ravel()).size

    @property
    def interaction_types(self):
        """``DataFrame`` of the 86 types with their counts and descriptions."""
        return (self.frame.groupby("interaction_type")
                .agg(count=("interaction_type", "size"),
                     description=("description", "first"))
                .sort_values("count", ascending=False))

    def arrays(self):
        """All pairs as ``(smiles1, smiles2, interaction_type)``."""
        return (self.frame["smiles1"].to_numpy(),
                self.frame["smiles2"].to_numpy(),
                self.frame["interaction_type"].to_numpy(dtype=int))

    def binary_task(self, interaction_type, negative_ratio=1.0, seed=None):
        """One-vs-rest view of a single interaction type.

        Pairs of the requested type are positives; an equal-sized (by
        ``negative_ratio``) random sample of pairs of *other* types
        forms the negatives — turning the 86-class problem into the
        binary setting the rest of this library expects.

        Args:
            interaction_type: The type id (1-86).
            negative_ratio: Negatives per positive.
            seed: Random seed for sampling the negatives.

        Returns:
            tuple: ``(smiles1, smiles2, y)`` aligned arrays.
        """
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be >= 0")
        is_positive = (self.frame["interaction_type"] == interaction_type).to_numpy()
        if not is_positive.any():
            available = sorted(self.frame["interaction_type"].unique())
            raise ValueError(f"no pairs with interaction_type={interaction_type!r}; "
                             f"available: {available[:5]}...{available[-1]}")
        positive_idx = np.flatnonzero(is_positive)
        negative_pool = np.flatnonzero(~is_positive)
        n_negatives = min(round(negative_ratio * len(positive_idx)), len(negative_pool))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(negative_pool, n_negatives, replace=False)

        index = np.concatenate([positive_idx, chosen])
        y = np.concatenate([np.ones(len(positive_idx), dtype=int),
                            np.zeros(n_negatives, dtype=int)])
        subset = self.frame.iloc[index]
        return subset["smiles1"].to_numpy(), subset["smiles2"].to_numpy(), y

    def __repr__(self):
        return (f"<DDIDataset: {len(self.frame)} pairs, {self.n_drugs} drugs, "
                f"{self.frame['interaction_type'].nunique()} interaction types>")


def load_drugbank_ddi(source=None, cache_dir=None):
    """Load the DrugBank drug-drug interaction dataset.

    Args:
        source: Path of a local file; downloaded once (~40 MB) and
            cached otherwise.
        cache_dir: Cache location (default ``~/.ikn_library/interactions``).

    Example:
        >>> data = load_drugbank_ddi()
        >>> smiles1, smiles2, y = data.binary_task(47, seed=0)
    """
    path = fetch(DRUGBANK_DDI_URL, "drugbank_ddi.csv", source, cache_dir or _CACHE)
    frame = pd.read_csv(path).rename(columns=_COLUMNS)
    return DDIDataset(frame)
