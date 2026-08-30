"""The Yamanishi (2008) drug-target interaction benchmarks.

Reference:
    Y. Yamanishi, M. Araki, A. Gutteridge, W. Honda, and M. Kanehisa,
    "Prediction of drug-target interaction networks from the
    integration of chemical and genomic spaces," Bioinformatics,
    24(13), i232-i240, 2008.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ikn_library.molecules.base import fetch

_BASE_URL = "http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget"
_CACHE = Path.home() / ".ikn_library" / "interactions"

SUBSETS = {
    "enzyme": "e",
    "ion_channel": "ic",
    "gpcr": "gpcr",
    "nuclear_receptor": "nr",
}


class YamanishiDataset:
    """One Yamanishi drug-target interaction network.

    The four classic benchmarks (enzyme, ion channel, GPCR, nuclear
    receptor) list the known interacting pairs between KEGG drugs
    (``D`` numbers) and human target proteins (``hsa`` ids). Only
    positive pairs are recorded; negatives must be sampled from the
    unobserved pairs — see :meth:`pairs`.

    Attributes:
        positives: ``DataFrame`` with columns ``target_id``, ``drug_id``.
        subset: The benchmark name.
    """

    def __init__(self, positives, subset=""):
        self.positives = positives
        self.subset = subset
        self.drugs = np.sort(positives["drug_id"].unique())
        self.targets = np.sort(positives["target_id"].unique())

    def interaction_matrix(self):
        """The full 0/1 interaction matrix (drugs x targets)."""
        matrix = pd.DataFrame(0, index=self.drugs, columns=self.targets, dtype=int)
        for target, drug in self.positives.itertuples(index=False):
            matrix.loc[drug, target] = 1
        return matrix

    def pairs(self, negative_ratio=1.0, seed=None):
        """Positive pairs plus randomly sampled negative pairs.

        Negatives are drawn (without replacement) from the drug-target
        combinations not recorded as interacting — the standard
        assumption in DTI benchmarking, with the caveat that an
        "unobserved" pair is not guaranteed to be truly non-interacting.

        Args:
            negative_ratio: Number of negatives per positive.
            seed: Random seed for the negative sampling.

        Returns:
            tuple: ``(drug_ids, target_ids, y)`` aligned arrays with
            ``y`` equal to 1 for known interactions, 0 for sampled
            negatives.
        """
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be >= 0")
        matrix = self.interaction_matrix().to_numpy()
        pos_drug, pos_target = np.nonzero(matrix)
        neg_drug, neg_target = np.nonzero(matrix == 0)
        n_negatives = min(round(negative_ratio * len(pos_drug)), len(neg_drug))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(neg_drug), n_negatives, replace=False)

        drug_ids = np.concatenate([self.drugs[pos_drug], self.drugs[neg_drug[chosen]]])
        target_ids = np.concatenate([self.targets[pos_target],
                                     self.targets[neg_target[chosen]]])
        y = np.concatenate([np.ones(len(pos_drug), dtype=int),
                            np.zeros(n_negatives, dtype=int)])
        return drug_ids, target_ids, y

    def __repr__(self):
        return (f"<YamanishiDataset {self.subset!r}: "
                f"{len(self.positives)} interactions, "
                f"{len(self.drugs)} drugs x {len(self.targets)} targets>")


def load_yamanishi(subset="enzyme", source=None, cache_dir=None):
    """Load one Yamanishi benchmark into a :class:`YamanishiDataset`.

    Args:
        subset: ``"enzyme"``, ``"ion_channel"``, ``"gpcr"``, or
            ``"nuclear_receptor"``.
        source: Path of a local pair file (tab-separated
            ``hsa:<id>\\t<D-number>`` lines); downloaded once from the
            authors' site and cached otherwise.
        cache_dir: Cache location (default ``~/.ikn_library/interactions``).

    Example:
        >>> data = load_yamanishi("nuclear_receptor")
        >>> drug_ids, target_ids, y = data.pairs(negative_ratio=1.0, seed=42)
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {sorted(SUBSETS)}")
    code = SUBSETS[subset]
    path = fetch(f"{_BASE_URL}/bind_orfhsa_drug_{code}.txt",
                 f"yamanishi_{subset}.txt", source, cache_dir or _CACHE)
    positives = pd.read_csv(path, sep="\t", header=None,
                            names=["target_id", "drug_id"])
    return YamanishiDataset(positives, subset=subset)
