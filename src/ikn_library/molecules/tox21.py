"""Load the Tox21 toxicity dataset (MoleculeNet version).

References:
    R. Huang et al., "Tox21 Challenge to build predictive models of
    nuclear receptor and stress response pathways as mediated by
    exposure to environmental chemicals and drugs," Frontiers in
    Environmental Science, 3:85, 2016.
    Z. Wu et al., "MoleculeNet: a benchmark for molecular machine
    learning," Chemical Science, 9(2), 513-530, 2018.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from ikn_library.molecules.base import MoleculeDataset

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


class Tox21Dataset(MoleculeDataset):
    """The Tox21 dataset: compounds (SMILES) and 12 toxicity assays.

    7,831 compounds from the Tox21 10K library, each labeled
    active (1) / inactive (0) on up to 12 assays: 7 nuclear-receptor
    pathways (``NR-*``) and 5 stress-response pathways (``SR-*``).
    Not every compound was tested in every assay, so labels contain
    missing values; :meth:`~ikn_library.molecules.base.MoleculeDataset.task`
    drops unlabeled molecules for the requested assay.
    """

    extra_columns = ("mol_id",)

    def __repr__(self):
        return (f"<Tox21Dataset: {len(self.smiles)} compounds x "
                f"{len(self.tasks)} assay tasks>")


def _download(cache_dir):
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ikn_library" / "molecules"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "tox21.csv.gz"
    if path.exists():
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(TOX21_URL, tmp)
    tmp.replace(path)
    return path


def load_tox21(source=None, cache_dir=None):
    """Load the Tox21 dataset into a :class:`Tox21Dataset`.

    Args:
        source: Path of a local ``tox21.csv[.gz]`` file. If omitted, the
            file is downloaded once from the MoleculeNet repository and
            cached under ``~/.ikn_library/molecules/``.
        cache_dir: Where the download is cached (when ``source`` is
            omitted).

    Example:
        >>> data = load_tox21()
        >>> smiles, y = data.task("NR-AhR")   # unlabeled compounds dropped
    """
    path = Path(source) if source is not None else _download(cache_dir)
    if not path.exists():
        raise ValueError(f"{source!r} is not an existing file")
    return Tox21Dataset(pd.read_csv(path))
