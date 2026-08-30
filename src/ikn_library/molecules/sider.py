"""Load the SIDER side-effect dataset (MoleculeNet version).

References:
    M. Kuhn, I. Letunic, L. J. Jensen, and P. Bork, "The SIDER database
    of drugs and side effects," Nucleic Acids Research, 44(D1), 2016.
    Z. Wu et al., "MoleculeNet: a benchmark for molecular machine
    learning," Chemical Science, 9(2), 513-530, 2018.
"""

import pandas as pd

from ikn_library.molecules.base import MoleculeDataset, fetch

SIDER_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"


class SIDERDataset(MoleculeDataset):
    """The SIDER dataset: drugs (SMILES) and their recorded side effects.

    1,427 marketed drugs, each labeled with 27 binary side-effect
    classes (MedDRA system-organ classes), as distributed by the
    MoleculeNet benchmark. Use
    :meth:`~ikn_library.molecules.base.MoleculeDataset.task` to get the
    SMILES and 0/1 labels of one side-effect class (e.g.
    ``data.task("hepato")`` for ``"Hepatobiliary disorders"``).
    """

    def __repr__(self):
        return (f"<SIDERDataset: {len(self.smiles)} drugs x "
                f"{len(self.tasks)} side-effect tasks>")


def load_sider(source=None, cache_dir=None):
    """Load the SIDER dataset into a :class:`SIDERDataset`.

    Args:
        source: Path of a local ``sider.csv[.gz]`` file. If omitted, the
            file is downloaded once from the MoleculeNet repository and
            cached under ``~/.ikn_library/molecules/``.
        cache_dir: Where the download is cached (when ``source`` is
            omitted).

    Example:
        >>> data = load_sider()
        >>> smiles, y = data.task("Hepatobiliary disorders")
    """
    path = fetch(SIDER_URL, "sider.csv.gz", source, cache_dir)
    return SIDERDataset(pd.read_csv(path))
