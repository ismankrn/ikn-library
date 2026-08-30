"""Additional MoleculeNet classification datasets: BBBP, ClinTox, HIV.

Reference:
    Z. Wu et al., "MoleculeNet: a benchmark for molecular machine
    learning," Chemical Science, 9(2), 513-530, 2018.
"""

import pandas as pd

from ikn_library.molecules.base import MoleculeDataset, fetch

_BUCKET = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


class BBBPDataset(MoleculeDataset):
    """BBBP: blood-brain barrier penetration.

    2,050 compounds with one binary task, ``p_np`` (1 = penetrates the
    blood-brain barrier). The compound ``name`` and ``num`` columns are
    kept in ``frame`` but excluded from the labels.
    """

    extra_columns = ("num", "name")


class ClinToxDataset(MoleculeDataset):
    """ClinTox: clinical-trial toxicity vs FDA approval.

    1,484 compounds with two binary tasks: ``FDA_APPROVED`` and
    ``CT_TOX`` (failed clinical trials for toxicity reasons).
    """


class HIVDataset(MoleculeDataset):
    """HIV: inhibition of HIV replication.

    41,127 compounds with one binary task, ``HIV_active`` — only ~3.5%
    of compounds are active, making this a prime case study for
    imbalanced-data techniques such as
    :class:`~ikn_library.sampling.UndersamplingProblem`. The raw
    three-way screening outcome (``activity``: CI/CM/CA) is kept in
    ``frame`` but excluded from the labels.
    """

    extra_columns = ("activity",)


def load_bbbp(source=None, cache_dir=None):
    """Load the BBBP dataset into a :class:`BBBPDataset`.

    Example:
        >>> data = load_bbbp()
        >>> smiles, y = data.task("p_np")
    """
    path = fetch(f"{_BUCKET}/BBBP.csv", "BBBP.csv", source, cache_dir)
    return BBBPDataset(pd.read_csv(path))


def load_clintox(source=None, cache_dir=None):
    """Load the ClinTox dataset into a :class:`ClinToxDataset`.

    Example:
        >>> data = load_clintox()
        >>> smiles, y = data.task("CT_TOX")
    """
    path = fetch(f"{_BUCKET}/clintox.csv.gz", "clintox.csv.gz", source, cache_dir)
    return ClinToxDataset(pd.read_csv(path))


def load_hiv(source=None, cache_dir=None):
    """Load the HIV dataset into a :class:`HIVDataset`.

    Example:
        >>> data = load_hiv()
        >>> smiles, y = data.task("HIV_active")
    """
    path = fetch(f"{_BUCKET}/HIV.csv", "HIV.csv", source, cache_dir)
    return HIVDataset(pd.read_csv(path))
