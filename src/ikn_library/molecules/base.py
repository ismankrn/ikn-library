"""Shared base for molecular datasets (SMILES + binary label columns)."""

import urllib.request
from pathlib import Path


def fetch(url, filename, source=None, cache_dir=None):
    """Return a local path for a dataset file, downloading it once if needed.

    ``source`` (a local path) short-circuits the download; otherwise the
    file is cached under ``cache_dir`` (default
    ``~/.ikn_library/molecules/``).
    """
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"{source!r} is not an existing file")
        return path
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ikn_library" / "molecules"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


class MoleculeDataset:
    """A table of molecules: a ``smiles`` column plus binary label columns.

    Attributes:
        frame: The full ``pandas.DataFrame``.
        smiles: Array of SMILES strings, one per molecule.
        labels: ``DataFrame`` of the label columns.
    """

    #: Non-label columns (besides ``smiles``) to exclude from ``labels``.
    extra_columns = ()

    def __init__(self, frame):
        if "smiles" not in frame.columns:
            raise ValueError(f"not a {type(self).__name__} table: no 'smiles' column")
        self.frame = frame
        self.smiles = frame["smiles"].to_numpy()
        drop = ["smiles"] + [c for c in self.extra_columns if c in frame.columns]
        self.labels = frame.drop(columns=drop)

    @property
    def tasks(self):
        """Names of the label columns."""
        return list(self.labels.columns)

    def _resolve_task(self, name):
        if name in self.labels.columns:
            return name
        matches = [t for t in self.tasks if name.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"no task matches {name!r}; available: {self.tasks}")
        raise KeyError(f"{name!r} is ambiguous; matches: {matches}")

    def task(self, name):
        """SMILES and binary labels for one task.

        ``name`` may be the exact column name or a case-insensitive
        substring that matches exactly one task. Molecules without a
        label for the task (missing values) are dropped, so the
        returned arrays may be shorter than the full dataset.

        Returns:
            tuple: ``(smiles, y)`` — SMILES strings and an integer 0/1
            label array of the same length.
        """
        column = self.labels[self._resolve_task(name)]
        mask = column.notna().to_numpy()
        return self.smiles[mask].copy(), column.to_numpy()[mask].astype(int)

    def __repr__(self):
        return (f"<{type(self).__name__}: {len(self.smiles)} molecules x "
                f"{len(self.tasks)} tasks>")
