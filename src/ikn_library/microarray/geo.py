"""Load NCBI GEO series-matrix files into ML-ready tables.

A GEO *series matrix* file bundles normalized expression values
(probes x samples) with per-sample metadata in one tab-separated text
file. :func:`load_geo` downloads it on demand (with local caching),
parses both blocks, and returns a :class:`GEODataset` whose ``X`` is a
samples x probes table ready for scikit-learn.
"""

import gzip
import re
import urllib.request
from pathlib import Path

import pandas as pd

_ACCESSION_RE = re.compile(r"^GSE\d+$", re.IGNORECASE)
_TABLE_BEGIN = "!series_matrix_table_begin"
_TABLE_END = "!series_matrix_table_end"


class GEODataset:
    """An ML-ready view of one GEO series.

    Attributes:
        X: ``pandas.DataFrame`` of shape ``(n_samples, n_probes)`` with
            GSM sample ids as the index — ready for scikit-learn.
        metadata: ``pandas.DataFrame`` indexed by the same sample ids,
            one column per sample characteristic (e.g. ``disease``,
            ``anatomic_location``), plus ``title`` and ``source_name``.
        accession: The GEO accession, when known (e.g. ``"GSE11223"``).
    """

    def __init__(self, X, metadata, accession=None):
        self.X = X
        self.metadata = metadata
        self.accession = accession

    def y(self, column):
        """Return one metadata column as labels aligned with ``X``.

        Args:
            column: Name of the metadata column (e.g. ``"disease"``).
        """
        if column not in self.metadata.columns:
            available = ", ".join(map(str, self.metadata.columns))
            raise KeyError(f"no metadata column {column!r}; available: {available}")
        return self.metadata.loc[self.X.index, column]

    def __repr__(self):
        name = self.accession or "GEODataset"
        return (f"<{name}: {self.X.shape[0]} samples x {self.X.shape[1]} probes, "
                f"{self.metadata.shape[1]} metadata columns>")


def matrix_url(accession):
    """Return the NCBI FTP URL of a series' matrix file.

    E.g. ``GSE11223`` maps to
    ``https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11223/matrix/GSE11223_series_matrix.txt.gz``.
    """
    acc = accession.upper()
    if not _ACCESSION_RE.match(acc):
        raise ValueError(f"not a GEO series accession: {accession!r}")
    stub = acc[:-3] + "nnn"
    return (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/"
            f"matrix/{acc}_series_matrix.txt.gz")


def _download(accession, cache_dir):
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ikn_library" / "geo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{accession.upper()}_series_matrix.txt.gz"
    if path.exists():
        return path
    url = matrix_url(accession)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.HTTPError as exc:
        raise OSError(
            f"could not download {url} ({exc}). Multi-platform series ship "
            f"several per-platform matrix files; download the one you need "
            f"from the GEO page and pass its local path to load_geo()."
        ) from exc
    tmp.replace(path)
    return path


def _strip_quotes(cell):
    cell = cell.strip()
    if len(cell) >= 2 and cell[0] == '"' and cell[-1] == '"':
        cell = cell[1:-1]
    return cell


def _parse(fileobj, accession=None):
    sample_lines = []  # (key, [values per sample])
    for line in fileobj:
        line = line.rstrip("\n")
        if line.startswith(_TABLE_BEGIN):
            break
        if line.startswith("!Sample_"):
            key, *values = line.split("\t")
            sample_lines.append((key[len("!Sample_"):], [_strip_quotes(v) for v in values]))
    else:
        raise ValueError("not a series matrix file: no expression table found")

    # The rest of the stream is the expression table (probes x samples).
    table = pd.read_csv(fileobj, sep="\t", index_col=0,
                        na_values=["null", "NULL"], low_memory=False)
    table = table[~table.index.astype(str).str.startswith(_TABLE_END)]
    X = table.apply(pd.to_numeric, errors="coerce").T

    fields = {}
    for key, values in sample_lines:
        fields.setdefault(key, []).append(values)
    sample_ids = fields.get("geo_accession", [list(X.index)])[0]

    records = {sid: {} for sid in sample_ids}
    for simple_key, column in [("title", "title"), ("source_name_ch1", "source_name")]:
        for values in fields.get(simple_key, []):
            for sid, cell in zip(sample_ids, values):
                records[sid][column] = cell or None
    for key, lines in fields.items():
        if not key.startswith("characteristics"):
            continue
        for values in lines:
            for sid, cell in zip(sample_ids, values):
                if not cell:
                    continue
                if ":" in cell:
                    name, value = cell.split(":", 1)
                    records[sid][name.strip()] = value.strip() or None
                else:
                    records[sid].setdefault("characteristics", cell)

    metadata = pd.DataFrame.from_dict(records, orient="index").reindex(X.index)
    return GEODataset(X, metadata, accession=accession)


def load_geo(source, cache_dir=None, dropna_threshold=None, impute=None):
    """Load a GEO series-matrix file into a :class:`GEODataset`.

    Args:
        source: A GEO series accession (e.g. ``"GSE11223"``, downloaded
            once and cached under ``~/.ikn_library/geo/``) or the path
            of a local ``*_series_matrix.txt[.gz]`` file.
        cache_dir: Where downloads are cached (accessions only).
        dropna_threshold: If given (0..1), drop probes whose fraction of
            missing values exceeds it.
        impute: ``"mean"`` or ``"median"`` — fill remaining missing
            values per probe; probes that are entirely missing are
            dropped.

    Returns:
        GEODataset: with ``X`` (samples x probes), ``metadata``, and
        the ``y(column)`` label helper.

    Example:
        >>> data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
        >>> X, y = data.X, data.y("disease")
    """
    accession = None
    path = Path(source)
    if path.exists():
        pass
    elif isinstance(source, str) and _ACCESSION_RE.match(source.strip()):
        accession = source.strip().upper()
        path = _download(accession, cache_dir)
    else:
        raise ValueError(
            f"{source!r} is neither an existing file nor a GEO series accession"
        )

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        data = _parse(f, accession=accession)

    if dropna_threshold is not None:
        if not 0.0 <= dropna_threshold <= 1.0:
            raise ValueError("dropna_threshold must be in [0, 1]")
        keep = data.X.isna().mean(axis=0) <= dropna_threshold
        data.X = data.X.loc[:, keep]
    if impute is not None:
        if impute not in ("mean", "median"):
            raise ValueError('impute must be "mean" or "median"')
        data.X = data.X.dropna(axis=1, how="all")
        fill = data.X.mean(axis=0) if impute == "mean" else data.X.median(axis=0)
        data.X = data.X.fillna(fill)
    return data
