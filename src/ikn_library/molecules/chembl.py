"""Bioactivity datasets from ChEMBL, ready for QSAR modelling."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPException
from pathlib import Path

import numpy as np

#: ChEMBL's public REST endpoint for bioactivity records.
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

#: Endpoint reporting which ChEMBL release the API is serving.
CHEMBL_STATUS_API = "https://www.ebi.ac.uk/chembl/api/data/status.json"

#: Fields requested from the API and kept in the cached table.
CHEMBL_FIELDS = (
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "target_organism",
)

_PAGE = 1000
_RETRIES = 4
_BACKOFF = 2.0      # seconds, doubled after each failure
_COURTESY = 0.2     # pause between pages, so a long paging run stays polite


def _get(url, timeout):
    """One GET with retries, because the public API rate-limits long runs.

    A paging run over a well-studied target makes dozens of requests;
    EBI will occasionally close the connection or answer 429/503. Those
    are transient, so they are retried with an exponential backoff
    rather than surfaced as a failed download.
    """
    delay = _BACKOFF
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, HTTPException, ConnectionError) as exc:
            status = getattr(exc, "code", None)
            fatal = status is not None and status not in (429, 500, 502, 503, 504)
            if fatal or attempt == _RETRIES - 1:
                raise RuntimeError(
                    f"ChEMBL request failed after {attempt + 1} attempt(s): {exc}. "
                    "The public API rate-limits sustained paging; retry later, "
                    "or pass `max_records` to fetch less."
                ) from exc
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


class ChemblDataset:
    """Cleaned bioactivity records for one ChEMBL target.

    Attributes:
        target_id: The ChEMBL target identifier the records come from.
        activity_type: The assay endpoint (``"IC50"``, ``"Ki"``, ...).
        frame: One row per molecule — ``smiles``, ``pchembl``,
            ``n_measurements`` and ``molecule_chembl_id``.
        smiles: Array of SMILES strings.
        y: Array of pChEMBL values, the regression target.
        funnel: ``dict`` recording how many records each cleaning step
            removed, so the drop from raw records to molecules is
            auditable rather than silent.
    """

    def __init__(self, target_id, activity_type, frame, funnel):
        self.target_id = target_id
        self.activity_type = activity_type
        self.frame = frame
        self.funnel = funnel
        self.smiles = frame["smiles"].to_numpy()
        self.y = frame["pchembl"].to_numpy(dtype=float)

    def report(self):
        """Human-readable version of :attr:`funnel`, one line per step."""
        width = max(len(step) for step in self.funnel)
        return "\n".join(f"{step:<{width}} : {count}"
                         for step, count in self.funnel.items())

    def __repr__(self):
        return (f"<ChemblDataset {self.target_id} {self.activity_type}: "
                f"{len(self.smiles)} molecules, pChEMBL "
                f"{self.y.min():.2f}-{self.y.max():.2f}>")


def _cache_path(target_id, activity_type, cache_dir):
    cache_dir = (Path(cache_dir) if cache_dir
                 else Path.home() / ".ikn_library" / "chembl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{target_id}_{activity_type}.csv.gz"


def _download(target_id, activity_type, max_records, timeout):
    """Page through the API, returning a list of record dicts."""
    records, offset = [], 0
    while True:
        query = urllib.parse.urlencode({
            "target_chembl_id": target_id,
            "standard_type": activity_type,
            "limit": _PAGE,
            "offset": offset,
        })
        payload = _get(f"{CHEMBL_API}?{query}", timeout)
        batch = payload["activities"]
        records.extend({field: record.get(field) for field in CHEMBL_FIELDS}
                       for record in batch)
        total = payload["page_meta"]["total_count"]
        offset += _PAGE
        if offset >= total or not batch:
            break
        if max_records is not None and len(records) >= max_records:
            break
        time.sleep(_COURTESY)
    return records[:max_records] if max_records else records


def fetch_chembl_activities(target_id, activity_type="IC50", cache_dir=None,
                            refresh=False, max_records=None, timeout=120):
    """Raw ChEMBL activity records for a target, as a ``DataFrame``.

    Downloads once and caches under ``~/.ikn_library/chembl/``; the
    cache holds the *unfiltered* records, so changing how
    :func:`load_chembl_target` cleans them never re-downloads.

    Args:
        target_id: ChEMBL target identifier, e.g. ``"CHEMBL279"``.
        activity_type: Assay endpoint to request (``"IC50"`` default).
        cache_dir: Override the cache location.
        refresh: Re-download even when a cached copy exists.
        max_records: Stop after roughly this many records. Useful for a
            quick look at a target with tens of thousands of them.
        timeout: Seconds to wait for each API request.
    """
    import pandas as pd

    path = _cache_path(target_id, activity_type, cache_dir)
    if refresh or not path.exists():
        records = _download(target_id, activity_type, max_records, timeout)
        if not records:
            raise ValueError(
                f"ChEMBL returned no {activity_type} records for {target_id!r}; "
                "check the identifier and the activity type"
            )
        frame = pd.DataFrame(records, columns=list(CHEMBL_FIELDS))
        tmp = path.with_suffix(".part")
        frame.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)
    return pd.read_csv(path)


def load_chembl_target(target_id, activity_type="IC50", relation="=",
                       units="nM", organism=None, aggregate="median",
                       cache_dir=None, refresh=False, max_records=None,
                       timeout=120):
    """Download and clean a ChEMBL target's bioactivity data for QSAR.

    The defaults apply the filtering that is standard in the QSAR
    literature, and every step is counted in
    :attr:`ChemblDataset.funnel` rather than applied silently:

    1. drop records with no SMILES,
    2. keep only exact measurements (``standard_relation`` of ``"="``),
       since ``>`` and ``<`` are censored values that a regression
       cannot use as if they were numbers,
    3. keep one unit system so the values are comparable,
    4. drop records with no ``pchembl_value`` (ChEMBL's
       :math:`-\\log_{10}` of the molar activity),
    5. collapse repeated measurements of the same molecule.

    Args:
        target_id: ChEMBL target identifier, e.g. ``"CHEMBL279"``
            (VEGFR2). Find one by searching the ChEMBL web interface.
        activity_type: ``"IC50"`` (default), ``"Ki"``, ``"EC50"``, ...
        relation: Keep only this ``standard_relation``; ``None`` keeps
            censored records too.
        units: Keep only this ``standard_units``; ``None`` keeps all.
        organism: Optional ``target_organism`` filter, e.g.
            ``"Homo sapiens"`` — some targets carry records from
            several species.
        aggregate: How to collapse repeated measurements of one
            molecule: ``"median"`` (default), ``"mean"``, ``"min"``,
            ``"max"``, or ``None`` to keep every measurement as its own
            row.
        cache_dir: Override the cache location.
        refresh: Re-download even when a cached copy exists.
        max_records: Cap on records fetched (see
            :func:`fetch_chembl_activities`).
        timeout: Seconds to wait for each API request.

    Returns:
        ChemblDataset: with ``smiles``, ``y`` (pChEMBL) and ``frame``.

    Example:
        >>> data = load_chembl_target("CHEMBL279")
        >>> smiles, y = data.smiles, data.y
        >>> print(data.report())
    """
    frame = fetch_chembl_activities(target_id, activity_type, cache_dir,
                                    refresh, max_records, timeout)
    funnel = {"records fetched": len(frame)}

    frame = frame[frame["canonical_smiles"].notna()]
    funnel["with SMILES"] = len(frame)

    if relation is not None:
        frame = frame[frame["standard_relation"] == relation]
        funnel[f"relation {relation!r}"] = len(frame)
    if units is not None:
        frame = frame[frame["standard_units"] == units]
        funnel[f"units {units!r}"] = len(frame)
    if organism is not None:
        frame = frame[frame["target_organism"] == organism]
        funnel[f"organism {organism!r}"] = len(frame)

    frame = frame[frame["pchembl_value"].notna()]
    funnel["with pChEMBL"] = len(frame)
    if frame.empty:
        raise ValueError(
            f"no {activity_type} records for {target_id!r} survived filtering; "
            "relax `relation`, `units` or `organism`"
        )

    frame = frame.assign(pchembl=frame["pchembl_value"].astype(float))
    if aggregate is None:
        table = frame[["molecule_chembl_id", "canonical_smiles", "pchembl"]].copy()
        table["n_measurements"] = 1
        table = table.rename(columns={"canonical_smiles": "smiles"})
    else:
        grouped = frame.groupby("molecule_chembl_id")
        table = grouped.agg(
            smiles=("canonical_smiles", "first"),
            pchembl=("pchembl", aggregate),
            n_measurements=("pchembl", "size"),
        ).reset_index()
    funnel["unique molecules"] = int(table["molecule_chembl_id"].nunique())

    table = table.reset_index(drop=True)
    return ChemblDataset(target_id, activity_type, table, funnel)


def chembl_status(timeout=60):
    """Which ChEMBL release the API is currently serving.

    ChEMBL is a living database: the same target returns more records
    after each release, so a count reported in a paper is only
    reproducible if the release is named alongside it. Record this with
    your results, together with the date you downloaded.

    Returns:
        dict: ``chembl_db_version``, ``chembl_release_date`` and the
        database-wide totals the endpoint reports.

    Example:
        >>> chembl_status()["chembl_db_version"]
        'ChEMBL_37'
    """
    return _get(CHEMBL_STATUS_API, timeout)


def pchembl_to_binary(y, threshold=6.0):
    """Turn pChEMBL values into an active/inactive label.

    ``threshold=6.0`` is the common convention — 1 micromolar — but it
    is a convention, not a property of the data. Report the threshold
    with any result that depends on it, and check how much the result
    moves when it changes.

    Args:
        y: Array of pChEMBL values.
        threshold: Values strictly above this count as active.

    Returns:
        numpy.ndarray: Integer array of 0/1 labels.
    """
    return (np.asarray(y, dtype=float) > float(threshold)).astype(int)
