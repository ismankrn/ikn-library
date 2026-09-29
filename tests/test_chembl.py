import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ikn_library.molecules import (
    ChemblDataset,
    load_chembl_target,
    pchembl_to_binary,
)

RECORDS = [
    # molecule, smiles, relation, units, pchembl, organism
    ("CHEMBL1", "CCO", "=", "nM", 7.0, "Homo sapiens"),
    ("CHEMBL1", "CCO", "=", "nM", 7.4, "Homo sapiens"),     # replicate
    ("CHEMBL2", "CCN", "=", "nM", 5.5, "Homo sapiens"),
    ("CHEMBL3", "CCC", ">", "nM", 6.1, "Homo sapiens"),     # censored
    ("CHEMBL4", "CCF", "=", "uM", 6.2, "Homo sapiens"),     # wrong units
    ("CHEMBL5", None, "=", "nM", 8.0, "Homo sapiens"),      # no structure
    ("CHEMBL6", "CCBr", "=", "nM", None, "Homo sapiens"),   # no pChEMBL
    ("CHEMBL7", "CCCl", "=", "nM", 6.9, "Rattus norvegicus"),
]


@pytest.fixture
def cache(tmp_path):
    """A pre-populated cache, so no test touches the network."""
    frame = pd.DataFrame(
        [{"molecule_chembl_id": m, "canonical_smiles": s, "standard_type": "IC50",
          "standard_relation": r, "standard_value": 100.0, "standard_units": u,
          "pchembl_value": p, "assay_chembl_id": "CHEMBL_A", "target_organism": o}
         for m, s, r, u, p, o in RECORDS])
    frame.to_csv(tmp_path / "CHEMBL279_IC50.csv.gz", index=False, compression="gzip")
    return tmp_path


def test_default_cleaning_and_funnel(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache)
    assert isinstance(data, ChemblDataset)
    assert data.funnel == {
        "records fetched": 8,
        "with SMILES": 7,
        "relation '='": 6,
        "units 'nM'": 5,
        "with pChEMBL": 4,
        "unique molecules": 3,
    }
    # CHEMBL1 (replicates), CHEMBL2, CHEMBL7 survive
    assert sorted(data.frame["molecule_chembl_id"]) == ["CHEMBL1", "CHEMBL2", "CHEMBL7"]


def test_replicates_are_collapsed_by_median(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache)
    row = data.frame.set_index("molecule_chembl_id").loc["CHEMBL1"]
    assert row["pchembl"] == pytest.approx(7.2)      # median of 7.0 and 7.4
    assert row["n_measurements"] == 2


def test_aggregate_none_keeps_every_measurement(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache, aggregate=None)
    assert len(data.frame) == 4                       # not 3
    assert list(data.frame["molecule_chembl_id"]).count("CHEMBL1") == 2


def test_keeping_censored_records_changes_the_funnel(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache, relation=None)
    assert "relation '='" not in data.funnel
    assert data.funnel["units 'nM'"] == 6             # the '>' record survives


def test_organism_filter(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache, organism="Homo sapiens")
    assert "CHEMBL7" not in set(data.frame["molecule_chembl_id"])
    assert data.funnel["organism 'Homo sapiens'"] == 4


def test_smiles_and_y_line_up(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache)
    assert len(data.smiles) == len(data.y) == len(data.frame)
    assert data.y.dtype == float
    assert all(isinstance(s, str) for s in data.smiles)


def test_over_filtering_raises_with_a_usable_message(cache):
    with pytest.raises(ValueError, match="survived filtering"):
        load_chembl_target("CHEMBL279", cache_dir=cache, organism="Mus musculus")


def test_report_lists_every_step(cache):
    data = load_chembl_target("CHEMBL279", cache_dir=cache)
    report = data.report()
    assert report.count("\n") == len(data.funnel) - 1
    assert "unique molecules" in report


@pytest.mark.parametrize("threshold, expected", [
    (6.0, [1, 0, 1]),
    (7.5, [0, 0, 0]),
    (5.0, [1, 1, 1]),
])
def test_pchembl_to_binary(threshold, expected):
    np.testing.assert_array_equal(
        pchembl_to_binary([7.2, 5.5, 6.9], threshold=threshold), expected)


def test_pchembl_to_binary_is_strict_above_the_threshold():
    np.testing.assert_array_equal(pchembl_to_binary([6.0, 6.0001]), [0, 1])
