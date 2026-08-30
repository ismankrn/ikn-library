import numpy as np
import pytest

pytest.importorskip("rdkit")

from ikn_library.molecules import CURATED_DESCRIPTORS, featurize

SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


def test_morgan_default():
    X = featurize(SMILES)
    assert X.shape == (3, 1024)
    assert list(X.columns[:2]) == ["morgan_0", "morgan_1"]
    assert set(np.unique(X.values)) <= {0, 1}


def test_fingerprint_methods_and_sizes():
    assert featurize(SMILES, method="maccs").shape == (3, 167)
    for method in ("rdkit", "atompair", "torsion"):
        X = featurize(SMILES, method=method, n_bits=256)
        assert X.shape == (3, 256)
        assert X.columns[0] == f"{method}_0"


def test_descriptors_curated():
    X = featurize(SMILES, method="descriptors")
    assert list(X.columns) == CURATED_DESCRIPTORS
    assert X.loc[0, "MolWt"] == pytest.approx(46.07, abs=0.01)   # ethanol
    assert X.loc[1, "NumAromaticRings"] == 1                     # benzene


def test_descriptors_all_and_custom():
    X_all = featurize(SMILES, method="descriptors", descriptor_names="all")
    assert X_all.shape[1] > 100
    X_two = featurize(SMILES, method="descriptors",
                      descriptor_names=["MolWt", "TPSA"])
    assert list(X_two.columns) == ["MolWt", "TPSA"]


def test_invalid_smiles_dropped_and_y_aligned():
    smiles = ["CCO", "not_a_smiles", "c1ccccc1"]
    y = np.array([1, 0, 1])
    with pytest.warns(UserWarning, match="dropped 1 invalid"):
        X, y_out = featurize(smiles, y, method="maccs")
    assert X.shape[0] == 2
    np.testing.assert_array_equal(y_out, [1, 1])


def test_invalid_smiles_raise():
    with pytest.raises(ValueError, match="invalid SMILES"):
        featurize(["CCO", "???"], method="maccs", on_invalid="raise")


def test_input_validation():
    with pytest.raises(ValueError):
        featurize(SMILES, y=np.array([1, 0]))
    with pytest.raises(ValueError):
        featurize(SMILES, method="fancy")
    with pytest.raises(ValueError):
        featurize(["CCO", "???"], method="maccs", on_invalid="ignore")


def test_mordred_backend():
    pytest.importorskip("mordred")
    X = featurize(SMILES, method="mordred")
    assert X.shape[0] == 3
    assert X.shape[1] > 1000
    assert X["MW"].iloc[0] == pytest.approx(46.04, abs=0.01)
