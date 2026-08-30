import numpy as np
import pandas as pd
import pytest

from ikn_library.interactions import (
    DDIDataset,
    load_drugbank_ddi,
    pair_features,
)


@pytest.fixture
def ddi_file(tmp_path):
    frame = pd.DataFrame({
        "ID1": ["DB1", "DB2", "DB3", "DB1"],
        "ID2": ["DB2", "DB3", "DB1", "DB3"],
        "Y": [1, 1, 2, 2],
        "Map": ["type one", "type one", "type two", "type two"],
        "X1": ["CCO", "CCC", "CCN", "CCO"],
        "X2": ["CCC", "CCN", "CCO", "CCN"],
    })
    path = tmp_path / "ddi.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_and_columns(ddi_file):
    data = load_drugbank_ddi(ddi_file)
    assert data.n_drugs == 3
    assert set(data.frame.columns) >= {"drug1_id", "drug2_id", "smiles1",
                                       "smiles2", "interaction_type", "description"}
    smiles1, smiles2, y = data.arrays()
    assert len(smiles1) == len(smiles2) == len(y) == 4


def test_interaction_types_summary(ddi_file):
    types = load_drugbank_ddi(ddi_file).interaction_types
    assert list(types.index) == [1, 2]              # tie, both count 2
    assert types.loc[1, "count"] == 2
    assert types.loc[1, "description"] == "type one"


def test_binary_task(ddi_file):
    data = load_drugbank_ddi(ddi_file)
    smiles1, smiles2, y = data.binary_task(1, negative_ratio=1.0, seed=0)
    assert y.sum() == 2 and (y == 0).sum() == 2
    assert len(smiles1) == len(smiles2) == 4
    # positives really are of the requested type
    positives = set(zip(smiles1[y == 1], smiles2[y == 1]))
    assert positives == {("CCO", "CCC"), ("CCC", "CCN")}


def test_binary_task_errors(ddi_file):
    data = load_drugbank_ddi(ddi_file)
    with pytest.raises(ValueError):
        data.binary_task(99)
    with pytest.raises(ValueError):
        data.binary_task(1, negative_ratio=-1)


def test_rejects_wrong_table():
    with pytest.raises(ValueError):
        DDIDataset(pd.DataFrame({"smiles1": ["CCO"]}))


# --- pair_features ---------------------------------------------------

A = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 0.0]})
B = pd.DataFrame({"f0": [4.0, 2.0], "f1": [1.0, 5.0]})


def test_concat_is_asymmetric():
    X = pair_features(A, B, method="concat")
    assert list(X.columns) == ["f0_1", "f1_1", "f0_2", "f1_2"]
    assert X.shape == (2, 4)
    flipped = pair_features(B, A, method="concat")
    assert not np.array_equal(X.to_numpy(), flipped.to_numpy())


@pytest.mark.parametrize("method", ["sum", "product", "absdiff", "mean", "max"])
def test_symmetric_methods(method):
    X = pair_features(A, B, method=method)
    flipped = pair_features(B, A, method=method)
    np.testing.assert_allclose(X.to_numpy(), flipped.to_numpy())
    assert X.shape == (2, 2)
    assert X.columns[0] == f"{method}_f0"


def test_symmetric_values():
    np.testing.assert_allclose(pair_features(A, B, "sum").to_numpy(),
                               [[5.0, 4.0], [4.0, 5.0]])
    np.testing.assert_allclose(pair_features(A, B, "absdiff").to_numpy(),
                               [[3.0, 2.0], [0.0, 5.0]])
    np.testing.assert_allclose(pair_features(A, B, "max").to_numpy(),
                               [[4.0, 3.0], [2.0, 5.0]])


def test_pair_features_validation():
    with pytest.raises(ValueError):
        pair_features(A, B, method="stack")
    with pytest.raises(ValueError):
        pair_features(A, B.iloc[:1])
    with pytest.raises(ValueError):     # element-wise needs equal widths
        pair_features(A, B[["f0"]], method="sum")
