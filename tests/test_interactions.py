import numpy as np
import pandas as pd
import pytest

from ikn_library.interactions import DTIDataset, load_davis, load_yamanishi
from ikn_library.interactions.dti import _COLUMNS


@pytest.fixture
def dti_file(tmp_path):
    frame = pd.DataFrame({
        "ID1": ["d1", "d1", "d2"],
        "X1": ["CCO", "CCO", "c1ccccc1"],
        "ID2": ["t1", "t2", "t1"],
        "X2": ["MKV", "ACDE", "MKV"],
        "Y": [10.0, 10000.0, 100.0],
    })
    path = tmp_path / "davis.tab"
    frame.to_csv(path, sep="\t", index=False)
    return path


def test_dti_dataset_arrays(dti_file):
    data = load_davis(dti_file, log_transform=False)
    smiles, sequences, y = data.arrays()
    assert len(smiles) == len(sequences) == len(y) == 3
    assert data.n_drugs == 2 and data.n_targets == 2
    np.testing.assert_allclose(y, [10.0, 10000.0, 100.0])


def test_davis_log_transform(dti_file):
    data = load_davis(dti_file, log_transform=True)
    _, _, y = data.arrays()
    np.testing.assert_allclose(y, [8.0, 5.0, 7.0])   # pKd = -log10(Kd nM * 1e-9)


def test_dti_rejects_wrong_columns():
    with pytest.raises(ValueError):
        DTIDataset(pd.DataFrame({"smiles": ["CCO"]}))
    assert set(_COLUMNS) == {"ID1", "X1", "ID2", "X2", "Y"}


@pytest.fixture
def yamanishi_file(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("hsa:1\tD001\nhsa:1\tD002\nhsa:2\tD001\n")
    return path


def test_yamanishi_matrix(yamanishi_file):
    data = load_yamanishi("enzyme", source=yamanishi_file)
    matrix = data.interaction_matrix()
    assert matrix.shape == (2, 2)                      # 2 drugs x 2 targets
    assert matrix.loc["D001", "hsa:1"] == 1
    assert matrix.loc["D002", "hsa:2"] == 0
    assert int(matrix.values.sum()) == 3


def test_yamanishi_pairs_sampling(yamanishi_file):
    data = load_yamanishi("enzyme", source=yamanishi_file)
    drugs, targets, y = data.pairs(negative_ratio=1.0, seed=42)
    assert y.sum() == 3                                # all positives kept
    assert (y == 0).sum() == 1                         # only 1 negative exists
    # sampled negative must not be a known positive
    positives = set(zip(data.positives["drug_id"], data.positives["target_id"]))
    for d, t, label in zip(drugs, targets, y):
        assert ((d, t) in positives) == bool(label)


def test_yamanishi_pairs_reproducible(yamanishi_file):
    data = load_yamanishi("enzyme", source=yamanishi_file)
    a = data.pairs(negative_ratio=1.0, seed=7)
    b = data.pairs(negative_ratio=1.0, seed=7)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)


def test_yamanishi_invalid_subset():
    with pytest.raises(ValueError):
        load_yamanishi("kinase")
