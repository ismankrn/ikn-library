import gzip

import numpy as np
import pytest

from ikn_library.molecules import load_sider, load_tox21

CONTENT = (
    "smiles,Hepatobiliary disorders,Eye disorders,Cardiac disorders\n"
    "CCO,1,0,1\n"
    "c1ccccc1,0,0,1\n"
    "CC(=O)O,1,1,0\n"
)


@pytest.fixture
def sider_file(tmp_path):
    path = tmp_path / "sider.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write(CONTENT)
    return path


def test_load_local_file(sider_file):
    data = load_sider(sider_file)
    assert len(data.smiles) == 3
    assert data.tasks == ["Hepatobiliary disorders", "Eye disorders",
                          "Cardiac disorders"]
    assert data.labels.shape == (3, 3)


def test_task_returns_smiles_and_labels(sider_file):
    data = load_sider(sider_file)
    smiles, y = data.task("Hepatobiliary disorders")
    np.testing.assert_array_equal(smiles, ["CCO", "c1ccccc1", "CC(=O)O"])
    np.testing.assert_array_equal(y, [1, 0, 1])
    assert y.dtype.kind == "i"


def test_task_substring_match(sider_file):
    data = load_sider(sider_file)
    _, y = data.task("hepato")   # case-insensitive substring
    np.testing.assert_array_equal(y, [1, 0, 1])


def test_task_no_match_and_ambiguous(sider_file):
    data = load_sider(sider_file)
    with pytest.raises(KeyError):
        data.task("nonexistent effect")
    with pytest.raises(KeyError):
        data.task("disorders")   # matches all three tasks


def test_task_smiles_is_a_copy(sider_file):
    data = load_sider(sider_file)
    smiles, _ = data.task("Eye disorders")
    smiles[0] = "changed"
    assert data.smiles[0] == "CCO"


def test_invalid_inputs(sider_file, tmp_path):
    with pytest.raises(ValueError):
        load_sider(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("not_smiles,task\nx,1\n")
    with pytest.raises(ValueError):
        load_sider(bad)


TOX21_CONTENT = (
    "NR-AR,SR-p53,mol_id,smiles\n"
    "1,0,TOX1,CCO\n"
    ",1,TOX2,c1ccccc1\n"
    "0,,TOX3,CC(=O)O\n"
)


@pytest.fixture
def tox21_file(tmp_path):
    path = tmp_path / "tox21.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write(TOX21_CONTENT)
    return path


def test_tox21_load_and_tasks(tox21_file):
    data = load_tox21(tox21_file)
    assert data.tasks == ["NR-AR", "SR-p53"]     # mol_id excluded from labels
    assert len(data.smiles) == 3
    assert "mol_id" in data.frame.columns


def test_tox21_task_drops_missing_labels(tox21_file):
    data = load_tox21(tox21_file)
    smiles, y = data.task("NR-AR")
    np.testing.assert_array_equal(smiles, ["CCO", "CC(=O)O"])   # TOX2 unlabeled
    np.testing.assert_array_equal(y, [1, 0])
    assert y.dtype.kind == "i"

    smiles, y = data.task("p53")   # substring match
    np.testing.assert_array_equal(smiles, ["CCO", "c1ccccc1"])
    np.testing.assert_array_equal(y, [0, 1])


def test_tox21_task_errors(tox21_file):
    data = load_tox21(tox21_file)
    with pytest.raises(KeyError):
        data.task("NR-ER")       # not in this synthetic file
    with pytest.raises(KeyError):
        data.task("r")           # ambiguous
