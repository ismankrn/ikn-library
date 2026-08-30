import numpy as np
import pytest

from ikn_library.proteins import featurize_protein
from ikn_library.proteins.featurize import CTD_PROPERTIES


def test_ctd_property_groups_cover_all_amino_acids():
    for prop, groups in CTD_PROPERTIES.items():
        letters = "".join(groups)
        assert len(letters) == 20, prop
        assert set(letters) == set("ACDEFGHIKLMNPQRSTVWY"), prop


def test_aac():
    X = featurize_protein(["AAAA", "ACAC"])
    assert X.shape == (2, 20)
    assert X.loc[0, "aac_A"] == 1.0
    assert X.loc[1, "aac_A"] == 0.5 and X.loc[1, "aac_C"] == 0.5
    np.testing.assert_allclose(X.sum(axis=1), 1.0)


def test_dpc():
    X = featurize_protein(["AAA", "ACA"], method="dpc")
    assert X.shape == (2, 400)
    assert X.loc[0, "dpc_AA"] == 1.0                  # both dipeptides are AA
    assert X.loc[1, "dpc_AC"] == 0.5 and X.loc[1, "dpc_CA"] == 0.5
    np.testing.assert_allclose(X.sum(axis=1), 1.0)


def test_ctd_shape_and_charge_values():
    X = featurize_protein(["KKDD"], method="ctd")
    assert X.shape == (1, 147)
    # charge groups: 1 = KR, 3 = DE
    assert X.loc[0, "charge_C1"] == pytest.approx(0.5)
    assert X.loc[0, "charge_C3"] == pytest.approx(0.5)
    # exactly one 1<->3 transition (K->D) among 3 adjacent pairs
    assert X.loc[0, "charge_T13"] == pytest.approx(1 / 3)
    # distribution: first K at position 1 of 4 -> 25%; last K at 2 of 4 -> 50%
    assert X.loc[0, "charge_D1_0"] == pytest.approx(25.0)
    assert X.loc[0, "charge_D1_100"] == pytest.approx(50.0)


def test_ctd_absent_group_is_zero():
    X = featurize_protein(["AAAA"], method="ctd")     # no charged residues
    assert X.loc[0, "charge_C1"] == 0.0
    assert X.loc[0, "charge_D1_50"] == 0.0


def test_nonstandard_characters_are_removed():
    X_clean = featurize_protein(["ACD"])
    X_noisy = featurize_protein(["ACXD-*"])
    np.testing.assert_allclose(X_clean.values, X_noisy.values)


def test_invalid_sequences_dropped_and_y_aligned():
    y = np.array([1, 0, 1])
    with pytest.warns(UserWarning, match="dropped 1 invalid"):
        X, y_out = featurize_protein(["ACD", "123", "MKV"], y)
    assert X.shape[0] == 2
    np.testing.assert_array_equal(y_out, [1, 1])


def test_input_validation():
    with pytest.raises(ValueError):
        featurize_protein(["ACD"], y=np.array([1, 2]))
    with pytest.raises(ValueError):
        featurize_protein(["ACD"], method="embedding")
    with pytest.raises(ValueError):
        featurize_protein(["123"], on_invalid="raise")
