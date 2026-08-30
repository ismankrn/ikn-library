import numpy as np
import pytest

from ikn_library.interactions import cold_split


@pytest.fixture
def pairs():
    """10 drugs x 5 targets, fully crossed (50 pairs)."""
    drugs = np.repeat([f"d{i}" for i in range(10)], 5)
    targets = np.tile([f"t{j}" for j in range(5)], 10)
    return drugs, targets


def test_cold_drug_has_no_shared_drugs(pairs):
    drugs, targets = pairs
    train, test = cold_split(drugs, targets, mode="drug", test_size=0.3, seed=0)
    assert not set(drugs[train]) & set(drugs[test])
    # targets may (and here do) overlap — only drugs are held out
    assert set(targets[train]) & set(targets[test])
    assert len(train) + len(test) == len(drugs)


def test_cold_target_has_no_shared_targets(pairs):
    drugs, targets = pairs
    train, test = cold_split(drugs, targets, mode="target", test_size=0.4, seed=0)
    assert not set(targets[train]) & set(targets[test])
    assert set(drugs[train]) & set(drugs[test])


def test_cold_both_shares_neither(pairs):
    drugs, targets = pairs
    train, test = cold_split(drugs, targets, mode="both", test_size=0.3, seed=0)
    assert not set(drugs[train]) & set(drugs[test])
    assert not set(targets[train]) & set(targets[test])
    # pairs touching exactly one held-out entity are dropped from both
    assert len(train) + len(test) < len(drugs)


def test_random_split_shares_entities(pairs):
    drugs, targets = pairs
    train, test = cold_split(drugs, targets, mode="random", test_size=0.2, seed=0)
    assert len(test) == 10
    assert len(train) + len(test) == len(drugs)
    assert set(drugs[train]) & set(drugs[test])   # leakage, by design


def test_indices_are_disjoint_and_sorted(pairs):
    drugs, targets = pairs
    for mode in ("random", "drug", "target", "both"):
        train, test = cold_split(drugs, targets, mode=mode, seed=1)
        assert not set(train) & set(test)
        assert list(train) == sorted(train)
        assert list(test) == sorted(test)
        assert len(test) > 0 and len(train) > 0


def test_reproducible_with_seed(pairs):
    drugs, targets = pairs
    a = cold_split(drugs, targets, mode="drug", seed=42)
    b = cold_split(drugs, targets, mode="drug", seed=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_tiny_entity_count_still_splits():
    drugs = np.array(["d1", "d1", "d2", "d2"])
    targets = np.array(["t1", "t2", "t1", "t2"])
    train, test = cold_split(drugs, targets, mode="drug", test_size=0.1, seed=0)
    assert len(test) > 0 and len(train) > 0        # at least one drug held out


def test_input_validation(pairs):
    drugs, targets = pairs
    with pytest.raises(ValueError):
        cold_split(drugs, targets[:-1])
    with pytest.raises(ValueError):
        cold_split(drugs, targets, mode="cold")
    with pytest.raises(ValueError):
        cold_split(drugs, targets, test_size=0.0)
