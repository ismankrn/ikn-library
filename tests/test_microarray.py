import gzip

import numpy as np
import pandas as pd
import pytest

from ikn_library.microarray import (
    load_geo,
    log2_transform,
    median_center,
    quantile_normalize,
    top_variance,
    zscore,
)
from ikn_library.microarray.geo import matrix_url

CONTENT = (
    '!Series_title\t"tiny test series"\n'
    '!Sample_title\t"biopsy 1"\t"biopsy 2"\t"biopsy 3"\n'
    '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\n'
    '!Sample_source_name_ch1\t"colon"\t"colon"\t"colon"\n'
    '!Sample_characteristics_ch1\t"disease: UC"\t"disease: Normal"\t"disease: UC"\n'
    '!Sample_characteristics_ch1\t"age: 40"\t"age: "\t"sex: M"\n'
    "!series_matrix_table_begin\n"
    '"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\n'
    "p1\t0.10\t0.20\t0.30\n"
    "p2\t0.50\tnull\t0.70\n"
    "p3\tnull\tnull\t9.00\n"
    "p4\tnull\tnull\tnull\n"
    "!series_matrix_table_end\n"
)


@pytest.fixture
def matrix_file(tmp_path):
    path = tmp_path / "TINY_series_matrix.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write(CONTENT)
    return path


def test_load_local_file(matrix_file):
    data = load_geo(matrix_file)
    assert data.X.shape == (3, 4)
    assert list(data.X.index) == ["GSM1", "GSM2", "GSM3"]
    assert list(data.X.columns) == ["p1", "p2", "p3", "p4"]
    assert data.X.loc["GSM3", "p2"] == pytest.approx(0.7)
    assert np.isnan(data.X.loc["GSM2", "p2"])


def test_metadata_and_labels(matrix_file):
    data = load_geo(matrix_file)
    assert list(data.y("disease")) == ["UC", "Normal", "UC"]
    assert data.metadata.loc["GSM1", "title"] == "biopsy 1"
    assert data.metadata.loc["GSM1", "age"] == "40"
    assert pd.isna(data.metadata.loc["GSM2", "age"])  # empty value
    assert data.metadata.loc["GSM3", "sex"] == "M"
    with pytest.raises(KeyError):
        data.y("nonexistent")


def test_dropna_threshold(matrix_file):
    data = load_geo(matrix_file, dropna_threshold=0.5)
    assert list(data.X.columns) == ["p1", "p2"]


def test_impute_mean(matrix_file):
    data = load_geo(matrix_file, impute="mean")
    assert "p4" not in data.X.columns  # entirely missing -> dropped
    assert not data.X.isna().any().any()
    assert data.X.loc["GSM2", "p2"] == pytest.approx(0.6)  # mean of 0.5, 0.7


def test_invalid_inputs(matrix_file):
    with pytest.raises(ValueError):
        load_geo("not-an-accession-or-file")
    with pytest.raises(ValueError):
        load_geo(matrix_file, dropna_threshold=2.0)
    with pytest.raises(ValueError):
        load_geo(matrix_file, impute="zero")


def test_matrix_url():
    assert matrix_url("GSE11223") == (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11223/"
        "matrix/GSE11223_series_matrix.txt.gz"
    )
    assert "GSEnnn/GSE123/" in matrix_url("gse123")
    with pytest.raises(ValueError):
        matrix_url("GPL1708")


def test_log2_transform():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [3.0, 7.0]})
    result = log2_transform(X)  # log2(x + 1)
    np.testing.assert_allclose(result.values, [[0.0, 2.0], [1.0, 3.0]])
    with pytest.raises(ValueError):
        log2_transform(pd.DataFrame({"a": [-2.0]}))


def test_quantile_normalize_makes_distributions_identical():
    X = pd.DataFrame({
        "p1": [5.0, 4.0, 3.0],
        "p2": [2.0, 1.0, 4.0],
        "p3": [3.0, 2.0, 6.0],
        "p4": [4.0, 3.0, 8.0],
    }, index=["s1", "s2", "s3"])  # no ties within any sample
    result = quantile_normalize(X)
    sorted_rows = np.sort(result.values, axis=1)
    for row in sorted_rows[1:]:
        np.testing.assert_allclose(row, sorted_rows[0])
    # Within each sample the ordering of probes is preserved.
    assert (result.rank(axis=1).values == X.rank(axis=1).values).all()


def test_quantile_normalize_ties_share_a_value():
    X = pd.DataFrame({
        "p1": [4.0, 1.0],
        "p2": [4.0, 2.0],
        "p3": [1.0, 3.0],
    })
    result = quantile_normalize(X)
    # Tied inputs map to the same (average-rank) normalized value.
    assert result.loc[0, "p1"] == result.loc[0, "p2"]
    assert result.loc[0, "p1"] > result.loc[0, "p3"]


def test_quantile_normalize_rejects_missing():
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    with pytest.raises(ValueError):
        quantile_normalize(X)


def test_zscore():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    result = zscore(X)
    np.testing.assert_allclose(result["a"].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(result["a"].std(ddof=0), 1.0)
    np.testing.assert_allclose(result["b"].values, 0.0)  # constant probe


def test_median_center():
    X = pd.DataFrame({"a": [1.0, 10.0], "b": [2.0, 20.0], "c": [3.0, 30.0]})
    result = median_center(X)
    np.testing.assert_allclose(result.median(axis=1).values, 0.0)
    np.testing.assert_allclose(result.loc[1].values, [-10.0, 0.0, 10.0])


def test_top_variance():
    X = pd.DataFrame({
        "a": [1.0, 1.0, 1.0],
        "b": [0.0, 5.0, 10.0],
        "c": [1.0, 2.0, 3.0],
    })
    assert list(top_variance(X, 2).columns) == ["b", "c"]
    assert top_variance(X, 99).shape == X.shape
    with pytest.raises(ValueError):
        top_variance(X, 0)
