import numpy as np
import pytest

from ikn_library.molecules import SmilesVectorizer, tokenize_smiles

SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


def test_tokenize_char_and_atom():
    assert tokenize_smiles("CClBr", "char") == list("CClBr")
    assert tokenize_smiles("CClBr", "atom") == ["C", "Cl", "Br"]
    assert tokenize_smiles("C[nH]1", "atom") == ["C", "[nH]", "1"]
    with pytest.raises(ValueError):
        tokenize_smiles("CCO", "word")


def test_fit_learns_vocabulary_and_length():
    vectorizer = SmilesVectorizer().fit(SMILES)
    assert set("CcO1(=)") <= set(vectorizer.vocabulary)
    assert vectorizer.max_length == len("c1ccccc1") + 5     # longest + 5
    assert vectorizer.vocab_size == len(vectorizer.vocabulary)
    # special tokens occupy the first indices
    assert vectorizer.vocabulary["E"] == 0
    assert vectorizer.vocabulary["!"] == 1


def test_vocabulary_is_deterministic():
    a = SmilesVectorizer().fit(SMILES).vocabulary
    b = SmilesVectorizer().fit(list(reversed(SMILES))).vocabulary
    assert a == b        # sorted vocabulary, independent of input order


def test_transform_shape_framing_and_padding():
    vectorizer = SmilesVectorizer().fit(SMILES)
    X = vectorizer.transform(SMILES)
    assert X.shape == (3, vectorizer.max_length)
    assert (X[:, 0] == vectorizer.vocabulary["!"]).all()     # start token
    # "CCO" occupies positions 1..3, the rest is padding
    assert X[0, 1] == vectorizer.vocabulary["C"]
    assert (X[0, 4:] == vectorizer.vocabulary["E"]).all()


def test_transform_one_hot():
    vectorizer = SmilesVectorizer().fit(SMILES)
    X = vectorizer.transform(SMILES, one_hot=True)
    assert X.shape == (3, vectorizer.max_length, vectorizer.vocab_size)
    assert (X.sum(axis=2) == 1).all()                        # exactly one hot
    np.testing.assert_array_equal(X.argmax(axis=2), vectorizer.transform(SMILES))


def test_inverse_transform_round_trip():
    vectorizer = SmilesVectorizer().fit(SMILES)
    assert vectorizer.inverse_transform(vectorizer.transform(SMILES)) == SMILES
    one_hot = vectorizer.transform(SMILES, one_hot=True)
    assert vectorizer.inverse_transform(one_hot) == SMILES


def test_unknown_tokens_map_to_question_mark():
    vectorizer = SmilesVectorizer().fit(["CCO"])
    X = vectorizer.transform(["CCN"])                          # N unseen
    assert X[0, 3] == vectorizer.vocabulary["?"]


def test_long_smiles_are_truncated():
    vectorizer = SmilesVectorizer(max_length=4).fit(SMILES)
    X = vectorizer.transform(["CCCCCCCCCC"])
    assert X.shape == (1, 4)
    assert X[0, 0] == vectorizer.vocabulary["!"]


def test_atom_tokenizer_end_to_end():
    vectorizer = SmilesVectorizer(tokenizer="atom").fit(["CCl", "CBr"])
    assert "Cl" in vectorizer.vocabulary and "Br" in vectorizer.vocabulary
    assert vectorizer.inverse_transform(vectorizer.transform(["CCl"])) == ["CCl"]


def test_vocabulary_table():
    vectorizer = SmilesVectorizer().fit(SMILES)
    table = vectorizer.vocabulary_table()
    assert list(table.columns) == ["token"]
    assert len(table) == vectorizer.vocab_size
    assert table.loc[0, "token"] == "E"


def test_errors_before_fit_and_on_bad_input():
    with pytest.raises(ValueError):
        SmilesVectorizer().transform(SMILES)
    with pytest.raises(ValueError):
        SmilesVectorizer().fit([])
    with pytest.raises(ValueError):
        SmilesVectorizer(tokenizer="bytes")
    with pytest.raises(ValueError):
        SmilesVectorizer(max_length=1)
