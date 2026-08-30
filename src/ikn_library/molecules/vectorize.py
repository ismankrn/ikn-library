"""Vectorize SMILES strings into sequences for deep-learning models.

The encoding follows the smiles2vec convention (Goh et al., 2017; see
also https://github.com/Abdulk084/Smiles2vec): each SMILES becomes a
sequence of token indices framed by a start token and padded with an
end/pad token, ready for an embedding layer or a one-hot input.
"""

import re

import numpy as np

#: Regex matching SMILES atoms and bond symbols as single tokens
#: (Schwaller et al., 2019) — keeps "Cl", "Br", "[nH]" intact.
ATOM_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%\d{2}|\d)"
)

START_TOKEN = "!"
PAD_TOKEN = "E"
UNKNOWN_TOKEN = "?"


def tokenize_smiles(smiles, tokenizer="char"):
    """Split a SMILES string into tokens.

    Args:
        smiles: The SMILES string.
        tokenizer: ``"char"`` treats every character as one token (the
            smiles2vec convention); ``"atom"`` keeps multi-character
            atoms such as ``Cl``, ``Br`` and bracket atoms like
            ``[nH]`` together.
    """
    if tokenizer == "char":
        return list(smiles)
    if tokenizer == "atom":
        return ATOM_PATTERN.findall(smiles)
    raise ValueError('tokenizer must be "char" or "atom"')


class SmilesVectorizer:
    """Turn SMILES strings into padded index or one-hot sequences.

    Learns its vocabulary from a set of SMILES (:meth:`fit`), then
    encodes any SMILES into a fixed-length sequence of token indices
    (:meth:`transform`) suitable for an embedding layer, or into a
    one-hot tensor for models that take one directly.

    Every sequence starts with the start token ``"!"`` and is padded to
    ``max_length`` with the pad token ``"E"`` — the framing used by
    smiles2vec — so a sequence looks like
    ``! C C O E E ... E``.

    Args:
        tokenizer: ``"char"`` (default, one token per character) or
            ``"atom"`` (multi-character atoms kept together).
        max_length: Sequence length. ``None`` (default) derives it from
            the training data as ``longest + 5``; longer SMILES are
            truncated at :meth:`transform` time.

    Attributes:
        vocabulary: ``dict`` mapping token to integer index.
        index_to_token: ``dict`` mapping integer index back to token.
        max_length: The padded sequence length.

    Example:
        >>> vectorizer = SmilesVectorizer().fit(smiles)
        >>> X = vectorizer.transform(smiles)          # (n, max_length) ints
        >>> vectorizer.vocab_size, vectorizer.max_length
    """

    def __init__(self, tokenizer="char", max_length=None):
        if tokenizer not in ("char", "atom"):
            raise ValueError('tokenizer must be "char" or "atom"')
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocabulary = None
        self.index_to_token = None

    def fit(self, smiles):
        """Learn the token vocabulary from ``smiles``. Returns ``self``."""
        smiles = list(smiles)
        if not smiles:
            raise ValueError("cannot fit on an empty list of SMILES")
        tokens = set()
        longest = 0
        for s in smiles:
            found = tokenize_smiles(str(s), self.tokenizer)
            tokens.update(found)
            longest = max(longest, len(found))
        # Sorted for determinism: Python's set iteration order varies
        # between runs, which would silently change the encoding.
        specials = [PAD_TOKEN, START_TOKEN, UNKNOWN_TOKEN]
        ordered = specials + sorted(tokens - set(specials))
        self.vocabulary = {token: i for i, token in enumerate(ordered)}
        self.index_to_token = {i: token for token, i in self.vocabulary.items()}
        if self.max_length is None:
            self.max_length = longest + 5
        return self

    @property
    def vocab_size(self):
        """Number of distinct tokens, including the special tokens."""
        self._check_fitted()
        return len(self.vocabulary)

    def _check_fitted(self):
        if self.vocabulary is None:
            raise ValueError("vectorizer is not fitted; call fit(smiles) first")

    def transform(self, smiles, one_hot=False):
        """Encode SMILES into padded sequences.

        Args:
            smiles: Iterable of SMILES strings.
            one_hot: When ``True``, return a one-hot tensor of shape
                ``(n, max_length, vocab_size)`` instead of the
                ``(n, max_length)`` integer matrix.

        Unknown tokens (absent from the training vocabulary) map to the
        ``"?"`` token rather than raising.
        """
        self._check_fitted()
        smiles = list(smiles)
        pad = self.vocabulary[PAD_TOKEN]
        unknown = self.vocabulary[UNKNOWN_TOKEN]
        X = np.full((len(smiles), self.max_length), pad, dtype=np.int32)
        for i, s in enumerate(smiles):
            X[i, 0] = self.vocabulary[START_TOKEN]
            tokens = tokenize_smiles(str(s), self.tokenizer)[: self.max_length - 1]
            for j, token in enumerate(tokens, start=1):
                X[i, j] = self.vocabulary.get(token, unknown)
        if not one_hot:
            return X
        encoded = np.zeros((len(smiles), self.max_length, self.vocab_size),
                           dtype=np.int8)
        rows, columns = np.indices(X.shape)
        encoded[rows, columns, X] = 1
        return encoded

    def fit_transform(self, smiles, one_hot=False):
        """Fit on ``smiles`` and encode them in one step."""
        return self.fit(smiles).transform(smiles, one_hot=one_hot)

    def inverse_transform(self, X):
        """Decode integer sequences back into SMILES strings."""
        self._check_fitted()
        X = np.asarray(X)
        if X.ndim == 3:      # one-hot
            X = X.argmax(axis=2)
        strings = []
        for row in X:
            tokens = [self.index_to_token[int(i)] for i in row]
            strings.append("".join(t for t in tokens
                                   if t not in (START_TOKEN, PAD_TOKEN)))
        return strings

    def vocabulary_table(self):
        """The token-to-index mapping as a ``pandas.DataFrame``.

        Handy for inspecting or printing the encoding:
        ``print(vectorizer.vocabulary_table())``.
        """
        import pandas as pd
        self._check_fitted()
        return pd.DataFrame(
            {"token": list(self.vocabulary), "index": list(self.vocabulary.values())}
        ).set_index("index")

    def __repr__(self):
        if self.vocabulary is None:
            return f"<SmilesVectorizer (unfitted, tokenizer={self.tokenizer!r})>"
        return (f"<SmilesVectorizer: vocab_size={self.vocab_size}, "
                f"max_length={self.max_length}, tokenizer={self.tokenizer!r}>")
