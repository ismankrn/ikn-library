# SMILES to Sequences (smiles2vec)

Fingerprints and descriptors summarize a molecule as a bag of features.
A **sequence model** instead reads the SMILES string itself, character
by character, the way a language model reads text.
`SmilesVectorizer` produces exactly the input such models need: a
padded sequence of token indices per molecule.

The encoding follows the smiles2vec convention (Goh et al., 2017): each
sequence starts with the start token `"!"` and is padded to a fixed
length with the pad token `"E"`.

## Vectorizing

```python
from ikn_library.molecules import load_bbbp, SmilesVectorizer

data = load_bbbp()
smiles, y = data.task("p_np")

vectorizer = SmilesVectorizer().fit(smiles)
print(vectorizer)

X = vectorizer.transform(smiles)
print(X.shape, X.dtype)
```

```text
<SmilesVectorizer: vocab_size=42, max_length=405, tokenizer='char'>
(2050, 405) int32
```

`max_length` defaults to *the longest training SMILES + 5*; pass
`max_length=120` to fix it yourself (longer SMILES are then truncated).
`fit_transform(smiles)` does both steps at once, and
`transform(smiles, one_hot=True)` returns a
`(n, max_length, vocab_size)` one-hot tensor for models that take one
directly.

## Inspecting the symbol dictionary

The learned token-to-index mapping is available as `vocabulary` (a
plain dict) or as a table:

```python
print(vectorizer.vocabulary_table().head(10))
```

```text
      token
index
0         E
1         !
2         ?
3         #
4         %
5         (
6         )
7         +
8         -
9         .
```

The three special tokens always occupy the first indices: `E` (pad) is
**0**, so Keras' `mask_zero=True` works out of the box; `!` is the
start token; `?` catches tokens unseen during `fit` — encoding never
crashes on a new molecule. Decoding is available too:

```python
vectorizer.inverse_transform(X[:1])   # -> ['[Cl].CC(C)NCC(O)COc1cccc2ccccc12']
```

!!! note "Two deviations from the reference notebook"
    - **Deterministic vocabulary.** The original builds the charset
      with a Python `set`, whose iteration order changes between
      interpreter runs — the same molecule would encode differently
      after a restart. Here the vocabulary is sorted, so encodings are
      reproducible (and a saved model keeps matching its inputs).
    - **An unknown token.** The original raises a `KeyError` on any
      character absent from the training set; `?` makes inference on
      new data safe.

## Character or atom tokens

```python
from ikn_library.molecules import tokenize_smiles

tokenize_smiles("CClBr", "char")   # ['C', 'C', 'l', 'B', 'r']
tokenize_smiles("CClBr", "atom")   # ['C', 'Cl', 'Br']
```

`SmilesVectorizer(tokenizer="atom")` keeps multi-character atoms
(`Cl`, `Br`, `[nH]`) as single tokens, which is chemically more
faithful — the character tokenizer splits `Cl` into an unrelated `C`
and `l`. The character mode is the smiles2vec default and remains the
default here.

## Example: an LSTM classifier

Vectorized SMILES feed straight into a recurrent network. Predicting
blood-brain barrier penetration (BBBP) with an embedding + LSTM:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from ikn_library.molecules import load_bbbp, SmilesVectorizer

data = load_bbbp()
smiles, y = data.task("p_np")

vectorizer = SmilesVectorizer(max_length=120).fit(smiles)
X = vectorizer.transform(smiles)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

keras.utils.set_random_seed(0)
model = keras.Sequential([
    keras.layers.Input(shape=(vectorizer.max_length,)),
    keras.layers.Embedding(vectorizer.vocab_size, 64, mask_zero=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, validation_split=0.1,
          epochs=10, batch_size=32, verbose=0)

proba = model.predict(X_test, verbose=0).ravel()
print("Test accuracy:", ((proba > 0.5) == y_test).mean())
print("Test ROC-AUC :", roc_auc_score(y_test, proba))
```

Output:

```text
Test accuracy: 0.8707
Test ROC-AUC : 0.9334
```

For reference, always predicting the majority class scores 0.7634
accuracy on this split — the LSTM learns real signal from the raw
strings, with no fingerprints or descriptors involved.

Points worth explaining to students:

- **`Embedding(vocab_size, 64)`** learns a 64-dimensional vector per
  token — the "2vec" part of smiles2vec. It replaces one-hot inputs
  and is learned jointly with the classifier.
- **`mask_zero=True`** tells Keras to ignore padding positions, which
  works precisely because the pad token has index 0.
- The same `X` also fits a 1D-CNN (as in the original smiles2vec paper)
  or a bidirectional LSTM; only the layer stack changes.
- TensorFlow is **not** a dependency of this library — install it
  yourself (`pip install tensorflow`) to run this example.

## Reference

G. B. Goh, N. O. Hodas, C. Siegel, and A. Vishnu, "SMILES2Vec: an
interpretable general-purpose deep neural network for predicting
chemical properties," arXiv:1712.02034, 2017.
Reference implementation:
[github.com/Abdulk084/Smiles2vec](https://github.com/Abdulk084/Smiles2vec).
