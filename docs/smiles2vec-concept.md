# SMILES2Vec Concept

This note explains the idea behind
[SMILES to Sequences](smiles2vec.md): how a molecule written as text
becomes a tensor a neural network can read, and why each step of
`SmilesVectorizer` is shaped the way it is. Everything is worked by
hand on three small molecules.

The premise is that a SMILES string is a *sentence about a molecule*.
`CCO` is ethanol, `CC(=O)O` is acetic acid, `c1ccccc1` is benzene. A
fingerprint throws the string away and keeps a bag of substructures; a
sequence model keeps the string and reads it left to right, the way a
language model reads text. Getting from a string to something a network
can multiply takes three transformations — **tokenize**, **index**,
**embed** — plus one piece of bookkeeping, **padding**.

## Step 1 — tokenize: what counts as one symbol?

A network needs discrete symbols. The obvious choice is one character
per symbol, which is what smiles2vec does and what
`SmilesVectorizer` does by default. It is not the only choice:

```python
from ikn_library.molecules import tokenize_smiles

print(tokenize_smiles("ClCCBr", "char"))
print(tokenize_smiles("ClCCBr", "atom"))
```

Output:

```text
['C', 'l', 'C', 'C', 'B', 'r']
['Cl', 'C', 'C', 'Br']
```

Look at what the character tokenizer did to chlorine: `Cl` became a
carbon `C` followed by a lowercase `l` — two symbols that, chemically,
mean nothing together. The atom tokenizer keeps `Cl`, `Br` and bracket
atoms like `[nH]` intact.

So why is the character mode the default? Because it is what the
smiles2vec paper used, and because it mostly works anyway: the network
sees `C` followed by `l` in a consistent context every time chlorine
appears, and an LSTM or CNN can learn that pair as a unit. The atom
tokenizer removes the need to learn it. The trade is between fidelity
(atom) and reproducing the published method (char) — which is why both
are available and neither is hidden.

## Step 2 — index: build a vocabulary

Symbols still are not numbers. The vectorizer collects every distinct
token in the **training** molecules and assigns each one an integer:

```python
from ikn_library.molecules import SmilesVectorizer

toy = ["CCO", "CC(=O)O", "c1ccccc1"]
vectorizer = SmilesVectorizer().fit(toy)

print(vectorizer.vocabulary)
print("max_length:", vectorizer.max_length)
```

Output:

```text
{'E': 0, '!': 1, '?': 2, '(': 3, ')': 4, '1': 5, '=': 6, 'C': 7, 'O': 8, 'c': 9}
```

```text
max_length: 13
```

Three properties of that dictionary are decisions, not accidents.

**The three special tokens come first, and `E` is 0.** `E` is padding,
`!` marks the start of a sequence, `?` stands for anything unseen.
Pinning padding to index 0 is what lets Keras' `mask_zero=True` work
without any extra configuration.

**The rest is sorted.** Building a vocabulary by iterating a Python
`set` — as the reference implementation does — gives a different order
on every interpreter run. That matters far more than it sounds: a
trained model's embedding matrix is *indexed by* this vocabulary. Row 7
holds what the model learned about carbon only because carbon was
number 7 at training time. Reload the model tomorrow with a
reshuffled vocabulary and every row points at the wrong token, silently.
Sorting makes the encoding reproducible.

**Case matters.** `C` (index 7) and `c` (index 9) are different tokens
because they are different chemistry — uppercase is an aliphatic carbon,
lowercase an aromatic one. Lowercasing this "text", the way one might
in NLP preprocessing, would destroy the distinction.

## Step 3 — pad: make the strings rectangular

Molecules differ in length; tensors do not. Every sequence is therefore
framed and padded to a fixed `max_length`:

```python
X = vectorizer.transform(toy)
print(X)
```

Output:

```text
[[1 7 7 8 0 0 0 0 0 0 0 0 0]
 [1 7 7 3 6 8 4 8 0 0 0 0 0]
 [1 9 5 9 9 9 9 9 5 0 0 0 0]]
```

Read the first row against the vocabulary: `1` is `!`, `7` is `C`, `7`
is `C`, `8` is `O`, then zeros. `CCO` became `! C C O E E E E E E E E E`.

| Position | 0 | 1 | 2 | 3 | 4 … 12 |
|---|---|---|---|---|---|
| Token | `!` | `C` | `C` | `O` | `E` |
| Index | 1 | 7 | 7 | 8 | 0 |

The **start token** gives every sequence a common first position, which
matters for models that carry state from the beginning of the string —
it is a "begin reading" signal that no molecule can produce by itself.
The **padding** is what makes 2050 molecules of different lengths into
one rectangular array. Default `max_length` is *longest training
molecule + 5*; anything longer is truncated rather than rejected.

## Step 4 — embed: from indices to vectors

An integer index is not yet a useful input — index 8 is not "one more
than" index 7 in any meaningful sense. There are two ways to fix that,
and they are mathematically the same thing.

**One-hot.** Replace each index with a vector of zeros carrying a single
1:

```python
one_hot = vectorizer.transform(["CCO"], one_hot=True)
print(one_hot.shape, one_hot.dtype)
print(one_hot[0][:5])
```

Output:

```text
(1, 13, 10) int8
[[0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 1 0]
 [1 0 0 0 0 0 0 0 0 0]]
```

Five positions, ten vocabulary slots, one 1 per row — you can read
`! C C O E` straight off the diagonal-ish pattern.

**Embedding.** Keep the integers and let the model hold a lookup table:
one learned vector per token, `vocab_size` rows deep. This is the
"2vec" in smiles2vec. The two are equivalent, which is worth
demonstrating rather than asserting:

```python
import numpy as np

rng = np.random.default_rng(0)
E = rng.normal(size=(vectorizer.vocab_size, 4)).round(2)   # 4 numbers per token

sequence = vectorizer.transform(["CCO"])[0]
lookup = E[sequence]                        # what an Embedding layer does
matmul = one_hot[0] @ E                     # what a one-hot input would need

print("sequence :", sequence[:4])
print("identical:", np.allclose(lookup, matmul))
```

Output:

```text
sequence : [1 7 7 8]
identical: True
```

Multiplying a one-hot row by a matrix *is* selecting a row of that
matrix. An `Embedding` layer skips the multiplication and indexes
directly — same result, a fraction of the work. That is why
`transform()` returns integers by default and one-hot only on request.

The size difference on a real dataset is not subtle. Encoding all 2050
BBBP molecules gives a `(2050, 405)` `int32` matrix of **3.3 MB**; the
same data one-hot with a 42-token vocabulary is **34.9 MB**, and every
one of those extra bytes is a zero.

## Step 5 — mask: teach the model to ignore the padding

Padding is scaffolding, not chemistry, and a recurrent layer will
happily read all of it. Here is how much there is to read:

```python
from ikn_library.molecules import load_bbbp

smiles, y = load_bbbp().task("p_np")
lengths = np.array([len(s) for s in smiles])
print("molecules      :", len(smiles))
print("median length  :", int(np.median(lengths)))
print("95th percentile:", int(np.percentile(lengths, 95)))
print("longest        :", int(lengths.max()))

fitted = SmilesVectorizer().fit(smiles)
X_all = fitted.transform(smiles)
pad_fraction = (X_all == fitted.vocabulary["E"]).mean()
print("max_length     :", fitted.max_length)
print("padding share  :", round(float(pad_fraction), 4))
```

Output:

```text
molecules      : 2050
median length  : 45
95th percentile: 105
longest        : 400
max_length     : 405
padding share  : 0.8704
```

The typical molecule is 45 characters long, but one outlier at 400
characters sets `max_length` to 405 for everybody — so **87% of that
matrix is padding**. Two independent fixes:

- **`mask_zero=True`** on the embedding layer. Because the pad token is
  index 0, Keras propagates a mask and the recurrent layers skip those
  positions. This fixes the *learning*: without it, the network spends
  most of its capacity on a symbol that means "nothing here".
- **Set `max_length` yourself.** `SmilesVectorizer(max_length=120)`
  cuts the padding share from 0.87 to 0.57 and truncates only 53 of the
  2050 molecules. That fixes the *compute*, and it is why the LSTM
  example on the [usage page](smiles2vec.md#example-an-lstm-classifier)
  passes 120 rather than accepting the default.

## Why an unknown token exists

The vocabulary is learned from training molecules, so a molecule seen
later can contain a token that was never there. The reference
implementation raises `KeyError`; here the token maps to `?`:

```python
print(vectorizer.transform(["CCN"]))              # N was never in the toy set
print(vectorizer.inverse_transform(vectorizer.transform(["CCN"])))
```

Output:

```text
[[1 7 7 2 0 0 0 0 0 0 0 0 0]]
['CC?']
```

Nitrogen becomes index 2, the `?` token, and encoding proceeds. This is
a deliberate trade: a model that has never seen nitrogen cannot say
anything useful about it, but crashing at inference time on one unusual
molecule is worse than degrading on it. The `inverse_transform` round
trip makes the loss visible — `CCN` comes back as `CC?`, which is the
honest report of what the model actually received.

## What this representation assumes

Two assumptions are worth stating out loud before trusting a sequence
model on molecules.

**The same molecule has many valid SMILES.** `CCO`, `OCC` and `C(C)O`
are all ethanol. A sequence model reads them as three different
sentences and can give three different predictions. Standard responses:
train on *canonical* SMILES so each molecule has one spelling, or the
opposite — deliberately train on several random spellings per molecule
as data augmentation, so the model learns the invariance. Either is a
decision; drifting between them by accident is not.

**Position is meaning, and the meaning is not local.** Ring closure
digits pair up across arbitrary distances — in `c1ccccc1` the two `1`s
are six tokens apart and denote a single bond — and branch parentheses
nest. This is exactly why the architecture is recurrent or
convolutional rather than a bag of tokens: order and distance carry the
chemistry.

## Where to go next

- [SMILES to Sequences](smiles2vec.md) — the API, and an LSTM that
  reaches 0.87 accuracy on BBBP from raw strings.
- [Molecular Descriptors](featurize.md) — the other representation:
  fingerprints and descriptors, where order is discarded on purpose.
- G. B. Goh, N. O. Hodas, C. Siegel, and A. Vishnu, "SMILES2Vec: an
  interpretable general-purpose deep neural network for predicting
  chemical properties," arXiv:1712.02034, 2017.
