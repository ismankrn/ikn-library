# Molecular Descriptors

`featurize` turns a list of SMILES strings into a numeric feature table
in one call — fingerprints or physicochemical descriptors, with RDKit
(and optionally Mordred) doing the chemistry behind the scenes.

Requires RDKit:

```bash
pip install "ikn-library[chem]"
```

## One call from SMILES to features

```python
from ikn_library.molecules import load_tox21, featurize

data = load_tox21()
smiles, y = data.task("SR-MMP")

X, y = featurize(smiles, y, method="morgan", n_bits=1024)
```

Two conveniences over using RDKit directly:

- **Invalid SMILES are handled explicitly.** RDKit silently returns
  ``None`` for unparseable SMILES; `featurize` either drops those
  molecules with a warning (default) or raises
  (``on_invalid="raise"``).
- **`y` stays aligned.** Pass the labels along and rows dropped from
  `X` are dropped from `y` too — no silent misalignment.

The result is a `pandas.DataFrame` with named columns, ready for
`top_variance`, `zscore`, and every `Problem` in this library.

## Available methods

| `method=` | Features | Size | Notes |
|---|---|---|---|
| `"morgan"` (default) | Morgan/ECFP circular fingerprint | `n_bits` (default 1024) | The de-facto standard for molecular ML; `radius` (default 2) |
| `"maccs"` | MACCS keys | 167 | Short, standardized substructure keys |
| `"rdkit"` | RDKit path-based fingerprint | `n_bits` | Daylight-style bond paths |
| `"atompair"` | Atom-pair fingerprint | `n_bits` | Atom pairs + topological distance |
| `"torsion"` | Topological torsion fingerprint | `n_bits` | Four-atom sequential fragments |
| `"descriptors"` | RDKit physicochemical descriptors | 15 curated (default) | Named columns; see below |
| `"mordred"` | Mordred 2D descriptors | ~1,613 | Needs `pip install mordredcommunity` |

## Physicochemical descriptors

The curated default covers the descriptors most used in QSAR teaching
and practice — molecular weight, LogP, TPSA, H-bond donors/acceptors,
rotatable bonds, ring counts, and friends:

```python
X = featurize(smiles, method="descriptors")
print(list(X.columns))
```

```text
['MolWt', 'MolLogP', 'MolMR', 'TPSA', 'NumHDonors', 'NumHAcceptors',
 'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
 'NumSaturatedRings', 'NumAliphaticRings', 'HeavyAtomCount',
 'NumHeteroatoms', 'FractionCSP3', 'NumValenceElectrons']
```

Pass ``descriptor_names="all"`` for every 2D descriptor RDKit offers
(200+), or your own list of RDKit descriptor names.

## The Mordred backend

For the massive descriptor sets common in QSAR papers,
``method="mordred"`` computes ~1,613 2D descriptors via the
community-maintained Mordred package:

```bash
pip install mordredcommunity
```

```python
X, y = featurize(smiles, y, method="mordred")
```

Descriptors that cannot be computed for a molecule come back as NaN
(and some columns may be entirely NaN for your dataset) — filter or
impute before modeling, e.g. ``X = X.dropna(axis=1)``.

## Where this leads

`featurize` closes the loop for molecular data: `load_sider` /
`load_tox21` provide SMILES and labels, `featurize` turns them into a
matrix, and the rest of the library takes over —
[feature selection](feature-selection.md) over fingerprint bits or
descriptors, [undersampling](undersampling.md) for the imbalanced
assays, and [parameter optimization](parameter-optimization.md) or
[ensemble weights](ensemble.md) for the model on top.
