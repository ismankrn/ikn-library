# Drug-Drug Interactions

`load_drugbank_ddi` loads the DrugBank drug-drug interaction benchmark
(the DeepDDI dataset, Ryu et al. 2018): **191,808 drug pairs** over
**1,706 drugs**, each labeled with one of **86 interaction types**, with
the SMILES of both drugs included.

```python
from ikn_library.interactions import load_drugbank_ddi

data = load_drugbank_ddi()      # downloads once (~40 MB), caches locally
print(data)                     # 191808 pairs, 1706 drugs, 86 interaction types

smiles1, smiles2, types = data.arrays()
```

## The 86 interaction types

`interaction_types` summarizes them, most frequent first:

```python
print(data.interaction_types.head(3))
```

```text
                  count                                        description
interaction_type
49                60751  The risk or severity of adverse effects can be ...
47                34360  The metabolism of #Drug2 can be decreased when ...
73                23779  The serum concentration of #Drug2 can be increa...
```

The classes are extremely unbalanced — the largest holds 60,751 pairs,
the smallest just 6 — so treat the 86-way problem with care, or work
one type at a time.

## One type at a time

`binary_task` turns the multi-class dataset into the binary setting the
rest of the library expects: pairs of the requested type are positives,
and a random sample of pairs of *other* types forms the negatives.

```python
smiles1, smiles2, y = data.binary_task(47, negative_ratio=1.0, seed=42)
print(len(y), int(y.sum()))
```

```text
68720 34360
```

## Pair features: mind the symmetry

A drug-drug interaction is **undirected**: the pair (A, B) is the same
as (B, A). But naively concatenating fingerprints is *not* symmetric —
`concat(fp_A, fp_B)` differs from `concat(fp_B, fp_A)`, so the model
would have to learn the same relation twice. `pair_features` offers
both behaviors:

```python
from ikn_library.molecules import featurize
from ikn_library.interactions import pair_features

X1 = featurize(smiles1, method="morgan", n_bits=256)
X2 = featurize(smiles2, method="morgan", n_bits=256)

X = pair_features(X1, X2, method="sum")       # symmetric, 256 features
```

| `method=` | Result | Symmetric? | Use for |
|---|---|---|---|
| `"concat"` (default) | `2 * n_features` | No | **drug-target** pairs (different entity kinds), or directed relations |
| `"sum"`, `"mean"`, `"max"`, `"product"`, `"absdiff"` | `n_features` | **Yes** | **drug-drug** pairs and other undirected relations |

Verified on real data: with `method="sum"`, `pair_features(X1, X2)` and
`pair_features(X2, X1)` are element-wise identical; with `"concat"`
they are not.

!!! tip "Which symmetric combination?"
    `"sum"` (or `"mean"`) keeps how often a substructure appears across
    the pair; `"product"` marks substructures present in **both**
    drugs; `"absdiff"` highlights what only **one** of them has. They
    encode different hypotheses about the interaction — worth comparing
    empirically.

## Splitting

Because the same drug appears in many pairs, the leakage caveat from
[drug-target interactions](interactions.md#splitting-paired-data-honestly)
applies here too — pass both drug columns to `cold_split`:

```python
from ikn_library.interactions import cold_split

train, test = cold_split(data.frame["drug1_id"], data.frame["drug2_id"],
                         mode="drug", seed=42)
```

With `mode="drug"`, the drugs held out on the *first* side are absent
from training — the closest analogue to "can we predict interactions
for a newly approved drug?".

## References

- J. Y. Ryu, H. U. Kim, and S. Y. Lee, "Deep learning improves
  prediction of drug-drug and drug-food interactions," *PNAS*, 115(18),
  E4304-E4311, 2018.
  [doi:10.1073/pnas.1803294115](https://doi.org/10.1073/pnas.1803294115).
- D. S. Wishart et al., "DrugBank 5.0: a major update to the DrugBank
  database for 2018," *Nucleic Acids Research*, 46(D1), D1074-D1082,
  2018. [doi:10.1093/nar/gkx1037](https://doi.org/10.1093/nar/gkx1037).
- File as harmonized by the Therapeutics Data Commons: K. Huang et al.,
  *NeurIPS Datasets and Benchmarks*, 2021.
