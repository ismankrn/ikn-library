# Drug-Target Interactions

The `ikn_library.interactions` module loads drug-target interaction
(DTI) benchmarks, and `ikn_library.proteins` turns protein sequences
into numeric descriptors — together they make **paired** data usable by
every `Problem` in this library.

## Affinity benchmarks: Davis and KIBA

```python
from ikn_library.interactions import load_davis, load_kiba

davis = load_davis()            # downloads once (~20 MB), caches locally
print(davis)                    # 25,772 pairs, 68 drugs x 379 targets

smiles, sequences, y = davis.arrays()
```

| Dataset | Pairs | Drugs x Targets | Target value |
|---|---|---|---|
| **Davis** | 25,772 | 68 x 379 | Kd, converted to **pKd** = `-log10(Kd * 1e-9)` (5.0–10.8) |
| **KIBA** | 117,657 | 2,068 x 229 | KIBA score (integrates Ki, Kd, IC50) |

Both are **regression** targets. `load_davis(log_transform=False)`
keeps the raw Kd in nM; the pKd convention (the default) follows DeepDTA
and its successors, where higher means stronger binding. Each row of
`frame` carries `drug_id`, `smiles`, `target_id`, `sequence`, and
`affinity`.

## Classification benchmarks: Yamanishi

The four classic Yamanishi (2008) networks record only **positive**
interactions between KEGG drugs and human proteins, so negatives must
be sampled:

```python
from ikn_library.interactions import load_yamanishi

data = load_yamanishi("nuclear_receptor")   # or "enzyme", "ion_channel", "gpcr"
print(data)                                 # 90 interactions, 54 drugs x 26 targets

drug_ids, target_ids, y = data.pairs(negative_ratio=1.0, seed=42)
matrix = data.interaction_matrix()           # 0/1 drugs x targets
```

| Subset | Interactions |
|---|---|
| `enzyme` | 2,926 |
| `ion_channel` | 1,476 |
| `gpcr` | 635 |
| `nuclear_receptor` | 90 |

!!! warning "Sampled negatives are assumptions, not facts"
    `pairs()` draws negatives from drug-target combinations that are
    *not recorded* as interacting — the standard benchmarking
    convention, but an unobserved pair may simply be untested rather
    than truly non-interacting. Report the `negative_ratio` you used;
    it changes both the class balance and the achievable scores.

## Protein descriptors

`featurize_protein` computes sequence descriptors in pure numpy — no
extra dependencies:

```python
from ikn_library.proteins import featurize_protein

X = featurize_protein(sequences, method="aac")    # or "dpc", "ctd"
```

| `method=` | Features | Description |
|---|---|---|
| `"aac"` (default) | 20 | Amino acid composition: fraction of each residue |
| `"dpc"` | 400 | Dipeptide composition: fraction of each ordered pair |
| `"ctd"` | 147 | Composition / Transition / Distribution over 7 physicochemical properties (Dubchak et al., 1995) |

Non-standard characters (`X`, gaps, digits) are stripped before
computing; sequences left with no standard amino acids are dropped with
a warning, and a `y` passed alongside is kept aligned — the same
contract as [`featurize`](featurize.md) for molecules.

## Putting a DTI pipeline together

Pair features are the concatenation of a drug representation and a
protein representation:

```python
import numpy as np
import pandas as pd

from ikn_library.interactions import load_davis
from ikn_library.molecules import featurize
from ikn_library.proteins import featurize_protein

data = load_davis()
smiles, sequences, y = data.arrays()

X_drug = featurize(smiles, method="morgan", n_bits=512)
X_protein = featurize_protein(sequences, method="ctd")
X = pd.concat([X_drug.reset_index(drop=True),
               X_protein.reset_index(drop=True)], axis=1)   # 659 features
```

From here the usual machinery applies — [feature
selection](feature-selection.md) over the combined descriptor block,
[parameter optimization](parameter-optimization.md) for the model, and
for the Yamanishi classification setting also
[undersampling](undersampling.md).

!!! note "Splitting paired data honestly"
    A random split lets the *same* drug appear in both train and test,
    which inflates scores. Publication-grade DTI work reports
    **cold-drug** and **cold-target** splits (unseen drugs / unseen
    proteins in the test set) as well. Decide the split before tuning
    anything.

## References

- M. I. Davis et al., "Comprehensive analysis of kinase inhibitor
  selectivity," *Nature Biotechnology*, 29(11), 1046-1051, 2011.
  [doi:10.1038/nbt.1990](https://doi.org/10.1038/nbt.1990).
- J. Tang et al., "Making sense of large-scale kinase inhibitor
  bioactivity data sets: a comparative and integrative analysis,"
  *Journal of Chemical Information and Modeling*, 54(3), 735-743, 2014.
  [doi:10.1021/ci400709d](https://doi.org/10.1021/ci400709d).
- Y. Yamanishi, M. Araki, A. Gutteridge, W. Honda, and M. Kanehisa,
  "Prediction of drug-target interaction networks from the integration
  of chemical and genomic spaces," *Bioinformatics*, 24(13),
  i232-i240, 2008.
  [doi:10.1093/bioinformatics/btn162](https://doi.org/10.1093/bioinformatics/btn162).
- I. Dubchak, I. Muchnik, S. R. Holbrook, and S.-H. Kim, "Prediction of
  protein folding class using global description of amino acid
  sequence," *PNAS*, 92(19), 8700-8704, 1995.
  [doi:10.1073/pnas.92.19.8700](https://doi.org/10.1073/pnas.92.19.8700).
- Davis and KIBA files as harmonized by the Therapeutics Data Commons:
  K. Huang et al., "Therapeutics Data Commons: machine learning
  datasets and tasks for drug discovery and development," *NeurIPS
  Datasets and Benchmarks*, 2021.
