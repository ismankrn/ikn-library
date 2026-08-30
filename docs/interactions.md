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

## Splitting paired data honestly

In paired data each drug appears in many rows (Davis: 68 drugs across
25,772 pairs, ~379 rows per drug). A **random** split therefore puts
the *same* drug in both train and test — the model can memorize
per-drug tendencies instead of learning the interaction, and the score
comes out inflated. `cold_split` holds out whole entities instead:

```python
from ikn_library.interactions import cold_split

train, test = cold_split(drug_ids, target_ids, test_size=0.2,
                         mode="drug", seed=42)
X_train, X_test = X[train], X[test]
```

| `mode=` | Held out | The question it answers |
|---|---|---|
| `"random"` | random pairs | "Fill in the blanks of an interaction matrix I already know" |
| `"drug"` (default) | whole drugs | "What does this **new drug** bind?" |
| `"target"` | whole proteins | "What binds this **new protein**?" |
| `"both"` | both | Hardest: neither entity seen in training |

### How much does it matter?

Same data, same features, same model (Random Forest on 4,000 Davis
pairs; Morgan-256 + amino-acid composition), only the split differs:

```text
 random: train= 3200 test= 800 | R2 = 0.288
   drug: train= 3204 test= 796 | R2 = -0.386
 target: train= 3185 test= 815 | R2 = 0.182
   both: train= 2582 test= 154 | R2 = -0.467
```

The random split looks like a working model; the cold-drug split
reveals that this feature set generalizes to unseen drugs *worse than
predicting the mean* (negative R²). Both numbers are "correct" — they
answer different questions, and only the cold ones speak to drug
discovery.

!!! warning "Decide the split before tuning anything"
    If hyperparameters, feature subsets, or ensemble weights are tuned
    under a random split and results are reported under a cold split,
    every one of those choices was optimized for the leaky setting.
    Fix the split scheme first, then use it consistently — including
    inside the fitness function of any metaheuristic.

    Note also that `mode="both"` keeps only pairs whose drug *and*
    target are held out, so its test set is much smaller (154 pairs
    above) — expect noisier estimates.

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
