# Molecule Data (BBBP, ClinTox, HIV)

Three more MoleculeNet classification datasets, loaded with the same
pattern as [SIDER](sider.md) and [Tox21](tox21.md): download once,
cache locally, and get `(smiles, y)` per task.

```python
from ikn_library.molecules import load_bbbp, load_clintox, load_hiv

bbbp = load_bbbp()
smiles, y = bbbp.task("p_np")            # blood-brain barrier penetration

clintox = load_clintox()
smiles, y = clintox.task("CT_TOX")       # failed trials for toxicity

hiv = load_hiv()
smiles, y = hiv.task("HIV_active")       # inhibition of HIV replication
```

## The datasets at a glance

All statistics computed from the actual files:

| Dataset | Compounds | Task (keyword for `task()`) | 1 (positive) | Imbalance ratio | Character |
|---|---|---|---|---|---|
| **BBBP** | 2,050 | `p_np` | 1,567 (76.4%) | 3.2 | The friendly starter: single task, mild imbalance — note the **positive class is the majority** |
| **ClinTox** | 1,484 | `FDA_APPROVED` | 1,390 (93.7%) | 14.8 | Heavily skewed toward approved drugs |
| **ClinTox** | 1,484 | `CT_TOX` | 112 (7.5%) | 12.2 | Rare-event prediction: drugs that failed trials for toxicity |
| **HIV** | 41,127 | `HIV_active` | 1,443 (3.5%) | 27.5 | Large and severely imbalanced — the prime case study for [undersampling](undersampling.md) |

Extra metadata columns stay available in `data.frame` but are excluded
from the labels: BBBP keeps the compound `name` and `num`, HIV keeps
the raw three-way screening outcome `activity` (CI/CM/CA).

## A complete case-study skeleton

The pieces chain together the same way for any of these datasets:

```python
from ikn_library.molecules import load_hiv, featurize

data = load_hiv()
smiles, y = data.task("HIV_active")
X, y = featurize(smiles, y, method="morgan", n_bits=1024)

# ... then split train/val/test and hand X, y to UndersamplingProblem,
# FeatureSelectionProblem, or a model-tuning Problem of your own.
```

!!! note "Invalid SMILES"
    A handful of SMILES in these files (BBBP is known for this) cannot
    be parsed by RDKit. `featurize` handles that: the affected rows are
    dropped from `X` *and* `y` with a warning, keeping the two aligned.

## Reference

Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
A. S. Pappu, K. Leswing, and V. Pande, "MoleculeNet: a benchmark for
molecular machine learning," *Chemical Science*, 9(2), 513-530, 2018.
[doi:10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A).
