# Molecule Data (SIDER)

The `ikn_library.molecules` module loads molecular datasets as SMILES
strings plus labels. Its first loader is **SIDER** — the Side Effect
Resource (Kuhn et al., 2016) in its MoleculeNet form (Wu et al., 2018):
**1,427 marketed drugs**, each labeled with **27 binary side-effect
classes** (MedDRA system-organ classes).

## Loading

```python
from ikn_library.molecules import load_sider

data = load_sider()      # downloads once (~34 KB), caches locally
print(data)              # <SIDERDataset: 1427 drugs x 27 side-effect tasks>
print(data.tasks[:3])    # side-effect class names
```

Passing a local path (`load_sider("sider.csv.gz")`) skips the network;
the default cache lives under `~/.ikn_library/molecules/`.

## Per-side-effect labels

SIDER is a multi-label dataset, but most workflows model it **one side
effect at a time** (binary relevance). `task()` returns the SMILES and
the 0/1 labels for one class — by exact name or by a case-insensitive
substring that matches exactly one task:

```python
smiles, y = data.task("Hepatobiliary disorders")
smiles, y = data.task("hepato")                    # same task

print(len(smiles), int(y.sum()))                   # 1427 drugs, 743 positive
```

Output:

```text
1427 743
```

The full label matrix and raw table remain available as `data.labels`
(DataFrame, 27 columns) and `data.frame`.

## From SMILES to features

The loader deliberately returns raw SMILES: featurization is a modeling
choice. A common route is Morgan/ECFP fingerprints via RDKit
(`pip install rdkit`):

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
mols = [Chem.MolFromSmiles(s) for s in smiles]
X = np.array([generator.GetFingerprint(m) for m in mols])   # (1427, 1024) bits
```

That produces exactly the kind of high-dimensional binary feature
matrix the rest of this library is built for:
[feature selection](feature-selection.md) over fingerprint bits,
[undersampling](undersampling.md) for the imbalanced side-effect
classes, and [parameter optimization](parameter-optimization.md) or
[ensemble weights](ensemble.md) for the classifier on top.

## References

- M. Kuhn, I. Letunic, L. J. Jensen, and P. Bork, "The SIDER database
  of drugs and side effects," *Nucleic Acids Research*, 44(D1),
  D1075-D1079, 2016.
  [doi:10.1093/nar/gkv1075](https://doi.org/10.1093/nar/gkv1075).
- Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
  A. S. Pappu, K. Leswing, and V. Pande, "MoleculeNet: a benchmark for
  molecular machine learning," *Chemical Science*, 9(2), 513-530, 2018.
  [doi:10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A).
