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

## The 27 side-effect tasks

The table below lists every task name — the keyword you pass to
`task()` (any unique case-insensitive substring of it also works) —
together with its class counts and **imbalance ratio**
(majority count / minority count; 1.0 means perfectly balanced):

| Task (keyword for `task()`) | 1 (positive) | 0 (negative) | Imbalance ratio |
|---|---|---|---|
| `Hepatobiliary disorders` | 743 | 684 | 1.1 |
| `Metabolism and nutrition disorders` | 996 | 431 | 2.3 |
| `Product issues` | 22 | 1405 | 63.9 |
| `Eye disorders` | 876 | 551 | 1.6 |
| `Investigations` | 1151 | 276 | 4.2 |
| `Musculoskeletal and connective tissue disorders` | 997 | 430 | 2.3 |
| `Gastrointestinal disorders` | 1298 | 129 | 10.1 |
| `Social circumstances` | 251 | 1176 | 4.7 |
| `Immune system disorders` | 1024 | 403 | 2.5 |
| `Reproductive system and breast disorders` | 727 | 700 | 1.0 |
| `Neoplasms benign, malignant and unspecified (incl cysts and polyps)` | 376 | 1051 | 2.8 |
| `General disorders and administration site conditions` | 1292 | 135 | 9.6 |
| `Endocrine disorders` | 323 | 1104 | 3.4 |
| `Surgical and medical procedures` | 213 | 1214 | 5.7 |
| `Vascular disorders` | 1108 | 319 | 3.5 |
| `Blood and lymphatic system disorders` | 885 | 542 | 1.6 |
| `Skin and subcutaneous tissue disorders` | 1318 | 109 | 12.1 |
| `Congenital, familial and genetic disorders` | 253 | 1174 | 4.6 |
| `Infections and infestations` | 1006 | 421 | 2.4 |
| `Respiratory, thoracic and mediastinal disorders` | 1060 | 367 | 2.9 |
| `Psychiatric disorders` | 1016 | 411 | 2.5 |
| `Renal and urinary disorders` | 911 | 516 | 1.8 |
| `Pregnancy, puerperium and perinatal conditions` | 125 | 1302 | 10.4 |
| `Ear and labyrinth disorders` | 659 | 768 | 1.2 |
| `Cardiac disorders` | 988 | 439 | 2.3 |
| `Nervous system disorders` | 1304 | 123 | 10.6 |
| `Injury, poisoning and procedural complications` | 946 | 481 | 2.0 |

Reading the table:

- **Most balanced**: `Reproductive system and breast disorders`
  (727 vs 700, ratio 1.0) — a comfortable binary task.
- **Most imbalanced**: `Product issues` (22 vs 1,405, ratio 63.9) —
  with so few positives, plain accuracy is meaningless and specialized
  handling such as [undersampling](undersampling.md) or class-weighted
  metrics is essential.
- Note that for many tasks the **positive class is the majority**
  (most drugs *do* have, e.g., gastrointestinal side effects recorded)
  — check which side is the minority before choosing metrics.

## References

- M. Kuhn, I. Letunic, L. J. Jensen, and P. Bork, "The SIDER database
  of drugs and side effects," *Nucleic Acids Research*, 44(D1),
  D1075-D1079, 2016.
  [doi:10.1093/nar/gkv1075](https://doi.org/10.1093/nar/gkv1075).
- Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
  A. S. Pappu, K. Leswing, and V. Pande, "MoleculeNet: a benchmark for
  molecular machine learning," *Chemical Science*, 9(2), 513-530, 2018.
  [doi:10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A).
