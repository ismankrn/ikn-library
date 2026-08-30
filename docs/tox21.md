# Molecule Data (Tox21)

**Tox21** (Toxicology in the 21st Century, an NIH/EPA/FDA program)
screened thousands of compounds — drugs, industrial chemicals,
pesticides — against toxicity-related cellular pathways. The
MoleculeNet version loaded here contains **7,831 compounds** labeled
active (1) / inactive (0) on up to **12 assays**: 7 nuclear-receptor
pathways (`NR-*`) and 5 stress-response pathways (`SR-*`).

Two traits distinguish it from [SIDER](sider.md):

- **Missing labels**: not every compound was tested in every assay, so
  `task()` drops unlabeled compounds — each task has its own effective
  sample size.
- **Severe imbalance**: active compounds are rare (5–12% at best, under
  3% at worst) — accuracy is meaningless here; use F1/AUC and consider
  [undersampling](undersampling.md).

## Loading

```python
from ikn_library.molecules import load_tox21

data = load_tox21()      # downloads once (~120 KB), caches locally
print(data)              # <Tox21Dataset: 7831 compounds x 12 assay tasks>
```

Passing a local path (`load_tox21("tox21.csv.gz")`) skips the network;
the default cache lives under `~/.ikn_library/molecules/`, next to the
SIDER file.

## Per-assay labels

```python
smiles, y = data.task("NR-AhR")   # exact name or unique substring: "ahr"
print(len(y), int(y.sum()))
```

Output:

```text
6549 768
```

Only the 6,549 compounds actually tested in the NR-AhR assay are
returned (1,282 unlabeled ones are dropped), of which 768 (11.7%) are
active.

## The 12 assay tasks

Keyword for `task()`, class counts among *labeled* compounds, number of
unlabeled compounds, and the imbalance ratio (majority / minority):

| Task (keyword for `task()`) | 1 (active) | 0 (inactive) | missing | Imbalance ratio |
|---|---|---|---|---|
| `NR-AR` | 309 | 6956 | 566 | 22.5 |
| `NR-AR-LBD` | 237 | 6521 | 1073 | 27.5 |
| `NR-AhR` | 768 | 5781 | 1282 | 7.5 |
| `NR-Aromatase` | 300 | 5521 | 2010 | 18.4 |
| `NR-ER` | 793 | 5400 | 1638 | 6.8 |
| `NR-ER-LBD` | 350 | 6605 | 876 | 18.9 |
| `NR-PPAR-gamma` | 186 | 6264 | 1381 | 33.7 |
| `SR-ARE` | 942 | 4890 | 1999 | 5.2 |
| `SR-ATAD5` | 264 | 6808 | 759 | 25.8 |
| `SR-HSE` | 372 | 6095 | 1364 | 16.4 |
| `SR-MMP` | 918 | 4892 | 2021 | 5.3 |
| `SR-p53` | 423 | 6351 | 1057 | 15.0 |

Reading the table:

- Unlike SIDER, **the active class is always the minority** — often a
  small one (`NR-PPAR-gamma`: 186 actives vs 6,264 inactives, ratio
  33.7). This is the regime the
  [undersampling module](undersampling.md) was built for.
- The *least* imbalanced tasks (`SR-ARE`, `SR-MMP`, ratio ~5) are the
  usual starting points for modeling exercises.

## References

- R. Huang, M. Xia, D.-T. Nguyen, T. Zhao, S. Sakamuru, J. Zhao,
  S. A. Shahane, A. Rossoshek, and A. Simeonov, "Tox21 Challenge to
  build predictive models of nuclear receptor and stress response
  pathways as mediated by exposure to environmental chemicals and
  drugs," *Frontiers in Environmental Science*, 3:85, 2016.
  [doi:10.3389/fenvs.2015.00085](https://doi.org/10.3389/fenvs.2015.00085).
- Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
  A. S. Pappu, K. Leswing, and V. Pande, "MoleculeNet: a benchmark for
  molecular machine learning," *Chemical Science*, 9(2), 513-530, 2018.
  [doi:10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A).
