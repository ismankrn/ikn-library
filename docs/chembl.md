# Bioactivity Data (ChEMBL)

`load_chembl_target` downloads every measured activity for one ChEMBL
target and returns a table ready for QSAR modelling: one row per
molecule, a SMILES string, and a pChEMBL value to regress on.

```python
from ikn_library.molecules import load_chembl_target

data = load_chembl_target("CHEMBL279", max_records=2000)
print(data)
print()
print(data.report())
```

Output:

```text
<ChemblDataset CHEMBL279 IC50: 1236 molecules, pChEMBL 4.04-9.70>

records fetched  : 2000
with SMILES      : 2000
relation '='     : 1615
units 'nM'       : 1615
with pChEMBL     : 1605
unique molecules : 1236
```

`CHEMBL279` is VEGFR2. Find an identifier by searching a protein name
on [the ChEMBL website](https://www.ebi.ac.uk/chembl/) and taking the
`CHEMBL…` code from the target page.

## What the funnel is for

Two thousand records became 1,236 molecules, and the report says
exactly where the rest went. Each step is a modelling decision, not
housekeeping:

- **`relation '='`** drops 385 records here. A measurement recorded as
  `>10000 nM` means "we stopped looking" — the compound was inactive
  beyond the assay's range. Feeding 10000 to a regression as though it
  were a measured value teaches the model something false. Pass
  `relation=None` to keep them, but then handle censoring deliberately.
- **`units 'nM'`** keeps one unit system so the numbers are comparable.
- **`with pChEMBL`** requires ChEMBL's own $-\log_{10}$ of the molar
  activity. It is the standard QSAR target because potencies span
  orders of magnitude: pChEMBL 9 is a thousand times more potent than
  pChEMBL 6, and the log scale makes that a difference of 3 rather
  than a factor the model has to learn.
- **`unique molecules`** collapses repeated measurements. 1,605
  measurements covered 1,236 molecules, so some compounds were tested
  several times — by default the **median** is kept, and
  `n_measurements` records how many there were. Use
  `aggregate=None` to keep every measurement as its own row.

Nothing is dropped silently. If a reviewer asks why your table has
fewer rows than ChEMBL reports for the target, `data.report()` is the
answer.

## Turning it into a classification task

```python
from ikn_library.molecules import pchembl_to_binary

y = pchembl_to_binary(data.y, threshold=6.0)
print(f"active (pChEMBL > 6.0)  : {int(y.sum())}")
print(f"inactive                : {int((1 - y).sum())}")
```

Output:

```text
active (pChEMBL > 6.0)  : 953
inactive                : 283
```

!!! warning "The threshold is a convention, and the classes are skewed"
    `6.0` (1 micromolar) is the common cut-off, but it is a choice, not
    a property of the data — report it with any result that depends on
    it, and check how much the result moves when it changes.

    Note the imbalance: 953 active against 283 inactive, the opposite
    of what a naive expectation would give. ChEMBL records what people
    measured and published, and compounds get measured because someone
    already had reason to think they would work. It is not a random
    sample of chemical space, and a classifier trained on it inherits
    that bias.

## End to end: fingerprints and a model

The output plugs straight into the rest of the library — this is the
[featurize](featurize.md) step followed by any scikit-learn model:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ikn_library.molecules import featurize

X, y = featurize(data.smiles, data.y, method="morgan")
print("fingerprints:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train)
pred = model.predict(X_test)

print(f"test MAE  : {mean_absolute_error(y_test, pred):.3f} pChEMBL units")
print(f"test R2   : {r2_score(y_test, pred):.3f}")
print(f"baseline  : {mean_absolute_error(y_test, np.full_like(y_test, y_train.mean())):.3f}"
      f" (predicting the training mean)")
```

Output:

```text
fingerprints: (1236, 1024)
test MAE  : 0.513 pChEMBL units
test R2   : 0.575
baseline  : 0.804 (predicting the training mean)
```

The baseline row is the point: 0.513 against 0.804 is the part the
fingerprints earned. From here the same table feeds
[feature selection](feature-selection.md) over the 1,024 bits, or
[SMILES sequences](smiles2vec.md) instead of fingerprints.

## Downloading, caching and the rate limit

The first call to a target is slow and the rest are instant:

| | time |
|---|---|
| first call, 2,000 records | 271 s |
| same call again | 0.01 s |

The cache lives in `~/.ikn_library/chembl/` and holds the **unfiltered**
records, so changing `relation`, `units`, `organism` or `aggregate`
re-cleans in memory without touching the network.

EBI's public API rate-limits sustained paging, and a well-studied
target has a lot of it — `CHEMBL279` alone has over 17,000 IC50
records. Requests are retried with an exponential backoff and spaced
politely, so a large download is slow rather than broken. Two ways to
keep it short: `max_records` caps the fetch, and the cache means you
pay only once.

!!! note "Record the release with your results"
    ChEMBL grows with every release, so the same target returns more
    records next year and your counts will not reproduce. Name the
    release in your methods section:

    ```python
    from ikn_library.molecules import chembl_status

    status = chembl_status()
    print(status["chembl_db_version"], status["chembl_release_date"])
    ```

    ```text
    ChEMBL_37 2026-05-01
    ```

    The numbers on this page were produced with **ChEMBL_37**
    (released 2026-05-01).

## Options

| Argument | Default | Effect |
|---|---|---|
| `activity_type` | `"IC50"` | `"Ki"`, `"EC50"`, `"Kd"`, … |
| `relation` | `"="` | `None` keeps censored `>` / `<` records |
| `units` | `"nM"` | `None` keeps all unit systems |
| `organism` | `None` | e.g. `"Homo sapiens"` — some targets mix species |
| `aggregate` | `"median"` | `"mean"`, `"min"`, `"max"`, or `None` for one row per measurement |
| `max_records` | `None` | cap the download |
| `refresh` | `False` | re-download over the cache |

`fetch_chembl_activities` returns the raw, unfiltered table if you want
to do the cleaning yourself.
