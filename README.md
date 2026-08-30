# ikn-library

A growing Python library of research and data-science tools. Its first
module provides nature-inspired metaheuristic algorithms for continuous
optimization, feature selection, and parameter optimization. More
components will be added over time.

## Installation

```bash
pip install ikn-library
```

Already installed? Upgrade to the latest release with:

```bash
pip install --upgrade ikn-library
```

Or from source (development mode):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import AntColonyOptimization

task = Task(problem=Sphere(dimension=10), max_evals=10000)
algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
best_x, best_fitness = algo.run(task)

print("Best fitness:", best_fitness)
print("Best solution:", best_x)
```

## Custom problems

Subclass `Problem` and implement `_evaluate` — for example, a
cross-validation score for hyperparameter optimization:

```python
import numpy as np
from ikn_library.problems import Problem

class MyProblem(Problem):
    def __init__(self, dimension=10):
        super().__init__(dimension, lower=-10.0, upper=10.0)

    def _evaluate(self, x):
        return float(np.sum(np.abs(x)))
```

Use `OptimizationType.MAXIMIZATION` in the `Task` when higher is better
(e.g. accuracy).

## Feature selection

Wrapper-based feature selection with a scikit-learn estimator
(`pip install ikn-library[ml]`):

```python
from sklearn.datasets import load_breast_cancer

from ikn_library import Task
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

X, y = load_breast_cancer(return_X_y=True)
problem = FeatureSelectionProblem(X, y, cv=5, alpha=0.99)
task = Task(problem=problem, max_evals=1000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)

print("Selected features:", problem.selected_features(best_x))
```

The fitness balances the cross-validated score against the subset size:
`alpha * (1 - cv_score) + (1 - alpha) * n_selected / n_features`.

## Parameter optimization

Tune model hyperparameters by subclassing `Problem`: each dimension is
one hyperparameter, and `_evaluate` returns the cross-validated score
(searched in log scale where appropriate):

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task
from ikn_library.problems import Problem
from ikn_library.algorithms import AntColonyOptimization

class SVMTuning(Problem):
    def __init__(self, X, y):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y = X, y

    def _evaluate(self, x):
        model = SVC(kernel="rbf", C=10.0 ** x[0], gamma=10.0 ** x[1])
        return cross_val_score(model, self.X, self.y, cv=5).mean()

task = Task(problem=SVMTuning(X, y), max_evals=150,
            optimization_type=OptimizationType.MAXIMIZATION)
best_x, best_score = AntColonyOptimization(population_size=10, seed=42).run(task)
```

See the full tutorial:
[Parameter Optimization](https://ikn-library.readthedocs.io/en/latest/parameter-optimization/)
and the runnable script [examples/parameter_optimization.py](examples/parameter_optimization.py).

## Algorithms

Currently available: `AntColonyOptimization` (ACO-R, continuous),
`BatAlgorithm` (continuous), `BinaryAntColonyOptimization`
(binary/subsets), `GeneticAlgorithm` (real-coded, continuous), and
`SimulatedAnnealing` (continuous). See
**[ALGORITHMS.md](ALGORITHMS.md)** for the full list with descriptions,
key parameters, and references.

## Microarray data

Load NCBI GEO microarray series into ML-ready tables — downloaded once,
cached locally, with missing-value handling built in:

```python
from ikn_library.microarray import load_geo, quantile_normalize, top_variance, zscore

data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
X = quantile_normalize(data.X)  # identical distribution for every sample
X = top_variance(X, 500)        # (202 samples, 500 most variable probes)
X = zscore(X)                   # standardize per probe
y = data.y("disease")           # UC vs Normal labels from sample metadata
```

Normalization helpers: `log2_transform`, `quantile_normalize`,
`median_center`, and `zscore`.

The result plugs directly into `FeatureSelectionProblem` — see
[examples/microarray_pipeline.py](examples/microarray_pipeline.py) for the
full GEO-to-feature-selection pipeline.

## Ensemble weight optimization

Replace a Random Forest's majority voting with metaheuristic-optimized
voting weights — train on one split, optimize weights on a validation
split, report on a test split:

```python
from ikn_library.ensemble import EnsembleWeightProblem, tree_proba_matrix

P_val = tree_proba_matrix(forest, X_val)     # (n_samples, n_trees) probabilities
problem = EnsembleWeightProblem(P_val, y_val)
task = Task(problem=problem, max_evals=4000)
best_x, _ = AntColonyOptimization(seed=42).run(task)

y_pred = problem.predict(best_x, tree_proba_matrix(forest, X_test))
```

Running the same problem with `BinaryAntColonyOptimization` performs
ensemble pruning (0/1 weights = drop/keep members). See
[examples/ensemble_weight_optimization.py](examples/ensemble_weight_optimization.py).

## Undersampling for imbalanced data

Balance an imbalanced dataset by optimizing *which* majority-class
samples to keep (evolutionary undersampling) — the minority class is
kept in full, and every candidate is repaired to an exact class ratio:

```python
from ikn_library.sampling import UndersamplingProblem

problem = UndersamplingProblem(X_train, y_train, X_val, y_val,
                               target_ratio=1.0, metric="f1")
task = Task(problem=problem, max_evals=3000)
best_x, _ = BinaryAntColonyOptimization(seed=42).run(task)

X_reduced, y_reduced = problem.resampled_data(best_x)
```

See [examples/undersampling.py](examples/undersampling.py).

## Molecule data (SIDER, Tox21)

Load molecular datasets (MoleculeNet versions) as SMILES strings plus
binary labels, one task at a time:

```python
from ikn_library.molecules import load_sider, load_tox21

sider = load_sider()                   # 1,427 drugs x 27 side-effect tasks
smiles, y = sider.task("Hepatobiliary disorders")  # or a substring: "hepato"

tox21 = load_tox21()                   # 7,831 compounds x 12 toxicity assays
smiles, y = tox21.task("NR-AhR")       # unlabeled compounds dropped per task
```

Also available: `load_bbbp` (blood-brain barrier), `load_clintox`
(clinical-trial toxicity), and `load_hiv` (41k compounds, 3.5% active —
a prime undersampling case study).

Turn SMILES into features in one call (`pip install ikn-library[chem]`):

```python
from ikn_library.molecules import featurize

X, y = featurize(smiles, y, method="morgan", n_bits=1024)
# methods: morgan, maccs, rdkit, atompair, torsion, descriptors, mordred
```

Or vectorize SMILES as sequences for deep learning (smiles2vec style —
no RDKit needed):

```python
from ikn_library.molecules import SmilesVectorizer

vectorizer = SmilesVectorizer().fit(smiles)
X = vectorizer.transform(smiles)          # (n, max_length) token indices
print(vectorizer.vocabulary_table())      # the symbol -> index dictionary
```

## Drug-target interactions

Load DTI benchmarks and turn protein sequences into descriptors (pure
numpy — no extra dependencies):

```python
from ikn_library.interactions import load_davis, load_yamanishi
from ikn_library.proteins import featurize_protein

davis = load_davis()                      # 25,772 pairs, 68 drugs x 379 targets
smiles, sequences, y = davis.arrays()     # y = pKd
X_protein = featurize_protein(sequences, method="ctd")   # aac / dpc / ctd

yam = load_yamanishi("nuclear_receptor")  # classification, negatives sampled
drug_ids, target_ids, y = yam.pairs(negative_ratio=1.0, seed=42)
```

Drug-drug interactions (DrugBank/DeepDDI: 191,808 pairs, 86 interaction
types) with symmetric pair features:

```python
from ikn_library.interactions import load_drugbank_ddi, pair_features
from ikn_library.molecules import featurize

ddi = load_drugbank_ddi()
smiles1, smiles2, y = ddi.binary_task(47, seed=42)   # one interaction type
X = pair_features(featurize(smiles1), featurize(smiles2), method="sum")
```

Split paired data without entity leakage — `cold_split` holds out whole
drugs or proteins instead of individual pairs:

```python
from ikn_library.interactions import cold_split

train, test = cold_split(drug_ids, target_ids, mode="drug", seed=42)
# modes: "drug" (unseen drugs), "target", "both", "random"
```

## Documentation

Full documentation: [ikn-library.readthedocs.io](https://ikn-library.readthedocs.io)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)

## Acknowledgments

- The `Problem` / `Task` / `Algorithm` workflow of this library is
  inspired by the API design of
  [NiaPy](https://github.com/NiaOrg/NiaPy): G. Vrbančič, L. Brezočnik,
  U. Mlakar, D. Fister, and I. Fister Jr., "NiaPy: Python microframework
  for building nature-inspired algorithms," *Journal of Open Source
  Software*, 3(23), 613, 2018.
  [doi:10.21105/joss.00613](https://doi.org/10.21105/joss.00613).
  All algorithms and implementations in this library are written
  independently.
- The weighted feature-selection fitness
  (`alpha * (1 - score) + (1 - alpha) * n_selected / n_features`) follows
  the standard wrapper formulation used, among others, in E. Emary,
  H. M. Zawbaa, and A. E. Hassanien, "Binary grey wolf optimization
  approaches for feature selection," *Neurocomputing*, 172, 371–381,
  2016, and in J. Too's
  [Wrapper-Feature-Selection-Toolbox](https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox).
- The ensemble weight-optimization scheme and its train/validation/test
  protocol follow D. Li, L. Luo, W. Zhang, F. Liu, and F. Luo, "A
  genetic algorithm-based weighted ensemble method for predicting
  transposon-derived piRNAs," *BMC Bioinformatics*, 17:329, 2016.
  [doi:10.1186/s12859-016-1206-3](https://doi.org/10.1186/s12859-016-1206-3).
