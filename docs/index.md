# ikn-library

A growing Python library of research and data-science tools. Its first
module provides nature-inspired metaheuristic algorithms for continuous
optimization, feature selection, and parameter optimization. More
components will be added over time.

## Installation

```bash
pip install ikn-library
```

With scikit-learn support for feature selection:

```bash
pip install "ikn-library[ml]"
```

Already installed? Upgrade to the latest release with:

```bash
pip install --upgrade ikn-library
```

Check the installed version with
`python -c "import ikn_library; print(ikn_library.__version__)"`.

## Quick example

```python
from ikn_library import Task
from ikn_library.problems import Sphere
from ikn_library.algorithms import AntColonyOptimization

task = Task(problem=Sphere(dimension=10), max_evals=10000)
algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
best_x, best_fitness = algo.run(task)

print("Best fitness:", best_fitness)
```

## How it works

The library follows a NiaPy-like workflow built from three pieces:

1. **`Problem`** — defines the search space and the objective function.
   Subclass it and implement `_evaluate` for custom problems.
2. **`Task`** — wraps a problem with a budget (`max_evals` / `max_iters`),
   counts evaluations, tracks the best solution, and records the
   convergence history.
3. **`Algorithm`** — a metaheuristic that consumes a task via
   `algorithm.run(task)` and returns `(best_x, best_fitness)`.

## Tutorials

- **[Getting Started](getting-started.md)** — tasks, custom problems,
  maximization, and the available algorithms.
- **[Feature Selection](feature-selection.md)** — wrapper-based feature
  selection with Binary ACO and a scikit-learn estimator.
- **[Parameter Optimization](parameter-optimization.md)** — tuning model
  hyperparameters (e.g. SVM `C` and `gamma`) with continuous ACO-R,
  including log-scale search spaces and convergence plotting.
- **[Microarray Data](microarray.md)** — loading NCBI GEO series into
  ML-ready tables (auto-download + cache, missing-value handling,
  variance filtering) and feeding them into feature selection.
- **[Ensemble Weights](ensemble.md)** — replacing majority voting with
  metaheuristic-optimized voting weights (and ensemble pruning with
  Binary ACO).
- **[Undersampling](undersampling.md)** — balancing imbalanced datasets
  by optimizing *which* majority samples to keep, with an exact-ratio
  constraint.
- **[Molecule Data (SIDER)](sider.md)** — loading the SIDER drug
  side-effect dataset as SMILES plus per-side-effect binary labels.
- **[Molecule Data (Tox21)](tox21.md)** — loading the Tox21 toxicity
  dataset (12 assays, missing labels handled per task).
- **[Molecule Data (BBBP, ClinTox, HIV)](moleculenet.md)** — three more
  MoleculeNet case-study datasets, from friendly (BBBP) to severely
  imbalanced (HIV).
- **[Molecular Descriptors](featurize.md)** — SMILES to features in one
  call: five fingerprints, physicochemical descriptors, and a Mordred
  backend.
- **[Drug-Target Interactions](interactions.md)** — Davis, KIBA, and
  Yamanishi benchmarks plus pure-numpy protein sequence descriptors,
  with leakage-aware cold splits.
- **[Drug-Drug Interactions](ddi.md)** — the DrugBank/DeepDDI benchmark
  and symmetric pair features.
- **[API Reference](api.md)** — full reference generated from the
  docstrings.

## Acknowledgments

The `Problem` / `Task` / `Algorithm` workflow is inspired by the API
design of [NiaPy](https://github.com/NiaOrg/NiaPy) — G. Vrbančič,
L. Brezočnik, U. Mlakar, D. Fister, and I. Fister Jr., "NiaPy: Python
microframework for building nature-inspired algorithms," *Journal of
Open Source Software*, 3(23), 613, 2018
([doi:10.21105/joss.00613](https://doi.org/10.21105/joss.00613)). All
algorithms and implementations in this library are written
independently; see the feature-selection and algorithm pages for the
literature each component is based on.
