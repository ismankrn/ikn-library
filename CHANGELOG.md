# Changelog

All notable changes to `ikn-library` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/).

Install a specific release with `pip install ikn-library==<version>`, or
upgrade to the latest with `pip install --upgrade ikn-library`.

## [0.15.0] — 2026-09-05

Documentation only — the package code is unchanged from 0.14.0, apart
from one docstring.

A single theme runs through this release: **every page that reports a
number now reports it on data the search never saw.** Applying that
consistently overturned three claims the docs had been making.

### Changed

- **Every example splits before it selects or tunes.** Feature
  selection, hyperparameter optimization, hyperparameter tuning,
  undersampling and the multi-objective walkthrough all carve out a test
  set first, put the scaler inside a `Pipeline` so it is refitted per
  fold, and use an explicit `StratifiedKFold(shuffle=True,
  random_state=...)` instead of a bare `cv=5`. The three example scripts
  and the README snippets follow the same pattern, since those are what
  get copied.
- **Comparisons are made symmetric.** A search's best score is the
  maximum over many candidates on one set of folds; a default model's
  score is a single unselected number. Comparing them favours the search
  on every dataset, whether or not it helped. Both tuning pages now
  refit each candidate and compare on the test set.
- **The Keras workflow ships a recipe, not weights.** The
  *Hyperparameter Optimization* page no longer saves the best model
  during the search. The fitness records the architecture and the epoch
  where it peaked; the weights are discarded. After the search, three
  seeds refit that recipe on train + validation, and the test set is
  opened once: 0.9620 ± 0.0134. The page also documents when the saved
  artefact would have been the right choice, and what to do when the
  fitness is a cross-validation rather than a validation split.

### Added

- **Plotting Learning Curve** teaching note — the two plots that share
  that name, drawn side by side: loss per epoch (where this run overfits
  after epoch 33) and score per training-set size (where more data stops
  paying at 176 rows), with a table mapping each question to the curve
  that answers it.
- **SMILES2Vec Concept** teaching note — the encoding `SmilesVectorizer`
  performs, worked by hand: why the vocabulary is sorted (a trained
  embedding matrix is indexed by it), why case is never folded, that an
  embedding lookup and a one-hot matrix product are the same operation,
  and why 87% of an unconfigured BBBP encoding is padding.

### Fixed

- **Three results that did not survive being checked properly.** On
  breast cancer, wrapper feature selection repeated over five splits
  gains −0.0070 ± 0.0116 — the headline gain was one split's luck. On
  the same data, tuning an SVM by grid search or by ACO-R beats the
  defaults on cross-validation and ties with them on the test set, with
  identical predictions on all 114 rows. And the microarray page's claim
  that selection "raises the cross-validated KNN accuracy well above the
  all-probes baseline" was measured on the data the selection was made
  from; on a held-out split it evaporates. All three pages now report
  what the data supports.
- The microarray example used 5-NN with 161 training rows and 200
  probes, which scores below the majority-class rate. It now uses a
  scaled logistic regression, and a search budget suited to a
  200-dimensional binary space.

### Documented

- `FeatureSelectionProblem`'s `cv` argument accepts a splitter object,
  not only a fold count. It always did; the docstring now says so.
- *Multi-Objective Optimization* was removed from the navigation. The
  page and its API remain and its URL still resolves.

## [0.14.0] — 2026-09-01

Documentation only — the package code is unchanged from 0.13.0.

### Added

- **Using a tuned model**, on the renamed *Hyperparameter Tuning* page:
  assign `best_estimator_`, evaluate it on a held-out test set, save it
  with `pickle`, load it back and predict. Includes the two things that
  bite people — unpickling untrusted files, and pickles not being
  portable across library versions.
- **Checkpointing the best Keras model during a search**, on the
  *Hyperparameter Optimization* page. Four lines in `_evaluate` write the
  model to `.h5` whenever the validation loss improves, so the search
  ends holding a trained model rather than just the winning vector — 30
  models trained, 3 written to disk. Loading it and predicting needs no
  retraining at all. The page also records that Keras 3 treats `.h5` as
  legacy and recommends `.keras`.

### Changed

- Two teaching pages renamed: *Comparing Tuning Results* is now
  **Hyperparameter Tuning**, and *Parameter Optimization* is now
  **Hyperparameter Optimization**. Page titles and every cross-reference
  were updated to match; filenames and URLs are unchanged, so existing
  links keep working.
- Both renamed pages now hold out a test set and fit their scaler inside
  a `Pipeline` or on the training split only. The previous examples
  scaled the full dataset before splitting, which leaks test statistics
  into training, and had no test set to demonstrate prediction on. All
  affected numbers were re-measured.

### Fixed

- The two comparison bar charts drew per-fold standard deviation as
  error-bar whiskers, which put a vertical line through the value label
  above each bar. The charts now show the value alone; the spread moved
  into the surrounding prose rather than being dropped.

## [0.13.0] — 2026-08-31

### Added

- **Grasshopper Optimisation Algorithm** — built around an explicit
  short-range **repulsion**, which nothing else in the library has:
  grasshoppers repel when close and attract at medium range, and the
  separation where the force changes sign is the swarm's *comfort zone*.
  Spacing comes from a force law rather than from randomness layered on
  attraction.

  Two properties are worth knowing before using it, both asserted by
  tests. Its published update rule contains **no random term at all**,
  so the algorithm is deterministic after initialisation and gets one
  shot at finding the right basin. And `intensity` and
  `attraction_length` do not tune the repulsion's strength — they decide
  whether it exists, because a badly chosen pair removes it entirely.

  Unlike the four other algorithms from the same research group in this
  library, GOA has **no origin bias**: its scores vary by under a factor
  of 1.7 across all four benchmark variants, the third-flattest profile
  here.

## [0.12.0] — 2026-08-31

### Added

- **Whale Optimization Algorithm** — each whale either swims toward the
  best solution in a straight line or spirals around it, chosen by a
  coin flip. Its two halves fail differently under transformation: the
  encircling move scales the target's absolute coordinates and breaks
  when the optimum is shifted, while the spiral uses a plain difference
  and does not. Having one robust half is why a shift costs it far less
  than it costs [Grey Wolf](https://ikn-library.readthedocs.io/en/latest/algorithm-details/gwo/)
  or [Sine Cosine](https://ikn-library.readthedocs.io/en/latest/algorithm-details/sca/).
  Best of the four Mirjalili algorithms in the library on every
  benchmark variant.
- This changelog, published to the documentation site.

### Fixed

- The README's algorithm list had silently lost `HybridBatAlgorithm` and
  `HybridSelfAdaptiveBatAlgorithm`. It is now generated from the package
  registry and cross-checked against `__all__`, so it cannot drift
  again.

## [0.11.0] — 2026-08-31

Eleven algorithms, a shared Lévy helper, and a benchmark methodology now
applied to every entry in the library.

### Added

- **Harris Hawks Optimization** — six moves gated by the prey's escaping
  energy. The strongest all-round performer measured here, and the only
  algorithm whose ranking survives every benchmark transformation.
- **Krill Herd** — three motion terms with inertia plus an
  inverse-fitness-weighted food centre. Translation- *and*
  rotation-invariant: its scores vary by under 15% across all four
  benchmark variants.
- **Particle Swarm Optimization** — the reference baseline. Several
  pages record published arguments that a later metaheuristic
  re-describes something already known; PSO is usually what it
  re-describes.
- **Self-Adaptive Differential Evolution (jDE)** — each individual
  carries its own `F` and `CR`, kept only if the trial they produced
  won. Beats this library's own hand-tuned DE on all four benchmark
  variants.
- **Hybrid Bat Algorithm** and **Hybrid Self-Adaptive Bat Algorithm** —
  with `BatAlgorithm`, these form one lineage, so the docs can report
  what each addition actually bought.
- **Lion Optimization Algorithm** — prides and nomads across seven
  operators; the only heterogeneous population in the library.
- **Monarch Butterfly Optimization**, **Monkey King Evolution**,
  **Moth-Flame Optimization**, **Sine Cosine Algorithm**.
- `levy_flight` is now public API, in `ikn_library.algorithms`.

### Changed

- **Benchmark methodology.** Earlier releases checked rotated variants
  alone, which cannot detect an origin bias: a rotation turns a function
  about the origin, so an optimum at zero stays where it was. Every
  algorithm is now measured on four variants — plain, rotated, shifted,
  and both — and the *Detail of Algorithm* index explains what each one
  isolates.
- Mantegna's Lévy routine, previously duplicated in Cuckoo Search and
  Flower Pollination, moved to a shared `levy_flight` helper. The
  refactor is bit-identical for seeded runs.
- `BatAlgorithm` and `DifferentialEvolution` gained internal extension
  points so their hybrid and self-adaptive variants can reuse their
  operators rather than copy them. Both refactors are bit-identical for
  seeded runs.

### Documented

Applying the four-variant check found that **Sine Cosine's entire
benchmark row is an artefact** — the origin is an exact fixed point of
its update rule, and shifting the optimum by 0.1 costs twenty-four
orders of magnitude — that **Monarch Butterfly** loses a factor of
200,000 under rotation, and that **Grey Wolf** and **Fireworks** share a
milder version of the same origin bias. The comparison table carries
four footnotes as a result.

## [0.10.0] — 2026-08-31

Eighteen algorithms, a multi-objective layer, and a documentation page
per algorithm.

### Added

- **Eighteen single-objective algorithms**: Artificial Bee Colony,
  Bacterial Foraging, Bees, Camel, Cat Swarm, Clonal Selection, Coral
  Reefs, Cuckoo Search, Differential Evolution, Firefly, Fireworks,
  Fish School Search, Flower Pollination, Forest Optimization,
  Gravitational Search, Grey Wolf, Harmony Search, and Komodo Mlipir.
- **Multi-objective optimization**: `MultiObjectiveProblem`,
  `MultiObjectiveTask` with a Pareto archive, `NSGA2`, and Pareto
  utilities including `pareto_sort_indices` — a drop-in replacement for
  `argsort` that makes population-based algorithms multi-objective
  without modifying them.
- **Detail of Algorithm** documentation: one page per algorithm with
  equations, pseudocode, parameters, measured benchmarks, and
  references.
- Worked example of turning a Pareto front into a trained model.

### Removed

- Flowcharts from the algorithm detail pages — Mermaid did not render
  in production.
- The multi-objective Simulated Annealing example: a single-solution
  method cannot map a Pareto front, so the recipe applies only to
  population-based algorithms.

## [0.9.0] — 2026-08-30

Applied modules for bioinformatics and cheminformatics.

### Added

- **Algorithms**: real-coded Genetic Algorithm with non-uniform
  mutation, and the Bat Algorithm.
- **`sampling`** — evolutionary undersampling with exact-ratio repair.
- **`molecules`** — SIDER, Tox21, BBBP, ClinTox, and HIV loaders;
  `featurize` for SMILES → fingerprints and descriptors via RDKit with a
  Mordred backend; `SmilesVectorizer` for smiles2vec-style sequence
  encoding.
- **`interactions`** — Davis, KIBA, and Yamanishi drug–target datasets,
  DrugBank drug–drug interactions, `pair_features`, and `cold_split`
  for leakage-aware splits of paired data.
- **`proteins`** — sequence descriptors.
- *Microarray Data Concept* teaching note.

## [0.8.0] — 2026-08-30

### Added

- `ALGORITHMS.md`, the full algorithm list with descriptions,
  parameters, and references, published to the docs site.

### Changed

- **The MIT license was restored.** It had been removed in 0.2.0;
  releases 0.2.0 through 0.7.0 carried no license.

## [0.7.0] — 2026-08-30

### Added

- **Acknowledgments** section with formal citations: NiaPy for the API
  design it inspired, Emary et al. for the feature-selection fitness
  formulation, and Li et al. (2016) for the weighted-ensemble method.
- Feature-selection tutorial: before/after `cross_val_score` comparison
  with a bar chart.
- Parameter-optimization tutorial: categorical and integer decoding, and
  an MLP architecture search in both scikit-learn and Keras with a
  discussion of validation-loss fitness.
- *Ensemble Weights Concept* teaching note.
- Upgrade instructions in the README and docs home.

## [0.6.0] — 2026-08-29

### Added

- **Simulated Annealing** with temperature-coupled step decay.

## [0.5.0] — 2026-08-29

### Added

- **`ensemble`** — metaheuristic-optimized voting weights and ensemble
  pruning, with the three-split train/validate/test protocol the docs
  recommend.

## [0.4.0] — 2026-08-29

### Added

- Expression normalization for microarray data: log2, quantile,
  z-score, and median centering.
- Teaching notes on comparing tuning results from `cv_results_` and on
  convergence plotting, both with real outputs and charts.

## [0.3.0] — 2026-08-29

### Added

- **`microarray`** — NCBI GEO loader producing ML-ready tables, with
  local caching and missing-value handling, plus a full
  GEO-to-feature-selection pipeline example.

## [0.2.0] — 2026-08-29

### Added

- Documentation site (MkDocs, published to Read the Docs).
- Parameter-optimization tutorial.

### Changed

- Library scope broadened beyond metaheuristics.

### Removed

- The MIT license (restored in 0.8.0).

## [0.1.0] — 2026-08-29

Initial release.

### Added

- `Problem`, `Task`, and `Algorithm` — the three-part framework every
  algorithm in the library is built on.
- **Ant Colony Optimization for continuous domains (ACO-R)** and
  **Binary Ant Colony Optimization**.
- `FeatureSelectionProblem` for wrapper-based feature selection with any
  scikit-learn estimator.
- GitHub Actions CI, and automated PyPI publishing on version tags via
  Trusted Publishing.

[0.15.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.15.0
[0.14.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.14.0
[0.13.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.13.0
[0.12.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.12.0
[0.11.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.11.0
[0.10.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.10.0
[0.9.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.9.0
[0.8.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.8.0
[0.7.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.7.0
[0.6.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.6.0
[0.5.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.5.0
[0.4.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.4.0
[0.3.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.3.0
[0.2.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.2.0
[0.1.0]: https://github.com/ismankrn/ikn-library/releases/tag/v0.1.0
