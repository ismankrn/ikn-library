# Changelog

All notable changes to `ikn-library` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/).

Install a specific release with `pip install ikn-library==<version>`, or
upgrade to the latest with `pip install --upgrade ikn-library`.

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
