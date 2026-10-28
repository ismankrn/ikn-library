---
title: 'ikn-library: nature-inspired metaheuristics with applied modules for cheminformatics and bioinformatics'
tags:
  - Python
  - metaheuristics
  - feature selection
  - hyperparameter optimization
  - cheminformatics
  - bioinformatics
  - benchmarking
authors:
  - name: Isman Kurniawan
    orcid: 0000-0003-3485-3063
    affiliation: 1
affiliations:
  - name: School of Computing, Telkom University, Bandung, Indonesia
    index: 1
date: 27 October 2026
bibliography: paper.bib
---

# Summary

`ikn-library` is a Python library of 38 nature-inspired metaheuristic
algorithms together with the applied components needed to point them at
real data: wrapper feature selection, hyperparameter search, ensemble
weight optimization, evolutionary undersampling, and loaders for gene
expression, molecular and drug–target datasets. Every algorithm
implements the same two methods behind one `Problem` / `Task` /
`Algorithm` interface, so swapping optimizers in an experiment is a
one-line change, and the same `Problem` can be driven by a population
method, a single-solution method such as simulated annealing, or by the
uniform-sampling baseline the library ships for comparison.

The library is distributed on PyPI, documented across 69 pages, and
covered by 796 automated tests running on Python 3.9–3.13.

# Statement of need

Several mature Python libraries already provide collections of
metaheuristics — `NiaPy` [@niapy], `MEALPY` [@mealpy] and `pymoo`
[@pymoo] among them — and a researcher who needs an optimizer is well
served by any of them. `ikn-library` is not an attempt to offer more
algorithms than these. It addresses two gaps that remain after an
optimizer has been chosen.

**The distance between an optimizer and a dataset.** Applying a
metaheuristic to feature selection on a gene-expression matrix, or to
tuning a model over a molecular fingerprint, requires glue code: an
objective that cross-validates an estimator on a candidate subset, a
decoder from a continuous search vector to hyperparameter values, a
repair operator that keeps a class ratio exact, a loader that turns a
GEO series matrix or a ChEMBL target identifier into a modelling table.
That glue is written and rewritten per project, and it is where
protocol errors enter. `ikn-library` ships it as tested components, and
documents the protocol each one assumes.

**The reliability of the comparison an optimizer is chosen by.**
Benchmark suites whose optima sit at the origin systematically flatter
algorithms whose update rules are not translation-invariant, a problem
documented across evolutionary computation [@kudela] and part of a
longer critique of metaphor-driven method proliferation [@sorensen].
Every algorithm in this library is therefore measured on four variants
of each benchmark — original, rotated, shifted, and both — and the
result is reported on its own documentation page, including where it is
unflattering. Applying this uncovered, for example, that the origin is
an exact fixed point of the Sine Cosine Algorithm's update rule, and
that Grey Wolf Optimizer [@gwo] loses 82 orders of magnitude on Sphere
when the optimum is translated away from zero. The library also ships
uniform random search [@bergstra] as a first-class algorithm, so the
control that makes a performance claim interpretable costs one line
rather than a bespoke script.

The library is used in the author's own research programme in QSAR
modelling, drug–target affinity prediction and expression-data
classification, and as teaching infrastructure for a research group of
43 student researchers. Its documentation is written for that audience:
every code example is accompanied by the output of actually running it,
and the evaluation protocol — hold out a test set before searching,
keep preprocessing inside the cross-validation, report the held-out
number — is stated rather than assumed.

# Functionality

- **Algorithms.** 38 metaheuristics for continuous and binary search
  spaces, including swarm, evolutionary, physics-based and
  immune-inspired families, the Komodo Mlipir Algorithm [@kma], NSGA-II
  for multi-objective problems [@nsga2], and uniform random search as a
  baseline. Optional early stopping (`patience`) is handled by the task
  rather than by each algorithm.
- **Applied problems.** Wrapper feature selection with any
  scikit-learn [@sklearn] estimator, following the weighted fitness
  formulation of @emary; hyperparameter optimization; ensemble weight
  optimization; evolutionary undersampling with exact-ratio repair.
- **Data modules.** NCBI GEO expression series, ChEMBL bioactivity by
  target identifier, MoleculeNet datasets, drug–target and drug–drug
  interaction data, molecular fingerprints and descriptors, and
  SMILES-to-sequence encoding.
- **Multi-objective layer.** Pareto utilities including
  `pareto_sort_indices`, a drop-in replacement for `argsort` that makes
  any population-based algorithm multi-objective without modifying it.

# Quality control

Correctness is covered by 796 automated tests run under continuous
integration on Python 3.9–3.13, including tests that assert each
algorithm's characteristic mechanism rather than only its output, and
tests that lock in structural findings such as translation invariance
where it holds. Documentation outputs are regenerated from execution
rather than written by hand.

# AI usage disclosure

**Tools used.** Anthropic Claude, accessed through the Claude Code
command-line interface. Two model versions were used over the
development period: Claude Fable 5 (101 commits) and Claude Opus 5
(37 commits). Each commit records the model that assisted it in a
`Co-Authored-By` trailer, so the distribution is auditable from the
public history.

**Nature and scope of assistance.** AI assistance was used for code
implementation and refactoring, test authoring, scripting of the
benchmark harnesses, drafting of the documentation, and drafting of
this paper. It was not used for the research questions, the choice of
evaluation protocol, or the interpretation of results.

**Confirmation of review.** The author reviewed, edited and validated
all AI-assisted output, and made the core design decisions: the scope
of the library, the four-variant evaluation protocol and the decision
to apply it to every algorithm, the standard that every documented
output be produced by execution, and the acceptance of findings
unfavourable to algorithms included in the library. All reported
numbers were produced by running the code and were verified against the
documentation before release.

# Acknowledgements

The API design was informed by `NiaPy` [@niapy]. Development was
supported by research grants from Telkom University, DRTPM RISTEKDIKTI,
and the Telkom University Matching Fund programme.

# References
