# Contributing to ikn-library

Contributions are welcome, and so are questions. This page covers the
three things people usually need: how to get help, how to report a
problem, and how to contribute code.

The project is maintained by Isman Kurniawan
(<ismankrn@telkomuniversity.ac.id>), School of Computing, Telkom
University.

## Seeking support

For questions about how to use the library — which algorithm suits a
problem, how to wrap your own objective in a `Problem`, how to read a
benchmark table — open an issue labelled `question` at
<https://github.com/ismankrn/ikn-library/issues>, or write to the
maintainer directly. Please check the
[documentation](https://ikn-library.readthedocs.io) first; most usage
questions are answered there with a worked example.

Questions are answered on a best-effort basis, usually within a week.

## Reporting a problem

Open an issue at
<https://github.com/ismankrn/ikn-library/issues>. A report is most
useful when it contains:

- what you expected and what happened instead,
- a minimal example that reproduces it, including the random `seed`,
- the output of `python -c "import ikn_library; print(ikn_library.__version__)"`,
  your Python version and operating system.

If the problem is a wrong *number* rather than a crash — a benchmark
result that does not reproduce, a score that contradicts the
documentation — say so explicitly and include both numbers. Those
reports are the most valuable kind, and the project has acted on
several of them.

## Contributing code

### Setting up

```bash
git clone https://github.com/ismankrn/ikn-library
cd ikn-library
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Before opening a pull request

All three must pass; continuous integration runs them on Python
3.9–3.13:

```bash
ruff check src tests
pytest
```

If you touched documentation, also:

```bash
pip install -e ".[docs]"
mkdocs build
```

## Project conventions

These are not stylistic preferences; they are what the library's claims
rest on. A pull request that ignores them will be asked to change.

### Documentation output must be real

Every code example in the documentation is accompanied by the output of
actually running it. Nothing is written by hand, rounded by eye, or
carried over from an earlier version of the code. If you change
behaviour that a documented example depends on, re-run the example and
paste what it printed.

The practical check: extract every `python` block from a page, run them
in order in one process, and compare against the page's `text` blocks.

### Report what you measure, including when it is unflattering

Several pages in this library document weaknesses of the algorithm they
describe — an origin bias, a benchmark row that turns out to be an
artefact, a method that loses to a baseline. That is deliberate. If
your contribution measures something inconvenient, the contribution is
the measurement, not a favourable framing of it.

### Adding an algorithm

1. `src/ikn_library/algorithms/<name>.py`, subclassing `Algorithm` and
   implementing `init_population` and `run_iteration`. Validate every
   parameter with `ValueError`. If the algorithm is a variant of one
   already present, subclass it and override a single hook rather than
   copying the loop.
2. Register it in `algorithms/__init__.py` — both the import and
   `__all__`, both alphabetically sorted.
3. **Benchmark it on four variants** of each standard function:
   original, rotated, shifted, and both, at full budget (20,000
   evaluations, 10 dimensions). Rotation alone is not sufficient: a
   rotation turns the function about the origin, so an optimum at zero
   stays at zero and an origin bias goes undetected. Choose defaults on
   the transformed variants, not on the plain scores.
4. Tests in `tests/test_<name>.py` covering the algorithm's
   *characteristic mechanism*, not only that it runs and improves.
5. A documentation page in `docs/algorithm-details/<name>.md` with
   equations, pseudocode matching the source, parameters, measured
   behaviour, and references.
6. Update the index files: `ALGORITHMS.md`, `README.md`, `docs/api.md`,
   `mkdocs.yml`, and `docs/algorithm-details/index.md`.

### Other notes

- Docstrings must not contain relative `.md` links; they break when
  rendered into `docs/api.md`.
- New dependencies are avoided. `numpy` and `pandas` are required;
  `scikit-learn` and `rdkit` are optional extras, imported inside the
  functions that need them.
- The `examples/` directory is not linted by CI, but new example
  scripts should still pass `ruff check`.

## Disclosure of AI assistance

Parts of this library were implemented with AI assistance, recorded in
the commit history. Contributors who use AI tools are asked to do the
same — a `Co-Authored-By` trailer, or a note in the pull request. The
expectation is disclosure, not avoidance: what matters is that you
understand, have tested, and can defend the code you submit.

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are accepted under the [MIT Licence](LICENSE), the
licence of the project.
