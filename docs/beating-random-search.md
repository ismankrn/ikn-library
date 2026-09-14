# Beating Random Search

[Hyperparameter Optimization](parameter-optimization.md) shows how to
drive a metaheuristic over a hyperparameter space, and ends by reporting
the test score rather than the search's best. This page covers what that
test score has to be compared against before the search can be said to
have contributed anything.

The natural comparison — tuned model versus default model — measures
tuning, not the algorithm. A metaheuristic is one way of spending an
evaluation budget; drawing the same number of configurations uniformly
at random is another, and it costs nothing to implement. Random search
is therefore the reference point for a *search* claim, in the same way
that `tuning alone` is the reference point for a feature-selection
claim in [Feature Selection Protocol](feature-selection-protocol.md).

Four measurements make the comparison interpretable.

## The comparison at a glance

Three arms, one evaluation budget, one sealed test set. What separates
them is only how the budget is spent, so the difference between any two
is attributable to that:

<figure>
<svg viewBox="0 0 880 612" width="100%" style="max-width:880px;height:auto;border:1px solid #C2CED7;border-radius:4px" role="img" aria-label="The search data feeds three arms that share one estimator factory, one cross-validation object and one search box. The defaults arm spends no evaluations, random search draws N configurations uniformly, and the metaheuristic spends the same N under guidance. Each arm is refit and scored once on the sealed test set. Subtracting the metaheuristic from the defaults measures tuning and search together; subtracting it from random search at the same budget isolates the contribution of the algorithm. Both differences are only readable above the objective's measured noise floor.">
  <defs>
    <marker id="f2a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#1F2933"/></marker>
    <marker id="f2g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#1C6450"/></marker>
    <marker id="f2s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8A2F26"/></marker>
  </defs>
  <rect x="0" y="0" width="880" height="612" fill="#FFFFFF"/>
  <g font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12.5" fill="#1F2933">

    <rect x="90" y="24" width="320" height="40" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
    <text x="250" y="49" text-anchor="middle" font-size="13" font-weight="600">X_search</text>

    <rect x="580" y="24" width="240" height="40" rx="3" fill="#F6E9E7" stroke="#8A2F26" stroke-width="1.6"/>
    <text x="700" y="49" text-anchor="middle" font-size="13" font-weight="600" fill="#8A2F26">X_test — sealed</text>

    <rect x="90" y="80" width="320" height="34" rx="3" fill="none" stroke="#1F2933" stroke-width="1.1" stroke-dasharray="4 4"/>
    <text x="250" y="101" text-anchor="middle" font-size="11.5" fill="#5A6B78">one CV object · one estimator factory · one box</text>

    <polyline points="250,114 250,132 48,132 48,192" fill="none" stroke="#1F2933" stroke-width="1.2"/>

    <rect x="30" y="150" width="530" height="292" rx="5" fill="none" stroke="#1F2933" stroke-width="1.6"/>
    <text x="48" y="174" font-size="11" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" letter-spacing=".9">IDENTICAL EVALUATION BUDGET — N = 150</text>

    <line x1="48" y1="192" x2="48" y2="402" stroke="#1F2933" stroke-width="1.2"/>

    <line x1="48" y1="214" x2="64" y2="214" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f2a)"/>
    <rect x="66" y="192" width="276" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <text x="80" y="211" font-weight="600">A · defaults</text>
    <text x="80" y="228" font-size="11.5" fill="#5A6B78">0 evaluations — no search at all</text>

    <line x1="48" y1="298" x2="64" y2="298" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f2a)"/>
    <rect x="66" y="276" width="276" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <text x="80" y="295" font-weight="600">B · random search</text>
    <text x="80" y="312" font-size="11.5" fill="#5A6B78">N uniform draws from the same box</text>

    <line x1="48" y1="382" x2="64" y2="382" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f2g)"/>
    <rect x="66" y="360" width="276" height="44" rx="3" fill="#E4EFEA" stroke="#1C6450" stroke-width="1.4"/>
    <text x="80" y="379" font-weight="600" fill="#1C6450">C · metaheuristic</text>
    <text x="80" y="396" font-size="11.5" fill="#1C6450">N guided evaluations</text>

    <line x1="342" y1="214" x2="374" y2="214" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f2a)"/>
    <line x1="342" y1="298" x2="374" y2="298" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f2a)"/>
    <line x1="342" y1="382" x2="374" y2="382" stroke="#1C6450" stroke-width="1.2" marker-end="url(#f2g)"/>

    <rect x="376" y="192" width="168" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <text x="460" y="218" text-anchor="middle" font-size="12">test score A</text>
    <rect x="376" y="276" width="168" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <text x="460" y="302" text-anchor="middle" font-size="12">test score B</text>
    <rect x="376" y="360" width="168" height="44" rx="3" fill="#E4EFEA" stroke="#1C6450" stroke-width="1.4"/>
    <text x="460" y="386" text-anchor="middle" font-size="12" fill="#1C6450">test score C</text>

    <polyline points="700,64 700,124 460,124 460,186" fill="none" stroke="#8A2F26" stroke-width="1.4" stroke-dasharray="5 5" marker-end="url(#f2s)"/>
    <text x="472" y="112" font-size="11.5" fill="#8A2F26">broken once, for all three arms</text>

    <polyline points="548,298 596,298 596,382 548,382" fill="none" stroke="#1C6450" stroke-width="1.6"/>
    <line x1="596" y1="340" x2="624" y2="340" stroke="#1C6450" stroke-width="1.6" marker-end="url(#f2g)"/>
    <text x="632" y="336" font-size="13" font-weight="600" fill="#1C6450">C − B</text>
    <text x="632" y="353" font-size="11.5" fill="#1C6450">the algorithm's contribution</text>
    <text x="632" y="369" font-size="11.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="#1C6450">+0.0018 in §2</text>

    <polyline points="548,214 668,214 668,420 548,420" fill="none" stroke="#5A6B78" stroke-width="1.2"/>
    <line x1="668" y1="252" x2="696" y2="252" stroke="#5A6B78" stroke-width="1.2" marker-end="url(#f2a)"/>
    <text x="704" y="248" font-size="13" font-weight="600" fill="#5A6B78">C − A</text>
    <text x="704" y="265" font-size="11.5" fill="#5A6B78">tuning and search together</text>
    <text x="704" y="281" font-size="11.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="#5A6B78">+0.0000 in §2</text>

    <rect x="30" y="480" width="820" height="96" rx="4" fill="#F4F6F8" stroke="#C2CED7" stroke-width="1.2"/>
    <text x="48" y="506" font-size="11" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" letter-spacing=".9" fill="#5A6B78">NOISE FLOOR — measure it before reading either difference</text>
    <line x1="140" y1="546" x2="760" y2="546" stroke="#1F2933" stroke-width="1.2"/>
    <rect x="318" y="530" width="264" height="32" fill="#E6EBEF"/>
    <line x1="450" y1="526" x2="450" y2="566" stroke="#5A6B78" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="450" y="578" text-anchor="middle" font-size="10.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="#5A6B78">0</text>
    <text x="326" y="578" font-size="10.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="#5A6B78">−0.0088</text>
    <text x="574" y="578" text-anchor="end" font-size="10.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="#5A6B78">+0.0088</text>
    <circle cx="450" cy="538" r="4" fill="#5A6B78"/>
    <circle cx="477" cy="554" r="4" fill="#1C6450"/>
    <text x="596" y="536" font-size="11.5" fill="#5A6B78">shaded band = one test observation (1/114)</text>
    <text x="596" y="553" font-size="11.5" fill="#1F2933">both differences land inside it</text>
  </g>
</svg>
<figcaption>
Arm A spends nothing, arm B draws uniformly, arm C searches — all three
from the same box, with the same estimator factory and the same
cross-validation object. <code>C − A</code> moves two things at once and
measures tuning and search together; <code>C − B</code> holds the budget
fixed and isolates the algorithm. The band at the bottom is the smallest
difference either subtraction can resolve.
</figcaption>
</figure>

The band is what makes the two numbers on the brackets readable. Both
land inside one test observation, so on this dataset neither subtraction
supports a claim — which is a result, and the rest of this page is how
to get it.

## 1. A best-of-N score improves with budget on its own

Before comparing algorithms, it helps to see how much of a search score
is produced by the budget alone. This problem ignores its input
entirely — every candidate draws an independent standard normal, so
there is nothing whatsoever to optimize:

```python
import numpy as np

from ikn_library import Task
from ikn_library.algorithms import (AntColonyOptimization, GreyWolfOptimizer,
                                    ParticleSwarmOptimization)
from ikn_library.problems import Problem


class PureNoise(Problem):
    """Fitness ignores x: every candidate draws N(0, 1)."""

    def __init__(self, seed=0):
        super().__init__(dimension=5, lower=-5.0, upper=5.0)
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, x):
        return float(self.rng.normal())


def random_search(n, seed):
    return float(np.random.default_rng(seed).normal(size=n).min())


ALG = {"ACO-R": AntColonyOptimization,
       "PSO": ParticleSwarmOptimization,
       "GWO": GreyWolfOptimizer}

print(f"{'budget':>7} " + " ".join(f"{k:>8}" for k in ALG) + f"{'random':>9}")
for n in (50, 200, 1000, 5000):
    row = []
    for i, (name, algorithm) in enumerate(ALG.items()):
        best = []
        for s in range(10):
            task = Task(problem=PureNoise(seed=1000 * i + s), max_evals=n)
            algorithm(population_size=20, seed=s).run(task)
            best.append(task.best_fitness)
        row.append(np.mean(best))
    rnd = np.mean([random_search(n, 9000 + s) for s in range(10)])
    print(f"{n:>7} " + " ".join(f"{v:>8.3f}" for v in row) + f"{rnd:>9.3f}")
```

Output:

```text
 budget    ACO-R      PSO      GWO   random
     50   -2.399   -2.077   -1.829   -2.389
    200   -2.681   -2.611   -2.506   -2.796
   1000   -3.313   -3.077   -3.249   -3.201
   5000   -3.570   -3.558   -3.547   -3.623
```

Every column improves monotonically, and every convergence curve drawn
from these runs would slope downward convincingly. The improvement is
the expected minimum of N draws, which grows with N whatever the
objective is doing — here, nothing.

Random search is level with all three algorithms at every budget, which
is the expected result when there is no structure to exploit. That is
precisely what makes it a useful reference: on a real objective, the
gap between a metaheuristic and random search at equal budget is the
part of the score that came from searching rather than from sampling.

!!! note "Report the budget with the score"
    Two searches are only comparable at equal evaluation counts. A
    population of 30 for 50 iterations is 1500 evaluations; a population
    of 10 for 50 iterations is 500. Comparing their best scores compares
    budgets as much as algorithms.

## 2. Random search at the same budget

The same comparison on a real objective — an RBF SVM tuned over
`log10(C)` and `log10(gamma)`, exactly the problem from
[Hyperparameter Optimization](parameter-optimization.md):

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task

X, y = load_breast_cancer(return_X_y=True)
X_search, X_test, y_search, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def build(x):
    return make_pipeline(StandardScaler(), SVC(C=10 ** x[0], gamma=10 ** x[1]))


class SVMTuning(Problem):
    def __init__(self):
        super().__init__(dimension=2, lower=np.array([-3.0, -4.0]),
                         upper=np.array([3.0, 1.0]))

    def _evaluate(self, x):
        return cross_val_score(build(x), X_search, y_search, cv=CV, n_jobs=-1).mean()


def random_tuning(n, seed):
    """Same box, same objective, same budget — uniform draws."""
    rng = np.random.default_rng(seed)
    problem = SVMTuning()
    best_x, best = None, -np.inf
    for _ in range(n):
        x = rng.uniform(problem.lower, problem.upper)
        score = problem._evaluate(x)
        if score > best:
            best, best_x = score, x
    return best, best_x


N = 150
print(f"{'method':>14} {'CV best':>18} {'test':>18}")
for name in ("ACO-R", "random"):
    cvs, tests = [], []
    for s in range(5):
        if name == "ACO-R":
            task = Task(problem=SVMTuning(), max_evals=N,
                        optimization_type=OptimizationType.MAXIMIZATION)
            best_x, best = AntColonyOptimization(population_size=20, seed=s).run(task)
        else:
            best, best_x = random_tuning(N, 100 + s)
        model = build(best_x).fit(X_search, y_search)
        cvs.append(best)
        tests.append(model.score(X_test, y_test))
    print(f"{name:>14} {np.mean(cvs):>10.4f} +/- {np.std(cvs):<6.4f}"
          f"{np.mean(tests):>10.4f} +/- {np.std(tests):<6.4f}")

default = make_pipeline(StandardScaler(), SVC())
default_cv = cross_val_score(default, X_search, y_search, cv=CV, n_jobs=-1).mean()
default.fit(X_search, y_search)
print(f"{'defaults':>14} {default_cv:>10.4f} {'':<10}"
      f"{default.score(X_test, y_test):>10.4f}")
```

Output:

```text
budget = 150 evaluations, 5 seeds

        method            CV best               test
         ACO-R     0.9771 +/- 0.0011    0.9825 +/- 0.0000
        random     0.9758 +/- 0.0000    0.9807 +/- 0.0035
      defaults     0.9692               0.9825
```

ACO-R does beat random search, by **+0.0013** on the objective it
optimized and **+0.0018** on the test set. Both gaps are smaller than
one test-set observation (1/114 = 0.0088), so the honest summary is that
the two methods are indistinguishable here.

The third row is the one worth carrying into a discussion section:
`SVC()`'s defaults reach **0.9825** on the test set, matching ACO-R
exactly, with no search at all. The search space was already centred on
a good region, so 150 evaluations bought CV score that did not convert
into test accuracy.

None of that makes the search pointless — it is how the fact that the
defaults were well placed got established. It does mean the claim to
report is "tuning confirmed the default region", not "the metaheuristic
improved accuracy".

## 3. The objective's own noise floor

When the model is stochastic, the same hyperparameters return different
scores on different training seeds. Any difference smaller than that
spread is not attributable to the hyperparameters:

```python
from sklearn.neural_network import MLPClassifier


def mlp(width, lr, seed):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(width,), learning_rate_init=lr,
                      max_iter=300, early_stopping=True, random_state=seed))


scores = [cross_val_score(mlp(64, 1e-3, s), X_search, y_search,
                          cv=CV, n_jobs=-1).mean() for s in range(10)]
print(f"one configuration (64, 1e-3), 10 training seeds")
print(f"  mean   : {np.mean(scores):.4f}")
print(f"  spread : SD {np.std(scores):.4f} | range {np.ptp(scores):.4f}")
```

Output:

```text
one configuration (64, 1e-3), 10 training seeds
  mean   : 0.9211
  spread : SD 0.0143 | range 0.0462
```

One configuration spans **0.0462** across seeds. A search reporting a
0.02 improvement over a baseline, with one training seed each, is
reporting something this spread could have produced on its own.

Averaging over seeds changes the comparison it supports:

```python
A, B = (64, 1e-3), (16, 1e-2)


def evaluate(cfg, seeds):
    return np.mean([cross_val_score(mlp(*cfg, s), X_search, y_search,
                                    cv=CV, n_jobs=-1).mean() for s in seeds])


print(f"{'':<22}{'A=(64, 1e-3)':>14}{'B=(16, 1e-2)':>14}{'A - B':>10}")
for label, seeds in (("single seed", [0]), ("mean of 10 seeds", range(10))):
    a, b = evaluate(A, seeds), evaluate(B, seeds)
    print(f"{label:<22}{a:>14.4f}{b:>14.4f}{a - b:>+10.4f}")
```

Output:

```text
                        A=(64, 1e-3)  B=(16, 1e-2)     A - B
single seed                   0.9209        0.9451   -0.0242
mean of 10 seeds              0.9211        0.9558   -0.0347
```

The seed-averaged gap is 0.0347 and the single-seed estimate misses it
by 0.010 — and 0.0242 sits inside the 0.0462 range from a single
configuration, so one seed could not have established the ordering even
though it happened to get the sign right.

Two ways to handle this, both fine as long as the choice is stated:
fix the training seed so the objective is deterministic and the search
ranks architectures rather than initializations; or average a few seeds
per evaluation, which costs proportionally more but produces differences
that survive re-running.

!!! tip "Measure the floor before reading the table"
    Re-evaluating one configuration a handful of times takes a fraction
    of a search budget and tells you the smallest difference worth
    interpreting. Quote it next to the results.

## 4. Can the search reach the baseline?

A comparison against a baseline configuration assumes the search could
have found it. When the decode cannot express it, the comparison has no
interpretation in either direction.

A common shortcut is to draw one layer width and repeat it for every
layer, which quietly removes every narrowing architecture from the
space:

```python
UNITS = [16, 32, 64, 128, 256]


def decode_uniform(x):
    """One width, repeated for every layer."""
    n_layers = int(np.clip(np.floor(x[0] * 3) + 1, 1, 3))
    width = UNITS[int(np.clip(np.floor(x[1] * len(UNITS)), 0, len(UNITS) - 1))]
    return tuple([width] * n_layers)


def decode_per_layer(x):
    """One width per layer."""
    n_layers = int(np.clip(np.floor(x[0] * 3) + 1, 1, 3))
    return tuple(UNITS[int(np.clip(np.floor(x[1 + i] * len(UNITS)), 0,
                                   len(UNITS) - 1))]
                 for i in range(n_layers))


BASELINES = [(64,), (128, 64), (256, 128, 64)]
rng = np.random.default_rng(0)

print(f"{'decode':>14} {'distinct architectures':>24}   baselines reachable")
for name, fn, dim in (("uniform width", decode_uniform, 2),
                      ("per layer", decode_per_layer, 4)):
    seen = {fn(rng.random(dim)) for _ in range(200000)}
    reachable = [b for b in BASELINES if b in seen]
    print(f"{name:>14} {len(seen):>24}   {len(reachable)}/{len(BASELINES)}  "
          f"{[str(b) for b in reachable]}")
```

Output:

```text
        decode   distinct architectures   baselines reachable
 uniform width                       15   1/3  ['(64,)']
     per layer                       155   3/3  ['(64,)', '(128, 64)', '(256, 128, 64)']
```

The uniform-width decode spans 15 architectures and can express one of
the three baselines. Two of the comparisons it would appear in are
therefore between a search and a configuration the search was never able
to consider.

The check is one assertion, and it belongs next to the decode:

```python
for baseline in BASELINES:
    assert baseline in seen, f"{baseline} is outside the search space"
```

## Best practice: the four measurements

1. **Run random search at the same evaluation budget**, and report the
   gap against it as the contribution of the algorithm. Report the
   budget itself, in evaluations, not in iterations.
2. **Keep the default configuration in the table.** It is the cheapest
   arm and frequently a strong one; a search that matches it has
   established something worth stating.
3. **Measure the objective's noise floor** by re-evaluating one
   configuration several times, and interpret no difference smaller
   than that spread.
4. **Assert that every baseline is reachable by the decode** before
   comparing against it.
5. **Read the test set once**, after the configuration is frozen — see
   [Hyperparameter Optimization](parameter-optimization.md#reporting-open-the-test-set-once).
6. **Repeat over seeds** and report mean and spread; one run of a
   stochastic search on a stochastic objective is one draw from two
   distributions at once.

## Checklist

- [ ] Random search ran at the same evaluation budget.
- [ ] The budget is reported in evaluations.
- [ ] The default configuration appears in the results table.
- [ ] The objective's seed-to-seed spread was measured and quoted.
- [ ] Reported differences exceed that spread.
- [ ] Every baseline configuration is reachable by the decode.
- [ ] The test set was read once, after all choices were frozen.
- [ ] Results are averaged over several seeds.

## Reference

- [Hyperparameter Optimization](parameter-optimization.md) — defining the
  problem, encoding integer and categorical dimensions, refitting, and
  reporting.
- [Hyperparameter Tuning](gridsearch-comparison.md) — grid search as the
  exhaustive counterpart to the budgets used here.
- [Feature Selection Protocol](feature-selection-protocol.md) — the
  same attribution question for feature subsets.
- [Setting the Annealing Temperature](annealing-temperature.md) — if the
  search is `SimulatedAnnealing`, calibrate it before benchmarking it.
- [Plotting Convergence](convergence-plot.md) — reading the curve that
  section 1 shows can slope downward on pure noise.
