# Feature Selection Protocol

[Feature Selection](feature-selection.md) shows how to *run* a wrapper
search. This page covers the question that follows: how much of the
resulting score is attributable to the selection itself.

A wrapper search evaluates hundreds of subsets against the same folds
and returns the best of them, so its score is a maximum over hundreds of
draws rather than an estimate of expected performance. The gap between
those two quantities grows with the number of candidates evaluated, and
it is present even when the features carry no signal.

Three measurements separate the attributable gain from that gap. None of
them is expensive.

!!! note "Which subtraction isolates selection"
    `selection + tuning` versus `defaults` moves two things at once, so
    the difference carries both effects. The subtraction that isolates
    selection is `selection + tuning` versus **tuning alone**. On the
    example below the two subtractions differ in sign.

## The protocol at a glance

The three measurements below sit inside a larger discipline: the folds
used to *choose* are kept separate from the folds used to *report*, and
the test set is opened once, at the end. Drawn as nested regions rather
than as a list of steps, because the nesting is the part that matters:

<figure>
<svg viewBox="0 0 880 1205" width="100%" style="max-width:880px;height:auto;border:1px solid #C2CED7;border-radius:4px" role="img" aria-label="The full dataset is split into X_dev and a sealed X_test. X_dev enters an outer loop that only reports; inside it an inner loop performs every step that learns from the labels — unsupervised filtering, class balancing, the hyperparameter grid, and then metaheuristic feature selection. The inner loop returns parameters and a feature mask, a model is refit on the full training folds and scored on the outer fold. The mean of those folds is the reported nested score. Only after every choice is frozen is the seal broken, once, to score four arms on X_test and compare them with paired tests.">
          <defs>
            <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10 z" fill="#1F2933"/>
            </marker>
            <marker id="arSeal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10 z" fill="#8A2F26"/>
            </marker>
            <marker id="arGate" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10 z" fill="#1C6450"/>
            </marker>
          </defs>

          <rect x="0" y="0" width="880" height="1205" fill="#FFFFFF"/>
          <g font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12.5" fill="#1F2933">

            <!-- root -->
            <rect x="300" y="28" width="280" height="42" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="440" y="54" text-anchor="middle" font-size="13" font-weight="600">Full dataset · n samples × p features</text>

            <line x1="440" y1="70" x2="440" y2="94" stroke="#1F2933" stroke-width="1.2"/>
            <line x1="250" y1="94" x2="700" y2="94" stroke="#1F2933" stroke-width="1.2"/>
            <line x1="250" y1="94" x2="250" y2="118" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <line x1="700" y1="94" x2="700" y2="118" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <text x="452" y="88" font-size="11" fill="#5A6B78" font-family="IBM Plex Mono, monospace">stratified split · fixed random_state</text>

            <!-- dev -->
            <rect x="120" y="120" width="260" height="42" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="250" y="140" text-anchor="middle" font-size="13" font-weight="600">X_dev — 80%</text>
            <text x="250" y="155" text-anchor="middle" font-size="11" fill="#5A6B78">every decision is made here</text>

            <!-- test -->
            <rect x="580" y="120" width="240" height="42" rx="3" fill="#F6E9E7" stroke="#8A2F26" stroke-width="1.6"/>
            <text x="700" y="146" text-anchor="middle" font-size="13" font-weight="600" fill="#8A2F26">X_test — 20%</text>
            <text x="700" y="186" text-anchor="middle" font-size="11" fill="#8A2F26" font-family="IBM Plex Mono, monospace" letter-spacing="1.6">S E A L E D</text>
            <text x="700" y="203" text-anchor="middle" font-size="11" fill="#8A2F26">opened once, at the very end</text>
            <line x1="700" y1="216" x2="700" y2="997" stroke="#8A2F26" stroke-width="1.4" stroke-dasharray="5 5"/>

            <!-- OUTER region -->
            <rect x="40" y="200" width="500" height="600" rx="5" fill="none" stroke="#1F2933" stroke-width="1.6"/>
            <text x="58" y="224" font-size="11" font-weight="600" font-family="IBM Plex Mono, monospace" letter-spacing=".9">OUTER LOOP × 5 — REPORTS ONLY</text>

            <line x1="250" y1="162" x2="250" y2="200" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <line x1="250" y1="200" x2="250" y2="226" stroke="#1F2933" stroke-width="1.2"/>
            <line x1="175" y1="226" x2="413" y2="226" stroke="#1F2933" stroke-width="1.2"/>
            <line x1="175" y1="226" x2="175" y2="246" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <line x1="413" y1="226" x2="413" y2="246" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>

            <rect x="62" y="246" width="226" height="38" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
            <text x="175" y="270" text-anchor="middle">training folds · 4/5</text>

            <rect x="308" y="246" width="210" height="38" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2" stroke-dasharray="4 4"/>
            <text x="413" y="270" text-anchor="middle">outer test fold · 1/5</text>
            <line x1="413" y1="284" x2="413" y2="666" stroke="#1F2933" stroke-width="1.1" stroke-dasharray="4 4" marker-end="url(#ar)"/>
            <text x="424" y="470" font-size="11" fill="#5A6B78">waits —</text>
            <text x="424" y="485" font-size="11" fill="#5A6B78">seen by no</text>
            <text x="424" y="500" font-size="11" fill="#5A6B78">step below</text>

            <!-- INNER region -->
            <line x1="175" y1="284" x2="175" y2="320" stroke="#1F2933" stroke-width="1.2" marker-end="url(#arGate)"/>
            <rect x="62" y="320" width="258" height="320" rx="5" fill="#E4EFEA" stroke="#1C6450" stroke-width="1.6"/>
            <text x="76" y="344" font-size="11" font-weight="600" fill="#1C6450" font-family="IBM Plex Mono, monospace" letter-spacing=".9">INNER LOOP × 5 — SELECTS ONLY</text>

            <rect x="78" y="358" width="226" height="36" rx="3" fill="#FFFFFF" stroke="#1C6450" stroke-width="1.1"/>
            <text x="191" y="381" text-anchor="middle" font-size="12">① unsupervised filter + scaling</text>

            <line x1="191" y1="394" x2="191" y2="406" stroke="#1C6450" stroke-width="1.2" marker-end="url(#arGate)"/>
            <rect x="78" y="406" width="226" height="36" rx="3" fill="#FFFFFF" stroke="#1C6450" stroke-width="1.1"/>
            <text x="191" y="429" text-anchor="middle" font-size="12">② SMOTE / RUS — training folds only</text>

            <line x1="191" y1="442" x2="191" y2="454" stroke="#1C6450" stroke-width="1.2" marker-end="url(#arGate)"/>
            <rect x="78" y="454" width="226" height="36" rx="3" fill="#FFFFFF" stroke="#1C6450" stroke-width="1.1"/>
            <text x="191" y="477" text-anchor="middle" font-size="12">③ hyperparameter grid — FIRST</text>

            <line x1="191" y1="490" x2="191" y2="502" stroke="#1C6450" stroke-width="1.2" marker-end="url(#arGate)"/>
            <rect x="78" y="502" width="226" height="36" rx="3" fill="#FFFFFF" stroke="#1C6450" stroke-width="1.1"/>
            <text x="191" y="525" text-anchor="middle" font-size="12">④ metaheuristic selects features</text>

            <line x1="191" y1="538" x2="191" y2="554" stroke="#1C6450" stroke-width="1.2" marker-end="url(#arGate)"/>
            <rect x="78" y="554" width="226" height="36" rx="3" fill="none" stroke="#1C6450" stroke-width="1.1" stroke-dasharray="4 4"/>
            <text x="191" y="577" text-anchor="middle" font-size="12" fill="#1C6450" font-weight="600">returns: params + feature mask</text>

            <text x="191" y="612" text-anchor="middle" font-size="11" fill="#1C6450">③ precedes ④ — features are chosen</text>
            <text x="191" y="627" text-anchor="middle" font-size="11" fill="#1C6450">for the model that will score them</text>

            <!-- retrain -->
            <line x1="191" y1="640" x2="191" y2="666" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <rect x="62" y="666" width="226" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="175" y="684" text-anchor="middle" font-size="12">refit on the full training folds</text>
            <text x="175" y="700" text-anchor="middle" font-size="11" fill="#5A6B78">using the inner loop's params + mask</text>

            <line x1="288" y1="688" x2="306" y2="688" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <rect x="308" y="666" width="210" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="413" y="692" text-anchor="middle" font-size="12">fold k score</text>

            <line x1="413" y1="710" x2="413" y2="742" stroke="#1F2933" stroke-width="1.2"/>
            <line x1="413" y1="742" x2="290" y2="742" stroke="#1F2933" stroke-width="1.2"/>
            <line x1="290" y1="742" x2="290" y2="800" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <text x="300" y="766" font-size="11" fill="#5A6B78">mean of 5 folds</text>

            <!-- reported -->
            <rect x="100" y="826" width="380" height="46" rx="3" fill="#E4EFEA" stroke="#1C6450" stroke-width="1.8"/>
            <text x="290" y="846" text-anchor="middle" font-size="13" font-weight="600" fill="#1C6450">NESTED SCORE — the reported number</text>
            <text x="290" y="862" text-anchor="middle" font-size="11" fill="#1C6450">not best_score_, not the best fitness</text>

            <line x1="290" y1="872" x2="290" y2="900" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <rect x="100" y="900" width="380" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="290" y="926" text-anchor="middle" font-size="12.5">freeze estimator · params · mask · scheme</text>

            <line x1="290" y1="944" x2="290" y2="972" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <rect x="160" y="972" width="260" height="46" rx="3" fill="#F6E9E7" stroke="#8A2F26" stroke-width="1.8"/>
            <text x="290" y="999" text-anchor="middle" font-size="13" font-weight="600" fill="#8A2F26">BREAK THE SEAL — once</text>
            <line x1="700" y1="997" x2="426" y2="997" stroke="#8A2F26" stroke-width="1.4" stroke-dasharray="5 5" marker-end="url(#arSeal)"/>

            <line x1="290" y1="1018" x2="290" y2="1046" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <rect x="80" y="1046" width="420" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="290" y="1072" text-anchor="middle" font-size="12.5">score arms P1–P4 on X_test</text>

            <line x1="290" y1="1090" x2="290" y2="1118" stroke="#1F2933" stroke-width="1.2" marker-end="url(#ar)"/>
            <rect x="60" y="1118" width="460" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
            <text x="290" y="1144" text-anchor="middle" font-size="12.5">McNemar · bootstrap CI · TOST</text>
          </g>
        </svg>
<figcaption>
Everything that learns from the labels — class balancing, tuning,
feature selection — lives in the green region. The outer test fold and
<code>X_test</code> never enter it. The reported score is produced
outside the green region; the score inside it is only used to choose.
</figcaption>
</figure>

Two orderings in that picture are easy to get backwards. Step ③ runs
before step ④, so the features are chosen for the model that will
actually score them rather than for the defaults. And the outer fold
sits idle through the whole inner loop — it is scored once, by a model
refitted from the recipe the inner loop returned.

## 1. Four arms, not two

Run the two interventions — tuning and selection — in a full factorial,
so each one has a partner that differs from it in exactly one place:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.problems import FeatureSelectionProblem

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
CV = StratifiedKFold(5, shuffle=True, random_state=42)
GRID = {"kneighborsclassifier__n_neighbors": [3, 5, 7, 11, 15],
        "kneighborsclassifier__weights": ["uniform", "distance"]}


def knn(**kw):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**kw))


def tune(Xa, ya):
    g = GridSearchCV(knn(), GRID, cv=CV, scoring="accuracy", n_jobs=-1).fit(Xa, ya)
    return {k.split("__")[1]: v for k, v in g.best_params_.items()}


def select(Xa, ya, params, seed=42):
    problem = FeatureSelectionProblem(Xa, ya, estimator=knn(**params), cv=CV,
                                      scoring="accuracy", alpha=0.99)
    task = Task(problem=problem, max_evals=1000)
    best_x, _ = BinaryAntColonyOptimization(population_size=20, evaporation=0.1,
                                            seed=seed).run(task)
    return problem.feature_mask(best_x)


ALL = np.ones(X_train.shape[1], dtype=bool)
tuned = tune(X_train, y_train)
arms = {
    "P1 default":       ({},    ALL),
    "P2 selection":     ({},    select(X_train, y_train, {})),
    "P4 tuning":        (tuned, ALL),
    "P3 sel + tuning":  (tuned, select(X_train, y_train, tuned)),
}

print("arm               n_feat   cv_acc   test_acc")
test_acc = {}
for name, (params, mask) in arms.items():
    cv = cross_val_score(knn(**params), X_train[:, mask], y_train,
                         cv=CV, scoring="accuracy", n_jobs=-1).mean()
    model = knn(**params).fit(X_train[:, mask], y_train)
    test_acc[name] = accuracy_score(y_test, model.predict(X_test[:, mask]))
    print(f"{name:<17} {int(mask.sum()):>5}   {cv:.4f}   {test_acc[name]:.4f}")

print(f"\nP3 - P1 = {test_acc['P3 sel + tuning'] - test_acc['P1 default']:+.4f}"
      "   <- both interventions at once")
print(f"P3 - P4 = {test_acc['P3 sel + tuning'] - test_acc['P4 tuning']:+.4f}"
      "   <- the effect of selection")
```

Output:

```text
arm               n_feat   cv_acc   test_acc
P1 default           30   0.9626   0.9561
P2 selection         13   0.9802   0.9649
P4 tuning            30   0.9626   0.9737
P3 sel + tuning      14   0.9802   0.9561

P3 - P1 = +0.0000   <- both interventions at once
P3 - P4 = -0.0175   <- the effect of selection
```

Three things worth reading off that table.

`P3 - P1` is **+0.0000**: the two interventions cancel, so the combined
subtraction carries no information about either. `P3 - P4`, which isolates
selection, is **−0.0175** — once the model is tuned, keeping 14 of 30
features costs accuracy rather than buying it.

`P2 - P1 = +0.0088` is positive, so selection improves on the untuned
baseline. `P4` supplies the missing context: tuning alone reaches
+0.0176 on the same test set, and it does so without discarding any
features.

The CV column shows +0.0176 for both selection arms, which the test set
does not reproduce. That gap between the selection score and held-out
performance is the subject of the next section.

!!! note "Keep the arms comparable"
    Every arm must use the same `CV` object, the same estimator factory
    and the same preprocessing — otherwise the difference between two
    arms includes whatever else changed. `select()` above takes the
    tuned parameters as an argument precisely so that `P3` searches for
    features that suit the model it will actually be scored with.

## 2. The search score is not a result

The score attached to `best_fitness` is the value the search maximized.
Re-running `cross_val_score` on the winning subset does not turn it into
an independent estimate — with the same folds it returns the same
number:

```python
problem = FeatureSelectionProblem(X_train, y_train, estimator=knn(), cv=CV,
                                  scoring="accuracy", alpha=0.99)
task = Task(problem=problem, max_evals=1000)
best_x, best_fitness = BinaryAntColonyOptimization(
    population_size=20, evaporation=0.1, seed=42).run(task)
mask = problem.feature_mask(best_x)

search_cv = 1.0 - (best_fitness - 0.01 * mask.sum() / problem.dimension) / 0.99
rerun_cv = cross_val_score(knn(), X_train[:, mask], y_train,
                           cv=CV, scoring="accuracy", n_jobs=-1).mean()

# Out-of-sample: repeat the whole selection inside every outer fold.
OUTER = StratifiedKFold(5, shuffle=True, random_state=7)
outer, sizes = [], []
for k, (tr, te) in enumerate(OUTER.split(X_train, y_train)):
    inner = StratifiedKFold(5, shuffle=True, random_state=42 + k)
    p_in = FeatureSelectionProblem(X_train[tr], y_train[tr], estimator=knn(),
                                   cv=inner, scoring="accuracy", alpha=0.99)
    t_in = Task(problem=p_in, max_evals=1000)
    bx, _ = BinaryAntColonyOptimization(population_size=20, evaporation=0.1,
                                        seed=42 + k).run(t_in)
    m_in = p_in.feature_mask(bx)
    model = knn().fit(X_train[tr][:, m_in], y_train[tr])
    outer.append(accuracy_score(y_train[te], model.predict(X_train[te][:, m_in])))
    sizes.append(int(m_in.sum()))

print(f"implied CV accuracy of the winner : {search_cv:.4f}")
print(f"cross_val_score on that winner    : {rerun_cv:.4f}")
print(f"nested CV accuracy                : {np.mean(outer):.4f} "
      f"+/- {np.std(outer):.4f}")
print(f"subset size per outer fold        : {sizes}")
```

Output:

```text
implied CV accuracy of the winner : 0.9802
cross_val_score on that winner    : 0.9802
nested CV accuracy                : 0.9626 +/- 0.0275
subset size per outer fold        : [14, 13, 16, 12, 12]
```

The first two lines are **the same number to four decimals**, because
they are the same computation. The third line answers a different
question — how the procedure performs on data it did not select on — and
it lands at 0.9626 — exactly the
`P1 default` CV accuracy from the previous section. On this dataset,
wrapper selection matches keeping all 30 features, and the difference
between 0.9802 and 0.9626 is the selection gap, measured.

The per-fold subset sizes carry the other half of the message: five runs
of the same procedure on slightly different data keep 12 to 16 features.
A single run's selected-feature list is one draw from that spread, and
its spread is worth reporting alongside it.

!!! tip "When nested CV is too expensive"
    Hold out an inner validation split for the search instead, and
    report the effect of selection from the test set (`P3 - P4`). The
    search's own CV score stays out of the results table either way.

## 3. Did the search actually move?

A search that never leaves its starting region still returns a subset,
a fitness and a convergence curve. The cheapest way to catch that is to
compare the subset size the optimizer *started* with against the one it
finished with.

Solutions live in `[0, 1]` and a feature counts as selected when it
exceeds `threshold`. With the default of 0.5, a uniform random start
selects half the features every time — on any problem, at any
dimension:

```python
rng = np.random.default_rng(0)
for thr in (0.5, 0.7, 0.85):
    n = (rng.random((2000, 30)) > thr).sum(axis=1)
    print(f"threshold={thr:<5} random start keeps {n.mean():5.1f} / 30 "
          f"= {100 * n.mean() / 30:5.1f}%")
```

Output:

```text
threshold=0.5   random start keeps  15.0 / 30 =  50.0%
threshold=0.7   random start keeps   9.0 / 30 =  29.9%
threshold=0.85  random start keeps   4.5 / 30 =  14.8%
```

Whether the search can escape that start depends on the dimension. At
30 features it can; at 500 it cannot:

```python
from sklearn.datasets import make_classification
from ikn_library.algorithms import SimulatedAnnealing

X_hd, y_hd = make_classification(n_samples=200, n_features=500,
                                 n_informative=20, n_redundant=0,
                                 random_state=0, shuffle=False)

print(f"{'threshold':>10} {'start':>7} {'end':>6} {'drift':>7}")
for thr in (0.5, 0.85):
    problem = FeatureSelectionProblem(X_hd, y_hd, estimator=knn(), cv=CV,
                                      scoring="accuracy", alpha=0.99,
                                      threshold=thr)
    task = Task(problem=problem, max_evals=500)
    start = int((np.random.default_rng(42).uniform(0, 1, 500) > thr).sum())
    best_x, _ = SimulatedAnnealing(initial_temperature=0.029, cooling=0.995,
                                   step_size=0.1, seed=42).run(task)
    end = int(problem.feature_mask(best_x).sum())
    print(f"{thr:>10} {start:>7} {end:>6} {end - start:>+7}")
```

Output:

```text
 threshold   start    end   drift
       0.5     251    247      -4
      0.85      66    101     +35
```

At `threshold=0.5` the run begins at 251 of 500 and ends at 247 — a
**drift of 4 features across 500 evaluations**. The reported subset is
the random initialization, lightly stirred. At `threshold=0.85` the same
budget moves the subset by 35 and the search is doing real work.

The reason is the fitness weighting. At `alpha=0.99`, halving the subset
is worth `0.01 * 0.5 = 0.005` of fitness, while the fold-to-fold spread
of the CV score is several times that — so the size term is below the
noise floor and only the score term steers. Raising `threshold` starts
the search where you want it; lowering `alpha` lets it stay there.

The algorithm's own settings matter here too. The run above uses
`initial_temperature=0.029`, which is calibrated to this objective —
[Setting the Annealing Temperature](annealing-temperature.md) covers how
that number is obtained and why the default of 1.0 would leave the
search accepting almost every proposal.

**Report both numbers.** A subset size that ends within a few percent of
its random start means the reported subset is close to the one the
initialization produced, whatever the fitness value suggests. The effect
grows with dimension, because a fixed evaluation budget covers a
vanishing fraction of a larger space.

## Best practice: the three measurements

1. **Run four arms, not two.** Defaults, selection, tuning, and both.
   Report the effect of selection as `P3 - P4`, and name the subtraction
   in the caption so the reader knows which one the figure shows.
2. **Report a nested score, never the search's own.** If the number in
   your table can be reproduced by `cross_val_score` on the winning
   subset with the same folds, it is the selection score, not a result.
3. **Report the subset size at the start and at the end.** Together with
   the per-fold sizes from nested CV, this is what tells a reader whether
   the search searched, and how stable its answer is.
4. **Test the difference.** Two arms scored on the same test set are
   paired: McNemar gives the difference, a paired bootstrap gives its
   confidence interval. Claiming the arms are *equivalent* — comparable
   score, far fewer features — is a separate question that needs an
   equivalence test, since a large p-value on its own does not
   establish it.
5. **Repeat over seeds.** One run of a stochastic search is one draw.
   Report mean and spread across at least five seeds, as
   [Feature Selection](feature-selection.md) does for splits.

## Checklist

- [ ] All four arms present; the caption says the effect is `P3 - P4`.
- [ ] Every arm shares one `CV` object, one estimator factory, one pipeline.
- [ ] The reported CV number comes from an outer loop the search never saw.
- [ ] Start and end subset sizes are both reported.
- [ ] Subset size moved by more than a few percent of the start.
- [ ] Differences carry a paired test, not just two numbers.
- [ ] Results are averaged over five or more seeds.

## Reference

- [Feature Selection](feature-selection.md) — running the search, the
  fitness function, and the `threshold` / `alpha` parameters.
- [Hyperparameter Tuning](gridsearch-comparison.md) — the tuning arm
  (`P4`), and how to compare tuning strategies fairly.
- [Microarray Data](microarray.md) — where the dimension is high enough
  for the `threshold=0.5` plateau in section 3 to bite.
- [Plotting Convergence](convergence-plot.md) — the curve that section 3
  tells you not to trust on its own.
- [Setting the Annealing Temperature](annealing-temperature.md) — if the
  search in section 3 is `SimulatedAnnealing`, calibrate it there first.
- [Early Stopping (patience)](early-stopping.md) — same principle,
  different parameter: measure it, do not guess it.
