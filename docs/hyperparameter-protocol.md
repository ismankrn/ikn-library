# Hyperparameter Optimization Protocol

[Hyperparameter Optimization](parameter-optimization.md) shows how to put
a model behind a `Problem` and let an optimizer drive it.
[Beating Random Search](beating-random-search.md) covers how much of the
resulting score is attributable to the search. This page is the sequence
in between, for the case where the model is a neural network — an ANN,
MLP or LSTM — and three properties of that setting change what a valid
run looks like:

- **The objective is stochastic.** The same architecture returns a
  different loss on a different initialization, so a difference between
  two candidates is only meaningful above that spread.
- **The objective is a minimum over epochs.** `min(val_loss)` is the
  best point of a trajectory, not a single measurement, and best-of-many
  is biased upward by construction.
- **The search space is heterogeneous.** Layer counts, unit counts,
  learning rates and categorical optimizers live on ranges that differ
  by two orders of magnitude, and distance-based algorithms feel that.

Each section below is one measurement that keeps one of those
properties from quietly producing the result.

## The protocol at a glance

<figure>
<svg viewBox="0 0 880 772" width="100%" style="max-width:880px;height:auto;border:1px solid #C2CED7;border-radius:4px" role="img" aria-label="The full dataset is split into X_dev and a sealed X_test, and X_dev is split again into X_train and X_val. The search loop decodes a candidate, trains it on X_train while monitoring X_val with early stopping, and takes the minimum validation loss as fitness while recording the epoch at which it occurred. After N evaluations the search returns a recipe — layers, units, learning rate, optimizer, regularisation and the epoch budget — and discards the weights. The final model is refit from scratch on X_train plus X_val for that epoch budget over several seeds, and only then is the seal broken to score it once on X_test.">
  <defs>
    <marker id="f3a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#1F2933"/></marker>
    <marker id="f3g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#1C6450"/></marker>
    <marker id="f3s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8A2F26"/></marker>
  </defs>
  <rect x="0" y="0" width="880" height="772" fill="#FFFFFF"/>
  <g font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12.5" fill="#1F2933">

    <rect x="300" y="22" width="280" height="40" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
    <text x="440" y="47" text-anchor="middle" font-size="13" font-weight="600">Full dataset</text>
    <line x1="440" y1="62" x2="440" y2="84" stroke="#1F2933" stroke-width="1.2"/>
    <line x1="230" y1="84" x2="700" y2="84" stroke="#1F2933" stroke-width="1.2"/>
    <line x1="230" y1="84" x2="230" y2="106" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>
    <line x1="700" y1="84" x2="700" y2="106" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>

    <rect x="70" y="108" width="320" height="38" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
    <text x="230" y="132" text-anchor="middle" font-size="13" font-weight="600">X_dev — 80%</text>

    <rect x="580" y="108" width="240" height="38" rx="3" fill="#F6E9E7" stroke="#8A2F26" stroke-width="1.6"/>
    <text x="700" y="132" text-anchor="middle" font-size="13" font-weight="600" fill="#8A2F26">X_test — 20%</text>
    <text x="700" y="168" text-anchor="middle" font-size="11" fill="#8A2F26" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" letter-spacing="1.6">S E A L E D</text>
    <line x1="700" y1="182" x2="700" y2="722" stroke="#8A2F26" stroke-width="1.4" stroke-dasharray="5 5"/>

    <line x1="230" y1="146" x2="230" y2="164" stroke="#1F2933" stroke-width="1.2"/>
    <line x1="140" y1="164" x2="330" y2="164" stroke="#1F2933" stroke-width="1.2"/>
    <line x1="140" y1="164" x2="140" y2="186" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>
    <line x1="330" y1="164" x2="330" y2="186" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>

    <rect x="60" y="188" width="160" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <text x="140" y="207" text-anchor="middle" font-weight="600">X_train · 60%</text>
    <text x="140" y="223" text-anchor="middle" font-size="11" fill="#5A6B78">the net trains here</text>

    <rect x="250" y="188" width="160" height="44" rx="3" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <text x="330" y="207" text-anchor="middle" font-weight="600">X_val · 20%</text>
    <text x="330" y="223" text-anchor="middle" font-size="11" fill="#5A6B78">the fitness scores here</text>

    <polyline points="140,232 140,250 235,250" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <polyline points="330,232 330,250 235,250" fill="none" stroke="#1F2933" stroke-width="1.2"/>
    <line x1="235" y1="250" x2="235" y2="352" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>

    <rect x="30" y="272" width="530" height="212" rx="5" fill="none" stroke="#1F2933" stroke-width="1.6"/>
    <text x="48" y="296" font-size="11" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" letter-spacing=".9">SEARCH LOOP — N evaluations</text>

    <rect x="58" y="310" width="472" height="34" rx="3" fill="none" stroke="#1F2933" stroke-width="1.1"/>
    <text x="294" y="332" text-anchor="middle" font-size="12">decode → layers · units · lr · optimizer · dropout · L2</text>

    <rect x="58" y="356" width="472" height="34" rx="3" fill="none" stroke="#1F2933" stroke-width="1.1"/>
    <text x="294" y="378" text-anchor="middle" font-size="12">train on X_train · monitor X_val · EarlyStopping(restore_best)</text>
    <line x1="294" y1="344" x2="294" y2="356" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>

    <rect x="58" y="402" width="472" height="46" rx="3" fill="#E4EFEA" stroke="#1C6450" stroke-width="1.4"/>
    <text x="294" y="422" text-anchor="middle" font-size="12" font-weight="600" fill="#1C6450">fitness = min(val_loss)</text>
    <text x="294" y="439" text-anchor="middle" font-size="11.5" fill="#1C6450">and record the epoch it occurred at</text>
    <line x1="294" y1="390" x2="294" y2="402" stroke="#1C6450" stroke-width="1.2" marker-end="url(#f3g)"/>

    <polyline points="530,425 546,425 546,327 532,327" fill="none" stroke="#5A6B78" stroke-width="1.2" marker-end="url(#f3a)"/>
    <text x="554" y="380" font-size="11" fill="#5A6B78">× N</text>

    <text x="586" y="330" font-size="11.5" fill="#8A2F26">X_val is consumed by</text>
    <text x="586" y="346" font-size="11.5" fill="#8A2F26">the search — it selects,</text>
    <text x="586" y="362" font-size="11.5" fill="#8A2F26">it does not estimate</text>

    <line x1="294" y1="484" x2="294" y2="512" stroke="#1C6450" stroke-width="1.4" marker-end="url(#f3g)"/>

    <rect x="90" y="512" width="410" height="76" rx="3" fill="#E4EFEA" stroke="#1C6450" stroke-width="1.8"/>
    <text x="295" y="534" text-anchor="middle" font-size="13" font-weight="600" fill="#1C6450">RECIPE — what the search returns</text>
    <text x="295" y="553" text-anchor="middle" font-size="11.5" fill="#1C6450">layers · units · lr · optimizer · dropout · L2 · epoch budget</text>
    <text x="295" y="574" text-anchor="middle" font-size="11.5" font-weight="600" fill="#8A2F26">the trained weights are discarded</text>

    <line x1="294" y1="588" x2="294" y2="616" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>
    <rect x="90" y="616" width="410" height="56" rx="3" fill="none" stroke="#1F2933" stroke-width="1.4"/>
    <text x="295" y="638" text-anchor="middle" font-size="12">refit from scratch on X_train + X_val</text>
    <text x="295" y="656" text-anchor="middle" font-size="11.5" fill="#5A6B78">for the recipe's epoch budget · k seeds · report mean ± SD</text>

    <line x1="294" y1="672" x2="294" y2="700" stroke="#1F2933" stroke-width="1.2" marker-end="url(#f3a)"/>
    <rect x="150" y="700" width="290" height="44" rx="3" fill="#F6E9E7" stroke="#8A2F26" stroke-width="1.8"/>
    <text x="295" y="727" text-anchor="middle" font-size="13" font-weight="600" fill="#8A2F26">BREAK THE SEAL — once</text>
    <line x1="700" y1="722" x2="446" y2="722" stroke="#8A2F26" stroke-width="1.4" stroke-dasharray="5 5" marker-end="url(#f3s)"/>
  </g>
</svg>
<figcaption>
Three splits, and a search that returns a <em>recipe</em> rather than a
model. <code>X_train</code> is what the network fits; <code>X_val</code>
is what the fitness reads, which means the search consumes it and it can
no longer estimate anything. The weights found during the search are
discarded — only the settings survive, including the epoch budget — and
the final model is rebuilt from scratch on <code>X_train + X_val</code>
before the seal is broken once.
</figcaption>
</figure>

The one arrow worth tracing twice is the loop-back. Every iteration
rereads `X_val`, so after N evaluations the best validation loss is the
minimum of N draws from a noisy quantity. That is why the box at the
bottom of the diagram is a refit and a fresh test score, not the
validation number carried forward.

!!! note "TensorFlow is not a dependency"
    `ikn-library` never sees the model — only a `Problem` with an
    `_evaluate` method — so any framework fits inside it. The examples
    below use Keras (`pip install tensorflow`) because the properties
    being measured are properties of network training.

## 1. `min(val_loss)` is a maximum over epochs

Scoring the best epoch of a run is the right choice — the last epoch can
be worse than the model's best point, which would make the fitness an
arbitrary stopping position. But taking a minimum over a trajectory is
a selection, and selections are optimistic.

How optimistic is measurable. Shuffle the labels so that no model can
honestly do better than the class prior, then record what `min(val_loss)`
reports anyway:

```python
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras import Sequential, layers
from keras.utils import set_random_seed
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
p = y.mean()
H = -(p * np.log(p) + (1 - p) * np.log(1 - p))   # the prior-entropy floor


def run(n_val, epochs, seed):
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)              # labels carry no information
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_shuffled, test_size=n_val, random_state=seed, stratify=y_shuffled)
    scaler = StandardScaler().fit(X_tr)
    set_random_seed(seed)
    model = Sequential([layers.Input((30,)),
                        layers.Dense(16, activation="relu"),
                        layers.Dense(1, activation="sigmoid")])
    model.compile("adam", "binary_crossentropy")
    history = model.fit(scaler.transform(X_tr), y_tr,
                        validation_data=(scaler.transform(X_va), y_va),
                        epochs=epochs, batch_size=32, verbose=0)
    v = np.array(history.history["val_loss"])
    return v.min(), v[-1]


print(f"class prior {p:.4f} -> prior entropy = {H:.4f}\n")
print(f"{'n_val':>7} {'epochs':>7} {'min(val_loss)':>15} {'last epoch':>12} "
      f"{'apparent gain':>15}")
for n_val in (57, 114, 228):
    for ep in (30, 100):
        r = np.array([run(n_val, ep, s) for s in range(8)])
        print(f"{n_val:>7} {ep:>7} {r[:, 0].mean():>15.4f} "
              f"{r[:, 1].mean():>12.4f} {H - r[:, 0].mean():>+15.4f}")
```

Output:

```text
class prior 0.6274 -> prior entropy = 0.6603
Labels are shuffled, so no model can honestly beat the prior entropy.

  n_val  epochs   min(val_loss)   last epoch   apparent gain
     57      30          0.6558       0.6655         +0.0045
     57     100          0.6518       0.6885         +0.0085
    114      30          0.6803       0.6890         -0.0200
    114     100          0.6799       0.7130         -0.0196
    228      30          0.6807       0.6883         -0.0204
    228     100          0.6803       0.7348         -0.0200
```

Two readings, and they point in different directions.

The **last-epoch** column gets steadily worse as the budget grows —
0.6883 to 0.7348 at `n_val=228` — because the network is busy memorising
shuffled labels. Scoring the final epoch measures how long training ran,
not what the architecture can do, which is why `min` over the history is
the right reduction.

The **minimum** is optimistic, but only where the validation set is
small. At `n_val=57` it dips 0.0045 to 0.0085 *below* a floor nothing can
honestly beat — that gap is pure selection over epochs. At `n_val=114`
and above it stays on the correct side of the floor, and the bias is no
longer visible at this budget. The inflation is a small-validation-set
effect, and a validation split of a few dozen rows is exactly where
architecture searches on clinical or gene-expression data tend to sit.

Two consequences for the protocol. The fitness may keep using
`min(val_loss)` — it is the better search signal, and the bias is
roughly constant across candidates, so the *ranking* mostly survives.
What must not happen is that number appearing in a results table: it is
a selection score, and the test set exists to replace it.

## 2. The objective is stochastic — decide how many seeds

A network's loss depends on its initialization. Re-evaluating one
architecture several times gives the smallest difference the search can
resolve:

```python
from keras.callbacks import EarlyStopping

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2,
                                                stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.25,
                                            stratify=y_dev, random_state=42)
scaler = StandardScaler().fit(X_tr)
A_tr, A_val = scaler.transform(X_tr), scaler.transform(X_val)


def net(units, lr, seed):
    set_random_seed(seed)
    model = Sequential([layers.Input((30,)),
                        layers.Dense(units, activation="relu"),
                        layers.Dense(1, activation="sigmoid")])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model


losses = []
for s in range(10):
    history = net(32, 1e-3, s).fit(
        A_tr, y_tr, validation_data=(A_val, y_val), epochs=60, batch_size=32,
        verbose=0, callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                            restore_best_weights=True)])
    losses.append(min(history.history["val_loss"]))

losses = np.array(losses)
print(f"one architecture (32 units, lr 1e-3), 10 training seeds")
print(f"  min val_loss : mean {losses.mean():.4f}  SD {losses.std():.4f}  "
      f"range {np.ptp(losses):.4f}")
for k in (1, 3, 5, 10):
    print(f"  SE of the mean over {k:>2} seed(s): {losses.std() / np.sqrt(k):.4f}")
```

Output:

```text
one architecture (32 units, lr 1e-3), 10 training seeds
  min val_loss : mean 0.0671  SD 0.0050  range 0.0146
  SE of the mean over  1 seed(s): 0.0050
  SE of the mean over  3 seed(s): 0.0029
  SE of the mean over  5 seed(s): 0.0022
  SE of the mean over 10 seed(s): 0.0016
```

One seed resolves differences of about 0.005; ten seeds resolve about
0.0016, for ten times the cost. That table is the budget decision, and
it is specific to the dataset and architecture — a wider net or a
smaller validation split moves it.

Both choices are defensible as long as one is made and stated:

- **Fix the seed** inside the fitness so the objective is deterministic.
  The search then ranks architectures rather than initializations, at
  the cost of tying the result to one initialization.
- **Average `k` seeds** per evaluation. The differences survive
  re-running, and the budget is `k` times larger.

What does not work is one seed per candidate with the seed left free:
the search then ranks a mixture of architecture and luck, and the
ranking changes when the run is repeated.

!!! warning "Seed the baselines too"
    The arms a search is compared against are trained by the same
    stochastic procedure. A baseline evaluated once carries the same
    0.005 of noise as a candidate, so the comparison needs the same
    treatment on both sides.

## 3. The recipe must carry the epoch budget

`EarlyStopping(restore_best_weights=True)` means each candidate is scored
at its own best epoch, and that epoch is part of what was selected. It
is not a constant across the search space:

```python
import keras

CANDIDATES = [(1, 16, 1e-2), (1, 16, 1e-4), (1, 128, 1e-3), (2, 32, 1e-3),
              (2, 128, 1e-2), (3, 64, 1e-4), (3, 256, 1e-3), (1, 64, 1e-3)]

print(f"{'layers':>7} {'units':>6} {'lr':>8} {'min(val_loss)':>14} {'at epoch':>10}")
for n_layers, units, lr in CANDIDATES:
    set_random_seed(0)
    model = Sequential(
        [layers.Input((30,))]
        + [layers.Dense(units, activation="relu") for _ in range(n_layers)]
        + [layers.Dense(1, activation="sigmoid")])
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy")
    history = model.fit(A_tr, y_tr, validation_data=(A_val, y_val),
                        epochs=150, batch_size=32, verbose=0)
    v = np.array(history.history["val_loss"])
    print(f"{n_layers:>7} {units:>6} {lr:>8.0e} {v.min():>14.4f} "
          f"{int(v.argmin()) + 1:>10}")
```

Output:

```text
layers  units       lr  min(val_loss)   at epoch
      1     16    1e-02         0.0444         44
      1     16    1e-04         0.2013        150
      1    128    1e-03         0.0546        106
      2     32    1e-03         0.0575         38
      2    128    1e-02         0.0575          4
      3     64    1e-04         0.0712        150
      3    256    1e-03         0.0720          2
      1     64    1e-03         0.0555        136

best epoch ranges 2 to 150 across the 8 candidates
```

The best epoch ranges from **2 to 150** across eight candidates in the
same space. A wide network at `lr=1e-2` peaks at epoch 4 and degrades
after; a narrow one at `lr=1e-4` is still improving when the budget runs
out at 150. There is no single epoch count that is correct for all of
them, and a refit that uses a fixed number would give some candidates a
different model from the one that was scored.

The last row is also worth noting: `(3, 256, 1e-4)` peaking at epoch 150
means the budget itself was binding, so that candidate was scored on an
unfinished run. A search space that contains such configurations needs a
larger epoch ceiling, or those candidates are penalised for the budget
rather than for their architecture.

So the object the search returns is a **recipe**, not a model:
architecture, learning rate, optimizer, regularisation, *and* the epoch
budget the recipe was scored at. The weights are discarded. The refit
rebuilds from scratch on `X_train + X_val` for that budget — see
[Refitting on train + validation](parameter-optimization.md#refitting-on-train-validation)
for why early stopping is dropped at that point and which direction the
resulting bias runs.

## 4. Normalise the search space before you search it

Swarm and trajectory algorithms move through the space using distances
and step sizes. A typical MLP space is heterogeneous enough that those
quantities stop meaning what they appear to:

```python
lower = np.array([1.0,  64.0, -4.0, 0.0])
upper = np.array([4.0, 256.0, -2.0, 3.0])
span = upper - lower
names = ["n_layers", "units", "log10(lr)", "optimizer"]

rng = np.random.default_rng(0)
pop = rng.uniform(lower, upper, size=(20, 4))
pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
d2 = np.array([np.sum((pop[i] - pop[j]) ** 2) for i, j in pairs])

print("Share of squared distance contributed by each dimension")
for k, name in enumerate(names):
    c = np.array([(pop[i, k] - pop[j, k]) ** 2 for i, j in pairs])
    print(f"  {name:<12} {100 * c.mean() / d2.mean():>6.1f}%")

print("\nAttractiveness exp(-gamma * d^2), gamma = 1, 190 pairs")
for label, g in (("raw coordinates", 1.0),
                 (f"scaled by mean(span)^2 = {span.mean() ** 2:,.0f}",
                  1.0 / span.mean() ** 2)):
    b = np.exp(-g * d2)
    print(f"  {label:<40} median {np.median(b):.4f}   "
          f"exactly zero: {(b == 0).sum():>3}/{len(b)}")

print("\nRandom-walk step on 'units', alpha = 0.25")
print(f"  alpha * (U - 0.5)         -> +/- {0.25 * 0.5:>7.3f} units")
print(f"  alpha * span * (U - 0.5)  -> +/- {0.25 * span[1] * 0.5:>7.3f} units")
```

Output:

```text
Share of squared distance contributed by each dimension
  n_layers        0.0%
  units         100.0%
  log10(lr)       0.0%
  optimizer       0.0%

Attractiveness exp(-gamma * d^2), gamma = 1, 190 pairs
  raw coordinates                          median 0.0000   exactly zero: 143/190
  scaled by mean(span)^2 = 2,500           median 0.1705   exactly zero:   0/190

Random-walk step on 'units', alpha = 0.25
  alpha * (U - 0.5)         -> +/-   0.125 units
  alpha * span * (U - 0.5)  -> +/-  24.000 units
```

`units` accounts for **100%** of the squared distance, so on raw
coordinates two candidates differing only in learning rate look
identical, and `exp(-d^2)` underflows to exactly zero on 143 of the 190
pairs — the attraction term vanishes and the algorithm degenerates into
a random walk. An unscaled step has the mirror problem: 0.125 moves
`log10(lr)` by 6% of its range and `units` by 0.07% of its, so the
narrow dimensions drift while the wide one never moves.

The algorithms in this library scale both quantities by the search
range, so `gamma` and `alpha` mean the same thing on any problem — see
`FireflyAlgorithm._attract` and its `alpha * span` step. The reason to
know the mechanism anyway is that a hand-written decode or a custom
`Problem` can reintroduce it, and the symptom is quiet: a converging
fitness curve while one dimension never leaves its initial value.

**The check.** Record the population's spread per dimension at the first
and last iteration. A dimension whose spread has not contracted was not
searched.

## 5. Search the dimensions that bind

An architecture search over width and depth answers the question "how
much capacity does this task need". If the network is already overfitting
at the capacity it has, that is not the binding question, and the search
will spend its budget on the dimension that matters least.

The signature is a train/validation gap that opens within a few epochs:
a training accuracy near 1.0 while the validation loss has been rising
since epoch 2 means the model memorised the training set almost
immediately, and more or fewer units will not change that. Dropout rate
and L2 strength belong in the search space alongside width and depth
whenever that pattern is present — they are continuous, cheap to encode,
and frequently the dimension with the most room in it.

The same applies to the arms the search is compared against: a baseline
architecture that the decode cannot express makes the comparison
uninterpretable in either direction. [Beating Random
Search](beating-random-search.md#4-can-the-search-reach-the-baseline)
has the reachability assertion.

## Best practice

1. **Split three ways** — train, validation, test — and keep the test
   set sealed until every choice is frozen.
2. **Score `min(val_loss)` but never report it.** It is a minimum over a
   trajectory and over N candidates; the test set replaces it.
3. **Measure the seed-to-seed spread first**, then either fix the seed
   inside the fitness or average `k` seeds, and say which.
4. **Return a recipe, not a model** — settings plus the epoch budget —
   and refit from scratch.
5. **Check that every dimension actually moved**, by comparing the
   population's per-dimension spread at the first and last iteration.
6. **Put regularisation in the search space** when the train/validation
   gap opens early, and confirm the baseline architectures are
   reachable by the decode.
7. **Report the budget in evaluations**, and remember each evaluation is
   a full training run: `population × iterations × folds × epochs` is
   the real cost.

## Checklist

- [ ] Train / validation / test are three disjoint splits.
- [ ] The fitness reads `X_val` only; `X_test` is untouched until the end.
- [ ] `restore_best_weights=True`, and the last epoch is never scored.
- [ ] The seed-to-seed spread of the objective was measured and quoted.
- [ ] The training seed is fixed inside the fitness, or `k` seeds averaged.
- [ ] Baselines were trained with the same seed discipline as candidates.
- [ ] The search returns settings **and** an epoch budget.
- [ ] The final model is refit from scratch, not carried over from the search.
- [ ] Every search dimension contracted between the first and last iteration.
- [ ] Dropout and L2 are in the space if the train/validation gap opens early.
- [ ] The reported number comes from the test set, read once.

## Reference

- [Hyperparameter Optimization](parameter-optimization.md) — defining the
  `Problem`, encoding integer and categorical dimensions, the Keras MLP
  example, refitting, and reporting.
- [Beating Random Search](beating-random-search.md) — the reference arm,
  and the reachability check for baselines.
- [Feature Selection Protocol](feature-selection-protocol.md) — the same
  discipline when the search picks columns rather than settings.
- [Setting the Annealing Temperature](annealing-temperature.md) — if the
  optimizer is `SimulatedAnnealing`, its temperature is on the scale of
  the validation loss, not of anything else.
- [Early Stopping (patience)](early-stopping.md) — `patience` for the
  *optimizer's* budget, distinct from Keras `EarlyStopping` for the
  network's.
