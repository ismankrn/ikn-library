# Pareto Concept

This note explains the machinery behind
[Multi-Objective Optimization](multiobjective.md) by working it out by
hand on seven solutions: what **dominance** actually asserts, how
solutions are sorted into **fronts**, what **crowding distance**
measures, and why the three together replace `argsort` in a
multi-objective algorithm. Every hand calculation is checked against the
library's own functions.

## The setup: seven candidate feature subsets

Take a feature-selection search with two objectives, exactly as
`MultiObjectiveFeatureSelection` reports them. **Both are minimized** —
that is the library's convention, so a problem that maximizes accuracy
returns `1 - accuracy`:

| | error (`1 - accuracy`) | fraction of features kept |
|---|---|---|
| **A** | 0.10 | 0.60 |
| **B** | 0.12 | 0.30 |
| **C** | 0.16 | 0.20 |
| **D** | 0.25 | 0.10 |
| **E** | 0.14 | 0.45 |
| **F** | 0.20 | 0.50 |
| **G** | 0.30 | 0.35 |

```python
import numpy as np

from ikn_library.multiobjective import (
    crowding_distance, dominates, non_dominated_sort, pareto_sort_indices,
)

labels = ["A", "B", "C", "D", "E", "F", "G"]
objectives = np.array([
    [0.10, 0.60],   # A
    [0.12, 0.30],   # B
    [0.16, 0.20],   # C
    [0.25, 0.10],   # D
    [0.14, 0.45],   # E
    [0.20, 0.50],   # F
    [0.30, 0.35],   # G
])
```

## Step 1 — dominance: when is one solution simply better?

**A dominates B when it is no worse in every objective and strictly
better in at least one.** Read that as: nobody could prefer B, whatever
they care about, because A matches or beats it everywhere.

Compare **B** (0.12, 0.30) with **E** (0.14, 0.45). B has the lower
error *and* keeps fewer features. There is no trade-off to discuss —
whatever weighting you had in mind, B is the better deal. **B dominates
E.**

Now compare **A** (0.10, 0.60) with **B** (0.12, 0.30). A has the lower
error; B keeps half as many features. Neither is better everywhere, so
**neither dominates the other**. That is not a failure of the
comparison — it is the entire point. The pair is a genuine trade-off,
and only someone who knows what a feature costs to measure can choose
between them.

```python
print("dominates(B, E):", dominates(objectives[1], objectives[4]))
print("dominates(A, B):", dominates(objectives[0], objectives[1]))
print("dominates(B, A):", dominates(objectives[1], objectives[0]))
```

Output:

```text
dominates(B, E): True
dominates(A, B): False
dominates(B, A): False
```

Two `False` values in a row is what "incomparable" looks like in code.
A single-objective `<` can never produce that answer, which is why
single-objective algorithms cannot be pointed at a multi-objective task.

## Step 2 — sorting into fronts

Dominance gives a ranking of *layers*, not of individuals:

- **Front 0** — the solutions nobody dominates. This is the Pareto
  front: the menu.
- **Front 1** — what remains non-dominated once front 0 is removed.
- and so on, peeling one layer at a time.

By hand: A, B, C and D are mutually incomparable — each has the lowest
error or the smallest subset relative to the others — so none of them is
dominated and all four form front 0. E is dominated by B, and G by C, so
both fall to front 1 once front 0 is peeled away. F is dominated by E,
which is itself in front 1, so F waits one more layer.

```python
fronts = non_dominated_sort(objectives)
for i, front in enumerate(fronts):
    print(f"front {i}: {[labels[j] for j in front]}")
```

Output:

```text
front 0: ['A', 'B', 'C', 'D']
front 1: ['E', 'G']
front 2: ['F']
```

This is the ranking step of NSGA-II. Note what it does **not** say:
inside front 0 there is no order at all. Four solutions, no winner —
which is exactly the situation step 3 exists to break.

## Step 3 — crowding distance: prefer the lonely solution

If ties inside a front were broken arbitrarily, a population would drift
into a cluster: many near-identical solutions in one corner of the
front, and nothing at the other end. Crowding distance breaks ties in
favour of **isolated** solutions, so the front stays spread out and the
user gets a menu instead of a huddle.

For each objective, sort the front by that objective and give every
solution the gap between its two neighbours, divided by the objective's
total spread. Then add the per-objective contributions.

Front 0, by **error**: A (0.10), B (0.12), C (0.16), D (0.25), spread
0.15.

- A and D are the extremes: infinite distance, so the endpoints of the
  front are never discarded.
- B: (0.16 − 0.10) / 0.15 = **0.400**
- C: (0.25 − 0.12) / 0.15 = **0.867**

Front 0, by **fraction**: D (0.10), C (0.20), B (0.30), A (0.60),
spread 0.50.

- D and A are the extremes again: infinite.
- C: (0.30 − 0.10) / 0.50 = **0.400**
- B: (0.60 − 0.20) / 0.50 = **0.800**

Adding up: B = 0.400 + 0.800 = 1.200, C = 0.867 + 0.400 = 1.267.

```python
front0 = fronts[0]
for label, distance in zip([labels[j] for j in front0],
                           crowding_distance(objectives[front0])):
    print(f"  {label}: {distance:.4f}")
```

Output:

```text
  A: inf
  B: 1.2000
  C: 1.2667
  D: inf
```

Three details in that arithmetic are decisions worth understanding:

- **Dividing by the spread** puts both objectives on one scale. Without
  it, an objective measured in thousands would drown out one measured in
  fractions, and the "spread" being preserved would be the spread of
  whichever objective happens to have big numbers.
- **Infinity for the endpoints** is what protects the extremes of the
  front — the most accurate solution and the smallest subset. They are
  the two answers a user is most likely to want, and a finite distance
  would eventually let them be discarded.
- **Summing across objectives** means crowding is about the *whole*
  objective space, not any one axis. C beats B here by 0.067, because C
  sits in a slightly emptier neighbourhood overall.

## Step 4 — the ranking: fronts first, crowding second

```python
order = pareto_sort_indices(objectives)
print("ranking:", [labels[j] for j in order])
```

Output:

```text
ranking: ['A', 'D', 'C', 'B', 'E', 'G', 'F']
```

Front 0 comes first, ordered by decreasing crowding distance: the two
endpoints A and D (infinite), then C (1.267), then B (1.200). Front 1
follows, then front 2.

That array is the whole point of the exercise. `pareto_sort_indices` is
a **drop-in replacement for `np.argsort(fitness)`**, so any algorithm
that ranks its population each iteration becomes multi-objective by
changing one line — which is how
[MO-KMA is built](multiobjective.md#worked-example-mo-kma) from the
single-objective Komodo Mlipir Algorithm.

## Why not just use a weighted sum?

Collapsing the objectives into `w × error + (1 − w) × fraction` looks
simpler, and for many fronts it works. But it can only ever find
solutions on the **convex hull** of the front — and a solution can be
Pareto-optimal while being invisible to every possible weight.

Three solutions make the point: P1 (0.0, 1.0), P2 (1.0, 0.0) and
P3 (0.6, 0.6). All three are non-dominated, so a decision maker should
be shown all three:

```python
points = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.6]])
names = ["P1", "P2", "P3"]

print("fronts:", [[names[j] for j in f] for f in non_dominated_sort(points)])
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    scores = w * points[:, 0] + (1 - w) * points[:, 1]
    print(f"  w={w:.2f} -> winner {names[int(np.argmin(scores))]} "
          f"(scores {np.round(scores, 3).tolist()})")
```

Output:

```text
fronts: [['P1', 'P2', 'P3']]
  w=0.00 -> winner P2 (scores [1.0, 0.0, 0.6])
  w=0.25 -> winner P2 (scores [0.75, 0.25, 0.6])
  w=0.50 -> winner P1 (scores [0.5, 0.5, 0.6])
  w=0.75 -> winner P1 (scores [0.25, 0.75, 0.6])
  w=1.00 -> winner P1 (scores [0.0, 1.0, 0.6])
```

**P3 never wins, for any weight.** Its score is 0.6 whatever `w` is,
while the better of P1 and P2 is at most 0.5. The compromise solution —
often the one a practitioner actually wants — sits in a dent in the
front, and no linear scalarization can reach into a dent. Running the
weighted sum with a hundred values of `w` would not help; the blind spot
is structural, not a matter of resolution.

Pareto dominance has no such blind spot, because it never adds the
objectives together in the first place.

## Where this shows up in the library

- [`dominates`, `non_dominated_sort`, `crowding_distance`,
  `pareto_sort_indices`](api.md#multi-objective) — the four functions
  used above, all public.
- [Multi-Objective Optimization](multiobjective.md) — NSGA-II, the
  Pareto archive, and the recipe for converting a population-based
  algorithm.
- K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
  multiobjective genetic algorithm: NSGA-II," *IEEE Transactions on
  Evolutionary Computation*, 6(2), 182–197, 2002.
