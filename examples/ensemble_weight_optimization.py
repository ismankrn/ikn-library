"""Example: optimize Random Forest voting weights with continuous ACO.

Protocol: the forest is trained on the train split, the weights are
optimized on the validation split, and all final numbers are reported
on an untouched test split.

The forest is kept deliberately weak (shallow trees) — that is where
re-weighting the members has the most room to help. A forest of large,
accurate trees is already near its ceiling and gains little.

Requires scikit-learn: pip install ikn-library[ml]
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ikn_library import Task
from ikn_library.algorithms import AntColonyOptimization
from ikn_library.ensemble import EnsembleWeightProblem, tree_proba_matrix

X, y = make_classification(n_samples=600, n_features=20, n_informative=5,
                           flip_y=0.1, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

forest = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=42)
forest.fit(X_train, y_train)

P_val = tree_proba_matrix(forest, X_val)
P_test = tree_proba_matrix(forest, X_test)

problem = EnsembleWeightProblem(P_val, y_val)
task = Task(problem=problem, max_evals=4000)
algo = AntColonyOptimization(population_size=20, archive_size=30, seed=42)
best_x, best_fitness = algo.run(task)

accuracy = lambda y_true, y_pred: float(np.mean(y_true == y_pred))  # noqa: E731
majority = accuracy(y_test, forest.predict(X_test))
uniform = accuracy(y_test, (P_test.mean(axis=1) > 0.5).astype(int))
weighted = accuracy(y_test, problem.predict(best_x, P_test))

print(f"Validation: uniform = {1 - problem.evaluate(np.full(30, 1 / 30)):.4f}, "
      f"optimized = {1 - best_fitness:.4f}")
print(f"Majority voting (RF default) : test accuracy = {majority:.4f}")
print(f"Uniform soft voting          : test accuracy = {uniform:.4f}")
print(f"ACO-optimized weights        : test accuracy = {weighted:.4f}")
