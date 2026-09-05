"""Example: Pareto-front feature selection with NSGA-II, end to end.

Builds the front on the training split, picks three candidate solutions
from it by different criteria, and compares them on an untouched test
split against using every feature.

Requires scikit-learn: pip install ikn-library[ml]
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library.algorithms import NSGA2
from ikn_library.multiobjective import (
    MultiObjectiveFeatureSelection,
    MultiObjectiveTask,
)


def knn():
    """A fresh scaler + KNN pipeline: the scaler is refitted per fold."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


CV = StratifiedKFold(5, shuffle=True, random_state=42)

X, y = make_classification(n_samples=400, n_features=30, n_informative=12,
                           n_redundant=5, flip_y=0.03, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 1. Build the Pareto front from the TRAINING data only.
problem = MultiObjectiveFeatureSelection(X_train, y_train, estimator=knn(), cv=CV)
solutions, objectives = NSGA2(population_size=40, seed=42).run(
    MultiObjectiveTask(problem=problem, max_evals=3000))

n_features = np.array([len(problem.selected_features(s)) for s in solutions])
cv_accuracy = 1.0 - objectives[:, 0]
order = np.argsort(n_features)
solutions, n_features, cv_accuracy = (solutions[order], n_features[order],
                                      cv_accuracy[order])


def knee_index(sizes, scores):
    """The point furthest from the line joining the front's two ends.

    A common heuristic for "where the curve stops paying off". Both
    axes are normalized first so the distance is not dominated by
    whichever axis has the larger range.
    """
    points = np.column_stack([sizes / sizes.max(), scores])
    start, end = points[0], points[-1]
    line = (end - start) / np.linalg.norm(end - start)
    offsets = points - start
    projections = np.outer(offsets @ line, line)
    return int(np.argmax(np.linalg.norm(offsets - projections, axis=1)))


# 2. Pick candidates from the front by three different criteria.
candidates = {
    "most accurate": int(np.argmax(cv_accuracy)),
    "knee point": knee_index(n_features, cv_accuracy),
    "smallest": 0,                      # the front is sorted by size
}

# 3. Train the final model on each candidate and score it on the test set.
baseline = accuracy_score(
    y_test, knn().fit(X_train, y_train).predict(X_test))
print(f"all {X.shape[1]} features            : test accuracy = {baseline:.4f}")

for label, index in candidates.items():
    features = problem.selected_features(solutions[index])
    model = knn().fit(X_train[:, features], y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test[:, features]))
    print(f"{label:<14} ({len(features):>2} features): "
          f"test accuracy = {accuracy:.4f}")
