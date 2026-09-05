"""Example: feature selection on the breast-cancer dataset with Binary ACO.

The test set is carved out before the search starts, the scaler lives
inside the pipeline so it is refitted per fold, and the before/after
comparison is made on the held-out test set — the cross-validated score
the search maximized is optimistically biased.

Requires scikit-learn: pip install ikn-library[ml]
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.problems import FeatureSelectionProblem


def knn():
    """A fresh scaler + KNN pipeline: the scaler is refitted per fold."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

problem = FeatureSelectionProblem(
    X_train, y_train,                 # training rows only
    estimator=knn(),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    alpha=0.99,
)
task = Task(problem=problem, max_evals=1000)
algo = BinaryAntColonyOptimization(population_size=20, evaporation=0.1, seed=42)
best_x, best_fitness = algo.run(task)

selected = problem.selected_features(best_x)
before = knn().fit(X_train, y_train).score(X_test, y_test)
after = knn().fit(X_train[:, selected], y_train).score(X_test[:, selected], y_test)

print(f"All {X.shape[1]} features      : test accuracy = {before:.4f}")
print(f"Selected {len(selected)} features : test accuracy = {after:.4f}")
print("Selected feature names:", [str(n) for n in data.feature_names[selected]])
